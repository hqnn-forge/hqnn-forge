"""
tests/test_checkpoint.py
========================
Round-trip and failure tests for hqnn_forge.utils.checkpoint.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, TypedDict, TypeVar

import pytest
import torch

import hqnn_forge
from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod
from hqnn_forge.models import (
    ClassicalBaseline,
    HybridBinaryClassifier,
    LinearClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.utils import checkpoint as ckpt
from hqnn_forge.utils import load_checkpoint, save_checkpoint


class _Backend(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


CPU: _Backend = {"device_name": "default.qubit", "diff_method": "backprop"}

MODELS = [
    pytest.param(HybridBinaryClassifier, dict(encoding_type="angle"), id="serial-angle"),
    pytest.param(
        HybridBinaryClassifier,
        dict(encoding_type="iqp", init_strategy="block_local"),
        id="serial-iqp",
    ),
    pytest.param(
        ParallelHybridClassifier, dict(classical_hidden_dim=5, dropout_p=0.2), id="parallel"
    ),
    pytest.param(
        HybridBinaryClassifier,
        dict(
            embedding_rotation="Y",
            entangler="strongly_entangling",
            readout="first",
            encoder_activation="sigmoid",
            init_strategy="normal",
        ),
        id="serial-published",
    ),
    pytest.param(
        MulticlassHybridClassifier,
        dict(
            n_classes=4,
            embedding_rotation="Y",
            entangler="strongly_entangling",
            readout="first",
            encoder_activation="sigmoid",
            init_strategy="normal",
        ),
        id="multiclass-published-trunk",
    ),
]


Classifier = HybridBinaryClassifier | ParallelHybridClassifier
C = TypeVar("C", bound=Classifier)


def _trained(cls: type[C], extra: dict[str, Any]) -> C:
    """A model whose weights differ from any fresh initialisation."""
    torch.manual_seed(0)
    model = cls(n_input_features=6, n_qubits=3, n_layers=2, **CPU, **extra)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    x, y = torch.randn(16, 6), torch.randint(0, 2, (16,))
    for _ in range(3):
        opt.zero_grad()
        logits = model(x)
        if isinstance(model, MulticlassHybridClassifier):
            loss = torch.nn.functional.cross_entropy(logits, y)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), y.float()
            )
        loss.backward()
        opt.step()
    return model


def _save_payload(payload: dict, path: Path) -> Path:
    torch.save(payload, path)
    return path


@pytest.fixture
def saved(tmp_path: Path) -> tuple[HybridBinaryClassifier, Path]:
    model = _trained(HybridBinaryClassifier, {})
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    return model, path


class TestRoundTrip:
    @pytest.mark.parametrize("init_strategy", ["restricted", "block_local"])
    def test_toy_size_reloads_without_the_init_warning(
        self, init_strategy: str, tmp_path: Path
    ) -> None:
        """The rebuild's weight draw is overwritten, so its #167 warning must not surface."""
        with pytest.warns(UserWarning, match="restricts nothing"):
            model = HybridBinaryClassifier(
                n_input_features=2, n_qubits=2, n_layers=1, init_strategy=init_strategy, **CPU
            )
        path = tmp_path / "toy.pt"
        save_checkpoint(model, path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = load_checkpoint(path)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)

    @pytest.mark.parametrize("cls, extra", MODELS)
    def test_identical_outputs_after_reload(
        self, cls: type[Classifier], extra: dict[str, Any], tmp_path: Path
    ) -> None:
        model = _trained(cls, extra)
        path = tmp_path / "model.pt"
        save_checkpoint(model, path)

        torch.manual_seed(123)  # a different RNG state must not matter
        loaded = load_checkpoint(path)

        assert type(loaded) is cls
        assert loaded.get_config() == model.get_config()
        x = torch.randn(5, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
        for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)

    def test_loaded_model_is_in_eval_mode(self, saved: tuple) -> None:
        _, path = saved
        assert not load_checkpoint(path).training

    def test_config_round_trips_every_constructor_argument(self) -> None:
        for cls in (HybridBinaryClassifier, ParallelHybridClassifier):
            model = cls(
                n_input_features=4, n_qubits=4, n_layers=1, use_classical_encoder=False, **CPU
            )
            assert set(model.get_config()) == ckpt._init_parameter_names(cls)
            rebuilt = cls(**model.get_config())
            assert rebuilt.get_config() == model.get_config()

    def test_get_config_returns_a_copy(self) -> None:
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=4, n_layers=1, **CPU)
        model.get_config()["n_qubits"] = 99
        assert model.get_config()["n_qubits"] == 4

    def test_override_device_on_load(self, saved: tuple) -> None:
        model, path = saved
        loaded = load_checkpoint(path, diff_method="parameter-shift")
        assert loaded.get_config()["diff_method"] == "parameter-shift"
        x = torch.randn(3, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=1e-6, atol=1e-6)

    def test_map_location_places_the_returned_model(self, saved: tuple) -> None:
        # cls(**config) always builds on the CPU, so without an explicit move
        # map_location only relocated the tensors that load_state_dict then
        # copied back into CPU parameters -- it had no effect on the result.
        _, path = saved
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = load_checkpoint(path, map_location=device)
        assert all(p.device.type == device.type for p in loaded.parameters())
        assert all(b.device.type == device.type for b in loaded.buffers())

    @pytest.mark.may_skip  # no CUDA device on the CI runners
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_map_location_cuda_does_not_return_a_cpu_model(self, saved: tuple) -> None:
        _, path = saved
        loaded = load_checkpoint(path, map_location="cuda", device_name="default.qubit")
        assert all(p.is_cuda for p in loaded.parameters())

    def test_file_loads_with_weights_only(self, saved: tuple) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        assert payload["format_version"] == ckpt.FORMAT_VERSION
        assert payload["hqnn_forge_version"] == hqnn_forge.__version__
        assert payload["class_name"] == "HybridBinaryClassifier"


class TestCheckpointsOlderThanAnOption:
    """
    A checkpoint written before the constructor gained an argument has no key
    for it.  Rebuilding it from ``_LEGACY_DEFAULTS`` gives back the model that
    was saved, because those values are what the circuit did before the
    argument existed -- and the warning says so out loud.
    """

    @staticmethod
    def _stripped(model: Classifier, path: Path, out: Path) -> Path:
        """
        ``path``'s payload as a file from before #131: every post-#131 key
        removed from its config, and no ``known_args``, which such a file
        never had.
        """
        save_checkpoint(model, path)
        payload = torch.load(path, weights_only=True)
        for name in ckpt._LEGACY_DEFAULTS:
            del payload["config"][name]
        del payload["known_args"]
        return _save_payload(payload, out)

    @pytest.mark.parametrize(
        "cls",
        [HybridBinaryClassifier, ParallelHybridClassifier, MulticlassHybridClassifier],
        ids=["serial", "parallel", "multiclass"],
    )
    def test_it_loads_and_predicts_what_the_saved_model_predicted(
        self, cls: type[Classifier], tmp_path: Path
    ) -> None:
        model = _trained(cls, {})
        old = self._stripped(model, tmp_path / "new.pt", tmp_path / "old.pt")

        with pytest.warns(RuntimeWarning, match="predates"):
            loaded = load_checkpoint(old)

        assert type(loaded) is cls
        assert loaded.get_config() == model.get_config()
        x = torch.randn(5, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)

    def test_the_warning_names_every_argument_it_filled(self, tmp_path: Path) -> None:
        model = _trained(HybridBinaryClassifier, {})
        old = self._stripped(model, tmp_path / "new.pt", tmp_path / "old.pt")

        with pytest.warns(RuntimeWarning) as record:
            load_checkpoint(old)

        message = str(record[0].message)
        for name, value in ckpt._LEGACY_DEFAULTS.items():
            assert name in message
            assert repr(value) in message

    def test_a_reload_that_filled_nothing_does_not_warn(self, saved: tuple) -> None:
        _, path = saved
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            load_checkpoint(path)

    def test_the_table_only_names_real_constructor_arguments(self) -> None:
        # Guards the table against rot: an argument renamed or dropped would
        # otherwise leave an entry here that silently never matches.
        for cls in (HybridBinaryClassifier, ParallelHybridClassifier, MulticlassHybridClassifier):
            assert set(ckpt._LEGACY_DEFAULTS) <= ckpt._init_parameter_names(cls)

    def test_re_saving_pins_the_filled_arguments(self, tmp_path: Path) -> None:
        model = _trained(HybridBinaryClassifier, {})
        old = self._stripped(model, tmp_path / "new.pt", tmp_path / "old.pt")
        with pytest.warns(RuntimeWarning):
            loaded = load_checkpoint(old)

        again = tmp_path / "pinned.pt"
        save_checkpoint(loaded, again)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            load_checkpoint(again)


class TestFailures:
    def test_unsupported_model(self, tmp_path: Path) -> None:
        with pytest.raises(
            TypeError,
            match="supports the classifiers in hqnn_forge.models.*got torch.nn.modules.linear.Linear",
        ):
            save_checkpoint(torch.nn.Linear(2, 1), tmp_path / "x.pt")  # type: ignore[arg-type]

    def test_subclass_is_not_silently_saved_as_parent(self, tmp_path: Path) -> None:
        class Custom(HybridBinaryClassifier):
            pass

        model = Custom(n_input_features=4, n_qubits=4, n_layers=1, **CPU)
        with pytest.raises(TypeError, match="Custom"):
            save_checkpoint(model, tmp_path / "x.pt")

    def test_version_mismatch(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["hqnn_forge_version"] = "0.0.1"
        other = _save_payload(payload, tmp_path / "old.pt")
        with pytest.raises(
            ValueError, match=r"written by hqnn_forge 0\.0\.1.*allow_version_mismatch=True"
        ):
            load_checkpoint(other)
        assert isinstance(
            load_checkpoint(other, allow_version_mismatch=True), HybridBinaryClassifier
        )

    def test_format_version_mismatch(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["format_version"] = ckpt.FORMAT_VERSION + 1
        with pytest.raises(ValueError, match="format version 2 is not supported"):
            load_checkpoint(_save_payload(payload, tmp_path / "future.pt"))

    def test_missing_constructor_field(self, saved: tuple, tmp_path: Path) -> None:
        # Not in _LEGACY_DEFAULTS, so it is a broken config rather than an old
        # one: back-filling it would rebuild an 'iqp' checkpoint as 'angle',
        # which fits the same weight shapes and predicts differently.  Without
        # known_args (a file from before it existed) that is all we can say.
        _, path = saved
        payload = torch.load(path, weights_only=True)
        del payload["config"]["encoding_type"]
        del payload["known_args"]
        with pytest.raises(ValueError, match=r"missing \['encoding_type'\]"):
            load_checkpoint(_save_payload(payload, tmp_path / "partial.pt"))

    def test_value_no_longer_accepted_is_refused(self, saved: tuple, tmp_path: Path) -> None:
        # A checkpoint written with embedding_rotation="Z" held a constant
        # quantum layer (#212).  The constructor now refuses the value, and a
        # load goes through the constructor, so the file is refused with the
        # same explanation rather than rebuilt as some other circuit.
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["config"]["embedding_rotation"] = "Z"
        with pytest.raises(ValueError, match="global phase"):
            load_checkpoint(_save_payload(payload, tmp_path / "z.pt"))

    def test_unexpected_constructor_field(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["config"]["n_heads"] = 4
        with pytest.raises(ValueError, match=r"unexpected \['n_heads'\]"):
            load_checkpoint(_save_payload(payload, tmp_path / "extra.pt"))

    def test_unknown_override(self, saved: tuple) -> None:
        _, path = saved
        with pytest.raises(ValueError, match=r"unknown constructor arguments.*\['n_heads'\]"):
            load_checkpoint(path, n_heads=4)

    def test_unknown_class(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["class_name"] = "os.system"
        with pytest.raises(ValueError, match="unknown class 'os.system'"):
            load_checkpoint(_save_payload(payload, tmp_path / "evil.pt"))

    def test_architecture_override_needs_the_opt_in(self, saved: tuple) -> None:
        _, path = saved
        with pytest.raises(
            ValueError, match=r"\['n_layers'\] describe the circuit the saved weights"
        ):
            load_checkpoint(path, n_layers=3)

    def test_opted_in_architecture_override_still_checks_shapes(self, saved: tuple) -> None:
        _, path = saved
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_checkpoint(path, n_layers=3, allow_architecture_override=True)

    def test_encoding_type_override_is_refused_not_silently_loaded(self, saved: tuple) -> None:
        """
        The override that shape checks cannot catch.

        QuantumEncodingLayer and IQPEncodingLayer both register
        quantum_layer.qlayer.weights at (n_layers, n_qubits, 3), so an angle
        checkpoint loads into an IQP model without a size mismatch and simply
        predicts something else.  Only the opt-in stands between a user and
        that model, so assert both halves: refused by default, and genuinely
        wrong once allowed.
        """
        model, path = saved
        with pytest.raises(
            ValueError, match=r"\['encoding_type'\].*allow_architecture_override=True"
        ):
            load_checkpoint(path, encoding_type="iqp")

        forced = load_checkpoint(path, encoding_type="iqp", allow_architecture_override=True)
        assert forced.get_config()["encoding_type"] == "iqp"
        x = torch.randn(5, 6)
        model.eval()
        with torch.no_grad():
            assert not torch.allclose(forced(x), model(x), rtol=1e-3, atol=1e-3)

    def test_init_strategy_override_is_refused(self, saved: tuple) -> None:
        # Not a shape change either: it only picks how fresh weights are drawn,
        # which the loaded state dict then overwrites -- so overriding it just
        # bakes a wrong init_strategy into the rebuilt get_config().
        _, path = saved
        with pytest.raises(ValueError, match=r"\['init_strategy'\]"):
            load_checkpoint(path, init_strategy="block_local")

    def test_init_seed_override_is_refused(self, saved: tuple) -> None:
        # The same reasoning as init_strategy: init_seed has no effect on the
        # loaded weights, but it records where the initial weights came from,
        # so an override would bake false provenance into get_config().  It
        # stays out of WEIGHT_SAFE_ARGS on purpose.
        _, path = saved
        assert "init_seed" not in ckpt.WEIGHT_SAFE_ARGS
        with pytest.raises(ValueError, match=r"\['init_seed'\]"):
            load_checkpoint(path, init_seed=3)

    def test_weight_safe_overrides_need_no_opt_in(self, saved: tuple) -> None:
        _, path = saved
        loaded = load_checkpoint(path, **CPU)
        assert loaded.get_config()["diff_method"] == "backprop"
        assert set(ckpt.WEIGHT_SAFE_ARGS) == {
            "device_name",
            "diff_method",
            "dropout_p",
            "noise_level",
            "noise_position",
            "noise_method",
            "noise_trajectories",
            "shots",
            "noise_channel",
            "seed",
            "readout_error",
        }

    def test_dropout_override_needs_no_opt_in_and_keeps_the_weights(self, saved: tuple) -> None:
        # nn.Dropout has no parameters, so this cannot invalidate a state dict
        # -- the point of WEIGHT_SAFE_ARGS.  Assert that, not just that it loads.
        model, path = saved
        loaded = load_checkpoint(path, dropout_p=0.5)
        assert isinstance(loaded, HybridBinaryClassifier)
        assert loaded.get_config()["dropout_p"] == 0.5
        assert loaded.dropout.p == 0.5
        for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        # And it stays inert in the eval-mode model that comes back.
        x = torch.randn(4, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)

    def test_training_noise_override_needs_no_opt_in_and_can_be_re_saved(
        self, saved: tuple, tmp_path: Path
    ) -> None:
        # Training noise only replaces the circuit in train mode, like dropout:
        # fine-tuning a saved model at another noise level must neither need the
        # architecture opt-in nor mark the model so save_checkpoint refuses it.
        model, path = saved
        loaded = load_checkpoint(path, noise_level=0.1, noise_position="end")
        assert isinstance(loaded, HybridBinaryClassifier)
        assert loaded.quantum_layer.noise_level == 0.1
        assert loaded.quantum_layer.noise_position == "end"
        for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        x = torch.randn(4, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
        save_checkpoint(loaded, tmp_path / "resaved.pt")
        assert load_checkpoint(tmp_path / "resaved.pt").get_config()["noise_level"] == 0.1

    def test_missing_state_dict(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        del payload["state_dict"]
        with pytest.raises(ValueError, match="has no 'state_dict'"):
            load_checkpoint(_save_payload(payload, tmp_path / "noweights.pt"))

    def test_not_a_checkpoint(self, tmp_path: Path) -> None:
        path = _save_payload({"weights": torch.zeros(2)}, tmp_path / "plain.pt")
        with pytest.raises(ValueError, match="is not an hqnn_forge checkpoint"):
            load_checkpoint(path)

    def test_foreign_file_is_a_value_error_not_a_torch_error(self, tmp_path: Path) -> None:
        # torch.load raises KeyError from its zip reader on a plain file, which
        # would escape load_checkpoint before the structural check runs.
        path = tmp_path / "notes.txt"
        path.write_text("this is not a checkpoint\n")
        with pytest.raises(ValueError, match="could not be read as a torch archive"):
            load_checkpoint(path)

    def test_missing_file_still_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nope.pt")

    def test_model_without_recorded_config(self) -> None:
        from hqnn_forge.models import BinaryClassifierBase

        class Bare(BinaryClassifierBase):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with pytest.raises(
            NotImplementedError, match="Bare does not record its constructor arguments"
        ):
            Bare().get_config()


class TestForcedOverrideProvenance:
    """
    A forced architecture override must not be launderable into a clean file.

    Without the mark, the sequence below produces a checkpoint whose config and
    state dict agree with each other and with nothing else: reloading it needs
    no override, so the guard that caught the mistake once can never fire
    again.
    """

    def test_forced_override_marks_the_model_with_the_forced_arguments(self, saved: tuple) -> None:
        _, path = saved
        forced = load_checkpoint(path, encoding_type="iqp", allow_architecture_override=True)
        assert getattr(forced, ckpt._FORCED_OVERRIDES_ATTR) == ("encoding_type",)

    def test_saving_a_forced_model_is_refused(self, saved: tuple, tmp_path: Path) -> None:
        model, path = saved
        forced = load_checkpoint(path, encoding_type="iqp", allow_architecture_override=True)

        # The weights really are the angle model's, and the model really does
        # predict something else -- this is what must not become a checkpoint.
        torch.testing.assert_close(
            model.state_dict()["quantum_layer.qlayer.weights"],
            forced.state_dict()["quantum_layer.qlayer.weights"],
            rtol=0,
            atol=0,
        )
        x = torch.randn(4, 6)
        model.eval()
        with torch.no_grad():
            assert not torch.allclose(forced(x), model(x), rtol=1e-3, atol=1e-3)

        with pytest.raises(
            ValueError, match=r"allow_architecture_override=True, forcing \['encoding_type'\]"
        ):
            save_checkpoint(forced, tmp_path / "laundered.pt")
        assert not (tmp_path / "laundered.pt").exists()

    def test_a_plainly_loaded_model_can_be_resaved(self, saved: tuple, tmp_path: Path) -> None:
        model, path = saved
        again = tmp_path / "again.pt"
        save_checkpoint(load_checkpoint(path), again)
        reloaded = load_checkpoint(again)
        x = torch.randn(4, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(reloaded(x), model(x), rtol=0, atol=0)

    @pytest.mark.parametrize("override", [{"diff_method": "parameter-shift"}, {"dropout_p": 0.3}])
    def test_weight_safe_overrides_stay_resavable(
        self, saved: tuple, tmp_path: Path, override: dict
    ) -> None:
        _, path = saved
        loaded = load_checkpoint(path, **override)
        assert not getattr(loaded, ckpt._FORCED_OVERRIDES_ATTR, ())
        save_checkpoint(loaded, tmp_path / "ok.pt")
        assert load_checkpoint(tmp_path / "ok.pt").get_config() == loaded.get_config()


class TestKnownArgs:
    """
    #185: ``save_checkpoint`` records the constructor arguments the writing
    version had, so a missing config key can be told apart as "written before
    this argument existed" (filled) or "edited out" (refused).
    """

    def test_save_records_every_constructor_argument(self, saved: tuple) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        assert payload["known_args"] == sorted(ckpt._init_parameter_names(HybridBinaryClassifier))

    def test_a_key_the_writer_knew_is_refused_not_filled(
        self, saved: tuple, tmp_path: Path
    ) -> None:
        """entangler is in _LEGACY_DEFAULTS, but this writer had it: an edit, not an old file."""
        _, path = saved
        payload = torch.load(path, weights_only=True)
        del payload["config"]["entangler"]
        with pytest.raises(ValueError, match=r"lacks \['entangler'\].*edited or is corrupt"):
            load_checkpoint(_save_payload(payload, tmp_path / "edited.pt"))

    def test_a_key_the_writer_did_not_know_is_filled(self, saved: tuple, tmp_path: Path) -> None:
        model, path = saved
        payload = torch.load(path, weights_only=True)
        del payload["config"]["entangler"]
        payload["known_args"].remove("entangler")
        with pytest.warns(RuntimeWarning, match="predates.*entangler"):
            loaded = load_checkpoint(_save_payload(payload, tmp_path / "older.pt"))
        assert loaded.get_config() == model.get_config()

    def test_a_predated_argument_without_legacy_default_is_refused(
        self, saved: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, path = saved
        monkeypatch.setattr(ckpt, "_NO_LEGACY_DEFAULT", frozenset({"readout"}))
        payload = torch.load(path, weights_only=True)
        del payload["config"]["readout"]
        payload["known_args"].remove("readout")
        with pytest.raises(ValueError, match=r"predates \['readout'\].*no legacy default"):
            load_checkpoint(_save_payload(payload, tmp_path / "older.pt"))

    @pytest.mark.parametrize("bad", ["entangler", [1, 2], None])
    def test_malformed_known_args(self, saved: tuple, tmp_path: Path, bad: object) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["known_args"] = bad
        with pytest.raises(ValueError, match="known_args"):
            load_checkpoint(_save_payload(payload, tmp_path / "bad.pt"))


DATA = Path(__file__).parent / "data"


class TestRealOldCheckpoint:
    """
    Files written by the tree before #159 (commit 12952f2), not payloads
    stripped by hand: they carry no known_args and none of the five arguments
    #159 added.  checkpoint_pre159_expected.pt holds the inputs and the
    outputs that tree computed for them.
    """

    @pytest.mark.parametrize("name", ["hybrid_angle", "hybrid_iqp", "parallel_angle"])
    def test_rebuilds_the_model_that_tree_saved(self, name: str) -> None:
        expected = torch.load(DATA / "checkpoint_pre159_expected.pt", weights_only=True)
        with pytest.warns(RuntimeWarning, match="predates") as record:
            model = load_checkpoint(
                DATA / f"checkpoint_pre159_{name}.pt", allow_version_mismatch=True
            )
        (backfill,) = [w for w in record if issubclass(w.category, RuntimeWarning)]
        message = str(backfill.message)
        for added in (
            "embedding_rotation",
            "entangler",
            "readout",
            "encoder_activation",
            "init_std",
        ):
            assert added in message
        with torch.no_grad():
            torch.testing.assert_close(
                model(expected["inputs"]), expected[name], rtol=0, atol=1e-6
            )

    def test_the_files_predate_known_args(self) -> None:
        payload = torch.load(DATA / "checkpoint_pre159_hybrid_angle.pt", weights_only=True)
        assert "known_args" not in payload
        assert "entangler" not in payload["config"]


#: Every classifier's constructor arguments, pinned.  A change here is a
#: checkpoint-compatibility decision: see "Compatibility rules" in
#: hqnn_forge/utils/checkpoint.py.
CONSTRUCTOR_ARGS = {
    HybridBinaryClassifier: {
        "n_input_features",
        "n_qubits",
        "n_layers",
        "use_classical_encoder",
        "dropout_p",
        "device_name",
        "diff_method",
        "init_strategy",
        "encoding_type",
        "embedding_rotation",
        "entangler",
        "readout",
        "encoder_activation",
        "init_std",
        "noise_level",
        "noise_position",
        "init_seed",
        "classical_encoder",
        "noise_method",
        "noise_trajectories",
        "trainable_input_scaling",
        "shots",
        "noise_channel",
        "seed",
        "readout_error",
    },
    ParallelHybridClassifier: {
        "n_input_features",
        "n_qubits",
        "n_layers",
        "classical_hidden_dim",
        "use_classical_encoder",
        "dropout_p",
        "device_name",
        "diff_method",
        "init_strategy",
        "encoding_type",
        "embedding_rotation",
        "entangler",
        "readout",
        "encoder_activation",
        "init_std",
        "noise_level",
        "noise_position",
        "init_seed",
        "classical_encoder",
        "noise_method",
        "noise_trajectories",
        "trainable_input_scaling",
        "shots",
        "noise_channel",
        "seed",
        "readout_error",
    },
    MulticlassHybridClassifier: {
        "n_input_features",
        "n_qubits",
        "n_layers",
        "n_classes",
        "strategy",
        "use_classical_encoder",
        "dropout_p",
        "device_name",
        "diff_method",
        "init_strategy",
        "init_std",
        "encoding_type",
        "init_seed",
        # The shared trunk's options (#226); each has a _LEGACY_DEFAULTS entry.
        "embedding_rotation",
        "entangler",
        "readout",
        "encoder_activation",
        "noise_level",
        "noise_position",
        "noise_method",
        "noise_trajectories",
        "classical_encoder",
        "trainable_input_scaling",
        "shots",
        "noise_channel",
        "seed",
        "readout_error",
    },
    ClassicalBaseline: {
        "n_input_features",
        "hidden_dims",
        "activation",
        "dropout_p",
        "init_seed",
    },
    LinearClassifier: {"n_input_features", "init_seed"},
}

#: The arguments each class had when checkpoints of it were first written: the
#: binary classifiers at #120 (31c9879), the multiclass one at #156 (11936cf),
#: ClassicalBaseline at #250, LinearClassifier at #501.
FIRST_CHECKPOINTED_ARGS = {
    HybridBinaryClassifier: {
        "n_input_features",
        "n_qubits",
        "n_layers",
        "use_classical_encoder",
        "dropout_p",
        "device_name",
        "diff_method",
        "init_strategy",
        "encoding_type",
    },
    ParallelHybridClassifier: {
        "n_input_features",
        "n_qubits",
        "n_layers",
        "classical_hidden_dim",
        "use_classical_encoder",
        "dropout_p",
        "device_name",
        "diff_method",
        "init_strategy",
        "encoding_type",
    },
    MulticlassHybridClassifier: {
        "n_input_features",
        "n_qubits",
        "n_layers",
        "n_classes",
        "strategy",
        "use_classical_encoder",
        "dropout_p",
        "device_name",
        "diff_method",
        "init_strategy",
        "init_std",
        "encoding_type",
    },
    ClassicalBaseline: {
        "n_input_features",
        "hidden_dims",
        "activation",
        "dropout_p",
        "init_seed",
    },
    LinearClassifier: {"n_input_features", "init_seed"},
}


class TestConstructorChangesAreDecided:
    @pytest.mark.parametrize("cls", list(CONSTRUCTOR_ARGS), ids=lambda c: c.__name__)
    def test_constructor_arguments_are_pinned(self, cls: type) -> None:
        current = ckpt._init_parameter_names(cls)
        assert current == CONSTRUCTOR_ARGS[cls], (
            f"{cls.__name__} gained {sorted(current - CONSTRUCTOR_ARGS[cls])} and lost "
            f"{sorted(CONSTRUCTOR_ARGS[cls] - current)}.  Checkpoints written before the "
            f"change do not carry a new argument: give it an entry in _LEGACY_DEFAULTS "
            f"(the old behaviour) or _NO_LEGACY_DEFAULT in hqnn_forge/utils/checkpoint.py, "
            f"then update CONSTRUCTOR_ARGS here."
        )

    def test_every_registered_class_is_pinned(self) -> None:
        assert set(ckpt._registry().values()) == set(CONSTRUCTOR_ARGS)

    @pytest.mark.parametrize("cls", list(CONSTRUCTOR_ARGS), ids=lambda c: c.__name__)
    def test_every_argument_added_since_has_a_decision(self, cls: type) -> None:
        added = CONSTRUCTOR_ARGS[cls] - FIRST_CHECKPOINTED_ARGS[cls]
        decided = set(ckpt._LEGACY_DEFAULTS) | ckpt._NO_LEGACY_DEFAULT
        assert added <= decided, sorted(added - decided)

    def test_the_two_tables_do_not_overlap(self) -> None:
        assert not set(ckpt._LEGACY_DEFAULTS) & ckpt._NO_LEGACY_DEFAULT
