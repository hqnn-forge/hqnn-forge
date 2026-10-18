# Benchmark

::: hqnn_forge.benchmark

## Optional robustness settings

`ValidationExtension` enables one validation-triggered restart; its default is
off. `pca_components` fits inner-training standardisation/SVD and rejects tuning.
`noise_channel`, `retune_noise_threshold` and `noise_batch_size` provide bounded
exact scoring under each channel. `FoldResult.noise_results` contains both
threshold policies; legacy `noise_mcc` and aggregate noise tables always use
the clean threshold. `fold_indices` executes a subset after deriving the full
seed tree. Shard summaries describe the actual completed folds.

`on_fold_end(model, fold_result, history)` observes each scored model. The history
dictionary contains attempt histories and local optimizer state; serializable
attempt diagnostics are also in the fold record. The callback cannot choose
fits or scores, and its Torch/NumPy/Python RNG use is isolated from later fits.
See the [frozen study](../results/noise-aware-robustness-study.md) for a complete
resumable example. None of these settings changes default preprocessing,
training budgets or legacy threshold semantics.
