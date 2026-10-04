"""
hqnn_forge.encoding
===================
Quantum feature-map modules for projecting classical tabular vectors into
an n-qubit Hilbert space.

Every encoder layer and QNode factory, the circuit helpers they share and
the option types in their signatures are exported here, and
``from hqnn_forge.encoding import …`` is the documented import path; a new
encoder adds its layer and QNode factory to this list.

Exported symbols
----------------
Encoders:

QuantumEncodingLayer    nn.Module: angle embedding + entangling VQC (TorchLayer).
build_encoding_qnode    Factory that wires the angle-embedding QNode to a device and diff method.
AngleEmbeddingQNode     Alias of build_encoding_qnode.
IQPEncodingLayer        nn.Module: IQP embedding (pairwise ZZ phases) + the same VQC.
build_iqp_qnode         Factory for the IQP-embedding QNode.
AmplitudeEncodingLayer  nn.Module: up to 2**n_qubits features as state amplitudes.
build_amplitude_qnode   Factory for the amplitude-embedding QNode.
DataReuploadingLayer    nn.Module: angle embedding repeated before every layer.
build_data_reuploading_qnode  Factory for the data re-uploading QNode.

Circuit building blocks shared by the encoders:

apply_variational_layers  The entangler + Rot blocks, inside a QNode.
readout_wires           Wires measured under a readout option.
measure_z               The ⟨Z_i⟩ measurements a circuit returns.
input_scaling_shape     Shape of the re-uploading QNode's input_scaling weights.

The encoder interface:

EncodingLayer           Protocol: qlayer, n_qubits, n_features, prepare_inputs;
                        forward is qlayer(prepare_inputs(x)).  The contract is
                        spelled out in hqnn_forge._encoding_contract.
CircuitLayer            Protocol: qlayer and n_qubits, the minimum the
                        diagnostics accept.
is_encoding_layer       Runtime check for EncodingLayer (isinstance cannot do it).
is_circuit_layer        Runtime check for CircuitLayer.

Option types, for annotating calls:

DeviceName              Any PennyLane device name (``str``); see ``KNOWN_DEVICES``.
DiffMethod              Literal of the supported differentiation methods.
Entangler               Literal of the entangler options.
Position                Literal of where training noise is inserted (from hqnn_forge.noise).
Readout                 Literal of the readout options.
RotationAxis            Literal of the embedding rotation axes.
"""

from hqnn_forge._encoding_contract import (
    CircuitLayer,
    EncodingLayer,
    is_circuit_layer,
    is_encoding_layer,
)
from hqnn_forge.encoding.amplitude_embedding import (
    AmplitudeEncodingLayer,
    build_amplitude_qnode,
)
from hqnn_forge.encoding.angle_embedding import (
    AngleEmbeddingQNode,
    DeviceName,
    DiffMethod,
    Entangler,
    QuantumEncodingLayer,
    Readout,
    RotationAxis,
    apply_variational_layers,
    build_encoding_qnode,
    measure_z,
    readout_wires,
)
from hqnn_forge.encoding.data_reuploading import (
    DataReuploadingLayer,
    build_data_reuploading_qnode,
    input_scaling_shape,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer, build_iqp_qnode
from hqnn_forge.noise import Position

__all__: list[str] = [
    "AmplitudeEncodingLayer",
    "AngleEmbeddingQNode",
    "CircuitLayer",
    "DataReuploadingLayer",
    "DeviceName",
    "DiffMethod",
    "EncodingLayer",
    "Entangler",
    "IQPEncodingLayer",
    "Position",
    "QuantumEncodingLayer",
    "Readout",
    "RotationAxis",
    "apply_variational_layers",
    "build_amplitude_qnode",
    "build_data_reuploading_qnode",
    "build_encoding_qnode",
    "build_iqp_qnode",
    "input_scaling_shape",
    "is_circuit_layer",
    "is_encoding_layer",
    "measure_z",
    "readout_wires",
]
