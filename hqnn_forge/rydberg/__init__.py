"""
hqnn_forge.rydberg
==================
A small simulated array of neutral atoms, evolving continuously in time under
the Rydberg Hamiltonian.  The model, its units and every convention are
specified in ``docs/rydberg-model.md``; this package implements it.

Units: ħ = 1, angular frequencies in rad/µs, lengths in µm, times in µs.

The matrices are dense and built in torch (``float64`` / ``complex128``),
without a PennyLane device: PennyLane's own ``qml.pulse.rydberg_interaction``
and ``rydberg_drive`` need JAX, which this library does not depend on.

Exported symbols
----------------
AtomRegister          Atom positions in µm; chain and ring constructors, C6/r^6 couplings, blockade radius.
rydberg_hamiltonian   Dense 2^n x 2^n Hamiltonian of a register, for one or a batch of pulses.
evolve                ρ(t) from all atoms in |g⟩: exact without dephasing, Strang splitting with it.
readout               ⟨n_i⟩ of every atom, optionally ⟨n_i n_j⟩ of every pair, from the diagonal of ρ.
DEFAULT_C6            C6 of two 87Rb atoms in 70S, 2π × 862 690 MHz µm^6, in rad/µs · µm^6.
MAX_ATOMS             Largest register rydberg_hamiltonian, evolve and readout accept (10).
"""

from hqnn_forge.rydberg.hamiltonian import DEFAULT_C6, MAX_ATOMS, rydberg_hamiltonian
from hqnn_forge.rydberg.lindblad import evolve, readout
from hqnn_forge.rydberg.register import AtomRegister

__all__: list[str] = [
    "DEFAULT_C6",
    "MAX_ATOMS",
    "AtomRegister",
    "evolve",
    "readout",
    "rydberg_hamiltonian",
]
