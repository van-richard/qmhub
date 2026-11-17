import numpy as np

from .utils.darray import DependArray
from .atoms import Atoms


class System(object):
    def __init__(self, n_atoms, n_qm_atoms, qm_charge, qm_mult):
        self.qm_index = np.s_[:n_qm_atoms]
        self.mm_index = np.s_[n_qm_atoms:]

        self.atoms = Atoms.new(n_atoms)

        self.qm = System.__new__(System)
        self.qm.atoms = self.atoms[self.qm_index]

        if n_atoms > n_qm_atoms:
            self.mm = System.__new__(System)
            self.mm.atoms = self.atoms[self.mm_index]
        elif n_atoms == n_qm_atoms:
            self.mm = None
        else:
            raise ValueError("The numer of QM atoms cannot be greater than the number of total atoms.")

        self.cell_basis = DependArray(np.zeros((3, 3)), name="cell_basis")
 
        self.qm_charge = qm_charge
        self.qm_mult = qm_mult

    def wrap_positions(self):
        self.atoms.positions -= self.qm.atoms.positions.mean(axis=1, keepdims=True)
        if not np.all(self.cell_basis == 0):
            self.atoms.positions -= np.around(self.atoms.positions / np.diagonal(self.cell_basis)[:, np.newaxis]) * np.diagonal(self.cell_basis)[:, np.newaxis]

    def map_energy_gradient(self, gradient):
        """Map a wrapped-coordinate gradient back to the input coordinates.

        Call this only for an energy evaluated after ``wrap_positions()``.
        Away from periodic image changes, moving one QM atom translates every
        wrapped coordinate by minus that displacement divided by N_QM. The
        resulting chain-rule term belongs to the QM columns only, including
        when the current centroid happens to be zero or the cell is nonperiodic.
        Direct Model callers that did not recenter must keep their raw gradient.
        """
        mapped = np.array(gradient, copy=True)
        n_qm_atoms = self.qm.atoms.positions.shape[1]
        # Copy before correcting so cached engine/model gradients stay intact.
        centroid_gradient = mapped.sum(axis=1, keepdims=True) / n_qm_atoms
        mapped[:, self.qm_index] -= centroid_gradient
        return mapped
