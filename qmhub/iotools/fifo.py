import os, stat, time
from pathlib import Path

import numpy as np

from ..system import System


def read_fifo(fin, dtype, count):
    buffer = fin.read(int(dtype[-1]) * count)
    return np.frombuffer(buffer, dtype=dtype, count=count)


def read_fifo_scalar(fin, dtype):
    # count=1 reads still return a one-element ndarray, but _step is stored as
    # a zero-dimensional array and needs a scalar assignment.
    return read_fifo(fin, dtype=dtype, count=1).item()


class IOFifo(object):
    def __init__(self, cwd=None):
        self.mode = "fifo"
        self.cwd = cwd

    def load_system(self, input, system=None, step=None):

        self.cwd = self.cwd or Path(input).parent

        if step is None:
            step = 0

        self._step = np.asarray(step)

        assert stat.S_ISFIFO(os.stat(input).st_mode)

        self._fin = open(input, "rb")

        self._n_atoms, self._n_qm_atoms, qm_charge, qm_mult, self._pbc = read_fifo(self._fin, dtype="i4", count=5)
        self._system = system or System(self._n_atoms, self._n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_mult)

        return self._system

    def return_results(self, energy, forces, output=None):
        assert self._system is not None

        output = output or Path(self._fin.name).with_suffix('.out')

        try:
            os.mkfifo(output)
        except:
            pass

        self._system.atoms.charges[:] = read_fifo(self._fin, dtype="f8", count=self._n_atoms)
        self._system.qm.atoms.elements[:] = read_fifo(self._fin, dtype="i4", count=self._n_qm_atoms)

        if self._pbc > 0:
            cell_basis = read_fifo(self._fin, dtype="f8", count=9).reshape(3, 3).copy()
            cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
            self._system.cell_basis[:] = cell_basis

        # First cycle
        self._step[()] = read_fifo_scalar(self._fin, dtype="i4")

        if self._pbc == 2 :
            cell_basis = read_fifo(self._fin, dtype="f8", count=9).reshape(3, 3).copy()
            cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
            self._system.cell_basis[:] = cell_basis

        self._system.atoms.positions[:] = read_fifo(self._fin, dtype="f8", count=self._n_atoms * 3).reshape(3, self._n_atoms)

        self._system.wrap_positions()

        self._fout = open(output, "wb")
        self._write_results(energy, forces)

        while True:

            try:
                self._step[()] = read_fifo_scalar(self._fin, dtype="i4")
            except:
                break

            if self._pbc == 2 :
                cell_basis = read_fifo(self._fin, dtype="f8", count=9).reshape(3, 3).copy()
                cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
                self._system.cell_basis[:] = cell_basis

            self._system.atoms.positions[:] = read_fifo(self._fin, dtype="f8", count=self._n_atoms * 3).reshape(3, self._n_atoms)

            self._system.wrap_positions()

            self._write_results(energy, forces)

    def _write_results(self, energy, forces):
        # FIFO writes are buffered by Python; flush each packet so the driver
        # can read small energy/force responses immediately.
        self._fout.write(energy.tobytes())
        self._fout.write(forces.tobytes(order="F"))
        self._fout.flush()

    @staticmethod
    def save_input(input):
        """Preserve the input file passed from the driver."""
        pass
