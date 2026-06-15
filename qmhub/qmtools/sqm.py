from pathlib import Path
import numpy as np

from ..units import AMBER_HARTREE_TO_KCAL, AMBER_BOHR_TO_A
from .templates.sqm import get_qm_template, default_options
from .qmbase import QMBase


class SQM(QMBase):

    OUTPUT = "sqm.out"
    default_options = default_options

    def gen_input(self):
        """Generate input file for QM software."""

        if not "qmcharge" in self.options:
            self.options["qmcharge"] = str(self.charge)

        if not "spin" in self.options:
            self.options["spin"] = str(self.mult)

        with open(Path(self.cwd).joinpath("sqm.inp"), 'w') as f:
            f.write(get_qm_template(self.options))

            for e, s, x, y, z, in zip(
                self.qm_elements,
                self.qm_element_symbols,
                self.qm_positions[0],
                self.qm_positions[1],
                self.qm_positions[2],
            ):
                f.write(f"{e:3} {s:>2} {x:21.14e} {y:21.14e} {z:21.14e}\n")

            if self.mm_charges is not None:
                f.write("#EXCHARGES\n")
                for x, y, z, c in zip(
                    self.mm_positions[0],
                    self.mm_positions[1],
                    self.mm_positions[2],
                    self.mm_charges,
                ):
                    f.write(f"  1  H {x:21.14e} {y:21.14e} {z:21.14e} {c:21.14e}\n")
                f.write("#END\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = f"cd {self.cwd}; "
        cmdline += "sqm -O -i sqm.inp -o sqm.out"

        return cmdline

    def _get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        output = self._read_output(qm_cache, output)

        for line in output:
            if "QMMM: SCF Energy" in line:
                return float(line.split()[4]) / AMBER_HARTREE_TO_KCAL

        self._raise_missing_output("SQM SCF energy", "QMMM: SCF Energy", output)

    def _get_qm_energy_gradient(self, qm_cache=None, output=None):
        """Get QM energy gradient from output of QM calculation."""

        output = self._read_output(qm_cache, output)

        for i in range(len(output)):
            if "Forces on QM atoms from SCF calculation" in output[i]:
                gradient = np.empty((3, len(self.qm_elements)), dtype=float)
                try:
                    for j in range(len(self.qm_elements)):
                        line = output[i + j + 1]
                        gradient[0, j] = float(line[18:38])
                        gradient[1, j] = float(line[38:58])
                        gradient[2, j] = float(line[58:78])
                except (IndexError, ValueError) as exc:
                    raise RuntimeError("Can not parse SQM energy gradient output.") from exc
                return gradient / (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)

        self._raise_missing_output(
            "SQM energy gradient",
            "Forces on QM atoms from SCF calculation",
            output,
        )

    def _get_mm_esp(self, qm_cache=None, output=None):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        output = self._read_output(qm_cache, output)

        mm_esp = np.zeros((4, len(self.mm_charges)))

        for i in range(len(output)):
            if "Electrostatic potential and field on MM atoms from QM Atoms" in output[i]:
                for j in range(len(self.mm_charges)):
                    line = output[i + j + 1]
                    mm_esp[0, j] = float(line[18:38])
                    mm_esp[1, j] = -float(line[38:58])
                    mm_esp[2, j] = -float(line[58:78])
                    mm_esp[3, j] = -float(line[78:98])
                break

            if i == len(output) - 1:
                raise ValueError("Can not find MM electrostatic potential and field.")

        mm_esp[0] /= AMBER_HARTREE_TO_KCAL
        mm_esp[1:] /= (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
        return mm_esp

    def _get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        output = self._read_output(qm_cache, output)

        for i in range(len(output)):
            if "Atomic Charges" in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                try:
                    for j in range(len(self.qm_elements)):
                        line = output[i + j + 2]
                        mulliken_charges[j] = float(line.split()[-1])
                except (IndexError, ValueError) as exc:
                    raise RuntimeError("Can not parse SQM Mulliken charge output.") from exc
                return mulliken_charges

        self._raise_missing_output("SQM Mulliken charges", "Atomic Charges", output)

    def _read_output(self, qm_cache=None, output=None):
        if qm_cache is not None:
            return qm_cache

        output_path = Path(output) if output is not None else Path(self.cwd).joinpath(self.OUTPUT)
        try:
            return output_path.read_text().split("\n")
        except OSError as exc:
            raise FileNotFoundError(f"Can not open SQM output: {output_path}") from exc

    def _raise_missing_output(self, label, marker, output):
        # In sander-driven QMHub runs, missing SQM markers must abort before
        # incomplete energy/force data can be returned to the external driver.
        message = [
            f"Could not find {label} in SQM output.",
            f"Expected marker: {marker}",
        ]
        tail = "\n".join(output[-20:])
        if tail:
            message.extend(["Last lines of sqm.out:", tail])

        raise RuntimeError("\n".join(message))

    def _raise_qm_command_error(self, returncode):
        message = [
            f"SQM command failed with exit code {returncode}.",
            f"Command: {self.cmdline}",
        ]

        # SQM writes parser/runtime failures to sqm.out. Include the tail so
        # SLURM logs show why QMHub exited without requiring a separate file.
        tail = self._read_text_tail(Path(self.cwd).joinpath(self.OUTPUT))
        if tail:
            message.extend([f"Last lines of {self.OUTPUT}:", tail])

        raise RuntimeError("\n".join(message))

    @staticmethod
    def _read_text_tail(path, n_lines=20):
        try:
            lines = Path(path).read_text(errors="replace").splitlines()
        except OSError:
            return ""

        return "\n".join(lines[-n_lines:])
