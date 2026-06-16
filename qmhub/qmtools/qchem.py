import os
from pathlib import Path

import numpy as np

from .templates.qchem import get_qm_template, default_options
from .qmbase import QMBase
from ..utils.sys import get_nthreads


class QChem(QMBase):

    OUTPUT = None
    default_options = default_options
    nproc_getter = staticmethod(get_nthreads)
    # Q-Chem's MM ESP/field scratch filenames vary by version/build.
    # Prefer known binary pairs over version sniffing.
    _MM_ESP_BINARY_OUTPUTS = (
        ("save/1521.0", "save/329.0"),
        ("save/5001.0", "save/5002.0"),
    )
    # Q-Chem 5.2 5001.0/5002.0 files have been observed with extra leading
    # rows. Keep that trimming limited to this known pair.
    _MM_ESP_TRAILING_MM_OUTPUTS = (
        ("save/5001.0", "save/5002.0"),
    )
    _CRAY_ENV_VARS = (
        "CRAYPE_VERSION",
        "CRAY_CPU_TARGET",
        "CRAY_LD_LIBRARY_PATH",
    )

    def gen_input(self):
        """Generate input file for QM software."""

        with open(Path(self.cwd).joinpath("qchem.inp"), "w") as f:
            f.write(get_qm_template(self.options))

            f.write("$molecule\n")
            f.write(f"{self.charge} {self.mult}\n")
            for e, x, y, z, in zip(
                self.qm_elements,
                self.qm_positions[0],
                self.qm_positions[1],
                self.qm_positions[2],
            ):
                f.write(f"{e:3} {x:21.14e} {y:21.14e} {z:21.14e}\n")
            f.write("$end" + "\n\n")

            f.write("$external_charges\n")
            if self.mm_charges is not None:
                for x, y, z, c in zip(
                    self.mm_positions[0],
                    self.mm_positions[1],
                    self.mm_positions[2],
                    self.mm_charges,
                ):
                    f.write(f"{x:21.14e} {y:21.14e} {z:21.14e} {c:21.14e}\n")
            f.write("$end" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        os.environ["QCSCRATCH"] = str(Path(self.cwd).resolve())

        cmdline = f"cd {self.cwd}; "
        cmdline += " ".join(self._get_qchem_env_assignments())
        cmdline += f" qchem -nt {self._get_qchem_nthreads()} qchem.inp qchem.out save > qchem_run.log"

        return cmdline

    def _get_qchem_nthreads(self):
        # Q-Chem -nt is an OpenMP thread count. Do not fall back to scheduler
        # task counts; Amber/Sander launches can export MPI task counts too.
        return self.nproc

    def _is_cray_openmp_environment(self):
        if os.environ.get("PE_ENV", "").upper() == "CRAY":
            return True

        return any(name in os.environ for name in self._CRAY_ENV_VARS)

    def _get_qchem_env_assignments(self):
        nthreads = str(self._get_qchem_nthreads())
        env = [
            ("QCTHREADS", nthreads),
            ("OMP_NUM_THREADS", nthreads),
        ]

        if self._is_cray_openmp_environment():
            if "OMP_PLACES" not in os.environ:
                env.append(("OMP_PLACES", "cores"))
            if "OMP_PROC_BIND" not in os.environ:
                env.append(("OMP_PROC_BIND", "close"))

        return [f"{key}={value}" for key, value in env]

    def _get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or "save/99.0"
        output_path = Path(self.cwd).joinpath(output)

        # Q-Chem writes the scalar energy after an 8-byte record prefix.
        # Validate before reading so a failed first step reports the real cause.
        self._require_binary_output(output_path, 16, "Q-Chem energy")
        energy = np.fromfile(output_path, dtype="f8", count=1, offset=8).item()
        os.remove(output_path)

        return energy

    def _get_qm_energy_gradient(self, qm_cache=None, output=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or "save/131.0"
        output_path = Path(self.cwd).joinpath(output)
        count = len(self.qm_elements) * 3

        # The gradient must contain three doubles per QM atom. Checking this
        # before deletion avoids hiding incomplete Q-Chem scratch output.
        self._require_binary_output(output_path, count * 8, "Q-Chem energy gradient")
        gradient = np.fromfile(output_path, dtype="f8", count=count).reshape(-1, 3).T
        os.remove(output_path)

        return gradient

    def _get_mm_esp(self, qm_cache=None, output=None):
        """Get electrostatic potential  at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            qm_cache.update_cache()

        n_mm = len(self.mm_charges)
        mm_esp = np.zeros((4, n_mm))

        if n_mm == 0:
            return mm_esp

        tried = []

        if output is not None:
            # Explicit output is an internal/direct parser hook; normal runs
            # discover Q-Chem's known binary scratch pairs below.
            potential, field = self._validate_mm_esp_output(output)
            potential_path = Path(self.cwd).joinpath(potential)
            field_path = Path(self.cwd).joinpath(field)
            return self._read_binary_mm_esp(potential_path, field_path, n_mm)

        # Do not auto-read text files such as esp.dat or plot.esp here. They
        # can be derived artifacts left over from an earlier MD step.
        for potential, field in self._MM_ESP_BINARY_OUTPUTS:
            potential_path = Path(self.cwd).joinpath(potential)
            field_path = Path(self.cwd).joinpath(field)
            allow_trailing = (potential, field) in self._MM_ESP_TRAILING_MM_OUTPUTS
            tried.append(
                self._describe_binary_mm_esp_pair(
                    potential_path,
                    field_path,
                    n_mm,
                    allow_trailing,
                )
            )

            if self._valid_binary_mm_esp_pair(
                potential_path,
                field_path,
                n_mm,
                allow_trailing,
            ):
                return self._read_binary_mm_esp(
                    potential_path,
                    field_path,
                    n_mm,
                    allow_trailing,
                )

        raise FileNotFoundError(self._format_mm_esp_error(tried, n_mm))

    @staticmethod
    def _validate_mm_esp_output(output):
        if len(output) != 2:
            raise ValueError("Q-Chem MM ESP output must contain potential and field paths.")

        return output

    @staticmethod
    def _valid_binary_mm_esp_pair(
        potential_path,
        field_path,
        n_mm,
        allow_trailing=False,
    ):
        row_count = QChem._binary_mm_esp_row_count(potential_path, field_path)

        if row_count is None:
            return False
        if row_count == n_mm:
            return True

        # Q-Chem 5.2 can prepend non-MM ESP rows to 5001.0/5002.0. QMHub
        # writes $external_charges after $molecule, so the current MM block is
        # the trailing n_mm rows when both files have a matching row count.
        return allow_trailing and row_count > n_mm

    @staticmethod
    def _binary_mm_esp_row_count(potential_path, field_path):
        potential_path = Path(potential_path)
        field_path = Path(field_path)

        if not potential_path.exists() or not field_path.exists():
            return None

        potential_size = potential_path.stat().st_size
        field_size = field_path.stat().st_size

        if potential_size % 8 or field_size % 8:
            return None

        potential_rows = potential_size // 8
        field_values = field_size // 8

        if field_values != potential_rows * 3:
            return None

        return potential_rows

    @staticmethod
    def _describe_binary_mm_esp_pair(
        potential_path,
        field_path,
        n_mm,
        allow_trailing=False,
    ):
        potential_size = potential_path.stat().st_size if potential_path.exists() else "missing"
        field_size = field_path.stat().st_size if field_path.exists() else "missing"
        expected = f"expected {n_mm * 8} bytes"
        field_expected = f"expected {n_mm * 3 * 8} bytes"

        if allow_trailing:
            expected += " or a larger matching file with trailing MM rows"
            field_expected += " or a larger matching file with trailing MM rows"

        return (
            f"{potential_path} ({potential_size}; {expected}), "
            f"{field_path} ({field_size}; {field_expected})"
        )

    @staticmethod
    def _read_binary_mm_esp(
        potential_path,
        field_path,
        n_mm,
        allow_trailing=False,
    ):
        potential_path = Path(potential_path)
        field_path = Path(field_path)
        mm_esp = np.zeros((4, n_mm))

        if not QChem._valid_binary_mm_esp_pair(
            potential_path,
            field_path,
            n_mm,
            allow_trailing,
        ):
            raise ValueError(
                "Invalid Q-Chem binary MM ESP files: "
                + QChem._describe_binary_mm_esp_pair(
                    potential_path,
                    field_path,
                    n_mm,
                    allow_trailing,
                )
            )

        row_count = QChem._binary_mm_esp_row_count(potential_path, field_path)
        skip_rows = row_count - n_mm

        mm_esp[0] = np.fromfile(
            potential_path,
            dtype="f8",
            count=n_mm,
            offset=skip_rows * 8,
        )
        mm_esp[1:] = -np.fromfile(
            field_path,
            dtype="f8",
            count=(n_mm * 3),
            offset=skip_rows * 3 * 8,
        ).reshape(-1, 3).T

        # Match the previous lifecycle for Q-Chem binary scratch outputs.
        os.remove(potential_path)
        os.remove(field_path)

        return mm_esp

    @staticmethod
    def _require_binary_output(path, min_size, label):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"{label} binary output was not produced: {path}"
            )

        size = path.stat().st_size
        if size < min_size:
            raise ValueError(
                f"{label} binary output is incomplete: {path} "
                f"has {size} bytes, expected at least {min_size} bytes."
            )

    def _format_mm_esp_error(self, tried, n_mm):
        save_path = Path(self.cwd).joinpath("save")
        save_files = []

        if save_path.exists():
            save_files = sorted(path.name for path in save_path.glob("*.0"))

        message = [
            "Could not find valid Q-Chem MM ESP output.",
            f"Expected potential size: {n_mm * 8} bytes.",
            f"Expected field size: {n_mm * 3 * 8} bytes.",
            "Tried:",
            *[f"  - {item}" for item in tried],
        ]

        if save_files:
            message.extend(["Files present in save/:", "  - " + ", ".join(save_files)])

        return "\n".join(message)

    def _raise_qm_command_error(self, returncode):
        message = [
            f"Q-Chem command failed with exit code {returncode}.",
            f"Command: {self.cmdline}",
        ]

        # qchem_run.log captures launch/module/runtime errors, while qchem.out
        # often contains the chemistry-level failure. Include tails of both.
        for name in ("qchem_run.log", "qchem.out"):
            tail = self._read_text_tail(Path(self.cwd).joinpath(name))
            if tail:
                message.extend([f"Last lines of {name}:", tail])

        raise RuntimeError("\n".join(message))

    @staticmethod
    def _read_text_tail(path, n_lines=20):
        try:
            lines = Path(path).read_text(errors="replace").splitlines()
        except OSError:
            return ""

        return "\n".join(lines[-n_lines:])

    def _get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or ("qchem.out")

        try:
            output = Path(self.cwd).joinpath(output).read_text().split("\n")
        except:
            output = Path(output).read_text().split("\n")

        for i in range(len(output)):
            if "Ground-State Mulliken Net Atomic Charges" in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + j + 4]
                    mulliken_charges[j] = float(line.split()[2])
                break

        return mulliken_charges
