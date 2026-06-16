QMHub
=====

QMHub is a Python QM/MM interface for coordinating molecular system I/O, QM/MM
model setup, simulation settings, and quantum engine integrations. It provides
both a command line entry point and a Python API for workflows that exchange
systems, energies, and gradients through text, binary, or FIFO files.

QMHub currently uses a local source installation workflow. Python 3.12.13 is
the preferred runtime, and Python 3.10.15 is retained as the compatibility
fallback. There is no released package on PyPI, and the source checkout workflow
below is the only supported installation path.

Installation
------------

Create a conda environment with one of the tested Python, NumPy, and SciPy
version sets. The preferred Python 3.12 environment is:

```bash
conda create --prefix ~/envs/qmhub python=3.12.13 numpy=2.4.6 scipy=1.17.1
conda activate ~/envs/qmhub
```

The compatibility fallback is:

```bash
conda create --prefix ~/envs/qmhub-py310 python=3.10.15 numpy=2.2.6 scipy=1.15.3
conda activate ~/envs/qmhub-py310
```

Clone QMHub under `~/github` and install from the source checkout:

```bash
mkdir -p ~/github
cd ~/github
git clone https://github.com/van-richard/qmhub.git
cd ~/github/qmhub
python -m pip install .
```

For PME-backed periodic electrostatics, build the helPME Python extension and
copy the compiled library into the installed QMHub package. If `helpmelib*.so`
is absent, QMHub falls back to the direct Ewald implementation and prints a
warning; that fallback is slower and should be validated for the target system:

```bash
cd ~/github
git clone https://github.com/andysim/helpme.git helPME

# After building helPME and producing helpmelib*.so for the active Python version:
cp /path/to/helpmelib*.so ~/envs/qmhub/lib/python3.12/site-packages/qmhub/
```

For the Python 3.10 fallback environment, use `~/envs/qmhub-py310` and the
matching `python3.10/site-packages/qmhub/` path instead.

Command Line Usage
------------------

QMHub installs a `qmhub` command. Pass a configuration file and choose one
exchange mode:

```bash
qmhub qmhub.ini --text qmmm.inp
qmhub qmhub.ini --bin qmmm.inp
qmhub qmhub.ini --fifo qmmm.inp
```

Optional flags include `--driver` to select a driver, `--cwd` to set the engine
working directory, and `--interactive` to open an IPython session after results
are returned.

Configuration Example
---------------------

The command line interface reads an INI-style configuration file. This minimal
example shows the expected sections:

```ini
[simulation]
protocol = md
save_input = false

[model]
switching_function = lrec
cutoff = 10.0
pbc = true

[engine]
qm = qchem
```

Python API Example
------------------

QMHub can also be driven from Python:

```python
from pathlib import Path
from qmhub import QMMM

qmmm = QMMM("text", driver=None, cwd=None)
qmmm.setup_simulation("md")
qmmm.load_system(Path("qmmm.inp"))
qmmm.build_model(switching_type="lrec", cutoff=10.0, pbc=True)
qmmm.add_engine("qchem", name="qm", group_name="engine")
qmmm.return_results()
```

Q-Chem Runtime Notes
--------------------

The Q-Chem backend sets `QCSCRATCH` to the QMHub engine working directory and
runs `qchem -nt ... qchem.inp qchem.out save` there. Q-Chem `-nt` is treated as
an OpenMP thread count, so QMHub prefers thread-oriented variables in this order:
`QCTHREADS`, `OMP_NUM_THREADS`, then `MKL_NUM_THREADS`. Scheduler variables are
not used for Q-Chem `-nt`.

For MM electrostatic potential and field output, QMHub reads Q-Chem binary
scratch files from `save/`. It checks the known pairs `1521.0`/`329.0` and
Q-Chem 5.2-style `5001.0`/`5002.0`; text artifacts such as `esp.dat` or
`plot.esp` are ignored during automatic parsing because they may be stale.

Documentation
-------------

Documentation sources are available in the [`docs`](docs) directory. The source
repository is available on [GitHub](https://github.com/van-richard/qmhub/).

AmberTools source patches for Sander/QMHub support are documented in
[patches/README.md](patches/README.md). The patch tree is organized by
AmberTools version and compiler family:

```text
patches/
|-- at23_gnu/
|-- at23_intel/
|-- at26_gnu/
`-- at26_intel/
```

External Amber/QMHub smoke tests for patched AmberTools installations live in
[`devtools/amber-qmhub-tests`](devtools/amber-qmhub-tests). Run them from a
repository checkout with the `qmhub` conda environment active; they are not
installed into `site-packages` and should not be run from an installed
`site-packages/qmhub` directory.

Development
-----------

Install the package with test dependencies and run the test suite:

```bash
python -m pip install -e ".[test]"
pytest -q
```

License
-------

QMHub is distributed under the MIT license. See [LICENSE](LICENSE).

Copyright
---------

Copyright (c) 2019, Xiaoliang Pan

Acknowledgements
----------------

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms)
version 1.1.
