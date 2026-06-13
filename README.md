QMHub
=====

QMHub is a Python QM/MM interface for coordinating molecular system I/O, QM/MM
model setup, simulation settings, and quantum engine integrations. It provides
both a command line entry point and a Python API for workflows that exchange
systems, energies, and gradients through text, binary, or FIFO files.

QMHub currently uses a local source installation workflow with Python 3.12.
There is no released package on PyPI, and the source checkout workflow below is
the only supported installation path.

Installation
------------

Create a conda environment with the required Python, NumPy, and SciPy versions:

```bash
conda create --prefix ~/envs/qmhub python=3.12 numpy=2.4.6 scipy=1.17.1
conda activate ~/envs/qmhub
```

Clone QMHub under `~/github` and install from the source checkout:

```bash
mkdir -p ~/github
cd ~/github
git clone https://github.com/panxl/qmhub.git
cd ~/github/qmhub
python -m pip install .
```

Before running `qmhub`, build the helPME Python extension and copy the compiled
library into the installed QMHub package. QMHub will not run correctly until
`helpmelib*.so` is present in the installed package directory:

```bash
cd ~/github
git clone https://github.com/andysim/helpme.git helPME

# After building helPME and producing helpmelib*.so:
cp /path/to/helpmelib*.so ~/envs/qmhub/lib/python3.12/site-packages/qmhub/
```

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

Documentation
-------------

The hosted documentation is available at
[panxl.github.io/qmhub](https://panxl.github.io/qmhub/). The source repository
is available on [GitHub](https://github.com/panxl/qmhub/).

AmberTools source patches for Sander/QMHub support are documented in
[patches/README.md](patches/README.md).

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
