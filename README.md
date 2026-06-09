QMHub
=====

QMHub is a Python QM/MM interface for coordinating molecular system I/O, QM/MM
model setup, simulation settings, and quantum engine integrations. It provides
both a command line entry point and a Python API for workflows that exchange
systems, energies, and gradients through text, binary, or FIFO files.

QMHub supports Python 3.9 through 3.14.

Installation
------------

Install the released package from PyPI:

```bash
python -m pip install qmhub
```

Install from source for development or local testing:

```bash
git clone https://github.com/panxl/qmhub.git
cd qmhub
python -m pip install -e .
```

Command Line Usage
------------------

QMHub installs a `qmhub` command. Pass a configuration file and choose one
exchange mode:

```bash
qmhub config.ini --text exchange.txt
qmhub config.ini --bin exchange.bin
qmhub config.ini --fifo exchange.fifo
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
qm = pyscf
```

Python API Example
------------------

QMHub can also be driven from Python:

```python
from pathlib import Path
from qmhub import QMMM

qmmm = QMMM("text", driver=None, cwd=None)
qmmm.setup_simulation("md")
qmmm.load_system(Path("exchange.txt"))
qmmm.build_model(switching_type="lrec", cutoff=10.0, pbc=True)
qmmm.add_engine("pyscf", name="qm", group_name="engine")
qmmm.return_results()
```

Documentation
-------------

The hosted documentation is available at
[panxl.github.io/qmhub](https://panxl.github.io/qmhub/). The source repository
is available on [GitHub](https://github.com/panxl/qmhub/).

AmberTools source patches for Sander/QMHub support are documented in
[patches/README.md](patches/README.md).

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
