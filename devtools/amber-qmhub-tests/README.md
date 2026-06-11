# Amber/QMHub smoke tests

This directory contains external smoke tests for QMHub's patched AmberTools
integration. They are intended to be run from a QMHub repository checkout
against an already installed and patched AmberTools environment.

These tests are not package unit tests and are not installed into
`site-packages` by `pip install qmhub` or `python -m pip install -e .`. Do not
run them from an installed `site-packages/qmhub` directory. Keep or clone a
QMHub checkout, activate the environment where QMHub is installed, then run the
tests from this directory.

## Recommended workflow

```bash
git clone https://github.com/panxl/qmhub.git
cd qmhub
conda activate qmhub
python -m pip install -e .

cd devtools/amber-qmhub-tests
export AMBERHOME=/path/to/patched/ambertools
export QMHUB_AMBER_PRMTOP=/path/to/system.prmtop
export QMHUB_AMBER_RST7=/path/to/system.rst7
export QMHUB_AMBER_QMMASK=':1'
export QMHUB_AMBER_QMCHARGE=0
export QMHUB_AMBER_SPIN=1

make test.serial
```

For the parallel smoke tests, also set an MPI launcher if the default
`mpirun -np 2` is not appropriate:

```bash
export MPI_LAUNCH='mpirun -np 2'
make test.parallel
```

## What the tests exercise

The serial target runs patched `sander` checks for:

- QMHub EXTERN communication modes `comm=0`, `comm=1`, and `comm=2`
- direct SQM `qmmm_int=6` and `qmmm_int=7` field output
- SINR `ntt=12`
- optional QMHub/Q-Chem EXTERN communication modes

The parallel target runs the same harness through `sander.MPI` and the
configured MPI launcher.

The harness writes temporary Amber input files and QMHub configuration files
under `QMHUB_AMBER_TEST_WORK`, or under a temporary directory when that variable
is not set.

## Required inputs

The harness no longer runs `tleap`; provide an existing Amber system:

- `QMHUB_AMBER_PRMTOP` or `--prmtop`: Amber topology file
- `QMHUB_AMBER_RST7` or `--rst7`: Amber restart file

The restart must contain periodic box values on its final line because these
tests run with `ntb=1`.

The QM region defaults are intentionally simple and should be adjusted for the
provided system:

- `QMHUB_AMBER_QMMASK`, default `:1`
- `QMHUB_AMBER_QMCHARGE`, default `0`
- `QMHUB_AMBER_SPIN`, default `1`

## External programs

Required for `make test.serial`:

- patched Amber `sander`
- QMHub installed in the active Python environment

Required for `make test.parallel`:

- patched Amber `sander`
- patched Amber `sander.MPI`
- MPI launcher from `MPI_LAUNCH`, default `mpirun -np 2`
- QMHub installed in the active Python environment

SQM handling:

- QMHub EXTERN SQM cases use a generated fake `sqm` shim by default.
- Direct `qmmm_int=6` and `qmmm_int=7` cases exercise Sander's internal SQM
  path through `qm_theory='PM3'`.
- Set `--sqm` only when you need QMHub's SQM backend to call a specific external
  SQM command instead of the fake shim.

Q-Chem is optional and disabled by default:

```bash
export QMHUB_RUN_QCHEM=0     # default
export QMHUB_RUN_QCHEM=auto  # run only if qchem is found
export QMHUB_RUN_QCHEM=1     # require qchem
```

Use `--qchem-command` to point the harness at a specific Q-Chem executable or
launcher command.

## Useful variables

- `AMBERHOME`: AmberTools installation prefix; the harness searches
  `$AMBERHOME/bin` for Amber executables.
- `QMHUB_AMBER_PRMTOP`: Amber topology file.
- `QMHUB_AMBER_RST7`: Amber restart file.
- `QMHUB_AMBER_QMMASK`: Amber QM mask.
- `QMHUB_AMBER_QMCHARGE`: QM region charge.
- `QMHUB_AMBER_SPIN`: QM region spin multiplicity.
- `MPI_LAUNCH`: command prefix for `sander.MPI`.
- `QMHUB_RUN_QCHEM`: `0`, `auto`, or `1`.
- `QMHUB_KEEP_WORK=1`: keep temporary work directories after successful runs.
- `QMHUB_AMBER_TEST_WORK`: explicit work directory for generated files.

Run `make help` for a short command summary.

## Unit tests versus smoke tests

The normal Python test suite under `qmhub/tests` contains safe unit and
parser-style tests, including small checks for helper functions used by this
harness. Those tests do not launch Amber or Q-Chem.

The `make test.serial` and `make test.parallel` targets in this directory are
external integration smoke tests. They launch patched Amber executables and may
optionally launch Q-Chem.
