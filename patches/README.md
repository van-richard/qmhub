# AmberTools patches

This directory contains source patches for building AmberTools with QMHub
support and related QM/MM updates. Apply these patches from the Amber source
root, such as `amber22_src` or `ambertools26_src`, so paths like
`AmberTools/src/sander/...` resolve correctly.

Patch files are grouped by AmberTools version and compiler family:

```text
patches/
├── at23_gnu/
├── at23_intel/
├── at26_gnu/
└── at26_intel/
```

## Compatibility

| AmberTools version | Compiler family | Patch directory | Apply in this order |
| --- | --- | --- | --- |
| AmberTools23 | GNU/gfortran | `at23_gnu/` | `asm_at23.patch`, `qmhub_at23_gnu.patch`, `sqm_at23.patch`, `sinr_at23.patch` |
| AmberTools23 | Intel/ifort/ifx | `at23_intel/` | `asm_at23.patch`, `qmhub_at23.patch`, `sqm_at23.patch`, `sinr_at23.patch` |
| AmberTools26 | GNU/gfortran | `at26_gnu/` | `qmhub_at26_gnu.patch`, `sqm_at26.patch`, `sinr_at26.patch` |
| AmberTools26 | Intel/ifort/ifx | `at26_intel/` | `qmhub_at26.patch`, `sqm_at26.patch`, `sinr_at26.patch` |

Pick exactly one row for your AmberTools version and compiler. Do not mix
AmberTools23 and AmberTools26 patch sets, and do not mix the GNU and Intel
QMHub patches.

## GNU vs Intel patch sets

The Intel patch sets are the baseline patches for Intel Fortran builds. The GNU
patch sets use compiler-specific QMHub patches named `qmhub_at23_gnu.patch` and
`qmhub_at26_gnu.patch`.

The GNU QMHub patches keep the same QMHub interface behavior, but include
GNU/gfortran-specific synchronization and I/O adjustments. In MPI Sander runs,
non-master ranks wait until the master rank finishes the external QMHub call,
then receive a completion signal before returning. The GNU patches also add
flushes around binary/FIFO writes so the QMHub driver sees complete records
promptly with GNU runtime buffering.

Support patches that do not differ by compiler are shared. In this repository,
some GNU directory entries are symlinks to the matching Intel patch files, such
as the SQM and SINR patches. Apply them through the selected GNU directory so
the command sequence stays version/compiler-specific.

## AmberTools23 installation

Start in a clean AmberTools23 source directory. Apply the patches from the
directory matching your compiler.

For GNU/gfortran builds:

```bash
cd amber22_src
patch -p1 < /path/to/qmhub/patches/at23_gnu/asm_at23.patch
patch -p1 < /path/to/qmhub/patches/at23_gnu/qmhub_at23_gnu.patch
patch -p1 < /path/to/qmhub/patches/at23_gnu/sqm_at23.patch
patch -p1 < /path/to/qmhub/patches/at23_gnu/sinr_at23.patch
```

For Intel Fortran builds:

```bash
cd amber22_src
patch -p1 < /path/to/qmhub/patches/at23_intel/asm_at23.patch
patch -p1 < /path/to/qmhub/patches/at23_intel/qmhub_at23.patch
patch -p1 < /path/to/qmhub/patches/at23_intel/sqm_at23.patch
patch -p1 < /path/to/qmhub/patches/at23_intel/sinr_at23.patch
```

After the patches apply cleanly, continue with the normal AmberTools build
procedure for your platform. Before running patched `sander`, make sure QMHub is
installed and the `qmhub` command is available on `PATH`.

## AmberTools26 installation

Start in a clean AmberTools26 source directory. Apply the patches from the
directory matching your compiler.

For GNU/gfortran builds:

```bash
cd ambertools26_src
patch -p1 < /path/to/qmhub/patches/at26_gnu/qmhub_at26_gnu.patch
patch -p1 < /path/to/qmhub/patches/at26_gnu/sqm_at26.patch
patch -p1 < /path/to/qmhub/patches/at26_gnu/sinr_at26.patch
```

For Intel Fortran builds:

```bash
cd ambertools26_src
patch -p1 < /path/to/qmhub/patches/at26_intel/qmhub_at26.patch
patch -p1 < /path/to/qmhub/patches/at26_intel/sqm_at26.patch
patch -p1 < /path/to/qmhub/patches/at26_intel/sinr_at26.patch
```

After the patches apply cleanly, continue with the normal AmberTools build
procedure for your platform. Before running patched `sander`, make sure QMHub is
installed and the `qmhub` command is available on `PATH`.

## Patch contents

The QMHub patches add the Sander QMHub external-QM interface. They add
`qm2_extern_qmhub_module.F90`, wire it into the Sander build, route
`qm_theory='EXTERN'` calculations through QMHub when an `&qmhub` namelist is
present, pass QM atom data, MM point charges, gradients, and unit-cell data
between Sander and QMHub, and support text, binary, and FIFO exchange modes.
For QMHub EXTERN runs, they also detect the `&qmhub` namelist separately from
other EXTERN backends. Amber stores lattice vectors in the columns of `ucell`;
the QMHub text/binary exchange writes those vectors in the row order consumed by
the current QMHub Sander readers.

`asm_at23.patch` updates AmberTools23 LAPACK build inputs needed by these
patched builds.

`sqm_at23.patch` and `sqm_at26.patch` update SQM/QMMM electrostatic handling.
They add storage for the electrostatic potential and field at MM atom positions
from QM atoms, extend the legal `qmmm_int` range from `0..5` to `0..7`, reduce
QM-MM electrostatic damping for `qmmm_int=6` and `qmmm_int=7`, include
`qmmm_int=7` in AM1/PM3/PM6 core-core correction paths, increase
external-charge input capacity, and report MM electrostatic potential/field
output when QMMM verbosity is high enough. They allocate `qm_resp_charges` with
link-atom slots so link-pair charges can be retained alongside QM atom charges.
The reported `mm_esp` rows are indexed by SQM external-charge/QM-MM pair-list
entry, not guaranteed original Amber atom IDs.

`sinr_at23.patch` and `sinr_at26.patch` extend Sander SINR thermostat support by
adding the `ntt=12` middle-scheme path. They update input validation, SINR
initialization, integration steps, restart velocity handling, trajectory
cleanup, and printed thermostat information for `ntt=12`. They also route
`ntt=12` through the SINR atom-partitioning path used for parallel setup when
`ntc=1`.

## New and relevant options

To use QMHub from patched AmberTools, set `qm_theory='EXTERN'` in `&qmmm` and
include an `&qmhub` namelist in the same `mdin` file. The presence of `&qmhub`
selects the QMHub EXTERN path; do not combine it with another EXTERN backend
namelist in the same input.

```text
&qmmm
  qmmask=':1',
  qm_theory='EXTERN',
  qmcharge=0,
  spin=1,
/
&qmhub
  config='qmhub.ini',
  basedir='qmhub',
  comm=2,
  debug=0,
/
```

The `&qmhub` namelist supports:

| Option | Default | Meaning |
| --- | --- | --- |
| `config` | `qmhub.ini` | QMHub configuration file passed to the `qmhub` command. |
| `basedir` | `qmhub` | Directory where Sander writes QMHub exchange files. |
| `comm` | `2` | Exchange mode: `0` text, `1` binary, `2` FIFO. |
| `debug` | `0` | Enables additional Sander-side QMHub interface logging when greater than zero. |

For `comm=0` and `comm=1`, Sander writes an input file, runs `qmhub` once per
QM call with `--text` or `--bin`, and reads the output file. For `comm=2`,
Sander creates FIFO files and launches one persistent `qmhub --fifo ... --driver
sander` process for the run.

The SQM patch allows `qmmm_int=6` and `qmmm_int=7`. Both reduce QM-MM
electrostatic damping; `qmmm_int=7` also follows the AM1/PM3/PM6 core-core
correction paths used by `qmmm_int=2`.

The SINR patch allows `ntt=12`. For `ntt=12`, the patched validation requires
`gamma_ln > 0`, `nkija >= 1`, `ntc=1`, `ntf=1`, `tempi <= temp0`, and
`sinrtau > 0`. The original `ntt=10` SINR path keeps its stricter
`sinrtau >= 0.5` requirement; `ntt=12` accepts smaller positive `sinrtau`
values and converts them internally before SINR initialization and restart
velocity I/O.

## Troubleshooting

- Pick the patch directory for the exact AmberTools version and compiler family
  you are building.
- Apply patches in the order shown in the compatibility table.
- Run `patch -p1` from the Amber source root, not from inside `AmberTools/`.
- Confirm `qmhub --help` works in the same environment used to run `sander`.
- If a patch fails, verify the AmberTools version and compiler selection, then
  start from a clean source tree before retrying.

## Possible future additions

- A minimal `qmhub.ini` example for a complete patched-Sander workflow.
- Tested AmberTools23 and AmberTools26 build transcripts for common compilers
  and platforms.
- Small regression inputs for the QMHub EXTERN path, `qmmm_int=6/7`, and
  `ntt=12`.
