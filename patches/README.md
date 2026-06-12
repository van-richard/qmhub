# AmberTools patches

This directory contains source patches for building AmberTools with QMHub
support and related QM/MM updates. Apply these patches from the Amber source
root, such as `amber22_src`, `amber24_src`, or `ambertools26_src`, so paths like
`AmberTools/src/sander/...` resolve correctly.

## Compatibility

| AmberTools version | Patch files |
| --- | --- |
| AmberTools23 | `qmhub_at23.patch` only |
| AmberTools26 | `qmhub_at26.patch`, `sqm_at26.patch`, `sinr_at26.patch` |

Do not mix the AmberTools23 patch with the AmberTools26 patch set.

## AmberTools26 installation

Start in the AmberTools26 source directory. Copy the three AT26 patch files
there, or reference them by absolute path from this repository, then apply them
in this order:

```bash
cd ambertools26_src
patch -p1 < /path/to/qmhub/patches/qmhub_at26.patch
patch -p1 < /path/to/qmhub/patches/sqm_at26.patch
patch -p1 < /path/to/qmhub/patches/sinr_at26.patch
```

After the patches apply cleanly, continue with the normal AmberTools build
procedure for your platform. Before running patched `sander`, make sure QMHub is
installed and the `qmhub` command is available on `PATH`.

If you prefer downloading the patches directly:

```bash
cd ambertools26_src
curl -OL https://raw.githubusercontent.com/panxl/qmhub/master/patches/qmhub_at26.patch
curl -OL https://raw.githubusercontent.com/panxl/qmhub/master/patches/sqm_at26.patch
curl -OL https://raw.githubusercontent.com/panxl/qmhub/master/patches/sinr_at26.patch
patch -p1 < qmhub_at26.patch
patch -p1 < sqm_at26.patch
patch -p1 < sinr_at26.patch
```

## AmberTools23 installation

The patch `qmhub_at23.patch` applies only to AmberTools23:

```bash
tar xf AmberTools23.tar.bz2
cd amber22_src
curl -OL https://raw.githubusercontent.com/panxl/qmhub/master/patches/qmhub_at23.patch
patch -p1 < qmhub_at23.patch
```

After the patch applies cleanly, continue with the normal AmberTools build.

## AmberTools26 patch contents

`qmhub_at26.patch` adds the Sander QMHub external-QM interface. It adds
`qm2_extern_qmhub_module.F90`, wires it into the Sander build, routes
`qm_theory='EXTERN'` calculations through QMHub when an `&qmhub` namelist is
present, passes QM atom data, MM point charges, gradients, and unit-cell data
between Sander and QMHub, and supports text, binary, and FIFO exchange modes.
For QMHub EXTERN runs, it also detects the `&qmhub` namelist separately from
other EXTERN backends, adjusts periodic QM/MM pair-list handling so the full set
of non-link MM atoms can be sent when the QMHub path disables the usual QM
cutoff, and uses the arithmetic QM center for periodic pair-list imaging to
match the AmberTools23 QMHub patch. Amber stores lattice vectors in the columns
of `ucell`; the AmberTools26 QMHub text/binary exchange writes `ucell(:,i)`,
matching AmberTools23 and the current QMHub readers' lattice-vector row order.
Binary exchange writes QM and MM coordinate rows in the packed order consumed by
the QMHub Sander driver. The patch also preserves link-pair MM charges in
`qm_resp_charges` and redistributes `adjust_q` charge corrections onto QM and
link atoms so subsequent QMHub charge handling has the adjusted values.

`sqm_at26.patch` updates SQM/QMMM electrostatic handling. It adds storage for
the electrostatic potential and field at MM atom positions from QM atoms,
extends the legal `qmmm_int` range from `0..5` to `0..7`, reduces QM-MM
electrostatic damping for `qmmm_int=6` and `qmmm_int=7`, includes `qmmm_int=7`
in AM1/PM3/PM6 core-core correction paths, increases external-charge input
capacity, and reports MM electrostatic potential/field output when QMMM
verbosity is high enough. It allocates `qm_resp_charges` with link-atom slots so
link-pair charges can be retained alongside QM atom charges. The reported
`mm_esp` rows are indexed by SQM external-charge/QM-MM pair-list entry, not
guaranteed original Amber atom IDs.

`sinr_at26.patch` extends Sander SINR thermostat support by adding the
`ntt=12` middle-scheme path. It updates input validation, SINR initialization,
integration steps, restart velocity handling, trajectory cleanup, and printed
thermostat information for `ntt=12`. The patch also routes `ntt=12` through the
SINR atom-partitioning path used for parallel setup when `ntc=1`.

## New and relevant options

To use QMHub from patched AmberTools26, set `qm_theory='EXTERN'` in `&qmmm` and
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

- Apply the AmberTools26 patches in the required order: `qmhub_at26.patch`,
  then `sqm_at26.patch`, then `sinr_at26.patch`.
- Run `patch -p1` from the Amber source root, not from inside `AmberTools/`.
- Confirm `qmhub --help` works in the same environment used to run `sander`.
- If a patch fails, verify the AmberTools version and start from a clean source
  tree before retrying.

## Possible future additions

- A minimal `qmhub.ini` example for a complete patched-Sander workflow.
- A tested AmberTools26 build transcript for common compilers and platforms.
- Small regression inputs for the QMHub EXTERN path, `qmmm_int=6/7`, and
  `ntt=12`.
