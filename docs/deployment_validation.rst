Deployment and Q-Chem validation
================================

This is the deployment audit and integration-test record for 12 November 2026.
The runtime baseline is commit
``f35aa813d461cf2209c20bed001bc5d9cfdebe17`` on the fork's ``vibing`` branch.
At audit time that commit was also on GitHub. The default branch, ``master``,
was not changed by this work.

The completed runtime work covers **OSU and HPCC**. Polaris was deferred at the
user's request after the EnzymeSim project quota prevented environment creation.
Local runtime setup was stopped when the user narrowed the scope to OSU and
HPCC. A local conda prefix had already been created with Python 3.12.13,
NumPy 2.4.6 and SciPy 1.17.1; QMHub and helPME were not installed into that new
prefix, and it is not a validated QMHub installation.

How the components fit together
-------------------------------

``QMMM`` coordinates text, binary or FIFO exchange through ``iotools``, loads a
``System``, constructs the electrostatic ``Model``, and attaches backend engines.
``DependArray`` and related cache objects recalculate quantities when their input
dependencies change. An engine group combines backend energies, QM gradients
and potentials/fields at embedding sites.

The model constructs near-field embedding charges and periodic or nonperiodic
electrostatics. Its result layer combines the backend quantities with switching,
projection and electrostatic corrections. ``Simulation`` then aggregates the
requested MD or MTS results for return to the driver. Consequently, a complete
QM/MM gradient also includes model response terms beyond the raw QM gradient.

The Intel/GNU AmberTools patch sets are organized separately for AmberTools23
and AmberTools26. GNU patches include MPI completion synchronization and I/O
flushes needed by GNU runtime buffering. Python package installation, patched
Sander integration, and the compiled helPME extension are separate components.
This validation did not rebuild or execute Amber/Sander.

Q-Chem output contract
----------------------

The existing Q-Chem adapter already implements both MM potential/field pairs;
no change to ``qmhub/qmtools/qchem.py`` was required:

.. list-table:: Native scratch files consumed by the adapter
   :header-rows: 1
   :widths: 22 24 54

   * - Quantity
     - File
     - Reader behavior
   * - Energy
     - ``save/99.0``
     - Reads one float64 at byte offset 8.
   * - QM gradient
     - ``save/131.0``
     - Reads three float64 components per QM atom; reshapes to 3 by NQM.
   * - Modern MM ESP / field
     - ``save/1521.0`` / ``save/329.0``
     - Preferred complete pair; requires NMM potentials and 3*NMM field values.
   * - Legacy MM ESP / field
     - ``save/5001.0`` / ``save/5002.0``
     - Fallback complete pair; larger matching arrays use their trailing NMM rows.

The potential is returned unchanged. The native electric field is negated and
reshaped into ``mm_esp[1:]`` to obtain the potential-gradient convention used
inside QMHub. Successfully consumed files are deleted, so the validation first
preserves raw copies. Automatic parsing ignores stale text exports such as
``esp.dat`` and ``plot.esp``.

The default Q-Chem job type is ``force``. QMHub reads Q-Chem's gradient instead
of finite-differencing energies. The live tests below also require the Q-Chem
output to identify the expected version and report an analytic SCF gradient.

Installed environments
----------------------

Both retained ``qmhub2`` installations use Python **3.12.13**, NumPy **2.4.6**,
and SciPy **1.17.1**. Their 47 runtime Python source files match the committed
baseline exactly. The installed package metadata is
``1.0.0+216.gf35aa81``. The executable and Python import remain ``qmhub``;
``qmhub2`` is the environment/module name.

.. list-table:: Validated deployment locations
   :header-rows: 1
   :widths: 15 55 30

   * - Host
     - Environment
     - Activation/runtime notes
   * - OSU
     - ``/scratch/van/shared_envs/qmhub2``
     - Use the absolute interpreter and one thread per numerical library.
   * - HPCC
     - ``/home/van/Programs/miniforge3/envs/qmhub2``
     - ``module load qmhub2 fftw/3.3.7`` inside a Slurm allocation.

Existing legacy ``qmhub`` environments were preserved. They use Python 3.10.15
and older adapter code that supports only the legacy MM ESP pair. They were not
upgraded or used to certify the current fork.

Native PME checks use ``helpmelib.cpython-312-x86_64-linux-gnu.so`` and require
``qmhub.electools.pme.Ewald`` rather than the direct-Ewald fallback. HPCC's
binary needs FFTW providing ``libfftw3.so.3.5.7``; ``fftw/3.3.7`` supplies that
dependency. The HPCC validation loaded ``gnu7/7.3.0``, ``openmpi3/3.1.0``
and ``fftw/3.3.7``; OpenBLAS 0.2.20 resolves through the extension's embedded
``/opt/ohpc/pub/libs/gnu7/openblas/0.2.20/lib`` RPATH. OSU's binary resolves its
dependencies through its existing runtime and environment libraries. Reuse an
existing extension only with its matching Python ABI and compatible host libraries; no extension was rebuilt or copied
between hosts during this scoped validation.

Q-Chem installations
--------------------

OSU's newer builds are outside the advertised Q-Chem module collection:

* 6.3: ``/scratch/chance/trunk6.3`` with ``intel/2021.2.0``.
* 6.4: ``/scratch/chance/trunk6.4`` with ``gcc/10.2.0-multilib``.
* Shared auxiliary data: ``/home/chance/software/qcaux``.

The 6.4 build uses its MPI launcher with one rank; the 6.3 build follows its
serial execution path. Each test uses one OpenMP thread and no outer MPI launch.
The validation records actual runtime banners and executable/library hashes,
rather than treating the directory or module label as proof of the version.

HPCC's module named ``qchem/6.2`` sets ``Version = "6.1"`` and selects the
6.1 installation. No 6.3/6.4 installation was found in the inspected supported
locations there, so HPCC has **no live 6.3/6.4 validation** in this record.
The previously examined 6.1 QM/MM external-charge calculation did not emit the
scalar ESP file required by this adapter's complete-pair contract and is not an
equivalent compatibility test.

Validation procedure and evidence
---------------------------------

The retained run directories are:

* OSU: ``/scratch/van/qhv-20261112T212848Z``.
* HPCC: ``/work/van/qhv-20261112T212848Z``.

The standalone ``replay_validate.py`` and fixture manifest in those directories
are task-local validation artifacts, not installed QMHub modules. They replay
copies of retained water calculations from Q-Chem 5.2.1 and development builds
6.3.0 and 6.4.0. Each fixture has three QM atoms and three external charges.
Original calculation datasets remain untouched.

The replay harness checks committed-source identity, fixture hashes, exact
energy/gradient/potential/field parsing, modern-pair precedence, fallback to the
legacy pair, synthetic legacy leading rows, rejection of extra modern rows,
incomplete-output rejection, and actual native PME execution. Source identity
and fixture checks are independent of the numeric comparisons. Synthesized
leading rows test the reader branch; they are not evidence that those rows
occurred in the retained Q-Chem jobs.

Change into the retained run directory first. On HPCC, load ``qmhub2`` and
``fftw/3.3.7``. For a fresh replay directory inside an allocation, run::

   export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
   export QCTHREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
   unset PYTHONPATH
   /absolute/path/to/qmhub2/bin/python -B replay_validate.py \
       --fixtures fixtures --work replay-new --output replay-new.json --host HOST

OSU live tests use B3LYP/6-31G*, neutral singlet water, and three explicit MM
charges, with one nonperiodic and one periodic embedding case for each newer
build. The periodic case requires native PME. The harness compares the parsed
backend arrays for exact numeric agreement against preserved native files, then checks complete
QM/MM energy and gradient shapes and finiteness. It records the raw scratch,
input, parsed arrays, output banners, runtime libraries and hashes. Q-Chem is
bounded per invocation; the enclosing allocation is one node, one rank, one
CPU, 4 GiB and 30 minutes in ``express``. HPCC's replay/native checks use one
CPU, 1 GiB and five minutes in ``guest-compute`` with QoS ``low``.

.. list-table:: Completed checks
   :header-rows: 1
   :widths: 20 38 42

   * - Host / test
     - Result
     - Final evidence
   * - HPCC replay and native PME
     - 11/11 PASS; 47 runtime source hashes match; zero parser error.
     - ``hpcc-results-v2.json``; job 10953700 on compute-8-2 completed in 5 seconds.
   * - OSU replay and native PME
     - 11/11 PASS; 47 runtime source hashes match; zero parser error.
     - ``replay-results-v2.json`` in the OSU run directory.
   * - OSU 6.3.0 development
     - Nonperiodic and periodic cases PASS; analytic-gradient message verified.
     - ``reparsed/6.3/nonperiodic/result.json`` and ``6.3/periodic/result.json``.
   * - OSU 6.4.0 development
     - Nonperiodic and periodic cases PASS; analytic-gradient message verified.
     - ``6.4/nonperiodic/result.json`` and ``6.4/periodic/result.json``.

OSU job **8329730** completed with exit code 0 on **n045**, using one allocated
CPU and 4 GiB; elapsed time was **5 minutes 20 seconds**. Both newer builds
produced complete modern ESP/field pairs. All four live cases have exact
native-versus-parser agreement for energy, the 3-by-3 QM gradient, three
potentials and the 3-by-3 negated field. All four complete QM/MM gradients are
finite with shape 3 by 6. Both periodic cases execute native PME and produce
finite reciprocal-space results.

The initial 6.3 nonperiodic calculation completed successfully, but its first
validation report falsely rejected the one-element energy array. The final
``reparsed`` result reconstructs the identical Q-Chem input (verified by hash)
and evaluates the installed parsers and complete QM/MM model against the
preserved scratch; it launches **zero additional Q-Chem calculations**. The
original report and calculation are retained alongside that final assessment.

Installed source and native-library hashes remained unchanged on both hosts.
HPCC's ``hpcc-installed-files-unchanged.json`` compares before/after hashes;
OSU's original and corrected replay reports record identical source and native
hashes. Executable, wrapper, selected runtime-library, input and raw-file hashes
are recorded in each live ``result.json``. Scheduler closure is retained in
``allocation-final.txt`` on OSU and ``hpcc-job-accounting.txt`` on HPCC.


Initial attempts exposed two validation-harness mistakes: raw NumPy arrays
were supplied where native PME expects dependency-aware arrays, and scalar
energies were compared with QMHub's one-element arrays. These were corrected
only in task-local harnesses. Initial logs and raw outputs were preserved;
no QMHub numerical or production code was changed. An HPCC metadata helper
also needed to tolerate the absence of ``git`` on compute nodes; repository
provenance was collected separately from the login node.

Scope of the conclusions
------------------------

Exact agreement with native scratch establishes the parser's file selection,
layout and the implemented field-negation transformation for these samples. Finite complete QM/MM results
and native PME execution establish integration smoke-test coverage. They do
not independently prove derivative accuracy, validate every method/build, or
replace a production-system force check against independently displaced
energies. No new finite-difference gradient campaign or Amber integration test
was run. The complete pytest suite was not rerun after the runtime scope was
narrowed to OSU and HPCC; standalone parser/native checks and the OSU live
calculations provide the stated coverage.
