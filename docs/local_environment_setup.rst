Local QMHub environments on lynnx
=================================

This setup record was prepared on 2026-11-16. Both environments use Python
3.12.13, NumPy 2.4.6, and SciPy 1.17.1. Environment names do not rename the
Python package or command: both remain ``qmhub``.

The branch selection for local reinstalls is:

.. list-table:: Environment sources
   :header-rows: 1

   * - Environment
     - Branch
     - Features
   * - ``qmhub2``
     - ``vibing``
     - Baseline QMHub with the recentering-gradient correction.
   * - ``qmhub2-mlpui``
     - ``vibe-pui``
     - The same correction plus the optional MLPUI adapter.

Resolve each branch to an immutable commit after committing its changes, then
build and install its wheel as described below. Exact installed commits,
versions, source hashes, and wheel hashes belong to the deployment manifests;
they are not inferred from an environment name or the current checkout.

The following snapshots describe the **initial installation before the
recentering-gradient correction**, rather than the targets for later reinstalls:

.. list-table:: Initial snapshots (historical)
   :header-rows: 1

   * - Environment
     - QMHub commit
     - Installed version
   * - ``qmhub2``
     - ``82c6d46a8aa9b1819a2845eaca638aebb4f409c5`` (tested ``vibing``)
     - ``1.0.0+217.g82c6d46``
   * - ``qmhub2-mlpui``
     - ``df11ca252ef0b47655a15bfcfbaa7d6de8a34bcd`` (``vibe-pui``)
     - ``1.0.0+227.gdf11ca2``

The existing ``qmhub`` environment was retained. Its older installed source
matches ``88eab36``; it was not renamed or replaced. Existing ``qmhub2``
dependencies were preserved, and the baseline QMHub package was added there.
The new environment started as an offline copy of that same conda dependency
stack, before the baseline QMHub package was installed.

MLPUI and native-library identity
---------------------------------

The new environment includes MLPUI 0.1.0 from commit
``fae948d1c8d78889a27cba9d213ffaac620f03ca``, its complete declared
dependencies, PyTorch ``2.11.0+cu128``, torch-geometric 2.8.1, e3nn 0.4.4,
and opt-einsum-fx 0.1.4. PyTorch requires ``setuptools<82``; the resolved
environment uses setuptools 78.1.0. System CUDA and NVIDIA drivers were not
changed. The CUDA wheel packages supply their own user-space runtime libraries.

Both installed packages received an identical copy of::

   /home/van/Programs/miniforge3/envs/qmhub/lib/python3.12/site-packages/qmhub/helpmelib.cpython-312-x86_64-linux-gnu.so

Its SHA-256 is::

   808ca4ca38e26c708ed355e75ee9bf119a5ec26d41b9a2bc8f78868820d25d6e

The filename and Python 3.12/x86_64 ABI were preserved; no native library was
rebuilt. Copying this binary to another machine requires a separate check of
architecture, Python ABI, linked libraries, and native computation.

``MLPUI_NEIGHBORS=python`` selects MLPUI's Python/PyTorch neighbor implementation.
It does not disable QMHub's native helPME reciprocal calculation. No trained
potential checkpoints are included in this installation.

Using the Lua modules
---------------------

The installed Lua files are under
``/home/van/github/PL888/modulefiles/sites/lynnx/conda``; their source templates
are in ``devtools/modulefiles/lynnx`` in this repository. The current default
shell loader is Tcl Modules, so initialize the existing Lmod installation::

   source /opt/apps/lmod/lmod/8.7.34/init/bash
   module use /home/van/github/PL888/modulefiles/sites/lynnx/conda
   module load qmhub2-mlpui
   python -I -c "import qmhub; print(qmhub.__file__)"
   module unload qmhub2-mlpui

Use ``module load qmhub2`` to select the baseline instead. Deactivate any active
conda environment before loading either module. Alternatively, use normal
``conda activate`` without either QMHub module loaded.

These modules select environment executables through ``PATH``. They preserve
conda's shell state, clear inherited ``PYTHONPATH`` while loaded, disable user
site packages, and prevent overlapping environment selections. They do not
autoload Amber, Sander, Q-Chem, or CUDA, modify thread counts, or add conda's
library directory to ``LD_LIBRARY_PATH``. Select Q-Chem versions explicitly;
the current unversioned local Q-Chem default is 5.2.

Lmod removes duplicate entries from inherited ``PATH`` while retaining first
occurrence precedence. Its ``pushenv`` primitive restores an originally empty
Python setting as unset; nonempty and unset settings restore exactly. Shell
startup files were not changed.

Reproduction records
--------------------

``devtools/conda-envs/qmhub2-mlpui.yaml`` describes the conda base;
``qmhub2-mlpui-linux-64.explicit.txt`` records its exact Linux package builds.
``qmhub2-mlpui-constraints.txt`` protects the numerical and ML compatibility
pins. ``qmhub2-mlpui-pip-lock.txt`` records the resolved pip dependency overlay;
unchanged conda-owned distributions are supplied by the explicit conda record.
The overlay retains pip replacements such as setuptools 78.1.0 and has no
build-machine file URLs. QMHub and MLPUI source wheels are installed separately.
The original complete pip freeze remains in the ignored record
``build/local-env-20261116/qmhub2-mlpui-pip-freeze.txt``.

For a fresh prefix, install the explicit conda base, then install PyTorch from
its official CUDA 12.8 index before the remaining locked dependencies. For
example, from this repository, with a fresh, explicitly chosen target prefix::

   /home/van/Programs/miniforge3/bin/conda create --prefix /absolute/new/prefix --file devtools/conda-envs/qmhub2-mlpui-linux-64.explicit.txt
   /absolute/new/prefix/bin/python -I -m pip install --index-url https://download.pytorch.org/whl/cu128 --constraint devtools/conda-envs/qmhub2-mlpui-constraints.txt 'torch==2.11.0+cu128'
   /absolute/new/prefix/bin/python -I -m pip install --constraint devtools/conda-envs/qmhub2-mlpui-constraints.txt --requirement devtools/conda-envs/qmhub2-mlpui-pip-lock.txt

The conda base's setuptools is replaced by the pinned pip version to satisfy
PyTorch. Install the pinned MLPUI wheel with its full dependencies and the
constraints file; its source archive is available at::

   https://codeload.github.com/MoriyaSuwako2524/MLPUI/tar.gz/fae948d1c8d78889a27cba9d213ffaac620f03ca

Build QMHub from each existing immutable Git commit without switching branches.
Commit the intended source and documentation first: the builder archives only
the selected commit and excludes uncommitted changes. For a fresh deployment
directory, resolve both branches and build their separate wheels::

   vibing_commit=$(git rev-parse --verify 'vibing^{commit}')
   vibe_pui_commit=$(git rev-parse --verify 'vibe-pui^{commit}')
   /home/van/Programs/miniforge3/envs/qmhub2/bin/python -I -B devtools/scripts/build_pinned_qmhub_wheel.py --commit "$vibing_commit" --output-dir build/publish-reinstall-20261116/qmhub2
   /home/van/Programs/miniforge3/envs/qmhub2/bin/python -I -B devtools/scripts/build_pinned_qmhub_wheel.py --commit "$vibe_pui_commit" --output-dir build/publish-reinstall-20261116/qmhub2-mlpui

The builder requires a new output directory. It archives only the selected
commit, embeds its version independently of the working tree, and verifies
wheel contents and hashes. Install each wheel noneditably into its matching
environment with ``--force-reinstall --no-deps`` after resolving dependencies,
then copy the ABI-matched helPME binary into the installed ``qmhub`` package.
Native binaries are excluded from the source wheel. A QMHub-only reinstall
preserves the dependency stack and does not install MLPUI into ``qmhub2``.

Actual build manifests, wheels, source archives, install reports, package
inventories, and validation artifacts for the initial installation are retained
in the ignored directory ``build/local-env-20261116``. The branch-based
reinstall uses ``build/publish-reinstall-20261116``. Keep a fresh output root
for subsequent deployments rather than overwriting earlier evidence. Each
build's ``manifest.json`` identifies its exact source commit, version, runtime
hashes, and wheel hash; installed-identity reports must compare the complete
runtime file set with that manifest independently of checkout imports.

Initial validation and diagnosis (historical)
---------------------------------------------

The following checks passed for the initial snapshots listed above:

* Isolated installed-package identity: 47 baseline and 48 MLPUI-branch runtime
  Python files match their pinned Git snapshots; both dependency checks pass.
* Native-enabled QMHub suites: 48 baseline tests and 51 MLPUI-branch tests.
* Six installed MLPUI checks, including all 145 pinned MLPUI Python files,
  NumPy/Torch interoperability, real adapter/unit conversion checks, native
  reciprocal PME, CUDA float64 autograd, and CPU/GPU neighbor agreement.
* Native reciprocal PME spatial gradients match finite differences, and a
  separate Fourier-sum oracle agrees to about ``4.4e-9`` in the probe's units.
* Actual CUDA execution on the RTX 3080, driver 580.173.02, CUDA runtime 12.8.
* 38 module checks, including both load orders with local ``sander/26``,
  ``qchem/6.1``, and ``qchem/6.3``, source/binary identity, conflicts, and unload
  restoration. Backend programs were not launched.

Python syntax and whitespace checks passed for the added setup tools. A local
Sphinx build was not run because Sphinx and docutils are absent; no documentation
packages were installed into the existing environments for that purpose.

The initial two upstream MLPUI analytic/synthetic test files produced **13 passes
and one failure**. The failure was
``test_qmhub_energy_gradient_consistency[True]``. The nonperiodic case passed.
For the first tested periodic QM component, the returned gradient was
``0.00249803848761397`` hartree/bohr, versus finite difference
``0.002492983692493033``: a discrepancy of about ``5.05e-6`` hartree/bohr.

The identical failure reproduced with the initial baseline ``qmhub2`` source.
An independent core-only analytic calculation bypassing MLPUI reproduced it
too. The finite PME grid is slightly sensitive to translation; QMHub recenters
coordinates on the QM centroid, while the pre-fix returned gradient omitted the
resulting centroid chain-rule contribution. At fixed origin the same component
agreed with finite differences to about ``3e-13`` hartree/bohr. A process-only
96-cubed mesh diagnostic and direct Ewald passed all nine tested components. These are
diagnostic comparisons, not production configuration changes or a general
accuracy guarantee.

The separately approved correction and its candidate validation are described in
:doc:`recentering_gradient_fix`. It maps the final gradient through the existing
coordinate transformation. **Returned QM gradients change**; energies and MM
gradients at the same input geometry are unchanged. PME defaults and test
tolerances are unchanged. The candidate passed all 14 upstream tests, including
the periodic case; its nine periodic finite-difference components had maximum
error ``1.87e-12`` hartree/bohr at the original mesh and tolerance. These candidate
results do not substitute for checking the final installed wheels.

Reinstall acceptance checks
---------------------------

Record these results for each final wheel under the reinstall artifact root:

* Verify isolated installed-source identity against the matching build manifest,
  including metadata and generated version, all runtime files, and the unchanged
  native binary hash. Run ``pip check`` without changing dependencies.
* Run the matching archived QMHub tests against the installed package with
  native helPME required: 63 tests for ``vibing`` and 66 for ``vibe-pui`` on
  Linux, including all 15 recentering regressions. Guard imports in subprocesses
  as well as the main test process; retained archive files provide test fixtures,
  not the runtime source under test.
* Execute native reciprocal PME and its finite-difference/Fourier checks in both
  environments. Verify that the native implementation, rather than the
  direct-Ewald fallback, was selected.
* Run the unchanged 14 upstream MLPUI tests separately for each QMHub wheel.
  For baseline ``qmhub2``, use MLPUI dependencies from the new environment in an
  isolated process selecting only the baseline QMHub package. This exercises
  MLPUI's compatibility launcher, not native MLPUI registration on ``vibing``.
* Repeat the six MLPUI/CUDA checks in ``qmhub2-mlpui`` and module selection,
  coexistence, conflicts, and unload checks against both new installed commits.

The retained initial failure is diagnostic history. The correction addresses
that fixture's missing coordinate derivative; production mesh convergence and
trained-potential accuracy remain separate validation requirements.

The reusable installed-MLPUI probe can be run as follows (the GPU must be
visible to the shell)::

   /home/van/Programs/miniforge3/envs/qmhub2-mlpui/bin/python -I -B devtools/scripts/validate_mlpui_environment.py --baseline-helpmelib /home/van/Programs/miniforge3/envs/qmhub2/lib/python3.12/site-packages/qmhub/helpmelib.cpython-312-x86_64-linux-gnu.so --mlpui-source build/local-env-20261116/MLPUI-fae948d1c8d78889a27cba9d213ffaac620f03ca --output build/publish-reinstall-20261116/validation-mlpui.json

Remaining manual validation
---------------------------

As requested, no licensed or external QM/MM calculations were run. Local
``amber/23-panxl`` was excluded because it is absent. Local Q-Chem 7 has no
module yet. Module coexistence and synthetic parser checks do not certify
real Sander or Q-Chem 6/7 output compatibility.

Future validation needs explicitly authorized Sander serial/MPI text, binary,
and FIFO cases, authentic Q-Chem 6/7 energy/QM-gradient/MM-potential-field outputs,
and a trained MLPUI specification/checkpoints with trusted reference results.
No OSU, HPCC, or OU deployments were performed.
