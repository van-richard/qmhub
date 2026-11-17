# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests, 
conda installation, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

You should test your code, but do not feel compelled to use these specific programs. You also may not need Unix and 
Windows testing if you only plan to deploy on specific platforms. These are just to help you get started.

### Conda Environment:

This directory contains the files to setup the Conda environment for testing purposes

* `conda-envs`: directory containing the YAML file(s) which fully describe Conda Environments, their dependencies, and those dependency provenance's
  * `test_env.yaml`: Simple test environment file with base dependencies. It declares the `conda-forge` and `defaults` channels used by CI.
  * `qmhub2-mlpui.yaml`: Python 3.12.13 numerical base for the local MLPUI stack;
    accompanying constraints and Linux explicit conda list preserve the numerical
    base. The pip lock contains only its tested pip overlay, including the
    setuptools replacement. Follow [local setup notes](../docs/local_environment_setup.rst)
    for branch-specific source wheels, deployment manifests, and the separate
    helPME binary copy.
  
### Additional Scripts:

This directory contains OS agnostic helper scripts which don't fall in any of the previous categories
* `scripts`
  * `create_conda_env.py`: Helper program for spinning up new conda environments based on a starter file with Python Version and Env. Name command-line options
  * `build_pinned_qmhub_wheel.py`: Build and verify an existing immutable Git
    commit without switching branches or including native binaries. Commit source
    changes before building; retain the manifest for installed-source verification.
  * `validate_mlpui_environment.py`: Check installed MLPUI source identity,
    NumPy/Torch interoperability, native PME, the real analytic MLPUI adapter,
    and CUDA calculations without external QM jobs. QMHub Git identity is
    verified separately against its build manifest.
  * `check_qmmm_stack.py`: Inspect the selected Python/QMHub, Sander, and
    Q-Chem installation without starting a licensed calculation. Run it after
    loading the intended modules, using the environment's Python:

    ```bash
    python -I devtools/scripts/check_qmmm_stack.py
    python -I devtools/scripts/check_qmmm_stack.py --json
    ```

    The checker reports interpreter/package/launcher locations, Amber and
    Q-Chem settings, thread controls, and Linux native linkage. It flags mixed
    installations and missing runtime libraries; missing optional `sander.MPI`
    is a warning. `--amber-home PREFIX` selects the intended Sander installation
    for comparison, while stale shell `AMBERHOME` settings still fail. Linux
    linkage inspection uses `ldd` only on the selected installation's resolved
    ELF executables; other platforms skip this check. The tool does not import
    QMHub, modify environment settings, load modules, or launch Amber/Q-Chem.
    It exits with status 0 when required checks pass and 1 when they fail.
    It is never called automatically during QM/MM calculations.

  * `validate_at26_cap_patches.py`: Validate required AT26 cap-storage growth
    and writer guards, exact GNU/Intel relevant-file patch application and
    optional-patch compatibility using external Amber source fixtures. It runs
    no compiler or scientific executable. Parser tests are in
    `tests/test_at26_cap_patches.py`; runtime allocation tests require a separate
    bounds-enabled Fortran build.

* `modulefiles/lynnx`: Lua templates selecting the baseline `qmhub2` and
  `qmhub2-mlpui` executables through Lmod, with reversible Python search settings.

### External Amber/QMHub smoke tests:

* `amber-qmhub-tests`: repository-local smoke tests for an already installed
  and patched AmberTools setup. These tests launch `sander` or `sander.MPI`
  and are not installed into `site-packages` as package unit tests. See
  `amber-qmhub-tests/README.md` for the required checkout-based workflow,
  Amber input files, environment variables, and optional Q-Chem settings.


## How to contribute changes
- Clone the repository if you have write access to the main repo, fork the repository if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code
- Ensure that the test environment dependencies (`conda-envs`) line up with the project dependencies in `pyproject.toml`
- Push the branch to the repo (either the main or your fork) with `git push -u origin {your branch name}`
  * Note that `origin` is the default name assigned to the remote, yours may be different
- Make a PR on GitHub with your changes
- We'll review the changes and get your code into the repo after lively discussion!


## Checklist for updates
- [ ] Make sure there is an/are issue(s) opened for your specific update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR as needed until tests pass
- [ ] Tag the final, debugged version 
   *  `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged in

## Versioningit auto-version
[versioningit](https://github.com/jwodder/versioningit) infers the installed
version from `git` tags using the configuration in `pyproject.toml`. Builds may
write the generated version module to `qmhub/_version.py`; that file is ignored
as local build output.
