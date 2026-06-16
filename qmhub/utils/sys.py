import os
import subprocess as sp

# Keep process and thread controls separate. SLURM_NTASKS and PBS_NP are
# process counts for process-based backends such as ORCA, not Q-Chem OpenMP
# thread counts.
_PROCESS_ENV_VARS = (
    "SLURM_NTASKS",
    "PBS_NP",
    "NCPUS",
)
# SLURM_CPUS_PER_TASK is the CPU allocation for one scheduler task, so it is a
# valid fallback for threaded Q-Chem runs after explicit thread variables.
_THREAD_ENV_VARS = (
    "QCTHREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
)


def run_cmdline(cmdline):
    """Run QM calculation."""

    proc = sp.Popen(args=cmdline, shell=True)
    proc.wait()
    return proc.returncode


def _get_positive_int_env(name):
    # Empty strings from launchers such as AMBER/srun should behave like an
    # unset variable, not crash int() during QMHub initialization.
    value = os.environ.get(name)
    if value is None:
        return None

    try:
        value = int(value)
    except ValueError:
        return None

    if value > 0:
        return value
    return None


def _get_first_positive_int_env(names):
    for name in names:
        value = _get_positive_int_env(name)
        if value is not None:
            return value
    return None


def get_nthreads():
    """Get the number of threads for threaded QM calculations."""
    nthreads = _get_first_positive_int_env(_THREAD_ENV_VARS) or 1

    # Some launchers export OMP_NUM_THREADS as an empty string. Normalize it
    # before OpenMP-backed libraries see an invalid environment value.
    if "OMP_NUM_THREADS" in os.environ and _get_positive_int_env("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)

    return nthreads


def get_nproc():
    """Get the number of processes for QM calculation."""
    # Process-oriented backends such as ORCA should use scheduler task counts.
    # Thread variables remain only as a fallback for non-scheduler launches.
    return _get_first_positive_int_env(_PROCESS_ENV_VARS + _THREAD_ENV_VARS) or 1
