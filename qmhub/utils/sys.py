import os
import subprocess as sp

# Threaded QM backends should use explicit thread variables only. Scheduler
# task-count variables describe MPI layout and can oversubscribe OpenMP codes.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "QCTHREADS",
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


def get_nproc():
    """Get the number of processes for QM calculation."""
    nproc = 1

    for name in _THREAD_ENV_VARS:
        value = _get_positive_int_env(name)
        if value is not None:
            nproc = value
            break

    # Some launchers export OMP_NUM_THREADS as an empty string. Normalize it
    # before OpenMP-backed libraries see an invalid environment value.
    if "OMP_NUM_THREADS" in os.environ and _get_positive_int_env("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = str(nproc)

    return nproc
