import os
import subprocess as sp

_THREAD_ENV_VARS = (
    "QCTHREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
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


def get_openmp_threads():
    """Get the thread count for backends that only honor OMP_NUM_THREADS."""
    nthreads = _get_positive_int_env("OMP_NUM_THREADS") or 1

    if "OMP_NUM_THREADS" in os.environ and _get_positive_int_env("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)

    return nthreads
