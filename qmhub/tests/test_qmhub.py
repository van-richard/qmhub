"""
Unit and regression test for the qmhub package.
"""

# Import package, test suite, and other packages as needed
import numpy as np
import qmhub
import pytest
import sys

from qmhub.utils.darray import DependArray


def test_qmhub_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qmhub" in sys.modules


def test_depend_array_operators():
    a = DependArray([1.0, 2.0, 4.0])
    b = DependArray([2.0, 2.0, 2.0])

    assert np.allclose(np.asarray(a / 2), [0.5, 1.0, 2.0])
    assert np.allclose(np.asarray(2 / a), [2.0, 1.0, 0.5])
    assert np.allclose(np.asarray(a + b), [3.0, 4.0, 6.0])
    assert np.allclose(np.asarray(a * b), [2.0, 4.0, 8.0])
    assert np.array_equal(np.asarray(a > 1), [False, True, True])

    a /= 2
    assert np.allclose(np.asarray(a), [0.5, 1.0, 2.0])


def test_depend_array_indexing_and_cache_invalidation():
    source = DependArray([1.0, 2.0, 4.0])
    dependent = DependArray(func=lambda array: np.asarray(array) * 2, dependencies=[source])

    assert source[1] == 2.0
    assert np.allclose(np.asarray(dependent), [2.0, 4.0, 8.0])

    source[1] = 3.0

    assert np.allclose(np.asarray(dependent), [2.0, 6.0, 8.0])
