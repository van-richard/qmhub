"""Q-Chem scratch-file compatibility and successful-read I/O budgets."""

from collections import Counter
import errno
from pathlib import Path

import numpy as np
import pytest

from qmhub.qmtools.qchem import QChem


MODERN = ("save/1521.0", "save/329.0")
LEGACY = ("save/5001.0", "save/5002.0")


def make_parser(tmp_path, n_mm=2):
    # Exercise parsing without constructing an engine or changing QCSCRATCH.
    parser = QChem.__new__(QChem)
    parser.cwd = tmp_path
    parser.mm_charges = np.ones(n_mm)
    return parser


def write_pair(tmp_path, names, potential=None, field=None):
    potential = np.asarray([1.0, 2.0] if potential is None else potential, dtype="f8")
    if field is None:
        field = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    field = np.asarray(field, dtype="f8")
    paths = tuple(tmp_path / name for name in names)
    for path, values in zip(paths, (potential, field)):
        path.parent.mkdir(parents=True, exist_ok=True)
        values.tofile(path)
    return paths, potential, field


@pytest.mark.parametrize("names", [MODERN, LEGACY])
def test_success_stats_each_candidate_once_without_diagnostics(tmp_path, monkeypatch, names):
    parser = make_parser(tmp_path)
    paths, potential, field = write_pair(tmp_path, names)
    candidates = {tmp_path / name for pair in (MODERN, LEGACY) for name in pair}
    stat_counts = Counter()
    output_allocations = []
    original_stat = Path.stat
    original_zeros = np.zeros

    def counted_stat(path, *args, **kwargs):
        if path in candidates:
            stat_counts[path] += 1
        return original_stat(path, *args, **kwargs)

    def counted_zeros(shape, *args, **kwargs):
        if shape == (4, 2):
            output_allocations.append(shape)
        return original_zeros(shape, *args, **kwargs)

    def unexpected_diagnostics(*args, **kwargs):
        pytest.fail("Successful reads must not construct failure diagnostics")

    monkeypatch.setattr(Path, "stat", counted_stat)
    monkeypatch.setattr(np, "zeros", counted_zeros)
    monkeypatch.setattr(QChem, "_describe_binary_mm_esp_pair", unexpected_diagnostics)
    result = parser._get_mm_esp()

    assert all(count == 1 for count in stat_counts.values())
    assert all(stat_counts[path] == 1 for path in paths)
    assert output_allocations == [(4, 2)]
    np.testing.assert_array_equal(result[0], potential)
    np.testing.assert_array_equal(result[1:], -field.T)


def test_zero_mm_does_not_probe_or_read_files(tmp_path, monkeypatch):
    parser = make_parser(tmp_path, n_mm=0)

    def unexpected_io(*args, **kwargs):
        pytest.fail("Zero-MM parsing must not access scratch files")

    monkeypatch.setattr(Path, "stat", unexpected_io)
    monkeypatch.setattr(np, "fromfile", unexpected_io)
    monkeypatch.setattr(QChem, "_describe_binary_mm_esp_pair", unexpected_io)
    assert parser._get_mm_esp().shape == (4, 0)


def test_modern_pair_keeps_priority_and_leaves_legacy_files(tmp_path):
    parser = make_parser(tmp_path)
    modern_paths, potential, field = write_pair(tmp_path, MODERN)
    legacy_paths, _, _ = write_pair(tmp_path, LEGACY, [9.0, 8.0], np.ones((2, 3)))
    legacy_bytes = [path.read_bytes() for path in legacy_paths]

    result = parser._get_mm_esp()

    np.testing.assert_array_equal(result[0], potential)
    np.testing.assert_array_equal(result[1:], -field.T)
    assert not any(path.exists() for path in modern_paths)
    assert [path.read_bytes() for path in legacy_paths] == legacy_bytes


def test_rejected_modern_pair_is_retained_when_legacy_trailing_rows_are_read(tmp_path):
    parser = make_parser(tmp_path)
    modern_paths, _, _ = write_pair(tmp_path, MODERN, [1.0, 2.0, 3.0], np.ones((3, 3)))
    modern_bytes = [path.read_bytes() for path in modern_paths]
    legacy_paths, potential, field = write_pair(
        tmp_path, LEGACY, [-10.0, -11.0, 4.0, 5.0], np.arange(12.0).reshape(4, 3)
    )

    result = parser._get_mm_esp()

    np.testing.assert_array_equal(result[0], potential[-2:])
    np.testing.assert_array_equal(result[1:], -field[-2:].T)
    assert [path.read_bytes() for path in modern_paths] == modern_bytes
    assert not any(path.exists() for path in legacy_paths)


@pytest.mark.parametrize("direct", [False, True])
def test_explicit_reader_remains_strict_for_legacy_filenames(tmp_path, direct):
    parser = make_parser(tmp_path)
    paths, _, _ = write_pair(tmp_path, LEGACY, [1.0, 2.0, 3.0], np.ones((3, 3)))
    before = [path.read_bytes() for path in paths]

    with pytest.raises(ValueError, match="Invalid Q-Chem binary MM ESP files"):
        if direct:
            QChem._read_binary_mm_esp(*paths, 2)
        else:
            parser._get_mm_esp(output=LEGACY)

    assert [path.read_bytes() for path in paths] == before


def test_direct_reader_preserves_explicit_trailing_row_permission(tmp_path):
    paths, potential, field = write_pair(
        tmp_path, LEGACY, [1.0, 2.0, 3.0], np.arange(9.0).reshape(3, 3)
    )

    result = QChem._read_binary_mm_esp(*paths, 2, allow_trailing=True)

    np.testing.assert_array_equal(result[0], potential[-2:])
    np.testing.assert_array_equal(result[1:], -field[-2:].T)
    assert not any(path.exists() for path in paths)


def test_direct_empty_pair_is_valid_and_consumed(tmp_path):
    paths, _, _ = write_pair(tmp_path, MODERN, [], np.empty((0, 3)))
    assert QChem._valid_binary_mm_esp_pair(*paths, 0)

    result = QChem._read_binary_mm_esp(*paths, 0)

    assert result.shape == (4, 0)
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("short_file", [0, 1])
def test_short_read_cannot_broadcast_or_delete_either_file(tmp_path, monkeypatch, direct, short_file):
    parser = make_parser(tmp_path)
    paths, _, _ = write_pair(tmp_path, MODERN)
    before = [path.read_bytes() for path in paths]
    original_fromfile = np.fromfile

    def shortened_read(path, *args, **kwargs):
        values = original_fromfile(path, *args, **kwargs)
        if Path(path) == paths[short_file]:
            # One scalar/one field row could otherwise silently broadcast to
            # all MM atoms if a file shrinks after its metadata was checked.
            return values[:1] if short_file == 0 else values[:3]
        return values

    monkeypatch.setattr(np, "fromfile", shortened_read)
    with pytest.raises(ValueError, match="Incomplete Q-Chem binary MM ESP read"):
        if direct:
            QChem._read_binary_mm_esp(*paths, 2)
        else:
            parser._get_mm_esp()

    assert [path.read_bytes() for path in paths] == before


def test_read_error_retains_both_files(tmp_path, monkeypatch):
    paths, _, _ = write_pair(tmp_path, MODERN)
    before = [path.read_bytes() for path in paths]
    original_fromfile = np.fromfile

    def failed_field_read(path, *args, **kwargs):
        if Path(path) == paths[1]:
            raise OSError("simulated field read failure")
        return original_fromfile(path, *args, **kwargs)

    monkeypatch.setattr(np, "fromfile", failed_field_read)
    with pytest.raises(OSError, match="simulated field read failure"):
        QChem._read_binary_mm_esp(*paths, 2)

    assert [path.read_bytes() for path in paths] == before


def test_missing_pair_descriptions_are_deferred_until_failure(tmp_path, monkeypatch):
    parser = make_parser(tmp_path)
    descriptions = []
    original_describe = QChem._describe_binary_mm_esp_pair

    def described(*args, **kwargs):
        descriptions.append(args[:2])
        return original_describe(*args, **kwargs)

    monkeypatch.setattr(QChem, "_describe_binary_mm_esp_pair", staticmethod(described))
    with pytest.raises(FileNotFoundError, match="Could not find valid Q-Chem MM ESP output") as error:
        parser._get_mm_esp()

    assert descriptions == [tuple(tmp_path / name for name in pair) for pair in (MODERN, LEGACY)]
    assert all(str(path) in str(error.value) for pair in descriptions for path in pair)


def test_unavailable_path_keeps_legacy_fallback(tmp_path, monkeypatch):
    parser = make_parser(tmp_path)
    _, potential, field = write_pair(tmp_path, LEGACY)
    unavailable = tmp_path / MODERN[0]
    original_stat = Path.stat

    def looped_path(path, *args, **kwargs):
        if path == unavailable:
            raise OSError(errno.ELOOP, "simulated symlink loop")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", looped_path)
    result = parser._get_mm_esp()

    np.testing.assert_array_equal(result[0], potential)
    np.testing.assert_array_equal(result[1:], -field.T)


def test_metadata_permission_errors_are_not_hidden(tmp_path, monkeypatch):
    potential = tmp_path / MODERN[0]
    field = tmp_path / MODERN[1]

    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "simulated access denial")

    monkeypatch.setattr(Path, "stat", denied)
    with pytest.raises(PermissionError, match="simulated access denial"):
        QChem._binary_mm_esp_row_count(potential, field)
