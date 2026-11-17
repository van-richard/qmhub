"""Exact patch checks without Amber sources, compilers or scientific programs."""
import difflib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/validate_at26_cap_patches.py'
SPEC = importlib.util.spec_from_file_location('at26_cap_patch_checks', SCRIPT)
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


@pytest.mark.parametrize('length', [0, 1, 5, 257])
@pytest.mark.parametrize('location', ['start', 'middle', 'end'])
def test_exact_insertion_roundtrip_for_varied_source_lengths(length, location):
    old = [f'line {index}\n' for index in range(length)]
    position = {'start': 0, 'middle': length // 2, 'end': length}[location]
    new = old[:position] + ['added\n', 'second addition\n'] + old[position:]
    patch = ''.join(difflib.unified_diff(old, new, fromfile='a/example.F90', tofile='b/example.F90'))
    hunks = CHECKS.sections(patch)['example.F90']
    assert CHECKS.apply(''.join(old), hunks) == ''.join(new)
    assert CHECKS.apply(''.join(new), hunks, reverse=True) == ''.join(old)


def test_context_drift_fails_instead_of_searching_nearby_lines():
    old = 'first\nsecond\nthird\n'
    new = 'first\nreplacement\nthird\n'
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                       fromfile='a/example.F90', tofile='b/example.F90'))
    hunks = CHECKS.sections(patch)['example.F90']
    with pytest.raises(ValueError, match='Exact patch context mismatch'):
        CHECKS.apply('unrelated\n' + old, hunks)


def test_malformed_hunk_count_is_rejected():
    patch = '--- a/example.F90\n+++ b/example.F90\n@@ -1,2 +1 @@\n-old\n+new\n'
    with pytest.raises(ValueError, match='Old hunk length'):
        CHECKS.sections(patch)


def test_new_file_section_does_not_merge_into_previous_hunk():
    patch = ('--- a/example.F90\n+++ b/example.F90\n@@ -1 +1 @@\n-old\n+new\n'
             '--- /dev/null\n+++ b/new.F90\n@@ -0,0 +1 @@\n+created\n')
    sections = CHECKS.sections(patch)
    assert CHECKS.apply('old\n', sections['example.F90']) == 'new\n'
    assert CHECKS.apply('', sections['new.F90']) == 'created\n'
