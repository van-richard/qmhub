#!/usr/bin/env python3
"""Check AT26 cap-storage patches using user-supplied Amber source fixtures.

Fixtures remain outside this repository because Amber source has separate
licensing. No compiler or scientific executable is invoked. Matching is exact:
line offsets, fuzz, or a second application cannot silently pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

TARGETS = (
    'AmberTools/src/sqm/qmmm_struct_module.F90',
    'AmberTools/src/sqm/qm_zero_charges.F90',
    'AmberTools/src/sander/cewmod.F90',
    'AmberTools/src/sander/putvalues.F90',
)
ROOT = Path(__file__).resolve().parents[2]


def sections(text):
    """Return file hunks without relying on vendor source line contents."""
    lines = text.splitlines(keepends=True)
    result = {}
    for index, line in enumerate(lines):
        if not line.startswith('--- '):
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith('+++ b/'):
            raise ValueError('Invalid paired patch headers')
        name = lines[index + 1][6:].rstrip('\n').split('\t', 1)[0]
        end = next((i for i in range(index + 2, len(lines))
                    if lines[i].startswith(('diff --git ', 'diff -ruN ', '--- '))), len(lines))
        if name in result:
            raise ValueError('Duplicate patch file section: ' + name)
        result[name] = parse_hunks(lines[index + 2:end])
    return result


def parse_hunks(lines):
    hunks = []
    index = 0
    while index < len(lines):
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[index])
        if not match:
            if lines[index].strip():
                raise ValueError('Unexpected text before patch hunk')
            index += 1
            continue
        start, count, new_start, new_count = map(int, (
            match[1], match[2] or '1', match[3], match[4] or '1'))
        index += 1
        body = []
        while index < len(lines) and not lines[index].startswith('@@'):
            line = lines[index]
            if line.startswith('\\ No newline'):
                raise ValueError('No-newline patch markers are unsupported for these fixtures')
            if line[:1] in (' ', '+', '-'):
                body.append(line)
            elif line.strip():
                raise ValueError('Unexpected text within patch hunk')
            index += 1
        if sum(line[0] in (' ', '-') for line in body) != count:
            raise ValueError('Old hunk length differs from header')
        if sum(line[0] in (' ', '+') for line in body) != new_count:
            raise ValueError('New hunk length differs from header')
        hunks.append((start, count, new_start, new_count, body))
    return hunks


def apply(text, hunks, reverse=False):
    """Apply exact unified hunks; never search for a nearby matching context."""
    lines = text.splitlines(keepends=True)
    delta = 0
    for start, count, new_start, new_count, body in hunks:
        if reverse:
            start, count = new_start, new_count
        remove, add = ('+', '-') if reverse else ('-', '+')
        old = [line[1:] for line in body if line[0] in (' ', remove)]
        new = [line[1:] for line in body if line[0] in (' ', add)]
        position = start - (1 if count else 0) + delta
        if lines[position:position + len(old)] != old:
            raise ValueError('Exact patch context mismatch at line ' + str(start))
        lines[position:position + len(old)] = new
        delta += len(new) - len(old)
    return ''.join(lines)


def apply_files(files, patch, reverse=False):
    return {name: apply(text, patch.get(name, []), reverse) for name, text in files.items()}


def source_contract(files):
    """Check dynamic storage and guard placement, not a fixed molecule size."""
    structure = files[TARGETS[0]]
    structure = structure[structure.index('  subroutine new_qmmm_struct_type('):
                          structure.index('  end subroutine new_qmmm_struct_type')]
    compact = re.sub(r'\s+', '', structure.lower())
    required = (
        'elseif(size(self%qm_resp_charges)<nquant+nlink)then',
        'allocate(new_resp_charges(nquant+nlink),stat=ier)',
        'new_resp_charges(:)=0.0d0',
        'new_resp_charges(1:old_resp_size)=self%qm_resp_charges(:)',
        'self%qm_resp_charges=>new_resp_charges',
        'nullify(new_resp_charges)',
    )
    if not all(text in compact for text in required):
        raise ValueError('Required generic charge-storage growth contract missing')
    writer = files[TARGETS[1]]
    writer = writer[writer.index('subroutine qm_zero_mm_link_pair_main_chg('):
                    writer.index('end subroutine qm_zero_mm_link_pair_main_chg')]
    associated = writer.index('if ( .not. associated(qmmm_struct%qm_resp_charges) ) then')
    capacity = writer.index('resp_size = size(qmmm_struct%qm_resp_charges)')
    required_capacity = writer.index('if ( resp_size < qmmm_struct%nquant + nlink ) then')
    stores = [match.start() for match in re.finditer(
        r'qm_resp_charges\(qmmm_struct%nquant\+i\)\s*=\s*charges\(y\)', writer)]
    if len(stores) != 2 or not associated < capacity < required_capacity < min(stores):
        raise ValueError('Writer must guard association and capacity before both cap-store branches')
    if 'end if' not in writer[associated:capacity]:
        raise ValueError('Association check must finish before SIZE evaluation')
    return {'dynamic_allocation': 'nquant+nlink', 'preserves_existing_entries': True,
            'initializes_added_entries': True, 'separate_association_and_capacity_guards': True,
            'both_writer_branches_guarded': True}


def validate(fixture_root, fixed_source):
    before = {name: (fixture_root / 'base' / name).read_text() for name in TARGETS}
    shared_path = ROOT / 'patches/at26_intel/sqm_at26.patch'
    optional_path = ROOT / 'patches/at26_intel/qmmm_internal_link_scratch_at26.patch'
    shared = sections(shared_path.read_text())
    optional = sections(optional_path.read_text())
    if TARGETS[0] in optional:
        raise ValueError('Optional patch still duplicates mandatory charge-storage growth')
    results = {}
    defaults = []
    for family, filename in [('gnu', 'at26_gnu/qmhub_at26_gnu.patch'),
                             ('intel', 'at26_intel/qmhub_at26.patch')]:
        primary_path = ROOT / 'patches' / filename
        primary = sections(primary_path.read_text())
        default = apply_files(apply_files(before, primary), shared)
        for name in TARGETS[:2]:
            if default[name] != (fixed_source / Path(name).name).read_text():
                raise ValueError('Default patch output differs from canonical source: ' + family + '/' + name)
        contract = source_contract(default)
        with_optional = apply_files(default, optional)
        if with_optional[TARGETS[0]] != default[TARGETS[0]]:
            raise ValueError('Optional patch changes the mandatory storage repair')
        source_contract(with_optional)
        if apply_files(apply_files(default, shared, True), primary, True) != before:
            raise ValueError('Default patch reverse round-trip failed')
        if apply_files(with_optional, optional, True) != default:
            raise ValueError('Optional patch reverse round-trip failed')
        defaults.append(default)
        results[family] = dict(default_matches_canonical=True, source_contract=contract,
            exact_default_reverse_roundtrip=True, exact_optional_reverse_roundtrip=True,
            optional_compatible=True, source_files=len(TARGETS))
    if defaults[0] != defaults[1]:
        raise ValueError('GNU and Intel relevant-file results differ')
    paths = [shared_path, optional_path, ROOT / 'patches/at26_gnu/qmhub_at26_gnu.patch',
             ROOT / 'patches/at26_intel/qmhub_at26.patch']
    return dict(status='PASS_AT26_RELEVANT_SOURCE_PATCH_COMPATIBILITY', checks=results,
                patch_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                scope='Four user-supplied relevant Amber files; source-only exact checks. No compiler, Amber, QMHub, or Q-Chem execution. Not a full-source or Intel runtime validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--fixed-source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.fixture_root, args.fixed_source)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
