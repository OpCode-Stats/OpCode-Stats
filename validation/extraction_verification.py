"""
Extraction correctness verification for OpCode-Stats.

Validates that the disassemble.py parsing pipeline produces results that
agree with a ground-truth, independent parse of raw objdump output.

Checks performed
----------------
1. Mnemonic count verification  — independent count vs extraction module count
2. Instruction boundary verification — per-function counts, monotonic addresses,
   multi-line continuation lines, (bad) pseudo-opcodes
3. Function boundary verification — raw function count vs extracted, PLT handling
4. Duplicate/hard-link filtering verification — filter_valid_binaries deduplication
5. Timeout verification — subprocess.TimeoutExpired path creates no Binary object

Usage
-----
    python3 validation/extraction_verification.py

Results are written to validation/results/extraction_verification.json.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, '/home/aaslyan/OpCode-Stats')

from extraction.disassemble import (
    DisassemblyError,
    disassemble_binary,
    parse_objdump_output,
    run_objdump,
)
from utils.helpers import filter_valid_binaries

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('extraction_verification')

# ── constants ────────────────────────────────────────────────────────────────
SAMPLE_BINARIES = ['grep', 'bash', 'find', 'tar', 'xz']
USR_BIN = Path('/usr/bin')
RESULTS_DIR = Path('/home/aaslyan/OpCode-Stats/validation/results')
OUTPUT_JSON = RESULTS_DIR / 'extraction_verification.json'

# Same patterns as disassemble.py — used only for the independent reference parser.
# The reference parser is deliberately simpler (line-by-line, no dataclasses) so
# it cannot share bugs with the extraction module.
_REF_FUNC_RE = re.compile(r'^([0-9a-f]+) <(.+?)>:$')
_REF_INSTR_RE = re.compile(
    r'^\s*([0-9a-f]+):\s*([0-9a-f ]+)\s+([a-zA-Z][a-zA-Z0-9]*)(.*)?$'
)
# Continuation lines: address + hex bytes only, no mnemonic
# e.g. "   31c82:	00 "
_CONTINUATION_RE = re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2} )+\s*$')


# ─────────────────────────────────────────────────────────────────────────────
# Reference parser (independent of extraction module)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_mnemonic(raw: str) -> str:
    """Apply the same normalization as Instruction.__post_init__."""
    return raw.split('@')[0].split('.')[0].lower()


class _RefFunction:
    """Lightweight function record from the reference parser."""
    __slots__ = ('name', 'address', 'instructions')

    def __init__(self, name: str, address: int) -> None:
        self.name = name
        self.address = address
        # Each element: (address: int, mnemonic: str)
        self.instructions: List[Tuple[int, str]] = []


def reference_parse(raw_output: str) -> List[_RefFunction]:
    """
    Independent reference parser.

    Walks raw objdump text line-by-line without touching any extraction module
    code.  Returns a list of _RefFunction objects with parsed addresses and
    normalised mnemonics.
    """
    functions: List[_RefFunction] = []
    current: Optional[_RefFunction] = None

    for raw_line in raw_output.split('\n'):
        line = raw_line.strip()
        if not line:
            continue

        func_m = _REF_FUNC_RE.match(line)
        if func_m:
            if current is not None and current.instructions:
                functions.append(current)
            current = _RefFunction(
                name=func_m.group(2),
                address=int(func_m.group(1), 16),
            )
            continue

        if current is None:
            continue

        instr_m = _REF_INSTR_RE.match(line)
        if instr_m:
            addr = int(instr_m.group(1), 16)
            mnemonic = _normalize_mnemonic(instr_m.group(3))
            current.instructions.append((addr, mnemonic))

    if current is not None and current.instructions:
        functions.append(current)

    return functions


def _run_objdump_safe(binary_path: Path, timeout: int = 60) -> Optional[str]:
    """
    Run objdump and return stdout, or None on failure/timeout.

    This is a local wrapper that does not depend on extraction.run_objdump so
    timeout-path tests can control behaviour independently.
    """
    try:
        result = subprocess.run(
            ['objdump', '-d', '-M', 'intel', str(binary_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning('objdump non-zero exit %d for %s', result.returncode, binary_path)
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning('objdump timed out for %s', binary_path)
        return None
    except FileNotFoundError:
        logger.error('objdump not found')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Mnemonic count verification
# ─────────────────────────────────────────────────────────────────────────────

def check_mnemonic_counts(binary_paths: List[Path]) -> Dict[str, Any]:
    """
    For each binary, compare total mnemonic counts between the reference parser
    and the extraction module.

    A "mismatch" means the two parsers disagree on how many times a given
    mnemonic appears across the whole binary.
    """
    logger.info('=== Check 1: Mnemonic count verification ===')
    per_binary: Dict[str, Any] = {}
    overall_pass = True

    for bp in binary_paths:
        name = bp.name
        raw = _run_objdump_safe(bp)
        if raw is None:
            per_binary[name] = {'status': 'SKIP', 'reason': 'objdump failed'}
            continue

        # Reference counts
        ref_funcs = reference_parse(raw)
        ref_counts: Counter = Counter()
        for f in ref_funcs:
            for _, mnemonic in f.instructions:
                ref_counts[mnemonic] += 1

        # Extraction module counts
        ext_funcs = parse_objdump_output(raw, name)
        ext_counts: Counter = Counter()
        for f in ext_funcs:
            for instr in f.instructions:
                ext_counts[instr.mnemonic] += 1

        # Compare
        all_mnemonics = set(ref_counts) | set(ext_counts)
        mismatches: Dict[str, Dict[str, int]] = {}
        for m in sorted(all_mnemonics):
            r, e = ref_counts.get(m, 0), ext_counts.get(m, 0)
            if r != e:
                mismatches[m] = {'reference': r, 'extracted': e}

        ref_total = sum(ref_counts.values())
        ext_total = sum(ext_counts.values())
        binary_pass = (ref_total == ext_total and not mismatches)
        if not binary_pass:
            overall_pass = False

        per_binary[name] = {
            'status': 'PASS' if binary_pass else 'FAIL',
            'reference_total': ref_total,
            'extracted_total': ext_total,
            'mnemonic_mismatches': mismatches,
        }
        logger.info(
            '%s: ref=%d ext=%d mismatches=%d [%s]',
            name, ref_total, ext_total, len(mismatches),
            'PASS' if binary_pass else 'FAIL',
        )

    return {
        'status': 'PASS' if overall_pass else 'FAIL',
        'binaries': per_binary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — Instruction boundary verification
# ─────────────────────────────────────────────────────────────────────────────

def check_instruction_boundaries(binary_paths: List[Path]) -> Dict[str, Any]:
    """
    For each binary:
    - Compare per-function instruction counts between reference and extraction.
    - Verify that instruction addresses are monotonically increasing within
      each function in the extracted data.
    - Count continuation lines (multi-byte instructions split across lines).
    - Count (bad) pseudo-opcode lines that the instruction_pattern will skip.
    """
    logger.info('=== Check 2: Instruction boundary verification ===')
    per_binary: Dict[str, Any] = {}
    overall_pass = True

    for bp in binary_paths:
        name = bp.name
        raw = _run_objdump_safe(bp)
        if raw is None:
            per_binary[name] = {'status': 'SKIP', 'reason': 'objdump failed'}
            continue

        ref_funcs = reference_parse(raw)
        ext_funcs = parse_objdump_output(raw, name)

        # Index extracted functions by name for fast lookup.
        ext_by_name: Dict[str, Any] = {f.name: f for f in ext_funcs}

        func_count_mismatch = 0
        instr_count_mismatches: List[Dict[str, Any]] = []
        non_monotonic_functions: List[str] = []

        for ref_f in ref_funcs:
            ext_f = ext_by_name.get(ref_f.name)
            if ext_f is None:
                # Function present in reference but missing from extraction.
                # This is covered in check 3; skip here to avoid double-counting.
                continue

            ref_ic = len(ref_f.instructions)
            ext_ic = len(ext_f.instructions)
            if ref_ic != ext_ic:
                func_count_mismatch += 1
                instr_count_mismatches.append({
                    'function': ref_f.name,
                    'reference': ref_ic,
                    'extracted': ext_ic,
                })

            # Monotonic address check on extracted data.
            addrs = [i.address for i in ext_f.instructions]
            if addrs != sorted(addrs):
                non_monotonic_functions.append(ref_f.name)

        # Scan raw output for structural anomalies.
        continuation_count = 0
        bad_opcode_count = 0
        for raw_line in raw.split('\n'):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if _CONTINUATION_RE.match(raw_line):
                continuation_count += 1
            if '(bad)' in stripped:
                bad_opcode_count += 1

        binary_pass = (
            func_count_mismatch == 0
            and not non_monotonic_functions
        )
        if not binary_pass:
            overall_pass = False

        per_binary[name] = {
            'status': 'PASS' if binary_pass else 'FAIL',
            'functions_with_instr_count_mismatch': func_count_mismatch,
            'instr_count_mismatches': instr_count_mismatches[:20],  # cap output
            'non_monotonic_address_functions': non_monotonic_functions[:20],
            'continuation_lines_in_raw': continuation_count,
            'bad_opcode_lines_in_raw': bad_opcode_count,
        }
        logger.info(
            '%s: func_instr_mismatches=%d non_monotonic=%d '
            'continuation_lines=%d bad_opcode_lines=%d [%s]',
            name, func_count_mismatch, len(non_monotonic_functions),
            continuation_count, bad_opcode_count,
            'PASS' if binary_pass else 'FAIL',
        )

    return {
        'status': 'PASS' if overall_pass else 'FAIL',
        'binaries': per_binary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — Function boundary verification
# ─────────────────────────────────────────────────────────────────────────────

def check_function_boundaries(binary_paths: List[Path]) -> Dict[str, Any]:
    """
    Compare raw function counts against extracted function counts and detect:
    - Functions present in raw output but absent from extraction (zero-instruction
      functions are dropped by parse_objdump_output; that is expected behaviour).
    - Functions present in extraction but not in raw output (should never happen).
    - PLT stub prevalence.
    - Potential split functions (where the extraction module emits more functions
      than the raw output shows — not expected with current regex).
    """
    logger.info('=== Check 3: Function boundary verification ===')
    per_binary: Dict[str, Any] = {}
    overall_pass = True

    for bp in binary_paths:
        name = bp.name
        raw = _run_objdump_safe(bp)
        if raw is None:
            per_binary[name] = {'status': 'SKIP', 'reason': 'objdump failed'}
            continue

        # Raw function count: every line matching the function header pattern,
        # regardless of whether it contains instructions.
        raw_func_names: List[str] = []
        for line in raw.split('\n'):
            m = _REF_FUNC_RE.match(line.strip())
            if m:
                raw_func_names.append(m.group(2))

        raw_func_set = set(raw_func_names)
        raw_plt = [n for n in raw_func_names if '@plt' in n or n == '.plt']

        ext_funcs = parse_objdump_output(raw, name)
        ext_func_set = {f.name for f in ext_funcs}
        ext_plt = [f.name for f in ext_funcs if '@plt' in f.name]

        # Functions in raw but not extracted: expected only for zero-instruction
        # headers (e.g. section labels with no code lines underneath).
        missing_from_extraction = raw_func_set - ext_func_set
        # Functions extracted that do not appear in raw: a parser bug.
        extra_in_extraction = ext_func_set - raw_func_set

        # Determine which "missing" entries are benign (zero-instruction headers).
        ref_funcs = reference_parse(raw)
        ref_zero_instr = {f.name for f in ref_funcs if not f.instructions}
        # Build a set of raw headers that had no following instructions at all
        # (not picked up by the reference parser even before the zero filter).
        ref_names = {f.name for f in ref_funcs}
        raw_no_instructions = raw_func_set - ref_names  # never entered reference parser

        benign_missing = missing_from_extraction & (ref_zero_instr | raw_no_instructions)
        problematic_missing = missing_from_extraction - benign_missing

        binary_pass = (not extra_in_extraction and not problematic_missing)
        if not binary_pass:
            overall_pass = False

        per_binary[name] = {
            'status': 'PASS' if binary_pass else 'FAIL',
            'raw_function_count': len(raw_func_names),
            'extracted_function_count': len(ext_funcs),
            'raw_plt_stub_count': len(raw_plt),
            'extracted_plt_stub_count': len(ext_plt),
            # Benign: zero-instruction headers skipped by both parsers.
            'benign_missing_from_extraction': len(benign_missing),
            # Problematic: had instructions in raw but were not extracted.
            'problematic_missing_from_extraction': sorted(problematic_missing)[:20],
            # Should always be empty; any entry here is a parser bug.
            'extra_in_extraction_not_in_raw': sorted(extra_in_extraction)[:20],
        }
        logger.info(
            '%s: raw=%d extracted=%d plt_raw=%d plt_ext=%d '
            'benign_missing=%d problematic_missing=%d extra=%d [%s]',
            name,
            len(raw_func_names), len(ext_funcs),
            len(raw_plt), len(ext_plt),
            len(benign_missing), len(problematic_missing),
            len(extra_in_extraction),
            'PASS' if binary_pass else 'FAIL',
        )

    return {
        'status': 'PASS' if overall_pass else 'FAIL',
        'binaries': per_binary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — Duplicate/hard-link filtering verification
# ─────────────────────────────────────────────────────────────────────────────

def _find_hardlinked_pairs(search_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Scan search_dir for regular files that share an inode.  Returns a list of
    (path_a, path_b) pairs where path_a sorts first lexicographically.
    """
    inode_to_paths: Dict[int, List[Path]] = defaultdict(list)
    try:
        for entry in search_dir.iterdir():
            if entry.is_file() and not entry.is_symlink():
                try:
                    inode = entry.stat().st_ino
                    inode_to_paths[inode].append(entry)
                except OSError:
                    pass
    except PermissionError:
        logger.warning('Permission denied scanning %s', search_dir)

    pairs: List[Tuple[Path, Path]] = []
    for paths in inode_to_paths.values():
        if len(paths) >= 2:
            paths_sorted = sorted(paths)
            for i in range(len(paths_sorted) - 1):
                pairs.append((paths_sorted[i], paths_sorted[i + 1]))
    return pairs


def check_hardlink_filtering() -> Dict[str, Any]:
    """
    Locate actual hard-linked binary pairs in /usr/bin and verify that
    filter_valid_binaries deduplicates them correctly.

    Three scenarios are tested:
    A. Both members of a pair passed as names → only one should survive.
    B. Only the first member passed → it should survive.
    C. All known pairs collectively → unique inode count should match output length.
    """
    logger.info('=== Check 4: Duplicate/hard-link filtering verification ===')

    pairs = _find_hardlinked_pairs(USR_BIN)
    if not pairs:
        return {
            'status': 'SKIP',
            'reason': 'No hard-linked binaries found in /usr/bin',
            'hard_linked_pairs': [],
        }

    pair_records: List[Dict[str, Any]] = []
    for a, b in pairs[:10]:  # cap to first 10 pairs
        inode_a = a.stat().st_ino
        inode_b = b.stat().st_ino
        pair_records.append({
            'path_a': str(a),
            'path_b': str(b),
            'inode_a': inode_a,
            'inode_b': inode_b,
            'same_inode': inode_a == inode_b,
        })

    # Scenario A: pass both names of the first pair; expect only 1 result.
    a0, b0 = pairs[0]
    result_a = filter_valid_binaries([a0.name, b0.name], USR_BIN)
    scenario_a_pass = (len(result_a) == 1)

    # Scenario B: pass only the first name; expect 1 result.
    result_b = filter_valid_binaries([a0.name], USR_BIN)
    scenario_b_pass = (len(result_b) == 1)

    # Scenario C: all left-side names from found pairs; unique inodes == output len.
    all_names = [str(a.name) for a, _ in pairs[:10]] + [str(b.name) for _, b in pairs[:10]]
    result_c = filter_valid_binaries(all_names, USR_BIN)
    seen_inodes_c = set()
    for p in result_c:
        seen_inodes_c.add(p.resolve().stat().st_ino)
    scenario_c_pass = (len(seen_inodes_c) == len(result_c))

    overall_pass = scenario_a_pass and scenario_b_pass and scenario_c_pass

    result = {
        'status': 'PASS' if overall_pass else 'FAIL',
        'hard_linked_pairs_found': len(pairs),
        'hard_linked_pairs_sample': pair_records,
        'scenario_a': {
            'description': 'Both hard-linked names passed → expect 1 result',
            'inputs': [a0.name, b0.name],
            'output_count': len(result_a),
            'status': 'PASS' if scenario_a_pass else 'FAIL',
        },
        'scenario_b': {
            'description': 'Only first hard-linked name passed → expect 1 result',
            'inputs': [a0.name],
            'output_count': len(result_b),
            'status': 'PASS' if scenario_b_pass else 'FAIL',
        },
        'scenario_c': {
            'description': 'All names from pairs → output length equals unique inode count',
            'input_count': len(all_names),
            'output_count': len(result_c),
            'unique_inodes': len(seen_inodes_c),
            'status': 'PASS' if scenario_c_pass else 'FAIL',
        },
    }
    logger.info(
        'Hard-link filtering: pairs_found=%d A=%s B=%s C=%s [%s]',
        len(pairs),
        'PASS' if scenario_a_pass else 'FAIL',
        'PASS' if scenario_b_pass else 'FAIL',
        'PASS' if scenario_c_pass else 'FAIL',
        'PASS' if overall_pass else 'FAIL',
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — Timeout verification
# ─────────────────────────────────────────────────────────────────────────────

def _make_slow_binary(tmp_dir: Path) -> Optional[Path]:
    """
    Create a tiny shell script that sleeps long enough to trigger the 60s timeout
    when called via objdump (objdump will read the file and exit quickly, so we
    instead mock the scenario using a fake objdump wrapper).

    Since objdump on a real binary never takes 60s in practice, we test the
    timeout code path by temporarily pointing PATH at a wrapper script that
    sleeps.
    """
    wrapper = tmp_dir / 'objdump'
    wrapper.write_text(
        '#!/bin/sh\nsleep 120\n',
        encoding='utf-8',
    )
    wrapper.chmod(0o755)
    return wrapper


def check_timeout_handling() -> Dict[str, Any]:
    """
    Verify that:
    1. When objdump exceeds the 60s timeout, run_objdump raises DisassemblyError.
    2. disassemble_binary returns None (no Binary object created) when objdump
       times out; i.e. partially-parsed data does not leak into the corpus.

    Implementation: patch PATH to point at a sleep-based fake objdump, then
    call the extraction functions with a shortened timeout for fast test execution.
    We monkey-patch subprocess.run to simulate TimeoutExpired so the test
    completes instantly.
    """
    logger.info('=== Check 5: Timeout verification ===')

    import unittest.mock as mock

    results: Dict[str, Any] = {}
    overall_pass = True

    # ── Sub-check 5a: run_objdump raises DisassemblyError on TimeoutExpired ──
    timeout_expired = subprocess.TimeoutExpired(cmd=['objdump'], timeout=60)
    with mock.patch('extraction.disassemble.subprocess.run', side_effect=timeout_expired):
        try:
            run_objdump(Path('/usr/bin/grep'))
            raised = False
        except DisassemblyError:
            raised = True
        except Exception as exc:
            raised = False
            results['unexpected_exception_5a'] = str(exc)

    subcheck_5a_pass = raised
    if not subcheck_5a_pass:
        overall_pass = False
    results['subcheck_5a'] = {
        'description': 'run_objdump raises DisassemblyError on TimeoutExpired',
        'status': 'PASS' if subcheck_5a_pass else 'FAIL',
        'raised_disassembly_error': raised,
    }
    logger.info('Timeout 5a (run_objdump raises): %s', 'PASS' if subcheck_5a_pass else 'FAIL')

    # ── Sub-check 5b: disassemble_binary returns None on timeout ─────────────
    with mock.patch('extraction.disassemble.subprocess.run', side_effect=timeout_expired):
        binary_obj = disassemble_binary(Path('/usr/bin/grep'))

    subcheck_5b_pass = (binary_obj is None)
    if not subcheck_5b_pass:
        overall_pass = False
    results['subcheck_5b'] = {
        'description': 'disassemble_binary returns None when objdump times out',
        'status': 'PASS' if subcheck_5b_pass else 'FAIL',
        'binary_object_is_none': (binary_obj is None),
    }
    logger.info('Timeout 5b (disassemble_binary→None): %s', 'PASS' if subcheck_5b_pass else 'FAIL')

    # ── Sub-check 5c: disassemble_binary returns None on non-zero exit code ──
    # Simulates a corrupted binary that makes objdump exit with an error.
    failed_result = mock.MagicMock()
    failed_result.returncode = 1
    failed_result.stderr = 'objdump: not an ELF file'
    with mock.patch('extraction.disassemble.subprocess.run', return_value=failed_result):
        binary_obj_err = disassemble_binary(Path('/usr/bin/grep'))

    subcheck_5c_pass = (binary_obj_err is None)
    if not subcheck_5c_pass:
        overall_pass = False
    results['subcheck_5c'] = {
        'description': 'disassemble_binary returns None on objdump non-zero exit',
        'status': 'PASS' if subcheck_5c_pass else 'FAIL',
        'binary_object_is_none': (binary_obj_err is None),
    }
    logger.info('Timeout 5c (non-zero exit→None): %s', 'PASS' if subcheck_5c_pass else 'FAIL')

    return {'status': 'PASS' if overall_pass else 'FAIL', **results}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def resolve_sample_binaries() -> List[Path]:
    """Return Path objects for SAMPLE_BINARIES that actually exist on disk."""
    found: List[Path] = []
    for name in SAMPLE_BINARIES:
        p = USR_BIN / name
        if p.exists() and p.is_file():
            found.append(p)
        else:
            logger.warning('Sample binary not found, skipping: %s', p)
    logger.info('Using %d/%d sample binaries: %s', len(found), len(SAMPLE_BINARIES),
                [bp.name for bp in found])
    return found


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    binary_paths = resolve_sample_binaries()

    if not binary_paths:
        logger.error('No sample binaries available — cannot run checks 1-3.')

    report: Dict[str, Any] = {
        'sample_binaries': [str(bp) for bp in binary_paths],
    }

    # Run all checks; a missing binary list results in SKIP for checks 1-3.
    if binary_paths:
        report['check_1_mnemonic_counts'] = check_mnemonic_counts(binary_paths)
        report['check_2_instruction_boundaries'] = check_instruction_boundaries(binary_paths)
        report['check_3_function_boundaries'] = check_function_boundaries(binary_paths)
    else:
        skip = {'status': 'SKIP', 'reason': 'No sample binaries found'}
        report['check_1_mnemonic_counts'] = skip
        report['check_2_instruction_boundaries'] = skip
        report['check_3_function_boundaries'] = skip

    report['check_4_hardlink_filtering'] = check_hardlink_filtering()
    report['check_5_timeout_handling'] = check_timeout_handling()

    # Overall summary
    statuses = [
        report['check_1_mnemonic_counts']['status'],
        report['check_2_instruction_boundaries']['status'],
        report['check_3_function_boundaries']['status'],
        report['check_4_hardlink_filtering']['status'],
        report['check_5_timeout_handling']['status'],
    ]
    fails = [s for s in statuses if s == 'FAIL']
    skips = [s for s in statuses if s == 'SKIP']
    overall = 'FAIL' if fails else ('SKIP' if len(skips) == len(statuses) else 'PASS')
    report['overall'] = {
        'status': overall,
        'checks_passed': statuses.count('PASS'),
        'checks_failed': len(fails),
        'checks_skipped': len(skips),
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)

    logger.info('Results written to %s', OUTPUT_JSON)
    logger.info(
        'Overall: %s  (passed=%d  failed=%d  skipped=%d)',
        overall,
        report['overall']['checks_passed'],
        report['overall']['checks_failed'],
        report['overall']['checks_skipped'],
    )

    return 0 if overall != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())
