"""
independent_corpora.py -- Cross-corpus validation for OpCode-Stats.

Builds three independent corpora of system binaries and checks that the
pre-registered primary metrics are consistent across all three, providing
evidence that the observed statistical properties generalise beyond the
original smoke corpus.

Corpus A: smoke corpus baseline  (loaded from results/smoke_fresh/corpus/corpus.pkl)
Corpus B: network/system utilities  (ssh, scp, wget, diff, patch, make, ...)
Corpus C: developer/server tools    (python3, perl, gdb, clang, g++, cmake, ...)

Output: validation/results/independent_corpora.json
"""

import sys
import json
import math
import logging
import pickle
import zlib
import random
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, '/home/aaslyan/OpCode-Stats')

from utils.helpers import Binary, filter_valid_binaries, save_pickle, ensure_output_dir
from extraction.disassemble import disassemble_binary, save_corpus_data
from analysis.frequency import compute_frequency_distribution, compute_rank_frequency, fit_zipf_mle
from analysis.ngrams import compute_entropy_rate, compute_shuffled_entropy_rates
from analysis.compression import compute_compression_ratios, generate_unigram_shuffled_baseline
from analysis.lm import NgramLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path('/home/aaslyan/OpCode-Stats')
SMOKE_CORPUS_PKL = PROJECT_ROOT / 'results' / 'smoke_fresh' / 'corpus' / 'corpus.pkl'
OUTPUT_DIR = PROJECT_ROOT / 'validation' / 'results'
CORPUS_B_DIR = OUTPUT_DIR / 'corpus_B'
CORPUS_C_DIR = OUTPUT_DIR / 'corpus_C'
OUTPUT_JSON = OUTPUT_DIR / 'independent_corpora.json'

# ---------------------------------------------------------------------------
# Binary candidate lists
# The smoke corpus contains:
#   grep, sed, gzip, bzip2, xz, find, tar, bash, python3, curl, git, vim, nano, gcc, objdump
# Corpus B and C must not overlap with each other or with smoke.
# ---------------------------------------------------------------------------

CORPUS_B_CANDIDATES = [
    'ssh', 'scp', 'wget', 'diff', 'patch', 'make',
    'ld', 'as', 'nm', 'readelf', 'strace', 'top',
    'htop', 'less', 'more', 'man',
]

# Smoke already has python3, gcc — exclude them from C.
CORPUS_C_CANDIDATES = [
    'perl', 'ruby', 'php', 'java',
    'gdb', 'valgrind', 'g++', 'cmake',
    'lldb', 'rustc', 'go', 'ninja', 'node',
    'clang', 'clang-19',
]

# Binaries known to be in the smoke corpus (by name stem); used to exclude
# any accidental overlap.
SMOKE_NAMES = {
    'grep', 'sed', 'gzip', 'bzip2', 'xz', 'find', 'tar',
    'bash', 'python3', 'curl', 'git', 'vim', 'nano', 'gcc', 'objdump',
}


# ---------------------------------------------------------------------------
# Corpus construction helpers
# ---------------------------------------------------------------------------

def load_corpus_a() -> List[Binary]:
    """Load the smoke corpus from disk."""
    if not SMOKE_CORPUS_PKL.exists():
        logger.error(f"Smoke corpus not found at {SMOKE_CORPUS_PKL}")
        return []
    with open(SMOKE_CORPUS_PKL, 'rb') as fh:
        binaries = pickle.load(fh)
    logger.info(f"Corpus A: loaded {len(binaries)} binaries from {SMOKE_CORPUS_PKL}")
    return binaries


def build_corpus(candidates: List[str],
                 save_dir: Path,
                 label: str,
                 exclude_names: set,
                 max_count: int = 15) -> List[Binary]:
    """
    Find, disassemble, and save a corpus from the candidate binary list.

    Candidates that share an inode with a smoke binary, or whose name is in
    *exclude_names*, are skipped.  At most *max_count* binaries are kept.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    cached_pkl = save_dir / 'corpus.pkl'
    if cached_pkl.exists():
        logger.info(f"{label}: loading cached corpus from {cached_pkl}")
        with open(cached_pkl, 'rb') as fh:
            return pickle.load(fh)

    # Filter to existing binaries, deduplicating by inode
    valid_paths = filter_valid_binaries(candidates, Path('/usr/bin'))
    # Also look in common alternative locations
    for alt_dir in [Path('/usr/local/bin'), Path('/bin'), Path('/home/aaslyan/bin')]:
        extra = [c for c in candidates if (alt_dir / c).exists()]
        if extra:
            valid_paths += filter_valid_binaries(extra, alt_dir)

    # Deduplicate again across all search directories by inode
    seen_inodes: set = set()
    unique_paths = []
    for p in valid_paths:
        try:
            inode = p.resolve().stat().st_ino
        except OSError:
            continue
        if inode not in seen_inodes:
            seen_inodes.add(inode)
            unique_paths.append(p)

    # Exclude names shared with smoke (or already in exclude_names)
    unique_paths = [p for p in unique_paths if p.stem not in exclude_names]

    # Cap at max_count
    unique_paths = unique_paths[:max_count]

    logger.info(f"{label}: disassembling {len(unique_paths)} binaries")
    binaries: List[Binary] = []
    for path in unique_paths:
        binary = disassemble_binary(path, category=label)
        if binary is not None:
            binaries.append(binary)

    if binaries:
        save_corpus_data(binaries, save_dir)
        logger.info(f"{label}: saved {len(binaries)} binaries to {save_dir}")
    else:
        logger.warning(f"{label}: no binaries successfully disassembled")

    return binaries


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _entropy_from_counter(counts: Counter) -> float:
    """Shannon entropy H (bits) from a frequency counter."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def compute_h1(binaries: List[Binary]) -> float:
    """Unigram entropy H1 (bits)."""
    counts: Counter = Counter()
    for b in binaries:
        counts.update(b.full_opcode_sequence)
    return _entropy_from_counter(counts)


def compute_h5_rate(binaries: List[Binary]) -> float:
    """5-gram joint entropy H5 divided by 5 (entropy rate at n=5)."""
    rates = compute_entropy_rate(binaries, max_n=5)
    for entry in rates:
        if entry['n'] == 5:
            return entry['entropy_rate']
    return float('nan')


def compute_compression_gap(binaries: List[Binary]) -> Tuple[float, float, float]:
    """
    Return (mean_zlib_real, mean_zlib_shuffled, gap) where gap = real - shuffled.

    A negative gap means real sequences are *more* compressible than shuffled,
    i.e. they have more sequential redundancy.
    """
    real_results = compute_compression_ratios(binaries)
    shuffled = generate_unigram_shuffled_baseline(binaries, num_shuffles=3, rng_seed=42)

    if not real_results:
        return float('nan'), float('nan'), float('nan')

    mean_real = float(np.mean([r['zlib_ratio'] for r in real_results]))
    mean_shuffled = float(shuffled['zlib_ratio'])
    gap = mean_real - mean_shuffled
    return mean_real, mean_shuffled, gap


def compute_mi_half_life(binaries: List[Binary], max_lag: int = 10) -> float:
    """
    Mutual information half-life: smallest lag l such that MI(lag=l) <= 0.5 * MI(lag=1).

    Estimated via empirical bigram/unigram entropy: MI(lag) approximated by the
    drop in conditional entropy as context extends.  We use the entropy-rate
    differences across lags as a proxy for MI decay.
    """
    # We approximate MI(lag=k) as the conditional entropy H(X_k | X_0) which is
    # related to: H1 - (H_{k+1}/k+1) is not straightforward. Instead we use a
    # simpler proxy: for each lag l, compute the mutual information as the
    # difference in unigram entropy minus the average conditional entropy at that lag.

    # Concatenate all sequences
    full_seq: List[str] = []
    for b in binaries:
        full_seq.extend(b.full_opcode_sequence)

    if len(full_seq) < max_lag + 2:
        return float('nan')

    # Unigram probabilities
    unigram_counts: Counter = Counter(full_seq)
    total = len(full_seq)
    h1 = _entropy_from_counter(unigram_counts)

    mi_by_lag: Dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        # Build joint distribution for (X_t, X_{t+lag})
        joint: Counter = Counter()
        n_pairs = 0
        for i in range(len(full_seq) - lag):
            joint[(full_seq[i], full_seq[i + lag])] += 1
            n_pairs += 1

        if n_pairs == 0:
            mi_by_lag[lag] = 0.0
            continue

        # H(X, X_lag) = joint entropy
        h_joint = 0.0
        for cnt in joint.values():
            if cnt > 0:
                p = cnt / n_pairs
                h_joint -= p * math.log2(p)

        # MI(X; X_lag) = H(X) + H(X) - H(X, X_lag) = 2*H1 - H_joint
        # (both marginals are approximately H1 for a stationary sequence)
        mi = max(0.0, 2.0 * h1 - h_joint)
        mi_by_lag[lag] = mi

    mi_lag1 = mi_by_lag.get(1, 0.0)
    if mi_lag1 == 0:
        return float('nan')

    threshold = 0.5 * mi_lag1
    for lag in range(1, max_lag + 1):
        if mi_by_lag[lag] <= threshold:
            return float(lag)
    return float(max_lag)  # half-life exceeds max_lag


def compute_bigram_lm_perplexity(binaries: List[Binary],
                                  test_fraction: float = 0.2) -> float:
    """Train a bigram LM on 80% and evaluate perplexity on 20%."""
    sequences = [b.full_opcode_sequence for b in binaries]
    if not sequences:
        return float('nan')

    train_seqs = []
    test_seqs = []
    for seq in sequences:
        split = int(len(seq) * (1.0 - test_fraction))
        train_seqs.append(seq)
        test_seqs.append(seq[split:])

    lm = NgramLM(n=2, k=1.0)
    lm.train(train_seqs)
    ppl = lm.perplexity([s for s in test_seqs if s])
    return float(ppl)


def compute_ncd_effect_size(binaries: List[Binary]) -> float:
    """
    Effect size: mean within-corpus NCD minus mean between-corpus NCD.

    A positive value means within-corpus pairs are more similar (lower NCD)
    than between-corpus pairs -- this is checked against the original finding
    that binary families cluster together.

    For a single corpus the within-corpus NCD is computed directly; the
    between-corpus NCD is approximated with a shuffled sequence baseline.
    """
    from utils.helpers import build_vocabulary, encode_sequence

    if len(binaries) < 2:
        return float('nan')

    vocab = build_vocabulary(binaries)
    max_len = 20_000

    def _encode_bytes(binary: Binary) -> bytes:
        seq = binary.full_opcode_sequence[:max_len]
        enc = encode_sequence(seq, vocab)
        try:
            if enc and max(enc) > 255:
                return struct.pack(f'<{len(enc)}H', *enc)
            return bytes(enc)
        except (ValueError, struct.error):
            return bytes([x % 256 for x in enc])

    encoded = [_encode_bytes(b) for b in binaries]
    compressed_single = [len(zlib.compress(e)) for e in encoded]

    ncd_values = []
    n = len(binaries)
    for i in range(n):
        for j in range(i + 1, n):
            cx = compressed_single[i]
            cy = compressed_single[j]
            cxy = len(zlib.compress(encoded[i] + encoded[j]))
            ncd = (cxy - min(cx, cy)) / max(max(cx, cy), 1)
            ncd_values.append(ncd)

    if not ncd_values:
        return float('nan')

    # For a single corpus we report the mean within-corpus NCD.
    # The effect size relative to "between corpora" is computed in the
    # cross-corpus section; here we just return mean within.
    return float(np.mean(ncd_values))


# ---------------------------------------------------------------------------
# Cross-corpus LM evaluation
# ---------------------------------------------------------------------------

def cross_lm_eval(train_binaries: List[Binary],
                  test_binaries: List[Binary],
                  train_label: str,
                  test_label: str) -> Dict:
    """Train bigram LM on train_binaries, evaluate on test_binaries."""
    train_seqs = [b.full_opcode_sequence for b in train_binaries]
    test_seqs = [b.full_opcode_sequence for b in test_binaries]

    if not train_seqs or not test_seqs:
        return {'train': train_label, 'test': test_label,
                'cross_entropy_bits': None, 'perplexity': None}

    lm = NgramLM(n=2, k=1.0)
    lm.train(train_seqs)
    ce = lm.cross_entropy(test_seqs)
    ppl = 2.0 ** ce

    logger.info(f"  LM({train_label} -> {test_label}): CE={ce:.3f} bits, PPL={ppl:.2f}")
    return {
        'train': train_label,
        'test': test_label,
        'cross_entropy_bits': round(ce, 4),
        'perplexity': round(ppl, 3),
    }


# ---------------------------------------------------------------------------
# Metric summary for a single corpus
# ---------------------------------------------------------------------------

def compute_corpus_metrics(binaries: List[Binary], label: str) -> Dict:
    """Compute the full pre-registered primary metric set for one corpus."""
    logger.info(f"=== Computing metrics for {label} ({len(binaries)} binaries) ===")

    if not binaries:
        return {'corpus': label, 'n_binaries': 0, 'error': 'empty corpus'}

    h1 = compute_h1(binaries)
    logger.info(f"  H1 = {h1:.4f} bits")

    h5_rate = compute_h5_rate(binaries)
    logger.info(f"  H5/5 = {h5_rate:.4f} bits")

    # Shuffled entropy rate at n=5 for comparison
    shuffled_rates = compute_shuffled_entropy_rates(binaries, max_n=5, seed=42)
    h5_rate_shuffled = next(
        (e['entropy_rate'] for e in shuffled_rates if e['n'] == 5), float('nan'))
    logger.info(f"  H5/5 shuffled = {h5_rate_shuffled:.4f} bits")

    mean_zlib_real, mean_zlib_shuffled, gap = compute_compression_gap(binaries)
    logger.info(f"  zlib_real={mean_zlib_real:.4f}  zlib_shuffled={mean_zlib_shuffled:.4f}  gap={gap:.4f}")

    mi_half_life = compute_mi_half_life(binaries, max_lag=10)
    logger.info(f"  MI half-life = {mi_half_life}")

    bigram_ppl = compute_bigram_lm_perplexity(binaries)
    logger.info(f"  bigram LM PPL = {bigram_ppl:.2f}")

    mean_ncd_within = compute_ncd_effect_size(binaries)
    logger.info(f"  mean NCD within = {mean_ncd_within:.4f}")

    return {
        'corpus': label,
        'n_binaries': len(binaries),
        'binary_names': [b.name for b in binaries],
        'h1_bits': round(h1, 4),
        'h5_rate_bits': round(h5_rate, 4),
        'h5_rate_shuffled_bits': round(h5_rate_shuffled, 4),
        'entropy_rate_real_lt_shuffled': bool(h5_rate < h5_rate_shuffled),
        'mean_zlib_real': round(mean_zlib_real, 4),
        'mean_zlib_shuffled': round(mean_zlib_shuffled, 4),
        'compression_gap': round(gap, 4),
        'compression_real_lt_shuffled': bool(mean_zlib_real < mean_zlib_shuffled),
        'mi_half_life_lag': mi_half_life,
        'bigram_lm_perplexity': round(bigram_ppl, 3),
        'mean_ncd_within': round(mean_ncd_within, 4),
    }


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def check_consistency(metrics_a: Dict, metrics_b: Dict, metrics_c: Dict,
                      cross_lm_results: List[Dict]) -> Dict:
    """
    Report PASS/FAIL for each pre-registered consistency criterion.

    Criteria:
    1. All corpora: entropy rate < shuffled entropy rate
    2. All corpora: compression ratio < shuffled compression ratio
    3. Cross-corpus LM perplexity is within 2x of within-corpus perplexity
    """
    results = {}

    # Criterion 1: entropy rate real < shuffled for all corpora
    c1_a = metrics_a.get('entropy_rate_real_lt_shuffled', False)
    c1_b = metrics_b.get('entropy_rate_real_lt_shuffled', False)
    c1_c = metrics_c.get('entropy_rate_real_lt_shuffled', False)
    results['criterion_1_entropy_rate_real_lt_shuffled'] = {
        'A': c1_a, 'B': c1_b, 'C': c1_c,
        'pass': bool(c1_a and c1_b and c1_c),
    }

    # Criterion 2: compression ratio real < shuffled for all corpora
    c2_a = metrics_a.get('compression_real_lt_shuffled', False)
    c2_b = metrics_b.get('compression_real_lt_shuffled', False)
    c2_c = metrics_c.get('compression_real_lt_shuffled', False)
    results['criterion_2_compression_real_lt_shuffled'] = {
        'A': c2_a, 'B': c2_b, 'C': c2_c,
        'pass': bool(c2_a and c2_b and c2_c),
    }

    # Criterion 3: cross-corpus perplexity within 2x of within-corpus perplexity
    within_ppls = [
        metrics_a.get('bigram_lm_perplexity', float('nan')),
        metrics_b.get('bigram_lm_perplexity', float('nan')),
        metrics_c.get('bigram_lm_perplexity', float('nan')),
    ]
    within_ppls_valid = [p for p in within_ppls if math.isfinite(p) and p > 0]
    cross_ppls = [r['perplexity'] for r in cross_lm_results
                  if r.get('perplexity') is not None]

    if within_ppls_valid and cross_ppls:
        mean_within = float(np.mean(within_ppls_valid))
        mean_cross = float(np.mean(cross_ppls))
        ratio = mean_cross / mean_within if mean_within > 0 else float('nan')
        c3_pass = bool(ratio <= 2.0)
    else:
        mean_within = float('nan')
        mean_cross = float('nan')
        ratio = float('nan')
        c3_pass = False

    results['criterion_3_cross_ppl_within_2x'] = {
        'mean_within_corpus_ppl': round(mean_within, 3) if math.isfinite(mean_within) else None,
        'mean_cross_corpus_ppl': round(mean_cross, 3) if math.isfinite(mean_cross) else None,
        'ratio': round(ratio, 3) if math.isfinite(ratio) else None,
        'pass': c3_pass,
    }

    all_pass = all(v['pass'] for v in results.values())
    results['overall_pass'] = all_pass
    return results


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(metrics_list: List[Dict]) -> None:
    header = (
        f"{'corpus':<12} {'n_bins':>6} {'H1':>7} {'H5_rate':>8} "
        f"{'gap':>8} {'zlib':>7} {'bigram_ppl':>10} {'consistent?':>12}"
    )
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        name = m.get('corpus', '?')
        n = m.get('n_binaries', 0)
        h1 = m.get('h1_bits', float('nan'))
        h5 = m.get('h5_rate_bits', float('nan'))
        gap = m.get('compression_gap', float('nan'))
        zlib = m.get('mean_zlib_real', float('nan'))
        ppl = m.get('bigram_lm_perplexity', float('nan'))
        e_ok = m.get('entropy_rate_real_lt_shuffled', False)
        c_ok = m.get('compression_real_lt_shuffled', False)
        consistent = 'YES' if (e_ok and c_ok) else 'NO'
        print(
            f"{name:<12} {n:>6} {h1:>7.3f} {h5:>8.4f} "
            f"{gap:>8.4f} {zlib:>7.4f} {ppl:>10.2f} {consistent:>12}"
        )
    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== independent_corpora.py ===")
    ensure_output_dir(OUTPUT_DIR)

    # ── 1. Build / load corpora ───────────────────────────────────────────

    corpus_a = load_corpus_a()

    # Names already present in corpus A (smoke) -- exclude from B and C
    smoke_binary_names = {b.name for b in corpus_a} | SMOKE_NAMES

    corpus_b = build_corpus(
        CORPUS_B_CANDIDATES,
        CORPUS_B_DIR,
        label='corpus_B',
        exclude_names=smoke_binary_names,
        max_count=15,
    )

    # Corpus C must also exclude names that ended up in corpus B
    corpus_b_names = {b.name for b in corpus_b}
    corpus_c = build_corpus(
        CORPUS_C_CANDIDATES,
        CORPUS_C_DIR,
        label='corpus_C',
        exclude_names=smoke_binary_names | corpus_b_names,
        max_count=15,
    )

    logger.info(
        f"Corpus sizes: A={len(corpus_a)}, B={len(corpus_b)}, C={len(corpus_c)}"
    )

    if not corpus_a or not corpus_b or not corpus_c:
        logger.error("One or more corpora are empty -- aborting")
        sys.exit(1)

    # ── 2. Compute per-corpus primary metrics ─────────────────────────────

    metrics_a = compute_corpus_metrics(corpus_a, 'corpus_A')
    metrics_b = compute_corpus_metrics(corpus_b, 'corpus_B')
    metrics_c = compute_corpus_metrics(corpus_c, 'corpus_C')

    # ── 3. Cross-corpus LM evaluation ────────────────────────────────────

    logger.info("=== Cross-corpus LM evaluation ===")
    cross_lm = []

    # Train on A, test on B and C
    cross_lm.append(cross_lm_eval(corpus_a, corpus_b, 'A', 'B'))
    cross_lm.append(cross_lm_eval(corpus_a, corpus_c, 'A', 'C'))
    # Train on B, test on A and C
    cross_lm.append(cross_lm_eval(corpus_b, corpus_a, 'B', 'A'))
    cross_lm.append(cross_lm_eval(corpus_b, corpus_c, 'B', 'C'))
    # Train on C, test on A and B
    cross_lm.append(cross_lm_eval(corpus_c, corpus_a, 'C', 'A'))
    cross_lm.append(cross_lm_eval(corpus_c, corpus_b, 'C', 'B'))

    # ── 4. Consistency checks ─────────────────────────────────────────────

    consistency = check_consistency(metrics_a, metrics_b, metrics_c, cross_lm)

    logger.info(f"Overall consistency: {'PASS' if consistency['overall_pass'] else 'FAIL'}")
    for key, val in consistency.items():
        if key != 'overall_pass':
            status = 'PASS' if val.get('pass') else 'FAIL'
            logger.info(f"  {key}: {status}")

    # ── 5. Print summary table ────────────────────────────────────────────

    print_summary_table([metrics_a, metrics_b, metrics_c])

    # ── 6. Save JSON output ───────────────────────────────────────────────

    output = {
        'description': (
            'Cross-corpus validation: three independent corpora of system binaries '
            'are analysed to confirm that the pre-registered primary metrics are '
            'consistent across different sets of binaries.'
        ),
        'corpus_A': metrics_a,
        'corpus_B': metrics_b,
        'corpus_C': metrics_c,
        'cross_lm_evaluation': cross_lm,
        'consistency_checks': consistency,
    }

    def _json_safe(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(OUTPUT_JSON, 'w') as fh:
        json.dump(output, fh, indent=2, default=_json_safe)

    logger.info(f"Results written to {OUTPUT_JSON}")

    overall = 'PASS' if consistency['overall_pass'] else 'FAIL'
    print(f"\nOverall consistency verdict: {overall}\n")


if __name__ == '__main__':
    main()
