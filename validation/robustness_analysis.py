"""
Statistical robustness and uncertainty analysis for OpCode-Stats.

Tests the stability of key corpus-level metrics via:
  1. Bootstrap CIs over binaries (resample binaries with replacement)
  2. Multiple-seed shuffled baselines (seeds 1-10)
  3. Corpus subsampling sensitivity (fractions 0.25 / 0.5 / 0.75 / 1.0)
  4. Sequence length truncation sensitivity
  5. Leave-one-binary-out stability

Output: validation/results/robustness_analysis.json
        Summary table printed to stdout.
"""

import sys
import json
import zlib
import lzma
import logging
import warnings
import random as _random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Must set Agg backend before importing matplotlib/pyplot elsewhere
import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, '/home/aaslyan/OpCode-Stats')

# Suppress noisy scipy/numpy warnings during bootstrap loops
warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BOOT = 200            # bootstrap iterations
N_SUBSAMPLE_DRAWS = 10  # random draws per corpus fraction
SUBSAMPLE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
TRUNCATION_LENGTHS = [1000, 5000, 10_000, 50_000]  # plus full length
SHUFFLED_SEEDS = list(range(1, 11))                 # seeds 1..10
ENTROPY_MAX_N = 5

CORPUS_CANDIDATES = [
    Path('/home/aaslyan/OpCode-Stats/results/smoke_fresh/corpus/corpus.pkl'),
    Path('/home/aaslyan/OpCode-Stats/results/smoke_rerun/corpus/corpus.pkl'),
    Path('/home/aaslyan/OpCode-Stats/results/smoke/corpus/corpus.pkl'),
    Path('/home/aaslyan/OpCode-Stats/results/system/corpus/corpus.pkl'),
    Path('/home/aaslyan/OpCode-Stats/results/coreutils_system/corpus/corpus.pkl'),
]

OUTPUT_PATH = Path(
    '/home/aaslyan/OpCode-Stats/validation/results/robustness_analysis.json'
)

# ---------------------------------------------------------------------------
# Imports from project modules (after sys.path is set)
# ---------------------------------------------------------------------------
from utils.helpers import load_pickle, build_vocabulary, encode_sequence, Binary

# ---------------------------------------------------------------------------
# Pre-computed cache structure
#
# Building these once at startup avoids re-encoding in every inner loop.
# ---------------------------------------------------------------------------

class CorpusCache:
    """
    Holds pre-computed per-binary arrays and compression ratios so that the
    inner loops (bootstrap, subsampling, leave-one-out) can operate on already-
    prepared data without repeating encoding or compression work.
    """
    def __init__(self, binaries: List[Binary], vocab: Dict[str, int]) -> None:
        self.binaries = binaries
        self.vocab = vocab
        self.n = len(binaries)
        self.V = len(vocab) + 1  # +1 for unknown token id

        print('  Pre-encoding sequences ...', flush=True)
        # Per-binary integer-encoded sequences as numpy uint16 arrays
        self.enc_seqs: List[np.ndarray] = []
        for b in binaries:
            seq = b.full_opcode_sequence
            enc = np.array(encode_sequence(seq, vocab), dtype=np.uint16)
            self.enc_seqs.append(enc)

        print('  Pre-computing compression ratios ...', flush=True)
        # Per-binary compression ratios (zlib, lzma)
        self.zlib_ratios: List[float] = []
        self.lzma_ratios: List[float] = []
        for enc in self.enc_seqs:
            z, l = _compress_encoded(enc)
            self.zlib_ratios.append(z)
            self.lzma_ratios.append(l)

        print('  Pre-computing truncated compression ratios ...', flush=True)
        # Per-binary per-length compression ratios for truncation analysis
        self.trunc_zlib: Dict[str, List[float]] = {}
        self.trunc_lzma: Dict[str, List[float]] = {}
        lengths: List[Optional[int]] = list(TRUNCATION_LENGTHS) + [None]
        for max_len in lengths:
            label = str(max_len) if max_len is not None else 'full'
            zs, ls = [], []
            for enc in self.enc_seqs:
                trunc = enc if max_len is None else enc[:max_len]
                z, l = _compress_encoded(trunc)
                zs.append(z)
                ls.append(l)
            self.trunc_zlib[label] = zs
            self.trunc_lzma[label] = ls

        print(f'  Cache ready: {self.n} binaries, vocab={self.V-1}', flush=True)


def _compress_encoded(enc: np.ndarray) -> Tuple[float, float]:
    """Return (zlib_ratio, lzma_ratio) for a uint16 numpy array."""
    if len(enc) == 0:
        return float('nan'), float('nan')
    raw = enc.tobytes()
    n = len(raw)
    try:
        z = len(zlib.compress(raw)) / n
    except Exception:
        z = float('nan')
    try:
        l = len(lzma.compress(raw)) / n
    except Exception:
        l = float('nan')
    return z, l


# ---------------------------------------------------------------------------
# Fast in-loop metric helpers (numpy-accelerated)
# ---------------------------------------------------------------------------

def _fast_entropy_rate_at_n(enc_seqs: List[np.ndarray], n: int,
                             V: int) -> float:
    """
    Compute H_n / n using numpy sliding-window views and a polynomial hash.

    For each sequence of length L, we view the (L-n+1) overlapping windows of
    width n, convert each window to a single int64 hash via a base-V polynomial,
    then count with Counter.  This avoids Python-level tuple creation and is
    ~13x faster than the equivalent tuple-based Counter approach.

    Returns NaN if there is no data.
    """
    big_V = np.int64(V)
    all_hashes: List[np.ndarray] = []

    for enc in enc_seqs:
        if len(enc) < n:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(enc.astype(np.int64), n)
        # Polynomial hash: h = w0*V^(n-1) + w1*V^(n-2) + ... + w_{n-1}
        h = np.zeros(len(windows), dtype=np.int64)
        for col in range(n):
            h = h * big_V + windows[:, col]
        all_hashes.append(h)

    if not all_hashes:
        return float('nan')

    counts = Counter(np.concatenate(all_hashes).tolist())
    total = sum(counts.values())
    if total == 0:
        return float('nan')

    c_arr = np.array(list(counts.values()), dtype=np.float64)
    p = c_arr / total
    entropy = float(-np.dot(p, np.log2(p)))
    return entropy / n


def _fast_zipf_alpha_mle(enc_seqs: List[np.ndarray]) -> float:
    """
    Discrete Zipf MLE on the unigram rank-frequency distribution.
    Returns NaN on failure.
    """
    counts = Counter()
    for enc in enc_seqs:
        counts.update(enc.tolist())
    if not counts:
        return float('nan')

    freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
    V = len(freqs)
    log_ranks = np.log(np.arange(1, V + 1, dtype=np.float64))
    N = freqs.sum()
    sum_wlogr = float(np.dot(freqs, log_ranks))

    def neg_ll(alpha: float) -> float:
        log_terms = -alpha * log_ranks
        shift = log_terms.max()
        log_Z = np.log(np.sum(np.exp(log_terms - shift))) + shift
        return alpha * sum_wlogr + N * log_Z

    try:
        res = minimize_scalar(neg_ll, bounds=(0.01, 15.0), method='bounded')
        return float(res.x)
    except Exception:
        return float('nan')


def _sample_compression_mean(cache: CorpusCache,
                              indices: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean zlib / lzma ratio for a bootstrap sample (list of binary indices).
    Uses pre-cached per-binary ratios — O(k) instead of O(total_tokens).
    """
    zlib_vals = [cache.zlib_ratios[i] for i in indices]
    lzma_vals = [cache.lzma_ratios[i] for i in indices]
    z = float(np.nanmean(zlib_vals)) if zlib_vals else float('nan')
    l = float(np.nanmean(lzma_vals)) if lzma_vals else float('nan')
    return z, l


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    arr = [v for v in values if not np.isnan(v)]
    return float(np.mean(arr)) if arr else float('nan')


def _safe_std(values: List[float]) -> float:
    arr = [v for v in values if not np.isnan(v)]
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else float('nan')


def _percentile_ci(values: List[float], lo: float = 2.5, hi: float = 97.5
                   ) -> Tuple[float, float]:
    arr = [v for v in values if not np.isnan(v)]
    if len(arr) < 2:
        v = arr[0] if arr else float('nan')
        return v, v
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


# ---------------------------------------------------------------------------
# 1. Bootstrap CIs over binaries
# ---------------------------------------------------------------------------

def run_bootstrap_ci(cache: CorpusCache, rng: np.random.Generator) -> Dict:
    """
    Resample binary list (with replacement) N_BOOT times.
    Metrics: Zipf alpha_mle, 5-gram entropy rate, mean zlib, mean lzma.
    Returns point estimates + 95% CI + SE for each metric.
    """
    print(f'\n[1/5] Bootstrap CIs ({N_BOOT} iterations, resampling binaries) ...',
          flush=True)
    n = cache.n
    V = cache.V

    boot_alpha: List[float] = []
    boot_h5: List[float] = []
    boot_zlib: List[float] = []
    boot_lzma: List[float] = []

    for i in range(N_BOOT):
        if (i + 1) % 50 == 0:
            print(f'      iteration {i + 1}/{N_BOOT}', flush=True)
        idx = rng.integers(0, n, size=n)
        seqs = [cache.enc_seqs[j] for j in idx]

        boot_alpha.append(_fast_zipf_alpha_mle(seqs))
        boot_h5.append(_fast_entropy_rate_at_n(seqs, n=5, V=V))
        z, l = _sample_compression_mean(cache, idx)
        boot_zlib.append(z)
        boot_lzma.append(l)

    # Point estimates on the full corpus
    pe_alpha = _fast_zipf_alpha_mle(cache.enc_seqs)
    pe_h5 = _fast_entropy_rate_at_n(cache.enc_seqs, n=5, V=V)
    pe_zlib = float(np.nanmean(cache.zlib_ratios))
    pe_lzma = float(np.nanmean(cache.lzma_ratios))

    def _stat(point: float, samples: List[float]) -> Dict:
        ci_lo, ci_hi = _percentile_ci(samples)
        return {
            'point_estimate': point,
            'ci_low_95': ci_lo,
            'ci_high_95': ci_hi,
            'standard_error': _safe_std(samples),
            'n_valid_boot': sum(1 for v in samples if not np.isnan(v)),
        }

    result = {
        'n_boot': N_BOOT,
        'resample_unit': 'binary',
        'metrics': {
            'zipf_alpha_mle': _stat(pe_alpha, boot_alpha),
            'entropy_rate_5gram': _stat(pe_h5, boot_h5),
            'mean_zlib_ratio': _stat(pe_zlib, boot_zlib),
            'mean_lzma_ratio': _stat(pe_lzma, boot_lzma),
        },
    }
    print('      Done.', flush=True)
    return result


# ---------------------------------------------------------------------------
# 2. Multiple-seed shuffled baselines
# ---------------------------------------------------------------------------

def run_multi_seed_shuffled(cache: CorpusCache) -> Dict:
    """
    Shuffle each binary's encoded sequence with seeds 1..10.
    Report mean ± std of shuffled entropy rate at each n, and compare with
    real rates to confirm the gap is stable across seeds.
    """
    print(f'\n[2/5] Multi-seed shuffled baselines (seeds {SHUFFLED_SEEDS}) ...',
          flush=True)
    V = cache.V

    # Real entropy rates (computed once on unshuffled data)
    real_by_n: Dict[int, float] = {}
    for n in range(1, ENTROPY_MAX_N + 1):
        real_by_n[n] = _fast_entropy_rate_at_n(cache.enc_seqs, n=n, V=V)

    # Collect shuffled entropy rates across seeds
    shuf_by_n: Dict[int, List[float]] = {n: [] for n in range(1, ENTROPY_MAX_N + 1)}

    for seed in SHUFFLED_SEEDS:
        print(f'      seed {seed}', flush=True)
        local_rng = np.random.default_rng(seed)
        shuffled = [local_rng.permutation(enc) for enc in cache.enc_seqs]
        for n in range(1, ENTROPY_MAX_N + 1):
            shuf_by_n[n].append(_fast_entropy_rate_at_n(shuffled, n=n, V=V))

    per_n = {}
    for n in range(1, ENTROPY_MAX_N + 1):
        vals = shuf_by_n[n]
        mean_shuf = _safe_mean(vals)
        std_shuf = _safe_std(vals)
        real_val = real_by_n.get(n, float('nan'))
        nan_real = np.isnan(real_val)
        nan_shuf = np.isnan(mean_shuf)
        gap = real_val - mean_shuf if not (nan_real or nan_shuf) else float('nan')
        nan_std = np.isnan(std_shuf) if isinstance(std_shuf, float) else False
        gap_cv = (std_shuf / abs(mean_shuf)
                  if mean_shuf and not (nan_shuf or nan_std)
                  else float('nan'))
        per_n[str(n)] = {
            'real_entropy_rate': real_val,
            'shuffled_mean': mean_shuf,
            'shuffled_std': std_shuf,
            'gap_real_minus_shuffled': gap,
            'shuffled_cv': gap_cv,
            'gap_stable': bool(not np.isnan(gap_cv) and gap_cv < 0.05),
            'seed_values': vals,
        }

    print('      Done.', flush=True)
    return {
        'seeds': SHUFFLED_SEEDS,
        'per_n': per_n,
        'summary': (
            'gap_stable=True when shuffled CV < 0.05 (5%): seed choice '
            'changes the shuffled baseline by less than 5% of its magnitude.'
        ),
    }


# ---------------------------------------------------------------------------
# 3. Corpus subsampling sensitivity
# ---------------------------------------------------------------------------

def run_subsampling_sensitivity(cache: CorpusCache,
                                rng: np.random.Generator) -> Dict:
    """
    For each fraction in SUBSAMPLE_FRACTIONS, draw N_SUBSAMPLE_DRAWS subsamples
    (without replacement) and compute Zipf alpha, entropy rate, and compression.
    """
    print(f'\n[3/5] Corpus subsampling sensitivity '
          f'(fractions={SUBSAMPLE_FRACTIONS}, draws={N_SUBSAMPLE_DRAWS}) ...',
          flush=True)
    n_total = cache.n
    V = cache.V

    results_by_fraction: Dict[str, Any] = {}

    for frac in SUBSAMPLE_FRACTIONS:
        k = max(2, int(round(frac * n_total)))
        print(f'      fraction={frac:.2f}, k={k} binaries', flush=True)
        alpha_vals, h5_vals, zlib_vals, lzma_vals = [], [], [], []

        n_draws = 1 if frac == 1.0 else N_SUBSAMPLE_DRAWS
        for _ in range(n_draws):
            if frac == 1.0:
                idx = np.arange(n_total)
            else:
                idx = rng.choice(n_total, size=k, replace=False)
            seqs = [cache.enc_seqs[i] for i in idx]

            alpha_vals.append(_fast_zipf_alpha_mle(seqs))
            h5_vals.append(_fast_entropy_rate_at_n(seqs, n=5, V=V))
            z, l = _sample_compression_mean(cache, idx)
            zlib_vals.append(z)
            lzma_vals.append(l)

        results_by_fraction[str(frac)] = {
            'k_binaries': k,
            'n_draws': n_draws,
            'zipf_alpha_mle': {
                'mean': _safe_mean(alpha_vals),
                'std': _safe_std(alpha_vals),
                'values': alpha_vals,
            },
            'entropy_rate_5gram': {
                'mean': _safe_mean(h5_vals),
                'std': _safe_std(h5_vals),
                'values': h5_vals,
            },
            'mean_zlib_ratio': {
                'mean': _safe_mean(zlib_vals),
                'std': _safe_std(zlib_vals),
                'values': zlib_vals,
            },
            'mean_lzma_ratio': {
                'mean': _safe_mean(lzma_vals),
                'std': _safe_std(lzma_vals),
                'values': lzma_vals,
            },
        }

    print('      Done.', flush=True)
    return {
        'fractions': SUBSAMPLE_FRACTIONS,
        'n_draws_per_fraction': N_SUBSAMPLE_DRAWS,
        'by_fraction': results_by_fraction,
        'interpretation': (
            'Metric std across draws at each fraction shows sensitivity to '
            'which binaries are included. Large std at low fractions is expected; '
            'convergence as fraction -> 1.0 confirms conclusions are not driven '
            'by individual binaries.'
        ),
    }


# ---------------------------------------------------------------------------
# 4. Sequence length truncation sensitivity
# ---------------------------------------------------------------------------

def run_truncation_sensitivity(cache: CorpusCache) -> Dict:
    """
    Truncate each binary's opcode sequence to fixed lengths and recompute
    per-binary zlib/lzma ratios and 5-gram entropy rates.
    Reports how metrics change with sequence length.
    """
    print(f'\n[4/5] Sequence length truncation sensitivity '
          f'(lengths={TRUNCATION_LENGTHS} + full) ...', flush=True)

    lengths: List[Optional[int]] = list(TRUNCATION_LENGTHS) + [None]
    V = cache.V
    results_by_length: Dict[str, Any] = {}

    for max_len in lengths:
        label = str(max_len) if max_len is not None else 'full'
        print(f'      length={label}', flush=True)

        # Truncated encoded sequences
        trunc_seqs = (
            [enc[:max_len] for enc in cache.enc_seqs]
            if max_len is not None
            else cache.enc_seqs
        )
        actual_lens = [len(s) for s in trunc_seqs]

        # Compression: use pre-cached values for 'full'; recompute for others
        zlib_per_binary: Dict[str, float] = {}
        lzma_per_binary: Dict[str, float] = {}
        zlib_vals: List[float] = []
        lzma_vals: List[float] = []

        cached_z = cache.trunc_zlib[label]
        cached_l = cache.trunc_lzma[label]
        for binary, z, l in zip(cache.binaries, cached_z, cached_l):
            zlib_per_binary[binary.name] = z
            lzma_per_binary[binary.name] = l
            zlib_vals.append(z)
            lzma_vals.append(l)

        h5_rate = _fast_entropy_rate_at_n(trunc_seqs, n=5, V=V)

        results_by_length[label] = {
            'max_tokens_per_binary': max_len,
            'mean_actual_length': float(np.mean(actual_lens)) if actual_lens else float('nan'),
            'entropy_rate_5gram': h5_rate,
            'zlib_ratio': {
                'mean': _safe_mean(zlib_vals),
                'std': _safe_std(zlib_vals),
                'per_binary': zlib_per_binary,
            },
            'lzma_ratio': {
                'mean': _safe_mean(lzma_vals),
                'std': _safe_std(lzma_vals),
                'per_binary': lzma_per_binary,
            },
        }

    print('      Done.', flush=True)
    return {
        'lengths_tested': [str(l) if l is not None else 'full' for l in lengths],
        'by_length': results_by_length,
        'interpretation': (
            'Stable metrics across truncation levels indicate that conclusions '
            'are not driven by long tails of large binaries.'
        ),
    }


# ---------------------------------------------------------------------------
# 5. Leave-one-binary-out (LOBO) stability
# ---------------------------------------------------------------------------

def run_leave_one_out(cache: CorpusCache) -> Dict:
    """
    For each binary, remove it and recompute corpus-level Zipf alpha and 5-gram
    entropy rate. Report range and flag influential binaries (|Δalpha| > 10%).
    """
    print(f'\n[5/5] Leave-one-binary-out stability ({cache.n} binaries) ...',
          flush=True)
    V = cache.V

    baseline_alpha = _fast_zipf_alpha_mle(cache.enc_seqs)
    baseline_h5 = _fast_entropy_rate_at_n(cache.enc_seqs, n=5, V=V)
    print(f'      Baseline: alpha={baseline_alpha:.4f}  h5_rate={baseline_h5:.4f}',
          flush=True)

    per_binary: Dict[str, Any] = {}

    for i, binary in enumerate(cache.binaries):
        subset = [cache.enc_seqs[j] for j in range(cache.n) if j != i]
        alpha = _fast_zipf_alpha_mle(subset)
        h5 = _fast_entropy_rate_at_n(subset, n=5, V=V)

        delta_alpha = alpha - baseline_alpha if not np.isnan(alpha) else float('nan')
        delta_h5 = h5 - baseline_h5 if not np.isnan(h5) else float('nan')
        pct_change_alpha = (
            abs(delta_alpha) / abs(baseline_alpha) * 100.0
            if not np.isnan(delta_alpha) and baseline_alpha
            else float('nan')
        )
        influential = bool(not np.isnan(pct_change_alpha) and pct_change_alpha > 10.0)

        per_binary[binary.name] = {
            'alpha_without': alpha,
            'h5_rate_without': h5,
            'delta_alpha': delta_alpha,
            'delta_h5_rate': delta_h5,
            'pct_change_alpha': pct_change_alpha,
            'influential': influential,
        }
        if influential:
            print(f'      [!] Influential: {binary.name}  '
                  f'Δalpha={delta_alpha:+.4f} ({pct_change_alpha:.1f}%)',
                  flush=True)

    all_alphas = [v['alpha_without'] for v in per_binary.values()
                  if not np.isnan(v['alpha_without'])]
    all_h5s = [v['h5_rate_without'] for v in per_binary.values()
               if not np.isnan(v['h5_rate_without'])]
    influential_names = [name for name, v in per_binary.items() if v['influential']]

    print('      Done.', flush=True)
    return {
        'baseline_alpha': baseline_alpha,
        'baseline_h5_rate': baseline_h5,
        'per_binary': per_binary,
        'alpha_range': {
            'min': float(np.min(all_alphas)) if all_alphas else float('nan'),
            'max': float(np.max(all_alphas)) if all_alphas else float('nan'),
        },
        'h5_range': {
            'min': float(np.min(all_h5s)) if all_h5s else float('nan'),
            'max': float(np.max(all_h5s)) if all_h5s else float('nan'),
        },
        'influential_binaries': influential_names,
        'n_influential': len(influential_names),
        'interpretation': (
            'Influential = removal changes alpha by >10%. '
            'Zero influential binaries means corpus-level findings are robust.'
        ),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _is_stable(ci_low: float, ci_high: float, point: float,
               threshold_pct: float = 10.0) -> str:
    """
    Stable if the 95% CI width is less than threshold_pct% of the point estimate.
    """
    if np.isnan(ci_low) or np.isnan(ci_high) or np.isnan(point) or point == 0:
        return 'unknown'
    width_pct = (ci_high - ci_low) / abs(point) * 100.0
    return 'yes' if width_pct < threshold_pct else 'no'


def print_summary_table(bootstrap_results: Dict) -> None:
    metrics = bootstrap_results['metrics']
    col_w = [28, 16, 10, 10, 10, 8]
    header = ['metric', 'point_estimate', 'CI_low', 'CI_high', 'SE', 'stable?']
    divider = '+' + '+'.join('-' * w for w in col_w) + '+'
    row_fmt = '|' + '|'.join(f'{{:<{w}}}' for w in col_w) + '|'

    print('\n' + '=' * 88)
    print(' BOOTSTRAP ROBUSTNESS SUMMARY (95% CI, N={})'.format(
        bootstrap_results['n_boot']))
    print('=' * 88)
    print(divider)
    print(row_fmt.format(*header))
    print(divider)

    for name, data in metrics.items():
        pe = data['point_estimate']
        ci_lo = data['ci_low_95']
        ci_hi = data['ci_high_95']
        se = data['standard_error']
        stable = _is_stable(ci_lo, ci_hi, pe)

        def _fmt(v: float) -> str:
            return f'{v:.4f}' if not np.isnan(v) else 'NaN'

        print(row_fmt.format(
            name[:col_w[0] - 1],
            _fmt(pe),
            _fmt(ci_lo),
            _fmt(ci_hi),
            _fmt(se),
            stable,
        ))

    print(divider)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_corpus() -> List[Binary]:
    """Load corpus from the first candidate path that exists."""
    for path in CORPUS_CANDIDATES:
        if path.exists():
            print(f'Loading corpus from: {path}', flush=True)
            binaries = load_pickle(path)
            print(f'  Loaded {len(binaries)} binaries, '
                  f'{sum(b.instruction_count for b in binaries):,} total instructions.',
                  flush=True)
            return binaries

    print('\nERROR: No corpus file found. Tried:')
    for p in CORPUS_CANDIDATES:
        print(f'  {p}')
    print('\nBuild a corpus first with the appropriate pipeline script, e.g.:')
    print('  python3 corpus/build_smoke.py')
    sys.exit(1)


def main() -> None:
    np.random.seed(42)
    rng = np.random.default_rng(42)

    binaries = load_corpus()

    if len(binaries) < 3:
        print(f'ERROR: Corpus has only {len(binaries)} binaries; need at least 3.')
        sys.exit(1)

    # Pre-build shared vocabulary
    print('Building shared vocabulary ...', flush=True)
    vocab = build_vocabulary(binaries)
    print(f'  Vocabulary size: {len(vocab)}', flush=True)

    # Build pre-computed cache (one-time encoding + compression)
    print('Building corpus cache ...', flush=True)
    cache = CorpusCache(binaries, vocab)

    all_results: Dict[str, Any] = {
        'corpus': {
            'n_binaries': len(binaries),
            'binary_names': [b.name for b in binaries],
            'total_instructions': sum(b.instruction_count for b in binaries),
        },
        'configuration': {
            'n_boot': N_BOOT,
            'n_subsample_draws': N_SUBSAMPLE_DRAWS,
            'subsample_fractions': SUBSAMPLE_FRACTIONS,
            'truncation_lengths': TRUNCATION_LENGTHS,
            'shuffled_seeds': SHUFFLED_SEEDS,
            'rng_seed': 42,
        },
    }

    # --- Test 1: Bootstrap CIs ---
    all_results['bootstrap_ci'] = run_bootstrap_ci(cache, rng)

    # --- Test 2: Multi-seed shuffled baselines ---
    all_results['multi_seed_shuffled'] = run_multi_seed_shuffled(cache)

    # --- Test 3: Subsampling sensitivity ---
    all_results['subsampling_sensitivity'] = run_subsampling_sensitivity(cache, rng)

    # --- Test 4: Truncation sensitivity ---
    all_results['truncation_sensitivity'] = run_truncation_sensitivity(cache)

    # --- Test 5: Leave-one-binary-out ---
    all_results['leave_one_out'] = run_leave_one_out(cache)

    # --- Save JSON ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None  # JSON null for NaN/Inf
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Not serializable: {type(obj)}')

    with open(OUTPUT_PATH, 'w') as fh:
        json.dump(all_results, fh, indent=2, default=_json_default)
    print(f'\nResults written to: {OUTPUT_PATH}', flush=True)

    # --- Summary table ---
    print_summary_table(all_results['bootstrap_ci'])

    # --- Additional quick summaries ---
    lobo = all_results['leave_one_out']
    a_min = lobo['alpha_range']['min']
    a_max = lobo['alpha_range']['max']
    a_base = lobo['baseline_alpha']
    print(f'Leave-one-out alpha range: [{a_min:.4f}, {a_max:.4f}]  '
          f'(baseline={a_base:.4f})')
    if lobo['influential_binaries']:
        print(f'  Influential binaries (>10% delta_alpha): '
              f'{lobo["influential_binaries"]}')
    else:
        print('  No influential binaries found (all removals change alpha <10%).')

    print('\nSubsampling: Zipf alpha mean by corpus fraction:')
    sub = all_results['subsampling_sensitivity']['by_fraction']
    for frac in SUBSAMPLE_FRACTIONS:
        data = sub[str(frac)]
        m = data['zipf_alpha_mle']['mean']
        s = data['zipf_alpha_mle']['std']
        s_str = f'  std={s:.4f}' if s is not None and not np.isnan(s) else ''
        print(f'  fraction={frac:.2f}  alpha_mean={m:.4f}{s_str}')

    print('\nShuffled baseline gap stability (CV < 0.05 => stable):')
    for n_key, entry in all_results['multi_seed_shuffled']['per_n'].items():
        gap = entry['gap_real_minus_shuffled']
        cv = entry['shuffled_cv']
        stable = entry['gap_stable']
        gap_str = f'{gap:.4f}' if gap is not None and not np.isnan(gap) else 'None'
        cv_str = f'{cv:.4f}' if cv is not None and not np.isnan(cv) else 'None'
        print(f'  n={n_key}  gap={gap_str}  CV={cv_str}  stable={stable}')


if __name__ == '__main__':
    main()
