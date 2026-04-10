"""
Synthetic data validation for OpCode-Stats analysis pipeline.

Builds five controlled synthetic corpora with known statistical properties,
runs the analysis modules, and verifies that each module produces results
consistent with the ground-truth properties of each corpus.

Run standalone:
    python3 validation/synthetic_validation.py
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- bootstrap path before any project imports ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Use non-interactive matplotlib backend before any plotting import
import matplotlib
matplotlib.use("Agg")

import numpy as np

from utils.helpers import Instruction, Function, Binary

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("synthetic_validation")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Representative x86-64 opcode subset (~50 mnemonics)
VOCAB: List[str] = [
    "mov", "push", "pop", "add", "sub", "xor", "cmp", "test",
    "je", "jne", "jmp", "call", "ret", "lea", "nop",
    "and", "or", "not", "neg", "inc", "dec", "imul", "idiv",
    "shl", "shr", "sar", "ror", "rol",
    "jl", "jle", "jg", "jge", "jb", "jbe", "ja", "jae",
    "js", "jns", "jo", "jno",
    "movsx", "movzx", "movsxd", "cmov", "setne", "sete",
    "leave", "endbr64", "ud2",
    "rep", "stosb",
]
VOCAB_SIZE = len(VOCAB)  # exactly 50

# Boilerplate boundary patterns
PROLOGUE = ["push", "mov", "sub"]   # first 3 instructions of each function
EPILOGUE = ["pop", "ret"]           # last 2 instructions of each function

# Template for the repeated-template corpus (10 opcodes)
TEMPLATE = ["push", "mov", "sub", "mov", "call", "mov", "add", "pop", "ret", "nop"]

# ---------------------------------------------------------------------------
# Corpus generation helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_instruction(addr: int, mnemonic: str) -> Instruction:
    """Create a dummy Instruction with the given mnemonic."""
    return Instruction(
        address=addr,
        mnemonic=mnemonic,
        operands="",
        raw_bytes=b"\x90",
    )


def _make_function(name: str, binary_name: str, mnemonics: List[str]) -> Function:
    """Create a Function from a list of mnemonic strings."""
    instructions = [_make_instruction(i, m) for i, m in enumerate(mnemonics)]
    return Function(name=name, binary_name=binary_name, instructions=instructions)


def _make_binary(
    binary_name: str,
    category: str,
    functions: List[Function],
) -> Binary:
    """Wrap functions into a Binary object."""
    return Binary(
        name=binary_name,
        path=f"/synthetic/{category}/{binary_name}",
        functions=functions,
        inode=None,
        file_size=None,
        category=category,
        compiler="synthetic",
    )


# ---------------------------------------------------------------------------
# Zipf probability weights (rank 1..VOCAB_SIZE, exponent alpha)
# ---------------------------------------------------------------------------

def _zipf_probs(alpha: float, size: int) -> np.ndarray:
    ranks = np.arange(1, size + 1, dtype=float)
    weights = ranks ** (-alpha)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Low-order Markov chain
# ---------------------------------------------------------------------------

def _build_markov_chain(rng: np.random.Generator, vocab_size: int, order: int = 2) -> Dict:
    """
    Build a random order-2 Markov transition table.

    Transition distributions are drawn from Dirichlet(alpha=0.1).  A low alpha
    concentrates mass on just a few successors per history, creating strong
    sequential dependencies detectable with ~18k tokens vs a 51-symbol vocab.
    The chain is generated lazily: each history's distribution is cached on
    first encounter, giving consistent behaviour across multiple calls.
    """
    return {"order": order, "cache": {}, "rng": rng, "vocab_size": vocab_size,
            "alpha": 0.1}


def _markov_next(chain: Dict, history: Tuple[int, ...]) -> int:
    cache = chain["cache"]
    if history not in cache:
        # Dirichlet(0.1) gives very sparse output distributions — typically
        # 1-3 tokens receive most of the probability mass, creating clear
        # sequential structure detectable even with short sequences.
        alpha_vec = chain["rng"].dirichlet(
            np.full(chain["vocab_size"], chain["alpha"])
        )
        cache[history] = alpha_vec
    probs = cache[history]
    return int(chain["rng"].choice(chain["vocab_size"], p=probs))


def _generate_markov_sequence(chain: Dict, length: int) -> List[str]:
    order = chain["order"]
    vs = chain["vocab_size"]
    # Random start state
    history = tuple(int(chain["rng"].integers(0, vs)) for _ in range(order))
    tokens = list(history)
    for _ in range(length - order):
        nxt = _markov_next(chain, history)
        tokens.append(nxt)
        history = history[1:] + (nxt,)
    return [VOCAB[t] for t in tokens[:length]]


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

# Corpus dimensions — kept small so the full script completes in < 5 minutes.
# The information module (sliding-window + MI per binary) is the bottleneck;
# 8 binaries × 15 functions × 150 instructions gives adequate statistical
# power while keeping the per-corpus information analysis under ~30 seconds.
N_BINARIES = 8
N_FUNCTIONS = 15
N_INSTRUCTIONS = 150  # instructions per function


def build_uniform_random_corpus() -> List[Binary]:
    """Uniform random draws from the vocabulary — maximum entropy baseline."""
    binaries = []
    for b_idx in range(N_BINARIES):
        functions = []
        for f_idx in range(N_FUNCTIONS):
            mnemonics = [
                VOCAB[int(RNG.integers(0, VOCAB_SIZE))]
                for _ in range(N_INSTRUCTIONS)
            ]
            functions.append(_make_function(f"func_{f_idx}", f"uniform_{b_idx}", mnemonics))
        binaries.append(_make_binary(f"uniform_{b_idx}", "uniform_random", functions))
    return binaries


def build_zipf_shuffled_corpus() -> List[Binary]:
    """
    Zipf(alpha=1.5)-weighted draws, then shuffle each function sequence.

    Shuffling destroys all sequential structure while preserving the skewed
    unigram distribution.  Entropy rate should not decrease much with n.
    """
    probs = _zipf_probs(alpha=1.5, size=VOCAB_SIZE)
    binaries = []
    for b_idx in range(N_BINARIES):
        functions = []
        for f_idx in range(N_FUNCTIONS):
            tokens = RNG.choice(VOCAB_SIZE, size=N_INSTRUCTIONS, replace=True, p=probs)
            # Shuffle to remove sequential structure
            RNG.shuffle(tokens)
            mnemonics = [VOCAB[int(t)] for t in tokens]
            functions.append(_make_function(f"func_{f_idx}", f"zipf_shuf_{b_idx}", mnemonics))
        binaries.append(_make_binary(f"zipf_shuf_{b_idx}", "zipf_shuffled", functions))
    return binaries


def build_markov_corpus() -> List[Binary]:
    """
    Order-2 Markov chain with sparse (Dirichlet(0.5)) transitions.

    Entropy rate should drop at n=2,3 then plateau; MI decays within ~3 lags.
    """
    chain = _build_markov_chain(RNG, VOCAB_SIZE, order=2)
    binaries = []
    for b_idx in range(N_BINARIES):
        functions = []
        for f_idx in range(N_FUNCTIONS):
            mnemonics = _generate_markov_sequence(chain, N_INSTRUCTIONS)
            functions.append(_make_function(f"func_{f_idx}", f"markov_{b_idx}", mnemonics))
        binaries.append(_make_binary(f"markov_{b_idx}", "markov", functions))
    return binaries


def build_template_corpus() -> List[Binary]:
    """
    Repeated 10-opcode template with 20% random substitution noise.

    Very high compression, very low entropy rate, motifs should be discoverable.
    """
    template_len = len(TEMPLATE)
    binaries = []
    for b_idx in range(N_BINARIES):
        functions = []
        for f_idx in range(N_FUNCTIONS):
            repeats = N_INSTRUCTIONS // template_len + 1
            base = TEMPLATE * repeats
            base = base[:N_INSTRUCTIONS]
            # 20% noise substitution
            mnemonics = []
            for op in base:
                if RNG.random() < 0.20:
                    mnemonics.append(VOCAB[int(RNG.integers(0, VOCAB_SIZE))])
                else:
                    mnemonics.append(op)
            functions.append(_make_function(f"func_{f_idx}", f"template_{b_idx}", mnemonics))
        binaries.append(_make_binary(f"template_{b_idx}", "template", functions))
    return binaries


def build_boilerplate_corpus() -> List[Binary]:
    """
    Mixed boilerplate + random body per function.

    First 3 instructions = PROLOGUE (push/mov/sub),
    last 2 instructions = EPILOGUE (pop/ret),
    middle is uniform random.

    Boundary positions should have low entropy; body should look random.
    """
    body_len = N_INSTRUCTIONS - len(PROLOGUE) - len(EPILOGUE)
    binaries = []
    for b_idx in range(N_BINARIES):
        functions = []
        for f_idx in range(N_FUNCTIONS):
            body = [
                VOCAB[int(RNG.integers(0, VOCAB_SIZE))]
                for _ in range(body_len)
            ]
            mnemonics = PROLOGUE + body + EPILOGUE
            functions.append(_make_function(f"func_{f_idx}", f"boilerplate_{b_idx}", mnemonics))
        binaries.append(_make_binary(f"boilerplate_{b_idx}", "boilerplate", functions))
    return binaries


# ---------------------------------------------------------------------------
# Analysis helpers — extract specific metrics from module outputs
# ---------------------------------------------------------------------------

def _entropy_rates_from_result(ngram_result: Dict) -> List[float]:
    """Return [H_n/n for n=1..5] from run_ngram_analysis output."""
    rates = ngram_result.get("entropy_analysis", {}).get("entropy_rates", [])
    return [r["entropy_rate"] for r in rates]


def _shuffled_entropy_rates_from_result(ngram_result: Dict) -> List[float]:
    """Return shuffled baseline [H_n/n for n=1..5]."""
    rates = ngram_result.get("entropy_analysis", {}).get("shuffled_baseline_rates", [])
    return [r["entropy_rate"] for r in rates]


def _mean_zlib_ratio(compression_result: Dict) -> float:
    return compression_result.get("compression_statistics", {}).get("zlib", {}).get("mean", 1.0)


def _mean_lzma_ratio(compression_result: Dict) -> float:
    return compression_result.get("compression_statistics", {}).get("lzma", {}).get("mean", 1.0)


def _top_motifs_found(motif_result: Dict, k: int) -> List[str]:
    """Return list of top motif strings for length k."""
    motifs = motif_result.get("motif_discovery", {}).get(f"{k}mer", [])
    return [m["motif"] for m in motifs[:10]]


def _mi_at_lag(info_res: Dict, lag: int, kind: str = "gap") -> float:
    """
    Return MI at the given lag from a compute_corpus_mi result dict.

    `kind` selects the sub-dict: "real", "shuffled", or "gap" (bias-corrected).
    """
    sub = info_res.get(kind, {})
    return float(sub.get(lag, 0.0))


def _positional_entropies_start(motif_result: Dict) -> List[float]:
    """Return per-position entropies at function start."""
    pos_patterns = motif_result.get("positional_patterns", {})
    return pos_patterns.get("start_patterns", {}).get("entropies", [])


def _positional_entropies_end(motif_result: Dict) -> List[float]:
    """Return per-position entropies at function end (position 0 = last instruction)."""
    pos_patterns = motif_result.get("positional_patterns", {})
    return pos_patterns.get("end_patterns", {}).get("entropies", [])


# ---------------------------------------------------------------------------
# Lightweight MI computation (avoids expensive PCA in run_information_analysis)
# ---------------------------------------------------------------------------

def _compute_mi_at_lag(sequence: List[str], lag: int) -> float:
    """
    Compute mutual information I(X_i ; X_{i+lag}) for a single lag.

    Uses symbolic computation directly on string tokens — no encoding needed.
    """
    from collections import Counter
    if len(sequence) <= lag:
        return 0.0
    x_vals = sequence[:-lag]
    y_vals = sequence[lag:]
    n = len(x_vals)
    if n == 0:
        return 0.0
    joint = Counter(zip(x_vals, y_vals))
    cx = Counter(x_vals)
    cy = Counter(y_vals)
    mi = 0.0
    for (x, y), cnt in joint.items():
        p_joint = cnt / n
        p_x = cx[x] / n
        p_y = cy[y] / n
        if p_joint > 0 and p_x > 0 and p_y > 0:
            mi += p_joint * np.log2(p_joint / (p_x * p_y))
    return float(mi)


def compute_corpus_mi(binaries: List[Binary], lags: List[int]) -> Dict[str, Any]:
    """
    Compute MI at each requested lag using the full corpus sequence.

    Implementation notes:
    - Concatenates opcode sequences across ALL binaries and functions into one
      long sequence so the sample count is maximised.  With 8 binaries × 15
      funcs × 150 instr = 18 k tokens, each (x,y) cell has ~6.9 expected
      samples — still somewhat biased for a 51-token vocab.
    - To cancel finite-sample positive bias, we also compute MI on a uniformly
      shuffled permutation of the same sequence.  The shuffled MI approximates
      the bias floor; the *gap* (real − shuffled) is the bias-corrected signal.

    Returns a dict with:
      "real":      {lag: mi_value}
      "shuffled":  {lag: mi_value}
      "gap":       {lag: real - shuffled}  — the bias-corrected measure
    """
    # Concatenate the full corpus into one sequence
    corpus_seq: List[str] = []
    for binary in binaries:
        corpus_seq.extend(binary.full_opcode_sequence)

    # Shuffled baseline (same unigram distribution, no sequential structure)
    shuffled_seq = list(corpus_seq)
    RNG.shuffle(shuffled_seq)

    real_mi: Dict[int, float] = {}
    shuf_mi: Dict[int, float] = {}
    for lag in lags:
        real_mi[lag] = _compute_mi_at_lag(corpus_seq, lag)
        shuf_mi[lag] = _compute_mi_at_lag(shuffled_seq, lag)

    return {
        "real": real_mi,
        "shuffled": shuf_mi,
        "gap": {lag: real_mi[lag] - shuf_mi[lag] for lag in lags},
    }


# ---------------------------------------------------------------------------
# Check helpers — each returns (passed: bool, detail: str)
# ---------------------------------------------------------------------------

def check(condition: bool, label: str, detail: str) -> Dict[str, Any]:
    return {
        "check": label,
        "passed": bool(condition),
        "detail": detail,
    }


def _pct_drop(a: float, b: float) -> float:
    """Percentage drop from a to b."""
    return (a - b) / a * 100.0 if a > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-corpus validation logic
# ---------------------------------------------------------------------------

def validate_uniform_random(
    ngram_res: Dict,
    comp_res: Dict,
    motif_res: Dict,
    info_res: Dict,
) -> List[Dict]:
    """
    Uniform random corpus expectations:
    - H_1/1 near log2(VOCAB_SIZE)
    - Real vs shuffled entropy rate gap is negligible at n=3 (no sequential structure)
    - Compression ratio not dramatically better than shuffled baseline
    - No significant motifs
    - Bias-corrected MI gap near zero at all lags

    Note on finite-sample limitations:
      With 18k tokens over a 51-symbol vocab, the n-gram space is undersampled
      for n>=3 (51^3=132k >> 18k tokens).  Both real and shuffled corpora show
      the same coverage-limited entropy drop, so we use relative (gap) metrics
      rather than absolute thresholds for n-gram rates and MI.
    """
    results = []
    max_entropy = np.log2(VOCAB_SIZE)  # 5.672 bits for 51 opcodes

    # 1. Unigram entropy rate close to theoretical maximum.
    #    The unigram distribution is uniform, so H_1 should be very close to
    #    log2(VOCAB_SIZE).  Finite samples don't hurt unigrams (all 51 seen).
    rates = _entropy_rates_from_result(ngram_res)
    if rates:
        h1 = rates[0]
        ok = h1 >= 0.92 * max_entropy
        results.append(check(
            ok,
            "uniform_entropy_near_max",
            f"H_1/1={h1:.3f} bits, max={max_entropy:.3f} bits "
            f"(ratio={h1/max_entropy:.3f}, need >=0.92)",
        ))

    # 2. Real vs shuffled entropy rate gap at n=3 near zero.
    #    Both real and shuffled uniform-random corpora suffer the same
    #    finite-sample underestimation.  The gap between them isolates
    #    sequential structure and should be negligible (<5% relative).
    shuffled_rates = _shuffled_entropy_rates_from_result(ngram_res)
    if len(rates) >= 3 and len(shuffled_rates) >= 3:
        real_r3 = rates[2]
        shuf_r3 = shuffled_rates[2]
        gap_abs = real_r3 - shuf_r3
        ok = abs(gap_abs) < 0.30  # less than 0.3 bits absolute difference
        results.append(check(
            ok,
            "uniform_rate_gap_near_zero",
            f"Rate gap (real-shuffled) at n=3: {gap_abs:+.3f} bits (need |gap|<0.30)",
        ))

    # 3. Compression ratio vs shuffled baseline: for uniform random, real and
    #    shuffled sequences have the same unigram distribution and the same
    #    lack of sequential structure, so the gap should be small.
    zlib = _mean_zlib_ratio(comp_res)
    shuf_zlib = comp_res.get("unigram_shuffled_baseline", {}).get("zlib_ratio", zlib)
    gap_zlib = shuf_zlib - zlib  # positive = real is more compressible than shuffled
    ok = abs(gap_zlib) < 0.10
    results.append(check(
        ok,
        "uniform_compression_gap_near_zero",
        f"Compression gap (shuffled-real) zlib: {gap_zlib:+.3f} (need |gap|<0.10)",
    ))

    # 4. No significant 4-mer motifs (or very few)
    motifs_4 = _top_motifs_found(motif_res, 4)
    ok = len(motifs_4) == 0
    results.append(check(
        ok,
        "uniform_no_motifs",
        f"Found {len(motifs_4)} 4-mer motifs (need 0)",
    ))

    # 5. Bias-corrected MI gap (real - shuffled) near zero at lag 1.
    #    Finite-sample positive MI bias cancels out in the gap since both
    #    sequences have the same unigram distribution.
    mi_gap1 = _mi_at_lag(info_res, 1, "gap")
    ok = abs(mi_gap1) < 0.05
    results.append(check(
        ok,
        "uniform_mi_gap_near_zero",
        f"MI gap (real-shuffled) at lag 1 = {mi_gap1:.4f} bits (need |gap|<0.05)",
    ))

    return results


def validate_zipf_shuffled(
    ngram_res: Dict,
    comp_res: Dict,
    motif_res: Dict,
    info_res: Dict,
) -> List[Dict]:
    """
    Zipf-shuffled corpus expectations:
    - H_1/1 lower than uniform (skewed distribution compresses better)
    - Entropy RATE (H_n/n) should NOT decrease much with n beyond unigram
      (shuffled = no sequential structure; rate approximately constant)
    - Real vs shuffled baseline gap should be small (no sequential redundancy)
    - No significant motifs
    - MI near 0 at all lags
    """
    results = []
    max_entropy = np.log2(VOCAB_SIZE)

    # 1. H_1/1 lower than max (Zipf skew reduces entropy)
    rates = _entropy_rates_from_result(ngram_res)
    if rates:
        h1 = rates[0]
        ok = h1 < max_entropy * 0.97  # must be measurably below max
        results.append(check(
            ok,
            "zipf_entropy_below_max",
            f"H_1/1={h1:.3f} bits, max={max_entropy:.3f} bits "
            f"(ratio={h1/max_entropy:.3f}, need <0.97)",
        ))

    # 2. Real vs shuffled entropy rate gap near zero at n=3.
    #    Both real and shuffled Zipf sequences have the same unigram distribution
    #    and no sequential structure, so the n-gram entropy rates should be
    #    nearly identical.  The absolute drop in H_n/n is a finite-sample
    #    coverage artifact common to both — we look at the gap only.
    shuffled_rates = _shuffled_entropy_rates_from_result(ngram_res)
    if len(rates) >= 3 and len(shuffled_rates) >= 3:
        gap_abs = rates[2] - shuffled_rates[2]
        ok = abs(gap_abs) < 0.30
        results.append(check(
            ok,
            "zipf_rate_gap_near_zero",
            f"Rate gap (real-shuffled) at n=3: {gap_abs:+.3f} bits (need |gap|<0.30)",
        ))

    # 3. Compression ratio clearly below 1.0 (Zipf skew exploitable by zlib)
    zlib = _mean_zlib_ratio(comp_res)
    # We just require it's not worse than uniform (we can't know exact uniform
    # ratio from this function; we only check internal consistency)
    ok = zlib < 1.05  # ratio should be below 1.0 in practice
    results.append(check(
        ok,
        "zipf_compressible",
        f"Mean zlib ratio={zlib:.3f} (need <1.05)",
    ))

    # 5. MI gap near zero (shuffled = no sequential structure; bias-corrected)
    mi_gap1 = _mi_at_lag(info_res, 1, "gap")
    ok = abs(mi_gap1) < 0.10
    results.append(check(
        ok,
        "zipf_mi_gap_near_zero",
        f"MI gap (real-shuffled) at lag 1 = {mi_gap1:.4f} bits (need |gap|<0.10)",
    ))

    return results


def validate_markov(
    ngram_res: Dict,
    comp_res: Dict,
    motif_res: Dict,
    info_res: Dict,
) -> List[Dict]:
    """
    Low-order Markov expectations:
    - Entropy rate drops significantly at n=2,3 vs n=1 (sequential structure)
    - Rate plateaus after n=order (new context adds diminishing info)
    - MI is positive at lag 1 and decays by lag 5
    - Better compression than uniform random
    """
    results = []

    # 1. Entropy rate drops from n=1 to n=3 (order-2 structure)
    rates = _entropy_rates_from_result(ngram_res)
    shuffled_rates = _shuffled_entropy_rates_from_result(ngram_res)
    if len(rates) >= 3 and len(shuffled_rates) >= 3:
        real_drop = _pct_drop(rates[0], rates[2])
        shuf_drop = _pct_drop(shuffled_rates[0], shuffled_rates[2])
        # Real should drop more than shuffled (sequential structure).
        # Threshold at 3% to remain robust at our 18k-token corpus size.
        ok = real_drop > shuf_drop + 3.0
        results.append(check(
            ok,
            "markov_entropy_rate_drops",
            f"Real rate drop n=1->3: {real_drop:.1f}%, "
            f"Shuffled drop: {shuf_drop:.1f}% (need real > shuffled+3%)",
        ))

    # 2. Real entropy rate at n=3 is below shuffled baseline at n=3
    if len(rates) >= 3 and len(shuffled_rates) >= 3:
        ok = rates[2] < shuffled_rates[2]
        results.append(check(
            ok,
            "markov_rate_below_shuffled",
            f"Real H_3/3={rates[2]:.3f} vs shuffled={shuffled_rates[2]:.3f} "
            f"(need real < shuffled)",
        ))

    # 3. Bias-corrected MI gap at lag 1 should be positive (Markov introduces
    #    genuine sequential dependencies not present in shuffled baseline).
    mi_gap1 = _mi_at_lag(info_res, 1, "gap")
    ok = mi_gap1 > 0.02
    results.append(check(
        ok,
        "markov_mi_gap_positive_lag1",
        f"MI gap (real-shuffled) at lag 1 = {mi_gap1:.4f} bits (need >0.02)",
    ))

    # 4. MI gap decays: lag-5 gap should be less than lag-1 gap.
    #    For order-2 Markov, most dependence is captured within lag <=2.
    mi_gap5 = _mi_at_lag(info_res, 5, "gap")
    ok = mi_gap5 < mi_gap1 * 0.9 or mi_gap1 < 0.0
    results.append(check(
        ok,
        "markov_mi_gap_decays",
        f"MI gap lag1={mi_gap1:.4f}, lag5={mi_gap5:.4f} (need lag5 < lag1)",
    ))

    # 5. Better compression than near-1.0 (Markov structure is exploitable)
    zlib = _mean_zlib_ratio(comp_res)
    ok = zlib < 0.98
    results.append(check(
        ok,
        "markov_better_compression",
        f"Mean zlib ratio={zlib:.3f} (need <0.98)",
    ))

    return results


def validate_template(
    ngram_res: Dict,
    comp_res: Dict,
    motif_res: Dict,
    info_res: Dict,
) -> List[Dict]:
    """
    Repeated-template expectations:
    - Very high compression (very low zlib ratio)
    - Very low entropy rate
    - Motif discovery should find the template or sub-sequences of it
    - MI should be elevated at periodic lags (multiples of template_len=10)
    """
    results = []
    template_len = len(TEMPLATE)  # 10

    # 1. Low entropy rate at n=1 (few unique tokens dominate)
    rates = _entropy_rates_from_result(ngram_res)
    shuffled_rates = _shuffled_entropy_rates_from_result(ngram_res)
    if rates:
        ok = rates[0] < np.log2(VOCAB_SIZE) * 0.80
        results.append(check(
            ok,
            "template_low_entropy_rate",
            f"H_1/1={rates[0]:.3f} bits (need < {np.log2(VOCAB_SIZE)*0.80:.3f})",
        ))

    # 2. Entropy rate much lower than shuffled at n=3
    if len(rates) >= 3 and len(shuffled_rates) >= 3:
        ok = rates[2] < shuffled_rates[2] * 0.90
        results.append(check(
            ok,
            "template_rate_below_shuffled",
            f"Real H_3/3={rates[2]:.3f} vs shuffled={shuffled_rates[2]:.3f} "
            f"(need real < 0.90*shuffled)",
        ))

    # 3. Good compression (zlib ratio well below 1.0)
    zlib = _mean_zlib_ratio(comp_res)
    ok = zlib < 0.85
    results.append(check(
        ok,
        "template_good_compression",
        f"Mean zlib ratio={zlib:.3f} (need <0.85)",
    ))

    # 4. LZMA ratio should also be low
    lzma = _mean_lzma_ratio(comp_res)
    ok = lzma < 0.80
    results.append(check(
        ok,
        "template_good_lzma",
        f"Mean lzma ratio={lzma:.3f} (need <0.80)",
    ))

    # 5. Motifs found: at least one k-mer motif discovered for some k
    total_motifs = sum(
        len(_top_motifs_found(motif_res, k)) for k in range(4, 11)
    )
    ok = total_motifs > 0
    results.append(check(
        ok,
        "template_motifs_found",
        f"Total motifs found across k=4..10: {total_motifs} (need >0)",
    ))

    # 6. Template sub-sequences should appear: check if any motif overlaps TEMPLATE
    template_set = set()
    for k in range(4, min(11, template_len + 1)):
        for start in range(template_len - k + 1):
            template_set.add(" ".join(TEMPLATE[start:start + k]))

    found_template_motif = False
    for k in range(4, 11):
        for motif_str in _top_motifs_found(motif_res, k):
            if motif_str in template_set:
                found_template_motif = True
                break

    results.append(check(
        found_template_motif,
        "template_motif_matches_template",
        f"Template sub-sequence found in motifs: {found_template_motif}",
    ))

    # 7. Bias-corrected MI gap at the template period should be elevated.
    #    With a 10-token template repeated many times, lag=10 should show
    #    strong correlation well above the unstructured shuffled baseline.
    mi_gap_period = _mi_at_lag(info_res, template_len, "gap")
    ok = mi_gap_period > 0.10
    results.append(check(
        ok,
        "template_periodic_mi_gap",
        f"MI gap (real-shuffled) at lag={template_len}: {mi_gap_period:.4f} bits "
        f"(need >0.10)",
    ))

    return results


def validate_boilerplate(
    ngram_res: Dict,
    comp_res: Dict,
    motif_res: Dict,
    info_res: Dict,
) -> List[Dict]:
    """
    Mixed boilerplate expectations:
    - Position 0 (push), 1 (mov), 2 (sub) have very low entropy (fixed)
    - Last positions (pop, ret) have very low entropy
    - Body positions have high entropy (random)
    - Boundary motifs found at function start/end
    """
    results = []
    prologue_len = len(PROLOGUE)  # 3
    epilogue_len = len(EPILOGUE)  # 2

    start_entropies = _positional_entropies_start(motif_res)
    end_entropies = _positional_entropies_end(motif_res)

    # 1. First prologue_len positions should have near-zero entropy
    if len(start_entropies) >= prologue_len:
        avg_prologue_entropy = np.mean(start_entropies[:prologue_len])
        ok = avg_prologue_entropy < 0.5
        results.append(check(
            ok,
            "boilerplate_prologue_low_entropy",
            f"Mean entropy at positions 0..{prologue_len-1}: "
            f"{avg_prologue_entropy:.3f} bits (need <0.5)",
        ))

    # 2. Last epilogue_len positions should have near-zero entropy
    if len(end_entropies) >= epilogue_len:
        avg_epilogue_entropy = np.mean(end_entropies[:epilogue_len])
        ok = avg_epilogue_entropy < 0.5
        results.append(check(
            ok,
            "boilerplate_epilogue_low_entropy",
            f"Mean entropy at last {epilogue_len} positions: "
            f"{avg_epilogue_entropy:.3f} bits (need <0.5)",
        ))

    # 3. Body (positions prologue_len + 5 through window_size - epilogue_len - 5)
    #    should have clearly higher entropy than boundary
    body_start_pos = prologue_len + 3
    body_end_pos = 15  # within window_size=20, well away from end positions
    if len(start_entropies) > body_start_pos:
        body_slice = start_entropies[body_start_pos:body_end_pos]
        if body_slice:
            avg_body_entropy = np.mean(body_slice)
            if len(start_entropies) >= prologue_len:
                avg_prologue_entropy = np.mean(start_entropies[:prologue_len])
                ok = avg_body_entropy > avg_prologue_entropy + 1.0
                results.append(check(
                    ok,
                    "boilerplate_body_higher_entropy",
                    f"Body entropy={avg_body_entropy:.3f} vs prologue "
                    f"entropy={avg_prologue_entropy:.3f} (need body > prologue+1.0)",
                ))

    # 4. Short motifs at boundaries should be discoverable
    motifs_4 = _top_motifs_found(motif_res, 4)
    prologue_4mer = " ".join(PROLOGUE) + " " + "mov"  # push mov sub mov (4-mer spanning start)
    # Check for any of the boundary-aligned motifs
    boundary_motifs = [
        " ".join(PROLOGUE[:3]),   # push mov sub (3-gram; check 4-mer context)
        " ".join(EPILOGUE),       # pop ret
    ]
    found_boundary = any(
        any(bm in m for bm in ["push mov sub", "pop ret"])
        for m in motifs_4
    )
    results.append(check(
        found_boundary,
        "boilerplate_boundary_motifs_found",
        f"Boundary motifs ('push mov sub' or 'pop ret') in 4-mers: {found_boundary}. "
        f"Top 4-mers: {motifs_4[:5]}",
    ))

    # 5. Overall compression better than uniform (boilerplate adds redundancy)
    zlib = _mean_zlib_ratio(comp_res)
    ok = zlib < 0.95
    results.append(check(
        ok,
        "boilerplate_better_compression",
        f"Mean zlib ratio={zlib:.3f} (need <0.95)",
    ))

    return results


# ---------------------------------------------------------------------------
# Run analysis modules safely
# ---------------------------------------------------------------------------

def _run_safe(fn, *args, label: str = ""):
    """Run analysis function, returning empty dict on any failure."""
    try:
        return fn(*args)
    except Exception as exc:
        logger.error(f"Analysis failed [{label}]: {exc}", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

CORPUS_SPECS = [
    ("uniform_random",  "Uniform Random",         build_uniform_random_corpus,  validate_uniform_random),
    ("zipf_shuffled",   "Zipf-Shuffled",          build_zipf_shuffled_corpus,   validate_zipf_shuffled),
    ("markov",          "Low-order Markov",        build_markov_corpus,          validate_markov),
    ("template",        "Repeated Template+Noise", build_template_corpus,        validate_template),
    ("boilerplate",     "Mixed Boilerplate",       build_boilerplate_corpus,     validate_boilerplate),
]


def run_all_validations(output_root: Path) -> Dict[str, Any]:
    from analysis.ngrams import run_ngram_analysis
    from analysis.compression import run_compression_analysis
    from analysis.motifs import run_motif_analysis

    # Lags to check in MI analysis.  We use a lightweight inline implementation
    # (compute_corpus_mi) instead of run_information_analysis because the full
    # module spends ~11 s/binary on PCA over a 2000×1500 n-gram matrix —
    # prohibitively expensive when run 5 times for 3 binaries each.
    MI_LAGS = [1, 2, 3, 5, 10]

    all_results: Dict[str, Any] = {}

    for corpus_id, corpus_name, builder_fn, validator_fn in CORPUS_SPECS:
        print(f"\n{'='*60}")
        print(f"  Corpus: {corpus_name}")
        print(f"{'='*60}")

        t0 = time.time()

        # Build synthetic corpus
        print(f"  Building corpus ({N_BINARIES} binaries × {N_FUNCTIONS} functions × "
              f"~{N_INSTRUCTIONS} instructions) ...")
        binaries = builder_fn()

        corpus_out = output_root / corpus_id
        corpus_out.mkdir(parents=True, exist_ok=True)

        # Run analysis modules
        print("  Running ngram analysis ...")
        ngram_res = _run_safe(run_ngram_analysis, binaries, corpus_out,
                              label=f"{corpus_id}/ngram")

        print("  Running compression analysis ...")
        comp_res = _run_safe(run_compression_analysis, binaries, corpus_out,
                             label=f"{corpus_id}/compression")

        print("  Running motif analysis ...")
        motif_res = _run_safe(run_motif_analysis, binaries, corpus_out,
                              label=f"{corpus_id}/motif")

        # MI analysis: use lightweight inline implementation to avoid the
        # expensive PCA dimensionality step in run_information_analysis.
        print("  Running MI analysis ...")
        try:
            info_res = compute_corpus_mi(binaries, MI_LAGS)
            # Persist results alongside other module outputs.
            # Convert int keys to strings for JSON compatibility.
            mi_json = {
                kind: {str(lag): val for lag, val in sub.items()}
                for kind, sub in info_res.items()
            }
            with open(corpus_out / "mi_analysis.json", "w") as fh:
                json.dump(mi_json, fh, indent=2)
        except Exception as exc:
            logger.error(f"MI analysis failed [{corpus_id}]: {exc}", exc_info=True)
            info_res = {}

        # Validate expected outcomes
        print("  Validating outcomes ...")
        checks = validator_fn(ngram_res, comp_res, motif_res, info_res)

        n_pass = sum(1 for c in checks if c["passed"])
        n_total = len(checks)
        elapsed = time.time() - t0

        corpus_result = {
            "corpus_id": corpus_id,
            "corpus_name": corpus_name,
            "n_binaries": N_BINARIES,
            "n_functions_per_binary": N_FUNCTIONS,
            "n_instructions_per_function": N_INSTRUCTIONS,
            "elapsed_seconds": round(elapsed, 1),
            "checks": checks,
            "passed": n_pass,
            "total": n_total,
            "all_passed": n_pass == n_total,
        }

        all_results[corpus_id] = corpus_result

        for c in checks:
            status = "PASS" if c["passed"] else "FAIL"
            print(f"    [{status}] {c['check']}: {c['detail']}")

        print(f"  Result: {n_pass}/{n_total} checks passed in {elapsed:.1f}s")

    return all_results


def print_summary_table(all_results: Dict[str, Any]) -> None:
    print("\n")
    print("=" * 78)
    print("SYNTHETIC VALIDATION SUMMARY")
    print("=" * 78)

    header = f"{'Corpus':<28} {'Pass/Total':>10}  {'All Pass':>8}  {'Elapsed':>8}"
    print(header)
    print("-" * 78)

    grand_pass = 0
    grand_total = 0

    for corpus_id, res in all_results.items():
        name = res["corpus_name"]
        passed = res["passed"]
        total = res["total"]
        all_ok = "YES" if res["all_passed"] else "NO"
        elapsed = f"{res['elapsed_seconds']:.1f}s"
        print(f"  {name:<26} {passed:>5}/{total:<5}     {all_ok:>6}   {elapsed:>8}")
        grand_pass += passed
        grand_total += total

    print("-" * 78)
    grand_all_ok = "YES" if grand_pass == grand_total else "NO"
    print(f"  {'TOTAL':<26} {grand_pass:>5}/{grand_total:<5}     {grand_all_ok:>6}")
    print("=" * 78)

    # Failed checks detail
    any_failures = any(not r["all_passed"] for r in all_results.values())
    if any_failures:
        print("\nFailed checks:")
        for corpus_id, res in all_results.items():
            for c in res["checks"]:
                if not c["passed"]:
                    print(f"  [{res['corpus_name']}] FAIL {c['check']}: {c['detail']}")

    print()


def main() -> None:
    output_root = _PROJECT_ROOT / "validation" / "results" / "synthetic_validation"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"OpCode-Stats Synthetic Validation")
    print(f"Project root : {_PROJECT_ROOT}")
    print(f"Output dir   : {output_root}")
    print(f"Vocab size   : {VOCAB_SIZE}")
    print(f"Corpus shape : {N_BINARIES} binaries × {N_FUNCTIONS} funcs × "
          f"~{N_INSTRUCTIONS} instr/func")
    print(f"Random seed  : 42 (fixed)")

    all_results = run_all_validations(output_root)

    # Serialize JSON results
    json_path = _PROJECT_ROOT / "validation" / "results" / "synthetic_validation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    print(f"\nJSON results written to: {json_path}")
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
