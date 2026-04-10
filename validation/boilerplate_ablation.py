"""
Boilerplate stripping ablation for OpCode-Stats.

Tests whether the statistical structure observed in x86-64 binary instruction
sequences (Zipf law, entropy rate gap, compression) is merely an artefact of
function prologues/epilogues, endbr64 guards, and other compiler boilerplate.

Each ablation variant strips a different category of boilerplate from every
function in the corpus, then recomputes the key metrics and compares against
the unmodified baseline.

Run standalone:
    python3 validation/boilerplate_ablation.py
"""

import sys
import json
import copy
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Bootstrap project root before any project imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Non-interactive backend must be set before any matplotlib import.
import matplotlib
matplotlib.use("Agg")

import numpy as np

from utils.helpers import Instruction, Function, Binary, load_pickle

from analysis.frequency import (
    compute_frequency_distribution,
    compute_rank_frequency,
    fit_zipf_mle,
)
from analysis.ngrams import compute_entropy_rate, compute_shuffled_entropy_rates
from analysis.compression import compute_compression_ratios

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("boilerplate_ablation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum instructions a function must retain after stripping to be included.
MIN_FUNC_INSTRUCTIONS = 5

# Corpus pickle paths tried in order.
CORPUS_PATHS = [
    _PROJECT_ROOT / "results" / "smoke_fresh" / "corpus" / "corpus.pkl",
    _PROJECT_ROOT / "results" / "smoke_rerun" / "corpus" / "corpus.pkl",
    _PROJECT_ROOT / "results" / "smoke" / "corpus" / "corpus.pkl",
    _PROJECT_ROOT / "results" / "system" / "corpus" / "corpus.pkl",
    _PROJECT_ROOT / "results" / "coreutils_system" / "corpus" / "corpus.pkl",
]

OUTPUT_DIR = _PROJECT_ROOT / "validation" / "results"
OUTPUT_FILE = OUTPUT_DIR / "boilerplate_ablation.json"

# Common prologue mnemonics (in order, prefix-match from function start).
PROLOGUE_MNEMONICS = {"endbr64", "push", "mov", "sub"}

# Common epilogue patterns (suffix-match from function end, rightmost first).
EPILOGUE_PATTERNS: List[Tuple[str, ...]] = [
    ("add", "pop", "ret"),
    ("leave", "ret"),
    ("pop", "ret"),
]

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus(paths: List[Path]) -> List[Binary]:
    """Load the first corpus pickle that exists from the given path list."""
    for p in paths:
        if p.exists():
            print(f"Loading corpus from {p} ...")
            binaries = load_pickle(p)
            print(
                f"  {len(binaries)} binaries, "
                f"{sum(b.instruction_count for b in binaries):,} total instructions"
            )
            return binaries
    raise FileNotFoundError(
        f"No corpus found. Tried:\n" + "\n".join(f"  {p}" for p in paths)
    )


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------


def _make_function(f: Function, new_instrs: List[Instruction]) -> Optional[Function]:
    """Return a new Function with new_instrs, or None if too short."""
    if len(new_instrs) < MIN_FUNC_INSTRUCTIONS:
        return None
    return Function(name=f.name, binary_name=f.binary_name, instructions=new_instrs)


def _make_binary(b: Binary, new_funcs: List[Function]) -> Optional[Binary]:
    """Return a new Binary with new_funcs, or None if no functions survive."""
    if not new_funcs:
        return None
    return Binary(
        name=b.name,
        path=b.path,
        functions=new_funcs,
        inode=b.inode,
        file_size=b.file_size,
        category=b.category,
        compiler=b.compiler,
    )


def _ablate_binary(b: Binary, strip_fn) -> Optional[Binary]:
    """Apply strip_fn to every function in b, drop short survivors, return new Binary."""
    new_funcs = []
    for f in b.functions:
        new_instrs = strip_fn(list(f.instructions))
        result = _make_function(f, new_instrs)
        if result is not None:
            new_funcs.append(result)
    return _make_binary(b, new_funcs)


def ablate_corpus(binaries: List[Binary], strip_fn) -> List[Binary]:
    """Apply strip_fn-based ablation to every binary; drop binaries with no survivors."""
    result = []
    for b in binaries:
        new_b = _ablate_binary(b, strip_fn)
        if new_b is not None:
            result.append(new_b)
    return result


# ---------------------------------------------------------------------------
# Strip functions (each takes List[Instruction] and returns List[Instruction])
# ---------------------------------------------------------------------------


def _strip_first_k(instrs: List[Instruction], k: int) -> List[Instruction]:
    return instrs[k:]


def _strip_last_k(instrs: List[Instruction], k: int) -> List[Instruction]:
    if k == 0:
        return instrs
    return instrs[:-k]


def _strip_both_k(instrs: List[Instruction], k: int) -> List[Instruction]:
    return instrs[k:-k] if k > 0 and len(instrs) > 2 * k else []


def _remove_endbr64(instrs: List[Instruction]) -> List[Instruction]:
    return [i for i in instrs if i.mnemonic != "endbr64"]


def _strip_prologue_pattern(instrs: List[Instruction]) -> List[Instruction]:
    """
    Remove a leading sequence of instructions whose mnemonics match a prefix of
    the canonical prologue pattern.

    The full pattern is [endbr64, push, mov, sub], but endbr64 is treated as
    optional so the matcher also handles sequences that start with push/mov/sub
    (e.g. after endbr64 has already been removed by _remove_endbr64).

    An instruction is consumed only if its mnemonic equals the next expected
    element of the pattern.  Matching stops at the first mismatch.
    """
    # endbr64 is optional: try with it first; if the first instruction is not
    # endbr64 start from "push" so that push/mov/sub prologues are also caught.
    full_pattern = ["endbr64", "push", "mov", "sub"]
    no_endbr_pattern = ["push", "mov", "sub"]

    def _match_pattern(seq: List[Instruction], pattern: List[str]) -> int:
        """Return the number of leading instructions that match the pattern prefix."""
        cut = 0
        for instr, expected in zip(seq, pattern):
            if instr.mnemonic == expected:
                cut += 1
            else:
                break
        return cut

    # Choose the pattern that yields more cuts (never fewer than 0).
    cut_full = _match_pattern(instrs, full_pattern)
    cut_no_endbr = _match_pattern(instrs, no_endbr_pattern)
    cut = max(cut_full, cut_no_endbr)
    return instrs[cut:]


def _strip_epilogue_pattern(instrs: List[Instruction]) -> List[Instruction]:
    """
    Remove a trailing sequence matching one of the epilogue patterns.

    Tries each pattern from longest to shortest.  Checks whether the tail of
    the instruction list matches the pattern; if so, removes those instructions.
    Only one pattern is applied.
    """
    mnemonics = [i.mnemonic for i in instrs]
    for pattern in EPILOGUE_PATTERNS:
        plen = len(pattern)
        if len(mnemonics) >= plen and tuple(mnemonics[-plen:]) == pattern:
            return instrs[:-plen]
    return instrs


def _full_boilerplate_strip(instrs: List[Instruction]) -> List[Instruction]:
    """Remove endbr64 anywhere, then strip prologue pattern, then strip epilogue pattern."""
    instrs = _remove_endbr64(instrs)
    instrs = _strip_prologue_pattern(instrs)
    instrs = _strip_epilogue_pattern(instrs)
    return instrs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(binaries: List[Binary]) -> Dict[str, Any]:
    """Compute the core ablation metrics for a corpus variant."""
    # --- Zipf MLE ---
    freqs = compute_frequency_distribution(binaries)
    rank_freq = compute_rank_frequency(freqs)
    zipf = fit_zipf_mle(rank_freq, n_bootstrap=100)

    # --- Entropy rates (real and shuffled) ---
    entropy_data = compute_entropy_rate(binaries, max_n=5)
    shuffled_data = compute_shuffled_entropy_rates(binaries, max_n=5, seed=42)

    entropy_rates = [d["entropy_rate"] for d in entropy_data]
    shuffled_rates = [d["entropy_rate"] for d in shuffled_data]

    # Entropy rate gap at each n: real - shuffled.
    entropy_gaps = [r - s for r, s in zip(entropy_rates, shuffled_rates)]

    # --- Compression ---
    comp_results = compute_compression_ratios(binaries)
    zlib_mean = float(np.mean([r["zlib_ratio"] for r in comp_results])) if comp_results else 1.0
    lzma_mean = float(np.mean([r["lzma_ratio"] for r in comp_results])) if comp_results else 1.0

    total_instructions = sum(b.instruction_count for b in binaries)
    n_functions = sum(b.function_count for b in binaries)

    return {
        "corpus_stats": {
            "n_binaries": len(binaries),
            "n_functions": n_functions,
            "total_instructions": total_instructions,
        },
        "zipf_alpha": zipf.get("alpha_mle", 0.0),
        "zipf_alpha_ci_low": zipf.get("alpha_ci_low", 0.0),
        "zipf_alpha_ci_high": zipf.get("alpha_ci_high", 0.0),
        "zipf_ks_stat": zipf.get("ks_stat", 0.0),
        "entropy_rates": entropy_rates,
        "shuffled_rates": shuffled_rates,
        "entropy_gaps": entropy_gaps,
        "compression": {
            "zlib_mean": zlib_mean,
            "lzma_mean": lzma_mean,
        },
    }


def _pct_change(new_val: float, orig_val: float) -> float:
    """Percent change from orig_val to new_val."""
    if orig_val == 0.0:
        return 0.0
    return 100.0 * (new_val - orig_val) / abs(orig_val)


def add_pct_changes(variant_metrics: Dict, original_metrics: Dict) -> Dict:
    """Augment variant_metrics with percentage-change entries vs. original."""
    pct: Dict[str, Any] = {}

    pct["zipf_alpha"] = _pct_change(variant_metrics["zipf_alpha"], original_metrics["zipf_alpha"])
    pct["entropy_rates"] = [
        _pct_change(v, o)
        for v, o in zip(variant_metrics["entropy_rates"], original_metrics["entropy_rates"])
    ]
    pct["shuffled_rates"] = [
        _pct_change(v, o)
        for v, o in zip(variant_metrics["shuffled_rates"], original_metrics["shuffled_rates"])
    ]
    pct["entropy_gaps"] = [
        _pct_change(v, o)
        for v, o in zip(variant_metrics["entropy_gaps"], original_metrics["entropy_gaps"])
    ]
    pct["zlib_mean"] = _pct_change(
        variant_metrics["compression"]["zlib_mean"],
        original_metrics["compression"]["zlib_mean"],
    )
    pct["lzma_mean"] = _pct_change(
        variant_metrics["compression"]["lzma_mean"],
        original_metrics["compression"]["lzma_mean"],
    )

    result = dict(variant_metrics)
    result["pct_change_from_original"] = pct
    return result


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def build_summary(original: Dict, variants: Dict[str, Dict]) -> Dict[str, Any]:
    """Decide whether findings survive ablation and identify most/least affected metric."""

    # A finding "survives" if:
    #   1. entropy gap at n=5 remains negative (real < shuffled = sequential structure)
    #   2. zlib compression is still below some loose threshold (< 0.95)
    #   3. zipf_alpha remains > 0.5 (non-trivial power-law)
    survivor_count = 0
    fail_reasons: List[str] = []

    gap5_orig = original["entropy_gaps"][4] if len(original["entropy_gaps"]) > 4 else 0.0
    zlib_orig = original["compression"]["zlib_mean"]

    for name, v in variants.items():
        gap5 = v["entropy_gaps"][4] if len(v["entropy_gaps"]) > 4 else 0.0
        zlib = v["compression"]["zlib_mean"]
        alpha = v["zipf_alpha"]

        if gap5 < 0 and zlib < 0.95 and alpha > 0.5:
            survivor_count += 1
        else:
            fail_reasons.append(
                f"{name}: gap5={gap5:.4f}, zlib={zlib:.3f}, alpha={alpha:.3f}"
            )

    findings_survive = survivor_count == len(variants)

    # Identify most and least affected metric by max absolute pct change across variants.
    metric_max_abs_pct: Dict[str, float] = {
        "zipf_alpha": 0.0,
        "entropy_rate_H5": 0.0,
        "entropy_gap_H5": 0.0,
        "zlib_mean": 0.0,
        "lzma_mean": 0.0,
    }

    for v in variants.values():
        pct = v.get("pct_change_from_original", {})
        metric_max_abs_pct["zipf_alpha"] = max(
            metric_max_abs_pct["zipf_alpha"], abs(pct.get("zipf_alpha", 0.0))
        )
        er = pct.get("entropy_rates", [0.0] * 5)
        eg = pct.get("entropy_gaps", [0.0] * 5)
        if len(er) >= 5:
            metric_max_abs_pct["entropy_rate_H5"] = max(
                metric_max_abs_pct["entropy_rate_H5"], abs(er[4])
            )
        if len(eg) >= 5:
            metric_max_abs_pct["entropy_gap_H5"] = max(
                metric_max_abs_pct["entropy_gap_H5"], abs(eg[4])
            )
        metric_max_abs_pct["zlib_mean"] = max(
            metric_max_abs_pct["zlib_mean"], abs(pct.get("zlib_mean", 0.0))
        )
        metric_max_abs_pct["lzma_mean"] = max(
            metric_max_abs_pct["lzma_mean"], abs(pct.get("lzma_mean", 0.0))
        )

    most_affected = max(metric_max_abs_pct, key=metric_max_abs_pct.get)
    least_affected = min(metric_max_abs_pct, key=metric_max_abs_pct.get)

    if findings_survive:
        interpretation = (
            "All boilerplate-stripped variants preserve the core statistical properties: "
            "positive entropy rate gap (sequential structure beyond unigrams), "
            "sub-random compression ratios, and Zipfian frequency distribution. "
            "The observed structure is intrinsic to the code logic, not an artefact "
            "of function prologues, epilogues, or endbr64 guards."
        )
    else:
        interpretation = (
            f"Some variants failed at least one survival criterion. "
            f"Failing variants: {'; '.join(fail_reasons)}. "
            f"Results should be interpreted with caution."
        )

    return {
        "findings_survive_ablation": findings_survive,
        "variants_passing": survivor_count,
        "variants_total": len(variants),
        "most_affected_metric": most_affected,
        "most_affected_metric_max_pct_change": round(metric_max_abs_pct[most_affected], 2),
        "least_affected_metric": least_affected,
        "least_affected_metric_max_pct_change": round(metric_max_abs_pct[least_affected], 2),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------


def print_summary_table(original: Dict, variants: Dict[str, Dict]) -> None:
    """Print an ASCII table: variant | zipf_α | H5_rate | gap_H5 | zlib_ratio | conclusion."""

    col_widths = [36, 8, 9, 9, 11, 12]
    headers = ["variant", "zipf_α", "H5_rate", "gap_H5", "zlib_ratio", "conclusion"]

    def row_str(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths))

    sep = "-" * sum(col_widths + [2 * (len(col_widths) - 1)])
    print()
    print("Boilerplate Ablation Summary")
    print(sep)
    print(row_str(headers))
    print(sep)

    def variant_row(name: str, m: Dict) -> None:
        alpha = m["zipf_alpha"]
        h5 = m["entropy_rates"][4] if len(m["entropy_rates"]) > 4 else float("nan")
        gap5 = m["entropy_gaps"][4] if len(m["entropy_gaps"]) > 4 else float("nan")
        zlib = m["compression"]["zlib_mean"]
        ok = gap5 < 0 and zlib < 0.95 and alpha > 0.5
        conclusion = "PASS" if ok else "FAIL"
        print(row_str([
            name[:36],
            f"{alpha:.4f}",
            f"{h5:.4f}",
            f"{gap5:.4f}",
            f"{zlib:.4f}",
            conclusion,
        ]))

    variant_row("original (no stripping)", original)
    print(sep)
    for name, m in variants.items():
        variant_row(name, m)

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus.
    binaries = load_corpus(CORPUS_PATHS)

    # 2. Baseline metrics (no stripping).
    print("Computing baseline metrics (original corpus) ...")
    t0 = time.time()
    original_metrics = compute_metrics(binaries)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 3. Define all ablation variants.
    #    Each entry: (label, strip_function)
    k_values = [1, 2, 3, 5]

    variant_defs: List[Tuple[str, Any]] = []

    # a. Strip first k
    for k in k_values:
        variant_defs.append((f"strip_first_{k}", lambda instrs, _k=k: _strip_first_k(instrs, _k)))

    # b. Strip last k
    for k in k_values:
        variant_defs.append((f"strip_last_{k}", lambda instrs, _k=k: _strip_last_k(instrs, _k)))

    # c. Strip both first k and last k
    for k in k_values:
        variant_defs.append((f"strip_both_{k}", lambda instrs, _k=k: _strip_both_k(instrs, _k)))

    # d. Remove all endbr64
    variant_defs.append(("remove_endbr64", _remove_endbr64))

    # e. Remove common prologue patterns
    variant_defs.append(("strip_prologue_pattern", _strip_prologue_pattern))

    # f. Remove common epilogue patterns
    variant_defs.append(("strip_epilogue_pattern", _strip_epilogue_pattern))

    # g. Full boilerplate strip (d + e + f)
    variant_defs.append(("full_boilerplate_strip", _full_boilerplate_strip))

    # 4. Run metrics for each variant.
    variants_output: Dict[str, Dict] = {}

    for label, strip_fn in variant_defs:
        print(f"Ablating: {label} ...", end="", flush=True)
        t0 = time.time()
        ablated = ablate_corpus(binaries, strip_fn)

        if not ablated:
            print(f" SKIPPED (no binaries survived)")
            continue

        total_instrs = sum(b.instruction_count for b in ablated)
        total_funcs = sum(b.function_count for b in ablated)
        print(
            f" {len(ablated)} binaries, {total_funcs} funcs, {total_instrs:,} instrs ... ",
            end="",
            flush=True,
        )

        metrics = compute_metrics(ablated)
        metrics_with_pct = add_pct_changes(metrics, original_metrics)
        variants_output[label] = metrics_with_pct
        print(f"done ({time.time() - t0:.1f}s)")

    # 5. Build summary.
    summary = build_summary(original_metrics, variants_output)

    # 6. Print table.
    print_summary_table(original_metrics, variants_output)

    # 7. Print summary conclusions.
    print("Summary:")
    print(f"  findings_survive_ablation : {summary['findings_survive_ablation']}")
    print(f"  variants passing          : {summary['variants_passing']} / {summary['variants_total']}")
    print(f"  most affected metric      : {summary['most_affected_metric']} "
          f"(max {summary['most_affected_metric_max_pct_change']:.1f}% change)")
    print(f"  least affected metric     : {summary['least_affected_metric']} "
          f"(max {summary['least_affected_metric_max_pct_change']:.1f}% change)")
    print()
    print(f"  {summary['interpretation']}")
    print()

    # 8. Write JSON output.
    output = {
        "original": original_metrics,
        "variants": variants_output,
        "summary": summary,
    }

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(OUTPUT_FILE, "w") as fh:
        json.dump(output, fh, indent=2, default=_json_default)

    print(f"Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
