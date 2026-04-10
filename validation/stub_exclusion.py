"""
Stub/Startup Function Exclusion Validation.

Classifies functions from the corpus into three categories and reruns the
core analysis on each, letting us quantify how much of the observed statistical
structure originates from generic toolchain boilerplate vs. application code.

Categories
----------
1. startup_runtime  - _start, __libc_*, _init/_fini, CRT boilerplate, etc.
2. stubs            - PLT/GOT trampolines and tiny stub functions (< 5 instrs)
3. internal         - everything else (application code)

Run standalone:
    python3 validation/stub_exclusion.py
"""

import sys
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, '/home/aaslyan/OpCode-Stats')

import matplotlib
matplotlib.use('Agg')

import numpy as np

from utils.helpers import (
    Instruction, Function, Binary, load_pickle, save_json
)
from analysis.frequency import (
    compute_frequency_distribution, compute_rank_frequency, fit_zipf_mle
)
from analysis.ngrams import compute_entropy_rate, compute_shuffled_entropy_rates
from analysis.compression import compute_compression_ratios

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stub_exclusion")

# ---------------------------------------------------------------------------
# Classification patterns
# ---------------------------------------------------------------------------

# Exact-name matches and prefix/suffix patterns for startup/runtime functions
_STARTUP_EXACT: frozenset = frozenset({
    "_start", "_init", "_fini", "__gmon_start__",
    "deregister_tm_clones", "register_tm_clones",
    "__libc_csu_init", "__libc_csu_fini",
    "frame_dummy",
})

_STARTUP_PREFIXES: Tuple[str, ...] = (
    "__libc_", "_IO_", "__cxa_", "_dl_", "__do_global_",
)

# Patterns that indicate PLT/GOT stubs
_STUB_SUBSTRINGS: Tuple[str, ...] = (".plt", "@plt", ".got")
_STUB_SUFFIXES: Tuple[str, ...] = (".plt.sec",)

# Tiny stub: fewer than this many instructions
_STUB_MAX_INSTRS = 5
# Mnemonics that appear in trivial trampolines
_TRAMPOLINE_MNEMONICS: frozenset = frozenset({"jmp", "endbr64", "nop", "ret"})


def _classify_function(func: Function) -> str:
    """Return 'startup_runtime', 'stubs', or 'internal'."""
    name = func.name

    # --- Startup / runtime ---
    if name in _STARTUP_EXACT:
        return "startup_runtime"
    if any(name.startswith(p) for p in _STARTUP_PREFIXES):
        return "startup_runtime"

    # --- Stubs ---
    name_lower = name.lower()
    if any(sub in name_lower for sub in _STUB_SUBSTRINGS):
        return "stubs"
    if any(name_lower.endswith(suf) for suf in _STUB_SUFFIXES):
        return "stubs"

    # Short functions that contain only jmp / trampoline mnemonics
    if len(func.instructions) < _STUB_MAX_INSTRS:
        mnems = {instr.mnemonic for instr in func.instructions}
        if mnems and mnems.issubset(_TRAMPOLINE_MNEMONICS):
            return "stubs"

    # --- Internal application code ---
    return "internal"


# ---------------------------------------------------------------------------
# Binary reconstruction helpers
# ---------------------------------------------------------------------------

def _rebuild_binary(orig: Binary, functions: List[Function], suffix: str) -> Binary:
    """Return a new Binary sharing metadata with *orig* but holding *functions*."""
    return Binary(
        name=orig.name + suffix,
        path=orig.path,
        functions=functions,
        inode=orig.inode,
        file_size=orig.file_size,
        category=orig.category,
        compiler=orig.compiler,
    )


def _split_binaries(binaries: List[Binary]) -> Dict[str, List[Binary]]:
    """
    For every original binary build three filtered copies (one per category).

    Returns a dict mapping category name -> list[Binary].  Only non-empty
    binaries are included.
    """
    buckets: Dict[str, List[Binary]] = {
        "startup_runtime": [],
        "stubs": [],
        "internal": [],
    }

    for binary in binaries:
        per_cat: Dict[str, List[Function]] = {k: [] for k in buckets}
        for func in binary.functions:
            cat = _classify_function(func)
            per_cat[cat].append(func)

        for cat, funcs in per_cat.items():
            if funcs:
                buckets[cat].append(_rebuild_binary(binary, funcs, f"__{cat}"))

    return buckets


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _safe_zipf_alpha(binaries: List[Binary]) -> float:
    """Return MLE Zipf alpha; 0.0 on error or insufficient data."""
    try:
        freqs = compute_frequency_distribution(binaries)
        if not freqs:
            return 0.0
        rank_freq = compute_rank_frequency(freqs)
        result = fit_zipf_mle(rank_freq, n_bootstrap=50)
        return float(result.get("alpha_mle", 0.0))
    except Exception as exc:
        logger.warning("Zipf fit failed: %s", exc)
        return 0.0


def _safe_entropy_rates(binaries: List[Binary], max_n: int = 5) -> List[float]:
    """Return list of entropy_rate values for n=1..max_n; zeros on error."""
    try:
        rates = compute_entropy_rate(binaries, max_n=max_n)
        return [r["entropy_rate"] for r in rates]
    except Exception as exc:
        logger.warning("Entropy rate computation failed: %s", exc)
        return [0.0] * max_n


def _safe_shuffled_entropy_rates(binaries: List[Binary], max_n: int = 5) -> List[float]:
    """Return shuffled entropy rates for n=1..max_n; zeros on error."""
    try:
        rates = compute_shuffled_entropy_rates(binaries, max_n=max_n, seed=42)
        return [r["entropy_rate"] for r in rates]
    except Exception as exc:
        logger.warning("Shuffled entropy rate computation failed: %s", exc)
        return [0.0] * max_n


def _safe_compression(binaries: List[Binary]) -> Tuple[float, float]:
    """Return (mean_zlib_ratio, mean_lzma_ratio); (1.0, 1.0) on error."""
    try:
        results = compute_compression_ratios(binaries)
        if not results:
            return 1.0, 1.0
        zlib = float(np.mean([r["zlib_ratio"] for r in results]))
        lzma = float(np.mean([r["lzma_ratio"] for r in results]))
        return zlib, lzma
    except Exception as exc:
        logger.warning("Compression computation failed: %s", exc)
        return 1.0, 1.0


def _compute_metrics(binaries: List[Binary], label: str) -> Dict:
    """Run all four analyses on *binaries* and return a metrics dict."""
    if not binaries:
        return {"error": "no binaries in this category"}

    total_funcs = sum(len(b.functions) for b in binaries)
    total_instrs = sum(b.instruction_count for b in binaries)
    logger.info("  [%s] %d binaries, %d funcs, %d instrs",
                label, len(binaries), total_funcs, total_instrs)

    alpha = _safe_zipf_alpha(binaries)
    ent_rates = _safe_entropy_rates(binaries)
    shuf_rates = _safe_shuffled_entropy_rates(binaries)
    zlib_ratio, lzma_ratio = _safe_compression(binaries)

    # Entropy gap at each n: real - shuffled
    gaps = [r - s for r, s in zip(ent_rates, shuf_rates)]

    return {
        "total_binaries": len(binaries),
        "total_functions": total_funcs,
        "total_instructions": total_instrs,
        "zipf_alpha": alpha,
        "entropy_rates": ent_rates,
        "shuffled_entropy_rates": shuf_rates,
        "entropy_gaps": gaps,
        "compression": {
            "zlib_ratio_mean": zlib_ratio,
            "lzma_ratio_mean": lzma_ratio,
        },
    }


# ---------------------------------------------------------------------------
# Function classification summary
# ---------------------------------------------------------------------------

def _classification_summary(binaries: List[Binary]) -> Dict:
    """Count functions and collect name samples per category."""
    summary: Dict[str, Dict] = {
        "startup_runtime": {"count": 0, "instruction_count": 0, "function_names_sample": []},
        "stubs":           {"count": 0, "instruction_count": 0, "function_names_sample": []},
        "internal":        {"count": 0, "instruction_count": 0, "function_names_sample": []},
    }
    for binary in binaries:
        for func in binary.functions:
            cat = _classify_function(func)
            summary[cat]["count"] += 1
            summary[cat]["instruction_count"] += len(func.instructions)
            # Collect up to 20 sample names per category
            if len(summary[cat]["function_names_sample"]) < 20:
                summary[cat]["function_names_sample"].append(func.name)

    return summary


# ---------------------------------------------------------------------------
# Comparison: how much structure is toolchain vs. application?
# ---------------------------------------------------------------------------

def _comparison_section(
    all_metrics: Dict,
    startup_metrics: Dict,
    stub_metrics: Dict,
    internal_metrics: Dict,
    classification: Dict,
) -> Dict:
    total_funcs = sum(c["count"] for c in classification.values())
    toolchain_funcs = (
        classification["startup_runtime"]["count"]
        + classification["stubs"]["count"]
    )
    pct_toolchain = 100.0 * toolchain_funcs / total_funcs if total_funcs else 0.0
    pct_internal = 100.0 - pct_toolchain

    # Compression gap: all vs. internal
    all_zlib = all_metrics.get("compression", {}).get("zlib_ratio_mean", 1.0)
    int_zlib = internal_metrics.get("compression", {}).get("zlib_ratio_mean", 1.0)
    comp_diff = all_zlib - int_zlib

    # Entropy gap at n=5
    all_h5 = all_metrics.get("entropy_rates", [0]*5)
    all_h5_val = all_h5[4] if len(all_h5) >= 5 else 0.0
    int_h5 = internal_metrics.get("entropy_rates", [0]*5)
    int_h5_val = int_h5[4] if len(int_h5) >= 5 else 0.0

    return {
        "structure_is_generic_toolchain": (
            f"{pct_toolchain:.1f}% of functions are toolchain (startup + stubs)"
        ),
        "structure_is_application_specific": (
            f"{pct_internal:.1f}% of functions are internal application code"
        ),
        "compression_ratio_change_internal_vs_all": (
            f"zlib ratio shifts by {comp_diff:+.4f} when toolchain functions are excluded"
        ),
        "h5_rate_change_internal_vs_all": (
            f"H5 entropy rate shifts from {all_h5_val:.4f} to {int_h5_val:.4f} "
            f"({int_h5_val - all_h5_val:+.4f}) after removing toolchain functions"
        ),
    }


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_corpus() -> List[Binary]:
    """Try smoke corpus first, fall back to coreutils_system, then other known paths."""
    candidates = [
        Path('/home/aaslyan/OpCode-Stats/results/smoke_fresh/corpus/corpus.pkl'),
        Path('/home/aaslyan/OpCode-Stats/results/smoke_rerun/corpus/corpus.pkl'),
        Path('/home/aaslyan/OpCode-Stats/results/smoke/corpus/corpus.pkl'),
        Path('/home/aaslyan/OpCode-Stats/results/coreutils_system/corpus/corpus.pkl'),
        Path('/home/aaslyan/OpCode-Stats/results/system/corpus/corpus.pkl'),
        Path('/home/aaslyan/OpCode-Stats/results/compiler_matrix_rerun/corpus/corpus.pkl'),
    ]
    for p in candidates:
        if p.exists():
            print(f"Loading corpus from {p} ...")
            binaries = load_pickle(p)
            print(f"  {len(binaries)} binaries loaded.")
            return binaries
    raise FileNotFoundError(
        "No corpus found. Expected one of:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _print_table(rows: List[Tuple]) -> None:
    header = ("category", "n_funcs", "n_instrs", "zipf_α", "H5_rate", "zlib_ratio")
    col_widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]

    def _fmt_row(row):
        return "  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))

    sep = "  ".join("-" * w for w in col_widths)
    print("\n" + _fmt_row(header))
    print(sep)
    for row in rows:
        print(_fmt_row(row))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)

    # --- load corpus ---
    try:
        binaries = _load_corpus()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not binaries:
        print("ERROR: corpus is empty.")
        sys.exit(1)

    # --- classify ---
    print("Classifying functions ...")
    classification = _classification_summary(binaries)
    for cat, info in classification.items():
        print(f"  {cat}: {info['count']} functions, {info['instruction_count']} instructions")

    # --- split into per-category binaries ---
    print("Splitting binaries by category ...")
    buckets = _split_binaries(binaries)

    # Convenience: all categories merged (original binaries, unmodified)
    all_binaries = binaries

    # --- compute metrics ---
    print("Computing metrics for ALL functions ...")
    all_metrics     = _compute_metrics(all_binaries,         "all")

    print("Computing metrics for STARTUP/RUNTIME functions ...")
    startup_metrics = _compute_metrics(buckets["startup_runtime"], "startup_runtime")

    print("Computing metrics for STUBS ...")
    stub_metrics    = _compute_metrics(buckets["stubs"],     "stubs")

    print("Computing metrics for INTERNAL functions ...")
    internal_metrics = _compute_metrics(buckets["internal"], "internal")

    # --- comparison ---
    comparison = _comparison_section(
        all_metrics, startup_metrics, stub_metrics, internal_metrics, classification
    )

    # --- assemble output ---
    output = {
        "function_classification": classification,
        "metrics_by_category": {
            "all":             all_metrics,
            "startup_runtime": startup_metrics,
            "stubs":           stub_metrics,
            "internal":        internal_metrics,
        },
        "comparison": comparison,
    }

    # --- save ---
    out_path = Path('/home/aaslyan/OpCode-Stats/validation/results/stub_exclusion.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_serialiser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=_default_serialiser)
    print(f"\nResults saved to {out_path}")

    # --- summary table ---
    def _row(label: str, m: Dict) -> Tuple:
        n_funcs  = m.get("total_functions", 0)
        n_instrs = m.get("total_instructions", 0)
        alpha    = m.get("zipf_alpha", 0.0)
        ent      = m.get("entropy_rates", [0.0] * 5)
        h5       = ent[4] if len(ent) >= 5 else 0.0
        zlib     = m.get("compression", {}).get("zlib_ratio_mean", 0.0)
        return (
            label,
            n_funcs,
            n_instrs,
            f"{alpha:.4f}",
            f"{h5:.4f}",
            f"{zlib:.4f}",
        )

    rows = [
        _row("all",             all_metrics),
        _row("startup_runtime", startup_metrics),
        _row("stubs",           stub_metrics),
        _row("internal",        internal_metrics),
    ]
    _print_table(rows)

    # Print comparison notes
    print("Comparison:")
    for k, v in comparison.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
