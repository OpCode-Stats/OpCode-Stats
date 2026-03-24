"""
Compiler-matrix-specific analysis:

  Q1. Does d95 (PCA intrinsic dimensionality) shrink with -O3?
  Q2. Do binaries cluster by project or by compiler?

Both use the Binary objects produced by compiler_matrix.py where
  binary.category = project name   (e.g. "sort")
  binary.compiler  = config label  (e.g. "gcc-O2")
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.ngrams import extract_ngrams
from analysis.compression import compute_compression_ratios, build_vocabulary, encode_sequence
from analysis.ngrams import run_ngram_analysis
from clustering.ncd import NCDCalculator, evaluate_category_separation
from utils.helpers import Binary, save_json

import zlib

logger = logging.getLogger(__name__)

PROJECTS  = ["sort", "hash", "compress", "search", "matrix"]
OPT_LEVELS = ["O0", "O2", "O3"]
COMPILERS  = ["gcc", "clang"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _opt(b: Binary) -> str:
    """Extract opt level string ('O0'/'O2'/'O3') from binary.compiler."""
    return b.compiler.split("-")[-1] if b.compiler else "unknown"

def _cc(b: Binary) -> str:
    """Extract compiler name ('gcc'/'clang') from binary.compiler."""
    return b.compiler.split("-")[0] if b.compiler else "unknown"


def _ngram_vectors(binaries: List[Binary], n: int = 3) -> np.ndarray:
    """L1-normalised n-gram frequency matrix, one row per binary."""
    docs = []
    for b in binaries:
        seq = b.full_opcode_sequence
        grams = extract_ngrams(seq, n)
        docs.append(" ".join("_".join(g) for g in grams))
    vec = TfidfVectorizer(use_idf=False, norm="l1", min_df=1,
                          token_pattern=r"(?u)\b[\w]+(?:_[\w]+)*\b")
    return vec.fit_transform(docs).toarray()


def _d95(matrix: np.ndarray) -> int:
    """Number of PCA components needed to explain 95 % variance."""
    if matrix.shape[0] < 2:
        return 1
    pca = PCA(n_components=min(matrix.shape))
    pca.fit(matrix)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    idx = np.searchsorted(cumvar, 0.95)
    return int(idx) + 1


def _compression_ratio(b: Binary, vocab: Dict) -> float:
    encoded = encode_sequence(b.full_opcode_sequence, vocab)
    if not encoded:
        return 1.0
    if max(encoded) < 256:
        data = bytes(encoded)
    else:
        import struct
        data = struct.pack(f'<{len(encoded)}H', *encoded)
    return len(zlib.compress(data)) / max(len(data), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Q1: metrics vs optimisation level
# ─────────────────────────────────────────────────────────────────────────────

def analyze_opt_trends(binaries: List[Binary]) -> Dict:
    """
    For each project, report compression ratio, instruction count,
    and n-gram entropy rate at O0 / O2 / O3.

    Also compute d95 across all binaries grouped by opt level.
    """
    vocab = build_vocabulary(binaries)

    # per-binary metrics
    per_binary = {}
    for b in binaries:
        per_binary[b.name] = {
            "project":      b.category,
            "compiler":     b.compiler,
            "opt":          _opt(b),
            "cc":           _cc(b),
            "instructions": b.instruction_count,
            "functions":    b.function_count,
            "zlib_ratio":   _compression_ratio(b, vocab),
        }

    # aggregate by opt level (across all projects and both compilers)
    by_opt: Dict[str, List] = defaultdict(list)
    for meta in per_binary.values():
        by_opt[meta["opt"]].append(meta)

    opt_summary = {}
    for opt, items in by_opt.items():
        opt_summary[opt] = {
            "n":               len(items),
            "mean_insns":      float(np.mean([x["instructions"] for x in items])),
            "mean_zlib_ratio": float(np.mean([x["zlib_ratio"]   for x in items])),
        }

    # d95 per opt level (using trigram vectors)
    d95_by_opt = {}
    for opt in OPT_LEVELS:
        subset = [b for b in binaries if _opt(b) == opt]
        if len(subset) >= 2:
            mat = _ngram_vectors(subset, n=3)
            d95_by_opt[opt] = _d95(mat)

    # d95 across full corpus
    full_mat = _ngram_vectors(binaries, n=3)
    d95_full = _d95(full_mat)

    # per-project: compression ratio by opt level
    by_project_opt: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for meta in per_binary.values():
        by_project_opt[meta["project"]][meta["opt"]].append(meta["zlib_ratio"])

    project_opt_compression = {
        proj: {opt: float(np.mean(vals)) for opt, vals in opts.items()}
        for proj, opts in by_project_opt.items()
    }

    return {
        "per_binary":               per_binary,
        "by_opt_level":             opt_summary,
        "d95_by_opt_level":         d95_by_opt,
        "d95_full_corpus":          d95_full,
        "project_opt_compression":  project_opt_compression,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Q2: project vs compiler clustering
# ─────────────────────────────────────────────────────────────────────────────

def analyze_project_vs_compiler(binaries: List[Binary]) -> Dict:
    """
    Mann-Whitney U tests on NCD distances:

    Groups:
      within-project  — same project, different compiler config
      within-compiler — same compiler config (e.g. gcc-O2), different project
      between         — different project AND different compiler config

    H1_project : within-project  < between  (project determines distance)
    H1_compiler: within-compiler < between  (compiler determines distance)
    """
    calculator = NCDCalculator(compressor="lzma")
    vocab = build_vocabulary(binaries)
    n = len(binaries)
    ncd_matrix = calculator.compute_ncd_matrix(binaries)

    within_project:  List[float] = []
    within_compiler: List[float] = []
    between:         List[float] = []

    pair_labels = []
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = binaries[i], binaries[j]
            d = float(ncd_matrix[i, j])
            same_proj = bi.category == bj.category
            same_cc   = bi.compiler  == bj.compiler
            if same_proj and not same_cc:
                within_project.append(d)
                pair_labels.append(("within_project", bi.name, bj.name, d))
            elif same_cc and not same_proj:
                within_compiler.append(d)
                pair_labels.append(("within_compiler", bi.name, bj.name, d))
            elif not same_proj and not same_cc:
                between.append(d)
                pair_labels.append(("between", bi.name, bj.name, d))
            # same project AND same compiler = same binary variant (skip)

    def mw(a, b, label):
        if len(a) < 2 or len(b) < 2:
            return {"note": "insufficient pairs"}
        stat, pval = mannwhitneyu(a, b, alternative="less")
        r = float(1.0 - 2.0 * stat / (len(a) * len(b)))
        return {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "U": float(stat),
            "p_value": float(pval),
            "p_value_note": "Mann-Whitney treats pairwise distances as independent; see perm_p_value",
            "effect_r": r,
            "significant": bool(pval < 0.05),
        }

    def perm_test(group_a, group_b, n_perm=10000):
        """Label-permutation test accounting for non-independence of pairwise distances."""
        if len(group_a) < 2 or len(group_b) < 2:
            return None
        observed = float(np.mean(group_b) - np.mean(group_a))
        rng = np.random.default_rng(42)
        categories = np.array([b.category for b in binaries])
        compilers = np.array([b.compiler for b in binaries])
        count_ge = 0
        for _ in range(n_perm):
            perm_cat = rng.permutation(categories)
            perm_cc = rng.permutation(compilers)
            pa, pb = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(ncd_matrix[i, j])
                    same_proj = perm_cat[i] == perm_cat[j]
                    same_cc = perm_cc[i] == perm_cc[j]
                    if same_proj and not same_cc:
                        pa.append(d)
                    elif not same_proj and not same_cc:
                        pb.append(d)
            if pa and pb:
                if (np.mean(pb) - np.mean(pa)) >= observed:
                    count_ge += 1
        return float((count_ge + 1) / (n_perm + 1))

    test_project  = mw(within_project,  between, "project < between")
    test_compiler = mw(within_compiler, between, "compiler < between")
    # Is within-project significantly closer than within-compiler?
    test_proj_vs_cc = mw(within_project, within_compiler, "project < compiler")

    # Permutation p-values (accounts for non-independence)
    perm_p_project = perm_test(within_project, between)
    perm_p_compiler = perm_test(within_compiler, between)
    if perm_p_project is not None:
        test_project["perm_p_value"] = perm_p_project
        test_project["significant"] = bool(perm_p_project < 0.05)
    if perm_p_compiler is not None:
        test_compiler["perm_p_value"] = perm_p_compiler
        test_compiler["significant"] = bool(perm_p_compiler < 0.05)

    # NCD matrix with rows/cols sorted for readability
    sorted_names  = sorted(b.name for b in binaries)
    name_to_idx   = {b.name: i for i, b in enumerate(binaries)}

    proj_sig = test_project.get("significant")
    cc_sig = test_compiler.get("significant")
    pvcc_sig = test_proj_vs_cc.get("significant")
    if proj_sig and cc_sig and not pvcc_sig:
        interpretation = "Both axes significant; no clear dominance between project and compiler"
    elif proj_sig and cc_sig and pvcc_sig:
        interpretation = "Both axes significant; project marginally tighter than compiler"
    elif proj_sig and not cc_sig:
        interpretation = "Project dominates"
    elif cc_sig and not proj_sig:
        interpretation = "Compiler dominates"
    else:
        interpretation = "Mixed / inconclusive"

    logger.info(
        f"[project-vs-compiler] "
        f"within-project mean={np.mean(within_project):.3f}  "
        f"within-compiler mean={np.mean(within_compiler):.3f}  "
        f"between mean={np.mean(between):.3f}  "
        f"→ {interpretation}"
    )

    return {
        "n_within_project":   len(within_project),
        "n_within_compiler":  len(within_compiler),
        "n_between":          len(between),
        "mean_within_project":  float(np.mean(within_project))  if within_project  else None,
        "mean_within_compiler": float(np.mean(within_compiler)) if within_compiler else None,
        "mean_between":         float(np.mean(between))         if between         else None,
        "test_project_vs_between":  test_project,
        "test_compiler_vs_between": test_compiler,
        "test_project_vs_compiler": test_proj_vs_cc,
        "interpretation":           interpretation,
        "ncd_matrix":   ncd_matrix.tolist(),
        "binary_names": [b.name for b in binaries],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_compiler_matrix_analysis(binaries: List[Binary], results_dir: Path) -> Dict:
    """Run both Q1 and Q2 analyses and save results."""
    results_dir = Path(results_dir)
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Compiler matrix analysis: Q1 (opt-level trends) ===")
    opt_trends = analyze_opt_trends(binaries)
    save_json(opt_trends, results_dir / "compiler_opt_trends.json")

    logger.info("=== Compiler matrix analysis: Q2 (project vs compiler) ===")
    proj_vs_cc = analyze_project_vs_compiler(binaries)
    save_json(proj_vs_cc, results_dir / "compiler_project_vs_compiler.json")

    # ── summary printout ──────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("COMPILER MATRIX RESULTS")
    print("═"*60)

    print("\n── d95 by optimisation level (trigram PCA) ──")
    for opt in OPT_LEVELS:
        d = opt_trends["d95_by_opt_level"].get(opt, "?")
        mean_r = opt_trends["by_opt_level"].get(opt, {}).get("mean_zlib_ratio", float("nan"))
        mean_i = opt_trends["by_opt_level"].get(opt, {}).get("mean_insns", float("nan"))
        print(f"  {opt}: d95={d:3}  mean_zlib={mean_r:.3f}  mean_insns={mean_i:.0f}")
    print(f"  Full corpus d95 = {opt_trends['d95_full_corpus']}")

    print("\n── Per-project compression ratio by opt level ──")
    print(f"  {'project':12s}", end="")
    for opt in OPT_LEVELS:
        print(f"  {opt:>8s}", end="")
    print()
    for proj, by_opt in sorted(opt_trends["project_opt_compression"].items()):
        print(f"  {proj:12s}", end="")
        for opt in OPT_LEVELS:
            print(f"  {by_opt.get(opt, float('nan')):8.3f}", end="")
        print()

    print("\n── Project vs compiler clustering (lzma NCD) ──")
    pv = proj_vs_cc
    print(f"  within-project  mean NCD = {pv['mean_within_project']:.4f}  "
          f"(n={pv['n_within_project']})")
    print(f"  within-compiler mean NCD = {pv['mean_within_compiler']:.4f}  "
          f"(n={pv['n_within_compiler']})")
    print(f"  between         mean NCD = {pv['mean_between']:.4f}  "
          f"(n={pv['n_between']})")
    tp = pv["test_project_vs_between"]
    tc = pv["test_compiler_vs_between"]
    tpc = pv["test_project_vs_compiler"]
    print(f"  H1(project<between):  p={tp.get('p_value','?'):.4f}  "
          f"r={tp.get('effect_r','?'):.3f}  sig={tp.get('significant','?')}")
    print(f"  H1(compiler<between): p={tc.get('p_value','?'):.4f}  "
          f"r={tc.get('effect_r','?'):.3f}  sig={tc.get('significant','?')}")
    print(f"  H1(project<compiler): p={tpc.get('p_value','?'):.4f}  "
          f"r={tpc.get('effect_r','?'):.3f}  sig={tpc.get('significant','?')}")
    print(f"  → {pv['interpretation']}")
    print("═"*60 + "\n")

    return {"opt_trends": opt_trends, "project_vs_compiler": proj_vs_cc}
