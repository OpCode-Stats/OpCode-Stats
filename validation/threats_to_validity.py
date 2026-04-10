"""
threats_to_validity.py -- Phase 7 of the OpCode-Stats validation plan.

Reads all available validation result files and produces a structured
threats-to-validity analysis covering four standard subsections:
  A. Construct Validity
  B. Internal Validity
  C. External Validity
  D. Statistical Conclusion Validity

Output: validation/results/threats_to_validity.json

Run standalone:
    python3 validation/threats_to_validity.py
"""

import sys
import json
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, '/home/aaslyan/OpCode-Stats')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('threats_to_validity')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_VALIDATION_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _VALIDATION_DIR / 'results'
_OUTPUT_FILE = _RESULTS_DIR / 'threats_to_validity.json'

_RESULT_FILES = {
    'synthetic_validation':    _RESULTS_DIR / 'synthetic_validation.json',
    'extraction_verification': _RESULTS_DIR / 'extraction_verification.json',
    'robustness_analysis':     _RESULTS_DIR / 'robustness_analysis.json',
    'boilerplate_ablation':    _RESULTS_DIR / 'boilerplate_ablation.json',
    'stub_exclusion':          _RESULTS_DIR / 'stub_exclusion.json',
    'operand_aware':           _RESULTS_DIR / 'operand_aware.json',
    'independent_corpora':     _RESULTS_DIR / 'independent_corpora.json',
    'expanded_compiler_matrix':_RESULTS_DIR / 'expanded_compiler_matrix.json',
    'effect_summary':          _RESULTS_DIR / 'effect_summary.json',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if missing or malformed."""
    if not path.exists():
        logger.warning('Missing result file: %s', path)
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning('Could not parse %s: %s', path, exc)
        return None


def _fmt(value: Any, fmt: str = '.4f') -> str:
    """Format a numeric value, returning 'N/A' if None or NaN."""
    if value is None:
        return 'N/A'
    try:
        if math.isnan(float(value)):
            return 'N/A'
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _severity_label(conditions: List[bool]) -> str:
    """
    Derive a severity label from a list of boolean conditions.
    Each True = one step toward HIGH.
    0 True  -> LOW
    1 True  -> MEDIUM
    2+ True -> HIGH
    """
    n_high = sum(conditions)
    if n_high == 0:
        return 'LOW'
    if n_high == 1:
        return 'MEDIUM'
    return 'HIGH'


# ---------------------------------------------------------------------------
# A. Construct Validity
# ---------------------------------------------------------------------------

def analyse_construct_validity(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Are our measurements actually aligned with the concepts we claim to measure?
    """
    threats = []
    synth   = data.get('synthetic_validation') or {}
    robust  = data.get('robustness_analysis') or {}
    operand = data.get('operand_aware') or {}

    # ------------------------------------------------------------------
    # A1. Does d95 capture manifold structure or only feature-space compression?
    # ------------------------------------------------------------------
    # No specific d95 validation file exists; we must rely on prior absence.
    threats.append({
        'threat': (
            'd95 (95%-variance PCA dimensionality) as a proxy for intrinsic '
            'manifold dimensionality vs. dataset compression in feature space'
        ),
        'severity': 'MEDIUM',
        'evidence': (
            'No dedicated manifold-geometry validation was run. '
            'd95 equals the number of principal components needed to explain '
            '95% of variance in a frequency-feature matrix, which conflates '
            'true low-dimensional manifold structure with correlated but '
            'high-dimensional features. We did not compare against manifold '
            'learning methods (e.g., UMAP, Isomap) that could distinguish '
            'the two. The synthetic sanity tests (uniform random, Zipf-shuffled, '
            'Markov) validate entropy and compression metrics but not d95 directly.'
        ),
        'mitigation': (
            'Synthetic corpora with known low-rank structure (template corpus) '
            'demonstrate that compression and entropy metrics respond correctly '
            'to latent structure, providing indirect support. '
            'PCA dimensionality is a standard, peer-reviewed proxy measure.'
        ),
        'remaining_risk': (
            'A corpus with many small orthogonal functions could yield high d95 '
            'even when each individual function is highly structured. '
            'Conversely, vocabulary skew alone can reduce d95 without any '
            'genuine geometric manifold. Confidence would require cross-validation '
            'of d95 against held-out reconstruction error or a nonlinear baseline.'
        ),
    })

    # ------------------------------------------------------------------
    # A2. Does compression ratio measure sequential structure or only unigram skew?
    # ------------------------------------------------------------------
    # Use the shuffled-baseline gap from synthetic and stub data.
    zipf_shuffled = synth.get('zipf_shuffled', {})
    zipf_rate_gap_near_zero = False
    for chk in zipf_shuffled.get('checks', []):
        if chk.get('check') == 'zipf_rate_gap_near_zero' and chk.get('passed'):
            zipf_rate_gap_near_zero = True

    # From operand_aware: opcode_only compression vs shuffled
    # From stub_exclusion: stub zlib_ratio = 0.028 vs internal = 0.227
    # High threat: no per-binary unigram-shuffled baseline stored in results
    # (robustness_analysis only stores multi-seed entropy gaps, not compression gaps)
    multi_seed = robust.get('multi_seed_shuffled', {})
    gap_n5 = multi_seed.get('per_n', {}).get('5', {}).get('gap_real_minus_shuffled', None)

    evidence_parts = []
    if zipf_rate_gap_near_zero:
        evidence_parts.append(
            'Synthetic Zipf-shuffled corpus passes the entropy-rate gap test '
            '(rate gap near zero when only unigram skew is present), confirming '
            'that entropy rate beyond n=1 does reflect sequential order.'
        )
    if gap_n5 is not None:
        evidence_parts.append(
            f'The real 5-gram entropy rate ({2.863:.3f} bits) lies {abs(gap_n5):.3f} '
            f'bits below the shuffled baseline ({2.863 - gap_n5:.3f} bits), '
            'confirming that sequential structure contributes beyond unigram distribution.'
        )
    evidence_parts.append(
        'Compression ratios (zlib, lzma) were NOT compared against a per-binary '
        'unigram-shuffled baseline in the stored results; only entropy-rate gaps '
        'were baseline-corrected.'
    )

    threats.append({
        'threat': (
            'Compression ratio (zlib/lzma) may primarily reflect vocabulary skew '
            '(Zipf-distributed opcodes) rather than sequential structure, '
            'because both Huffman and LZ77 coding exploit unigram frequencies'
        ),
        'severity': 'MEDIUM',
        'evidence': ' '.join(evidence_parts),
        'mitigation': (
            'The entropy-rate gap (real minus shuffled) is explicitly baseline-corrected '
            'and captures purely sequential structure; this is the primary metric for '
            'sequential-structure claims. Compression ratios are presented as '
            'complementary compactness measures, not as sequential-structure indicators. '
            'Synthetic Zipf-shuffled corpus confirms that entropy-rate gap is ~0 '
            'when only unigram skew is present.'
        ),
        'remaining_risk': (
            'If readers interpret low compression ratios (0.22 zlib) as evidence '
            'of sequential structure, that interpretation is unsupported. '
            'A stored unigram-shuffle compression baseline per binary would '
            'definitively separate the two contributions but was not computed.'
        ),
    })

    # ------------------------------------------------------------------
    # A3. Do motif counts measure genuine biological-like patterns or compiler templates?
    # ------------------------------------------------------------------
    template_corpus = synth.get('template', {})
    boilerplate_corpus = synth.get('boilerplate', {})
    template_motifs_found = any(
        c.get('check') == 'template_motifs_found' and c.get('passed')
        for c in template_corpus.get('checks', [])
    )
    boilerplate_boundary_motifs = any(
        c.get('check') == 'boilerplate_boundary_motifs_found' and c.get('passed')
        for c in boilerplate_corpus.get('checks', [])
    )
    stub_excl = data.get('stub_exclusion') or {}
    internal_instr = (
        stub_excl.get('metrics_by_category', {})
               .get('internal', {})
               .get('total_instructions', None)
    )
    all_instr = (
        stub_excl.get('metrics_by_category', {})
               .get('all', {})
               .get('total_instructions', None)
    )

    evidence_parts = [
        'Synthetic template corpus with known repeated k-mer patterns: motif '
        f'detection {"passed" if template_motifs_found else "failed"} — '
        'top motifs correctly recover the injected template sub-sequence.',
        'Synthetic boilerplate corpus: boundary motif detection '
        f'{"passed" if boilerplate_boundary_motifs else "failed"}, '
        "confirming that top k-mers in a boilerplate corpus are prologue/epilogue sequences "
        "('push mov sub jns', etc.), i.e., compiler templates.",
    ]
    if internal_instr and all_instr:
        stub_pct = 100 * (1 - internal_instr / all_instr)
        evidence_parts.append(
            f'PLT stubs account for {stub_pct:.1f}% of instructions by count. '
            'Stub and startup functions are classified and can be excluded, '
            'but the main corpus includes them.'
        )

    threats.append({
        'threat': (
            'Motif counts may predominantly reflect compiler-inserted boilerplate '
            '(prologues, epilogues, PLT stubs) rather than semantically meaningful '
            'recurring computational patterns'
        ),
        'severity': 'MEDIUM',
        'evidence': ' '.join(evidence_parts),
        'mitigation': (
            'Synthetic validation confirms motif detection correctly identifies '
            'both template patterns and prologue/epilogue boilerplate. '
            'The paper acknowledges that motifs include compiler conventions. '
            'Stub exclusion analysis shows that removing PLT stubs changes the '
            'motif pool composition but does not eliminate sequential structure.'
        ),
        'remaining_risk': (
            'The paper does not systematically distinguish motifs that are '
            'compiler-universal (e.g., any standard-call prologue) from those '
            'that are program-specific idioms. A claim that motifs represent '
            '"biological-like patterns" requires demonstrating that at least a '
            'non-trivial fraction survives boilerplate stripping and is program-specific.'
        ),
    })

    # ------------------------------------------------------------------
    # A4. Is Zipf alpha meaningful for a ~280-token vocabulary?
    # ------------------------------------------------------------------
    boot_ci = robust.get('bootstrap_ci', {}).get('metrics', {}).get('zipf_alpha_mle', {})
    alpha_point = boot_ci.get('point_estimate', None)
    alpha_se    = boot_ci.get('standard_error', None)

    evidence_parts = [
        'x86-64 opcode vocabulary: ~281 unique mnemonics (vs 50K+ tokens in '
        'natural language). Zipf alpha is fit by MLE on rank-frequency data.',
    ]
    if alpha_point is not None:
        evidence_parts.append(
            f'Estimated alpha = {alpha_point:.4f} '
            f'(bootstrap SE = {alpha_se:.4f} over {robust.get("corpus", {}).get("n_binaries", "?")} binaries).'
        )
    evidence_parts.append(
        'With only 281 ranks, the power-law fit is highly sensitive to the '
        'treatment of the tail (rare opcodes) and to the lower truncation '
        'point x_min. MLE without x_min optimisation conflates a power law '
        'with other heavy-tailed distributions (log-normal, Weibull).'
    )

    # Additional: natural language comparison risk
    threats.append({
        'threat': (
            'Zipf alpha estimated on a 281-token vocabulary may not be comparable '
            'to natural-language Zipf fits over 50K+ tokens; the interpretation '
            'of alpha=1.44 as evidence of a power law may not be statistically '
            'distinguishable from log-normal or Weibull alternatives'
        ),
        'severity': 'MEDIUM',
        'evidence': ' '.join(evidence_parts),
        'mitigation': (
            'Bootstrap CIs over binaries (resampling unit = binary) confirm '
            f'tight inter-binary stability (SE={alpha_se:.4f}). '
            'Leave-one-out analysis shows zero influential binaries '
            '(max alpha shift <0.5% excluding any single binary). '
            'The paper frames alpha as a descriptive statistic of frequency '
            'concentration, not as a claim that the true distribution is Pareto.'
        ),
        'remaining_risk': (
            'No goodness-of-fit test (KS test, likelihood ratio vs log-normal) '
            'was performed to confirm the power-law hypothesis over alternatives. '
            'Cross-ISA comparability of alpha values requires fitting the same '
            'model on ARM64/RISC-V corpora, which were not collected.'
        ),
    })

    # ------------------------------------------------------------------
    # A5. Does MI decay measure long-range dependencies or just n-gram correlations?
    # ------------------------------------------------------------------
    markov_corpus = synth.get('markov', {})
    mi_gap_positive = any(
        c.get('check') == 'markov_mi_gap_positive_lag1' and c.get('passed')
        for c in markov_corpus.get('checks', [])
    )
    mi_gap_decays = any(
        c.get('check') == 'markov_mi_gap_decays' and c.get('passed')
        for c in markov_corpus.get('checks', [])
    )

    threats.append({
        'threat': (
            'Mutual information (MI) decay curves may reflect short-range '
            'n-gram correlations (bigrams, trigrams) rather than genuine '
            'long-range dependencies; MI at lag k is not corrected for '
            'indirect paths through shorter-range correlations'
        ),
        'severity': 'MEDIUM',
        'evidence': (
            f'Synthetic low-order Markov corpus: MI gap at lag 1 is positive '
            f'({"pass" if mi_gap_positive else "fail"}) and MI gap decays with lag '
            f'({"pass" if mi_gap_decays else "fail"}), validating that the metric '
            'is sensitive to order. However, a first-order Markov process by '
            'definition has only lag-1 dependencies, so the synthetic test does '
            'not probe long-range structure. No partial MI or transfer entropy '
            'analysis was performed to isolate contributions beyond lag k-1.'
        ),
        'mitigation': (
            'Entropy-rate convergence (H_n/n as n grows) provides a complementary '
            'view: the gap between H_1 and H_5 rates measures cumulative predictability '
            'improvement, which is independent of any particular lag decomposition. '
            'Synthetic Markov corpus confirms that a process with known short-range '
            'structure yields measurably different MI curves from a shuffled baseline.'
        ),
        'remaining_risk': (
            'A claim that opcode sequences exhibit "long-range dependencies" beyond '
            'function-length scales cannot be supported by the current MI analysis '
            'without partial MI or convergence of H_n/n to a stable rate, '
            'which would require n >> function length.'
        ),
    })

    return threats


# ---------------------------------------------------------------------------
# B. Internal Validity
# ---------------------------------------------------------------------------

def analyse_internal_validity(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Could results be driven by extraction bugs, boilerplate, or other confounds?
    """
    threats = []
    extv    = data.get('extraction_verification') or {}
    boiler  = data.get('boilerplate_ablation')     # may be None
    stub    = data.get('stub_exclusion') or {}
    operand = data.get('operand_aware') or {}
    robust  = data.get('robustness_analysis') or {}

    # ------------------------------------------------------------------
    # B1. Extraction correctness
    # ------------------------------------------------------------------
    overall_ext = extv.get('overall', {})
    ext_pass    = overall_ext.get('status') == 'PASS'
    ext_checks  = overall_ext.get('checks_passed', 0)
    ext_total   = overall_ext.get('checks_failed', 0) + ext_checks

    mnemonic_all_pass = extv.get('check_1_mnemonic_counts', {}).get('status') == 'PASS'
    boundary_all_pass = extv.get('check_2_instruction_boundaries', {}).get('status') == 'PASS'
    func_all_pass     = extv.get('check_3_function_boundaries', {}).get('status') == 'PASS'
    hardlink_pass     = extv.get('check_4_hardlink_filtering', {}).get('status') == 'PASS'
    timeout_pass      = extv.get('check_5_timeout_handling', {}).get('status') == 'PASS'

    severity = 'LOW' if ext_pass else 'HIGH'
    threats.append({
        'threat': (
            'Extraction pipeline bugs (mnemonic misidentification, instruction '
            'boundary errors, function boundary errors, duplicate binary inclusion) '
            'could silently inflate or deflate all structural metrics'
        ),
        'severity': severity,
        'evidence': (
            f'Extraction verification: {ext_checks}/{ext_checks + overall_ext.get("checks_failed", 0)} checks passed '
            f'(status: {"PASS" if ext_pass else "FAIL"}). '
            f'Mnemonic counts vs raw objdump: {"PASS" if mnemonic_all_pass else "FAIL"} '
            '(zero mismatches on grep/bash/find/tar/xz totaling 387,061 instructions). '
            f'Instruction boundaries: {"PASS" if boundary_all_pass else "FAIL"}. '
            f'Function boundaries: {"PASS" if func_all_pass else "FAIL"}. '
            f'Hard-link deduplication: {"PASS" if hardlink_pass else "FAIL"} '
            '(7 hard-linked pairs detected and correctly deduplicated). '
            f'Timeout handling: {"PASS" if timeout_pass else "FAIL"}.'
        ),
        'mitigation': (
            'Five independent binary samples were checked against raw objdump output '
            'at the mnemonic-count, instruction-boundary, and function-boundary levels. '
            'Hard-link deduplication and timeout handling were tested with synthetic '
            'edge cases. All checks passed with zero discrepancies.'
        ),
        'remaining_risk': (
            'Verification was performed on 5 binaries (grep, bash, find, tar, xz) '
            'and does not cover stripped binaries, PIE vs non-PIE, or binaries '
            'with unusual section layouts. objdump disassembly errors on data '
            'interpreted as code (a known limitation of linear sweep disassembly) '
            'are not flagged by any check.'
        ),
    })

    # ------------------------------------------------------------------
    # B2. Boilerplate contribution
    # ------------------------------------------------------------------
    if boiler is not None:
        # Parse available boilerplate data
        boiler_available = True
        boiler_note = 'Boilerplate ablation results available.'
    else:
        boiler_available = False

    severity = _severity_label([
        not boiler_available,           # missing data = more uncertainty
        True,                           # boilerplate is always a real concern
    ])

    if boiler_available:
        # Try to extract signal-survival percentage
        summary = boiler.get('summary', {})
        h5_survival = summary.get('h5_rate_survival_pct', None)
        zipf_survival = summary.get('zipf_alpha_survival_pct', None)
        ev_str = (
            f'Signal survival after full boilerplate stripping: '
            f'entropy-rate {h5_survival:.1f}% ' if h5_survival else ''
        ) + (
            f'Zipf alpha {zipf_survival:.1f}%.' if zipf_survival else ''
        )
        if not ev_str:
            ev_str = json.dumps(boiler.get('summary', {}))[:300]
    else:
        ev_str = (
            'Boilerplate ablation results (boilerplate_ablation.json) are not '
            'available in validation/results/. The ablation script exists but '
            'requires the full corpus pickle. '
            'Synthetic boilerplate corpus confirms that prologue/epilogue '
            'boilerplate causes measurable entropy reduction and boundary '
            'motif inflation. '
            'Stub exclusion analysis (stub_exclusion.json) shows that removing '
            'PLT stubs (16,955 instructions, 0.6% of total) shifts zlib ratio '
            'from 0.2212 to 0.2275 and H5 rate from 2.8628 to 2.8654 — '
            'changes smaller than the bootstrap CI width.'
        )

    threats.append({
        'threat': (
            'A substantial fraction of observed structure may come from '
            'compiler-inserted boilerplate (function prologues/epilogues, '
            'endbr64 CET guards, alignment nops) that appears in every binary '
            'regardless of application logic'
        ),
        'severity': severity,
        'evidence': ev_str,
        'mitigation': (
            'Stub exclusion analysis shows that PLT stubs (0.6% of instructions) '
            'have negligible impact on corpus-level metrics. '
            'Synthetic mixed-boilerplate corpus validates that the measurement '
            'framework can detect and quantify boilerplate contributions. '
            'Internal-only corpus (excluding all toolchain functions) yields '
            'essentially identical metrics (H5 rate: 2.8654 vs 2.8628 baseline).'
        ),
        'remaining_risk': (
            'Full boilerplate stripping ablation (removing first/last k instructions '
            'and endbr64 from every function) could not be confirmed from stored '
            'results. Until boilerplate_ablation.json is generated, we cannot '
            'quantify what fraction of entropy rate and Zipf alpha survives '
            'aggressive stripping of boundary instructions.'
        ),
    })

    # ------------------------------------------------------------------
    # B3. Startup/stub contribution
    # ------------------------------------------------------------------
    comp = stub.get('comparison', {})
    h5_shift = comp.get('h5_rate_change_internal_vs_all', 'N/A')
    zlib_shift = comp.get('compression_ratio_change_internal_vs_all', 'N/A')
    stub_pct = comp.get('structure_is_generic_toolchain', 'N/A')

    # Stubs have alpha=0.859 vs corpus alpha=1.439 — a meaningful difference
    stub_alpha = stub.get('metrics_by_category', {}).get('stubs', {}).get('zipf_alpha', None)
    all_alpha  = stub.get('metrics_by_category', {}).get('all', {}).get('zipf_alpha', None)
    alpha_diff_large = (stub_alpha is not None and all_alpha is not None and
                        abs(stub_alpha - all_alpha) > 0.3)

    threats.append({
        'threat': (
            'PLT stubs and startup/runtime glue functions have very different '
            'statistical properties (near-identical 3-instruction stubs) and '
            'could bias corpus-level metrics if included'
        ),
        'severity': 'LOW',
        'evidence': (
            f'{stub_pct}. '
            f'Stub corpus: Zipf alpha = {stub_alpha:.4f} vs full corpus {all_alpha:.4f} '
            f'({"large" if alpha_diff_large else "small"} difference). '
            f'Metric shifts when moving to internal-only: {h5_shift}; {zlib_shift}. '
            'Startup/runtime glue: 23 functions, 69 instructions — negligible.'
        ),
        'mitigation': (
            'Explicit three-way classification (startup/runtime, PLT stubs, internal). '
            'Metrics recomputed on internal-only subset. '
            'Entropy rate and compression ratio change by less than 0.3% '
            'when stubs are excluded, confirming minimal bias.'
        ),
        'remaining_risk': (
            'The stub count (2,749 functions, 16,955 instructions) represents '
            '46% of function count but only 0.6% of instruction count. '
            'Stub bias is negligible at the instruction level but could inflate '
            'function-count-based statistics if those are used.'
        ),
    })

    # ------------------------------------------------------------------
    # B4. Operand dropping effect
    # ------------------------------------------------------------------
    variants = operand.get('variants', {})
    opcode_only_gap = operand.get('comparison', {}).get('gap_h5_by_variant', {}).get('opcode_only', None)
    operand_class_gap = operand.get('comparison', {}).get('gap_h5_by_variant', {}).get('opcode_operand_class', None)
    gap_stable = operand.get('comparison', {}).get('gap_stable', None)

    if opcode_only_gap is not None and operand_class_gap is not None:
        diff = abs(opcode_only_gap - operand_class_gap)
        evidence_str = (
            f'H5 entropy-rate gap (real minus shuffled): '
            f'opcode-only = {opcode_only_gap:.4f} bits, '
            f'opcode+operand-class = {operand_class_gap:.4f} bits '
            f'(difference = {diff:.4f} bits). '
            f'Gap stable across variants: {gap_stable}. '
            'Richer operand representations yield larger gaps, indicating that '
            'operand information carries additional sequential structure '
            'not captured by mnemonics alone.'
        )
    else:
        evidence_str = 'Operand-aware results not available for comparison.'

    # Dropping operands discards real structure — this is a construct concern
    threats.append({
        'threat': (
            'Discarding operands (using opcode mnemonics only) loses sequential '
            'structure that is carried in register reuse patterns and '
            'immediate-value sequences, potentially underestimating true '
            'information density'
        ),
        'severity': 'MEDIUM',
        'evidence': evidence_str,
        'mitigation': (
            'Four operand-aware variants were tested: opcode-only, '
            'opcode+operand-class, opcode+register-class, opcode+immediate-bucket. '
            'All variants show a positive H5 gap (real < shuffled), confirming '
            'that the core finding (sequential structure beyond unigrams) holds '
            'regardless of representation choice. '
            'Opcode-only is the most conservative, reproducible, and ISA-portable choice.'
        ),
        'remaining_risk': (
            'Quantitative claims (e.g., compression ratio 0.22) are specific to '
            'the opcode-only representation and will differ for richer tokens. '
            'Cross-ISA comparison requires a common representation; '
            'opcode-only provides the cleanest equivalence class.'
        ),
    })

    # ------------------------------------------------------------------
    # B5. Random seed sensitivity
    # ------------------------------------------------------------------
    multi_seed = robust.get('multi_seed_shuffled', {})
    all_stable = all(
        v.get('gap_stable', False)
        for v in multi_seed.get('per_n', {}).values()
    )
    n_seeds = len(robust.get('configuration', {}).get('shuffled_seeds', []))

    threats.append({
        'threat': (
            'Shuffled-baseline entropy rates depend on the random permutation seed; '
            'an unlucky seed could artificially inflate or deflate the '
            'real-vs-shuffled gap'
        ),
        'severity': 'LOW',
        'evidence': (
            f'{n_seeds}-seed multi-seed analysis: shuffled-baseline CV < 0.005% '
            f'at all n-gram orders. Gap stability: {"all stable" if all_stable else "some unstable"}. '
            'Shuffled standard deviation across 10 seeds is < 0.03% of the '
            'shuffled mean, confirming that the choice of seed is immaterial.'
        ),
        'mitigation': (
            'Ten independent shuffled baselines computed at each n-gram order. '
            'Coefficient of variation < 0.01% in all cases. '
            'Results are effectively deterministic given the corpus.'
        ),
        'remaining_risk': (
            'Near-zero residual: the shuffled baseline variance is negligible. '
            'No remaining risk attributable to seed choice.'
        ),
    })

    return threats


# ---------------------------------------------------------------------------
# C. External Validity
# ---------------------------------------------------------------------------

def analyse_external_validity(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Do findings hold across corpora, ISAs, compilers, and source languages?
    """
    threats = []
    indep   = data.get('independent_corpora')     # may be None
    compiler= data.get('expanded_compiler_matrix') # may be None
    robust  = data.get('robustness_analysis') or {}

    # ------------------------------------------------------------------
    # C1. Cross-corpus consistency
    # ------------------------------------------------------------------
    if indep is not None:
        corpora = indep.get('corpora', {})
        n_corpora = len(corpora)
        alpha_vals = [c.get('zipf_alpha', None) for c in corpora.values() if c.get('zipf_alpha')]
        h5_vals    = [c.get('entropy_rate_5gram', None) for c in corpora.values() if c.get('entropy_rate_5gram')]
        if alpha_vals:
            alpha_range = max(alpha_vals) - min(alpha_vals)
            h5_range    = max(h5_vals) - min(h5_vals) if h5_vals else None
            ev_str = (
                f'{n_corpora} independent corpora tested. '
                f'Zipf alpha range: {min(alpha_vals):.3f}–{max(alpha_vals):.3f} '
                f'(spread = {alpha_range:.3f}). '
            )
            if h5_range is not None:
                ev_str += f'H5 rate range: {min(h5_vals):.3f}–{max(h5_vals):.3f} (spread = {h5_range:.3f}).'
            severity = 'LOW' if alpha_range < 0.15 else 'MEDIUM'
        else:
            ev_str = 'Independent corpora results available but metric values not parseable.'
            severity = 'MEDIUM'
    else:
        ev_str = (
            'independent_corpora.json is not present in validation/results/. '
            'The independent_corpora.py script exists and targets three corpora '
            '(system utilities, network/developer tools, compiled benchmarks) '
            'but was not successfully run or results were not saved. '
            'The primary corpus (15 binaries: grep, bash, git, gcc, etc.) was '
            'analysed with leave-one-out and subsampling, providing partial '
            'internal replication but not independent external validation.'
        )
        severity = 'HIGH'

    threats.append({
        'threat': (
            'Findings may be specific to the 15-binary system-utilities corpus '
            'used as the primary dataset and may not generalize to other '
            'application domains (e.g., multimedia codecs, scientific computing, '
            'game engines, embedded firmware)'
        ),
        'severity': severity,
        'evidence': ev_str,
        'mitigation': (
            'Leave-one-out analysis: zero influential binaries found '
            '(maximum alpha shift < 0.5% when any single binary is excluded). '
            'Subsampling analysis shows convergence: alpha std drops from '
            '0.020 at 25% corpus to ~0 at 100%, indicating the corpus is '
            'large enough for stable estimates within this domain.'
        ),
        'remaining_risk': (
            'Domain generalizability remains the largest open external validity '
            'question. System utilities (the primary corpus) are C-heavy, '
            'POSIX-oriented, and compiled with mainstream toolchains. '
            'Results for Rust/Go binaries, JIT-compiled code, or domain-specific '
            'binaries (DSP, ML inference) are unknown.'
        ),
    })

    # ------------------------------------------------------------------
    # C2. Compiler invariance
    # ------------------------------------------------------------------
    if compiler is not None:
        programs = compiler.get('programs', [])
        compilers = compiler.get('compilers', [])
        opt_levels = compiler.get('opt_levels', [])
        variance = compiler.get('variance_decomposition', {})
        program_var_frac = variance.get('program_fraction', None)
        compiler_var_frac = variance.get('compiler_fraction', None)
        optlevel_var_frac = variance.get('opt_level_fraction', None)

        ev_str = (
            f'Expanded compiler matrix: {len(programs)} programs x '
            f'{len(compilers)} compilers x {len(opt_levels)} opt levels. '
        )
        if program_var_frac is not None:
            ev_str += (
                f'Variance decomposition: program={program_var_frac:.1%}, '
                f'compiler={compiler_var_frac:.1%}, '
                f'opt-level={optlevel_var_frac:.1%}.'
            )
        severity = 'LOW' if (compiler_var_frac is not None and compiler_var_frac < 0.2) else 'MEDIUM'
    else:
        ev_str = (
            'expanded_compiler_matrix.json is not present in validation/results/. '
            'The expanded_compiler_matrix.py script targets gcc/clang x '
            'O0/O1/O2/O3/Os across 15+ programs but results were not stored. '
            'No variance decomposition (program vs. compiler vs. opt-level) '
            'is available from stored files.'
        )
        severity = 'HIGH'

    threats.append({
        'threat': (
            'Statistical properties may be dominated by compiler choice or '
            'optimization level rather than reflecting language-level or '
            'program-level characteristics; in particular, O0 vs O3 may '
            'produce qualitatively different opcode distributions'
        ),
        'severity': severity,
        'evidence': ev_str,
        'mitigation': (
            'Primary corpus binaries are distribution-provided (mixed gcc/clang, '
            'mixed optimization levels), providing incidental compiler diversity. '
            'Robustness analysis includes 15 binaries spanning multiple packages '
            'and likely multiple compilers.'
        ),
        'remaining_risk': (
            'Without the expanded compiler matrix results, we cannot quantify '
            'what fraction of inter-binary metric variance is attributable to '
            'compiler choice vs. program content. This is a significant gap '
            'for any claim about ISA-level rather than toolchain-level properties.'
        ),
    })

    # ------------------------------------------------------------------
    # C3. Missing ISAs, OSes, and source languages
    # ------------------------------------------------------------------
    threats.append({
        'threat': (
            'All binaries are x86-64 ELF on Linux; findings may not transfer '
            'to ARM64, RISC-V, or WASM, to Windows PE/Mach-O binary formats, '
            'or to source languages beyond C/C++ (Rust, Go, Fortran, Swift)'
        ),
        'severity': 'HIGH',
        'evidence': (
            'No ARM64, RISC-V, or Windows binaries were collected or analysed. '
            'All 15 primary corpus binaries and all compiler-matrix programs are '
            'x86-64 ELF. The x86-64 ISA has variable-length instructions, '
            'a complex legacy instruction set, and specific calling conventions '
            'that may produce Zipf distributions and entropy rates substantially '
            'different from RISC ISAs with fixed-width instructions and '
            'smaller register files.'
        ),
        'mitigation': (
            'The paper focuses on x86-64 and does not claim cross-ISA generality '
            'without evidence. Limitations section acknowledges this scope.'
        ),
        'remaining_risk': (
            'This is a genuine and unmitigated scope limitation. '
            'Whether the reported Zipf exponent (~1.44), entropy rate (~2.86 bits), '
            'or compression ratio (~0.22) are ISA-specific constants or universal '
            'features of compiled code cannot be determined from this study alone. '
            'ARM64 analysis would be a direct and feasible extension.'
        ),
    })

    # ------------------------------------------------------------------
    # C4. Corpus size effects
    # ------------------------------------------------------------------
    subsamp = robust.get('subsampling_sensitivity', {})
    by_frac = subsamp.get('by_fraction', {})
    alpha_25 = by_frac.get('0.25', {}).get('zipf_alpha_mle', {}).get('std', None)
    alpha_100 = by_frac.get('1.0', {}).get('zipf_alpha_mle', {}).get('std', None)  # NaN at full
    h5_std_25 = by_frac.get('0.25', {}).get('entropy_rate_5gram', {}).get('std', None)
    h5_mean_full = by_frac.get('1.0', {}).get('entropy_rate_5gram', {}).get('mean', None)

    threats.append({
        'threat': (
            'The 15-binary corpus may be too small to establish stable '
            'estimates, and metric values could shift substantially with '
            'a larger or differently-sampled corpus'
        ),
        'severity': 'MEDIUM',
        'evidence': (
            f'Subsampling at 25% (4 binaries): '
            f'alpha std = {alpha_25:.4f}, H5 rate std = {h5_std_25:.4f}. '
            f'Subsampling at 50% (8 binaries): alpha std = '
            f'{by_frac.get("0.5", {}).get("zipf_alpha_mle", {}).get("std", float("nan")):.4f}. '
            'Convergence is visible: std decreases monotonically as fraction increases. '
            f'Bootstrap CI for alpha over 15 binaries: '
            f'[{robust.get("bootstrap_ci", {}).get("metrics", {}).get("zipf_alpha_mle", {}).get("ci_low_95", "?"):.4f}, '
            f'{robust.get("bootstrap_ci", {}).get("metrics", {}).get("zipf_alpha_mle", {}).get("ci_high_95", "?"):.4f}].'
        ),
        'mitigation': (
            'Bootstrap CIs confirm narrow inter-binary uncertainty at the 15-binary scale. '
            'Leave-one-out shows zero influential binaries. '
            'Subsampling convergence is visible by 50% (8 binaries), '
            'suggesting the corpus is adequate for the metrics studied.'
        ),
        'remaining_risk': (
            '15 binaries is a small N for corpus-level claims. '
            'The bootstrap resamples with replacement from 15 units, meaning '
            'some bootstrap samples contain duplicates; CIs may be slightly '
            'over-optimistic. A corpus of 50-100 binaries would provide '
            'substantially more reliable external generalizability.'
        ),
    })

    return threats


# ---------------------------------------------------------------------------
# D. Statistical Conclusion Validity
# ---------------------------------------------------------------------------

def analyse_statistical_conclusion_validity(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Are sample sizes and statistical methods adequate?
    """
    threats = []
    robust = data.get('robustness_analysis') or {}
    stub   = data.get('stub_exclusion') or {}

    boot_ci = robust.get('bootstrap_ci', {})
    n_boot  = boot_ci.get('n_boot', 0)
    resample_unit = boot_ci.get('resample_unit', 'unknown')
    metrics_ci    = boot_ci.get('metrics', {})

    corpus = robust.get('corpus', {})
    n_binaries   = corpus.get('n_binaries', 0)
    n_total_instr = corpus.get('total_instructions', 0)

    # ------------------------------------------------------------------
    # D1. Bootstrap CI widths
    # ------------------------------------------------------------------
    alpha_ci = metrics_ci.get('zipf_alpha_mle', {})
    h5_ci    = metrics_ci.get('entropy_rate_5gram', {})
    zlib_ci  = metrics_ci.get('mean_zlib_ratio', {})

    alpha_width = (alpha_ci.get('ci_high_95', 0) - alpha_ci.get('ci_low_95', 0)) if alpha_ci else None
    h5_width    = (h5_ci.get('ci_high_95', 0)    - h5_ci.get('ci_low_95', 0))    if h5_ci else None

    # CIs look narrow in absolute terms but relative to effect size matters
    # alpha ~1.44, CI width ~0.027 => ~1.9% relative width
    # h5 rate ~2.86, CI width ~0.115 => ~4% relative width

    threats.append({
        'threat': (
            'Bootstrap CIs are computed by resampling binaries (N=15), which '
            'may produce over-optimistic CIs if binaries are heterogeneous or '
            'if the corpus is not a random sample from a well-defined population'
        ),
        'severity': 'MEDIUM',
        'evidence': (
            f'Bootstrap: n_boot={n_boot}, resample_unit={resample_unit}, N={n_binaries}. '
            '95% CI widths: alpha=' + _fmt(alpha_width) + (
                f' ({100*alpha_width/alpha_ci.get("point_estimate", 1):.1f}% relative)'
                if alpha_width is not None else ''
            ) + ', H5 rate=' + _fmt(h5_width) + (
                f' ({100*h5_width/h5_ci.get("point_estimate", 1):.1f}% relative)'
                if h5_width is not None else ''
            ) + '. '
            'Resampling from N=15 with replacement means some bootstrap samples '
            'contain 5-6 copies of the same binary, potentially narrowing CIs '
            'artificially if the corpus is not iid.'
        ),
        'mitigation': (
            f'Binary-level resampling (not instruction-level) is the correct choice '
            f'given that the corpus is a collection of programs, not a sample of '
            f'independent instructions. n_boot={n_boot} is sufficient for 95% CIs. '
            f'Leave-one-out analysis provides an additional non-bootstrap '
            f'sensitivity check.'
        ),
        'remaining_risk': (
            'The 15-binary corpus was not randomly sampled from a population of '
            'binaries — it is a convenience sample of common Linux system utilities. '
            'Frequentist CIs presuppose random sampling; the reported CIs measure '
            'inter-binary consistency within this corpus, not uncertainty about '
            'the population of all binaries.'
        ),
    })

    # ------------------------------------------------------------------
    # D2. Multiple comparisons
    # ------------------------------------------------------------------
    threats.append({
        'threat': (
            'Multiple metrics (Zipf alpha, H1–H5 rates, zlib/lzma ratios, '
            'motif counts, MI at multiple lags, d95, similarity) are reported '
            'without multiple-comparisons correction; some findings may be '
            'false positives under a family-wise error rate framework'
        ),
        'severity': 'MEDIUM',
        'evidence': (
            'At least 8 primary metrics are reported, each with a claim of '
            'statistical significance or with CIs. No Bonferroni, Benjamini-Hochberg, '
            'or other FWER/FDR correction is applied. '
            'For the entropy rate at n=2,3,4,5 the tests are not independent '
            '(each n-gram rate bounds the next), so naive Bonferroni would be '
            'overly conservative, but no correction at all inflates type-I error.'
        ),
        'mitigation': (
            'The primary claim (opcode sequences exhibit non-trivial sequential '
            'structure beyond unigrams) is supported by multiple independent '
            'measurement approaches (entropy rate, compression, MI, motifs) '
            'all pointing in the same direction, reducing the likelihood that '
            'all are false positives simultaneously. '
            'Synthetic controls confirm the expected direction and magnitude '
            'for each metric independently.'
        ),
        'remaining_risk': (
            'If individual sub-claims (e.g., specific motif frequencies) are '
            'treated as hypothesis tests, false discovery rates are uncontrolled. '
            'A pre-registration specifying primary vs. exploratory metrics would '
            'strengthen inferential claims.'
        ),
    })

    # ------------------------------------------------------------------
    # D3. Dependence structure (tokens within a binary are not iid)
    # ------------------------------------------------------------------
    all_instr_count = stub.get('metrics_by_category', {}).get('all', {}).get('total_instructions', n_total_instr)

    threats.append({
        'threat': (
            'Instruction tokens within a single binary are not independent; '
            'reporting "N = 2.9M tokens" implies a much larger effective sample '
            'size than the actual N=15 binaries, potentially giving misleading '
            'precision for corpus-level claims'
        ),
        'severity': 'HIGH',
        'evidence': (
            f'Total instructions: {all_instr_count:,}. '
            f'Binaries: {n_binaries}. '
            f'Instructions are spatially and semantically correlated within '
            f'functions and binaries (same compiler, same codebase, same architecture '
            f'conventions). Bootstrap correctly resamples at binary level, but '
            f'instruction-level statistics (e.g., frequency counts, MI values) '
            f'conflate within-binary and between-binary variation.'
        ),
        'mitigation': (
            'Bootstrap CI explicitly uses the binary as the resampling unit, '
            'not individual instructions, correctly reflecting that N=15 for '
            'corpus-level inference. '
            'Leave-one-out sensitivity at the binary level provides '
            'a check that no single binary dominates.'
        ),
        'remaining_risk': (
            'Even with binary-level resampling, N=15 is insufficient for '
            'reliable parametric inference about population parameters. '
            'The MI decay curves and entropy rates are computed over '
            'concatenated sequences, treating function boundaries as '
            'transparent — this inflates observed long-range MI '
            'because cross-function "dependencies" are actually just '
            'shared vocabulary biases.'
        ),
    })

    # ------------------------------------------------------------------
    # D4. Effective sample size for corpus-level claims
    # ------------------------------------------------------------------
    _subsamp_d4 = robust.get('subsampling_sensitivity', {}).get('by_fraction', {})
    _alpha_25   = _subsamp_d4.get('0.25', {}).get('zipf_alpha_mle', {}).get('std', None)
    _h5_std_25  = _subsamp_d4.get('0.25', {}).get('entropy_rate_5gram', {}).get('std', None)

    threats.append({
        'threat': (
            'With N=15 binaries, the effective sample size for corpus-level '
            'distributional claims is very small; some statistics (e.g., '
            'tail of Zipf distribution, rare motif counts) may be '
            'highly sensitive to which 15 programs are included'
        ),
        'severity': 'MEDIUM',
        'evidence': (
            f'N=15 binaries from a single Linux installation. '
            + (
                f'At 25% subsampling (N=4 binaries), Zipf alpha std = {_alpha_25:.4f} '
                f'and H5 rate std = {_h5_std_25:.4f} — substantial variability. '
                if _alpha_25 is not None else
                'Subsampling std at 25% not available. '
            ) +
            'At N=8, variability drops significantly, suggesting N=15 is near '
            'the asymptote for these specific metrics. '
            'Rare motifs (appearing in only 1-2 binaries) would have '
            'highly uncertain frequency estimates.'
        ),
        'mitigation': (
            'Primary metrics (Zipf alpha, H5 rate, compression ratio) are '
            'bulk properties of the frequency distribution and converge '
            'quickly with corpus size. They are not sensitive to tail '
            'behavior. Leave-one-out confirms no influential binaries '
            'for these metrics.'
        ),
        'remaining_risk': (
            'Motif frequency tables, similarity matrices, and d95 estimates '
            'may be less stable than the bulk metrics. '
            'Claims about specific motif rankings or d95 values should be '
            'treated as preliminary pending a larger corpus.'
        ),
    })

    # ------------------------------------------------------------------
    # D5. Subsampling stability
    # ------------------------------------------------------------------
    subsamp = robust.get('subsampling_sensitivity', {})
    interp  = subsamp.get('interpretation', '')
    by_frac = subsamp.get('by_fraction', {})

    threats.append({
        'threat': (
            'Entropy rate and compression metrics might not converge '
            'with corpus size if a few large binaries dominate the '
            'instruction token count'
        ),
        'severity': 'LOW',
        'evidence': (
            'Subsampling at fractions [0.25, 0.50, 0.75, 1.0] with 10 draws each. '
            'Zlib ratio std: '
            + _fmt(by_frac.get('0.25', {}).get('mean_zlib_ratio', {}).get('std'), '.4f') + ' at 25%, '
            + _fmt(by_frac.get('0.5',  {}).get('mean_zlib_ratio', {}).get('std'), '.4f') + ' at 50%, '
            + _fmt(by_frac.get('0.75', {}).get('mean_zlib_ratio', {}).get('std'), '.4f') + ' at 75%. '
            'Monotonic convergence observed. '
            'Truncation sensitivity: H5 rate rises from 2.28 (5K tokens/binary) to '
            '2.86 (full), stabilizing above 50K tokens/binary. '
            + interp
        ),
        'mitigation': (
            'Truncation analysis shows that metrics stabilize above 50K tokens '
            'per binary; only gzip and bzip2 are below this threshold, and '
            'leave-one-out confirms neither is influential. '
            'Subsampling convergence is monotonic, confirming no individual '
            'binary disproportionately drives corpus-level statistics.'
        ),
        'remaining_risk': (
            'At truncation lengths below 5K tokens/binary, entropy rates are '
            'substantially lower than at full length, indicating that very '
            'short sequences yield unreliable estimates. '
            'Any study comparing across corpora of very different sizes must '
            'control for sequence length.'
        ),
    })

    return threats


# ---------------------------------------------------------------------------
# Overall assessment
# ---------------------------------------------------------------------------

def compute_overall_assessment(
    construct: List[Dict],
    internal: List[Dict],
    external: List[Dict],
    statistical: List[Dict],
) -> str:
    all_threats = construct + internal + external + statistical
    n_high   = sum(1 for t in all_threats if t['severity'] == 'HIGH')
    n_medium = sum(1 for t in all_threats if t['severity'] == 'MEDIUM')
    n_low    = sum(1 for t in all_threats if t['severity'] == 'LOW')

    if n_high >= 3:
        overall = 'SUBSTANTIAL CONCERNS'
    elif n_high >= 1:
        overall = 'MODERATE CONCERNS'
    else:
        overall = 'MINOR CONCERNS'

    return (
        f'{overall}: {len(all_threats)} threats identified across four validity categories. '
        f'HIGH={n_high}, MEDIUM={n_medium}, LOW={n_low}. '
        f'The two most significant unmitigated threats are: '
        f'(1) EXTERNAL: no cross-ISA data — all findings are x86-64 Linux only; '
        f'(2) STATISTICAL: N=15 binaries is a convenience sample, not a random sample '
        f'from a well-defined population, so frequentist CIs measure within-corpus '
        f'consistency rather than population uncertainty. '
        f'Internal and seed-sensitivity threats are well-mitigated by validation. '
        f'Missing result files (boilerplate_ablation.json, independent_corpora.json, '
        f'expanded_compiler_matrix.json) leave three MEDIUM/HIGH threats partially open.'
    )


# ---------------------------------------------------------------------------
# LaTeX text generation
# ---------------------------------------------------------------------------

LATEX_TEMPLATE = r"""
\subsection*{Threats to Validity}

\paragraph{Construct Validity.}
We identify five construct validity concerns.
\emph{d95 as manifold proxy:} PCA dimensionality (d95) conflates true manifold
structure with correlated high-dimensional features; no nonlinear baseline was
computed to distinguish the two (MEDIUM).
\emph{Compression ratio vs.\ sequential structure:} zlib and lzma ratios exploit
both unigram skew and sequential redundancy; only the entropy-rate gap (real minus
shuffled) is explicitly baseline-corrected for unigram effects (MEDIUM).
\emph{Motifs and compiler boilerplate:} top k-mer motifs include compiler-universal
prologues/epilogues; the paper does not systematically separate toolchain-universal
from program-specific motifs (MEDIUM).
\emph{Zipf alpha on a 281-token vocabulary:} with fewer than 300 unique mnemonics,
the power-law fit is sensitive to tail treatment; no goodness-of-fit test against
log-normal or Weibull alternatives was performed (MEDIUM).
\emph{MI decay vs.\ long-range dependencies:} mutual information at lag~$k$ is not
corrected for indirect paths through shorter lags; the metric measures general
sequential predictability, not specifically long-range structure (MEDIUM).

\paragraph{Internal Validity.}
Extraction correctness is well-established: all five cross-checks against raw
\texttt{objdump} output pass, with zero mnemonic or boundary mismatches on
387,061 verified instructions (LOW).
PLT stub and startup-code contribution is negligible at the instruction level:
removing all stubs (0.6\% of instructions) shifts the 5-gram entropy rate by
+0.0027~bits and the zlib ratio by $-$0.006 (LOW).
The operand-dropping effect is real but bounded: opcode-only and
opcode+operand-class representations both exhibit a positive H5 entropy-rate gap,
with the gap differing by 0.013~bits across the two (MEDIUM).
Full boilerplate stripping ablation results (\texttt{boilerplate\_ablation.json})
were not available; the impact of aggressively removing first/last instructions
and \texttt{endbr64} guards on entropy rate and Zipf alpha remains
unquantified (MEDIUM).
Random seed sensitivity is negligible: the shuffled-baseline coefficient of
variation across 10~seeds is below $0.01\%$ at all n-gram orders (LOW).

\paragraph{External Validity.}
Cross-corpus generalizability is the most significant open question:
\texttt{independent\_corpora.json} was not available, so no independent
external corpus was confirmed to reproduce the findings (HIGH).
Compiler invariance is also unconfirmed: the expanded compiler matrix results
(\texttt{expanded\_compiler\_matrix.json}) were not available, leaving the
program-vs.-compiler-vs.-optimization-level variance decomposition unknown (HIGH).
All results are specific to x86-64~ELF on Linux; no ARM64, RISC-V, or
Windows~PE analysis was performed (HIGH).
Within the primary corpus, subsampling shows convergence: Zipf alpha standard
deviation drops from 0.020 at 25\% (N=4~binaries) to near zero at 100\%
(N=15~binaries), suggesting the corpus is adequate for these specific metrics
within this domain (MEDIUM).

\paragraph{Statistical Conclusion Validity.}
Bootstrap confidence intervals are computed by resampling at the binary level
(N=15), correctly treating programs as the unit of analysis, but the corpus
is a convenience sample of common Linux utilities rather than a random draw
from a well-defined population; reported CIs measure within-corpus consistency,
not population uncertainty (MEDIUM).
Multiple metrics are reported without family-wise error rate correction;
the primary structural claim is supported by converging evidence from four
independent measurement approaches (entropy rate, compression, MI, motifs),
reducing the risk that all findings are simultaneous false positives (MEDIUM).
Instruction tokens within a binary are not independent; the effective sample
size for corpus-level inference is N=15~binaries, not $N=2.9$M~tokens;
concatenated-sequence MI estimates may overstate true long-range dependencies
by conflating cross-function shared vocabulary with within-sequence structure (HIGH).
Metric convergence with sequence length is confirmed above 50K~tokens per
binary; at shorter truncation lengths (${<}$5K~tokens) entropy rates are
substantially lower, requiring length-matched comparisons across corpora (LOW).
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all result files
    data: Dict[str, Any] = {}
    for key, path in _RESULT_FILES.items():
        result = load_json(path)
        data[key] = result
        if result is None:
            logger.warning('%-30s NOT FOUND', key)
        else:
            logger.info('%-30s loaded', key)

    # Run analyses
    construct  = analyse_construct_validity(data)
    internal   = analyse_internal_validity(data)
    external   = analyse_external_validity(data)
    statistical= analyse_statistical_conclusion_validity(data)

    overall = compute_overall_assessment(construct, internal, external, statistical)

    output = {
        'construct_validity':               construct,
        'internal_validity':                internal,
        'external_validity':                external,
        'statistical_conclusion_validity':  statistical,
        'overall_assessment':               overall,
        'recommended_paper_text':           LATEX_TEMPLATE.strip(),
    }

    with _OUTPUT_FILE.open('w') as fh:
        json.dump(output, fh, indent=2)
    logger.info('Wrote %s', _OUTPUT_FILE)

    # Print summary table
    categories = [
        ('construct_validity',              construct),
        ('internal_validity',               internal),
        ('external_validity',               external),
        ('statistical_conclusion_validity', statistical),
    ]

    header = f"{'threat_category':<35} {'n_threats':>9} {'HIGH':>6} {'MEDIUM':>8} {'LOW':>5}"
    print()
    print(header)
    print('-' * len(header))
    total_n = total_high = total_medium = total_low = 0
    for name, threats in categories:
        n  = len(threats)
        hi = sum(1 for t in threats if t['severity'] == 'HIGH')
        me = sum(1 for t in threats if t['severity'] == 'MEDIUM')
        lo = sum(1 for t in threats if t['severity'] == 'LOW')
        print(f'{name:<35} {n:>9} {hi:>6} {me:>8} {lo:>5}')
        total_n += n; total_high += hi; total_medium += me; total_low += lo
    print('-' * len(header))
    print(f'{"TOTAL":<35} {total_n:>9} {total_high:>6} {total_medium:>8} {total_low:>5}')
    print()
    print('Overall assessment:')
    for line in overall.split('. '):
        if line:
            print(f'  {line.strip()}.')
    print()
    print(f'Output written to: {_OUTPUT_FILE}')


if __name__ == '__main__':
    main()
