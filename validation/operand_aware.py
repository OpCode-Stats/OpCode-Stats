"""
Operand-Aware Representation Validation.

The paper deliberately drops operands and works with opcode mnemonics only.
This script tests whether the core findings (Zipfian distribution, entropy
rate gap between real and shuffled sequences, high compressibility) hold
under four progressively richer token representations:

1. opcode_only           - just the mnemonic                (current paper baseline)
2. opcode_operand_class  - mnemonic + coarse operand types  (reg/mem/imm/label)
3. opcode_register_class - mnemonic + full register names
4. opcode_immediate_bucket - mnemonic + bucketed immediates (0/small/medium/large)

For each variant we compute:
  - vocabulary size
  - unigram entropy  (H1)
  - entropy rates    n=1..5
  - shuffled entropy rates (unigram-preserving baseline)
  - entropy gap (real - shuffled) at each n
  - zlib and lzma compression ratios on the text encoding

Key question: does the entropy-rate gap (sequential structure) persist, grow,
or disappear when tokens carry more information?

Run standalone:
    python3 validation/operand_aware.py
"""

import sys
import re
import json
import zlib
import lzma
import logging
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, '/home/aaslyan/OpCode-Stats')

import matplotlib
matplotlib.use('Agg')

import numpy as np

from utils.helpers import Instruction, Function, Binary, load_pickle
from analysis.ngrams import compute_entropy_rate, compute_shuffled_entropy_rates

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("operand_aware")

# ---------------------------------------------------------------------------
# x86-64 register name patterns
# ---------------------------------------------------------------------------

# All canonical x86-64 register names, ordered longest-first so the regex
# prefers "rax" over "ax" in left-to-right matching.
_X86_REGISTERS: List[str] = sorted([
    # 64-bit GPRs
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp", "rbp",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
    # 32-bit
    "eax", "ebx", "ecx", "edx", "esi", "edi", "esp", "ebp",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
    # 16-bit
    "ax",  "bx",  "cx",  "dx",  "si",  "di",  "sp",  "bp",
    "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
    # 8-bit
    "al",  "bl",  "cl",  "dl",  "sil", "dil", "spl", "bpl",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
    "ah",  "bh",  "ch",  "dh",
    # XMM / YMM / ZMM (just number suffix; we match "xmm0" .. "xmm15")
    *[f"xmm{i}" for i in range(16)],
    *[f"ymm{i}" for i in range(16)],
    *[f"zmm{i}" for i in range(16)],
    # Segment / special
    "rip", "eip", "cs", "ds", "es", "fs", "gs", "ss",
    "st0", "st1", "st2", "st3", "st4", "st5", "st6", "st7",
    "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7",
], key=len, reverse=True)

_REG_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(r) for r in _X86_REGISTERS) + r')\b',
    re.IGNORECASE,
)

# Immediate value pattern: optional sign, hex or decimal
_IMM_PATTERN = re.compile(r'-?(?:0x[0-9a-fA-F]+|[0-9]+)')

# Memory operand: contains '[' ']'
_MEM_PATTERN = re.compile(r'\[')

# Jump/call targets that look like labels (symbol names, not pure hex)
_LABEL_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_@.<>+\-]*$')


# ---------------------------------------------------------------------------
# Operand classification helpers
# ---------------------------------------------------------------------------

def _classify_operand(op: str) -> str:
    """Return coarse operand class: reg | mem | imm | label."""
    op = op.strip()
    if not op:
        return ""
    # Memory reference
    if _MEM_PATTERN.search(op):
        return "mem"
    # Pure register
    if _REG_PATTERN.fullmatch(op):
        return "reg"
    # Immediate / numeric
    if _IMM_PATTERN.fullmatch(op.strip()):
        return "imm"
    # Label / symbol
    if _LABEL_PATTERN.match(op):
        return "label"
    # Mixed (e.g. "BYTE PTR [rsp+8]" — already caught by MEM above;
    # fall-through handles edge cases like "QWORD PTR" without brackets)
    if _REG_PATTERN.search(op):
        return "reg"
    return "imm"  # default for unrecognised numerics / expressions


def _split_operands(operands_str: str) -> List[str]:
    """Split comma-separated operand string, respecting brackets."""
    # Simple split — brackets don't contain commas in x86 asm, so plain split works
    if not operands_str:
        return []
    return [o.strip() for o in operands_str.split(",") if o.strip()]


def _imm_bucket(value_str: str) -> Optional[str]:
    """Parse an immediate string and return its bucket name."""
    s = value_str.strip()
    try:
        val = int(s, 16) if s.startswith("0x") or s.startswith("-0x") else int(s)
        val = abs(val)
        if val == 0:
            return "0"
        if val <= 15:
            return "small"
        if val <= 255:
            return "medium"
        return "large"
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Token generators for each variant
# ---------------------------------------------------------------------------

def _token_opcode_only(instr: Instruction) -> str:
    """Variant 1: just the mnemonic."""
    return instr.mnemonic


def _token_operand_class(instr: Instruction) -> str:
    """Variant 2: mnemonic + coarse operand type for each operand."""
    ops = _split_operands(instr.operands)
    if not ops:
        return instr.mnemonic
    classes = [_classify_operand(o) for o in ops]
    classes = [c for c in classes if c]
    if not classes:
        return instr.mnemonic
    return instr.mnemonic + "_" + "_".join(classes)


def _token_register_class(instr: Instruction) -> str:
    """Variant 3: mnemonic + all register names found in operands."""
    regs = _REG_PATTERN.findall(instr.operands.lower()) if instr.operands else []
    if not regs:
        return instr.mnemonic
    return instr.mnemonic + "_" + "_".join(regs)


def _token_immediate_bucket(instr: Instruction) -> str:
    """Variant 4: for instructions with immediates, bucket the value."""
    ops = _split_operands(instr.operands)
    parts = [instr.mnemonic]
    for op in ops:
        op = op.strip()
        if _MEM_PATTERN.search(op):
            parts.append("mem")
        elif _REG_PATTERN.fullmatch(op):
            parts.append("reg")
        elif _IMM_PATTERN.fullmatch(op):
            bucket = _imm_bucket(op)
            parts.append(f"imm_{bucket}" if bucket is not None else "imm")
        else:
            # Could be a complex expression; check for registers inside
            if _REG_PATTERN.search(op):
                parts.append("reg")
            else:
                # Could be a label/symbol
                imm_match = _IMM_PATTERN.search(op)
                if imm_match:
                    bucket = _imm_bucket(imm_match.group())
                    parts.append(f"imm_{bucket}" if bucket is not None else "imm")
                else:
                    parts.append("label")
    if len(parts) == 1:
        return parts[0]
    return "_".join(parts)


# Mapping from variant name to token generator
_VARIANTS: Dict[str, callable] = {
    "opcode_only":            _token_opcode_only,
    "opcode_operand_class":   _token_operand_class,
    "opcode_register_class":  _token_register_class,
    "opcode_immediate_bucket": _token_immediate_bucket,
}


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def _build_sequences(binaries: List[Binary], token_fn) -> List[List[str]]:
    """Apply *token_fn* to every instruction; return per-binary sequences."""
    seqs = []
    for binary in binaries:
        seq = []
        for func in binary.functions:
            for instr in func.instructions:
                tok = token_fn(instr)
                if tok:
                    seq.append(tok)
        if seq:
            seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Entropy computation on plain sequences via wrapper
# ---------------------------------------------------------------------------

class _SequenceWrapper:
    """Lightweight wrapper so ngrams module can consume a plain list[str]."""
    def __init__(self, seq: List[str]):
        self._seq = seq

    @property
    def full_opcode_sequence(self) -> List[str]:
        return self._seq


def _wrapped_binaries(seqs: List[List[str]]) -> List[_SequenceWrapper]:
    return [_SequenceWrapper(s) for s in seqs]


def _compute_entropy_rates_for_seqs(seqs: List[List[str]], max_n: int = 5) -> List[float]:
    wrapped = _wrapped_binaries(seqs)
    try:
        rates = compute_entropy_rate(wrapped, max_n=max_n)
        return [r["entropy_rate"] for r in rates]
    except Exception as exc:
        logger.warning("Entropy rate failed: %s", exc)
        return [0.0] * max_n


def _compute_shuffled_rates_for_seqs(seqs: List[List[str]], max_n: int = 5,
                                     seed: int = 42) -> List[float]:
    wrapped = _wrapped_binaries(seqs)
    try:
        rates = compute_shuffled_entropy_rates(wrapped, max_n=max_n, seed=seed)
        return [r["entropy_rate"] for r in rates]
    except Exception as exc:
        logger.warning("Shuffled entropy rate failed: %s", exc)
        return [0.0] * max_n


# ---------------------------------------------------------------------------
# Compression on plain sequences (text encoding)
# ---------------------------------------------------------------------------

def _compress_seqs(seqs: List[List[str]]) -> Tuple[float, float]:
    """Concatenate all sequences as space-separated text and compress."""
    if not seqs:
        return 1.0, 1.0
    text = " ".join(" ".join(s) for s in seqs).encode("utf-8")
    if not text:
        return 1.0, 1.0
    n = len(text)
    try:
        zlib_ratio = len(zlib.compress(text)) / n
    except Exception:
        zlib_ratio = 1.0
    try:
        lzma_ratio = len(lzma.compress(text)) / n
    except Exception:
        lzma_ratio = 1.0
    return zlib_ratio, lzma_ratio


# ---------------------------------------------------------------------------
# Per-variant metrics
# ---------------------------------------------------------------------------

def _variant_metrics(seqs: List[List[str]], max_n: int = 5) -> Dict:
    """Compute all metrics for a given set of sequences."""
    if not seqs:
        return {"error": "no sequences"}

    # Vocabulary and unigram entropy
    token_counts: Counter = Counter()
    total_tokens = 0
    for seq in seqs:
        token_counts.update(seq)
        total_tokens += len(seq)

    vocab_size = len(token_counts)
    if vocab_size == 0:
        return {"error": "empty vocabulary"}

    # Unigram Shannon entropy
    probs = np.array(list(token_counts.values()), dtype=float)
    probs /= probs.sum()
    h1 = float(-np.sum(probs * np.log2(probs + 1e-300)))

    # Entropy rates
    ent_rates = _compute_entropy_rates_for_seqs(seqs, max_n=max_n)
    shuf_rates = _compute_shuffled_rates_for_seqs(seqs, max_n=max_n, seed=42)

    # Gaps: real - shuffled
    gaps = [float(r - s) for r, s in zip(ent_rates, shuf_rates)]

    # Compression
    zlib_ratio, lzma_ratio = _compress_seqs(seqs)

    return {
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "h1_unigram_entropy": h1,
        "entropy_rates": [float(r) for r in ent_rates],
        "shuffled_rates": [float(r) for r in shuf_rates],
        "gap": gaps,
        "compression": {
            "zlib_ratio": float(zlib_ratio),
            "lzma_ratio": float(lzma_ratio),
        },
    }


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_corpus() -> List[Binary]:
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

def _print_table(rows) -> None:
    header = ("variant", "vocab_size", "H1", "H5_rate", "gap_H5", "interpretation")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]

    def _fmt(row):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))

    sep = "  ".join("-" * w for w in widths)
    print("\n" + _fmt(header))
    print(sep)
    for row in rows:
        print(_fmt(row))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    random.seed(42)

    # --- load corpus ---
    try:
        binaries = _load_corpus()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not binaries:
        print("ERROR: corpus is empty.")
        sys.exit(1)

    # --- compute per-variant metrics ---
    variants_output: Dict[str, Dict] = {}
    MAX_N = 5

    for variant_name, token_fn in _VARIANTS.items():
        print(f"Building sequences for variant: {variant_name} ...")
        seqs = _build_sequences(binaries, token_fn)
        if not seqs:
            print(f"  WARNING: no sequences produced for {variant_name}.")
            variants_output[variant_name] = {"error": "no sequences"}
            continue

        total_tokens = sum(len(s) for s in seqs)
        print(f"  {len(seqs)} sequences, {total_tokens} tokens total.")
        print(f"  Computing metrics ...")
        metrics = _variant_metrics(seqs, max_n=MAX_N)
        variants_output[variant_name] = metrics

    # --- comparison across variants ---
    # Gap at H5 for each variant (index 4 = n=5)
    gaps_h5 = {}
    for vname, m in variants_output.items():
        if "gap" in m and len(m["gap"]) >= 5:
            gaps_h5[vname] = m["gap"][4]

    # Determine if gap is stable (all variants within 20% of the opcode_only gap)
    baseline_gap = gaps_h5.get("opcode_only", 0.0)
    gap_stable = True
    if abs(baseline_gap) > 1e-6:
        for vname, g in gaps_h5.items():
            if abs(g - baseline_gap) / abs(baseline_gap) > 0.20:
                gap_stable = False
                break

    # Interpretation
    gap_values = list(gaps_h5.values())
    if not gap_values:
        interpretation = "Insufficient data."
    elif gap_stable:
        interpretation = (
            "The entropy-rate gap (real minus shuffled at n=5) is stable across "
            "all four representations, indicating that the observed sequential "
            "structure is a robust property of opcode sequences and does not "
            "depend on dropping operand information."
        )
    else:
        max_v = max(gaps_h5, key=lambda k: abs(gaps_h5[k]))
        interpretation = (
            f"The gap varies across representations (max at '{max_v}'). "
            "Richer tokens capture additional sequential dependencies, suggesting "
            "that operand information carries structure beyond the mnemonic alone."
        )

    comparison = {
        "gap_h5_by_variant": gaps_h5,
        "gap_stable": gap_stable,
        "interpretation": interpretation,
    }

    # --- assemble output ---
    output = {
        "variants": variants_output,
        "comparison": comparison,
    }

    # --- save ---
    out_path = Path('/home/aaslyan/OpCode-Stats/validation/results/operand_aware.json')
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
    table_rows = []
    for vname in _VARIANTS:
        m = variants_output.get(vname, {})
        if "error" in m:
            table_rows.append((vname, "err", "err", "err", "err", m["error"]))
            continue
        vocab = m.get("vocab_size", 0)
        h1    = f"{m.get('h1_unigram_entropy', 0.0):.4f}"
        ent   = m.get("entropy_rates", [0.0] * 5)
        h5    = f"{ent[4]:.4f}" if len(ent) >= 5 else "N/A"
        gaps  = m.get("gap", [0.0] * 5)
        gh5   = f"{gaps[4]:.4f}" if len(gaps) >= 5 else "N/A"
        zlib  = m.get("compression", {}).get("zlib_ratio", 0.0)
        interp = (
            "gap stable" if gap_stable else "gap varies"
        )
        table_rows.append((vname, vocab, h1, h5, gh5, interp))

    _print_table(table_rows)

    print("Comparison:")
    print(f"  gap_stable: {gap_stable}")
    print(f"  interpretation: {interpretation}")


if __name__ == "__main__":
    main()
