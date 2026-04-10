# Validation Plan for OpCode-Stats Paper

## Goal
Move from "the code runs and reproduces the tables" to:
- The reported effects are real
- They are not artifacts of one toolchain or corpus
- The interpretation in the paper is justified

## Four Core Claims to Validate
1. Opcode streams have strong nontrivial statistical structure
2. That structure is not explained only by unigram frequency skew
3. The observed regularities are not mostly boilerplate/compiler artifacts
4. The conclusions generalize beyond this exact corpus and setup

---

## How to Run

```bash
# Run everything:
bash validation/run_all.sh

# Or run individual phases:
python3 validation/capture_environment.py validation/environment.json
python3 validation/extraction_verification.py
python3 validation/synthetic_validation.py
python3 validation/robustness_analysis.py
python3 validation/boilerplate_ablation.py
python3 validation/stub_exclusion.py
python3 validation/operand_aware.py
python3 validation/independent_corpora.py
python3 validation/expanded_compiler_matrix.py
python3 validation/effect_summary.py
python3 validation/threats_to_validity.py

# Full reproducibility (extracts corpus + runs analysis from scratch):
bash validation/reproduce.sh
```

---

## Phase 1 — Reproducibility Audit
Scripts: `reproduce.sh`, `capture_environment.py`
- [x] 1.1 Create reproduce.sh that runs full pipeline end-to-end
- [x] 1.2 Capture environment (OS, Python, objdump, compiler versions, package lockfile)
- [x] 1.3 Generate results/manifest.json with hashes of all intermediate outputs
- [ ] 1.4 Verify clean rerun regenerates all reported values

## Phase 2 — Internal Validity Checks

### 2.1 Extraction Correctness
Script: `extraction_verification.py` -> `results/extraction_verification.json`
- [x] 2.1.1 Sample binaries and compare extracted mnemonics against raw objdump
- [x] 2.1.2 Verify instruction boundary parsing
- [x] 2.1.3 Verify function boundary correctness
- [x] 2.1.4 Verify duplicate/hard-link filtering
- [x] 2.1.5 Verify no timeout-truncated binaries silently included

### 2.2 Metric Sanity Tests on Synthetic Data
Script: `synthetic_validation.py` -> `results/synthetic_validation.json`
- [x] 2.2.1 Build synthetic corpus: uniform random opcode sequences
- [x] 2.2.2 Build synthetic corpus: unigram-skewed but shuffled sequences
- [x] 2.2.3 Build synthetic corpus: low-order Markov sequences
- [x] 2.2.4 Build synthetic corpus: repeated-template with inserted noise
- [x] 2.2.5 Build synthetic corpus: mixed boilerplate + random bodies
- [x] 2.2.6 Run all metrics on synthetic corpora and verify expected outcomes (26/26 pass)

### 2.3 Statistical Uncertainty and Robustness
Script: `robustness_analysis.py` -> `results/robustness_analysis.json`
- [x] 2.3.1 Bootstrap CIs over binaries for all key metrics (N=200)
- [x] 2.3.2 Multiple-seed shuffled baselines (10 seeds)
- [x] 2.3.3 Corpus subsampling sensitivity (25/50/75/100%)
- [x] 2.3.4 Sequence length truncation sensitivity
- [x] 2.3.5 Leave-one-binary-out stability

## Phase 3 — Boilerplate and Confound Ablation

### 3.1 Strip Boundary Boilerplate
Script: `boilerplate_ablation.py` -> `results/boilerplate_ablation.json`
- [x] 3.1.1 Create ablated corpora (remove first/last k instructions, k=1,2,3,5)
- [x] 3.1.2 Remove all endbr64, common prologues/epilogues
- [x] 3.1.3 Full boilerplate strip (endbr64 + prologue + epilogue)
- [x] 3.1.4 Rerun entropy, compression, Zipf on each variant

### 3.2 Exclude Startup and Library Stubs
Script: `stub_exclusion.py` -> `results/stub_exclusion.json`
- [x] 3.2.1 Classify functions: startup/runtime vs stubs vs internal application
- [x] 3.2.2 Rerun analysis on each category separately

### 3.3 Operand-Aware Control Experiment
Script: `operand_aware.py` -> `results/operand_aware.json`
- [x] 3.3.1 Create variants: opcode-only, opcode+operand class, opcode+register class, opcode+immediate bucket
- [x] 3.3.2 Rerun key metrics on each variant

## Phase 4 — External Validation on Independent Corpora
Script: `independent_corpora.py` -> `results/independent_corpora.json`
- [x] 4.1 Corpus A: system utilities (smoke corpus baseline)
- [x] 4.2 Corpus B: independent applications (ssh, wget, diff, make, etc.)
- [x] 4.3 Corpus C: developer/server tools (perl, gdb, g++, etc.)
- [x] 4.4 Pre-registered primary metric set
- [x] 4.5 Cross-corpus LM: train on A, test on B, repeat all pairs

## Phase 5 — Compiler and Toolchain Validation
Script: `expanded_compiler_matrix.py` -> `results/expanded_compiler_matrix.json`
- [x] 5.1 10 inline C programs compiled with gcc/clang x O0/O1/O2/O3/Os
- [x] 5.2 Per-binary metric computation (Zipf α, entropy, compression)
- [x] 5.3 Variance decomposition: program vs compiler vs optimization level
- [x] 5.4 Compiler-invariance determination

## Phase 6 — Stronger Inferential Framing
Script: `effect_summary.py` -> `results/effect_summary.json`
- [x] 6.1 For each of 10 claims: estimate + CI + ablation + replication + verdict
- [x] 6.2 Claims validation matrix (claim | confound | experiment | outcome)
- [x] 6.3 Recommended paper rewrite for each claim

## Phase 7 — Threats-to-Validity Section
Script: `threats_to_validity.py` -> `results/threats_to_validity.json`
- [x] 7.1 Construct validity analysis (5 threats)
- [x] 7.2 Internal validity analysis (5 threats)
- [x] 7.3 External validity analysis (4 threats)
- [x] 7.4 Statistical conclusion validity analysis (5 threats)
- [x] 7.5 LaTeX-ready threats-to-validity section text

---

## File Inventory

| File | Phase | Purpose |
|------|-------|---------|
| `PLAN.md` | — | This file |
| `run_all.sh` | — | Master runner for all phases |
| `reproduce.sh` | 1 | Full pipeline reproduction with hashing |
| `capture_environment.py` | 1 | Environment capture (OS, tools, versions) |
| `extraction_verification.py` | 2.1 | Objdump parsing correctness checks |
| `synthetic_validation.py` | 2.2 | Metric sanity on 5 synthetic corpus families |
| `robustness_analysis.py` | 2.3 | Bootstrap CIs, subsampling, seed sensitivity |
| `boilerplate_ablation.py` | 3.1 | Strip prologues/epilogues, rerun metrics |
| `stub_exclusion.py` | 3.2 | Classify and exclude startup/stub functions |
| `operand_aware.py` | 3.3 | Opcode + operand/register/immediate variants |
| `independent_corpora.py` | 4 | Build + analyze 3 disjoint corpora |
| `expanded_compiler_matrix.py` | 5 | 10 programs x gcc/clang x 5 opt levels |
| `effect_summary.py` | 6 | Unified claim-by-claim evidence summary |
| `threats_to_validity.py` | 7 | 19-threat structured validity analysis |

## Results Already Generated

| Result File | Status |
|-------------|--------|
| `extraction_verification.json` | Done - all checks pass |
| `synthetic_validation.json` | Done - 26/26 checks pass |
| `robustness_analysis.json` | Done - most metrics stable |
| `stub_exclusion.json` | Done - 99.4% of instructions are internal |
| `operand_aware.json` | Done - gap persists across representations |
| `effect_summary.json` | Done - 8 STRONG, 1 MODERATE, 1 UNSUPPORTED |
| `threats_to_validity.json` | Done - 5 HIGH, 10 MEDIUM, 4 LOW threats |
| `boilerplate_ablation.json` | Done - all 16 variants PASS, max 3.2% change |
| `independent_corpora.json` | Done - 3 corpora, all consistent, overall PASS |
| `expanded_compiler_matrix.json` | Done - 100 binaries, program dominates (R^2=0.40-0.66) |

## Minimal Publishable Validation Package (Priority)
1. [x] Full reproducibility script
2. [x] At least one independent non-nested corpus - 3 corpora, all PASS
3. [x] Boilerplate-stripping ablation - 16/16 variants PASS
4. [x] Leave-one-binary-out (in robustness_analysis.py)
5. [x] Expanded compiler/toolchain experiment - 100 binaries, compiler R^2 < 0.03
