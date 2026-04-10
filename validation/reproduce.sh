#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Full pipeline reproducibility script for OpCode-Stats
#
# Usage (from project root):
#   bash validation/reproduce.sh
#
# What this script does:
#   1. Captures the full software environment to validation/environment.json
#   2. Runs every pipeline stage in order, recording timing per phase
#   3. After each stage, SHA-256-hashes all output JSON files
#   4. Saves a consolidated manifest to validation/manifest.json
#
# Exit behaviour: set -euo pipefail — any unhandled failure aborts the run.
# Per-phase failures are caught explicitly so the manifest is always written.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# All Python commands run from the project root so relative imports work.
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python3}"
VALIDATION_DIR="${PROJECT_ROOT}/validation"
ENV_JSON="${VALIDATION_DIR}/environment.json"
MANIFEST_JSON="${VALIDATION_DIR}/manifest.json"

# Results locations (match config.yaml corpus output_dir entries)
SMOKE_DIR="results/smoke"
COREUTILS_DIR="results/coreutils_system"
COMPILER_MATRIX_DIR="results/compiler_matrix"

# Temp directory for per-phase JSON blobs used when building the manifest
PHASE_TMP_DIR="$(mktemp -d)"
# Clean up the temp directory on exit, even on error
trap 'rm -rf "${PHASE_TMP_DIR}"' EXIT

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log()  { printf '[%s] %s\n'       "$(_ts)" "$*"; }
info() { printf '[%s] INFO  %s\n' "$(_ts)" "$*"; }
warn() { printf '[%s] WARN  %s\n' "$(_ts)" "$*" >&2; }
fail() { printf '[%s] ERROR %s\n' "$(_ts)" "$*" >&2; }

phase_banner() {
    echo ""
    echo "============================================================"
    printf '  PHASE: %s\n' "$1"
    printf '  Started: %s\n' "$(_ts)"
    echo "============================================================"
}

# Return wall-clock seconds elapsed since a Unix epoch value
elapsed() { echo $(( $(date -u +%s) - $1 )); }

# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

# Compute SHA-256 of every *.json under a directory tree.
# Prints a JSON object: { "relative/path.json": "sha256hex", ... }
# Paths in the object are relative to PROJECT_ROOT for portability.
hash_json_files() {
    local search_dir="$1"
    "${PYTHON}" - "${search_dir}" "${PROJECT_ROOT}" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

search = Path(sys.argv[1])
root   = Path(sys.argv[2])
hashes = {}

for p in sorted(search.rglob("*.json")):
    try:
        rel = str(p.relative_to(root))
    except ValueError:
        rel = str(p)
    try:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[rel] = digest
    except OSError as e:
        hashes[rel] = f"ERROR: {e}"

print(json.dumps(hashes, indent=4))
PYEOF
}

# ---------------------------------------------------------------------------
# Phase recording
# ---------------------------------------------------------------------------

# Save a single phase record to PHASE_TMP_DIR/<key>.json
# Args: key  start_epoch  status  [search_dir]
record_phase() {
    local key="$1"
    local start_epoch="$2"
    local status="$3"
    local search_dir="${4:-}"

    local end_ts; end_ts="$(_ts)"
    local secs;   secs="$(elapsed "${start_epoch}")"
    local hashes_json="{}"

    if [[ -n "${search_dir}" && -d "${search_dir}" ]]; then
        hashes_json="$(hash_json_files "${search_dir}")"
    fi

    "${PYTHON}" - "${key}" "${status}" "${end_ts}" "${secs}" \
            "${PHASE_TMP_DIR}/${key}.json" <<PYEOF
import json, sys
from pathlib import Path

key, status, end_ts, secs, out_path = sys.argv[1:]
hashes = ${hashes_json}

record = {
    "status": status,
    "completed_at_utc": end_ts,
    "elapsed_seconds": int(secs),
    "output_hashes": hashes,
}

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(record, indent=2))
PYEOF
}

# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

write_manifest() {
    local completed_at; completed_at="$(_ts)"

    "${PYTHON}" - "${completed_at}" "${MANIFEST_JSON}" "${PHASE_TMP_DIR}" <<'PYEOF'
import json, sys
from pathlib import Path

completed_at = sys.argv[1]
out_path     = Path(sys.argv[2])
tmp_dir      = Path(sys.argv[3])

phases = {}
for f in sorted(tmp_dir.glob("*.json")):
    try:
        phases[f.stem] = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        phases[f.stem] = {"status": "parse_error", "error": str(e)}

manifest = {
    "schema_version": "1",
    "generated_at_utc": completed_at,
    "phases": phases,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(manifest, indent=2))
print(f"[manifest] Written to {out_path}", file=sys.stderr)
PYEOF
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

preflight_checks() {
    info "Running pre-flight checks..."

    if ! command -v "${PYTHON}" &>/dev/null; then
        fail "Python interpreter not found: ${PYTHON}"
        exit 1
    fi
    info "Python       : $(${PYTHON} --version 2>&1)"

    if [[ ! -f "${PROJECT_ROOT}/binary_dna.py" ]]; then
        fail "binary_dna.py not found in ${PROJECT_ROOT}"
        exit 1
    fi
    info "binary_dna.py: found"

    if [[ ! -f "${PROJECT_ROOT}/config.yaml" ]]; then
        fail "config.yaml not found in ${PROJECT_ROOT}"
        exit 1
    fi
    info "config.yaml  : found"

    if ! command -v objdump &>/dev/null; then
        warn "objdump not found — extraction phases will likely fail"
    else
        info "objdump      : $(objdump --version 2>&1 | head -1)"
    fi

    if ! command -v gcc &>/dev/null; then
        warn "gcc not found — compiler-matrix phase may produce no results"
    else
        info "gcc          : $(gcc --version 2>&1 | head -1)"
    fi

    if ! command -v clang &>/dev/null; then
        warn "clang not found — compiler-matrix phase may produce limited results"
    else
        info "clang        : $(clang --version 2>&1 | head -1)"
    fi

    info "Pre-flight checks passed."
}

# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

# Phase 0 — Capture environment snapshot
phase_capture_environment() {
    phase_banner "0 — Capture Environment"
    local t0; t0=$(date -u +%s)
    local status="ok"

    if "${PYTHON}" validation/capture_environment.py "${ENV_JSON}" "${PROJECT_ROOT}"; then
        info "Environment snapshot written to ${ENV_JSON}"
    else
        warn "Environment capture script exited non-zero; continuing"
        status="failed"
    fi

    record_phase "00_environment" "${t0}" "${status}" "${VALIDATION_DIR}"
    log "Phase 0 done in $(elapsed "${t0}")s  [status=${status}]"
}

# Phase 1 — Smoke corpus: extraction + full analysis (15 system binaries)
phase_smoke() {
    phase_banner "1 — Smoke Corpus  (extraction + analysis)"
    local t0; t0=$(date -u +%s)
    local status="ok"

    info "Command: python3 binary_dna.py smoke --output-dir ${SMOKE_DIR}"
    if "${PYTHON}" binary_dna.py smoke --output-dir "${SMOKE_DIR}"; then
        info "Smoke corpus completed."
    else
        warn "Smoke phase exited non-zero."
        status="failed"
    fi

    record_phase "01_smoke" "${t0}" "${status}" "${SMOKE_DIR}"
    log "Phase 1 done in $(elapsed "${t0}")s  [status=${status}]"
}

# Phase 2 — coreutils_system corpus (limit-based, ~24 unique system binaries)
phase_coreutils_system() {
    phase_banner "2 — coreutils_system Corpus"
    local t0; t0=$(date -u +%s)
    local status="ok"

    info "Command: python3 binary_dna.py corpus coreutils_system"
    if "${PYTHON}" binary_dna.py corpus coreutils_system; then
        info "coreutils_system corpus completed."
    else
        warn "coreutils_system phase exited non-zero."
        status="failed"
    fi

    record_phase "02_coreutils_system" "${t0}" "${status}" "${COREUTILS_DIR}"
    log "Phase 2 done in $(elapsed "${t0}")s  [status=${status}]"
}

# Phase 3 — N-gram language model analysis on the smoke corpus
# Requires: smoke extraction and ngram_analysis.json to exist.
phase_lm_smoke() {
    phase_banner "3 — Language Model Analysis  (smoke corpus)"
    local t0; t0=$(date -u +%s)
    local status="ok"

    local corpus_dir="${SMOKE_DIR}/corpus"
    local results_dir="${SMOKE_DIR}/results"
    local lm_output="${results_dir}/lm_analysis.json"
    local ngram_json="${results_dir}/ngram_analysis.json"

    if [[ ! -f "${ngram_json}" ]]; then
        warn "Prerequisite missing: ${ngram_json}"
        warn "Skipping LM phase — re-run after phase 1 succeeds."
        record_phase "03_lm_smoke" "${t0}" "skipped" ""
        log "Phase 3 skipped  [prerequisite missing]"
        return
    fi

    info "Command: python3 binary_dna.py lm \\"
    info "           --corpus-dir ${corpus_dir} \\"
    info "           --results-dir ${results_dir} \\"
    info "           --output ${lm_output}"

    if "${PYTHON}" binary_dna.py lm \
            --corpus-dir  "${corpus_dir}" \
            --results-dir "${results_dir}" \
            --output      "${lm_output}"; then
        info "LM analysis written to ${lm_output}"
    else
        warn "LM analysis exited non-zero."
        status="failed"
    fi

    record_phase "03_lm_smoke" "${t0}" "${status}" "${results_dir}"
    log "Phase 3 done in $(elapsed "${t0}")s  [status=${status}]"
}

# Phase 4 — Clone detection on the smoke corpus
# Requires: corpus.pkl produced by the smoke extraction step.
phase_clones_smoke() {
    phase_banner "4 — Clone Detection  (smoke corpus)"
    local t0; t0=$(date -u +%s)
    local status="ok"

    local corpus_dir="${SMOKE_DIR}/corpus"
    local clones_dir="${SMOKE_DIR}/clones"
    local corpus_pkl="${corpus_dir}/corpus.pkl"

    if [[ ! -f "${corpus_pkl}" ]]; then
        warn "Prerequisite missing: ${corpus_pkl}"
        warn "Skipping clone-detection — re-run after phase 1 succeeds."
        record_phase "04_clones_smoke" "${t0}" "skipped" ""
        log "Phase 4 skipped  [prerequisite missing]"
        return
    fi

    info "Command: python3 binary_dna.py clones \\"
    info "           --corpus-dir ${corpus_dir} \\"
    info "           --output-dir ${clones_dir}"

    if "${PYTHON}" binary_dna.py clones \
            --corpus-dir  "${corpus_dir}" \
            --output-dir  "${clones_dir}"; then
        info "Clone detection completed. Output: ${clones_dir}"
    else
        warn "Clone detection exited non-zero."
        status="failed"
    fi

    record_phase "04_clones_smoke" "${t0}" "${status}" "${clones_dir}"
    log "Phase 4 done in $(elapsed "${t0}")s  [status=${status}]"
}

# Phase 5 — Compiler matrix experiment (5 projects × gcc/clang × O0/O2/O3)
phase_compiler_matrix() {
    phase_banner "5 — Compiler Matrix Experiment"
    local t0; t0=$(date -u +%s)
    local status="ok"

    info "Command: python3 binary_dna.py compiler-matrix --output-dir ${COMPILER_MATRIX_DIR}"
    if "${PYTHON}" binary_dna.py compiler-matrix --output-dir "${COMPILER_MATRIX_DIR}"; then
        info "Compiler-matrix experiment completed. Output: ${COMPILER_MATRIX_DIR}"
    else
        warn "Compiler-matrix experiment exited non-zero."
        status="failed"
    fi

    record_phase "05_compiler_matrix" "${t0}" "${status}" "${COMPILER_MATRIX_DIR}"
    log "Phase 5 done in $(elapsed "${t0}")s  [status=${status}]"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    echo ""
    echo "============================================================"
    echo "  REPRODUCTION SUMMARY"
    echo "============================================================"

    for f in $(ls "${PHASE_TMP_DIR}"/*.json 2>/dev/null | sort); do
        local key; key="$(basename "${f}" .json)"
        local status; status="$("${PYTHON}" -c \
            "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status','?'))" \
            "${f}" 2>/dev/null || echo "?")"
        local secs; secs="$("${PYTHON}" -c \
            "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('elapsed_seconds','?'))" \
            "${f}" 2>/dev/null || echo "?")"
        printf '  %-35s  status=%-8s  %ss\n' "${key}" "${status}" "${secs}"
    done

    echo "------------------------------------------------------------"
    printf '  Environment : %s\n' "${ENV_JSON}"
    printf '  Manifest    : %s\n' "${MANIFEST_JSON}"
    echo "============================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local run_start; run_start=$(date -u +%s)

    log "OpCode-Stats reproducibility run starting"
    log "Project root : ${PROJECT_ROOT}"
    log "Python       : ${PYTHON}"
    log "Working dir  : $(pwd)"
    echo ""

    preflight_checks

    # Run all phases.  Each phase records its own status so a failure in one
    # does not prevent subsequent phases from running or the manifest from
    # being written.
    phase_capture_environment
    phase_smoke
    phase_coreutils_system
    phase_lm_smoke
    phase_clones_smoke
    phase_compiler_matrix

    info "Writing manifest ..."
    write_manifest

    print_summary

    log "Full reproduction run complete in $(elapsed "${run_start}")s"
    log "Manifest    : ${MANIFEST_JSON}"
    log "Environment : ${ENV_JSON}"
}

main "$@"
