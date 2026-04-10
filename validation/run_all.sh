#!/usr/bin/env bash
# Master runner for the full validation suite.
# Usage: bash validation/run_all.sh
#
# Runs all validation phases in order. Each phase is independent
# and produces results in validation/results/.
# Phases that have already produced output are skipped unless
# FORCE_RERUN=1 is set.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
RESULTS="validation/results"
FORCE="${FORCE_RERUN:-0}"
LOG_PREFIX="[validation]"

mkdir -p "$RESULTS"

log() { echo "$LOG_PREFIX $(date +%H:%M:%S) $*"; }
hr()  { echo "────────────────────────────────────────────────────────"; }

should_run() {
    local output_file="$1"
    if [[ "$FORCE" == "1" ]]; then return 0; fi
    if [[ -f "$output_file" ]]; then
        log "SKIP  $output_file already exists (set FORCE_RERUN=1 to override)"
        return 1
    fi
    return 0
}

run_phase() {
    local name="$1"
    local script="$2"
    local output="$3"
    hr
    if should_run "$output"; then
        log "START $name"
        local start=$SECONDS
        if $PYTHON "$script"; then
            local elapsed=$(( SECONDS - start ))
            log "DONE  $name (${elapsed}s)"
        else
            log "FAIL  $name (exit code $?)"
        fi
    fi
}

log "=== OpCode-Stats Validation Suite ==="
log "Python: $($PYTHON --version 2>&1)"
log "Working directory: $(pwd)"
log ""

# Phase 1: Environment capture
hr
log "Phase 1: Reproducibility audit"
if should_run "validation/environment.json"; then
    $PYTHON validation/capture_environment.py validation/environment.json
    log "DONE  Environment captured"
fi

# Phase 2.1: Extraction verification
run_phase "Phase 2.1: Extraction correctness" \
    "validation/extraction_verification.py" \
    "$RESULTS/extraction_verification.json"

# Phase 2.2: Synthetic data validation
run_phase "Phase 2.2: Synthetic data tests" \
    "validation/synthetic_validation.py" \
    "$RESULTS/synthetic_validation.json"

# Phase 2.3: Robustness analysis
run_phase "Phase 2.3: Statistical robustness" \
    "validation/robustness_analysis.py" \
    "$RESULTS/robustness_analysis.json"

# Phase 3.1: Boilerplate ablation
run_phase "Phase 3.1: Boilerplate ablation" \
    "validation/boilerplate_ablation.py" \
    "$RESULTS/boilerplate_ablation.json"

# Phase 3.2: Stub exclusion
run_phase "Phase 3.2: Startup/stub exclusion" \
    "validation/stub_exclusion.py" \
    "$RESULTS/stub_exclusion.json"

# Phase 3.3: Operand-aware control
run_phase "Phase 3.3: Operand-aware experiment" \
    "validation/operand_aware.py" \
    "$RESULTS/operand_aware.json"

# Phase 4: Independent corpora
run_phase "Phase 4: Independent corpora" \
    "validation/independent_corpora.py" \
    "$RESULTS/independent_corpora.json"

# Phase 5: Expanded compiler matrix
run_phase "Phase 5: Compiler matrix" \
    "validation/expanded_compiler_matrix.py" \
    "$RESULTS/expanded_compiler_matrix.json"

# Phase 6: Effect summary (reads all prior results)
hr
log "Phase 6: Effect summary (aggregation)"
# Always rerun since it reads other results
$PYTHON validation/effect_summary.py
log "DONE  Effect summary"

# Phase 7: Threats to validity (reads all prior results)
hr
log "Phase 7: Threats to validity"
$PYTHON validation/threats_to_validity.py
log "DONE  Threats to validity"

hr
log ""
log "=== Validation Suite Complete ==="
log "Results in: $RESULTS/"
log ""
log "Files produced:"
ls -1 "$RESULTS"/*.json 2>/dev/null | while read f; do
    size=$(stat --printf='%s' "$f" 2>/dev/null || stat -f'%z' "$f" 2>/dev/null)
    echo "  $(basename "$f")  (${size} bytes)"
done
