#!/usr/bin/env bash
# =============================================================================
# Decision-before-Feature: 完整正式数据生成脚本
#
# 规模:
#   BBOB train:      54 shards,  19440 runs, ~454M FE
#   BBOB validation:  18 shards,   6480 runs, ~151M FE
#   CEC2017 test:     87 shards,  10440 runs, ~313M FE
#   Total:           159 shards, 36360 runs, ~918M FE
#
# 预计运行时间 (取决于 CPU 核数):
#   - 8 workers: 约 60-80 小时
#   - 16 workers: 约 30-40 小时
#   - 32 workers: 约 15-20 小时
#
# 用法:
#   chmod +x scripts/run_full_experiment.sh
#   ./scripts/run_full_experiment.sh [WORKERS]
#
#   WORKERS: 并行 worker 数 (默认 8)
#
# 可选环境变量:
#   SKIP_CEC2017=1   跳过 CEC2017 (只跑 BBOB)
#   SKIP_QUERY=1     跳过 query/action-loss/selection-reference 阶段
#   QUERY_ID         query id (默认 descriptor_cheap_invariant)
#   SAMPLE_DESIGN   sample design id (默认 lhs_50d)
# =============================================================================
set -euo pipefail

# ── 参数 ──
WORKERS="${1:-8}"
QUERY_ID="${QUERY_ID:-descriptor_cheap_invariant}"
SAMPLE_DESIGN="${SAMPLE_DESIGN:-lhs_50d}"
SKIP_CEC2017="${SKIP_CEC2017:-0}"
SKIP_QUERY="${SKIP_QUERY:-0}"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }
step() { echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"; echo -e "${BLUE}  $*${NC}"; echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"; }

# ── 检查 ──
cd "$(dirname "$0")/.."
log "Working directory: $(pwd)"
log "Workers: ${WORKERS}"
log "Query ID: ${QUERY_ID}"
log "Sample design: ${SAMPLE_DESIGN}"
test -f pyproject.toml || { err "pyproject.toml not found"; exit 1; }
test -f configs/phase1_bbob_train.yaml || { err "train config not found"; exit 1; }
test -f configs/phase1_bbob_validation.yaml || { err "validation config not found"; exit 1; }

# ── 配置 ──
TRAIN_CFG="configs/phase1_bbob_train.yaml"
VAL_CFG="configs/phase1_bbob_validation.yaml"
CEC_CFG="configs/phase1_cec2017_test.yaml"
ALL_CFGS=("${TRAIN_CFG}" "${VAL_CFG}")
if [ "${SKIP_CEC2017}" != "1" ]; then
    ALL_CFGS+=("${CEC_CFG}")
fi

# ── 计数器 ──
STEP=0
TOTAL_STEPS=10
if [ "${SKIP_CEC2017}" = "1" ]; then
    TOTAL_STEPS=$((TOTAL_STEPS - 1))
fi
if [ "${SKIP_QUERY}" = "1" ]; then
    TOTAL_STEPS=$((TOTAL_STEPS - 5))
fi

# =============================================================================
# Phase 1: Trajectory Collection
# =============================================================================

step "Step $((++STEP))/${TOTAL_STEPS}: Planning BBOB train shards"
uv run phase1-plan-shards --config "${TRAIN_CFG}"
log "BBOB train shard plan complete."

step "Step $((++STEP))/${TOTAL_STEPS}: Collecting BBOB train trajectories (54 shards, 19440 runs)"
log "This is the longest step. Estimated 30-50% of total runtime."
uv run phase1-collect-batch --config "${TRAIN_CFG}" --sharded --workers "${WORKERS}"
log "BBOB train trajectories complete."

step "Step $((++STEP))/${TOTAL_STEPS}: Planning BBOB validation shards"
uv run phase1-plan-shards --config "${VAL_CFG}"
log "BBOB validation shard plan complete."

step "Step $((++STEP))/${TOTAL_STEPS}): Collecting BBOB validation trajectories (18 shards, 6480 runs)"
uv run phase1-collect-batch --config "${VAL_CFG}" --sharded --workers "${WORKERS}"
log "BBOB validation trajectories complete."

if [ "${SKIP_CEC2017}" != "1" ]; then
    step "Step $((++STEP))/${TOTAL_STEPS}: Collecting CEC2017 test trajectories (87 shards, 10440 runs)"
    uv run phase1-plan-shards --config "${CEC_CFG}"
    uv run phase1-collect-batch --config "${CEC_CFG}" --sharded --workers "${WORKERS}"
    log "CEC2017 trajectories complete."
fi

# =============================================================================
# Phase 2: Data Quality Checks
# =============================================================================

step "Step $((++STEP))/${TOTAL_STEPS}: Checking trajectory shard data quality"
for cfg in "${ALL_CFGS[@]}"; do
    log "Checking ${cfg}..."
    uv run phase1-check-trajectory-shards "${cfg}"
done
log "All trajectory shards validated."

step "Step $((++STEP))/${TOTAL_STEPS}: Running consistency checks"
log "optimizer-state-check..."
uv run optimizer-state-check
log "behavior-permutation-check..."
uv run behavior-permutation-check
log "algorithm-consistency-check..."
uv run algorithm-consistency-check
log "Consistency checks passed."

if [ "${SKIP_QUERY}" = "1" ]; then
    log "SKIP_QUERY=1, skipping query/action-loss/selection-reference phases."
    log "Full data generation (trajectory phase) complete."
    exit 0
fi

# =============================================================================
# Phase 3: Behavior Extraction
# =============================================================================

step "Step $((++STEP))/${TOTAL_STEPS}: Extracting behavior states"
for cfg in "${ALL_CFGS[@]}"; do
    log "Extracting behavior for ${cfg}..."
    uv run behavior-extract-batch --config "${cfg}"
done
log "Behavior extraction complete."

# =============================================================================
# Phase 4: Query Sampling & Features
# =============================================================================

step "Step $((++STEP))/${TOTAL_STEPS}: Generating query samples and features"
for cfg in "${ALL_CFGS[@]}"; do
    log "Generating query samples for ${cfg}..."
    uv run query-sample-batch --config "${cfg}" --sample-design-id "${SAMPLE_DESIGN}"
done
log "Extracting query features..."
uv run query-extract-cheap \
    --samples results/landscape_queries/samples/"${SAMPLE_DESIGN}"/bbob_train/samples.parquet \
    --output results/landscape_queries/features/"${SAMPLE_DESIGN}"/bbob_train/features.parquet
if [ "${SKIP_CEC2017}" != "1" ]; then
    uv run query-extract-cheap \
        --samples results/landscape_queries/samples/"${SAMPLE_DESIGN}"/cec2017_test/samples.parquet \
        --output results/landscape_queries/features/"${SAMPLE_DESIGN}"/cec2017_test/features.parquet
fi
log "Query consistency check..."
SAMPLE_PATHS=""
FEATURE_PATHS=""
for cfg in "${ALL_CFGS[@]}"; do
    split=$(basename "$(dirname "$(grep -o 'output: .*' "${cfg}" | awk '{print $2}')" )")
    s="results/landscape_queries/samples/${SAMPLE_DESIGN}/${split}/samples.parquet"
    f="results/landscape_queries/features/${SAMPLE_DESIGN}/${split}/features.parquet"
    if [ -f "$s" ]; then SAMPLE_PATHS="${SAMPLE_PATHS} --samples ${s}"; fi
    if [ -f "$f" ]; then FEATURE_PATHS="${FEATURE_PATHS} --features ${f}"; fi
done
uv run query-consistency ${SAMPLE_PATHS} ${FEATURE_PATHS}
log "Query samples and features complete."

# =============================================================================
# Phase 5: Action Losses
# =============================================================================

step "Step $((++STEP))/${TOTAL_STEPS}): Generating action losses (this is very compute-intensive)"
# query_adjusted_budget
log "Generating query_adjusted_budget action losses..."
for cfg in "${ALL_CFGS[@]}"; do
    split=$(basename "$(dirname "$(grep -o 'output: .*' "${cfg}" | awk '{print $2}')" )")
    out_dir="results/action_losses/query_adjusted/${split}"
    mkdir -p "${out_dir}"
    # Get functions from config
    functions=$(python3 -c "import yaml; cfg=yaml.safe_load(open('${cfg}')); print(' '.join(str(f) for f in cfg['functions']))")
    dims=$(python3 -c "import yaml; cfg=yaml.safe_load(open('${cfg}')); print(' '.join(str(d) for d in cfg['dimensions']))")
    for f in ${functions}; do
        for d in ${dims}; do
            out="${out_dir}/f${f}_d${d}.parquet"
            if [ -f "${out}" ]; then
                log "  skip existing ${out}"
                continue
            fi
            log "  ${split} f${f} d${d}..."
            uv run selection-reference-evaluate-actions \
                --config "${cfg}" \
                --train-config "${TRAIN_CFG}" \
                --action-budget-mode query_adjusted_budget \
                --sample-design-id "${SAMPLE_DESIGN}" \
                --output "${out}" \
                --only-function "${f}" \
                --only-dimension "${d}" \
                --all-prefixes
        done
    done
done
log "query_adjusted_budget action losses complete."

# behavior_only_full_budget
log "Generating behavior_only_full_budget action losses..."
for cfg in "${ALL_CFGS[@]}"; do
    split=$(basename "$(dirname "$(grep -o 'output: .*' "${cfg}" | awk '{print $2}')" )")
    out_dir="results/action_losses/behavior_only/${split}"
    mkdir -p "${out_dir}"
    functions=$(python3 -c "import yaml; cfg=yaml.safe_load(open('${cfg}')); print(' '.join(str(f) for f in cfg['functions']))")
    dims=$(python3 -c "import yaml; cfg=yaml.safe_load(open('${cfg}')); print(' '.join(str(d) for d in cfg['dimensions']))")
    for f in ${functions}; do
        for d in ${dims}; do
            out="${out_dir}/f${f}_d${d}.parquet"
            if [ -f "${out}" ]; then
                log "  skip existing ${out}"
                continue
            fi
            log "  ${split} f${f} d${d}..."
            uv run selection-reference-evaluate-actions \
                --config "${cfg}" \
                --train-config "${TRAIN_CFG}" \
                --action-budget-mode behavior_only_full_budget \
                --output "${out}" \
                --only-function "${f}" \
                --only-dimension "${d}" \
                --all-prefixes
        done
    done
done
log "behavior_only_full_budget action losses complete."

# pre_run_query_adjusted_budget
log "Generating pre_run_query_adjusted_budget action losses..."
for cfg in "${ALL_CFGS[@]}"; do
    split=$(basename "$(dirname "$(grep -o 'output: .*' "${cfg}" | awk '{print $2}')" )")
    out_dir="results/action_losses/pre_run/${split}"
    mkdir -p "${out_dir}"
    functions=$(python3 -c "import yaml; cfg=yaml.safe_load(open('${cfg}')); print(' '.join(str(f) for f in cfg['functions']))")
    dims=$(python3 -c "import yaml; cfg=yaml.safe_load(open('${cfg}')); print(' '.join(str(d) for d in cfg['dimensions']))")
    for f in ${functions}; do
        for d in ${dims}; do
            out="${out_dir}/f${f}_d${d}.parquet"
            if [ -f "${out}" ]; then
                log "  skip existing ${out}"
                continue
            fi
            log "  ${split} f${f} d${d}..."
            uv run selection-reference-evaluate-actions \
                --config "${cfg}" \
                --train-config "${TRAIN_CFG}" \
                --action-budget-mode pre_run_query_adjusted_budget \
                --sample-design-id "${SAMPLE_DESIGN}" \
                --output "${out}" \
                --only-function "${f}" \
                --only-dimension "${d}"
        done
    done
done
log "pre_run_query_adjusted_budget action losses complete."

# =============================================================================
# Phase 6: Selection Reference Build
# =============================================================================

step "Step $((++STEP))/${TOTAL_STEPS}): Building selection references"

build_selection_reference() {
    local mode="$1"
    local action_dir="$2"
    local out_subdir="$3"

    local train_files=""
    local val_files=""

    # Gather all train action-loss files
    for f in $(ls "${action_dir}/bbob_train/"*.parquet 2>/dev/null | sort); do
        train_files="${train_files} --train-action-losses ${f}"
    done

    # Gather validation action-loss files (if they exist)
    if [ -d "${action_dir}/bbob_validation" ]; then
        for f in $(ls "${action_dir}/bbob_validation/"*.parquet 2>/dev/null | sort); do
            val_files="${val_files} --predict-action-losses ${f}"
        done
    fi

    # Gather behavior files
    local beh_files=""
    for cfg in "${TRAIN_CFG}" "${VAL_CFG}"; do
        output_dir=$(grep -o 'output: .*' "${cfg}" | awk '{print $2}')
        beh_dir=$(dirname "${output_dir}")
        for f in $(find "${beh_dir}" -name "behavior.parquet" | sort); do
            beh_files="${beh_files} --behavior ${f}"
        done
    done

    # Gather query feature files
    local qf_files=""
    for f in $(find "results/landscape_queries/features/${SAMPLE_DESIGN}" -name "features.parquet" | sort); do
        qf_files="${qf_files} --query-features ${f}"
    done

    local out_path="results/selection_reference/${out_subdir}/selection_reference.parquet"
    local model_path="results/selection_reference/${out_subdir}/statewise_selector.joblib"

    log "Building ${out_subdir}..."
    uv run selection-reference-build \
        --query-id "${QUERY_ID}" \
        ${train_files} \
        ${val_files} \
        ${beh_files} \
        ${qf_files} \
        --output "${out_path}" \
        --model-output "${model_path}" \
        --overwrite
}

# Query-adjusted selection reference
build_selection_reference "query" "results/action_losses/query_adjusted" "${QUERY_ID}"

# Behavior-only selection reference
build_selection_reference "behavior" "results/action_losses/behavior_only" "behavior_only_full_budget"

log "Selection references built."

# =============================================================================
# Done
# =============================================================================

step "FULL DATA GENERATION COMPLETE"
log "Outputs:"
log "  Trajectories:   results/phase1_refined_sampling/bbob_train/"
log "                   results/phase1_refined_sampling/bbob_validation/"
if [ "${SKIP_CEC2017}" != "1" ]; then
log "                   results/phase1_cec2017_test/"
fi
log "  Behavior:        results/phase1_refined_sampling/*/bbob_f*/dimension_*/behavior.parquet"
log "  Query samples:   results/landscape_queries/samples/${SAMPLE_DESIGN}/"
log "  Query features:  results/landscape_queries/features/${SAMPLE_DESIGN}/"
log "  Action losses:   results/action_losses/"
log "  Selection ref:  results/selection_reference/${QUERY_ID}/"
log "                   results/selection_reference/behavior_only_full_budget/"
log ""
log "Next steps:"
log "  1. Build utility labels (utility-labels-generate-batch)"
log "  2. Train decision model (decision-train-full)"
log "  3. Run pilot coverage check"
log "  4. External evaluation (decision-online-controller-evaluate)"
