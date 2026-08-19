#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

usage() {
  cat <<'EOF'
Usage: run_action_losses.sh <50h|100h|both>

Environment overrides:
  MAX_BATCH_JOBS   Outer concurrency across tasks (default: 8)
  TASK_WORKERS     Inner workers passed to each evaluation task (default: 1)
  MAX_RETRIES      Automatic retries per task after the first failure (default: 1)
  OUTPUT_ROOT_BASE Base output directory (default: results/action_losses)
  LOG_ROOT_BASE    Base log directory (default: results/action_losses_logs)
EOF
}

PROFILE="${1:-${RUN_PROFILE:-}}"
case "$PROFILE" in
  50h|100h|both) ;;
  "")
    usage >&2
    exit 1
    ;;
  *)
    echo "Unknown profile '$PROFILE'; expected 50h, 100h, or both" >&2
    usage >&2
    exit 1
    ;;
esac

MAX_BATCH_JOBS="${MAX_BATCH_JOBS:-8}"
TASK_WORKERS="${TASK_WORKERS:-1}"
MAX_RETRIES="${MAX_RETRIES:-1}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-results/action_losses}"
LOG_ROOT_BASE="${LOG_ROOT_BASE:-results/action_losses_logs}"

BBOB_TRAIN_FUNCS=(1 2 3 4 6 7 8 10 11 12 15 16 17 18 20 21 22 23)
BBOB_VALID_FUNCS=(5 9 13 14 19 24)
DIMENSIONS=(10 20 40)

if ! [[ "$MAX_BATCH_JOBS" =~ ^[0-9]+$ ]] || (( MAX_BATCH_JOBS < 1 )); then
  echo "MAX_BATCH_JOBS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TASK_WORKERS" =~ ^[0-9]+$ ]] || (( TASK_WORKERS < 1 )); then
  echo "TASK_WORKERS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || (( MAX_RETRIES < 0 )); then
  echo "MAX_RETRIES must be a non-negative integer" >&2
  exit 1
fi

case "$PROFILE" in
  50h)  PROFILES=(50h) ;;
  100h) PROFILES=(100h) ;;
  both) PROFILES=(50h 100h) ;;
esac

SEM_FIFO="$(mktemp -u "${TMPDIR:-/tmp}/run_action_losses.sem.XXXXXX")"
mkfifo "$SEM_FIFO"
exec 9<>"$SEM_FIFO"
rm -f "$SEM_FIFO"
cleanup() {
  exec 9>&- 9<&-
}
trap cleanup EXIT

for ((i = 0; i < MAX_BATCH_JOBS; i++)); do
  printf '.' >&9
done

acquire_slot() {
  IFS= read -r -n 1 -u 9 _
}

release_slot() {
  printf '.' >&9
}

TASK_PIDS=()
TASK_LABELS=()
TASK_STATUS_FILES=()

run_task() {
  local profile="$1"
  local split_name="$2"
  local mode_name="$3"
  local config_path="$4"
  local train_config_path="$5"
  local function_id="$6"
  local dimension="$7"
  local action_budget_mode="$8"
  local sample_design_id="$9"
  local use_all_prefixes="${10}"
  local status_file="${11}"

  local func_tag
  func_tag=$(printf "f%03d" "$function_id")
  local dim_tag="d${dimension}"

  local out_dir="${OUTPUT_ROOT_BASE}/${profile}/${split_name}/${mode_name}/${func_tag}/${dim_tag}"
  local final_out="${out_dir}/action_losses.parquet"
  local log_file="${LOG_ROOT_BASE}/${profile}/${split_name}_${mode_name}_${func_tag}_${dim_tag}.log"

  mkdir -p "$out_dir"

  if [[ -f "$final_out" ]]; then
    echo "skipped" > "$status_file"
    echo "[skip] ${profile} ${split_name} ${mode_name} ${func_tag} ${dim_tag}"
    return 0
  fi

  local attempt
  local max_attempts=$((MAX_RETRIES + 1))
  for attempt in $(seq 1 "$max_attempts"); do
    local tmp_out="${out_dir}/action_losses.parquet.tmp.$$.${attempt}"
    rm -f "$tmp_out"

    echo "[run ] ${profile} ${split_name} ${mode_name} ${func_tag} ${dim_tag} (attempt ${attempt}/${max_attempts})"

    local -a cmd=(
      uv run selection-reference-evaluate-actions
      --config "$config_path"
      --train-config "$train_config_path"
      --action-budget-mode "$action_budget_mode"
      --only-function "$function_id"
      --only-dimension "$dimension"
      --output "$tmp_out"
      --workers "$TASK_WORKERS"
      --overwrite
    )

    if [[ -n "$sample_design_id" ]]; then
      cmd+=(--sample-design-id "$sample_design_id")
    fi
    if [[ "$use_all_prefixes" == "yes" ]]; then
      cmd+=(--all-prefixes)
    fi

    if "${cmd[@]}" >"$log_file" 2>&1; then
      mv "$tmp_out" "$final_out"
      if (( attempt > 1 )); then
        echo "ok:retried" > "$status_file"
        echo "[ok ] ${profile} ${split_name} ${mode_name} ${func_tag} ${dim_tag} (retried)"
      else
        echo "ok" > "$status_file"
        echo "[ok ] ${profile} ${split_name} ${mode_name} ${func_tag} ${dim_tag}"
      fi
      return 0
    fi

    rm -f "$tmp_out"
    if (( attempt < max_attempts )); then
      echo "[retry] ${profile} ${split_name} ${mode_name} ${func_tag} ${dim_tag}" >&2
      sleep 2
    fi
  done

  echo "fail" > "$status_file"
  echo "[fail] ${profile} ${split_name} ${mode_name} ${func_tag} ${dim_tag}; see $log_file" >&2
  return 1
}

launch_task() {
  local label="$1"
  shift
  local status_file="$1"
  shift

  acquire_slot
  (
    trap 'release_slot' EXIT
    "$@"
  ) &

  TASK_PIDS+=("$!")
  TASK_LABELS+=("$label")
  TASK_STATUS_FILES+=("$status_file")
}

submit_batch() {
  local profile="$1"
  local split_name="$2"
  local mode_name="$3"
  local config_path="$4"
  local train_config_path="$5"
  local action_budget_mode="$6"
  local sample_design_id="$7"
  local use_all_prefixes="$8"
  shift 8
  local funcs=("$@")

  local function_id
  local dimension
  for function_id in "${funcs[@]}"; do
    for dimension in "${DIMENSIONS[@]}"; do
      local func_tag
      func_tag=$(printf "f%03d" "$function_id")
      local dim_tag="d${dimension}"
      local status_file="${LOG_ROOT_BASE}/${profile}/.${split_name}_${mode_name}_${func_tag}_${dim_tag}.status"
      mkdir -p "$(dirname "$status_file")"
      rm -f "$status_file"
      launch_task \
        "${profile}/${split_name}/${mode_name}/${func_tag}/${dim_tag}" \
        "$status_file" \
        run_task \
        "$profile" \
        "$split_name" \
        "$mode_name" \
        "$config_path" \
        "$train_config_path" \
        "$function_id" \
        "$dimension" \
        "$action_budget_mode" \
        "$sample_design_id" \
        "$use_all_prefixes" \
        "$status_file"
    done
  done
}

submit_profile() {
  local profile="$1"

  mkdir -p "${OUTPUT_ROOT_BASE}/${profile}" "${LOG_ROOT_BASE}/${profile}"
  echo "[info] submitting profile=${profile} outer_parallel=${MAX_BATCH_JOBS} task_workers=${TASK_WORKERS} retries=${MAX_RETRIES}"

  submit_batch "$profile" "bbob_train" "query_adjusted" "configs/phase1_bbob_train.yaml" "configs/phase1_bbob_train.yaml" "query_adjusted_budget" "lhs_50d" "yes" "${BBOB_TRAIN_FUNCS[@]}"
  submit_batch "$profile" "bbob_validation" "query_adjusted" "configs/phase1_bbob_validation.yaml" "configs/phase1_bbob_train.yaml" "query_adjusted_budget" "lhs_50d" "yes" "${BBOB_VALID_FUNCS[@]}"

  submit_batch "$profile" "bbob_train" "behavior_only" "configs/phase1_bbob_train.yaml" "configs/phase1_bbob_train.yaml" "behavior_only_full_budget" "" "yes" "${BBOB_TRAIN_FUNCS[@]}"
  submit_batch "$profile" "bbob_validation" "behavior_only" "configs/phase1_bbob_validation.yaml" "configs/phase1_bbob_train.yaml" "behavior_only_full_budget" "" "yes" "${BBOB_VALID_FUNCS[@]}"

  submit_batch "$profile" "bbob_train" "pre_run" "configs/phase1_bbob_train.yaml" "configs/phase1_bbob_train.yaml" "pre_run_query_adjusted_budget" "lhs_50d" "no" "${BBOB_TRAIN_FUNCS[@]}"
  submit_batch "$profile" "bbob_validation" "pre_run" "configs/phase1_bbob_validation.yaml" "configs/phase1_bbob_train.yaml" "pre_run_query_adjusted_budget" "lhs_50d" "no" "${BBOB_VALID_FUNCS[@]}"
}

START_EPOCH=$(date +%s)
for profile in "${PROFILES[@]}"; do
  submit_profile "$profile"
done

# Wait for all tasks and collect summary.
total=${#TASK_PIDS[@]}
succeeded=0
retried_ok=0
skipped=0
failed=0
failed_labels=()

for i in "${!TASK_PIDS[@]}"; do
  pid="${TASK_PIDS[$i]}"
  label="${TASK_LABELS[$i]}"
  status_file="${TASK_STATUS_FILES[$i]}"

  if wait "$pid"; then
    if [[ -f "$status_file" ]]; then
      status_content="$(tr -d '\n' < "$status_file")"
      case "$status_content" in
        ok)
          succeeded=$((succeeded + 1))
          ;;
        ok:retried)
          succeeded=$((succeeded + 1))
          retried_ok=$((retried_ok + 1))
          ;;
        skipped)
          skipped=$((skipped + 1))
          ;;
        fail)
          failed=$((failed + 1))
          failed_labels+=("$label")
          ;;
        *)
          succeeded=$((succeeded + 1))
          ;;
      esac
    else
      succeeded=$((succeeded + 1))
    fi
  else
    failed=$((failed + 1))
    failed_labels+=("$label")
  fi

  rm -f "$status_file"
done

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

printf '\n'
echo "════════════════════════════════════════"
echo "[summary] profiles          = ${PROFILES[*]}"
echo "[summary] total tasks       = ${total}"
echo "[summary] succeeded         = ${succeeded}"
echo "[summary] retried-success   = ${retried_ok}"
echo "[summary] skipped           = ${skipped}"
echo "[summary] failed            = ${failed}"
echo "[summary] outer concurrency  = ${MAX_BATCH_JOBS}"
echo "[summary] task workers       = ${TASK_WORKERS}"
echo "[summary] retries per task   = ${MAX_RETRIES}"
echo "[summary] elapsed            = ${ELAPSED_MIN}m${ELAPSED_SEC}s"
echo "[summary] output_root        = ${OUTPUT_ROOT_BASE}"
echo "[summary] log_root           = ${LOG_ROOT_BASE}"
echo "════════════════════════════════════════"

if (( failed > 0 )); then
  echo ""
  echo "Failed tasks:" >&2
  for label in "${failed_labels[@]}"; do
    echo "  - ${label}" >&2
  done
  exit 1
fi

echo "[done] all batches completed successfully"
