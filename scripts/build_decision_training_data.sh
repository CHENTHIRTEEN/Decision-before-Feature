#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUERY_ID="${QUERY_ID:-descriptor_cheap_invariant}"
SAMPLE_DESIGN_ID="${SAMPLE_DESIGN_ID:-lhs_50d}"
WORKERS="${WORKERS:-32}"
OVERWRITE="${OVERWRITE:-0}"
CONFIG_TRAIN="${CONFIG_TRAIN:-configs/phase1_bbob_train.yaml}"
CONFIG_VALIDATION="${CONFIG_VALIDATION:-configs/phase1_bbob_validation.yaml}"
BASE_RESULTS="${BASE_RESULTS:-results}"

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${BASE_RESULTS}/phase1_refined_sampling}"
SAMPLES_ROOT="${SAMPLES_ROOT:-${BASE_RESULTS}/landscape_queries/samples/${SAMPLE_DESIGN_ID}}"
FEATURE_ROOT="${FEATURE_ROOT:-${BASE_RESULTS}/landscape_queries/features/${QUERY_ID}}"
SELECTION_ROOT="${SELECTION_ROOT:-${BASE_RESULTS}/selection_reference/${QUERY_ID}}"
UTILITY_ROOT="${UTILITY_ROOT:-${BASE_RESULTS}/utility_labels/${QUERY_ID}}"
DECISION_ROOT="${DECISION_ROOT:-${BASE_RESULTS}/decision/${QUERY_ID}}"
MATERIALIZED_DIR="$DECISION_ROOT/materialized_training_data"

ACTION_LOSS_TRAIN="$SELECTION_ROOT/action_losses_train.parquet"
ACTION_LOSS_VALIDATION="$SELECTION_ROOT/action_losses_validation.parquet"
TRAIN_SAMPLE_PATH="$SAMPLES_ROOT/bbob_train/samples.parquet"
VALIDATION_SAMPLE_PATH="$SAMPLES_ROOT/bbob_validation/samples.parquet"
TRAIN_FEATURE_PATH="$FEATURE_ROOT/bbob_train/features.parquet"
VALIDATION_FEATURE_PATH="$FEATURE_ROOT/bbob_validation/features.parquet"

if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  OVERWRITE_FLAG="--overwrite"
else
  OVERWRITE_FLAG=""
fi

mkdir -p "$SELECTION_ROOT" "$UTILITY_ROOT" "$DECISION_ROOT" "$SAMPLES_ROOT" "$FEATURE_ROOT"

run_py() {
  uv run python -m "$@"
}

echo "[0/8] Check that the frozen protocols are available"
run_py decision.check_model_protocol >/dev/null
run_py landscape_queries.consistency >/dev/null 2>&1 || true

echo "[1/8] Generate query samples for train and validation"
run_py landscape_queries.batch_sampling \
  --config "$CONFIG_TRAIN" \
  --config "$CONFIG_VALIDATION" \
  --sample-design-id "$SAMPLE_DESIGN_ID" \
  ${OVERWRITE_FLAG}

if [[ ! -f "$TRAIN_SAMPLE_PATH" || ! -f "$VALIDATION_SAMPLE_PATH" ]]; then
  echo "Expected sample files were not produced:" >&2
  echo "  $TRAIN_SAMPLE_PATH" >&2
  echo "  $VALIDATION_SAMPLE_PATH" >&2
  exit 1
fi

echo "[2/8] Extract query features for train and validation"
run_py landscape_queries.batch_features \
  --samples "$TRAIN_SAMPLE_PATH" \
  ${OVERWRITE_FLAG}
run_py landscape_queries.batch_features \
  --samples "$VALIDATION_SAMPLE_PATH" \
  ${OVERWRITE_FLAG}

if [[ ! -f "$TRAIN_FEATURE_PATH" || ! -f "$VALIDATION_FEATURE_PATH" ]]; then
  echo "Expected query feature files were not produced:" >&2
  echo "  $TRAIN_FEATURE_PATH" >&2
  echo "  $VALIDATION_FEATURE_PATH" >&2
  exit 1
fi

echo "[3/8] Extract behavior shards for train and validation"
run_py behavior.batch_extraction \
  --config "$CONFIG_TRAIN" \
  --config "$CONFIG_VALIDATION" \
  --workers "$WORKERS" \
  ${OVERWRITE_FLAG}

echo "[4/8] Generate selection-reference action losses for train and validation"
run_py selection_reference.action_losses \
  --config "$CONFIG_TRAIN" \
  --train-config "$CONFIG_TRAIN" \
  --sample-design-id "$SAMPLE_DESIGN_ID" \
  --output "$ACTION_LOSS_TRAIN" \
  --all-prefixes \
  ${OVERWRITE_FLAG}
run_py selection_reference.action_losses \
  --config "$CONFIG_VALIDATION" \
  --train-config "$CONFIG_TRAIN" \
  --sample-design-id "$SAMPLE_DESIGN_ID" \
  --output "$ACTION_LOSS_VALIDATION" \
  --all-prefixes \
  ${OVERWRITE_FLAG}

if [[ ! -f "$ACTION_LOSS_TRAIN" || ! -f "$ACTION_LOSS_VALIDATION" ]]; then
  echo "Expected action-loss files were not produced:" >&2
  echo "  $ACTION_LOSS_TRAIN" >&2
  echo "  $ACTION_LOSS_VALIDATION" >&2
  exit 1
fi

echo "[5/8] Build the selection reference and selector model"
run_py selection_reference.build \
  --query-id "$QUERY_ID" \
  --train-action-losses "$ACTION_LOSS_TRAIN" \
  --predict-action-losses "$ACTION_LOSS_VALIDATION" \
  --behavior "$BEHAVIOR_ROOT" \
  --query-features "$FEATURE_ROOT" \
  --output "$SELECTION_ROOT/selection_reference.parquet" \
  --model-output "$SELECTION_ROOT/statewise_selector.joblib"

echo "[6/8] Generate utility labels for train and validation"
run_py utility_labels.batch_generation \
  --query-id "$QUERY_ID" \
  --config "$CONFIG_TRAIN" \
  --config "$CONFIG_VALIDATION" \
  --selection-reference "$SELECTION_ROOT/selection_reference.parquet" \
  --output-root "$UTILITY_ROOT" \
  --report-dir "$UTILITY_ROOT/quality" \
  --workers "$WORKERS" \
  ${OVERWRITE_FLAG}

echo "[7/8] Materialize Decision training data"
run_py decision.materialize_training_data \
  --query-id "$QUERY_ID" \
  --utility-root "$UTILITY_ROOT" \
  --behavior-root "$BEHAVIOR_ROOT" \
  --output-dir "$MATERIALIZED_DIR" \
  --expected-utility-shards 72 \
  --expected-behavior-shards 72 \
  ${OVERWRITE_FLAG}

echo "[8/8] Check the Decision model protocol"
run_py decision.check_model_protocol \
  --training-dir "$MATERIALIZED_DIR"

echo "Decision training-data generation completed successfully."
