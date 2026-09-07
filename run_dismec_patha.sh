#!/usr/bin/env bash
# Unified DiSMEC PTQ pipeline.  Usage: bash run_pipeline.sh <dataset|all>

set -Eeuo pipefail
IFS=$'\n\t'

BASE="${BASE:-/DATA2/rudra1/dismec_quantization/dismecpp}"
QUANT_DIR="${QUANT_DIR:-/DATA2/rudra1/dismec_quantization/quantization}"
PYTHON="${PYTHON:-python3}"
TOPK="${TOPK:-5}"
DEFAULT_THREADS="${THREADS:-20}"

PREDICT_BIN="$BASE/build/bin/predict"
QUANT_BIN="$QUANT_DIR/quant_infer_dismec_v2"
PATHB_BIN="$QUANT_DIR/quant_infer_dismec_v2_fixed"
QUANT_SCRIPT="$QUANT_DIR/run_quantize.py"
DEQUANT_SCRIPT="$QUANT_DIR/dequant_npz_to_weights.py"
EVALUATION_SCRIPT="$BASE/python/evaluation.py"

PATH_A_CONFIGS=(
  int8_row_sym int8_row_asym int8_row_sym_clip
  int8_group_sym int8_group_sym_clip int8_row_sym_clip_mixed
  int4_row_sym int4_row_asym int4_row_sym_clip
  int4_group_sym int4_group_sym_clip int4_row_sym_clip_mixed
)
PATH_B_CONFIGS=(
  int8_group_sym_clip_act int8_group_sym_clip_act_intinfer int8_group_sym_clip_act_intinfer_bias
  int4_group_sym_clip_act int4_group_sym_clip_act_intinfer int4_group_sym_clip_act_intinfer_bias
)
CONFIGS=("${PATH_A_CONFIGS[@]}" "${PATH_B_CONFIGS[@]}")
DATASETS=(eurlex wiki10 amazoncat13k amazon670k delicious200k wiki500k)
CSV_METRICS=(P@1 P@3 P@5 nDCG@1 nDCG@3 nDCG@5 PSP@1 PSP@3 PSP@5 PSnDCG@1 PSnDCG@3 PSnDCG@5)
CSV_HEADER=(dataset config path bits model_size_mb quant_time_s inference_time_s prediction_time_s fp32_inference_s speedup throughput "${CSV_METRICS[@]}")

DATASET="${1:-eurlex}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="" EXPERIMENT_DIR="" NPZ_DIR="" PRED_DIR="" TIMING_DIR="" METRIC_DIR="" TMP_DIR="" SUMMARY=""
WEIGHTS="" TEST="" ORIG_MODEL="" PROP_WEIGHTS="" NUM_FEATURES=0
LARGE=false CHUNK_SIZE=10000 ONE_BASED=false THREADS="$DEFAULT_THREADS" BATCH_SIZE=50000
FP32_INFER_S="" SAMPLE_COUNT=""

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
warn() { log "WARNING: $*" >&2; }
on_error() {
  local line=$1 command=$2 status=$3
  log "ERROR: command failed (exit $status) at line $line: $command" >&2
}
on_exit() {
  local status=$?
  cleanup || true
  return "$status"
}
trap 'on_error "$LINENO" "$BASH_COMMAND" "$?"' ERR
trap on_exit EXIT

config_uses_path_b() {
  local config=$1 candidate
  for candidate in "${PATH_B_CONFIGS[@]}"; do [[ $candidate == "$config" ]] && return 0; done
  return 1
}
config_path() { config_uses_path_b "$1" && printf B || printf A; }
config_bits() { [[ $1 == int4* ]] && printf '4' || printf '8'; }
file_size_mb() { [[ -f $1 ]] && "$PYTHON" -c 'import os,sys; print(f"{os.path.getsize(sys.argv[1])/1048576:.2f}")' "$1" || printf '0'; }
metric() { [[ -f $2 ]] && grep -oP "${1}:\s*\K[0-9]+\.[0-9]+" "$2" 2>/dev/null | head -n1 || true; }
config_is_complete() {
  local config=$1
  [[ -s "$NPZ_DIR/$config.npz" \
     && -s "$PRED_DIR/pred_v2_$config.txt" \
     && -s "$METRIC_DIR/eval_v2_$config.txt" \
     && -s "$TIMING_DIR/${config}_predict.txt" \
     && -n $(metric 'P@1' "$METRIC_DIR/eval_v2_$config.txt") ]]
}

reset_dataset_config() {
  NUM_FEATURES=0; LARGE=false; CHUNK_SIZE=10000; ONE_BASED=false
  THREADS="$DEFAULT_THREADS"; BATCH_SIZE=50000; FP32_INFER_S=""; SAMPLE_COUNT=""
}

configure_eurlex() {
  WEIGHTS="$BASE/eurlex_bow_baseline.model.weights-0-3992"; TEST="$BASE/data/eurlex/eurlex_test.txt"; ORIG_MODEL="$BASE/eurlex_bow_baseline.model"
  PROP_WEIGHTS="$BASE/python/eurlex-weights-test-pos.txt"; NUM_FEATURES=5002; ONE_BASED=true; SAMPLE_COUNT=3809; FP32_INFER_S=0.554
}
configure_wiki10() {
  WEIGHTS="$BASE/wiki10_bow_baseline.model.weights-0-30937"; TEST="$BASE/data/wiki10/test.txt"; ORIG_MODEL="$BASE/wiki10_bow_baseline.model"
  PROP_WEIGHTS="$BASE/python/wiki10-weights-test-pos.txt"; SAMPLE_COUNT=6616; FP32_INFER_S=5.707
}
configure_amazoncat13k() {
  WEIGHTS="$BASE/amazoncat13k_bow.model.weights-0-13329"; TEST="$BASE/data/amazoncat13k/test_amazoncat13k.txt"; ORIG_MODEL="$BASE/amazoncat13k_bow.model"
  PROP_WEIGHTS="$BASE/python/amazoncat13k-weights-test-pos.txt"; THREADS=64; BATCH_SIZE=10000; SAMPLE_COUNT=306782; FP32_INFER_S=28.318
}
configure_amazon670k() {
  WEIGHTS="$BASE/amazon670k_bow.model.weights-0-670090"; TEST="$BASE/data/amazon670k/Amazon670K_test.txt"; ORIG_MODEL="$BASE/amazon670k_bow.model"
  PROP_WEIGHTS="$BASE/python/amazon670k-weights-test-pos.txt"; THREADS=64; BATCH_SIZE=10000; SAMPLE_COUNT=153025; FP32_INFER_S=331.0
}
configure_delicious200k() {
  WEIGHTS="$BASE/delicious_bow_baseline.model.weights-0-205442"; TEST="$BASE/data/deliciouslarge/deliciousLarge_test.txt"; ORIG_MODEL="$BASE/delicious_bow_baseline.model"
  PROP_WEIGHTS="$BASE/python/delicious200k-weights-test-pos.txt"; NUM_FEATURES=782586; LARGE=true; CHUNK_SIZE=10000; THREADS=64; BATCH_SIZE=5000; SAMPLE_COUNT=100095; FP32_INFER_S=309.0
}
configure_wiki500k() {
  WEIGHTS="$BASE/wiki500k_bow.model.weights-0-501070"; TEST="$BASE/data/wiki500k/wiki500_test.txt"; ORIG_MODEL="$BASE/wiki500k_bow.model"
  PROP_WEIGHTS="$BASE/python/wiki500k-weights-test-pos.txt"; LARGE=true; CHUNK_SIZE=10000; ONE_BASED=true; THREADS=64; BATCH_SIZE=10000; SAMPLE_COUNT=769421
}

load_dataset_config() {
  reset_dataset_config
  [[ $EXPERIMENT_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid EXPERIMENT_NAME: $EXPERIMENT_NAME"
  case "$DATASET" in
    eurlex) configure_eurlex ;;
    wiki10) configure_wiki10 ;;
    amazoncat13k) configure_amazoncat13k ;;
    amazon670k) configure_amazon670k ;;
    delicious200k) configure_delicious200k ;;
    wiki500k) configure_wiki500k ;;
    *) die "Unknown dataset '$DATASET'. Valid datasets: ${DATASETS[*]}" ;;
  esac
  RESULTS_DIR="$BASE/results/$DATASET"
  NPZ_DIR="$RESULTS_DIR/models"
  EXPERIMENT_DIR="$RESULTS_DIR/experiments/$EXPERIMENT_NAME"
  PRED_DIR="$EXPERIMENT_DIR/predictions"
  TIMING_DIR="$EXPERIMENT_DIR/timing"
  METRIC_DIR="$EXPERIMENT_DIR/metrics"
  TMP_DIR="$EXPERIMENT_DIR/tmp_weights"
  SUMMARY="$EXPERIMENT_DIR/${DATASET}_dismec_results.csv"
}

verify_requirements() {
  local required=("$PREDICT_BIN" "$QUANT_BIN" "$PATHB_BIN" "$QUANT_SCRIPT" "$DEQUANT_SCRIPT" "$EVALUATION_SCRIPT" "$WEIGHTS" "$TEST" "$ORIG_MODEL") item
  for item in "${required[@]}"; do [[ -f $item ]] || die "Required file not found: $item"; done
  command -v "$PYTHON" >/dev/null 2>&1 || die "Python executable not found: $PYTHON"
  [[ -f $PROP_WEIGHTS ]] || warn "Propensity weights missing; PSP and PSnDCG will be skipped: $PROP_WEIGHTS"
}

create_directories() { mkdir -p "$NPZ_DIR" "$PRED_DIR" "$TIMING_DIR" "$METRIC_DIR" "$TMP_DIR"; }

strip_mode_key() {
  local npz=$1
  "$PYTHON" - "$npz" <<'PY'
import sys, numpy as np
path = sys.argv[1]
with np.load(path, allow_pickle=True) as archive:
    if 'mode' in archive.files:
        np.savez(path, **{key: archive[key] for key in archive.files if key != 'mode'})
PY
}

run_quantization() {
  local config npz log_file
  for config in "${CONFIGS[@]}"; do
    npz="$NPZ_DIR/$config.npz"; log_file="$TIMING_DIR/${config}_quant.txt"
    [[ -f $npz ]] && { log "Shared model reuse: $config already exists"; continue; }
    log "Quantizing $config"
    local args=("$PYTHON" "$QUANT_SCRIPT" --weights "$WEIGHTS" --test "$TEST" --out-dir "$NPZ_DIR" --config "$config")
    [[ $LARGE == true ]] && args+=(--chunked --chunk-size "$CHUNK_SIZE")
    { time "${args[@]}"; } 2>&1 | tee "$log_file"
    [[ -f $npz ]] || die "Quantization did not produce: $npz"
  done
  # cnpy used by the C++ inference binaries cannot read the metadata-only mode key.
  # Apply this also to NPZs produced by an earlier interrupted run.
  for config in "${CONFIGS[@]}"; do
    [[ -f "$NPZ_DIR/$config.npz" ]] && strip_mode_key "$NPZ_DIR/$config.npz"
  done
}

convert_prediction() {
  local raw=$1 final=$2
  "$PYTHON" - "$raw" "$final" "$TOPK" <<'PY'
import sys
raw, final, topk = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(raw) as source:
    lines = [line.strip() for line in source if line.strip()]
first = lines[0].split() if lines else []
start = 1 if len(first) == 2 and all(x.isdigit() for x in first) else 0
with open(final, 'w') as target:
    target.write(f'{len(lines)-start} {topk}\n')
    for line in lines[start:]:
        labels = line.split()[:topk]
        target.write(' '.join(label if ':' in label else f'{label}:{topk-i:.1f}' for i, label in enumerate(labels)) + '\n')
PY
}

run_evaluation() {
  local prediction=$1 evaluation=$2
  local args=("$PYTHON" "$EVALUATION_SCRIPT" --pred-path "$prediction" --data "$TEST")
  [[ -f $PROP_WEIGHTS ]] && args+=(--weights "$PROP_WEIGHTS")
  (cd "$BASE/python" && "${args[@]}") >"$evaluation" 2>&1
}

run_pathA() {
  local config=$1 npz="$NPZ_DIR/$1.npz" eval_file="$METRIC_DIR/eval_v2_$1.txt" weights="$TMP_DIR/$1.weights" model="$TMP_DIR/$1.model.json" raw="$TMP_DIR/${1}_raw_pred.txt" prediction="$PRED_DIR/pred_v2_$1.txt" predict_log="$TIMING_DIR/${1}_predict.txt"
  config_is_complete "$config" && { log "Path A resume: $config complete"; return; }
  [[ -f $npz ]] || { warn "Path A skip; NPZ missing: $npz"; return; }
  local one_based=(); [[ $ONE_BASED == true ]] && one_based=(--one-based)
  log "Path A dequantizing $config"
  { time "$PYTHON" "$DEQUANT_SCRIPT" --npz "$npz" --out "$weights" --model-json "$model" --original-model "$ORIG_MODEL" --num-features "$NUM_FEATURES" "${one_based[@]}"; } 2>&1 | tee "$TIMING_DIR/${config}_dequant.txt"
  [[ -f $weights && -f $model ]] || die "Dequantization failed for $config"
  log "Path A predicting $config"
  { time "$PREDICT_BIN" --augment-for-bias --normalize-instances --topk "$TOPK" --threads "$THREADS" "$TEST" "$model" "$raw"; } 2>&1 | tee "$predict_log"
  convert_prediction "$raw" "$prediction"
  run_evaluation "$prediction" "$eval_file"
  rm -f -- "$weights" "$model" "$raw"
}

run_pathB() {
  local config=$1 npz="$NPZ_DIR/$1.npz" eval_file="$METRIC_DIR/eval_v2_$1.txt" prediction="$PRED_DIR/pred_v2_$1.txt" timing="$TIMING_DIR/${1}_predict.txt"
  config_is_complete "$config" && { log "Path B resume: $config complete"; return; }
  [[ -f $npz ]] || { warn "Path B skip; NPZ missing: $npz"; return; }
  log "Path B direct activation-aware inference: $config"
  "$PATHB_BIN" "$npz" "$TEST" "$prediction" --topk "$TOPK" --nthreads "$THREADS" --batch_size "$BATCH_SIZE" --act_quant >"$timing" 2>&1
  grep -q '\[Timing\] inference=' "$timing" || die "Path B inference failed for $config; see $timing"
  run_evaluation "$prediction" "$eval_file"
}

quant_time_s() { grep -oP 'real\s+(?:(\d+)m)?([0-9.]+)s' "$1" 2>/dev/null | head -n1 | "$PYTHON" -c 'import re,sys; s=sys.stdin.read(); m=re.search(r"(?:(\d+)m)?([0-9.]+)s",s); print(round((int(m.group(1) or 0)*60)+float(m.group(2)),2) if m else "")' || true; }
path_a_inference_s() { "$PYTHON" - "$1" <<'PY'
import re, sys
from datetime import datetime
try: text=open(sys.argv[1]).read()
except OSError: print(''); raise SystemExit
prefix=r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\].*'
a=re.search(prefix + r'Calculating top-5',text); b=re.search(prefix + r'Finished prediction',text); load=re.search(r'read weight file.*?in (\d+)ms',text)
if not a or not b: print('')
else: print(round(max(0,(datetime.strptime(b.group(1),'%Y-%m-%d %H:%M:%S.%f')-datetime.strptime(a.group(1),'%Y-%m-%d %H:%M:%S.%f')).total_seconds()-(float(load.group(1)) if load else 0)/1000),4))
PY
}
path_b_inference_s() { grep '\[Timing\] inference=' "$1" 2>/dev/null | grep -oP '[\d.]+(?= sec)' | head -n1 || true; }
extract_timing() {
  local config=$1 q i; q=$(quant_time_s "$TIMING_DIR/${config}_quant.txt")
  if config_uses_path_b "$config"; then i=$(path_b_inference_s "$TIMING_DIR/${config}_predict.txt"); else i=$(path_a_inference_s "$TIMING_DIR/${config}_predict.txt"); fi
  printf '%s,%s' "$q" "$i"
}

write_csv() {
  local IFS=,
  printf '%s\n' "${CSV_HEADER[*]}" >"$SUMMARY"
  local config eval_file quant inference speedup throughput metric_name
  for config in "${CONFIGS[@]}"; do
    eval_file="$METRIC_DIR/eval_v2_${config}.txt"; [[ -f $eval_file ]] || continue
    read -r quant inference <<<"$(extract_timing "$config")"
    speedup=; throughput=
    if [[ -n $inference && -n $FP32_INFER_S ]]; then speedup=$("$PYTHON" -c "print(f'{$FP32_INFER_S/float(\"$inference\"):.3f}')"); fi
    if [[ -n $inference && -n $SAMPLE_COUNT ]]; then throughput=$("$PYTHON" -c "print(f'{$SAMPLE_COUNT/float(\"$inference\"):.0f}')"); fi
    local row=("$DATASET" "$config" "$(config_path "$config")" "$(config_bits "$config")" "$(file_size_mb "$NPZ_DIR/$config.npz")" "$quant" "$inference" "$inference" "$FP32_INFER_S" "$speedup" "$throughput")
    for metric_name in "${CSV_METRICS[@]}"; do row+=("$(metric "$metric_name" "$eval_file")"); done
    printf '%s\n' "${row[*]}" >>"$SUMMARY"
  done
}

verify_completion() {
  local config failed=false
  printf '%-45s %-5s %-8s %s\n' Config Path Status Missing_artifacts
  for config in "${CONFIGS[@]}"; do
    local missing=()
    [[ -s "$NPZ_DIR/$config.npz" ]] || missing+=(npz)
    [[ -s "$PRED_DIR/pred_v2_$config.txt" ]] || missing+=(prediction)
    [[ -s "$METRIC_DIR/eval_v2_$config.txt" ]] || missing+=(evaluation)
    [[ -s "$TIMING_DIR/${config}_predict.txt" ]] || missing+=(inference_log)
    [[ -n $(metric 'P@1' "$METRIC_DIR/eval_v2_$config.txt") ]] || missing+=(P@1)
    if ((${#missing[@]})); then
      printf '%-45s %-5s %-8s %s\n' "$config" "$(config_path "$config")" MISSING "${missing[*]}"
      failed=true
    else
      printf '%-45s %-5s %-8s %s\n' "$config" "$(config_path "$config")" OK '-'
    fi
  done
  if [[ $failed == true ]]; then
    return 1
  fi
  return 0
}

cleanup() { find "$TMP_DIR" -maxdepth 1 -type f \( -name '*.weights' -o -name '*.model.json' -o -name '*_raw_pred.txt' \) -delete; }

run_dataset() {
  load_dataset_config; verify_requirements; create_directories
  log "Starting DiSMEC PTQ pipeline: dataset=$DATASET experiment=$EXPERIMENT_NAME"
  run_quantization
  local config
  for config in "${CONFIGS[@]}"; do
    if config_uses_path_b "$config"; then run_pathB "$config"; else run_pathA "$config"; fi
  done
  write_csv; verify_completion
  log "Completed experiment. Results CSV: $SUMMARY"
}

main() {
  if [[ $DATASET == all ]]; then local dataset; for dataset in "${DATASETS[@]}"; do DATASET=$dataset; run_dataset; done; else run_dataset; fi
}
main "$@"
