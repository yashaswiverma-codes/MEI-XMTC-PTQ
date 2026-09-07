#!/bin/bash
# run_inference_timing_stage4.sh
# =======================
# Same methodology as runs_inference_timing of  (3 seeds x {100,500,1000} for larger datasets {amazoncat13k, amazon670k, deliciouslarge200k}
#
# Usage:
#   bash run_inference_timing_stage4.sh amazoncat13k
#   bash run_inference_timing_stage4.sh all   (all 6, but eurlex/wiki10
#                                               already covered separately
#                                               today via the per-config
#                                               one-by-one runs)
DATASET_ARG=${1:-amazoncat13k}
BASE_DIR=${2:-/DATA2/rudra1/dismec_quantization/dismecpp}
THREADS=20
TOPK=5
SEEDS=(42 123 7)
SAMPLE_SIZES=(100 500 1000)
PYTHON=python3
PREDICT_BIN="$BASE_DIR/build/bin/predict"
QUANT_DIR="/DATA2/rudra1/dismec_quantization/quantization"
DEQUANT_SCRIPT="$QUANT_DIR/dequant_npz_to_weights.py"
# THE SUBSTITUTION: our verified Stage 4 kernel instead of _fixed.
QUANT_BIN="$QUANT_DIR/quant_infer_dismec_v2_stage4_sparsequant"
# NEW output directory -- avoids stale-cache collision with the earlier
# run_inference_timing.sh output (which used a different Path B binary).
OUT_DIR="$BASE_DIR/results/inference_timing_stage4"
mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
[ ! -f "$PREDICT_BIN" ]    && log "WARN: predict binary not found: $PREDICT_BIN"
[ ! -f "$QUANT_BIN" ]      && log "WARN: quant_infer binary not found: $QUANT_BIN"
[ ! -f "$DEQUANT_SCRIPT" ] && log "WARN: dequant script not found: $DEQUANT_SCRIPT"
COMBINED_CSV="$OUT_DIR/inference_timing_all.csv"
HEADER="dataset,config,bits,path"
for SEED in "${SEEDS[@]}"; do
    for N in "${SAMPLE_SIZES[@]}"; do
        HEADER="${HEADER},fp32_seed${SEED}_n${N}_s"
    done
done
for SEED in "${SEEDS[@]}"; do
    for N in "${SAMPLE_SIZES[@]}"; do
        HEADER="${HEADER},quant_seed${SEED}_n${N}_s"
    done
done
HEADER="${HEADER},fp32_full_s,quant_full_s"
setup_dataset() {
    case "$1" in
    eurlex)
        TEST="$BASE_DIR/data/eurlex/eurlex_test.txt"
        MODEL="$BASE_DIR/eurlex_bow_baseline.model"
        AUG="--augment-for-bias"; LARGE=false
        NUM_FEATURES=5002; ONE_BASED=true ;;
    wiki10)
        TEST="$BASE_DIR/data/wiki10/test.txt"
        MODEL="$BASE_DIR/wiki10_bow_baseline.model"
        AUG="--augment-for-bias"; LARGE=false
        NUM_FEATURES=0; ONE_BASED=true ;;
    amazoncat13k)
        TEST="$BASE_DIR/data/amazoncat13k/test_amazoncat13k.txt"
        MODEL="$BASE_DIR/amazoncat13k_bow.model"
        AUG="--augment-for-bias"; LARGE=true
        NUM_FEATURES=0; ONE_BASED=true ;;
    amazon670k)
        TEST="$BASE_DIR/data/amazon670k/Amazon670K_test.txt"
        MODEL="$BASE_DIR/amazon670k_bow.model"
        AUG="--augment-for-bias"; LARGE=true
        NUM_FEATURES=0; ONE_BASED=true ;;
    delicious200k)
        TEST="$BASE_DIR/data/deliciouslarge/deliciousLarge_test.txt"
        MODEL="$BASE_DIR/delicious_bow_baseline.model"
        AUG="--augment-for-bias"; LARGE=true
        NUM_FEATURES=782586; ONE_BASED=false ;;
    amazon3m)
        TEST="$BASE_DIR/data/amazon3m/test.txt"
        MODEL="$BASE_DIR/amazon3m_combined.model"
        AUG="--augment-for-bias"; LARGE=true
        NUM_FEATURES=337068; ONE_BASED=false ;;
    *)
        echo "Unknown dataset: $1"; exit 1 ;;
    esac
    NPZ_DIR="$BASE_DIR/results/$1/models"
    DS_DIR="$OUT_DIR/$1"
    SAMPLE_DIR="$DS_DIR/samples"
    CSV="$DS_DIR/${1}_inference_timing.csv"
    mkdir -p "$SAMPLE_DIR"
}
get_test_info() {
    N_FULL=$(awk 'NR==1{print $1}' "$TEST")
    NF=$(awk 'NR==1{print $2}' "$TEST")
    NL=$(awk 'NR==1{print $3}' "$TEST")
}
get_fp32_pure_time() {
    $PYTHON - << PYEOF
import re
from datetime import datetime
try:
    lines = open('$1').read()
except: print(0); exit()
fmt = '%Y-%m-%d %H:%M:%S.%f'
t1 = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\].*Calculating top-5', lines)
t2 = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\].*Finished prediction', lines)
wt = re.search(r'read weight file.*?in (\d+)ms', lines)
if not t1 or not t2: print(0)
else:
    total = (datetime.strptime(t2.group(1), fmt) - datetime.strptime(t1.group(1), fmt)).total_seconds()
    wt_ms = float(wt.group(1)) if wt else 0.0
    print(round(max(0.0, total - wt_ms/1000.0), 4))
PYEOF
}
get_quant_time() {
    grep "\[Timing\] inference=" "$1" 2>/dev/null | grep -oP '[\d.]+(?= sec)' | head -1
}
strip_mode_key() {
    local NPZ="$1"
    $PYTHON - << PYEOF
import numpy as np
d = np.load('$NPZ', allow_pickle=True)
if 'mode' in d.files:
    data = {k: d[k] for k in d.files if k != 'mode'}
    np.savez('${NPZ%.npz}', **data)
PYEOF
}
dequant_config() {
    local NPZ="$1" WEIGHTS_OUT="$2" MODEL_JSON_OUT="$3"
    [ -f "$WEIGHTS_OUT" ] && [ -f "$MODEL_JSON_OUT" ] && return
    strip_mode_key "$NPZ"
    local ONE_BASED_FLAG=""
    [ "$ONE_BASED" = "true" ] && ONE_BASED_FLAG="--one-based"
    $PYTHON "$DEQUANT_SCRIPT" \
        --npz "$NPZ" \
        --out "$WEIGHTS_OUT" \
        --model-json "$MODEL_JSON_OUT" \
        --original-model "$MODEL" \
        --num-features "$NUM_FEATURES" \
        $ONE_BASED_FLAG \
        > "${WEIGHTS_OUT}.dequant.log" 2>&1
}
make_sample() {
    local SEED="$1" N="$2" DST="$3"
    [ -f "$DST" ] && return
    echo "$N $NF $NL" > "$DST"
    tail -n +2 "$TEST" | $PYTHON -c "
import sys, random
random.seed($SEED)
lines = sys.stdin.readlines()
sys.stdout.writelines(random.sample(lines, min($N, len(lines))))
" >> "$DST"
}
run_dataset_timing() {
    local DATASET="$1"
    setup_dataset "$DATASET"
    echo ""
    echo "========================================================"
    echo " Dataset: $DATASET  (Path B binary: quant_infer_dismec_v2_stage4_sparsequant)"
    echo "========================================================"
    [ ! -f "$TEST" ]  && echo "ERROR: $TEST not found"  && return
    [ ! -f "$MODEL" ] && echo "ERROR: $MODEL not found" && return
    get_test_info
    echo "  N=$N_FULL  NF=$NF  NL=$NL"
    echo "  Mode: $([ "$LARGE" = true ] && echo 'subsampling (no extrapolation)' || echo 'full test set')"
    echo "$HEADER" > "$CSV"
    FP32_FULL=""
    declare -A FP32_TIMES
    FP32_CACHE_DIR="$DS_DIR/fp32_cache"
    mkdir -p "$FP32_CACHE_DIR"
    echo ""; echo "  --- FP32 Baseline ---"
    if [ "$LARGE" = false ]; then
        CACHE="$FP32_CACHE_DIR/full.cache"
        if [ -f "$CACHE" ]; then
            FP32_FULL=$(cat "$CACHE"); echo "  CACHED: ${FP32_FULL}s"
        else
            LOG="$FP32_CACHE_DIR/full.log"
            $PREDICT_BIN $AUG --normalize-instances \
                --topk $TOPK --threads $THREADS \
                "$TEST" "$MODEL" "$FP32_CACHE_DIR/pred.txt" > "$LOG" 2>&1
            FP32_FULL=$(get_fp32_pure_time "$LOG")
            rm -f "$FP32_CACHE_DIR/pred.txt"
            echo "  Full: ${FP32_FULL}s"
            echo "$FP32_FULL" > "$CACHE"
        fi
    else
        for SEED in "${SEEDS[@]}"; do
            for N in "${SAMPLE_SIZES[@]}"; do
                KEY="${SEED}_${N}"
                CACHE="$FP32_CACHE_DIR/seed${SEED}_n${N}.cache"
                DST="$SAMPLE_DIR/fp32_seed${SEED}_n${N}.txt"
                make_sample "$SEED" "$N" "$DST"
                if [ -f "$CACHE" ]; then
                    FP32_TIMES[$KEY]=$(cat "$CACHE")
                    echo "  CACHED seed=$SEED n=$N: ${FP32_TIMES[$KEY]}s"
                else
                    LOG="$FP32_CACHE_DIR/seed${SEED}_n${N}.log"
                    $PREDICT_BIN $AUG --normalize-instances \
                        --topk $TOPK --threads $THREADS \
                        "$DST" "$MODEL" "$FP32_CACHE_DIR/pred_s.txt" > "$LOG" 2>&1
                    T=$(get_fp32_pure_time "$LOG")
                    rm -f "$FP32_CACHE_DIR/pred_s.txt"
                    FP32_TIMES[$KEY]="$T"
                    echo "  seed=$SEED n=$N: ${T}s"
                    echo "$T" > "$CACHE"
                fi
            done
        done
    fi
    CONFIGS_PA=(
        "int8_row_sym" "int8_row_asym" "int8_row_sym_clip"
        "int8_group_sym" "int8_group_sym_clip" "int8_row_sym_clip_mixed"
        "int4_row_sym" "int4_row_asym" "int4_row_sym_clip"
        "int4_group_sym" "int4_group_sym_clip" "int4_row_sym_clip_mixed"
    )
    CONFIGS_PB=(
        "int8_group_sym_clip_act" "int8_group_sym_clip_act_intinfer"
        "int8_group_sym_clip_act_intinfer_bias"
        "int4_group_sym_clip_act" "int4_group_sym_clip_act_intinfer"
        "int4_group_sym_clip_act_intinfer_bias"
    )
    time_config_pathA() {
        local CONFIG="$1"
        local NPZ="$NPZ_DIR/${CONFIG}.npz"
        local BITS=8; [[ "$CONFIG" == int4* ]] && BITS=4
        local QCACHE="$DS_DIR/quant_cache/${CONFIG}"
        mkdir -p "$QCACHE"
        local WEIGHTS_FILE="$QCACHE/${CONFIG}.weights"
        local MODEL_JSON="$QCACHE/${CONFIG}.model.json"
        dequant_config "$NPZ" "$WEIGHTS_FILE" "$MODEL_JSON"
        if [ ! -f "$WEIGHTS_FILE" ] || [ ! -f "$MODEL_JSON" ]; then
            echo "  FAIL $CONFIG — dequant failed, see ${WEIGHTS_FILE}.dequant.log"
            return
        fi
        declare -A QUANT_TIMES
        QUANT_FULL=""
        if [ "$LARGE" = false ]; then
            CACHE="$QCACHE/full.cache"
            if [ -f "$CACHE" ]; then
                QUANT_FULL=$(cat "$CACHE"); echo "  CACHED: ${QUANT_FULL}s"
            else
                LOG="$QCACHE/full.log"
                $PREDICT_BIN $AUG --normalize-instances \
                    --topk $TOPK --threads $THREADS \
                    "$TEST" "$MODEL_JSON" "$QCACHE/pred.txt" > "$LOG" 2>&1
                QUANT_FULL=$(get_fp32_pure_time "$LOG")
                rm -f "$QCACHE/pred.txt"
                echo "  Full: ${QUANT_FULL}s"
                echo "$QUANT_FULL" > "$CACHE"
            fi
        else
            for SEED in "${SEEDS[@]}"; do
                for N in "${SAMPLE_SIZES[@]}"; do
                    KEY="${SEED}_${N}"
                    CACHE="$QCACHE/seed${SEED}_n${N}.cache"
                    DST="$SAMPLE_DIR/fp32_seed${SEED}_n${N}.txt"
                    make_sample "$SEED" "$N" "$DST"
                    if [ -f "$CACHE" ]; then
                        QUANT_TIMES[$KEY]=$(cat "$CACHE")
                        echo "  CACHED seed=$SEED n=$N: ${QUANT_TIMES[$KEY]}s"
                    else
                        LOG="$QCACHE/seed${SEED}_n${N}.log"
                        $PREDICT_BIN $AUG --normalize-instances \
                            --topk $TOPK --threads $THREADS \
                            "$DST" "$MODEL_JSON" "$QCACHE/pred_s.txt" > "$LOG" 2>&1
                        T=$(get_fp32_pure_time "$LOG")
                        rm -f "$QCACHE/pred_s.txt"
                        QUANT_TIMES[$KEY]="$T"
                        echo "  seed=$SEED n=$N: ${T}s"
                        echo "$T" > "$CACHE"
                    fi
                done
            done
        fi
        write_row "$CONFIG" "$BITS" "A"
    }
    time_config_pathB() {
        local CONFIG="$1"
        local NPZ="$NPZ_DIR/${CONFIG}.npz"
        local BITS=8; [[ "$CONFIG" == int4* ]] && BITS=4
        local QCACHE="$DS_DIR/quant_cache/${CONFIG}"
        mkdir -p "$QCACHE"
        [ ! -f "$NPZ" ] && echo "  SKIP $CONFIG — NPZ missing" && return
        echo ""; echo "  $CONFIG (PATH B, Stage 4 kernel)..."
        declare -A QUANT_TIMES
        QUANT_FULL=""
        if [ "$LARGE" = false ]; then
            CACHE="$QCACHE/full.cache"
            if [ -f "$CACHE" ]; then
                QUANT_FULL=$(cat "$CACHE"); echo "  CACHED: ${QUANT_FULL}s"
            else
                LOG="$QCACHE/full.log"
                $QUANT_BIN "$NPZ" "$TEST" "$QCACHE/pred.txt" \
                    --topk $TOPK --nthreads $THREADS \
                    --batch_size 5000 --act_quant > "$LOG" 2>&1
                QUANT_FULL=$(get_quant_time "$LOG")
                rm -f "$QCACHE/pred.txt"
                echo "  Full: ${QUANT_FULL}s"
                echo "$QUANT_FULL" > "$CACHE"
            fi
        else
            for SEED in "${SEEDS[@]}"; do
                for N in "${SAMPLE_SIZES[@]}"; do
                    KEY="${SEED}_${N}"
                    CACHE="$QCACHE/seed${SEED}_n${N}.cache"
                    DST="$SAMPLE_DIR/fp32_seed${SEED}_n${N}.txt"
                    make_sample "$SEED" "$N" "$DST"
                    if [ -f "$CACHE" ]; then
                        QUANT_TIMES[$KEY]=$(cat "$CACHE")
                        echo "  CACHED seed=$SEED n=$N: ${QUANT_TIMES[$KEY]}s"
                    else
                        LOG="$QCACHE/seed${SEED}_n${N}.log"
                        $QUANT_BIN "$NPZ" "$DST" "$QCACHE/pred_s.txt" \
                            --topk $TOPK --nthreads $THREADS \
                            --batch_size 5000 --act_quant > "$LOG" 2>&1
                        T=$(get_quant_time "$LOG")
                        rm -f "$QCACHE/pred_s.txt"
                        QUANT_TIMES[$KEY]="$T"
                        echo "  seed=$SEED n=$N: ${T}s"
                        echo "$T" > "$CACHE"
                    fi
                done
            done
        fi
        write_row "$CONFIG" "$BITS" "B"
    }
    write_row() {
        local CONFIG="$1" BITS="$2" PATH_TYPE="$3"
        local ROW="$DATASET,$CONFIG,$BITS,$PATH_TYPE"
        for SEED in "${SEEDS[@]}"; do
            for N in "${SAMPLE_SIZES[@]}"; do
                KEY="${SEED}_${N}"
                ROW="${ROW},${FP32_TIMES[$KEY]:-$FP32_FULL}"
            done
        done
        for SEED in "${SEEDS[@]}"; do
            for N in "${SAMPLE_SIZES[@]}"; do
                KEY="${SEED}_${N}"
                ROW="${ROW},${QUANT_TIMES[$KEY]:-$QUANT_FULL}"
            done
        done
        ROW="${ROW},${FP32_FULL},${QUANT_FULL}"
        echo "$ROW" >> "$CSV"
    }
    echo ""; echo "  --- PATH A timing (dequant -> predict, unchanged) ---"
    for CONFIG in "${CONFIGS_PA[@]}"; do time_config_pathA "$CONFIG"; done
    echo ""; echo "  --- PATH B timing (Stage 4 kernel, --act_quant) ---"
    for CONFIG in "${CONFIGS_PB[@]}"; do time_config_pathB "$CONFIG"; done
    echo ""; echo "  CSV: $CSV"
}
ALL_DATASETS=(eurlex wiki10 amazoncat13k amazon670k delicious200k)
if [ "$DATASET_ARG" = "all" ]; then
    for DS in "${ALL_DATASETS[@]}"; do run_dataset_timing "$DS"; done
else
    run_dataset_timing "$DATASET_ARG"
fi
echo "$HEADER" > "$COMBINED_CSV"
for DS in "${ALL_DATASETS[@]}"; do
    F="$OUT_DIR/$DS/${DS}_inference_timing.csv"
    [ -f "$F" ] && tail -n +2 "$F" >> "$COMBINED_CSV"
done
echo ""
echo "========================================================"
echo " RESULTS LAYOUT"
echo "========================================================"
echo " Combined (all datasets): $COMBINED_CSV"
for DS in "${ALL_DATASETS[@]}"; do
    F="$OUT_DIR/$DS/${DS}_inference_timing.csv"
    if [ -f "$F" ]; then
        ROWS=$(($(wc -l < "$F") - 1))
        echo " $DS: $F  ($ROWS config rows)"
    fi
done
echo "========================================================"
echo ""
cat "$COMBINED_CSV"
echo "ALL DONE — $(date)"
