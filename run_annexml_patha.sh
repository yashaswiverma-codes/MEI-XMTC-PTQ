#!/bin/bash
# run_pipeline_annexml.sh
# =======================
# AnnexML PTQ Pipeline — all datasets
#
# Steps:
#   1. Quantize → NPZ files (run_quantized_annexml.py --skip_inference)
#   2. PATH A: Float query inference (no --act_quant)
#      Configs: row_sym, row_asym, group_sym, group_sym_clip, row_sym_clip_mixed
#   3. PATH B: INT8 query inference (--act_quant)
#      Configs: group_sym_clip_act, group_sym_clip_act_intinfer,
#               group_sym_clip_act_intinfer_bias
#   4. Evaluate → P@k, nDCG@k, PSP@k, PSnDCG@k
#   5. Summary CSV with accuracy + quant time + inference time
#
# Timing:
#   quant_time_s: extracted from timing/<config>_quant.txt  (real Xm Y.Zs)
#   infer_time_s: extracted from timing/<config>_infer.txt  ([Timing] inference=X sec)
#
# Usage:
#   bash run_pipeline_annexml.sh eurlex
#   bash run_pipeline_annexml.sh wiki10
#   bash run_pipeline_annexml.sh amazoncat13k
#   bash run_pipeline_annexml.sh amazon670k
#   bash run_pipeline_annexml.sh delicious200k
#   bash run_pipeline_annexml.sh amazon3m

set -e

DATASET=${1:-eurlex}

# ============================================================
# PATHS — update if needed
# ============================================================
BASE="/DATA2/rudra1/annexml_quantization/annexml"
QUANT_DIR="/DATA2/rudra1/annexml_quantization/quantization"
PYTHON="python3"
THREADS=20
TOPK=5

INFER_BIN="$QUANT_DIR/quant_annexml_intinfer_multibatch_v2"
EVAL_SCRIPT="$QUANT_DIR/evaluate_annexml.py"
QUANT_SCRIPT="$QUANT_DIR/run_quantized_annexml.py"

# ============================================================
# DATASET CONFIG
# A and B: propensity parameters
# ============================================================
case "$DATASET" in
eurlex)
    MODEL="$BASE/annexml-model-eurlex_baseline.bin"
    TEST="$BASE/data/eurlex/eurlex_test.txt"
    TRAIN="$BASE/data/eurlex/eurlex_train.txt"
    A="0.55"; B="1.5" ;;
wiki10)
    MODEL="$BASE/annexml-model-wiki10_baseline.bin"
    TEST="$BASE/data/wiki10/test.txt"
    TRAIN="$BASE/data/wiki10/train.txt"
    A="0.55"; B="1.5" ;;
amazoncat13k)
    MODEL="$BASE/annexml-model-amazoncat13k_baseline.bin"
    TEST="$BASE/data/amazoncat13k/test_amazoncat13k.txt"
    TRAIN="$BASE/data/amazoncat13k/train_amazoncat13k.txt"
    A="0.55"; B="1.5" ;;
amazon670k)
    MODEL="$BASE/annexml-model-amazon670k_baseline.bin"
    TEST="$BASE/data/amazon670k/Amazon670K_test.txt"
    TRAIN="$BASE/data/amazon670k/Amazon670K_train.txt"
    A="0.6";  B="2.6" ;;
delicious200k)
    MODEL="$BASE/annexml-model-delicious200k_baseline.bin"
    TEST="$BASE/data/deliciouslarge/deliciousLarge_test.txt"
    TRAIN="$BASE/data/deliciouslarge/deliciousLarge_train.txt"
    A="0.55"; B="1.5" ;;
amazon3m)
    MODEL="$BASE/annexml-model-amazon3m_baseline.bin"
    TEST="$BASE/data/amazon3m/test.txt"
    TRAIN="$BASE/data/amazon3m/train.txt"
    A="0.6";  B="2.6" ;;
*)
    echo "Unknown dataset: $DATASET"
    echo "Valid: eurlex wiki10 amazoncat13k amazon670k delicious200k amazon3m"
    exit 1 ;;
esac

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
OUT_DIR="$BASE/results/$DATASET"
NPZ_DIR="$OUT_DIR/models"
PRED_DIR="$OUT_DIR/predictions"
TIMING_DIR="$OUT_DIR/timing"
METRIC_DIR="$OUT_DIR/metrics"

mkdir -p "$NPZ_DIR" "$PRED_DIR" "$TIMING_DIR" "$METRIC_DIR"

echo "========================================================"
echo " AnnexML PTQ Pipeline — $DATASET"
echo " A=$A  B=$B"
echo " Started: $(date)"
echo "========================================================"

# ============================================================
# VERIFY PATHS
# ============================================================
[ ! -f "$INFER_BIN" ]    && echo "ERROR: infer binary missing: $INFER_BIN" \
                         && echo "       Build with: g++ -O3 -march=native -funroll-loops -ffast-math -mavx2 -mfma -fopenmp -std=c++17 -I../annexml/include $QUANT_DIR/quant_annexml_intinfer_multibatch_v2.cpp -lz -lopenblas -o $INFER_BIN" \
                         && exit 1
[ ! -f "$EVAL_SCRIPT" ]  && echo "ERROR: eval script missing: $EVAL_SCRIPT"  && exit 1
[ ! -f "$QUANT_SCRIPT" ] && echo "ERROR: quant script missing: $QUANT_SCRIPT" && exit 1
[ ! -f "$MODEL" ]        && echo "ERROR: model missing: $MODEL"               && exit 1
[ ! -f "$TEST" ]         && echo "ERROR: test missing: $TEST"                 && exit 1
[ ! -f "$TRAIN" ]        && echo "ERROR: train missing: $TRAIN"               && exit 1
echo "Paths verified ✓"
echo "Model:  $MODEL"
echo "Test:   $TEST"
echo "Train:  $TRAIN"

# ============================================================
# ALL 18 CONFIGS
# PATH A = no --act_quant (float query, dequantize embedding on-the-fly)
# PATH B = --act_quant    (INT8 query, integer dot product)
# ============================================================

# PATH A configs (12 configs — float query)
CONFIGS_PA=(
    "int8_row_sym"
    "int8_row_asym"
    "int8_row_sym_clip"
    "int8_group_sym"
    "int8_group_sym_clip"
    "int8_row_sym_clip_mixed"
    "int4_row_sym"
    "int4_row_asym"
    "int4_row_sym_clip"
    "int4_group_sym"
    "int4_group_sym_clip"
    "int4_row_sym_clip_mixed"
)

# PATH B configs (6 configs — INT8 query)
CONFIGS_PB=(
    "int8_group_sym_clip_act"
    "int8_group_sym_clip_act_intinfer"
    "int8_group_sym_clip_act_intinfer_bias"
    "int4_group_sym_clip_act"
    "int4_group_sym_clip_act_intinfer"
    "int4_group_sym_clip_act_intinfer_bias"
)

# Full ordered list for CSV
ALL_CONFIGS=(
    "int8_row_sym"    "int8_row_asym"    "int8_row_sym_clip"
    "int8_group_sym"  "int8_group_sym_clip" "int8_row_sym_clip_mixed"
    "int8_group_sym_clip_act" "int8_group_sym_clip_act_intinfer"
    "int8_group_sym_clip_act_intinfer_bias"
    "int4_row_sym"    "int4_row_asym"    "int4_row_sym_clip"
    "int4_group_sym"  "int4_group_sym_clip" "int4_row_sym_clip_mixed"
    "int4_group_sym_clip_act" "int4_group_sym_clip_act_intinfer"
    "int4_group_sym_clip_act_intinfer_bias"
)

# Quantization args per config
declare -A QUANT_ARGS
QUANT_ARGS["int8_row_sym"]="--bits 8 --quant_mode sym"
QUANT_ARGS["int8_row_asym"]="--bits 8 --quant_mode asym"
QUANT_ARGS["int8_row_sym_clip"]="--bits 8 --quant_mode sym --clip_pct 99.99"
QUANT_ARGS["int8_group_sym"]="--bits 8 --quant_mode group --group_size 32"
QUANT_ARGS["int8_group_sym_clip"]="--bits 8 --quant_mode group --group_size 32 --clip_pct 99.99"
QUANT_ARGS["int8_row_sym_clip_mixed"]="--bits 8 --quant_mode sym --clip_pct 99.99 --mixed"
QUANT_ARGS["int8_group_sym_clip_act"]="--bits 8 --quant_mode group --group_size 32 --clip_pct 99.99 --act_quant"
QUANT_ARGS["int8_group_sym_clip_act_intinfer"]="--bits 8 --quant_mode group --group_size 32 --clip_pct 99.99 --act_quant --intinfer"
QUANT_ARGS["int8_group_sym_clip_act_intinfer_bias"]="--bits 8 --quant_mode group --group_size 32 --clip_pct 99.99 --act_quant --intinfer --bias_correction"
QUANT_ARGS["int4_row_sym"]="--bits 4 --quant_mode sym"
QUANT_ARGS["int4_row_asym"]="--bits 4 --quant_mode asym"
QUANT_ARGS["int4_row_sym_clip"]="--bits 4 --quant_mode sym --clip_pct 99.99"
QUANT_ARGS["int4_group_sym"]="--bits 4 --quant_mode group --group_size 32"
QUANT_ARGS["int4_group_sym_clip"]="--bits 4 --quant_mode group --group_size 32 --clip_pct 99.99"
QUANT_ARGS["int4_row_sym_clip_mixed"]="--bits 4 --quant_mode sym --clip_pct 99.99 --mixed"
QUANT_ARGS["int4_group_sym_clip_act"]="--bits 4 --quant_mode group --group_size 32 --clip_pct 99.99 --act_quant"
QUANT_ARGS["int4_group_sym_clip_act_intinfer"]="--bits 4 --quant_mode group --group_size 32 --clip_pct 99.99 --act_quant --intinfer"
QUANT_ARGS["int4_group_sym_clip_act_intinfer_bias"]="--bits 4 --quant_mode group --group_size 32 --clip_pct 99.99 --act_quant --intinfer --bias_correction"

# ============================================================
# HELPERS
# ============================================================
get_metric() { grep -oP "${1}:\s*\K[0-9]+\.[0-9]+" "$2" 2>/dev/null | head -1; }

get_npz_mb() {
    local DIR="$1"
    [ -d "$DIR" ] && $PYTHON -c "
import os
total = sum(os.path.getsize(os.path.join('$DIR', f))
            for f in os.listdir('$DIR') if f.endswith('.npz'))
print(f'{total/1024/1024:.2f}')
" || echo "0"
}

get_quant_time_s() {
    local LOG="$1"
    [ ! -f "$LOG" ] && echo "" && return
    $PYTHON - << PYEOF
import re
line = open('$LOG').read()
m = re.search(r'real\s+(\d+)m([\d.]+)s', line)
if m:
    print(round(int(m.group(1))*60 + float(m.group(2)), 2))
else:
    print("")
PYEOF
}

get_infer_time_s() {
    local LOG="$1"
    [ ! -f "$LOG" ] && echo "" && return
    $PYTHON - << PYEOF
import re
line = open('$LOG').read()
m = re.search(r'\[Timing\]\s+inference=([\d.]+)\s+sec', line)
if m:
    print(round(float(m.group(1)), 4))
else:
    print("")
PYEOF
}

npz_exists() {
    local DIR="$1"
    [ -d "$DIR" ] && [ "$(ls $DIR/*.npz 2>/dev/null | wc -l)" -gt "0" ]
}

# ============================================================
# PART 1 — QUANTIZE ALL 18 CONFIGS
# Runs run_quantized_annexml.py with --skip_inference
# Saves NPZ files to $NPZ_DIR/<config>/learner*_cluster*.npz
# ============================================================
echo ""
echo "========================================================"
echo " PART 1: Quantization → NPZ"
echo "========================================================"

for config in "${ALL_CONFIGS[@]}"; do
    NPZ_CONFIG_DIR="$NPZ_DIR/$config"
    QUANT_LOG="$TIMING_DIR/${config}_quant.txt"

    if npz_exists "$NPZ_CONFIG_DIR"; then
        echo "  SKIP $config (NPZ exists)"
        continue
    fi

    echo ""
    echo "  Quantizing: $config"
    echo "  Args: ${QUANT_ARGS[$config]}"

    (time $PYTHON $QUANT_SCRIPT \
        --model      "$MODEL" \
        --test       "$TEST" \
        --train      "$TRAIN" \
        --out_dir    "$OUT_DIR" \
        --out_prefix "$config" \
        --topk       $TOPK \
        --A          "$A" \
        --B          "$B" \
        --save_npz \
        --skip_inference \
        ${QUANT_ARGS[$config]} \
    ) 2>&1 | tee "$QUANT_LOG"

    if npz_exists "$NPZ_CONFIG_DIR"; then
        echo "  ✓ Done in $(get_quant_time_s $QUANT_LOG)s"
    else
        echo "  ✗ FAILED: NPZ not found at $NPZ_CONFIG_DIR"
    fi
done

# ============================================================
# PART 2A — PATH A INFERENCE (float query, 12 configs)
# z stays float32 after projection + normalization
# Embeddings dequantized on-the-fly in C++ binary
# score = z_float · dequant(e_int8)
# ============================================================
echo ""
echo "========================================================"
echo " PART 2A: PATH A — Float Query Inference (12 configs)"
echo "          z = float32, embeddings dequantized on-the-fly"
echo "========================================================"

for config in "${CONFIGS_PA[@]}"; do
    NPZ_CONFIG_DIR="$NPZ_DIR/$config"
    PRED_FILE="$PRED_DIR/pred_${config}.txt"
    EVAL_FILE="$METRIC_DIR/eval_${config}.txt"
    INFER_LOG="$TIMING_DIR/${config}_infer.txt"

    echo ""
    echo "--- [PATH A] $config ---"

    if [ -f "$EVAL_FILE" ]; then
        P1=$(get_metric "P@1" "$EVAL_FILE")
        if [ -n "$P1" ] && [ "$(echo "$P1 > 0" | bc -l 2>/dev/null)" = "1" ]; then
            echo "  SKIP (P@1=$P1)"
            continue
        fi
        rm -f "$EVAL_FILE"
    fi

    if ! npz_exists "$NPZ_CONFIG_DIR"; then
        echo "  SKIP (NPZ missing)"
        continue
    fi

    # PATH A: NO --act_quant flag
    echo "  Running PATH A inference (float query)..."
    (time $INFER_BIN \
        "$MODEL" \
        "$TEST" \
        "$PRED_FILE" \
        --npz_dir  "$NPZ_CONFIG_DIR" \
        --topk     $TOPK \
        --nthreads $THREADS \
    ) 2>&1 | tee "$INFER_LOG"
    echo -n "  Infer time: "; get_infer_time_s "$INFER_LOG"

    [ ! -f "$PRED_FILE" ] && echo "  SKIP eval (no prediction)" && continue

    echo "  Evaluating..."
    $PYTHON $EVAL_SCRIPT \
        --pred  "$PRED_FILE" \
        --data  "$TEST" \
        --train "$TRAIN" \
        --topk  $TOPK \
        --A     "$A" \
        --B     "$B" \
        > "$EVAL_FILE" 2>&1

    echo -n "  P@1="; get_metric "P@1" "$EVAL_FILE"
    echo -n "  P@3="; get_metric "P@3" "$EVAL_FILE"
    echo -n "  P@5="; get_metric "P@5" "$EVAL_FILE"
done

# ============================================================
# PART 2B — PATH B INFERENCE (INT8 query, 6 configs)
# z quantized to INT8 after projection + normalization
# Integer dot product with INT8 embeddings in C++ binary
# score = (z_int8 · e_int8) × act_scale × emb_scale
# ============================================================
echo ""
echo "========================================================"
echo " PART 2B: PATH B — INT8 Query Inference (6 configs)"
echo "          z = INT8, integer dot product scoring"
echo "========================================================"

for config in "${CONFIGS_PB[@]}"; do
    NPZ_CONFIG_DIR="$NPZ_DIR/$config"
    PRED_FILE="$PRED_DIR/pred_${config}.txt"
    EVAL_FILE="$METRIC_DIR/eval_${config}.txt"
    INFER_LOG="$TIMING_DIR/${config}_infer.txt"

    echo ""
    echo "--- [PATH B] $config ---"

    if [ -f "$EVAL_FILE" ]; then
        P1=$(get_metric "P@1" "$EVAL_FILE")
        if [ -n "$P1" ] && [ "$(echo "$P1 > 0" | bc -l 2>/dev/null)" = "1" ]; then
            echo "  SKIP (P@1=$P1)"
            continue
        fi
        rm -f "$EVAL_FILE"
    fi

    if ! npz_exists "$NPZ_CONFIG_DIR"; then
        echo "  SKIP (NPZ missing)"
        continue
    fi

    # PATH B: --act_quant flag — INT8 query
    echo "  Running PATH B inference (INT8 query)..."
    (time $INFER_BIN \
        "$MODEL" \
        "$TEST" \
        "$PRED_FILE" \
        --npz_dir  "$NPZ_CONFIG_DIR" \
        --topk     $TOPK \
        --nthreads $THREADS \
        --act_quant \
    ) 2>&1 | tee "$INFER_LOG"
    echo -n "  Infer time: "; get_infer_time_s "$INFER_LOG"

    [ ! -f "$PRED_FILE" ] && echo "  SKIP eval (no prediction)" && continue

    echo "  Evaluating..."
    $PYTHON $EVAL_SCRIPT \
        --pred  "$PRED_FILE" \
        --data  "$TEST" \
        --train "$TRAIN" \
        --topk  $TOPK \
        --A     "$A" \
        --B     "$B" \
        > "$EVAL_FILE" 2>&1

    echo -n "  P@1="; get_metric "P@1" "$EVAL_FILE"
    echo -n "  P@3="; get_metric "P@3" "$EVAL_FILE"
    echo -n "  P@5="; get_metric "P@5" "$EVAL_FILE"
done

# ============================================================
# PART 3 — SUMMARY CSV
# Columns:
#   config, bits, npz_mb,
#   quant_time_s,   ← timing/<config>_quant.txt  (real Xm Y.Zs)
#   infer_time_s,   ← timing/<config>_infer.txt  ([Timing] inference=X sec)
#   P@1..PSnDCG@5
# ============================================================
echo ""
echo "========================================================"
echo " PART 3: Summary CSV"
echo "========================================================"

SUMMARY="$OUT_DIR/${DATASET}_annexml_results.csv"
echo "config,bits,npz_mb,quant_time_s,infer_time_s,P@1,P@3,P@5,nDCG@1,nDCG@3,nDCG@5,PSP@1,PSP@3,PSP@5,PSnDCG@1,PSnDCG@3,PSnDCG@5" \
    > "$SUMMARY"

for config in "${ALL_CONFIGS[@]}"; do
    EVAL_FILE="$METRIC_DIR/eval_${config}.txt"
    [ ! -f "$EVAL_FILE" ] && continue

    BITS=8; [[ "$config" == int4* ]] && BITS=4
    SZ=$(get_npz_mb  "$NPZ_DIR/$config")
    QT=$(get_quant_time_s "$TIMING_DIR/${config}_quant.txt")
    IT=$(get_infer_time_s  "$TIMING_DIR/${config}_infer.txt")

    P1=$(get_metric "P@1"      "$EVAL_FILE"); P3=$(get_metric "P@3"      "$EVAL_FILE"); P5=$(get_metric "P@5"      "$EVAL_FILE")
    N1=$(get_metric "nDCG@1"   "$EVAL_FILE"); N3=$(get_metric "nDCG@3"   "$EVAL_FILE"); N5=$(get_metric "nDCG@5"   "$EVAL_FILE")
    SP1=$(get_metric "PSP@1"   "$EVAL_FILE"); SP3=$(get_metric "PSP@3"   "$EVAL_FILE"); SP5=$(get_metric "PSP@5"   "$EVAL_FILE")
    SN1=$(get_metric "PSnDCG@1" "$EVAL_FILE"); SN3=$(get_metric "PSnDCG@3" "$EVAL_FILE"); SN5=$(get_metric "PSnDCG@5" "$EVAL_FILE")

    echo "${config},${BITS},${SZ},${QT},${IT},${P1},${P3},${P5},${N1},${N3},${N5},${SP1},${SP3},${SP5},${SN1},${SN3},${SN5}" \
        >> "$SUMMARY"
done

# ============================================================
# VERIFICATION
# ============================================================
echo ""
echo "========================================================"
echo " VERIFICATION"
echo "========================================================"
all_ok=true
printf "%-45s %-6s %-8s %-12s %-12s %-8s %-6s\n" "Config" "Bits" "Path" "Quant(s)" "Infer(s)" "P@1" "Status"
printf "%-45s %-6s %-8s %-12s %-12s %-8s %-6s\n" "------" "----" "----" "--------" "--------" "---" "------"

for config in "${ALL_CONFIGS[@]}"; do
    EVAL_FILE="$METRIC_DIR/eval_${config}.txt"
    p1=$(get_metric "P@1" "$EVAL_FILE")
    BITS=8; [[ "$config" == int4* ]] && BITS=4
    PATH_TYPE="A (float)"
    [[ "$config" == *"_act"* ]] && PATH_TYPE="B (INT8)"
    QT=$(get_quant_time_s "$TIMING_DIR/${config}_quant.txt")
    IT=$(get_infer_time_s  "$TIMING_DIR/${config}_infer.txt")
    if [ -f "$EVAL_FILE" ] && [ -n "$p1" ]; then
        STATUS="✓"
    else
        STATUS="✗ MISSING"
        all_ok=false
    fi
    printf "%-45s %-6s %-8s %-12s %-12s %-8s %-6s\n" \
        "$config" "$BITS" "$PATH_TYPE" "${QT:-N/A}" "${IT:-N/A}" "${p1:-N/A}" "$STATUS"
done

echo ""
[ "$all_ok" = true ] && echo "ALL 18 COMPLETE ✓" || echo "SOME MISSING — rerun script"
echo ""
echo "CSV: $SUMMARY"
echo ""
echo "========================================================"
echo " FINAL SUMMARY"
echo "========================================================"
cat "$SUMMARY"
echo ""
echo "ALL DONE — $(date)"
