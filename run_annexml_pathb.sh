#!/bin/bash
# run_annexml_pathb.sh
# ====================
# AnnexML PATH B — INT8 embeddings → AnnexML NGT inference
#
# Uses quant_annexml_intinfer_multibatch_v2 --act_quant:
#   - Keeps embeddings as INT8 (no dequantization)
#
# Usage:
#   bash run_annexml_pathb.sh wiki10
#   bash run_annexml_pathb.sh all

set -e
DATASET_ARG=${1:-wiki10}

BASE=/DATA2/rudra1/annexml_quantization
ANNEXML_BASE=$BASE/annexml
QUANT_DIR=$BASE/quantization
INFER_BIN=$QUANT_DIR/quant_annexml_intinfer_multibatch_v2
TOPK=5
NTHREADS=20

# PATH B configs (6 _act configs only)
CONFIGS_PB=(
    int8_group_sym_clip_act
    int8_group_sym_clip_act_intinfer
    int8_group_sym_clip_act_intinfer_bias
    int4_group_sym_clip_act
    int4_group_sym_clip_act_intinfer
    int4_group_sym_clip_act_intinfer_bias
)

log()        { echo "[$(date '+%H:%M:%S')] $*"; }
get_metric() { grep -oP "${1}:\s+\K[0-9]+\.[0-9]+" "$2" 2>/dev/null | head -1; }
get_bits()   { [[ "$1" == int4* ]] && echo 4 || echo 8; }
sep()        { echo "================================================================"; }

# ============================================================
# DATASET CONFIG
# ============================================================
run_dataset() {
    local DS=$1

    case "$DS" in
    eurlex)
        MODEL="$ANNEXML_BASE/annexml-model-eurlex_baseline.bin"
        TEST="$ANNEXML_BASE/data/eurlex/eurlex_test.txt"
        TRAIN="$ANNEXML_BASE/data/eurlex/eurlex_train.txt"
        NPZ_DIR="$ANNEXML_BASE/results/eurlex/models"
        A="0.55"; B="1.5"
        FP32_P1=84.78; FP32_TIME=0.202 ;;
    wiki10)
        MODEL="$ANNEXML_BASE/annexml-model-wiki10_baseline.bin"
        TEST="$ANNEXML_BASE/data/wiki10/test.txt"
        TRAIN="$ANNEXML_BASE/data/wiki10/train.txt"
        NPZ_DIR="$ANNEXML_BASE/results/wiki10/models"
        A="0.55"; B="1.5"
        FP32_P1=84.75; FP32_TIME=0.8837 ;;
    amazoncat13k)
        MODEL="$ANNEXML_BASE/annexml-model-amazoncat13k_baseline.bin"
        TEST="$ANNEXML_BASE/data/amazoncat13k/test_amazoncat13k.txt"
        TRAIN="$ANNEXML_BASE/data/amazoncat13k/train_amazoncat13k.txt"
        NPZ_DIR="$ANNEXML_BASE/results/amazoncat13k/models"
        A="0.55"; B="1.5"
        FP32_P1=92.42; FP32_TIME=22.006 ;;
    amazon670k)
        MODEL="$ANNEXML_BASE/annexml-model-amazon670k_baseline.bin"
        TEST="$ANNEXML_BASE/data/amazon670k/Amazon670K_test.txt"
        TRAIN="$ANNEXML_BASE/data/amazon670k/Amazon670K_train.txt"
        NPZ_DIR="$ANNEXML_BASE/results/amazon670k/models"
        A="0.6";  B="2.6"
        FP32_P1=43.23; FP32_TIME=10.8064 ;;
    delicious200k)
        MODEL="$ANNEXML_BASE/annexml-model-delicious200k_baseline.bin"
        TEST="$ANNEXML_BASE/data/deliciouslarge/deliciousLarge_test.txt"
        TRAIN="$ANNEXML_BASE/data/deliciouslarge/deliciousLarge_train.txt"
        NPZ_DIR="$ANNEXML_BASE/results/delicious200k/models"
        A="0.55"; B="1.5"
        FP32_P1=42.19; FP32_TIME=52.86 ;;
    amazon3m)
        MODEL="$ANNEXML_BASE/annexml-model-amazon3m_baseline.bin"
        TEST="$ANNEXML_BASE/data/amazon3m/test.txt"
        TRAIN="$ANNEXML_BASE/data/amazon3m/train.txt"
        NPZ_DIR="$ANNEXML_BASE/results/amazon3m/models"
        A="0.6";  B="2.6"
        # NOTE: FP32_P1 / FP32_TIME baseline values for amazon3m are not yet
        # recorded anywhere in the original script — fill these in before
        # running PATH B on amazon3m, or the drop/summary columns will be wrong.
        FP32_P1=0.00; FP32_TIME=0.000 ;;
    *)
        echo "Unknown dataset: $DS"
        echo "Valid: eurlex wiki10 amazoncat13k amazon670k delicious200k amazon3m"
        return 1 ;;
    esac

    OUT_DIR="$ANNEXML_BASE/results/$DS/pathb_ngt"
    PRED_DIR="$OUT_DIR/predictions"
    METRIC_DIR="$OUT_DIR/metrics"
    TIMING_DIR="$OUT_DIR/timing"
    mkdir -p "$PRED_DIR" "$METRIC_DIR" "$TIMING_DIR"

    CSV="$OUT_DIR/pathb_results.csv"
    echo "config,bits,infer_time_s,fp32_time_s,P@1,P@3,P@5,nDCG@1,nDCG@3,nDCG@5,P1_drop" \
        > "$CSV"

    sep
    log " AnnexML PATH B — $DS"
    sep
    log " Binary  : quant_annexml_intinfer_multibatch_v2 --act_quant"
    log " Approach: INT8 embeddings → AnnexML NGT (float graph)"
    log " Expected: accuracy PRESERVED — float NGT graph gives correct candidates, INT8xINT8 scoring preserves ranking"
    log " FP32 baseline: P@1=$FP32_P1  time=${FP32_TIME}s"
    log " Propensity   : A=$A  B=$B"
    sep

    [ ! -f "$INFER_BIN" ] && log "ERROR: $INFER_BIN not found" && return 1
    [ ! -f "$MODEL"     ] && log "ERROR: $MODEL not found: $MODEL" && return 1
    [ ! -f "$TEST"      ] && log "ERROR: $TEST not found" && return 1
    [ ! -f "$TRAIN"     ] && log "ERROR: $TRAIN not found: $TRAIN" && return 1

    for cfg in "${CONFIGS_PB[@]}"; do
        NPZ="$NPZ_DIR/${cfg}"  # directory of per-cluster NPZ files
        PRED="$PRED_DIR/${cfg}_pred.txt"
        MFILE="$METRIC_DIR/${cfg}_eval.txt"
        TFILE="$TIMING_DIR/${cfg}_timing.log"
        BITS=$(get_bits $cfg)

        echo ""; sep
        log "  [PATH B] ─── Config: $cfg ($DS) ───"
        log "  [PATH B]     Bits  : INT${BITS}"
        log "  [PATH B]     Step  : INT8 embeddings → NGT (float graph)"
        sep

        if [ -f "$MFILE" ] && [ -n "$(get_metric 'P@1' $MFILE)" ]; then
            log "  SKIP — already done (P@1=$(get_metric 'P@1' $MFILE))"
            continue
        fi

        [ ! -d "$NPZ" ] && log "  SKIP — NPZ dir not found: $NPZ" && continue

        # Run PATH B inference using same args as run_pipeline.sh
        log "  Running PATH B (INT8 → NGT)..."
        T0=$(date +%s%3N)
        (time $INFER_BIN \
            "$MODEL" \
            "$TEST" \
            "$PRED" \
            --npz_dir "$NPZ" \
            --topk $TOPK \
            --nthreads $NTHREADS \
            --act_quant \
        ) > "$TFILE" 2>&1 || { log "  ERROR: inference failed for $cfg ($DS) — see $TFILE"; continue; }
        T1=$(date +%s%3N)
        INFER_S=$(echo "scale=3; ($T1-$T0)/1000" | bc)
        echo "INFER_S=$INFER_S" >> "$TFILE"
        log "  Inference time: ${INFER_S}s"

        # Evaluate using same approach as run_pipeline.sh
        log "  Evaluating accuracy..."
        python3 $QUANT_DIR/evaluate_annexml.py \
            --pred  "$PRED" \
            --data  "$TEST" \
            --train "$TRAIN" \
            --topk  $TOPK \
            --A "$A" --B "$B" \
            > "$MFILE" 2>&1 || { log "  ERROR: evaluate_annexml.py failed for $cfg ($DS) — see $MFILE"; continue; }

        P1=$(get_metric "P@1" "$MFILE"); P3=$(get_metric "P@3" "$MFILE"); P5=$(get_metric "P@5" "$MFILE")
        N1=$(get_metric "nDCG@1" "$MFILE"); N3=$(get_metric "nDCG@3" "$MFILE"); N5=$(get_metric "nDCG@5" "$MFILE")
        D1=$(python3 -c "print(f'{$FP32_P1-float(\"${P1:-0}\"):.2f}')" 2>/dev/null || echo "N/A")

        echo ""
        log "  ┌─────────────────────────────────────────────┐"
        log "  │ [PATH B] RESULT: $cfg ($DS)"
        log "  │  Bits       : INT${BITS}"
        log "  │  Infer time : ${INFER_S}s  (FP32=${FP32_TIME}s)"
        log "  │  P@1        : $P1  (FP32=$FP32_P1  drop=$D1)"
        log "  │  P@3        : $P3"
        log "  │  P@5        : $P5"
        log "  │  nDCG@1/3/5 : $N1 / $N3 / $N5"
        log "  └─────────────────────────────────────────────┘"

        echo "$cfg,$BITS,$INFER_S,$FP32_TIME,$P1,$P3,$P5,$N1,$N3,$N5,$D1" >> "$CSV"
    done

    # Summary
    echo ""; sep
    echo " PATH B SUMMARY — $DS"
    sep
    printf "%-45s %4s %9s %7s %7s %7s %7s\n" \
        "Config" "Bits" "Infer(s)" "P@1" "P@3" "P@5" "P1_drop"
    printf "%-45s %4s %9s %7s %7s %7s %7s\n" \
        "------" "----" "--------" "---" "---" "---" "-------"
    printf "%-45s %4s %9s %7s %7s %7s %7s\n" \
        "FP32_baseline" "-" "${FP32_TIME}s" "$FP32_P1" "-" "-" "0.00"
    for cfg in "${CONFIGS_PB[@]}"; do
        MFILE="$METRIC_DIR/${cfg}_eval.txt"; [ ! -f "$MFILE" ] && continue
        TFILE="$TIMING_DIR/${cfg}_timing.log"
        P1=$(get_metric "P@1" "$MFILE"); P3=$(get_metric "P@3" "$MFILE"); P5=$(get_metric "P@5" "$MFILE")
        IS=$(grep "^INFER_S=" "$TFILE" 2>/dev/null | cut -d= -f2 || echo "N/A")
        D1=$(python3 -c "print(f'{$FP32_P1-float(\"${P1:-0}\"):.2f}')" 2>/dev/null || echo "N/A")
        printf "%-45s %4s %9s %7s %7s %7s %7s\n" \
            "$cfg" "$(get_bits $cfg)" "${IS}s" \
            "${P1:-N/A}" "${P3:-N/A}" "${P5:-N/A}" "$D1"
    done
    echo "CSV → $CSV"
    echo ""; sep; echo " DONE $DS — $(date)"; sep
}

# ============================================================
# MAIN
# ============================================================
ALL_DATASETS=(wiki10 eurlex amazoncat13k amazon670k delicious200k)

if [ "$DATASET_ARG" = "all" ]; then
    for DS in "${ALL_DATASETS[@]}"; do
        run_dataset "$DS"
    done
else
    run_dataset "$DATASET_ARG"
fi
