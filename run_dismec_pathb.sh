#!/bin/bash
# run_pathb_dismec.sh
# ===================
# DiSMEC PATH B: Direct INT8xINT8 Inference Benchmark
# Uses quant_infer_dismec_v2 --act_quant
#
# Usage:
#   bash run_pathb_dismec.sh eurlex
#   bash run_pathb_dismec.sh all

set -e

DATASET_ARG=${1:-eurlex}

# ============================================================
# CONFIG
# ============================================================
BASE="/DATA2/rudra1/dismec_quantization/dismecpp"
QUANT_DIR="/DATA2/rudra1/dismec_quantization/quantization"
BINARY="$QUANT_DIR/quant_infer_dismec_v2_corrected"
RESULTS_DIR="/DATA2/rudra1/dismec_quantization/results/speedup"
TOPK=5
PYTHON="python3"

mkdir -p $RESULTS_DIR

# PATH B configs only (act_quant=1 configs)
CONFIGS=(
    int8_group_sym_clip_act_intinfer_bias
    int4_group_sym_clip_act
    int4_group_sym_clip_act_intinfer
    int4_group_sym_clip_act_intinfer_bias
)

# ============================================================
# DATASET CONFIG
# ============================================================
declare -A DS_MODELS DS_TEST DS_PROP DS_FP32 DS_SAMPLES DS_THREADS DS_BATCH DS_FP32_P1

DS_MODELS[eurlex]=$BASE/results/eurlex/models
DS_TEST[eurlex]=$BASE/data/eurlex/eurlex_test.txt
DS_PROP[eurlex]=$BASE/python/eurlex-weights-test-pos.txt
DS_FP32[eurlex]=0.554
DS_SAMPLES[eurlex]=3809
DS_THREADS[eurlex]=20
DS_BATCH[eurlex]=50000
DS_FP32_P1[eurlex]=83.93

DS_MODELS[wiki10]=$BASE/results/wiki10/models
DS_TEST[wiki10]=$BASE/data/wiki10/test.txt
DS_PROP[wiki10]=$BASE/python/wiki10-weights-test-pos.txt
DS_FP32[wiki10]=5.707
DS_SAMPLES[wiki10]=6616
DS_THREADS[wiki10]=20
DS_BATCH[wiki10]=50000
DS_FP32_P1[wiki10]=84.75

DS_MODELS[amazoncat13k]=$BASE/results/amazoncat13k/models
DS_TEST[amazoncat13k]=$BASE/data/amazoncat13k/test_amazoncat13k.txt
DS_PROP[amazoncat13k]=$BASE/python/amazoncat13k-weights-test-pos.txt
DS_FP32[amazoncat13k]=28.318
DS_SAMPLES[amazoncat13k]=306782
DS_THREADS[amazoncat13k]=64
DS_BATCH[amazoncat13k]=10000
DS_FP32_P1[amazoncat13k]=93.23

DS_MODELS[amazon670k]=$BASE/results/amazon670k/models
DS_TEST[amazon670k]=$BASE/data/amazon670k/Amazon670K_test.txt
DS_PROP[amazon670k]=$BASE/python/amazon670k-weights-test-pos.txt
DS_FP32[amazon670k]=331.0
DS_SAMPLES[amazon670k]=153025
DS_THREADS[amazon670k]=64
DS_BATCH[amazon670k]=10000
DS_FP32_P1[amazon670k]=45.66

DS_MODELS[delicious200k]=$BASE/results/delicious200k/models
DS_TEST[delicious200k]=$BASE/data/deliciouslarge/deliciousLarge_test.txt
DS_PROP[delicious200k]=$BASE/python/delicious200k-weights-test-pos.txt
DS_FP32[delicious200k]=309.0
DS_SAMPLES[delicious200k]=100095
DS_THREADS[delicious200k]=64
DS_BATCH[delicious200k]=5000
DS_FP32_P1[delicious200k]=46.46

log() { echo "[$(date '+%H:%M:%S')] $*"; }
get_metric() { grep -oP "${1}:\s+\K[0-9]+\.[0-9]+" "$2" 2>/dev/null | head -1; }

# ============================================================
# RUN ONE DATASET
# ============================================================
run_dataset() {
    local DS=$1
    local DS_OUT=$RESULTS_DIR/$DS
    local DS_PRED=$DS_OUT/pathb_predictions
    local DS_METRICS=$DS_OUT/pathb_metrics
    local DS_TIMING=$DS_OUT/pathb_timing

    mkdir -p $DS_PRED $DS_METRICS $DS_TIMING

    local SUMMARY=$DS_OUT/pathb_summary.txt
    local CSV=$DS_OUT/pathb_results.csv

    log "======== Starting PATH B: $DS ========"
    log "FP32 baseline: ${DS_FP32[$DS]}s | FP32 P@1: ${DS_FP32_P1[$DS]}"
    log "Threads: ${DS_THREADS[$DS]} | Batch: ${DS_BATCH[$DS]}"

    cat > $SUMMARY << HDR
=================================================================
 DiSMEC PATH B (INT8xINT8) Results — $DS
 Binary: $BINARY --act_quant
 FP32 baseline: ${DS_FP32[$DS]}s  |  FP32 P@1: ${DS_FP32_P1[$DS]}
 Threads: ${DS_THREADS[$DS]} | Batch: ${DS_BATCH[$DS]}
 Started: $(date)
=================================================================
HDR
    printf "%-45s %10s %10s %8s %6s %8s %8s %8s\n" \
        "config" "infer(s)" "fp32(s)" "speedup" "status" "P@1" "P@3" "P@5" >> $SUMMARY
    printf "%-45s %10s %10s %8s %6s %8s %8s %8s\n" \
        "------" "--------" "-------" "-------" "------" "---" "---" "---" >> $SUMMARY

    echo "dataset,config,bits,infer_s,fp32_s,speedup,beats_fp32,throughput,P@1,P@3,P@5,nDCG@1,nDCG@3,nDCG@5,PSP@1,PSP@3,PSP@5,PSnDCG@1,PSnDCG@3,PSnDCG@5" > $CSV

    for CONFIG in "${CONFIGS[@]}"; do
        NPZ="${DS_MODELS[$DS]}/${CONFIG}.npz"
        PRED_FILE=$DS_PRED/${CONFIG}_pred.txt
        METRIC_FILE=$DS_METRICS/${CONFIG}_eval.txt
        TIMING_FILE=$DS_TIMING/${CONFIG}_timing.txt

        log "  Running $CONFIG..."

        if [ ! -f "$NPZ" ]; then
            log "  SKIP $CONFIG — NPZ not found: $NPZ"
            continue
        fi

        # Run PATH B inference
        $BINARY \
            $NPZ \
            ${DS_TEST[$DS]} \
            $PRED_FILE \
            --topk $TOPK \
            --nthreads ${DS_THREADS[$DS]} \
            --batch_size ${DS_BATCH[$DS]} \
            --act_quant \
            > $TIMING_FILE 2>&1

        # Extract timing
        T=$(grep "\[Timing\] inference=" $TIMING_FILE | grep -oP '[\d.]+(?= sec)' | head -1)
        if [ -z "$T" ]; then
            log "  FAIL $CONFIG — check $TIMING_FILE"
            continue
        fi

        FP32=${DS_FP32[$DS]}
        SPD=$($PYTHON -c "print(f'{$FP32/float(\"$T\"):.3f}')")
        TP=$($PYTHON -c "print(f'{${DS_SAMPLES[$DS]}/float(\"$T\"):.0f}')")
        BITS=8; [[ "$CONFIG" == int4* ]] && BITS=4

        if $PYTHON -c "exit(0 if $FP32 >= float('$T') else 1)"; then
            ST="FASTER"; BEATS=true
        else
            ST="SLOWER"; BEATS=false
        fi

        log "  $CONFIG: ${T}s | FP32=${FP32}s | speedup=${SPD}x | $ST"

        # Evaluate accuracy
        cd $BASE/python
        PROP=""
        [ -f "${DS_PROP[$DS]}" ] && PROP="--weights ${DS_PROP[$DS]}"
        $PYTHON evaluation.py \
            --pred-path $PRED_FILE \
            --data ${DS_TEST[$DS]} \
            $PROP > $METRIC_FILE 2>&1
        cd $QUANT_DIR

        P1=$(get_metric "P@1"      $METRIC_FILE)
        P3=$(get_metric "P@3"      $METRIC_FILE)
        P5=$(get_metric "P@5"      $METRIC_FILE)
        N1=$(get_metric "nDCG@1"   $METRIC_FILE)
        N3=$(get_metric "nDCG@3"   $METRIC_FILE)
        N5=$(get_metric "nDCG@5"   $METRIC_FILE)
        SP1=$(get_metric "PSP@1"   $METRIC_FILE)
        SP3=$(get_metric "PSP@3"   $METRIC_FILE)
        SP5=$(get_metric "PSP@5"   $METRIC_FILE)
        SN1=$(get_metric "PSnDCG@1" $METRIC_FILE)
        SN3=$(get_metric "PSnDCG@3" $METRIC_FILE)
        SN5=$(get_metric "PSnDCG@5" $METRIC_FILE)

        log "  Accuracy: P@1=${P1:-N/A} (FP32 baseline: ${DS_FP32_P1[$DS]})"

        printf "%-45s %10s %10s %8s %6s %8s %8s %8s\n" \
            "$CONFIG" "${T}s" "${FP32}s" "${SPD}x" "$ST" \
            "${P1:-N/A}" "${P3:-N/A}" "${P5:-N/A}" >> $SUMMARY

        echo "$DS,$CONFIG,$BITS,$T,$FP32,$SPD,$BEATS,$TP,${P1:-},${P3:-},${P5:-},${N1:-},${N3:-},${N5:-},${SP1:-},${SP3:-},${SP5:-},${SN1:-},${SN3:-},${SN5:-}" >> $CSV
    done

    echo "" >> $SUMMARY
    echo "Completed: $(date)" >> $SUMMARY
    log "  Results: $DS_OUT"
    echo ""
    cat $SUMMARY
}

# ============================================================
# MAIN
# ============================================================

ALL_DATASETS=(eurlex wiki10 amazoncat13k amazon670k delicious200k)

if [ "$DATASET_ARG" = "all" ]; then
    for DS in "${ALL_DATASETS[@]}"; do
        run_dataset "$DS"
    done
else
    run_dataset "$DATASET_ARG"
fi

# Combined CSV
COMBINED=$RESULTS_DIR/pathb_combined_summary.csv
echo "dataset,config,bits,infer_s,fp32_s,speedup,beats_fp32,throughput,P@1,P@3,P@5,nDCG@1,nDCG@3,nDCG@5,PSP@1,PSP@3,PSP@5,PSnDCG@1,PSnDCG@3,PSnDCG@5" > $COMBINED
for DS in "${ALL_DATASETS[@]}"; do
    [ -f "$RESULTS_DIR/$DS/pathb_results.csv" ] && \
        tail -n +2 $RESULTS_DIR/$DS/pathb_results.csv >> $COMBINED
done

log "All done. Combined: $COMBINED"
cat $COMBINED
