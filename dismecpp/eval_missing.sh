#!/bin/bash

# Define paths matching your machine layout
BASE_DIR="/DATA1/rudra1/dismec_quantization"
PYTHON_DIR="$BASE_DIR/dismecpp/python"

# Paths to script, data, and weight files
EVAL_SCRIPT="$PYTHON_DIR/evaluation.py"
DATA_TEST="$BASE_DIR/dismecpp/data/deliciouslarge/deliciousLarge_test.txt"
WEIGHTS_FILE="$PYTHON_DIR/delicious200k-weights-test-pos.txt" # or weights-test-pos.txt based on your setup

PRED_DIR="$BASE_DIR/dismecpp/results/delicious200k/predictions"
METRICS_DIR="$BASE_DIR/dismecpp/results/delicious200k/metrics"

mkdir -p "$METRICS_DIR"

echo "=== Running Python Evaluation for Missing final files ==="

# List of missing configurations
missing_configs=(
    "int4_group_sym_clip_act"
    "int4_group_sym_clip_act_intinfer"
    "int4_group_sym_clip_act_intinfer_bias"
    "int8_group_sym_clip_act"
    "int8_group_sym_clip_act_intinfer"
    "int8_group_sym_clip_act_intinfer_bias"
)

# Remember to activate your conda/python environment first if dismec is not in global path
for config in "${missing_configs[@]}"; do
    PRED_FILE="$PRED_DIR/pred_final_${config}.txt"
    EVAL_FILE="$METRICS_DIR/eval_final_${config}.txt"
    
    if [ -f "$PRED_FILE" ]; then
        echo "Processing evaluation: $PRED_FILE -> $EVAL_FILE"
        
        # Run using python with your required flags
        python "$EVAL_SCRIPT" \
            --data "$DATA_TEST" \
            --pred-path "$PRED_FILE" \
            --weights "$WEIGHTS_FILE" > "$EVAL_FILE" 2>&1
            
    else
        echo "Skipping: $PRED_FILE not found."
    fi
done

echo "=== Evaluation Suite Done ==="
