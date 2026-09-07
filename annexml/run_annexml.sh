#!/bin/bash

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Timestamp function for logging
log_with_timestamp() {
  while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
  done
}

#echo "Compiling annexml..."
#make -C src/ annexml 2>&1 | log_with_timestamp | tee logs/compile_wiki10_baseline.log

#echo "Running training..."
#src/annexml train annexml-example.json 2>&1 | log_with_timestamp | tee logs/train_wiki10_baseline.log

echo "Running prediction..."
src/annexml predict annexml-example.json 2>&1 | log_with_timestamp | tee logs/predict_wiki10_baseline.log

#echo "Evaluating predictions..."
#cat  annexml-result-wiki10_baseline.txt | python3 scripts/learning-evaluate_predictions.py 2>&1 | log_with_timestamp | tee logs/eval_wiki10_baseline_1.log

#echo "Evaluating predictions with propensity scoring..."
#cat  annexml-result-wiki10_baseline.txt | python3 /DATA2/rudra1/AnnexML/learning-evaluate_predictions_propensity_scored_fixed.py /DATA1/rudra1/bonsai/sandbox/data/wiki10/train.txt    -A 0.55 -B 1.5 2>&1 | log_with_timestamp | tee logs/eval_wiki10_baseline_2.log

echo "All steps completed successfully."

