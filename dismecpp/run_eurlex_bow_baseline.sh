#!/bin/bas
# --- DiSMEC Training Step (1) ---
# Trains the model on the Amazon670K dataset.
/mnt/sdb/rudra1/dismecpp/build/bin/train /mnt/sdb/rudra1/dismecpp/data/deliciouslarge/deliciousLarge_train.txt delicious_bow_baseline.model --augment-for-bias --normalize-instances --save-sparse-txt  --weight-culling=0.01 --threads 20 > delicious_bow_baseline_training.txt
# --- DiSMEC Prediction Step (2) ---
# Uses the trained model to generate predictions on the test set.
# Corrected model name from amazon6670k.model to amazon670k.model
/mnt/sdb/rudra1/dismecpp/build/bin/predict /mnt/sdb/rudra1/dismecpp/data/deliciouslarge/deliciousLarge_test.txt delicious_bow_baseline.model delicious_predictions.txt --augment-for-bias --normalize-instances --topk=5 --threads 20 > delicious_bow_baseline.txt
# --- Change Directory (3) ---
# Moves to the Python directory to run helper scripts.
#cd python

# --- Weight Preparation Step (4) ---
# Calculates propensity weights for the test set (needed for PSP/PSnDCG).
#python3 make_weights.py --dataset_dir /DATA/rudra1/dismecpp --train-data eurlex_train.txt --test-data eurlex_test.txt > eurlex_bow_step3_5_pred.txt

# --- Evaluation Step (5) ---
# Computes Precision, nDCG, PSP, and PSnDCG and redirects the output to a file.
#python3 evaluation.py --pred-path /DATA/rudra1/dismecpp/eurlex_knn_pred_5.txt --data /DATA/rudra1/dismecpp/eurlex_test.txt --weights weights-test-pos.txt > eurlex_evaluation_scores_bow_5_pred.txt
#--init-mode msi
