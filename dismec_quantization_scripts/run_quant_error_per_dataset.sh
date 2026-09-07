#!/bin/bash
# run_quant_error_per_dataset.sh
# ================================
# Runs compute_quant_error.py ONCE PER DATASET as a fresh subprocess
# (single-dataset mode), instead of one long-lived process looping over
# all 5 via --datasets-config.
#
# WHY: del + gc.collect() inside the long-lived process only frees
# Python-level references. glibc's malloc allocator can still retain
# freed heap arenas internally rather than returning them to the OS —
# so RSS climbs across dataset iterations even though live memory is
# small and bounded. Confirmed: RSS hit ~140GB after only eurlex+
# wiki10+amazoncat13k finished (combined data is a few hundred MB at
# most), before amazon670k even started. Spawning a fresh process per
# dataset guarantees the OS reclaims ALL memory on exit, sidestepping
# the allocator-retention issue entirely rather than trying to out-fix
# it from inside Python.
#
# Usage:
#   bash run_quant_error_per_dataset.sh datasets_config.json

set -e

CONFIG_JSON="${1:-datasets_config.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMBINED_OUT="/DATA2/rudra1/dismec_quantization/dismecpp/results/quant_error_dismec_combined_full.json"

if [ ! -f "$CONFIG_JSON" ]; then
    echo "ERROR: config file not found: $CONFIG_JSON"
    exit 1
fi

# Pull the ordered list of dataset names out of the JSON
DATASETS=$(python3 -c "
import json
cfg = json.load(open('$CONFIG_JSON'))
print(' '.join(cfg.keys()))
")

echo "Datasets to run: $DATASETS"
echo ""

for DS in $DATASETS; do
    echo "============================================================"
    echo " Running: $DS  (fresh process)"
    echo " Started: $(date)"
    echo "============================================================"

    # Extract this dataset's fields from the JSON
    read -r WEIGHTS MODELS_DIR OUT_JSON ONE_BASED MAX_ROWS <<< "$(python3 -c "
import json
cfg = json.load(open('$CONFIG_JSON'))['$DS']
one_based_flag = '--one-based' if cfg.get('one_based', True) else '--zero-based'
print(cfg['weights'], cfg['models_dir'], cfg['out_json'], one_based_flag, cfg.get('max_rows', 0))
")"

    # Note: the read above breaks on spaces in paths — if any of your
    # paths contain spaces, replace this block with a small Python
    # driver script instead. Your paths so far don't have spaces, so
    # this is fine as-is.

    ONE_BASED_FLAG="$ONE_BASED"

    python3 "$SCRIPT_DIR/compute_quant_error.py" \
        --dataset "$DS" \
        --weights "$WEIGHTS" \
        --models-dir "$MODELS_DIR" \
        --out-json "$OUT_JSON" \
        --max-rows "$MAX_ROWS" \
        $ONE_BASED_FLAG

    echo ""
    echo " Finished: $DS  at $(date)"
    echo " (process exits here — OS fully reclaims all memory before next dataset)"
    echo ""
done

# Merge all per-dataset JSONs into one combined file
echo "Merging per-dataset JSONs into: $COMBINED_OUT"
python3 -c "
import json
cfg = json.load(open('$CONFIG_JSON'))
combined = {}
for ds, c in cfg.items():
    try:
        d = json.load(open(c['out_json']))
        combined.update(d)
    except FileNotFoundError:
        print(f'WARNING: missing output for {ds}: {c[\"out_json\"]}')
import os
os.makedirs(os.path.dirname('$COMBINED_OUT'), exist_ok=True)
json.dump(combined, open('$COMBINED_OUT', 'w'), indent=2, default=str)
print('Saved combined:', '$COMBINED_OUT')
"

echo ""
echo "All done — $(date)"
