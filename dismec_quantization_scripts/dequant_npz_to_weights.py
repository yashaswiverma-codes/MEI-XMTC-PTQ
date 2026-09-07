"""
dequant_npz_to_weights.py
==========================
Unified dequantization for DiSMEC PTQ.
Converts quantized NPZ → DiSMEC SparseTXT weights + model JSON.

IMPORTANT — index format differs by dataset:
  Small datasets (EURLex, Wiki10, AmazonCat-13K, Amazon-670K):
    Original weights are 1-based → use --one-based
    dequant writes: indices[k] + 1

  Large datasets (Delicious-200K, Amazon-3M):
    Original weights are 0-based → do NOT use --one-based
    predict binary uses --augment-for-bias separately
    dequant writes: indices[k] (0-based)

Usage:
  # Small datasets (1-based):
  python3 dequant_npz_to_weights.py \
      --npz            model_int8_row_sym.npz \
      --out            dequant.weights \
      --model-json     dequant.model.json \
      --original-model eurlex_bow_baseline.model \
      --num-features   5002 \
      --one-based

  # Large datasets (0-based):
  python3 dequant_npz_to_weights.py \
      --npz            model_int8_row_sym.npz \
      --out            dequant.weights \
      --model-json     dequant.model.json \
      --original-model delicious_bow_baseline.model \
      --num-features   782586
"""

import argparse
import json
import os
import numpy as np


def dequantize_npz(npz_path, out_weights_path, model_json_path,
                   original_model_path, num_features, one_based=False):
    print(f"Loading NPZ: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)

    # int64 to avoid overflow on Amazon-3M (2.8B nnz > int32 max)
    indices    = npz['indices'].astype(np.int64)
    indptr     = npz['indptr'].astype(np.int64)
    bits       = int(npz['bits'][0])
    symmetric  = int(npz['symmetric'][0])
    group_size = int(npz['group_size'][0])

    L           = len(indptr) - 1
    data        = npz['data']
    scales      = npz['scales']
    zero_points = npz['zero_points'] if 'zero_points' in npz \
                  else np.zeros(L, dtype=np.float64)

    idx_offset = 1 if one_based else 0

    print(f"rows={L}  nnz={len(data)}  bits={bits}  "
          f"sym={symmetric}  group={group_size}  "
          f"index_format={'1-based (+1)' if one_based else '0-based'}")
    print(f"Writing: {out_weights_path}")

    written = 0
    sid     = np.int64(0)

    with open(out_weights_path, 'w') as f:
        for r in range(L):
            s = int(indptr[r])
            e = int(indptr[r + 1])

            if s == e:
                f.write('\n')
                continue

            parts = []
            nnz_r = e - s

            if group_size <= 0:
                sc = float(scales[r])
                zp = float(zero_points[r])
                for k in range(s, e):
                    if symmetric:
                        w = sc * float(data[k])
                    else:
                        w = float(data[k]) * sc + zp
                    if w != 0.0:
                        parts.append(f"{int(indices[k]) + idx_offset}:{w:.6f}")
            else:
                sid_base = sid
                for k in range(s, e):
                    g_idx   = np.int64(k - s) // np.int64(group_size)
                    cur_sid = int(sid_base + g_idx)
                    sc = float(scales[cur_sid])
                    zp = float(zero_points[cur_sid])
                    if symmetric:
                        w = sc * float(data[k])
                    else:
                        w = float(data[k]) * sc + zp
                    if w != 0.0:
                        parts.append(f"{int(indices[k]) + idx_offset}:{w:.6f}")
                sid += np.int64((nnz_r + group_size - 1) // group_size)

            f.write(' '.join(parts) + '\n')
            written += 1

            if r % 50000 == 0 and r > 0:
                print(f"  Progress: {r}/{L} rows ({r*100//L}%)  sid={sid}")

    print(f"Written {written} rows")

    # Write JSON model file
    with open(original_model_path) as f:
        orig = json.load(f)

    num_labels = orig.get('num-labels', L)
    nf         = num_features if num_features > 0 else orig.get('num-features', 0)

    model_dict = {
        'date':  orig.get('date', '2026'),
        'files': [{
            'count':         num_labels,
            'file':          os.path.abspath(out_weights_path),
            'first':         0,
            'weight-format': 'SparseTXT',
        }],
        'num-features': nf,
        'num-labels':   num_labels,
    }

    with open(model_json_path, 'w') as f:
        json.dump(model_dict, f, indent=4)
    print(f"Model JSON written: {model_json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',            required=True)
    parser.add_argument('--out',            required=True)
    parser.add_argument('--model-json',     required=True)
    parser.add_argument('--original-model', required=True)
    parser.add_argument('--num-features',   type=int, default=0)
    parser.add_argument('--one-based',      action='store_true',
                        help='Write 1-based indices (EURLex, Wiki10, AmazonCat, Amazon-670K). '
                             'Default: 0-based (Delicious-200K, Amazon-3M)')
    args = parser.parse_args()

    dequantize_npz(args.npz, args.out, args.model_json,
                   args.original_model, args.num_features,
                   one_based=args.one_based)
    print("Done.")


if __name__ == '__main__':
    main()
