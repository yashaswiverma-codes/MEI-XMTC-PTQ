"""
run_quantize.py
===============
DiSMEC PTQ — quantization entry point + core functions.

Small datasets  → load full weights, quantize all 18 configs directly
Large datasets  → chunked streaming (--chunked flag), merge after

Usage:
  # Small (EURLex, Wiki10, AmazonCat-13K, Amazon-670K):
  python3 run_quantize.py \
      --weights /path/to/model.weights-0-L \
      --test    /path/to/test.txt \
      --out-dir /path/to/results/eurlex/models

  # Large single file (Delicious-200K):
  python3 run_quantize.py \
      --weights    /path/to/model.weights-0-205442 \
      --test       /path/to/test.txt \
      --out-dir    /path/to/results/delicious200k/models \
      --chunked \
      --chunk-size 10000
"""

import argparse
import os
import glob
import gc
import time
import numpy as np
import scipy.sparse as sp

from data_loader import load_sparse_data, load_sparse_weights


# ============================================================
# CONSTANTS
# ============================================================
ALL_CONFIGS = [
    'int8_row_sym',    'int8_row_asym',    'int8_row_sym_clip',
    'int8_group_sym',  'int8_group_sym_clip', 'int8_row_sym_clip_mixed',
    'int8_group_sym_clip_act', 'int8_group_sym_clip_act_intinfer',
    'int8_group_sym_clip_act_intinfer_bias',
    'int4_row_sym',    'int4_row_asym',    'int4_row_sym_clip',
    'int4_group_sym',  'int4_group_sym_clip', 'int4_row_sym_clip_mixed',
    'int4_group_sym_clip_act', 'int4_group_sym_clip_act_intinfer',
    'int4_group_sym_clip_act_intinfer_bias',
]

CLIP_PCT   = 99.99
GROUP_SIZE = 8


# ============================================================
# CONFIG PARSING
# ============================================================

def parse_bits(config):
    c = config.lower()
    if 'int4'  in c: return 4
    if 'int8'  in c: return 8
    if 'int16' in c: return 16
    if 'int32' in c: return 32
    raise ValueError(f"Cannot parse bits from: {config}")


def dtype_for(bits, symmetric):
    if bits <= 8:  return np.int8  if symmetric else np.uint8
    if bits == 16: return np.int16 if symmetric else np.uint16
    return np.int32 if symmetric else np.uint32


def qrange(bits, symmetric):
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        return -qmax, qmax
    return 0, (1 << bits) - 1


def config_flags(config):
    c = config.lower()
    return {
        'bits':     parse_bits(c),
        'row':      'row' in c or 'group' not in c,
        'group':    'group' in c,
        'sym':      'sym' in c and 'asym' not in c,
        'clip':     'clip' in c,
        'act':      '_act' in c,
        'intinfer': 'intinfer' in c,
        'mixed':    'mixed' in c,
        'bias':     c.endswith('bias'),
    }


# ============================================================
# QUANTIZATION FUNCTIONS (vectorized numpy)
# ============================================================

def quant_row_sym(data, indptr, bits, clip=False):
    Q       = (2 ** (bits - 1)) - 1
    rl      = np.diff(indptr)
    ab      = np.abs(data)
    nz_rows = np.where(rl > 0)[0]
    maxes   = np.zeros(len(rl), dtype=np.float64)

    if len(nz_rows):
        if clip:
            for ri in nz_rows:
                s, e = indptr[ri], indptr[ri + 1]
                maxes[ri] = float(np.percentile(ab[s:e], CLIP_PCT)) \
                            if (e - s) > 1 else (float(ab[s]) if e > s else 1.0)
        else:
            maxes[nz_rows] = np.maximum.reduceat(ab, indptr[nz_rows])

    maxes  = np.where(maxes == 0, 1.0, maxes)
    scales = maxes / Q
    spe    = np.repeat(scales, rl)
    q      = np.clip(np.round(data / spe), -Q, Q).astype(np.int8)
    return q, scales, np.zeros(len(rl), dtype=np.float64)


def quant_row_asym(data, indptr, bits):
    Q    = (2 ** bits) - 1
    rl   = np.diff(indptr)
    nz   = np.where(rl > 0)[0]
    mins  = np.zeros(len(rl), dtype=np.float64)
    maxes = np.zeros(len(rl), dtype=np.float64)

    if len(nz):
        mins[nz]  = np.minimum.reduceat(data.astype(np.float64), indptr[nz])
        maxes[nz] = np.maximum.reduceat(data.astype(np.float64), indptr[nz])

    rng    = np.where(maxes - mins == 0, 1.0, maxes - mins)
    scales = rng / Q
    zp     = mins
    spe    = np.repeat(scales, rl)
    zpe    = np.repeat(zp,     rl)
    q      = np.clip(np.round((data - zpe) / spe), 0, Q).astype(np.uint8)
    return q, scales, zp


def quant_group_sym(data, indptr, bits, clip=False, gs=GROUP_SIZE):
    Q      = (2 ** (bits - 1)) - 1
    rl     = np.diff(indptr)
    nrows  = len(rl)
    q_out  = np.zeros(len(data), dtype=np.int8)
    all_sc = []
    all_zp = []

    for i in range(nrows):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            all_sc.append(1.0); all_zp.append(0.0)
            continue
        row   = data[s:e].astype(np.float64)
        row_q = np.zeros(e - s, dtype=np.float64)

        for g0 in range(0, e - s, gs):
            g1    = min(g0 + gs, e - s)
            chunk = row[g0:g1]
            am    = float(np.percentile(np.abs(chunk), CLIP_PCT)) \
                    if clip and len(chunk) > 1 else float(np.abs(chunk).max())
            if am == 0: am = 1.0
            sc = am / Q
            all_sc.append(sc); all_zp.append(0.0)
            row_q[g0:g1] = np.clip(np.round(chunk / sc), -Q, Q)

        q_out[s:e] = row_q.astype(np.int8)

    return q_out, np.array(all_sc, dtype=np.float64), np.array(all_zp, dtype=np.float64)


def quant_mixed(data, indptr, base_bits, high_bits=8, low_bits=4,
                high_pct=10.0, clip=True):
    rl    = np.diff(indptr)
    norms = np.zeros(len(rl), dtype=np.float64)
    nz    = np.where(rl > 0)[0]
    if len(nz):
        norms[nz] = np.sqrt(
            np.add.reduceat(data.astype(np.float64) ** 2, indptr[nz]))

    thresh    = max(1, int(high_pct / 100.0 * len(rl)))
    high_rows = np.argsort(norms)[::-1][:thresh]
    mask_high = np.zeros(len(rl), dtype=bool)
    mask_high[high_rows] = True

    q8, sc8, _ = quant_row_sym(data, indptr, high_bits, clip=clip)
    q4, sc4, _ = quant_row_sym(data, indptr, low_bits,  clip=clip)

    q_out  = np.where(np.repeat(mask_high, rl), q8, q4)
    sc_out = np.where(mask_high, sc8, sc4)
    zp_out = np.zeros(len(rl), dtype=np.float64)
    return q_out, sc_out, zp_out


def quantize_weights(W, config):
    """Quantize weight matrix W for given config string."""
    f    = config_flags(config)
    bits = f['bits']

    if f['mixed']:
        q, sc, zp = quant_mixed(W.data, W.indptr, bits,
                                 high_bits=8, low_bits=4,
                                 high_pct=10.0, clip=True)
        gs = -1
    elif f['group']:
        q, sc, zp = quant_group_sym(W.data, W.indptr, bits, clip=f['clip'])
        gs = GROUP_SIZE
    elif not f['sym']:
        q, sc, zp = quant_row_asym(W.data, W.indptr, bits)
        gs = -1
    else:
        q, sc, zp = quant_row_sym(W.data, W.indptr, bits, clip=f['clip'])
        gs = -1

    return {'data': q, 'scales': sc, 'zero_points': zp, 'group_size': gs}


def save_npz(path, W, config, result):
    """
    Save quantized model NPZ.
    Compatible with quant_infer_dismec_v2 C++ binary.
    Does NOT save 'mode' key (causes cnpy crash in C++ binary).
    """
    f    = config_flags(config)
    bits = f['bits']
    gs   = result['group_size']
    eff_act_bits = bits if f['intinfer'] else 8

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    save_dict = {
        'indices':     W.indices.astype(np.int32),
        'indptr':      W.indptr.astype(np.int32),
        'shape':       np.array(W.shape,                           dtype=np.int32),
        'bits':        np.array([bits],                            dtype=np.int32),
        'symmetric':   np.array([1 if f['sym'] else 0],           dtype=np.int32),
        'group_size':  np.array([gs],                             dtype=np.int32),
        'act_quant':   np.array([1 if f['act'] else 0],           dtype=np.int32),
        'act_bits':    np.array([eff_act_bits if f['act'] else 8], dtype=np.int32),
        'clip_pct':    np.array([CLIP_PCT if f['clip'] else -1.],  dtype=np.float32),
        'intinfer':    np.array([1 if f['intinfer'] else 0],       dtype=np.int32),
        'mixed':       np.array([1 if f['mixed'] else 0],          dtype=np.int32),
        'bias':        np.array([1 if f['bias'] else 0],           dtype=np.int32),
        'nnz':         np.array([len(W.data)],                     dtype=np.int32),
        'data':        result['data'],
        'scales':      result['scales'].astype(np.float64),
        'zero_points': result['zero_points'].astype(np.float64),
    }

    np.savez(path, **save_dict)
    npz_path = path + '.npz' if not path.endswith('.npz') else path
    sz = os.path.getsize(npz_path)
    print(f"  Saved: {os.path.basename(npz_path)}  ({sz/1024/1024:.1f} MB)")


# ============================================================
# SMALL DATASET: direct quantization
# ============================================================
def run_small(weights_path, test_path, out_dir, configs):
    os.makedirs(out_dir, exist_ok=True)

    _, num_features = load_sparse_data(test_path)
    num_features_with_bias = num_features + 1

    print(f"Loading weights: {weights_path}")
    W = load_sparse_weights(weights_path, num_features_with_bias)
    print(f"W shape: {W.shape}  nnz={W.nnz}")

    for config in configs:
        out_path = os.path.join(out_dir, f"{config}.npz")
        if os.path.exists(out_path):
            print(f"  SKIP {config} (exists)")
            continue
        print(f"\n  Quantizing: {config}")
        t0     = time.time()
        result = quantize_weights(W, config)
        save_npz(out_path, W, config, result)
        print(f"  Done in {time.time()-t0:.1f}s")


# ============================================================
# CHUNKED: stream single large weights file
# ============================================================
def stream_single(weights_path, chunk_size, num_features):
    rows, cols, vals = [], [], []
    row_in_chunk     = 0
    chunk_idx        = 0

    with open(weights_path) as f:
        for line in f:
            for tok in line.split():
                if ':' not in tok: continue
                c, v = tok.split(':', 1)
                try:
                    ci = int(c)
                    if 0 <= ci < num_features:
                        rows.append(row_in_chunk)
                        cols.append(ci)
                        vals.append(float(v))
                except ValueError:
                    continue
            row_in_chunk += 1

            if row_in_chunk == chunk_size:
                yield chunk_idx, sp.csr_matrix(
                    (np.array(vals, dtype=np.float32),
                     (np.array(rows, dtype=np.int32),
                      np.array(cols, dtype=np.int32))),
                    shape=(row_in_chunk, num_features))
                chunk_idx        += 1
                rows, cols, vals  = [], [], []
                row_in_chunk      = 0

    if row_in_chunk > 0:
        yield chunk_idx, sp.csr_matrix(
            (np.array(vals, dtype=np.float32),
             (np.array(rows, dtype=np.int32),
              np.array(cols, dtype=np.int32))),
            shape=(row_in_chunk, num_features))


# ============================================================
# CHUNKED: stream pre-split weight files (Amazon-3M)
# ============================================================
def stream_split(weights_dir, num_features):
    split_files = sorted(glob.glob(os.path.join(weights_dir, '*weights*')))
    if not split_files:
        split_files = sorted(glob.glob(os.path.join(weights_dir, '*')))

    for chunk_idx, fpath in enumerate(split_files):
        rows, cols, vals = [], [], []
        row_in_chunk     = 0
        with open(fpath) as f:
            for line in f:
                for tok in line.split():
                    if ':' not in tok: continue
                    c, v = tok.split(':', 1)
                    try:
                        ci = int(c)
                        if 0 <= ci < num_features:
                            rows.append(row_in_chunk)
                            cols.append(ci)
                            vals.append(float(v))
                    except ValueError:
                        continue
                row_in_chunk += 1
        if row_in_chunk > 0:
            yield chunk_idx, sp.csr_matrix(
                (np.array(vals, dtype=np.float32),
                 (np.array(rows, dtype=np.int32),
                  np.array(cols, dtype=np.int32))),
                shape=(row_in_chunk, num_features))


# ============================================================
# SAVE CHUNK NPZ
# ============================================================
def save_chunk(config, chunk_idx, W, result, chunk_dir):
    os.makedirs(chunk_dir, exist_ok=True)
    path = os.path.join(chunk_dir, f"{config}_chunk{chunk_idx:04d}.npz")
    np.savez(path,
        data        = result['data'],
        indices     = W.indices.astype(np.int32),
        indptr      = W.indptr.astype(np.int32),
        scales      = result['scales'],
        zero_points = result['zero_points'],
        nrows       = np.array([W.shape[0]], dtype=np.int32),
    )


# ============================================================
# MERGE CHUNKS → FINAL NPZ (int64 safe for Amazon-3M)
# ============================================================
def merge_config(config, chunk_dir, out_dir, n_chunks):
    f    = config_flags(config)
    bits = f['bits']
    gs   = GROUP_SIZE if f['group'] else -1

    out_path = os.path.join(out_dir, f"{config}.npz")
    if os.path.exists(out_path):
        print(f"  SKIP {config} (exists)")
        return True

    print(f"  Merging {config} ({n_chunks} chunks)...")
    all_data    = []
    all_indices = []
    all_indptr  = [np.int64(0)]
    all_scales  = []
    all_zp      = []
    nnz_total   = np.int64(0)
    total_rows  = 0
    max_col     = 0

    for ci in range(n_chunks):
        cpf = os.path.join(chunk_dir, f"{config}_chunk{ci:04d}.npz")
        if not os.path.exists(cpf):
            print(f"    ERROR: chunk missing: {cpf}")
            return False
        c = np.load(cpf, allow_pickle=True)
        all_data.append(c['data'])
        all_indices.append(c['indices'])
        if len(c['indices']) > 0:
            max_col = max(max_col, int(c['indices'].max()) + 1)
        all_indptr.extend(
            (c['indptr'][1:].astype(np.int64) + nnz_total).tolist())
        all_scales.append(c['scales'])
        all_zp.append(c['zero_points'])
        nnz_total  += np.int64(len(c['data']))
        total_rows += int(c['nrows'][0])
        del c; gc.collect()

    eff_act_bits = bits if f['intinfer'] else 8
    save_dict = {
        'indices':     np.concatenate(all_indices).astype(np.int64),
        'indptr':      np.array(all_indptr, dtype=np.int64),
        'shape':       np.array([total_rows, max_col],             dtype=np.int32),
        'bits':        np.array([bits],                            dtype=np.int32),
        'symmetric':   np.array([1 if f['sym'] else 0],           dtype=np.int32),
        'group_size':  np.array([gs],                             dtype=np.int32),
        'act_quant':   np.array([1 if f['act'] else 0],           dtype=np.int32),
        'act_bits':    np.array([eff_act_bits if f['act'] else 8], dtype=np.int32),
        'clip_pct':    np.array([CLIP_PCT if f['clip'] else -1.],  dtype=np.float32),
        'intinfer':    np.array([1 if f['intinfer'] else 0],       dtype=np.int32),
        'mixed':       np.array([1 if f['mixed'] else 0],          dtype=np.int32),
        'bias':        np.array([1 if f['bias'] else 0],           dtype=np.int32),
        'nnz':         np.array([nnz_total],                       dtype=np.int64),
        'data':        np.concatenate(all_data),
        'scales':      np.concatenate(all_scales).astype(np.float64),
        'zero_points': np.concatenate(all_zp).astype(np.float64),
    }

    base = out_path[:-4] if out_path.endswith('.npz') else out_path
    np.savez(base, **save_dict)
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"    Saved: {config}.npz  ({sz:.0f} MB)  rows={total_rows}  nnz={nnz_total}")
    return True


# ============================================================
# CHUNKED MAIN
# ============================================================
def run_chunked(weights_path, weights_dir, test_path, out_dir, configs, chunk_size):
    os.makedirs(out_dir, exist_ok=True)
    chunk_dir = os.path.join(out_dir, 'chunks')
    os.makedirs(chunk_dir, exist_ok=True)

    _, num_features = load_sparse_data(test_path)
    num_features_with_bias = num_features + 1
    print(f"num_features (with bias): {num_features_with_bias}")

    pending = [c for c in configs
               if not os.path.exists(os.path.join(out_dir, f"{c}.npz"))]
    if not pending:
        print("All configs already quantized.")
        return 0

    print(f"Pending configs: {len(pending)}")

    gen = stream_split(weights_dir, num_features_with_bias) \
          if weights_dir \
          else stream_single(weights_path, chunk_size, num_features_with_bias)

    n_chunks = 0
    t_all    = time.time()

    for chunk_idx, W in gen:
        n_chunks = chunk_idx + 1
        print(f"\n{'='*50}\nChunk {chunk_idx:04d}: shape={W.shape} nnz={W.nnz}")

        done = all(
            os.path.exists(os.path.join(chunk_dir, f"{c}_chunk{chunk_idx:04d}.npz"))
            for c in pending)
        if done:
            print(f"SKIP chunk {chunk_idx:04d} (all configs done)")
            continue

        t0 = time.time()
        for config in pending:
            cpf = os.path.join(chunk_dir, f"{config}_chunk{chunk_idx:04d}.npz")
            if os.path.exists(cpf): continue
            result = quantize_weights(W, config)
            save_chunk(config, chunk_idx, W, result, chunk_dir)
            print(f"  {config}: {time.time()-t0:.1f}s")

        del W; gc.collect()
        print(f"Chunk {chunk_idx:04d} done in {time.time()-t0:.1f}s")

    print(f"\nAll chunks done in {(time.time()-t_all)/60:.1f} min")
    print(f"\nMerging chunks → final NPZs")
    for config in pending:
        merge_config(config, chunk_dir, out_dir, n_chunks)

    return n_chunks


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='DiSMEC PTQ — quantize weights')
    parser.add_argument('--weights',     help='Weights file (small / single large dataset)')
    parser.add_argument('--weights-dir', help='Split weights directory (Amazon-3M)')
    parser.add_argument('--test',        required=True, help='Test file (for num_features)')
    parser.add_argument('--out-dir',     required=True, help='Output directory for NPZs')
    parser.add_argument('--config',      help='Single config (default: all 18)')
    parser.add_argument('--chunked',     action='store_true',
                        help='Chunked streaming for large datasets')
    parser.add_argument('--chunk-size',  type=int, default=10000,
                        help='Labels per chunk (default: 10000)')
    parser.add_argument('--merge-only',  action='store_true',
                        help='Only merge existing chunks, skip quantization')
    args = parser.parse_args()

    configs = [args.config] if args.config else ALL_CONFIGS

    if args.merge_only:
        chunk_dir = os.path.join(args.out_dir, 'chunks')
        existing  = sorted(glob.glob(
            os.path.join(chunk_dir, f"{configs[0]}_chunk*.npz")))
        n_chunks  = len(existing)
        print(f"Merge-only: {n_chunks} chunks found")
        for config in configs:
            merge_config(config, chunk_dir, args.out_dir, n_chunks)
        return

    if not args.weights and not args.weights_dir:
        parser.error('Provide --weights or --weights-dir')

    if args.chunked or args.weights_dir:
        run_chunked(args.weights, args.weights_dir, args.test,
                    args.out_dir, configs, args.chunk_size)
    else:
        run_small(args.weights, args.test, args.out_dir, configs)

    # Verify
    print(f"\nVerification:")
    ok = 0
    for config in configs:
        p = os.path.join(args.out_dir, f"{config}.npz")
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1024 / 1024
            print(f"  ✓ {config}: {sz:.0f} MB")
            ok += 1
        else:
            print(f"  ✗ MISSING: {config}")
    print(f"{ok}/{len(configs)} configs complete")


if __name__ == '__main__':
    main()
