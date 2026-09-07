"""
run_quantized_annexml.py
------------------------
Post-Training Quantization inference pipeline for AnnexML XMC models.
"""

from __future__ import annotations
import argparse, os, time, math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

from annexml_data_loader import (
    AnnexMLModel, AnnexMLLearner,
    quantize_sym, quantize_asym, quantize_group,
    dequantize_matrix, build_dense_wmat_per_cluster,
)


# ============================================================
# Data loading
# ============================================================

def load_test_data(path: str):
    """Load XMC test file. Returns list of (labels, sparse_features)."""
    data = []
    with open(path) as f:
        first = f.readline().strip().split()
        if not (len(first) == 3 and all(t.isdigit() for t in first)):
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line: continue
            toks = line.split()
            if ':' in toks[0]:
                labels, fi = [], 0
            else:
                labels = [int(l) for l in toks[0].split(',') if l.strip()]
                fi = 1
            feats = []
            for tok in toks[fi:]:
                if ':' not in tok: continue
                fid, val = tok.split(':', 1)
                feats.append((int(fid), float(val)))
            data.append((labels, feats))
    return data


def normalize_sparse(feats):
    norm = sum(v*v for _, v in feats) ** 0.5
    if norm == 0: return feats
    return [(f, v/norm) for f, v in feats]


# ============================================================
# Per-group embedding quantization (consistent pack + scale)
# ============================================================

def quantize_emb_grouped(
    E: np.ndarray,
    group_size: int,
    bits: int,
    clip_pct: float = None,
) -> tuple:
    """
    Quantize embedding matrix E with per-group scales.
    """
    num_emb, embed_size = E.shape
    Q = (2 ** (bits - 1)) - 1          # 127 for INT8
    num_groups = (embed_size + group_size - 1) // group_size

    q_int  = np.zeros((num_emb, embed_size), dtype=np.int8)
    scales = np.zeros((num_emb, num_groups), dtype=np.float32)
    E_dq   = np.zeros((num_emb, embed_size), dtype=np.float32)

    for i in range(num_emb):
        for g in range(num_groups):
            start = g * group_size
            end   = min(embed_size, start + group_size)
            chunk = E[i, start:end]

            if clip_pct is not None:
                alpha = float(np.percentile(np.abs(chunk), clip_pct))
            else:
                alpha = float(np.abs(chunk).max())

            if alpha == 0.0:
                alpha = 1.0

            sc = alpha / Q
            scales[i, g] = sc

            q = np.clip(np.round(chunk / sc), -Q, Q).astype(np.int8)
            q_int[i, start:end]  = q
            E_dq[i, start:end]   = q.astype(np.float32) * sc

    return q_int, scales, E_dq


# ============================================================
# Quantized AnnexML model
# ============================================================

class QuantizedAnnexML:
    """Holds quantized w_mat and embeddings for all learners."""

    def __init__(self, model: AnnexMLModel,
                 bits: int = 8, bits_emb: int = None,
                 quant_mode: str = 'sym', per_row: bool = True,
                 clip_pct: float = None, group_size: int = 32):
        self.model       = model
        self.bits        = bits
        self.bits_emb    = bits_emb if bits_emb is not None else bits
        self.quant_mode  = quant_mode
        self.per_row     = per_row
        self.clip_pct    = clip_pct
        self.group_size  = group_size

        self.q_wmat: List[dict] = []
        self.q_emb:  List[dict] = []

        self.emb_group_scales: List[Optional[np.ndarray]] = []
        self.emb_group_qint:   List[Optional[np.ndarray]] = []

        self._dq_emb:  List[Optional[np.ndarray]] = [None] * model.num_learner
        self._quantize()

    def _quant_matrix(self, arr: np.ndarray, bits: int) -> dict:
        mode = self.quant_mode
        if mode == 'sym':
            dq, sc = quantize_sym(arr, bits, self.per_row, self.clip_pct)
            return {'mode': 'sym', 'data_q': dq, 'scales': sc,
                    'shape': arr.shape, 'per_row': self.per_row}
        elif mode == 'asym':
            dq, sc, zp = quantize_asym(arr, bits, self.per_row, self.clip_pct)
            return {'mode': 'asym', 'data_q': dq, 'scales': sc,
                    'zero_points': zp, 'shape': arr.shape, 'per_row': self.per_row}
        elif mode in ('group', 'group_asym'):
            sym_g = (mode == 'group')
            dq, sc, zp = quantize_group(arr, bits, self.group_size,
                                         sym_g, self.clip_pct)
            return {'mode': mode, 'data_q': dq, 'scales': sc,
                    'zero_points': zp, 'shape': arr.shape,
                    'group_size': self.group_size, 'per_row': self.per_row}
        raise ValueError(f"Unknown quant_mode: {mode}")

    def _is_grouped(self) -> bool:
        return self.quant_mode in ('group', 'group_asym')

    def _quantize(self):
        print(f"[PTQ] Quantizing {self.model.num_learner} learners...")
        t0 = time.time()
        self.cluster_feat_ids = []

        for l, lr in enumerate(self.model.learners):
            feat_ids_per_cluster, wmats_per_cluster = \
                build_dense_wmat_per_cluster(lr)
            self.cluster_feat_ids.append(feat_ids_per_cluster)

            q_clusters = []
            for W in wmats_per_cluster:
                if W.shape[0] == 0:
                    q_clusters.append(None)
                else:
                    q_clusters.append(self._quant_matrix(W, self.bits))
            self.q_wmat.append(q_clusters)

            E = lr.embeddings

            if self._is_grouped():
                q_int, gs, E_dq = quantize_emb_grouped(
                    E,
                    self.group_size,
                    self.bits_emb,
                    self.clip_pct,
                )
                q_emb_dict = {
                    'mode':      self.quant_mode,
                    'data_q':    q_int,
                    'scales':    np.array([1.0], dtype=np.float32),
                    'zero_points': np.array([0.0], dtype=np.float64),
                    'shape':     E.shape,
                    'per_row':   self.per_row,
                    'group_size': self.group_size,
                }
                self.q_emb.append(q_emb_dict)
                self.emb_group_scales.append(gs)
                self.emb_group_qint.append(q_int)
                self._dq_emb[l] = E_dq
                print(f"  Learner {l+1}: {len(wmats_per_cluster)} clusters | grouped emb {E.shape}")
            else:
                q_emb_dict = self._quant_matrix(E, self.bits_emb)
                self.q_emb.append(q_emb_dict)
                self.emb_group_scales.append(None)
                self.emb_group_qint.append(None)
                print(f"  Learner {l+1}: {len(wmats_per_cluster)} clusters | emb {E.shape}")

        print(f"[PTQ] Done in {time.time()-t0:.2f}s")

    def get_dq_wmat_cluster(self, l: int, cluster: int) -> tuple:
        feat_ids = self.cluster_feat_ids[l][cluster]
        q = self.q_wmat[l][cluster]
        if q is None:
            return feat_ids, np.zeros((0, self.model.embed_size), dtype=np.float32)
        return feat_ids, dequantize_matrix(q)

    def get_dq_emb(self, l: int) -> np.ndarray:
        if self._dq_emb[l] is not None:
            return self._dq_emb[l]
        self._dq_emb[l] = dequantize_matrix(self.q_emb[l])
        return self._dq_emb[l]


# ============================================================
# Inference
# ============================================================

def get_nearest_cluster(feats, w_index):
    ip = defaultdict(float)
    for fid, val in feats:
        if fid >= len(w_index): continue
        for cluster, weight in w_index[fid]:
            ip[cluster] += val * weight
    return max(ip, key=ip.get) if ip else 0


def predict_instance_quantized(
    feats_norm: List[Tuple[int, float]],
    model: AnnexMLModel,
    qmodel: Optional[QuantizedAnnexML],
    topk: int = 5,
) -> List[Tuple[int, float]]:
    label_scores = defaultdict(float)
    embed_size = model.embed_size

    for l, lr in enumerate(model.learners):
        cluster = get_nearest_cluster(feats_norm, lr.w_index)

        z = np.zeros(embed_size, dtype=np.float32)
        if qmodel is not None:
            feat_ids, W_dq = qmodel.get_dq_wmat_cluster(l, cluster)
            feat_to_row = {int(fid): j for j, fid in enumerate(feat_ids)}
            for fid, val in feats_norm:
                row = feat_to_row.get(fid, -1)
                if row >= 0:
                    z += val * W_dq[row]
        else:
            if cluster < len(lr.w_mat_vec):
                entries = lr.w_mat_vec[cluster]
                entry_map = {fid: evec for fid, evec in entries}
                for fid, val in feats_norm:
                    if fid in entry_map:
                        ev = entry_map[fid]
                        z[:len(ev)] += val * ev

        norm = float(np.linalg.norm(z))
        if norm > 0: z /= norm

        if cluster >= len(lr.cluster_assign): continue
        ca = np.array(lr.cluster_assign[cluster], dtype=np.int64)
        num_nn = 10
        if len(ca) == 0: continue
        if qmodel is not None:
            E_dq = qmodel.get_dq_emb(l)
            scores_arr = E_dq[ca].dot(z)
        else:
            scores_arr = lr.embeddings[ca].dot(z)
        k_nn = min(num_nn, len(scores_arr))
        top_j = np.argpartition(scores_arr, -k_nn)[-k_nn:]
        for j in top_j:
            idx = int(ca[j])
            for lbl in model.labels[idx]:
                label_scores[lbl] += 1

    result = sorted(label_scores.items(), key=lambda x: -x[1])
    return result[:topk]


# Global state for parallel inference
_ANNEXML_STATE = {}

def _init_annexml_worker(state):
    global _ANNEXML_STATE
    _ANNEXML_STATE = state

def _predict_one_annexml(args):
    i, labels, feats = args
    feats_norm = normalize_sparse(feats)
    preds = predict_instance_quantized(
        feats_norm,
        _ANNEXML_STATE["model"],
        _ANNEXML_STATE["qmodel"],
        _ANNEXML_STATE["topk"],
    )
    return i, preds

def predict_all(data, model, qmodel, topk=5, verbose=True, num_workers=1):
    n = len(data)
    results = [None] * n
    t0 = time.time()
    if num_workers > 1:
        import multiprocessing as _mp
        _ANNEXML_STATE.update({"model": model, "qmodel": qmodel, "topk": topk})
        args_list = [(i, labels, feats) for i, (labels, feats) in enumerate(data)]
        print(f"  Parallel inference: {num_workers} workers, {n} instances")
        with _mp.Pool(processes=num_workers,
                      initializer=_init_annexml_worker,
                      initargs=(_ANNEXML_STATE,)) as pool:
            for done, (i, preds) in enumerate(
                    pool.imap_unordered(_predict_one_annexml, args_list, chunksize=50)):
                results[i] = preds
                if verbose and (done+1) % 500 == 0:
                    el = time.time() - t0
                    print(f"  [{done+1}/{n}] {el:.1f}s  ({(done+1)/el:.0f} samp/s)")
    else:
        for i, (labels, feats) in enumerate(data):
            feats_norm = normalize_sparse(feats)
            preds = predict_instance_quantized(feats_norm, model, qmodel, topk)
            results[i] = preds
            if verbose and (i+1) % 500 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{n}] {el:.1f}s  ({(i+1)/el:.0f} samp/s)")
    return results


# ============================================================
# Metrics
# ============================================================

def calc_propensity(train_path, A=0.55, B=1.5):
    num_inst = 0
    freqs    = defaultdict(int)
    with open(train_path) as f:
        for line in f:
            if ':' not in line: continue
            num_inst += 1
            idx    = line.find(' ')
            labels = line[:idx] if idx >= 0 else line
            for l in labels.split(','):
                l = l.strip()
                if not l: continue
                freqs[int(l)] += 1
    C  = (math.log(num_inst) - 1) * (B + 1) ** A
    pw = {k: 1.0 + C * (v + B) ** (-A) for k, v in freqs.items()}
    return pw, 1.0 + C * B ** (-A)


def precision_at_k(preds, y_true, k):
    total, n = 0.0, 0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        n += 1
        ps = {p[0] for p in pred[:k]}
        total += sum(1 for l in labels if l in ps) / k
    return total / n * 100 if n > 0 else 0.0


def ndcg_at_k(preds, y_true, k):
    total, n = 0.0, 0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        n += 1
        ls = set(labels)
        dcg  = sum(1.0/math.log2(i+2) for i,(l,_) in enumerate(pred[:k]) if l in ls)
        idcg = sum(1.0/math.log2(i+2) for i in range(min(k, len(labels))))
        if idcg > 0: total += dcg / idcg
    return total / n * 100 if n > 0 else 0.0


def psp_at_k(preds, y_true, pw, dpw, k):
    total, n = 0.0, 0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        n += 1
        ls  = set(labels)
        num = sum(pw.get(p[0], dpw) for p in pred[:k] if p[0] in ls)
        den = sum(sorted([pw.get(l, dpw) for l in ls], reverse=True)[:k])
        if den > 0: total += num / den
    return total / n * 100 if n > 0 else 0.0


def psndcg_at_k(preds, y_true, pw, dpw, k):
    total, n = 0.0, 0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        n += 1
        ls = set(labels)
        dcg  = sum((pw.get(lbl, dpw)/math.log2(i+2))
                   for i,(lbl,_) in enumerate(pred[:k]) if lbl in ls)
        idcg = sum((pw.get(l, dpw)/math.log2(i+2))
                   for i,l in enumerate(
                       sorted(ls, key=lambda l: pw.get(l, dpw), reverse=True)[:k]))
        if idcg > 0: total += dcg / idcg
    return total / n * 100 if n > 0 else 0.0


def evaluate(preds, y_true, pw, dpw, ks=(1,3,5)):
    m = {}
    skip = sum(1 for y in y_true if not y)
    print(f"[Eval] Skipping {skip} samples with no labels")
    for k in ks:
        m[f'P@{k}']      = precision_at_k(preds, y_true, k)
        m[f'nDCG@{k}']   = ndcg_at_k(preds, y_true, k)
        m[f'PSP@{k}']    = psp_at_k(preds, y_true, pw, dpw, k)
        m[f'PSnDCG@{k}'] = psndcg_at_k(preds, y_true, pw, dpw, k)
    return m


# ============================================================
# Model size
# ============================================================

def model_size_mb(qmodel: QuantizedAnnexML) -> float:
    total = 0
    for l in range(qmodel.model.num_learner):
        for q in qmodel.q_wmat[l]:
            if q is None: continue
            total += q['data_q'].nbytes
            total += q['scales'].nbytes
            if 'zero_points' in q: total += q['zero_points'].nbytes
        q = qmodel.q_emb[l]
        total += q['data_q'].nbytes
        total += q['scales'].nbytes
        if 'zero_points' in q: total += q['zero_points'].nbytes
        if qmodel.emb_group_scales[l] is not None:
            total += qmodel.emb_group_scales[l].nbytes
    return total / 1e6


def fp32_size_mb(model: AnnexMLModel) -> float:
    total = 0
    for lr in model.learners:
        total += lr.embeddings.nbytes
        for cluster_entries in lr.w_mat_vec:
            for _, evec in cluster_entries:
                total += evec.nbytes
    return total / 1e6


# ============================================================
# Write predictions
# ============================================================

def write_predictions(preds, path, topk):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        for pred in preds:
            toks = [f"{lbl}:{sc:.4f}" for lbl, sc in pred[:topk]]
            f.write(' '.join(toks) + '\n')
    print(f"[Output] Written to {path}")


# ============================================================
# TRUE packed INT4 helpers
# ============================================================

def pack_int4_matrix(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.int8)
    rows, cols = q.shape
    packed_cols = (cols + 1) // 2
    out = np.zeros((rows, packed_cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(0, cols, 2):
            v0 = int(q[r, c]) & 0xF
            if c + 1 < cols:
                v1 = int(q[r, c + 1]) & 0xF
            else:
                v1 = 0
            out[r, c // 2] = v0 | (v1 << 4)
    return out


# ============================================================
# NPZ saving — TRUE packed INT4 + grouped embedding support
# ============================================================

def save_quantized_learner_npz_with_group_scales(
    q_wmat: dict,
    q_emb: dict,
    bits: int,
    bits_emb: int,
    path: str,
    learner_idx: int,
    feat_ids: np.ndarray,
    emb_group_scales: Optional[np.ndarray] = None,
    emb_group_qint: Optional[np.ndarray] = None,
    emb_fp32=None,
):
    arrays = {}

    # Metadata
    arrays['bits'] = np.array([bits], dtype=np.int32)
    arrays['bits_emb'] = np.array([bits_emb], dtype=np.int32)
    arrays['mode_int'] = np.array(
        [0 if q_wmat['mode'] in ('sym', 'group') else 1],
        dtype=np.int32
    )
    arrays['per_row'] = np.array([int(q_wmat['per_row'])], dtype=np.int32)
    arrays['feat_ids'] = feat_ids.astype(np.int32)

    # w_mat
    if bits == 4:
        arrays['wmat_data'] = pack_int4_matrix(q_wmat['data_q'])
    else:
        arrays['wmat_data'] = q_wmat['data_q'].astype(np.int8)

    arrays['wmat_scales'] = q_wmat['scales'].astype(np.float32)
    arrays['wmat_zp'] = (
        q_wmat.get('zero_points', np.zeros(1)).astype(np.float32)
    )
    arrays['wmat_nrows'] = np.array([q_wmat['shape'][0]], dtype=np.int32)
    arrays['wmat_ncols'] = np.array([q_wmat['shape'][1]], dtype=np.int32)

    # Group metadata for w_mat
    if q_wmat['mode'] in ('group', 'group_asym'):
        arrays['group_size'] = np.array(
            [q_wmat.get('group_size', 32)], dtype=np.int32
        )
        # group_scales same as wmat_scales — skip duplicate
        arrays['group_zp'] = (
            q_wmat.get('zero_points', np.zeros(1)).astype(np.float32)
        )

    # Embeddings — only saved in first cluster per learner (q_emb=None for others)
    if q_emb is not None:
        is_grouped_emb = (emb_group_scales is not None)
        if is_grouped_emb:
            assert emb_group_qint is not None
            if bits_emb == 4:
                arrays['emb_data'] = pack_int4_matrix(emb_group_qint)
            else:
                arrays['emb_data'] = emb_group_qint.astype(np.int8)
        else:
            if bits_emb == 4:
                arrays['emb_data'] = pack_int4_matrix(q_emb['data_q'])
            else:
                arrays['emb_data'] = q_emb['data_q']
        arrays['emb_nrows'] = np.array([q_emb['shape'][0]], dtype=np.int32)
        arrays['emb_ncols'] = np.array([q_emb['shape'][1]], dtype=np.int32)
        if emb_fp32 is not None and bits == 32:
            arrays['emb_fp32'] = emb_fp32.astype(np.float32)

    if q_emb is not None:
        # Non-grouped embedding metadata
        if not is_grouped_emb:
            arrays['emb_scales'] = q_emb['scales'].astype(np.float32)
            arrays['emb_zp'] = (
                q_emb.get('zero_points', np.zeros(1)).astype(np.float64)
            )

        # Grouped embedding scales (FP16)
        if is_grouped_emb:
            arrays['emb_group_scales'] = (
                emb_group_scales.astype(np.float16).ravel()
            )
            arrays['emb_num_groups'] = np.array(
                [emb_group_scales.shape[1]], dtype=np.int32
            )

    np.savez(path, **arrays)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',        required=True)
    p.add_argument('--test',         required=True)
    p.add_argument('--train',        required=True)
    p.add_argument('--out_dir',      required=True)
    p.add_argument('--out_prefix',   default=None)
    p.add_argument('--bits',         type=int, default=8, choices=[4,8,16,32])
    p.add_argument('--bits_emb',     type=int, default=None)
    p.add_argument('--quant_mode',   default='sym',
                   choices=['sym','asym','group','group_asym'])
    p.add_argument('--global_scale', action='store_true')
    p.add_argument('--clip_pct',     type=float, default=None)
    p.add_argument('--group_size',   type=int, default=32)
    p.add_argument('--topk',         type=int, default=5)
    p.add_argument('--float_baseline', action='store_true')
    p.add_argument('--save_npz',     action='store_true', default=True)
    p.add_argument('--A',            type=float, default=0.55)
    p.add_argument('--B',            type=float, default=1.5)
    p.add_argument('--act_quant',    action='store_true')
    p.add_argument('--act_bits',     type=int, default=8)
    p.add_argument('--intinfer',     action='store_true')
    p.add_argument('--mixed',        action='store_true')
    p.add_argument('--high_bits',    type=int, default=8)
    p.add_argument('--low_bits',     type=int, default=4)
    p.add_argument('--high_pct',     type=float, default=90.0)
    p.add_argument('--bias_correction', action='store_true')
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--skip_inference', action='store_true', default=False)
    return p.parse_args()


def main():
    args   = parse_args()
    per_row = not args.global_scale
    prefix  = args.out_prefix or \
              f"int{args.bits}_{'row' if per_row else 'global'}_{args.quant_mode}"
    os.makedirs(args.out_dir, exist_ok=True)

    model = AnnexMLModel.load(args.model)

    print(f"[Data] Loading {args.test} ...")
    data   = load_test_data(args.test)
    y_true = [labels for labels, _ in data]
    print(f"[Data] {len(data)} samples")

    print(f"[Propensity] From {args.train} ...")
    pw, dpw = calc_propensity(args.train, args.A, args.B)

    # FP32 baseline
    if args.float_baseline:
        print("\n=== FP32 Baseline ===")
        t0 = time.time()
        preds_fp = predict_all(data, model, None, args.topk,
                               num_workers=args.num_workers)
        t_fp = time.time() - t0
        m_fp = evaluate(preds_fp, y_true, pw, dpw)
        print(f"[FP32] time={t_fp:.2f}s")
        for k, v in m_fp.items(): print(f"  {k} = {v:.2f}")
        fp_path = os.path.join(args.out_dir, "pred_fp32_baseline.txt")
        write_predictions(preds_fp, fp_path, args.topk)

    eff_bits      = args.bits
    eff_bits_emb  = args.bits_emb
    eff_clip_pct  = args.clip_pct
    eff_mode      = args.quant_mode

    if args.mixed:
        eff_bits     = args.high_bits
        eff_bits_emb = args.low_bits
        print(f"[Mixed] projection={eff_bits}bit, embeddings={eff_bits_emb}bit")

    if args.act_quant:
        if eff_mode not in ('group', 'group_asym'):
            eff_mode = 'asym'
        print(f"[ActQuant] mode={eff_mode}")

    if args.intinfer:
        if eff_mode not in ('group', 'group_asym'):
            eff_mode = 'sym'
        print(f"[IntInfer] mode={eff_mode}")

    if args.bias_correction:
        eff_clip_pct = 99.9
        print(f"[BiasCorr] clip_pct={eff_clip_pct}")

    # Quantize
    print(f"\n=== Quantization: {prefix} ===")
    t_q0 = time.time()
    qmodel = QuantizedAnnexML(
        model, bits=eff_bits, bits_emb=eff_bits_emb,
        quant_mode=eff_mode, per_row=per_row,
        clip_pct=eff_clip_pct, group_size=args.group_size,
    )
    t_quant = time.time() - t_q0

    sz_fp32 = fp32_size_mb(model)
    sz_q    = model_size_mb(qmodel)
    print(f"[Size] FP32={sz_fp32:.1f}MB  Quant={sz_q:.1f}MB | Ratio={sz_fp32/sz_q:.1f}x")

    # Save NPZ
    if args.save_npz:
        npz_dir = os.path.join(args.out_dir, 'models', prefix)
        os.makedirs(npz_dir, exist_ok=True)
        is_grouped = eff_mode in ('group', 'group_asym')
        for l in range(model.num_learner):
            feat_ids_per_cluster = qmodel.cluster_feat_ids[l]
            gs_l      = qmodel.emb_group_scales[l]
            qint_l    = qmodel.emb_group_qint[l]
            first_cluster_saved = False
            emb_saved = False  # Save emb only in first cluster
            for c, q_c in enumerate(qmodel.q_wmat[l]):
                if q_c is None: continue
                path = os.path.join(npz_dir, f"learner{l}_cluster{c}.npz")
                gs_to_save   = None
                qint_to_save = None
                if is_grouped and gs_l is not None and not first_cluster_saved:
                    gs_to_save   = gs_l
                    qint_to_save = qint_l
                    first_cluster_saved = True
                save_quantized_learner_npz_with_group_scales(
                    q_c,
                    qmodel.q_emb[l] if not emb_saved else None,
                    args.bits,
                    qmodel.bits_emb,
                    path,
                    learner_idx=l,
                    feat_ids=feat_ids_per_cluster[c],
                    emb_group_scales=gs_to_save,
                    emb_group_qint=qint_to_save,
                    emb_fp32=(
                        model.learners[l].embeddings
                        if (args.bits == 32 and not emb_saved) else None
                    ),
                )
                emb_saved = True
        print(f"[NPZ] Saved to {npz_dir}/")

    if args.skip_inference:
        print("[skip_inference] NPZ files generated successfully.")
        import sys; sys.exit(0)

    print(f"\n=== Inference (INT{args.bits}) ===")
    t_i0 = time.time()
    preds_q = predict_all(data, model, qmodel, args.topk,
                          num_workers=args.num_workers)
    t_infer = time.time() - t_i0
    print(f"[Infer] Done in {t_infer:.2f}s | Speed: {len(data)/t_infer:.0f} samp/s")
    m_q = evaluate(preds_q, y_true, pw, dpw)
    print("[Metrics]")
    for k, v in m_q.items(): print(f"  {k} = {v:.2f}")


if __name__ == '__main__':
    main()
