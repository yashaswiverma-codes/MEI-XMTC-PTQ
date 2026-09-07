"""
annexml_data_loader.py
----------------------
Loads AnnexML binary model and provides quantization utilities.

AnnexML model binary format (mirrors C++ ReadFromStream):
  AnnexMLParameter  (key-value fields)
  labels_           (per data-point label list)
  num_learner
  For each learner:
    DataPartitioner::ReadFromStream
      w_index_: sparse cluster centroids [feat_id] -> [(cluster, weight)]
    LLEmbedding::ReadFromStream
      embed_size, num_data
      w_mat_vec_: [cluster] -> [(feat_id, float[embed_size])]
      embeddings_: [data_idx, embed_size]
      cluster_assign_: [cluster] -> [data_idx]
"""

from __future__ import annotations
import struct, time, os
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# Binary reading helpers
# ============================================================

def _read_sizet(fp):
    return struct.unpack('<Q', fp.read(8))[0]

def _read_int(fp):
    return struct.unpack('<i', fp.read(4))[0]

def _read_float(fp):
    return struct.unpack('<f', fp.read(4))[0]

def _read_double(fp):
    return struct.unpack('<d', fp.read(8))[0]


def _skip_param(fp):
    """Skip AnnexMLParameter fixed-field block.
    Order matches AnnexMLParameter::WriteToStream:
      int: emb_size, num_learner, num_nn, cls_type, cls_iter, emb_iter, label_normalize
      float: eta0, lambda, gamma
      int: pred_type, num_edge
      float: search_eps
      int: num_thread, seed, verbose
      string x4: train_file, predict_file, model_file, result_file
    Each int/float = 4 bytes. String = int(4B) + chars.
    """
    # 7 ints
    fp.read(4 * 7)
    # 3 floats
    fp.read(4 * 3)
    # 2 ints
    fp.read(4 * 2)
    # 1 float
    fp.read(4 * 1)
    # 3 ints
    fp.read(4 * 3)
    # 4 strings
    for _ in range(4):
        slen = struct.unpack('<i', fp.read(4))[0]
        fp.read(slen)


# ============================================================
# Model data structures
# ============================================================

@dataclass
class AnnexMLLearner:
    """One learner (tree) in the AnnexML forest."""
    # DataPartitioner: w_index_[feat_id] = list of (cluster_id, weight)
    w_index:        List[List[Tuple[int, float]]] = field(default_factory=list)

    # LLEmbedding
    embed_size:     int = 0
    num_data:       int = 0

    # w_mat_vec_[cluster] = list of (feat_id, np.ndarray[embed_size])
    w_mat_vec:      List[List[Tuple[int, np.ndarray]]] = field(default_factory=list)

    # embeddings_: shape (num_data, embed_size)
    embeddings:     Optional[np.ndarray] = None

    # cluster_assign_[cluster] = list of data indices
    cluster_assign: List[List[int]] = field(default_factory=list)

    @property
    def num_clusters(self):
        return len(self.w_mat_vec)


@dataclass
class AnnexMLModel:
    """Full AnnexML model loaded from .bin file."""
    num_learner: int = 0
    embed_size:  int = 0
    # labels_[data_idx] = list of label ids
    labels:      List[List[int]] = field(default_factory=list)
    learners:    List[AnnexMLLearner] = field(default_factory=list)

    @staticmethod
    def load(path: str) -> 'AnnexMLModel':
        model = AnnexMLModel()
        print(f"[Model] Loading {path} ...")
        t0 = time.time()
        with open(path, 'rb') as fp:
            _skip_param(fp)

            # labels_
            num_entries = _read_sizet(fp)
            model.labels = []
            for _ in range(num_entries):
                n = _read_sizet(fp)
                lbls = [_read_int(fp) for _ in range(n)]
                model.labels.append(lbls)

            num_learner = _read_sizet(fp)
            model.num_learner = int(num_learner)
            model.learners = []

            for l in range(num_learner):
                print(f"  Loading learner {l+1}/{num_learner} ...", flush=True)
                lr = AnnexMLLearner()

                # DataPartitioner::ReadFromStream
                # Writes: K_ (size_t), w_index_.size() (size_t),
                #   for each feat: n_pairs (size_t),
                #     for each pair: cluster (size_t), weight (float)
                K = _read_sizet(fp)          # K_ = num clusters
                num_feats = _read_sizet(fp)  # w_index_.size() = num features
                lr.w_index = []
                for i in range(num_feats):
                    n_pairs = _read_sizet(fp)
                    pairs = []
                    for _ in range(n_pairs):
                        cluster = _read_int(fp)   # w_index_[i][j].first is int
                        weight  = _read_float(fp) # w_index_[i][j].second is float
                        pairs.append((int(cluster), float(weight)))
                    lr.w_index.append(pairs)

                # LLEmbedding::ReadFromStream
                # Field order: embed_size(Q), seed(i), verbose(i),
                #   num_clusters(Q), [num_entries(Q), [feat_id(i), elen(Q), floats]],
                #   num_data(Q), [elen(Q), embed_size floats],
                #   num_ca(Q), [n_items(Q), [size_t items]]
                lr.embed_size = int(_read_sizet(fp))  # embed_size_ (size_t)
                _read_int(fp)                          # seed_ (int)
                _read_int(fp)                          # verbose_ (int)
                if model.embed_size == 0:
                    model.embed_size = lr.embed_size

                # w_mat_vec_
                num_clusters_wmat = _read_sizet(fp)
                lr.w_mat_vec = []
                for c in range(num_clusters_wmat):
                    n_entries = _read_sizet(fp)
                    entries = []
                    for _ in range(n_entries):
                        feat_id = _read_int(fp)        # .first is int
                        elen    = _read_sizet(fp)      # .second.size()
                        evec    = np.array([_read_float(fp) for _ in range(elen)],
                                           dtype=np.float32)
                        entries.append((int(feat_id), evec))
                    lr.w_mat_vec.append(entries)

                # embeddings_: num_data_ (size_t), then for each:
                #   elen (size_t) = embed_size, then embed_size floats
                lr.num_data = int(_read_sizet(fp))
                emb = np.zeros((lr.num_data, lr.embed_size), dtype=np.float32)
                for i in range(lr.num_data):
                    elen = _read_sizet(fp)             # always = embed_size
                    for j in range(lr.embed_size):
                        emb[i, j] = _read_float(fp)
                lr.embeddings = emb

                # cluster_assign_
                num_ca = _read_sizet(fp)
                lr.cluster_assign = []
                for c in range(num_ca):
                    n_items = _read_sizet(fp)
                    items = [int(_read_sizet(fp)) for _ in range(n_items)]
                    lr.cluster_assign.append(items)

                model.learners.append(lr)

        t_load = time.time() - t0
        print(f"[Model] Loaded in {t_load:.2f}s  "
              f"embed_size={model.embed_size}  "
              f"num_learner={model.num_learner}  "
              f"num_data={model.learners[0].num_data if model.learners else 0}  "
              f"num_label_entries={len(model.labels)}")
        return model


# ============================================================
# Quantization functions
# ============================================================

def quantize_sym(arr: np.ndarray, bits: int,
                 per_row: bool = True,
                 clip_pct: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric quantization of a 2D float array.
    Returns (data_q, scales).
    data_q: int8/int16/int32, shape same as arr
    scales: float32, shape (num_rows,) if per_row else (1,)
    """
    max_val = (1 << (bits - 1)) - 1
    if bits <= 8:   dtype = np.int8
    elif bits <= 16: dtype = np.int16
    else:           dtype = np.int32

    arr64 = arr.astype(np.float64)

    def _abs_max(a):
        if a.size == 0: return 1.0
        if clip_pct is not None:
            v = float(np.percentile(np.abs(a), clip_pct))
            return v if v > 0 else float(np.max(np.abs(a)))
        return float(np.max(np.abs(a)))

    if per_row:
        scales = np.ones(arr64.shape[0], dtype=np.float32)
        data_q = np.zeros_like(arr64, dtype=dtype)
        for r in range(arr64.shape[0]):
            am = _abs_max(arr64[r])
            if am == 0: continue
            s = am / max_val
            scales[r] = np.float32(s)
            data_q[r] = np.clip(np.round(arr64[r] / s),
                                 -max_val, max_val).astype(dtype)
    else:
        am = _abs_max(arr64)
        s  = am / max_val if am > 0 else 1.0
        scales = np.array([s], dtype=np.float32)
        data_q = np.clip(np.round(arr64 / s),
                         -max_val, max_val).astype(dtype)

    return data_q, scales


def quantize_asym(arr: np.ndarray, bits: int,
                  per_row: bool = True,
                  clip_pct: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Asymmetric quantization.
    Returns (data_q, scales, zero_points).
    data_q: uint8/uint16/uint32
    scales, zero_points: float64
    """
    n_levels = (1 << bits) - 1
    if bits <= 8:   dtype = np.uint8
    elif bits <= 16: dtype = np.uint16
    else:           dtype = np.uint32

    arr64 = arr.astype(np.float64)

    def _clip(a):
        if clip_pct is not None and a.size > 0:
            lo = float(np.percentile(a, 100 - clip_pct))
            hi = float(np.percentile(a, clip_pct))
            return np.clip(a, lo, hi)
        return a

    if per_row:
        scales      = np.ones(arr64.shape[0],  dtype=np.float64)
        zero_points = np.zeros(arr64.shape[0], dtype=np.float64)
        data_q      = np.zeros_like(arr64, dtype=dtype)
        for r in range(arr64.shape[0]):
            a  = _clip(arr64[r])
            lo, hi = float(a.min()), float(a.max())
            rng = hi - lo
            if rng == 0: continue
            s = rng / n_levels
            scales[r]      = s
            zero_points[r] = lo
            data_q[r] = np.clip(np.round((arr64[r] - lo) / s),
                                 0, n_levels).astype(dtype)
    else:
        a  = _clip(arr64)
        lo = float(a.min()); hi = float(a.max())
        rng = hi - lo
        s   = rng / n_levels if rng > 0 else 1.0
        scales      = np.array([s],  dtype=np.float64)
        zero_points = np.array([lo], dtype=np.float64)
        data_q = np.clip(np.round((arr64 - lo) / s),
                         0, n_levels).astype(dtype)

    return data_q, scales, zero_points


def quantize_group(arr: np.ndarray, bits: int,
                   group_size: int = 32, sym: bool = True,
                   clip_pct: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group-wise quantization along columns.
    Returns (data_q int32, scales float32 (nr,ng), zero_points float32 (nr,ng)).
    """
    nr, nc    = arr.shape
    max_val   = (1 << (bits - 1)) - 1 if sym else (1 << bits) - 1
    n_levels  = (1 << bits) - 1
    num_groups = (nc + group_size - 1) // group_size

    data_q      = np.zeros((nr, nc),          dtype=np.int32)
    scales      = np.ones((nr, num_groups),   dtype=np.float32)
    zero_points = np.zeros((nr, num_groups),  dtype=np.float32)

    for r in range(nr):
        for g in range(num_groups):
            cs = g * group_size
            ce = min(cs + group_size, nc)
            seg = arr[r, cs:ce].astype(np.float64)

            if sym:
                if clip_pct is not None:
                    am = float(np.percentile(np.abs(seg), clip_pct))
                    am = am if am > 0 else float(np.max(np.abs(seg)))
                else:
                    am = float(np.max(np.abs(seg)))
                if am == 0: continue
                s = am / max_val
                scales[r, g] = np.float32(s)
                data_q[r, cs:ce] = np.clip(np.round(seg / s),
                                            -max_val, max_val).astype(np.int32)
            else:
                if clip_pct is not None:
                    lo = float(np.percentile(seg, 100 - clip_pct))
                    hi = float(np.percentile(seg, clip_pct))
                    seg2 = np.clip(seg, lo, hi)
                else:
                    seg2 = seg
                lo2, hi2 = float(seg2.min()), float(seg2.max())
                rng = hi2 - lo2
                if rng == 0: continue
                s = rng / n_levels
                scales[r, g]      = np.float32(s)
                zero_points[r, g] = np.float32(lo2)
                data_q[r, cs:ce]  = np.clip(np.round((seg - lo2) / s),
                                             0, n_levels).astype(np.int32)

    return data_q, scales, zero_points


def dequantize_matrix(q: dict) -> np.ndarray:
    """Dequantize a stored quantized matrix back to float32."""
    mode = q['mode']
    dq   = q['data_q'].astype(np.float64)

    if mode == 'sym':
        sc = q['scales'].astype(np.float64)
        if q['per_row']:
            return (dq * sc[:, None]).astype(np.float32)
        return (dq * sc[0]).astype(np.float32)

    elif mode == 'asym':
        sc = q['scales'].astype(np.float64)
        zp = q['zero_points'].astype(np.float64)
        if q['per_row']:
            return (dq * sc[:, None] + zp[:, None]).astype(np.float32)
        return (dq * sc[0] + zp[0]).astype(np.float32)

    elif mode in ('group', 'group_asym'):
        nr, nc   = dq.shape
        gs       = q['group_size']
        sc       = q['scales'].astype(np.float64)
        zp       = q['zero_points'].astype(np.float64)
        out      = np.zeros((nr, nc), dtype=np.float64)
        ng       = sc.shape[1]
        for g in range(ng):
            cs = g * gs; ce = min(cs + gs, nc)
            if mode == 'group':
                out[:, cs:ce] = dq[:, cs:ce] * sc[:, g:g+1]
            else:
                out[:, cs:ce] = dq[:, cs:ce] * sc[:, g:g+1] + zp[:, g:g+1]
        return out.astype(np.float32)

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Build dense w_mat per cluster from w_mat_vec (per learner)
# ============================================================

def build_dense_wmat_per_cluster(learner: AnnexMLLearner):
    """
    Build a list of dense matrices, one per cluster.
    Each matrix has shape (num_entries_in_cluster, embed_size).
    Also returns feat_ids per cluster for sparse dot.

    Returns:
      cluster_feat_ids: List[np.ndarray]  feat_ids for each cluster
      cluster_wmats:    List[np.ndarray]  (n_entries, embed_size) per cluster
    """
    embed_size = learner.embed_size
    cluster_feat_ids = []
    cluster_wmats    = []
    for cluster_entries in learner.w_mat_vec:
        if not cluster_entries:
            cluster_feat_ids.append(np.array([], dtype=np.int32))
            cluster_wmats.append(np.zeros((0, embed_size), dtype=np.float32))
            continue
        feat_ids = np.array([fid for fid, _ in cluster_entries], dtype=np.int32)
        W = np.zeros((len(cluster_entries), embed_size), dtype=np.float32)
        for j, (fid, evec) in enumerate(cluster_entries):
            W[j, :len(evec)] = evec[:embed_size]
        cluster_feat_ids.append(feat_ids)
        cluster_wmats.append(W)
    return cluster_feat_ids, cluster_wmats


# ============================================================
# Save quantized model to npz (one file per learner)
# ============================================================

def save_quantized_learner_npz(q_wmat: dict, q_emb: dict,
                                bits: int, bits_emb: int,
                                path: str, learner_idx: int = 0,
                                feat_ids: np.ndarray = None):
    """
    Save quantized weights for one learner to npz.
    Schema mirrors what quant_annexml_infer.cpp loads.
    """
    mode_int = 0 if 'sym' in q_wmat['mode'] else 1
    is_group = 'group' in q_wmat['mode']

    def _get_dtype(b, asym):
        if asym: return {4: np.uint8, 8: np.uint8, 16: np.uint16, 32: np.uint32}[b]
        return {4: np.int8, 8: np.int8, 16: np.int16, 32: np.int32}[b]

    wm_asym = q_wmat['mode'] in ('asym', 'group_asym')
    em_asym = q_emb['mode']  in ('asym', 'group_asym')
    wm_dtype = _get_dtype(bits,     wm_asym)
    em_dtype = _get_dtype(bits_emb, em_asym)

    wmat_dq = q_wmat['data_q'].astype(wm_dtype)
    emb_dq  = q_emb['data_q'].astype(em_dtype)

    wmat_sc = q_wmat['scales'].astype(np.float32)
    emb_sc  = q_emb['scales'].astype(np.float32)
    wmat_zp = q_wmat.get('zero_points', np.zeros(1)).astype(np.float64)
    emb_zp  = q_emb.get('zero_points',  np.zeros(1)).astype(np.float64)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fids = feat_ids.astype(np.int32) if feat_ids is not None else np.array([], dtype=np.int32)
    np.savez(path,
        bits        = np.array([bits],     dtype=np.int32),
        bits_emb    = np.array([bits_emb], dtype=np.int32),
        mode_int    = np.array([mode_int], dtype=np.int32),
        per_row     = np.array([int(q_wmat.get('per_row', True))], dtype=np.int32),
        is_group    = np.array([int(is_group)], dtype=np.int32),
        group_size  = np.array([int(q_wmat.get('group_size', 32))], dtype=np.int32),
        feat_ids    = fids,
        wmat_data   = wmat_dq.ravel(),
        wmat_scales = wmat_sc.ravel(),
        wmat_zp     = wmat_zp.ravel(),
        wmat_nrows  = np.array([q_wmat['data_q'].shape[0]], dtype=np.int32),
        wmat_ncols  = np.array([q_wmat['data_q'].shape[1]], dtype=np.int32),
        emb_data    = emb_dq.ravel(),
        emb_scales  = emb_sc.ravel(),
        emb_zp      = emb_zp.ravel(),
        emb_nrows   = np.array([q_emb['data_q'].shape[0]], dtype=np.int32),
        emb_ncols   = np.array([q_emb['data_q'].shape[1]], dtype=np.int32),
    )
    print(f"[NPZ] Saved learner {learner_idx} -> {path}  "
          f"wmat={wmat_dq.shape} emb={emb_dq.shape}")
