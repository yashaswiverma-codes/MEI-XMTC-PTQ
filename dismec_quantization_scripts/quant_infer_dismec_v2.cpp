/*
 * quant_infer_dismec_v2.cpp
 * ==========================
 * Unified DiSMEC Quantized Inference Binary
 * Works for ALL datasets: EURLex, Wiki10, AmazonCat-13K, Amazon-670K,
 *                         Delicious-200K, Amazon-3M
 *
 * PATH A (default):
 *   Dequantize weights â†’ Eigen sparse float â†’ SpMM via BLAS
 *   Used for all configs without --act_quant
 *
 * PATH B (--act_quant):
 *   Quantize activations to INT8 â†’ integer dot product â†’ rescale
 *   Used for _act configs
 *
 * Memory management:
 *   --batch_size controls samples per batch (default: auto)
 *   Auto mode: targets ~4GB scores matrix per batch
 *   Use small batch_size for large datasets:
 *     Delicious-200K: --batch_size 5000
 *     Amazon-3M:      --batch_size 1000
 *
 * Supports:
 *   - All 18 PTQ configs (row/group sym/asym, mixed, act, intinfer)
 *   - int32 and int64 indptr/indices (Amazon-3M has nnz > int32 max)
 *   - Mixed precision (per-row INT8/INT4 selection)
 *
 * -----------------------------------------------------------------
 *
 * Build:
 *   g++ -O3 -march=native -funroll-loops -ffast-math \
 *       -mavx2 -mfma -fopenmp -std=c++17 \
 *       -I../deps/cnpy -I../deps/eigen \
 *       quant_infer_dismec_v2.cpp ../deps/cnpy/cnpy.cpp \
 *       -lz -lopenblas \
 *       -o quant_infer_dismec_v2
 *
 * Usage:
 *   quant_infer_dismec_v2 model.npz test.txt output.txt \
 *       [--topk N]         default: 5
 *       [--nthreads N]     default: all cores
 *       [--batch_size N]   default: auto (targets 4GB scores matrix)
 *       [--act_quant]      PATH B: quantize activations to INT8
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <stdexcept>
#include <omp.h>
#include <cnpy.h>

// Eigen
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;
using hrc = chrono::high_resolution_clock;

typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMatF;
typedef Eigen::MatrixXf                              DenseMatF;
typedef Eigen::VectorXf                              VecF;


// ============================================================
// MODEL â€” supports both int32 and int64 indptr/indices
// ============================================================
struct Model {
    // Quantized data
    vector<int8_t>   data_i8;
    vector<uint8_t>  data_u8;

    // Indices and pointers (stored as int64 for Amazon-3M compatibility)
    vector<int64_t>  indices;
    vector<int64_t>  indptr;

    // Scales and zero points
    vector<double>   scales;
    vector<double>   zero_points;
    vector<int>      group_offset;   // pre-computed per-row group scale start

    int rows       = 0;
    int cols       = 0;
    int bits       = 8;
    int symmetric  = 1;
    int group_size = -1;
    int act_quant  = 0;
    int act_bits   = 8;
    int intinfer   = 0;
    int mixed      = 0;
    int bias_flag  = 0;   // corresponds to npz "bias" field / config's "_bias" suffix.
                           // Previously never loaded, so nothing in the C++ ever
                           // branched on it. Now gates whether the bias column is
                           // kept in full precision (see path_b_batch) vs quantized
                           // like any other activation at m.act_bits.

    // Mixed precision fields
    vector<uint8_t>  mask_high;
    vector<int8_t>   data_high;
    vector<int8_t>   data_low;
    vector<double>   scale_high;
    vector<double>   scale_low;
    vector<double>   zero_high;
    vector<double>   zero_low;
    vector<int>      off_high;
    vector<int>      off_low;

    // Dequantized Eigen sparse matrix (built once for PATH A)
    SpMatF W_eigen;
    bool   eigen_built = false;
};


// ============================================================
// LOAD MODEL from NPZ
// Handles both int32 and int64 stored indptr/indices
// ============================================================
Model load_model(const string& path) {
    cnpy::npz_t npz = cnpy::npz_load(path);
    Model m;

    // Load indices/indptr â€” check word_size explicitly. Do NOT rely on
    // try/catch here: cnpy::NpyArray::as_vec<T>() does not validate that
    // word_size == sizeof(T) before copying, so calling as_vec<int64_t>()
    // on an array actually stored as int32 silently reads 2x the allocated
    // bytes (heap-buffer-overflow / UB) instead of throwing. Must branch
    // on the real on-disk word_size instead.
    if (npz.count("indices")) {
        auto& arr = npz["indices"];
        if (arr.word_size == 8) {
            m.indices = arr.as_vec<int64_t>();
        } else if (arr.word_size == 4) {
            auto v = arr.as_vec<int32_t>();
            m.indices.assign(v.begin(), v.end());
        } else {
            throw runtime_error("indices: unexpected word_size=" +
                                 to_string(arr.word_size));
        }
    }
    if (npz.count("indptr")) {
        auto& arr = npz["indptr"];
        if (arr.word_size == 8) {
            m.indptr = arr.as_vec<int64_t>();
        } else if (arr.word_size == 4) {
            auto v = arr.as_vec<int32_t>();
            m.indptr.assign(v.begin(), v.end());
        } else {
            throw runtime_error("indptr: unexpected word_size=" +
                                 to_string(arr.word_size));
        }
    }

    m.rows       = (int)m.indptr.size() - 1;
    m.bits       = npz["bits"].as_vec<int>()[0];
    m.symmetric  = npz["symmetric"].as_vec<int>()[0];
    m.group_size = npz["group_size"].as_vec<int>()[0];
    m.act_quant  = npz["act_quant"].as_vec<int>()[0];
    m.act_bits   = npz["act_bits"].as_vec<int>()[0];
    m.bias_flag  = npz.count("bias") ? npz["bias"].as_vec<int>()[0] : 0;
    m.intinfer   = npz["intinfer"].as_vec<int>()[0];
    m.mixed      = npz.count("mixed") ? npz["mixed"].as_vec<int>()[0] : 0;

    // Determine cols
    m.cols = 0;
    if (npz.count("shape")) {
        try {
            auto sv = npz["shape"].as_vec<int>();
            if ((int)sv.size() >= 2) m.cols = sv[1];
        } catch (...) {}
    }
    if (m.cols <= 0 && !m.indices.empty())
        m.cols = (int)(*max_element(m.indices.begin(), m.indices.end()) + 1);
    if (m.cols <= 0)
        throw runtime_error("Cannot determine model cols from NPZ");

    // Load quantized data
    if (m.mixed && npz.count("q_data_high")) {
        // Mixed precision: separate high/low arrays
        m.mask_high  = npz["mask_high"].as_vec<uint8_t>();
        m.data_high  = npz["q_data_high"].as_vec<int8_t>();
        m.data_low   = npz["q_data_low"].as_vec<int8_t>();
        m.scale_high = npz["scale_high"].as_vec<double>();
        m.scale_low  = npz["scale_low"].as_vec<double>();
        if (npz.count("zero_high")) m.zero_high = npz["zero_high"].as_vec<double>();
        if (npz.count("zero_low"))  m.zero_low  = npz["zero_low"].as_vec<double>();

        // Build per-row offsets into data_high/data_low
        m.off_high.resize(m.rows + 1, 0);
        m.off_low.resize(m.rows + 1, 0);
        int ch = 0, cl = 0;
        for (int r = 0; r < m.rows; r++) {
            m.off_high[r] = ch;
            m.off_low[r]  = cl;
            int nnz = (int)(m.indptr[r+1] - m.indptr[r]);
            if (m.mask_high[r]) ch += nnz;
            else                cl += nnz;
        }
        m.off_high[m.rows] = ch;
        m.off_low[m.rows]  = cl;

        cerr << "[Model] MIXED rows=" << m.rows << " cols=" << m.cols << endl;
    } else {
        // Standard: single data array
        if (npz.count("scales"))      m.scales      = npz["scales"].as_vec<double>();
        if (npz.count("zero_points")) m.zero_points = npz["zero_points"].as_vec<double>();

        if (m.symmetric)
            m.data_i8 = npz["data"].as_vec<int8_t>();
        else
            m.data_u8 = npz["data"].as_vec<uint8_t>();

        // Pre-compute group scale offsets
        if (m.group_size > 0) {
            m.group_offset.resize(m.rows, 0);
            int sid = 0;
            for (int r = 0; r < m.rows; r++) {
                m.group_offset[r] = sid;
                int nnz = (int)(m.indptr[r+1] - m.indptr[r]);
                sid += (nnz + m.group_size - 1) / m.group_size;
            }
        }

        cerr << "[Model] rows=" << m.rows << " cols=" << m.cols
             << " bits=" << m.bits << " sym=" << m.symmetric
             << " group=" << m.group_size
             << " act_quant=" << m.act_quant
             << " nnz=" << m.indices.size() << endl;
    }

    return m;
}


// ============================================================
// DEQUANTIZE WEIGHT VALUE â€” inline helpers
// ============================================================
inline float dequant_standard(const Model& m, int64_t j, int sid) {
    double sc = m.scales[sid];
    double zp = m.zero_points.empty() ? 0.0 : m.zero_points[sid];
    if (m.symmetric)
        return (float)(sc * (double)m.data_i8[j]);
    else
        return (float)((double)m.data_u8[j] * sc + zp);
}

inline int64_t weight_int(const Model& m, int64_t j) {
    return m.symmetric ? (int64_t)m.data_i8[j] : (int64_t)m.data_u8[j];
}

inline float dequant_mixed(const Model& m, int r, int k_in_row) {
    if (m.mask_high[r]) {
        int64_t j = m.off_high[r] + k_in_row;
        double sc = m.scale_high[r];
        double zp = m.zero_high.empty() ? 0.0 : m.zero_high[r];
        return (float)(sc * (double)m.data_high[j] - zp);
    } else {
        int64_t j = m.off_low[r] + k_in_row;
        double sc = m.scale_low[r];
        double zp = m.zero_low.empty() ? 0.0 : m.zero_low[r];
        return (float)(sc * (double)m.data_low[j] - zp);
    }
}


// ============================================================
// BUILD EIGEN SPARSE MATRIX (PATH A â€” dequantized float)
// Called once, stored in model for reuse across batches
// ============================================================
void build_eigen_sparse(Model& m) {
    cerr << "[Build] Dequantizing â†’ Eigen sparse (PATH A)..." << endl;

    vector<Eigen::Triplet<float>> trips;
    trips.reserve(m.indices.size());

    for (int r = 0; r < m.rows; r++) {
        int64_t start = m.indptr[r];
        int64_t end   = m.indptr[r + 1];
        if (start == end) continue;
        int nnz_r = (int)(end - start);

        if (m.mixed) {
            for (int k = 0; k < nnz_r; k++) {
                float w = dequant_mixed(m, r, k);
                if (w != 0.f)
                    trips.emplace_back(r, (int)m.indices[start + k], w);
            }
        } else if (m.group_size <= 0) {
            for (int64_t k = start; k < end; k++) {
                float w = dequant_standard(m, k, r);
                if (w != 0.f)
                    trips.emplace_back(r, (int)m.indices[k], w);
            }
        } else {
            int sid_base = m.group_offset[r];
            for (int g = 0; g < nnz_r; g += m.group_size) {
                int gend = min(g + m.group_size, nnz_r);
                int sid  = sid_base + g / m.group_size;
                for (int k = g; k < gend; k++) {
                    float w = dequant_standard(m, start + k, sid);
                    if (w != 0.f)
                        trips.emplace_back(r, (int)m.indices[start + k], w);
                }
            }
        }
    }

    m.W_eigen.resize(m.rows, m.cols);
    m.W_eigen.setFromTriplets(trips.begin(), trips.end());
    m.W_eigen.makeCompressed();
    m.eigen_built = true;

    cerr << "[Build] Eigen sparse: " << m.W_eigen.rows() << "Ã—" << m.W_eigen.cols()
         << " nnz=" << m.W_eigen.nonZeros() << endl;
}


// ============================================================
// TEST DATA
// ============================================================
struct TestData {
    vector<vector<pair<int,float>>> X;
    int n_features = 0;
    int n_labels   = 0;
};

TestData load_test(const string& path) {
    TestData td;
    ifstream f(path);
    if (!f) throw runtime_error("Cannot open test file: " + path);

    string line;
    getline(f, line);
    {
        istringstream ss(line);
        int ns, nf, nl;
        if (ss >> ns >> nf >> nl) {
            td.n_features = nf;
            td.n_labels   = nl;
        }
    }

    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        string tok;
        vector<pair<int,float>> feats;
        bool first = true;
        while (ss >> tok) {
            if (first) {
                first = false;
                if (tok.find(':') == string::npos) continue;
            }
            size_t p = tok.find(':');
            if (p != string::npos) {
                int   idx = stoi(tok.substr(0, p)) - 1;  // 1-based â†’ 0-based
                float val = stof(tok.substr(p + 1));
                if (idx >= 0) feats.push_back({idx, val});
            }
        }
        // Normalize
        float nm = 0.f;
        for (auto& [i,v]: feats) nm += v * v;
        nm = sqrtf(nm) + 1e-8f;
        for (auto& [i,v]: feats) v /= nm;
        td.X.push_back(move(feats));
    }

    // Infer n_features if not in header
    if (td.n_features == 0 && !td.X.empty())
        for (auto& feats: td.X)
            for (auto& [i,v]: feats)
                td.n_features = max(td.n_features, i + 1);

    cerr << "[Test] " << td.X.size() << " samples, "
         << td.n_features << " features" << endl;
    return td;
}


// ============================================================
// AUTO BATCH SIZE
// Target: ~4GB scores matrix per batch
// scores = L Ã— batch_n Ã— sizeof(float)
// ============================================================
int auto_batch_size(int L, size_t target_bytes = 4ULL * 1024 * 1024 * 1024) {
    size_t bytes_per_sample = (size_t)L * sizeof(float);
    int    batch            = (int)(target_bytes / bytes_per_sample);
    batch = max(batch, 1);
    batch = min(batch, 50000);
    cerr << "[Auto] batch_size=" << batch
         << " (scores=" << (size_t)L * batch * 4 / 1024 / 1024 << "MB)" << endl;
    return batch;
}


// ============================================================
// BUILD DENSE BATCH MATRIX X [batch_n Ã— cols]
// Adds bias at col (cols-1)
// ============================================================
DenseMatF build_batch(const TestData& td, int batch_start, int batch_n, int cols) {
    DenseMatF X = DenseMatF::Zero(batch_n, cols);
    int BIAS    = cols - 1;
    for (int i = 0; i < batch_n; i++) {
        for (auto& [fid, val]: td.X[batch_start + i])
            if (fid < BIAS) X(i, fid) = val;
        X(i, BIAS) = 1.0f;
    }
    return X;
}


// ============================================================
// PATH A INFERENCE â€” Eigen sparse Ã— dense batch (BLAS SGEMM)
// scores[L Ã— batch] = W[L Ã— cols] Ã— X[cols Ã— batch]
// ============================================================
void path_a_batch(const Model& m, const DenseMatF& X_batch,
                  vector<float>& scores, int batch_n) {
    // W [L Ã— C] Ã— X.T [C Ã— batch_n] â†’ scores [L Ã— batch_n]
    DenseMatF S = m.W_eigen * X_batch.transpose();   // L Ã— batch_n

    // Copy to scores vector (row-major: scores[r * batch_n + i])
    for (int r = 0; r < m.rows; r++)
        for (int i = 0; i < batch_n; i++)
            scores[(size_t)r * batch_n + i] = S(r, i);
}


// ============================================================
// PATH B INFERENCE â€” INT8 activations, manual OMP loop
// Needed when act_quant=1: quantize X â†’ INT8, integer dot product
//
// FIXED in this revision:
//   - added a dedicated m.mixed branch (previously indexed empty
//     m.data_i8/m.data_u8/m.scales for mixed-precision models -> UB)
//   - removed (int) truncation on the index passed to weight_int(),
//     which corrupted weight lookups once total nnz exceeded INT32_MAX
// ============================================================
void path_b_batch(const Model& m, const DenseMatF& X_float,
                  const vector<int8_t>& X_int8, float x_scale,
                  vector<float>& scores, int batch_n) {
    int cols = m.cols;
    const int BIAS_COL = cols - 1;   // build_batch() always sets X(:, BIAS_COL) = 1.0

    #pragma omp parallel for schedule(dynamic, 64)
    for (int r = 0; r < m.rows; r++) {
        int64_t start = m.indptr[r];
        int64_t end   = m.indptr[r + 1];
        if (start == end) continue;
        int nnz_r = (int)(end - start);

        float* row_scores = scores.data() + (size_t)r * batch_n;

        if (m.mixed) {
            // ---- MIXED PRECISION ROW ----
            // A whole row lives entirely in data_high[] or data_low[],
            // selected by m.mask_high[r], with per-row scale/zero_point.
            // Matches dequant_mixed()'s convention: w = sc*code - zp
            bool     high = m.mask_high[r] != 0;
            int64_t  base = high ? m.off_high[r] : m.off_low[r];
            double   sc   = high ? m.scale_high[r] : m.scale_low[r];
            double   zp   = high
                              ? (m.zero_high.empty() ? 0.0 : m.zero_high[r])
                              : (m.zero_low.empty()  ? 0.0 : m.zero_low[r]);

            for (int kr = 0; kr < nnz_r; kr++) {
                int      feat = (int)m.indices[start + kr];
                int64_t  j    = base + kr;
                int64_t  wq   = high ? (int64_t)m.data_high[j]
                                      : (int64_t)m.data_low[j];

                if (feat == BIAS_COL && m.bias_flag) {
                    // Only for "_bias"-suffixed configs (m.bias_flag set from
                    // the npz "bias" field): keep the bias term in full
                    // precision, since its activation is always exactly 1.0
                    // with no quantization uncertainty. Non-"_bias" configs
                    // fall through below and quantize bias like any other
                    // activation, at act_Q -- matching the naming convention
                    // that "_bias" is what makes this behavior different.
                    float w_real = (float)((double)wq * sc - zp);
                    if (w_real != 0.f)
                        for (int i = 0; i < batch_n; i++)
                            row_scores[i] += w_real;
                    continue;
                }

                // Can only skip on wq==0 if there's no zero-point offset;
                // with zp != 0 a "zero code" still contributes -zp*x.
                if (wq == 0 && zp == 0.0) continue;

                // zero_point is an additive term on the dequantized weight,
                // so (unlike the pure-symmetric case) it can't be folded
                // into a single wq*xq integer product â€” dequantize the
                // weight once per (row, k), then scale by the real
                // (dequantized) activation value.
                double w_real        = (double)wq * sc - zp;
                double w_real_scaled = w_real * (double)x_scale;

                for (int i = 0; i < batch_n; i++) {
                    int64_t xq = (int64_t)X_int8[(size_t)i * cols + feat];
                    row_scores[i] += (float)(w_real_scaled * (double)xq);
                }
            }
        } else if (m.group_size <= 0) {
            double sc      = m.scales[r];
            double w_scale = sc * (double)x_scale;
            for (int64_t k = start; k < end; k++) {
                int     feat = (int)m.indices[k];
                int64_t wq   = weight_int(m, k);   // fixed: full int64_t index, no truncation

                if (feat == BIAS_COL && m.bias_flag) {
                    // _bias: the bias activation is exactly 1.0, so add the
                    // dequantized weight directly rather than quantizing it.
                    float contrib = dequant_standard(m, k, r);
                    if (contrib != 0.f)
                        for (int i = 0; i < batch_n; i++)
                            row_scores[i] += contrib;
                    continue;
                }

                if (wq == 0) continue;
                for (int i = 0; i < batch_n; i++) {
                    int64_t xq = (int64_t)X_int8[(size_t)i * cols + feat];
                    row_scores[i] += (float)((double)(wq * xq) * w_scale);
                }
            }
        } else {
            int sid_base = m.group_offset[r];
            for (int g = 0; g < nnz_r; g += m.group_size) {
                int gend   = min(g + m.group_size, nnz_r);
                int sid    = sid_base + g / m.group_size;
                double sc   = m.scales[sid];
                double w_sc = sc * (double)x_scale;
                for (int k = g; k < gend; k++) {
                    int     feat = (int)m.indices[start + k];
                    int64_t wq   = weight_int(m, start + k);   // fixed: full int64_t index

                    if (feat == BIAS_COL && m.bias_flag) {
                        // _bias: the bias activation is exactly 1.0, so add the
                        // dequantized weight directly rather than quantizing it.
                        float contrib = dequant_standard(m, start + k, sid);
                        if (contrib != 0.f)
                            for (int i = 0; i < batch_n; i++)
                                row_scores[i] += contrib;
                        continue;
                    }

                    if (wq == 0) continue;
                    for (int i = 0; i < batch_n; i++) {
                        int64_t xq = (int64_t)X_int8[(size_t)i * cols + feat];
                        row_scores[i] += (float)((double)(wq * xq) * w_sc);
                    }
                }
            }
        }
    }
}


// ============================================================
// TOPK and OUTPUT
// ============================================================
void write_topk(ofstream& out, const vector<float>& scores,
                int batch_n, int L, int topk) {
    for (int i = 0; i < batch_n; i++) {
        vector<pair<float,int>> sv(L);
        for (int r = 0; r < L; r++)
            sv[r] = {scores[(size_t)r * batch_n + i], r};
        partial_sort(sv.begin(), sv.begin() + topk, sv.end(),
            [](const pair<float,int>& a, const pair<float,int>& b) {
                return a.first > b.first; });
        for (int k = 0; k < topk; k++)
            out << sv[k].second << ":" << sv[k].first << " ";
        out << "\n";
    }
}


// ============================================================
// MAIN INFERENCE LOOP
// ============================================================
void run_inference(Model& m, const TestData& td,
                   const string& out_path,
                   int topk, bool act_quant, int batch_size) {
    int N    = (int)td.X.size();
    int cols = m.cols;
    int L    = m.rows;

    cerr << "[Inference] "
         << (act_quant ? "PATH B (INT8 query)" : "PATH A (float, BLAS)")
         << " N=" << N << " L=" << L << " cols=" << cols
         << " batch=" << batch_size << endl;

    // PATH A: build Eigen sparse once
    if (!act_quant && !m.eigen_built)
        build_eigen_sparse(m);

    // PATH B: preserve the original global activation scaling method.
    // act_bits selects the symmetric activation range: 127 for INT8, 7 for INT4.
    int act_Q = (1 << (m.act_bits - 1)) - 1;
    if (act_Q < 1) act_Q = 1;

    float x_scale = 1.0f;
    if (act_quant) {
        float absmax = 0.f;
        for (auto& feats: td.X)
            for (auto& [fid, val]: feats)
                absmax = max(absmax, fabsf(val));

        absmax  = max(absmax, 1.0f);
        x_scale = absmax / (float)act_Q;

        cerr << "[Scale] x_scale=" << x_scale
             << " (absmax=" << absmax
             << ", act_bits=" << m.act_bits << " (Q=" << act_Q << "))" << endl;
    }

    // Set Eigen threads (only affects PATH A BLAS multiply)
    Eigen::setNbThreads(omp_get_max_threads());

    ofstream out(out_path);
    out << N << " " << topk << "\n";

    int n_batches = (N + batch_size - 1) / batch_size;

    for (int b = 0; b < n_batches; b++) {
        int batch_start = b * batch_size;
        int batch_end   = min(N, batch_start + batch_size);
        int batch_n     = batch_end - batch_start;

        cerr << "[Batch] " << b+1 << "/" << n_batches
             << " (samples " << batch_start << "-" << batch_end << ")" << endl;

        // Build dense X batch
        DenseMatF X_batch = build_batch(td, batch_start, batch_n, cols);

        // Quantize every activation, including the bias column, with the
        // same scale and range. _bias configurations bypass this value only
        // when adding the bias contribution directly in path_b_batch().
        vector<int8_t> X_int8;
        if (act_quant) {
            X_int8.resize((size_t)batch_n * cols, 0);
            for (int i = 0; i < batch_n; i++)
                for (int c = 0; c < cols; c++) {
                    int v = (int)roundf(X_batch(i, c) / x_scale);
                    X_int8[(size_t)i * cols + c] =
                        (int8_t)max(-act_Q, min(act_Q, v));
                }
        }

        // Scores matrix: L Ã— batch_n (row-major)
        vector<float> scores((size_t)L * batch_n, 0.0f);

        if (!act_quant)
            path_a_batch(m, X_batch, scores, batch_n);
        else
            path_b_batch(m, X_batch, X_int8, x_scale, scores, batch_n);

        write_topk(out, scores, batch_n, L, topk);
    }

    cerr << "[Output] Written: " << out_path << endl;
}


// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0]
             << " model.npz test.txt output.txt"
             << " [--topk N] [--nthreads N]"
             << " [--batch_size N] [--act_quant]\n"
             << "\n"
             << "batch_size guidance:\n"
             << "  Small datasets (EURLex, Wiki10): 50000 (default auto)\n"
             << "  AmazonCat-13K, Amazon-670K:      10000\n"
             << "  Delicious-200K:                   5000\n"
             << "  Amazon-3M:                        1000\n";
        return 1;
    }

    string model_path = argv[1];
    string test_path  = argv[2];
    string out_path   = argv[3];
    int    topk       = 5;
    int    nthreads   = -1;
    bool   act_quant  = false;
    int    batch_size = -1;   // -1 = auto

    for (int i = 4; i < argc; i++) {
        string a = argv[i];
        if (a == "--topk"       && i+1 < argc) topk       = stoi(argv[++i]);
        if (a == "--nthreads"   && i+1 < argc) nthreads   = stoi(argv[++i]);
        if (a == "--batch_size" && i+1 < argc) batch_size = stoi(argv[++i]);
        if (a == "--act_quant")                act_quant  = true;
    }

    if (nthreads > 0) {
        omp_set_num_threads(nthreads);
        cerr << "[OMP] " << nthreads << " threads" << endl;
    }

    auto t0 = hrc::now();

    cerr << "[Load] Model: " << model_path << endl;
    Model model = load_model(model_path);

    cerr << "[Load] Test: " << test_path << endl;
    TestData td = load_test(test_path);

    // Auto batch size if not specified
    if (batch_size <= 0)
        batch_size = auto_batch_size(model.rows);

    size_t scores_mb = (size_t)model.rows * batch_size * sizeof(float) / 1024 / 1024;
    cerr << "[Memory] scores per batch: " << scores_mb << " MB" << endl;

    auto t1 = hrc::now();
    run_inference(model, td, out_path, topk, act_quant, batch_size);

    double infer_s = chrono::duration<double>(hrc::now() - t1).count();
    double total_s = chrono::duration<double>(hrc::now() - t0).count();
    int    N       = (int)td.X.size();

    cerr << "[Timing] inference=" << infer_s << " sec" << endl;
    cerr << "[Timing] throughput=" << (int)(N / infer_s) << " samples/sec" << endl;
    cerr << "[Total]  " << total_s << " sec" << endl;
    return 0;
}

