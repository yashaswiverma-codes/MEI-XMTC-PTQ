#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <numeric>
#include <queue>
#include <immintrin.h>
#include <cnpy.h>
#include <omp.h>
#include <cblas.h>
#include "int_kernels.h"
#include "activation_quant.h"
using namespace std;
using hrc = chrono::high_resolution_clock;
static inline int aligned_size(int n) { return (n + 7) & ~7; }
static inline int aligned32(int n)    { return (n + 31) & ~31; }
// ============================================================
// SIMD helpers
// ============================================================
inline float dot_f32_u(const float* __restrict__ a,
                       const float* __restrict__ b, int n){
    __m256 acc=_mm256_setzero_ps();
    for(int i=0;i<n;i+=8)
        acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_load_ps(b+i),acc);
    alignas(32) float t[8]; _mm256_store_ps(t,acc);
    return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
}
inline int32_t dot_int8_avx2(const int8_t* __restrict__ w,
                              const int8_t* __restrict__ z, int n){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<n;i+=32){
        __m256i wv=_mm256_loadu_si256((__m256i*)(w+i));
        __m256i zv=_mm256_loadu_si256((__m256i*)(z+i));
        __m256i wlo=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv,0));
        __m256i whi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv,1));
        __m256i zlo=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(zv,0));
        __m256i zhi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(zv,1));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(wlo,zlo));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(whi,zhi));
    }
    __m128i lo=_mm256_extracti128_si256(acc,0);
    __m128i hi=_mm256_extracti128_si256(acc,1);
    __m128i s=_mm_add_epi32(lo,hi);
    s=_mm_add_epi32(s,_mm_srli_si128(s,8));
    s=_mm_add_epi32(s,_mm_srli_si128(s,4));
    return _mm_cvtsi128_si32(s);
}
// ============================================================
// Model structures
// ============================================================
struct PartEntry { int cluster_id; float weight; };
struct WMatEntry { int feat_id; vector<float> evec; };
struct Learner {
    vector<vector<PartEntry>>  w_index;
    vector<vector<WMatEntry>>  w_mat_vec;
    vector<unordered_map<int,const float*>> feat_map;
    int embed_size=0, num_data=0;
    vector<vector<size_t>> cluster_assign;
};
struct AnnexModel {
    vector<Learner>      learners;
    vector<vector<int>>  labels;
    int num_learner=0, embed_size=0;
};
// ============================================================
// Quantized cluster embedding storage
// ============================================================
struct QCluster {
    bool   fp32_mode=false;
    bool   emb_grouped=false;
    int    num_egroups=0, emb_group_size=32, emb_ncols=0, AS=0;
    vector<float>   emb_local_fp32;
    vector<int8_t>  emb_local_int8;
    vector<float>   emb_local_gs;
    vector<float>   emb_local_row_sc;
};
// ============================================================
// Binary helpers
// ============================================================
static size_t rs(FILE* fp){size_t v;fread(&v,8,1,fp);return v;}
static int    ri(FILE* fp){int    v;fread(&v,4,1,fp);return v;}
static float  rf(FILE* fp){float  v;fread(&v,4,1,fp);return v;}
static void skip_param(FILE* fp){
    fseek(fp,4*7,SEEK_CUR);fseek(fp,4*3,SEEK_CUR);
    fseek(fp,4*2,SEEK_CUR);fseek(fp,4*1,SEEK_CUR);
    fseek(fp,4*3,SEEK_CUR);
    for(int i=0;i<4;i++){int sl;fread(&sl,4,1,fp);fseek(fp,sl,SEEK_CUR);}
}
// ============================================================
// Load model
// ============================================================
AnnexModel load_model(const string& path){
    AnnexModel m;
    FILE* fp=fopen(path.c_str(),"rb");
    if(!fp) throw runtime_error("Cannot open: "+path);
    skip_param(fp);
    size_t ne=rs(fp); m.labels.resize(ne);
    for(size_t i=0;i<ne;i++){
        size_t n=rs(fp); m.labels[i].resize(n);
        for(size_t j=0;j<n;j++) m.labels[i][j]=ri(fp);
    }
    size_t nl=rs(fp); m.num_learner=(int)nl; m.learners.resize(nl);
    for(size_t l=0;l<nl;l++){
        Learner& lr=m.learners[l];
        rs(fp);
        size_t nf=rs(fp); lr.w_index.resize(nf);
        for(size_t i=0;i<nf;i++){
            size_t np=rs(fp); lr.w_index[i].resize(np);
            for(size_t j=0;j<np;j++){
                lr.w_index[i][j].cluster_id=ri(fp);
                lr.w_index[i][j].weight=rf(fp);
            }
        }
        size_t es=rs(fp); lr.embed_size=(int)es;
        ri(fp); ri(fp);
        if(m.embed_size==0) m.embed_size=(int)es;
        size_t nc=rs(fp); lr.w_mat_vec.resize(nc);
        for(size_t c=0;c<nc;c++){
            size_t nent=rs(fp); lr.w_mat_vec[c].resize(nent);
            for(size_t j=0;j<nent;j++){
                lr.w_mat_vec[c][j].feat_id=ri(fp);
                size_t el=rs(fp); lr.w_mat_vec[c][j].evec.resize(el);
                for(size_t k=0;k<el;k++) lr.w_mat_vec[c][j].evec[k]=rf(fp);
            }
        }
        lr.feat_map.resize(nc);
        for(size_t c=0;c<nc;c++){
            lr.feat_map[c].reserve(lr.w_mat_vec[c].size());
            for(auto& e:lr.w_mat_vec[c])
                lr.feat_map[c][e.feat_id]=e.evec.data();
        }
        size_t nd=rs(fp); lr.num_data=(int)nd;
        for(size_t i=0;i<nd;i++){
            size_t el=rs(fp); fseek(fp,el*sizeof(float),SEEK_CUR);
        }
        size_t nca=rs(fp); lr.cluster_assign.resize(nca);
        for(size_t c=0;c<nca;c++){
            size_t ni=rs(fp); lr.cluster_assign[c].resize(ni);
            for(size_t j=0;j<ni;j++) lr.cluster_assign[c][j]=rs(fp);
        }
        cerr<<"Learner "<<l<<" clusters="<<nc<<" data="<<nd<<endl;
    }
    fclose(fp);
    cerr<<"[Model] embed_size="<<m.embed_size<<" learners="<<m.num_learner<<endl;
    return m;
}
// ============================================================
// Load quantized embeddings from NPZ
// ============================================================
vector<vector<QCluster>> load_npz(const string& npz_dir, const AnnexModel& model){
    int NL=model.num_learner;
    int ES=model.embed_size;
    int AS_g=aligned_size(ES);
    vector<vector<QCluster>> qclusters(NL);
    for(int l=0;l<NL;l++){
        int nc=(int)model.learners[l].cluster_assign.size();
        int nd=model.learners[l].num_data;
        qclusters[l].resize(nc);
        bool emb_loaded=false, is_fp32=false;
        vector<float>  l_emb_fp32;
        vector<int8_t> l_emb_int8;
        vector<float>  l_emb_gs;
        int l_num_eg=0, l_eg32=32, l_emb_nc=ES;
        bool l_emb_grouped=false;
        for(int c=0;c<nc;c++){
            string path=npz_dir+"/learner"+to_string(l)+"_cluster"+to_string(c)+".npz";
            if(!ifstream(path).good()){cerr<<"[WARN] Missing "<<path<<endl;continue;}
            cnpy::npz_t npz=cnpy::npz_load(path);
            QCluster& qc=qclusters[l][c];
            qc.AS=AS_g;
            int bits=(int)npz.at("bits").as_vec<int32_t>()[0];
            bool this_fp32=(bits==32);
            if(c==0) is_fp32=this_fp32;
            qc.fp32_mode=this_fp32;
            if(!emb_loaded && npz.count("emb_data")){
                int be=(int)npz.at("bits_emb").as_vec<int32_t>()[0];
                int mode_i=(int)npz.at("mode_int").as_vec<int32_t>()[0];
                int emb_nc=(int)npz.at("emb_ncols").as_vec<int32_t>()[0];
                l_emb_nc=emb_nc;
                auto& darr_e=npz.at("emb_data");
                if(be==32){
                    l_emb_fp32.assign((size_t)nd*AS_g,0.f);
                    if(npz.count("emb_fp32")){
                        auto& ef=npz.at("emb_fp32");
                        const float* src=(const float*)ef.data_holder->data();
                        for(int i=0;i<nd;i++)
                            memcpy(l_emb_fp32.data()+(size_t)i*AS_g,
                                   src+(size_t)i*emb_nc,emb_nc*sizeof(float));
                    } else {
                        auto sc=npz.at("emb_scales").as_vec<float>();
                        const int32_t* src=(const int32_t*)darr_e.data_holder->data();
                        for(int i=0;i<nd;i++){
                            float s=(i<(int)sc.size())?sc[i]:sc[0];
                            float* dst=l_emb_fp32.data()+(size_t)i*AS_g;
                            for(int d=0;d<emb_nc;d++) dst[d]=float(src[(size_t)i*emb_nc+d])*s;
                        }
                    }
                    l_emb_grouped=false; l_num_eg=1;
                } else if(npz.count("emb_group_scales")){
                    // INT grouped
                    l_emb_grouped=true;
                    l_emb_int8.assign((size_t)nd*AS_g,0);
                    const uint8_t* src=(const uint8_t*)darr_e.data_holder->data();
                    if(be==4){
                        int srb=(emb_nc+1)/2;
                        for(int i=0;i<nd;i++){
                            const uint8_t* rs2=src+(size_t)i*srb;
                            int8_t* rd=l_emb_int8.data()+(size_t)i*AS_g;
                            for(int d=0;d<emb_nc;d++){
                                uint8_t byte=rs2[d/2];
                                uint8_t nib=(d%2==0)?(byte&0x0F):(byte>>4);
                                int8_t sv=int8_t(nib); if(sv>7) sv=int8_t(int(nib)-16);
                                rd[d]=sv;
                            }
                        }
                    } else {
                        const int8_t* src8=(const int8_t*)src;
                        for(int i=0;i<nd;i++)
                            memcpy(l_emb_int8.data()+(size_t)i*AS_g,
                                   src8+(size_t)i*emb_nc,emb_nc*sizeof(int8_t));
                    }
                    auto gs_raw=npz.at("emb_group_scales").as_vec<uint16_t>();
                    l_num_eg=(int)npz.at("emb_num_groups").as_vec<int32_t>()[0];
                    l_emb_gs.resize(gs_raw.size());
                    for(size_t gi=0;gi<gs_raw.size();gi++) l_emb_gs[gi]=_cvtsh_ss(gs_raw[gi]);
                    if(npz.count("group_size")) l_eg32=(int)npz.at("group_size").as_vec<int32_t>()[0];
                } else {
                    // INT non-grouped (sym or asym)
                    l_emb_grouped=false;
                    auto emb_sc=npz.at("emb_scales").as_vec<float>();
                    vector<double> emb_zp_v;
                    if(npz.count("emb_zp")) emb_zp_v=npz.at("emb_zp").as_vec<double>();
                    bool emb_asym=(mode_i!=0);
                    const uint8_t* src=(const uint8_t*)darr_e.data_holder->data();
                    l_emb_int8.assign((size_t)nd*AS_g,0);
                    l_emb_gs.resize(nd);
                    int srb=(be==4)?(emb_nc+1)/2:emb_nc;
                    for(int i=0;i<nd;i++){
                        float sc=(i<(int)emb_sc.size())?emb_sc[i]:emb_sc[0];
                        float zp=(float)((!emb_zp_v.empty())?
                            (i<(int)emb_zp_v.size()?emb_zp_v[i]:emb_zp_v[0]):0.0);
                        float absmax=0.f;
                        vector<float> tmp(emb_nc);
                        if(be==4 && emb_asym){
                            const uint8_t* rs2=src+(size_t)i*srb;
                            for(int d=0;d<emb_nc;d++){
                                uint8_t byte=rs2[d/2];
                                uint8_t nib=(d%2==0)?(byte&0x0F):(byte>>4);
                                tmp[d]=float(nib)*sc+zp;
                                absmax=max(absmax,fabs(tmp[d]));
                            }
                        } else if(be==4){
                            const uint8_t* rs2=src+(size_t)i*srb;
                            for(int d=0;d<emb_nc;d++){
                                uint8_t byte=rs2[d/2];
                                uint8_t nib=(d%2==0)?(byte&0x0F):(byte>>4);
                                int8_t sv=int8_t(nib); if(sv>7) sv=int8_t(int(nib)-16);
                                tmp[d]=float(sv)*sc;
                                absmax=max(absmax,fabs(tmp[d]));
                            }
                        } else if(emb_asym){
                            const uint8_t* rs2=src+(size_t)i*srb;
                            for(int d=0;d<emb_nc;d++){
                                tmp[d]=float(rs2[d])*sc+zp;
                                absmax=max(absmax,fabs(tmp[d]));
                            }
                        } else {
                            const int8_t* rs2=(const int8_t*)src+(size_t)i*srb;
                            for(int d=0;d<emb_nc;d++){
                                tmp[d]=float(rs2[d])*sc;
                                absmax=max(absmax,fabs(tmp[d]));
                            }
                        }
                        if(absmax==0.f) absmax=1.f;
                        float newsc=absmax/127.f;
                        l_emb_gs[i]=newsc;
                        int8_t* dst=l_emb_int8.data()+(size_t)i*AS_g;
                        for(int d=0;d<emb_nc;d++){
                            int q=int(round(tmp[d]/newsc));
                            dst[d]=int8_t(max(-127,min(127,q)));
                        }
                    }
                    l_num_eg=1;
                }
                emb_loaded=true;
                cerr<<"Learner "<<l<<" emb loaded (bits_emb="<<be<<" grouped="<<l_emb_grouped<<")"<<endl;
            }
            qc.emb_grouped=l_emb_grouped;
            qc.num_egroups=l_num_eg;
            qc.emb_group_size=l_eg32;
            qc.emb_ncols=l_emb_nc;
        }
        for(int c=0;c<nc;c++){
            const auto& ca=model.learners[l].cluster_assign[c];
            int n=(int)ca.size();
            if(n==0) continue;
            QCluster& qc=qclusters[l][c];
            if(is_fp32){
                qc.emb_local_fp32.resize((size_t)n*AS_g);
                for(int i=0;i<n;i++)
                    memcpy(qc.emb_local_fp32.data()+(size_t)i*AS_g,
                           l_emb_fp32.data()+(size_t)ca[i]*AS_g,AS_g*sizeof(float));
            } else {
                qc.emb_local_int8.resize((size_t)n*AS_g,0);
                for(int i=0;i<n;i++)
                    memcpy(qc.emb_local_int8.data()+(size_t)i*AS_g,
                           l_emb_int8.data()+(size_t)ca[i]*AS_g,AS_g*sizeof(int8_t));
                if(l_emb_grouped){
                    qc.emb_local_gs.resize((size_t)n*l_num_eg);
                    for(int i=0;i<n;i++)
                        for(int g=0;g<l_num_eg;g++)
                            qc.emb_local_gs[(size_t)i*l_num_eg+g]=
                                l_emb_gs[(size_t)ca[i]*l_num_eg+g];
                } else {
                    qc.emb_local_row_sc.resize(n);
                    for(int i=0;i<n;i++)
                        qc.emb_local_row_sc[i]=l_emb_gs[ca[i]];
                }
            }
        }
        cerr<<"Learner "<<l<<" loaded (fp32="<<is_fp32<<")"<<endl;
    }
    return qclusters;
}
// ============================================================
// Test data
// ============================================================
struct TestData {
    vector<vector<pair<int,float>>> X;
    vector<vector<int>> y;
};
TestData load_test(const string& path){
    TestData td;
    ifstream f(path);
    if(!f) throw runtime_error("Cannot open: "+path);
    string line;
    getline(f,line);
    {istringstream ss(line);int a,b,c;
     if(!(ss>>a>>b>>c)){f.clear();f.seekg(0);}}
    while(getline(f,line)){
        if(line.empty()) continue;
        istringstream ss(line); string tok;
        vector<int> labels; vector<pair<int,float>> feats;
        bool first=true;
        while(ss>>tok){
            if(first){first=false;
                if(tok.find(':')==string::npos){
                    istringstream ls(tok);string lb;
                    while(getline(ls,lb,',')) if(!lb.empty()) labels.push_back(stoi(lb));
                    continue;
                }
            }
            size_t p=tok.find(':');
            if(p!=string::npos) feats.push_back({stoi(tok.substr(0,p)),stof(tok.substr(p+1))});
        }
        float nm=0.f;
        for(auto&[f,v]:feats) nm+=v*v; nm=sqrtf(nm);
        if(nm>0) for(auto&[f,v]:feats) v/=nm;
        td.X.push_back(feats); td.y.push_back(labels);
    }
    cerr<<"[Data] "<<td.X.size()<<" samples loaded"<<endl;
    return td;
}
// ============================================================
// PATH A: Float query scoring
// Dequantize embedding on-the-fly, float dot product
// Used for: row_sym, row_asym, group_sym, group_sym_clip
//           row_sym_clip_mixed (any config WITHOUT --act_quant)
// ============================================================
static inline float score_emb_float(
    const QCluster& qc, int li,
    const float* zf, int AS)
{
    int nc = qc.emb_ncols;
    if(qc.emb_grouped){
        // Dequantize group by group, accumulate float score
        const int8_t* w = qc.emb_local_int8.data() + (size_t)li*AS;
        float score = 0.f;
        for(int g=0; g<qc.num_egroups; g++){
            int s  = g * qc.emb_group_size;
            int e2 = min(nc, s + qc.emb_group_size);
            float gs = qc.emb_local_gs[(size_t)li*qc.num_egroups+g];
            // Use AVX for inner loop if possible
            float group_score = 0.f;
            for(int d=s; d<e2; d++)
                group_score += float(w[d]) * gs * zf[d];
            score += group_score;
        }
        return score;
    } else {
        // Dequantize row, float dot product
        const int8_t* w = qc.emb_local_int8.data() + (size_t)li*AS;
        float sc = qc.emb_local_row_sc[li];
        // Use cblas_sdot equivalent via manual loop (avoids allocation)
        float score = 0.f;
        for(int d=0; d<nc; d++)
            score += float(w[d]) * sc * zf[d];
        return score;
    }
}
// ============================================================
// PATH B: INT8 query scoring (act_quant mode)
// Integer dot product, rescale by act_scale × emb_scale
// Used for: group_sym_clip_act, group_sym_clip_act_intinfer,
//           group_sym_clip_act_intinfer_bias (any config WITH --act_quant)
// ============================================================
static inline float score_emb_int8(
    const QCluster& qc, int li,
    const int8_t* zq, float act_scale, int AS)
{
    int nc = qc.emb_ncols;
    if(qc.fp32_mode){
        // Should not happen in act_quant mode but handle gracefully
        return 0.f;
    }
    if(qc.emb_grouped){
        const int8_t* w = qc.emb_local_int8.data() + (size_t)li*AS;
        float score = 0.f;
        for(int g=0; g<qc.num_egroups; g++){
            int s  = g * qc.emb_group_size;
            int e2 = min(nc, s + qc.emb_group_size);
            int32_t acc = 0;
            int avx_end = s + ((e2-s)/32)*32;
            if(avx_end > s) acc += dot_int8_avx2(w+s, zq+s, avx_end-s);
            for(int d=avx_end; d<e2; d++) acc += int32_t(w[d]) * int32_t(zq[d]);
            float gs = qc.emb_local_gs[(size_t)li*qc.num_egroups+g];
            score += float(acc) * gs * act_scale;
        }
        return score;
    } else {
        const int8_t* w = qc.emb_local_int8.data() + (size_t)li*AS;
        int32_t acc = 0;
        int avx_end = (nc/32)*32;
        if(avx_end > 0) acc += dot_int8_avx2(w, zq, avx_end);
        for(int d=avx_end; d<nc; d++) acc += int32_t(w[d]) * int32_t(zq[d]);
        return float(acc) * qc.emb_local_row_sc[li] * act_scale;
    }
}
// ============================================================
// Project one sample: sparse w_mat → z vector
// ============================================================
static inline void project_sample(
    const Learner& lr, int cluster,
    const vector<pair<int,float>>& feats,
    float* z, int ES)
{
    fill_n(z, aligned_size(ES), 0.f);
    const auto& fmap = lr.feat_map[cluster];
    for(auto&[fid,val]:feats){
        auto it = fmap.find(fid);
        if(it == fmap.end()) continue;
        const float* ev = it->second;
        for(int d=0; d<ES; d++) z[d] += val * ev[d];
    }
    float nm = 0.f;
    for(int d=0; d<ES; d++) nm += z[d]*z[d];
    nm = sqrtf(nm);
    if(nm > 0) for(int d=0; d<ES; d++) z[d] /= nm;
}
// ============================================================
// Main inference
//
// act_quant_mode = false (default):
//   PATH A — Float query inference
//   z stays as float32 after projection+normalization
//   embeddings dequantized on-the-fly before scoring
//   score = z_float · dequant(e_int8)
//   Used for: row_sym, row_asym, group_sym, group_sym_clip,
//             row_sym_clip_mixed (all configs WITHOUT --act_quant)
//
// act_quant_mode = true (--act_quant flag):
//   PATH B — INT8 query inference (fully quantized)
//   z quantized to INT8 after projection+normalization
//   integer dot product with INT8 embeddings
//   score = (z_int8 · e_int8) × act_scale × emb_scale
//   Used for: group_sym_clip_act, group_sym_clip_act_intinfer,
//             group_sym_clip_act_intinfer_bias (all WITH --act_quant)
// ============================================================
void run_inference(
    const AnnexModel& model,
    const vector<vector<QCluster>>& qclusters,
    const TestData& td,
    vector<vector<pair<int,float>>>& predictions,
    int topk,
    bool act_quant_mode)
{
    int N  = (int)td.X.size();
    int ES = model.embed_size;
    int AS = aligned_size(ES);
    cerr << "[Inference] mode="
         << (act_quant_mode ? "INT8-query (act_quant)" : "Float-query (weight-only)")
         << endl;
    vector<unordered_map<int,int>> votes(N);
    for(int l=0; l<model.num_learner; l++){
        const Learner& lr = model.learners[l];
        int nc = (int)lr.cluster_assign.size();
        // ====================================================
        // STEP 1: Assign all samples to clusters (parallel)
        // ====================================================
        vector<int> cls_assign(N);
        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<N; i++){
            unordered_map<int,float> ip;
            for(auto&[fid,val]:td.X[i]){
                if(fid >= (int)lr.w_index.size()) continue;
                for(auto& pe:lr.w_index[fid])
                    ip[pe.cluster_id] += val * pe.weight;
            }
            int best_c=0; float best_v=-1e30f;
            for(auto&[c,v]:ip) if(v>best_v){best_v=v; best_c=c;}
            cls_assign[i] = best_c;
        }
        // ====================================================
        // STEP 2: Group samples by cluster
        // ====================================================
        vector<vector<int>> cluster_samples(nc);
        for(int i=0; i<N; i++)
            cluster_samples[cls_assign[i]].push_back(i);
        // ====================================================
        // STEP 3: Score each cluster's samples
        // ====================================================
        for(int c=0; c<nc; c++){
            const auto& samples = cluster_samples[c];
            if(samples.empty()) continue;
            int ns = (int)samples.size();
            const QCluster& qc = qclusters[l][c];
            int n_emb = (int)lr.cluster_assign[c].size();
            if(n_emb == 0) continue;
            if(qc.fp32_mode){
                // ============================================
                // FP32 embeddings: cblas_sgemv (unchanged)
                // ============================================
                #pragma omp parallel for schedule(static)
                for(int si=0; si<ns; si++){
                    int i = samples[si];
                    alignas(32) float z[64]={};
                    project_sample(lr, c, td.X[i], z, ES);
                    vector<float> scores(n_emb);
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        n_emb, AS,
                        1.0f,
                        qc.emb_local_fp32.data(), AS,
                        z, 1,
                        0.0f,
                        scores.data(), 1);
                    int k = min(10, n_emb);
                    vector<pair<float,int>> sc(n_emb);
                    for(int e=0; e<n_emb; e++) sc[e] = {scores[e], e};
                    partial_sort(sc.begin(), sc.begin()+k, sc.end(),
                        [](const pair<float,int>& a, const pair<float,int>& b){
                            return a.first > b.first;});
                    for(int ki=0; ki<k; ki++){
                        size_t idx = lr.cluster_assign[c][sc[ki].second];
                        for(int lbl:model.labels[idx]) votes[i][lbl]++;
                    }
                }
            } else if(!act_quant_mode){
                // ============================================
                // PATH A: Float query, dequantize embedding
                // For: row_sym, row_asym, group_sym,
                //      group_sym_clip, row_sym_clip_mixed
                //      (any config WITHOUT --act_quant)
                // z stays float32, embedding dequantized
                // score = z_float · dequant(e_int8)
                // ============================================
                #pragma omp parallel for schedule(static)
                for(int si=0; si<ns; si++){
                    int i = samples[si];
                    alignas(32) float z[64]={};
                    project_sample(lr, c, td.X[i], z, ES);
                    // z is normalized float — NOT quantized
                    int k = min(10, n_emb);
                    vector<pair<float,int>> sc(n_emb);
                    for(int e=0; e<n_emb; e++)
                        sc[e] = {score_emb_float(qc, e, z, AS), e};
                    partial_sort(sc.begin(), sc.begin()+k, sc.end(),
                        [](const pair<float,int>& a, const pair<float,int>& b){
                            return a.first > b.first;});
                    for(int ki=0; ki<k; ki++){
                        size_t idx = lr.cluster_assign[c][sc[ki].second];
                        for(int lbl:model.labels[idx]) votes[i][lbl]++;
                    }
                }
            } else {
                // ============================================
                // PATH B: INT8 query, integer dot product
                // For: group_sym_clip_act,
                //      group_sym_clip_act_intinfer,
                //      group_sym_clip_act_intinfer_bias
                //      (any config WITH --act_quant flag)
                // z quantized to INT8, integer scoring
                // score = (z_int8 · e_int8) × scales
                // ============================================
                #pragma omp parallel for schedule(static)
                for(int si=0; si<ns; si++){
                    int i = samples[si];
                    alignas(32) float  z[64]={};
                    alignas(32) int8_t zq[64]={};
                    project_sample(lr, c, td.X[i], z, ES);
                    // Quantize activation z → zq (INT8)
                    float absmax = 0.f;
                    for(int d=0; d<ES; d++) absmax = max(absmax, fabs(z[d]));
                    if(absmax == 0.f) absmax = 1.f;
                    float act_scale = absmax / 127.f;
                    for(int d=0; d<ES; d++){
                        int v = int(round(z[d] / act_scale));
                        zq[d] = int8_t(max(-127, min(127, v)));
                    }
                    for(int d=ES; d<AS; d++) zq[d] = 0;
                    // Integer scoring
                    int k = min(10, n_emb);
                    vector<pair<float,int>> sc(n_emb);
                    for(int e=0; e<n_emb; e++)
                        sc[e] = {score_emb_int8(qc, e, zq, act_scale, AS), e};
                    partial_sort(sc.begin(), sc.begin()+k, sc.end(),
                        [](const pair<float,int>& a, const pair<float,int>& b){
                            return a.first > b.first;});
                    for(int ki=0; ki<k; ki++){
                        size_t idx = lr.cluster_assign[c][sc[ki].second];
                        for(int lbl:model.labels[idx]) votes[i][lbl]++;
                    }
                }
            }
        }
        cerr<<"Learner "<<l<<" done"<<endl;
    }
    // Collect top-k predictions
    predictions.resize(N);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<N; i++){
        vector<pair<int,float>> v;
        for(auto&[lb,cnt]:votes[i]) v.push_back({lb,(float)cnt});
        partial_sort(v.begin(), v.begin()+min(topk,(int)v.size()), v.end(),
            [](const pair<int,float>& a, const pair<int,float>& b){
                return a.second > b.second;});
        if((int)v.size() > topk) v.resize(topk);
        predictions[i] = move(v);
    }
}
// ============================================================
// Main
// ============================================================
int main(int argc, char** argv){
    if(argc < 5){
        cerr << "Usage: " << argv[0]
             << " <model.bin> <test.txt> <output.txt>"
             << " --npz_dir <dir>"
             << " [--topk N]"
             << " [--nthreads N]"
             << " [--act_quant]"
             << "\n\n"
             << "Inference modes:\n"
             << "  (default)    Float-query: z stays float32, embeddings dequantized\n"
             << "               Use for: row_sym, row_asym, group_sym, group_sym_clip,\n"
             << "                        row_sym_clip_mixed\n"
             << "  --act_quant  INT8-query: z quantized to INT8, integer dot product\n"
             << "               Use for: group_sym_clip_act, group_sym_clip_act_intinfer,\n"
             << "                        group_sym_clip_act_intinfer_bias\n";
        return 1;
    }
    string model_path=argv[1], test_file=argv[2], output_file=argv[3];
    string npz_dir="";
    int topk=5, nthreads=-1;
    bool act_quant_mode=false;
    for(int i=4; i<argc; i++){
        string a=argv[i];
        if(a=="--npz_dir"   && i+1<argc) npz_dir        = argv[++i];
        if(a=="--topk"      && i+1<argc) topk           = stoi(argv[++i]);
        if(a=="--nthreads"  && i+1<argc) nthreads       = stoi(argv[++i]);
        if(a=="--act_quant")             act_quant_mode  = true;
    }
    if(nthreads > 0){
        omp_set_num_threads(nthreads);
        cerr<<"[OMP] Using "<<nthreads<<" threads"<<endl;
    } else {
        cerr<<"[OMP] Using "<<omp_get_max_threads()<<" threads"<<endl;
    }
    // openblas_set_num_threads(1); // not available with cblas
    cerr<<"[Mode] act_quant="<<(act_quant_mode?"true (INT8-query)":"false (Float-query)")<<endl;
    auto t0=hrc::now();
    cerr<<"[Model] Loading "<<model_path<<"..."<<endl;
    AnnexModel model=load_model(model_path);
    cerr<<"[NPZ] Loading from "<<npz_dir<<"..."<<endl;
    auto qclusters=load_npz(npz_dir, model);
    cerr<<"[Test] Loading "<<test_file<<"..."<<endl;
    TestData td=load_test(test_file);
    int N=(int)td.X.size();
    vector<vector<pair<int,float>>> predictions;
    auto t1=hrc::now();
    run_inference(model, qclusters, td, predictions, topk, act_quant_mode);
    double infer_time=chrono::duration<double>(hrc::now()-t1).count();
    cerr<<"[Timing] inference="<<infer_time<<" sec"<<endl;
    cerr<<"[Timing] throughput="<<int(N/infer_time)<<" samples/sec"<<endl;
    ofstream out(output_file);
    for(int i=0; i<N; i++){
        for(auto&[l,s]:predictions[i]) out<<l<<":"<<s<<" ";
        out<<"\n";
    }
    cerr<<"Predictions written to "<<output_file<<endl;
    cerr<<"[TOTAL] "<<chrono::duration<double>(hrc::now()-t0).count()<<" sec"<<endl;
    return 0;
}
