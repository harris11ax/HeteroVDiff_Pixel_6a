// npu_block — T_N harness: a token-tiled NNAPI FULLY_CONNECTED chain for ONE DiT block.
//
// Why this exists
// ---------------
// T_N (the NPU per-block stage rate) is the missing half of the block cut. Gate 0
// measured a single FC at M=384 on `google-edgetpu` (37.43 ms, 282 GOPS) and found
// a hard per-graph activation ceiling of ~384 tokens: M=448 -> OP_FAILED, M>=512 ->
// DEAD_OBJECT. So Preset A (M=1530) must be tiled 4x on the token axis, and the open
// question (Gate 0 open question #2) is whether a *whole block's* 10 GEMMs can live
// in one <=384-token graph or must be one compilation per GEMM.
//
// This binary answers that and times it:
//   chain  — one model, 10 FC ops chained (out of op i -> in of op i+1), M tokens/tile,
//            `tiles` fresh executions. Per-dispatch overhead paid once per tile.
//   single — 10 separate one-op models (the fallback: 40 dispatches/block).
//   support— build + validate the model, dump the per-op Edge-TPU support vector. No
//            compile, so it cannot take the HAL down.
//   down2 / down4 — the `ffn.down` workaround: `ffn.down` (N=1536, K=8960) does not compile
//            on this HAL at all (DEAD_OBJECT after ~62 s / OP_FAILED after 47 s, both
//            reproduced fresh). Split the K dimension into 2 x 4480 or 4 x 2240 FCs and
//            accumulate with ANEURALNETWORKS_ADD. Same MAC count, same output shape; the
//            extra cost is one ADD per extra chunk. Weights for the split layout come from
//            a separate chunk-major blob (the row-major [N_out,K] blob cannot be sliced
//            along K without a re-layout) — synthetic LCG like the main blob, since the
//            real INT8 artifact does not exist yet. Pass iters=0 for a compile-only probe.
//
// Weights are a CONSTANT operand served from ONE file-mapped INT8 blob
// (/data/local/tmp/hv_block_int8.bin, 46,399,488 B = the block's 10 base weight
// matrices concatenated). That is route B (createFromFd), i.e. the same one-page-cache
// artifact the GPU imports — one stored copy, no second buffer. Per-output-column
// scales go in as SYMM_PER_CHANNEL (channelDim = 0), the production quantisation.
//
// Traps already paid for (docs/device/RESULTS_gate0.md):
//   * FC takes EXACTLY 4 inputs {input, weights[N_out,K], bias[N_out], activation};
//     bias is mandatory, there is no activationParams operand.
//   * You must call addOperation; operands + identifyInputsAndOutputs is not a model.
//   * Weights must be a constant operand via setOperandValueFromMemory.
//   * A reused Execution returns BAD_STATE -> a fresh Execution per compute.
//   * NEVER submit a graph above the measured token ceiling. Fresh process per config.
//
// Build/run: tools/android/npublock/build.sh
//            adb shell /data/local/tmp/npu_block chain 4 3 384

#include <android/neuralnetworks.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <algorithm>
#include <vector>

static const char* W_PATH = "/data/local/tmp/hv_block_int8.bin";
// Chunk-major copy of the same 13,762,560 B of `ffn.down` weights, laid out as
// `nchunks` contiguous [1536, 8960/nchunks] matrices so each chunk is a legal
// constant operand. Synthetic LCG fill, same as W_PATH.
static const char* W_DOWN_PATH = "/data/local/tmp/hv_down_chunks.bin";
static const int DOWN_N = 1536;
static const int DOWN_K = 8960;

// VAE mid-block attention — the last unmeasured term of `T_V`. The attention is
// 6120 tokens x 384 ch: QK^T and AV are [M=384 tile, N=6120, K=384] and the four
// projections (q,k,v,o) are [M=384, N=384, K=384]. Both matrices live in ONE
// file-mapped blob at fixed offsets, same route-B discipline as the block blob.
// LCG fill (latency is shape-determined; the real INT8 artifact is Phase 1).
static const char* W_ATTN_PATH = "/data/local/tmp/hv_attn_int8.bin";
static const int ATTN_SCORE_N = 6120;
static const int ATTN_SCORE_K = 384;
static const int ATTN_PROJ_N  = 384;
static const int ATTN_PROJ_K  = 384;
static const int64_t ATTN_SCORE_OFF = 0;
static const int64_t ATTN_PROJ_OFF  = (int64_t)ATTN_SCORE_N * (int64_t)ATTN_SCORE_K;
static const int64_t ATTN_RAW_BYTES = ATTN_PROJ_OFF + (int64_t)ATTN_PROJ_N * (int64_t)ATTN_PROJ_K;
static const int64_t ATTN_BYTES = (ATTN_RAW_BYTES + 4095) / 4096 * 4096;   // page-aligned

// The Preset-A DiT block's 10 base GEMMs, in execution order. Layout is row-major
// [N_out, K] INT8 — the layout both arm_dot (Mali) and NNAPI FC want.
struct Gemm { int N; int K; int64_t off; const char* name; };

static const Gemm GEMMS[10] = {
    {1536, 1536,          0, "attn1.to_q"},
    {1536, 1536,    2359296, "attn1.to_k"},
    {1536, 1536,    4718592, "attn1.to_v"},
    {1536, 1536,    7077888, "attn1.to_out"},
    {1536, 1536,    9437184, "attn2.to_q"},
    {1536, 1536,   11796480, "attn2.to_k"},
    {1536, 1536,   14155776, "attn2.to_v"},
    {1536, 1536,   16515072, "attn2.to_out"},
    {8960, 1536,   18874368, "ffn.up"},
    {1536, 8960,   32636928, "ffn.down"},
};
// 8 * 1536*1536 + 2 * 8960*1536 = 46,399,488 bytes == the block's INT8 base weights.
static const int64_t W_BYTES = 46399488;
static const int64_t BLOCK_MACS_PER_TOKEN = 46399488;  // base GEMM MACs, one token

static const float SA = 1.0f / 127.0f;   // activation scale (per-GEMM calibrated in prod)
static const float SW = 1.0f / 127.0f;   // nominal per-output-column weight scale

// 1 = per-output-column scales (SYMM_PER_CHANNEL, the locked production scheme)
// 0 = per-tensor scale (what every Gate-0 FC measurement actually used)
static int G_PC = 1;

static int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static void P(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n"); fflush(stdout);
}
static const char* rcs(int rc) {
    switch (rc) {
        case 0: return "NO_ERROR";
        case 1: return "OUT_OF_MEMORY";
        case 2: return "INCOMPLETE";
        case 3: return "UNEXPECTED_NULL";
        case 4: return "BAD_DATA";
        case 5: return "OP_FAILED";
        case 6: return "BAD_STATE";
        case 7: return "UNMAPPABLE";
        case 8: return "OUTPUT_INSUFFICIENT_SIZE";
        case 9: return "UNAVAILABLE_DEVICE";
        case 10: return "DEAD_OBJECT";
        default: return "OTHER";
    }
}

// ---------------------------------------------------------------- weight blob
// One file-mapped INT8 blob for the whole block. Values are a deterministic
// pseudo-random fill: this is a *latency* harness and the real INT8 artifact does
// not exist yet (Phase 1 quantiser outstanding), so the numbers here are shape-
// determined, not value-determined. Stated in the report.
static int64_t ensure_weights(const char** why) {
    struct stat st;
    if (stat(W_PATH, &st) == 0 && st.st_size == W_BYTES) return 0;
    int fd = open(W_PATH, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { *why = "wfile_open"; return -1; }
    if (ftruncate(fd, (off_t)W_BYTES) != 0) { *why = "wfile_trunc"; close(fd); return -1; }
    void* p = mmap(nullptr, (size_t)W_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { *why = "wfile_mmap"; close(fd); return -1; }
    int8_t* q = (int8_t*)p;
    uint32_t x = 0x12345678u;
    for (int64_t i = 0; i < W_BYTES; ++i) {
        x = x * 1664525u + 1013904223u;
        q[i] = (int8_t)((x >> 24) & 0xFF);          // uniform in [-128,127]
    }
    msync(p, (size_t)W_BYTES, MS_SYNC);
    munmap(p, (size_t)W_BYTES);
    close(fd);
    P("WFILE|path=%s|bytes=%lld|fill=lcg", W_PATH, (long long)W_BYTES);
    return 0;
}

// One blob for the mid-block attention: score matrix [6120,384] at offset 0,
// projection matrix [384,384] after it, tail zero-padded to a page boundary.
static int64_t ensure_attn_blob(const char** why) {
    struct stat st;
    if (stat(W_ATTN_PATH, &st) == 0 && st.st_size == ATTN_BYTES) return 0;
    int fd = open(W_ATTN_PATH, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { *why = "attn_open"; return -1; }
    if (ftruncate(fd, (off_t)ATTN_BYTES) != 0) { *why = "attn_trunc"; close(fd); return -1; }
    void* p = mmap(nullptr, (size_t)ATTN_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { *why = "attn_mmap"; close(fd); return -1; }
    int8_t* q = (int8_t*)p;
    uint32_t x = 0xC0FFEE11u;
    for (int64_t i = 0; i < ATTN_RAW_BYTES; ++i) {
        x = x * 1664525u + 1013904223u;
        q[i] = (int8_t)((x >> 24) & 0xFF);
    }
    for (int64_t i = ATTN_RAW_BYTES; i < ATTN_BYTES; ++i) q[i] = 0;
    msync(p, (size_t)ATTN_BYTES, MS_SYNC);
    munmap(p, (size_t)ATTN_BYTES);
    close(fd);
    P("WFILE|path=%s|bytes=%lld|fill=lcg|score=%dx%d|proj=%dx%d",
      W_ATTN_PATH, (long long)ATTN_BYTES, ATTN_SCORE_N, ATTN_SCORE_K, ATTN_PROJ_N, ATTN_PROJ_K);
    return 0;
}

// The chunk-major `ffn.down` blob. Same total bytes as the down region of W_PATH
// (DOWN_N * DOWN_K), different order: chunk 0's full [1536, K/nchunks] matrix,
// then chunk 1's, etc. Plus a filler block zeroed first so chunk LENGTHS are the
// thing that varies — nothing here is real weight data (LCG fill).
static int64_t ensure_down_blob(int nchunks, const char** path_out, const char** why) {
    static std::string dyn[2];
    int idx = (nchunks == 4) ? 1 : 0;
    dyn[idx] = "/data/local/tmp/hv_down_chunk";
    dyn[idx] += std::to_string(nchunks);
    dyn[idx] += ".bin";
    const char* path = dyn[idx].c_str();
    *path_out = path;
    const int64_t bytes = (int64_t)DOWN_N * (int64_t)DOWN_K;

    struct stat st;
    if (stat(path, &st) == 0 && st.st_size == bytes) return 0;
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { *why = "down_open"; return -1; }
    if (ftruncate(fd, (off_t)bytes) != 0) { *why = "down_trunc"; close(fd); return -1; }
    void* p = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { *why = "down_mmap"; close(fd); return -1; }
    int8_t* q = (int8_t*)p;
    uint32_t x = 0x9E3779B9u;
    for (int64_t i = 0; i < bytes; ++i) {
        x = x * 1664525u + 1013904223u;
        q[i] = (int8_t)((x >> 24) & 0xFF);
    }
    msync(p, (size_t)bytes, MS_SYNC);
    munmap(p, (size_t)bytes);
    close(fd);
    P("WFILE|path=%s|bytes=%lld|fill=lcg|chunks=%d", path, (long long)bytes, nchunks);
    return 0;
}

// ------------------------------------------------------------------- builder
struct Holder {
    ANeuralNetworksModel* m = nullptr;
    uint32_t next = 0;
    std::vector<std::vector<float>> scales;
    std::vector<std::vector<int32_t>> biases;
    int32_t fused = ANEURALNETWORKS_FUSED_NONE;
};

static int add_operand(Holder& h, const ANeuralNetworksOperandType* t, uint32_t* id,
                       const char** why) {
    uint32_t idx = h.next;
    int rc = ANeuralNetworksModel_addOperand(h.m, t);
    if (rc) { P("FATAL|addOperand|idx=%u|rc=%d(%s)", idx, rc, rcs(rc)); *why = "addOperand"; return -1; }
    h.next = idx + 1;
    *id = idx;
    return 0;
}

// Append one FULLY_CONNECTED whose input is operand `in`; returns its output id.
static int add_fc(Holder& h, ANeuralNetworksMemory* wmem, int M, const Gemm& g,
                  uint32_t in, uint32_t* out, const char** why) {
    h.scales.emplace_back((size_t)g.N, SW);
    h.biases.emplace_back((size_t)g.N, 0);
    const std::vector<float>& sc = h.scales.back();
    const std::vector<int32_t>& bi = h.biases.back();

    uint32_t dW[2] = {(uint32_t)g.N, (uint32_t)g.K};
    uint32_t dB[1] = {(uint32_t)g.N};
    uint32_t dO[2] = {(uint32_t)M, (uint32_t)g.N};

    ANeuralNetworksOperandType tW = {G_PC ? ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL
                                          : ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
                                     2, dW, G_PC ? 0.0f : SW, 0};
    ANeuralNetworksOperandType tB = {ANEURALNETWORKS_TENSOR_INT32, 1, dB, SA * SW, 0};
    ANeuralNetworksOperandType tAct = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
    ANeuralNetworksOperandType tO = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dO, SA, 0};

    uint32_t iw, ib, ia, io;
    if (add_operand(h, &tW, &iw, why)) return -1;
    if (add_operand(h, &tB, &ib, why)) return -1;
    if (add_operand(h, &tAct, &ia, why)) return -1;
    if (add_operand(h, &tO, &io, why)) return -1;

    // weights: CONSTANT operand backed by the shared file mapping (route B)
    const size_t wlen = (size_t)g.N * (size_t)g.K;
    int rc = ANeuralNetworksModel_setOperandValueFromMemory(h.m, iw, wmem, (size_t)g.off, wlen);
    if (rc) { P("FATAL|setOperandValueFromMemory|%s|pc=%d|rc=%d(%s)", g.name, G_PC, rc, rcs(rc));
              *why = "w_from_memory"; return -1; }

    if (G_PC) {
        ANeuralNetworksSymmPerChannelQuantParams pch = {0, (uint32_t)g.N, sc.data()};
        rc = ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(h.m, iw, &pch);
        if (rc) { P("FATAL|setPerChannel|%s|rc=%d(%s)", g.name, rc, rcs(rc));
                  *why = "per_channel"; return -1; }
    }

    rc = ANeuralNetworksModel_setOperandValue(h.m, ib, bi.data(), bi.size() * sizeof(int32_t));
    if (rc) { P("FATAL|setBias|%s|rc=%d(%s)", g.name, rc, rcs(rc)); *why = "bias"; return -1; }

    rc = ANeuralNetworksModel_setOperandValue(h.m, ia, &h.fused, sizeof(h.fused));
    if (rc) { P("FATAL|setAct|%s|rc=%d(%s)", g.name, rc, rcs(rc)); *why = "activation"; return -1; }

    uint32_t op_in[4] = {in, iw, ib, ia};
    uint32_t op_out[1] = {io};
    rc = ANeuralNetworksModel_addOperation(h.m, ANEURALNETWORKS_FULLY_CONNECTED, 4, op_in, 1, op_out);
    if (rc) { P("FATAL|addOperation|%s|pc=%d|in=%u,%u,%u,%u|out=%u|nops_in_model=%u|rc=%d(%s)",
                g.name, G_PC, in, iw, ib, ia, io, h.next, rc, rcs(rc));
              *why = "addOperation"; return -1; }
    *out = io;
    return 0;
}

// mode=chain: all 10 FCs in one model. mode=single: caller loops one op per model.
static int build_model(ANeuralNetworksMemory* wmem, int M, const Gemm* gems, int nops,
                       Holder& h, uint32_t* in_id, uint32_t* out_id, const char** why) {
    int rc = ANeuralNetworksModel_create(&h.m);
    if (rc) { *why = "model_create"; return -rc; }

    uint32_t dA[2] = {(uint32_t)M, (uint32_t)gems[0].K};
    ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, SA, 0};
    uint32_t ia;
    if (add_operand(h, &tA, &ia, why)) return -1;
    *in_id = ia;

    uint32_t prev = ia;
    for (int i = 0; i < nops; ++i) {
        if (add_fc(h, wmem, M, gems[i], prev, &prev, why)) return -1;
    }
    *out_id = prev;

    // identifyInputsAndOutputs is what makes it a model: without it model_finish
    // fails BAD_DATA (operands + operations alone are not a model).
    uint32_t ins[1] = {ia};
    uint32_t outs[1] = {prev};
    rc = ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
    if (rc) { P("FATAL|identifyIO|rc=%d(%s)", rc, rcs(rc)); *why = "identify_io"; return -rc; }
    return 0;
}

// ANEURALNETWORKS_ADD for the chunked-K down projection. On this HAL's quant8
// path all three tensors must share scale/zeroPoint (per-tensor SA), which is
// why the chunk outputs are requantised to SA between adds — acceptable here
// because this measures latency; the accuracy consequence belongs to Gate 1.
static int add_add(Holder& h, int M, int N, uint32_t a, uint32_t b, uint32_t* out,
                   const char** why) {
    uint32_t d[2] = {(uint32_t)M, (uint32_t)N};
    ANeuralNetworksOperandType t = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, d, SA, 0};
    ANeuralNetworksOperandType tAct = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
    uint32_t iact, io;
    // Both inputs are existing operands (a, b) — do NOT allocate a fresh
    // operand for the second input: an operand that no op consumes and no value
    // backs is "never written" and fails model_finish with BAD_DATA.
    if (add_operand(h, &tAct, &iact, why)) return -1;
    if (add_operand(h, &t, &io, why)) return -1;
    int rc = ANeuralNetworksModel_setOperandValue(h.m, iact, &h.fused, sizeof(h.fused));
    if (rc) { P("FATAL|add_setAct|rc=%d(%s)", rc, rcs(rc)); *why = "add_act"; return -1; }
    uint32_t op_in[3] = {a, b, iact};
    uint32_t op_out[1] = {io};
    rc = ANeuralNetworksModel_addOperation(h.m, ANEURALNETWORKS_ADD, 3, op_in, 1, op_out);
    if (rc) { P("FATAL|addOperation|ADD|rc=%d(%s)", rc, rcs(rc)); *why = "add_op"; return -1; }
    *out = io;
    return 0;
}

// nchunks x FC(1536, 8960/nchunks) accumulated by ADD. The activation is the
// same [M, 8960/nchunks] tile for every chunk (the harness feeds ones; a real
// pipeline hands each chunk its slice of the 8960-wide activation).
static int build_down(ANeuralNetworksMemory* wmem, int M, int nchunks, Holder& h,
                      uint32_t* in_id, uint32_t* out_id, const char** why) {
    int rc = ANeuralNetworksModel_create(&h.m);
    if (rc) { *why = "model_create"; return -rc; }
    const int Kc = DOWN_K / nchunks;
    uint32_t dA[2] = {(uint32_t)M, (uint32_t)Kc};
    ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, SA, 0};
    uint32_t ia;
    if (add_operand(h, &tA, &ia, why)) return -1;
    *in_id = ia;

    uint32_t acc = 0;
    for (int c = 0; c < nchunks; ++c) {
        Gemm g = {DOWN_N, Kc, (int64_t)c * (int64_t)DOWN_N * (int64_t)Kc, "ffn.down"};
        uint32_t o;
        if (add_fc(h, wmem, M, g, ia, &o, why)) return -1;
        if (c == 0) { acc = o; continue; }
        uint32_t no;
        if (add_add(h, M, DOWN_N, acc, o, &no, why)) return -1;
        acc = no;
    }
    *out_id = acc;

    uint32_t ins[1] = {ia};
    uint32_t outs[1] = {acc};
    rc = ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
    if (rc) { P("FATAL|identifyIO|rc=%d(%s)", rc, rcs(rc)); *why = "identify_io"; return -rc; }
    return 0;
}

// mode=downsep: the only chunked-K shape that fits the compiler's K ceiling.
// K=8960 does not compile, and neither does K=1792 (measured: K<=1664 compiles
// in ~0.5 s, K=1792 fails OP_FAILED after 48.7 s), so the down projection is
// split into DOWN_K/Kc chunks of Kc=1280 — and each chunk is compiled as its OWN
// one-op model, because a multi-FC graph of this size is exactly what rebooted
// the phone in Compilation_finish during the chain experiment.
static int run_downsep(ANeuralNetworksDevice* tpu, ANeuralNetworksMemory* wmem,
                       int M, int nchunks, int tiles, int iters) {
    const int Kc = DOWN_K / nchunks;
    const char* why = "";

    std::vector<ANeuralNetworksCompilation*> fc(nchunks, nullptr);
    std::vector<ANeuralNetworksModel*> fcm(nchunks, nullptr);
    double compile_fc_ms = 0;
    for (int c = 0; c < nchunks; ++c) {
        Holder h; uint32_t in_id = 0, out_id = 0;
        if (ANeuralNetworksModel_create(&h.m)) { P("FATAL|downsep_model|%d", c); return 5; }
        uint32_t dA[2] = {(uint32_t)M, (uint32_t)Kc};
        ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, SA, 0};
        add_operand(h, &tA, &in_id, &why);
        Gemm g = {DOWN_N, Kc, (int64_t)c * (int64_t)DOWN_N * (int64_t)Kc, "ffn.down"};
        if (add_fc(h, wmem, M, g, in_id, &out_id, &why)) { P("FATAL|downsep_fc|%d", c); return 5; }
        uint32_t ins[1] = {in_id}, outs[1] = {out_id};
        ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
        if (ANeuralNetworksModel_finish(h.m)) { P("FATAL|downsep_finish|%d", c); return 6; }
        ANeuralNetworksCompilation* comp = nullptr;
        int64_t c0 = now_ns();
        int rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
        if (rc) { P("FATAL|downsep_comp_create|%d|rc=%d(%s)", c, rc, rcs(rc)); return 7; }
        ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        rc = ANeuralNetworksCompilation_finish(comp);
        double cms = (now_ns() - c0) / 1e6;
        compile_fc_ms += cms;
        P("COMPILE|mode=downsep|c=%d|K=%d|ms=%.1f|rc=%d(%s)", c, Kc, cms, rc, rcs(rc));
        if (rc) { P("FATAL|downsep_compile|%d|rc=%d(%s)", c, rc, rcs(rc)); return 7; }
        fc[c] = comp; fcm[c] = h.m;
    }

    // One ADD compilation, reused for every accumulate (shapes are identical).
    Holder ha; uint32_t a0 = 0, a1 = 0, ao = 0;
    if (ANeuralNetworksModel_create(&ha.m)) { P("FATAL|addsep_model"); return 5; }
    uint32_t dO[2] = {(uint32_t)M, (uint32_t)DOWN_N};
    ANeuralNetworksOperandType tO = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dO, SA, 0};
    add_operand(ha, &tO, &a0, &why);
    add_operand(ha, &tO, &a1, &why);
    if (add_add(ha, M, DOWN_N, a0, a1, &ao, &why)) { P("FATAL|addsep_build|%s", why); return 5; }
    uint32_t ains[2] = {a0, a1}, aouts[1] = {ao};
    ANeuralNetworksModel_identifyInputsAndOutputs(ha.m, 2, ains, 1, aouts);
    if (ANeuralNetworksModel_finish(ha.m)) { P("FATAL|addsep_finish"); return 6; }
    ANeuralNetworksCompilation* addc = nullptr;
    int64_t a0ns = now_ns();
    int rc = ANeuralNetworksCompilation_createForDevices(ha.m, &tpu, 1, &addc);
    if (rc) { P("FATAL|addsep_comp_create|rc=%d(%s)", rc, rcs(rc)); return 7; }
    ANeuralNetworksCompilation_setPreference(addc, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    rc = ANeuralNetworksCompilation_finish(addc);
    double compile_add_ms = (now_ns() - a0ns) / 1e6;
    P("COMPILE|mode=downsep|ADD|ms=%.1f|rc=%d(%s)", compile_add_ms, rc, rcs(rc));
    if (rc) { P("FATAL|addsep_compile|rc=%d(%s)", rc, rcs(rc)); return 7; }

    if (iters <= 0) {
        P("SUMMARY|mode=downsep|M=%d|chunks=%d|Kc=%d|compile_ms=%.1f|probe_only=1",
          M, nchunks, Kc, compile_fc_ms + compile_add_ms);
        P("DONE");
        return 0;
    }

    const size_t asz = (size_t)M * (size_t)Kc;
    const size_t osz = (size_t)M * (size_t)DOWN_N;
    std::vector<int8_t> A(asz, 1);
    std::vector<std::vector<int8_t>> O(nchunks, std::vector<int8_t>(osz, 0));
    std::vector<int8_t> bufA(osz, 0), bufB(osz, 0);

    std::vector<double> tile_ms;
    for (int it = 0; it < iters; ++it) {
        for (int t = 0; t < tiles; ++t) {
            int64_t t0 = now_ns();
            // 7 chunk FCs + 6 ADDs, each a separate dispatch, exactly the shape
            // the single-mode fallback already pays for.
            for (int c = 0; c < nchunks; ++c) {
                ANeuralNetworksExecution* ex = nullptr;
                if (ANeuralNetworksExecution_create(fc[c], &ex)) { P("FATAL|downsep_exec|%d", c); return 8; }
                ANeuralNetworksExecution_setInput(ex, 0, nullptr, A.data(), asz);
                ANeuralNetworksExecution_setOutput(ex, 0, nullptr, O[c].data(), osz);
                int erc = ANeuralNetworksExecution_compute(ex);
                ANeuralNetworksExecution_free(ex);
                if (erc) { P("FATAL|downsep_compute_fc|c=%d|tile=%d|rc=%d(%s)", c, t, erc, rcs(erc)); return 8; }
            }
            int8_t* prev = O[0].data();
            for (int c = 1; c < nchunks; ++c) {
                int8_t* dst = (c & 1) ? bufA.data() : bufB.data();
                ANeuralNetworksExecution* ex = nullptr;
                if (ANeuralNetworksExecution_create(addc, &ex)) { P("FATAL|downsep_add_exec"); return 8; }
                ANeuralNetworksExecution_setInput(ex, 0, nullptr, prev, osz);
                ANeuralNetworksExecution_setInput(ex, 1, nullptr, O[c].data(), osz);
                ANeuralNetworksExecution_setOutput(ex, 0, nullptr, dst, osz);
                int erc = ANeuralNetworksExecution_compute(ex);
                ANeuralNetworksExecution_free(ex);
                if (erc) { P("FATAL|downsep_compute_add|c=%d|tile=%d|rc=%d(%s)", c, t, erc, rcs(erc)); return 8; }
                prev = dst;
            }
            int64_t t1 = now_ns();
            tile_ms.push_back((t1 - t0) / 1e6);
            // Absolute CLOCK_MONOTONIC stamps, every tile of every iteration: the
            // co-run driver classifies each tile as "ran alone" vs "ran while the
            // GPU was busy" from these, instead of inferring it from the arm.
            P("TS|downsep|tile=%d|iter=%d|t0=%.3f|t1=%.3f|ms=%.3f", t, it,
              (double)t0 / 1e6, (double)t1 / 1e6, (t1 - t0) / 1e6);
            if (it == 0)
                P("TILE|downsep|tile=%d|ms=%.3f|out0=%d", t, (t1 - t0) / 1e6, (int)prev[0]);
        }
    }

    double sum = 0;
    for (double v : tile_ms) sum += v;
    double per_tile = sum / (double)tile_ms.size();
    double block_s = per_tile * tiles / 1000.0;
    double gmacs = (double)tiles * (double)M * (double)DOWN_N * (double)DOWN_K;
    P("OP|ffn.down|chunked_sep=%d|N=%d|K=%d|compile_ms=%.1f|ms_per_tile=%.3f|per_block_s=%.4f|gops=%.1f",
      nchunks, DOWN_N, DOWN_K, compile_fc_ms + compile_add_ms, per_tile, block_s,
      2.0 * gmacs / block_s / 1e9);
    P("SUMMARY|mode=downsep|M=%d|tiles=%d|iters=%d|chunks=%d|Kc=%d|compile_ms=%.1f|"
      "ms_per_tile=%.3f|per_block_s=%.4f|gops=%.1f|samples=%zu",
      M, tiles, iters, nchunks, Kc, compile_fc_ms + compile_add_ms, per_tile, block_s,
      2.0 * gmacs / block_s / 1e9, tile_ms.size());
    P("DONE");
    return 0;
}

// ------------------------------------------------- T_V: one CONV_2D op (tap)
// The VAE decoder is a stack of causal 3-D convs and NNAPI has no CONV_3D. The
// decomposition each kT tap becomes is a plain 2-D CONV_2D over the (h,w) plane
// of one frame, accumulated into the output frame with ADD. So the unit of
// measurement is one CONV_2D at the layer's real shape, exactly like the T_N
// table's unit is one FULLY_CONNECTED.
//
// Filter layout for CONV_2D is [Cout, kH, kW, Cin] (OHWI), the transpose of the
// checkpoint's [Cout, Cin, kT, kH, kW] slice — getting this backwards produces
// a graph that runs and is numerically wrong, so it is filled explicitly here.
struct ConvCfg { int N, H, W, Cin, Cout, k; };

static int run_conv(ANeuralNetworksDevice* tpu, const ConvCfg& c, int tiles, int iters) {
    const char* why = "";
    // int64: a Preset-A decoder conv is ~97 GMAC per tap and overflows int32
    const int64_t macs = (int64_t)c.N * c.H * c.W * c.Cout * c.Cin * c.k * c.k;

    Holder h;
    int rc = ANeuralNetworksModel_create(&h.m);
    if (rc) { P("FATAL|conv_model_create|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 5; }

    const uint32_t dI[4] = {(uint32_t)c.N, (uint32_t)c.H, (uint32_t)c.W, (uint32_t)c.Cin};
    const uint32_t dF[4] = {(uint32_t)c.Cout, (uint32_t)c.k, (uint32_t)c.k, (uint32_t)c.Cin};
    const uint32_t dB[1] = {(uint32_t)c.Cout};
    const uint32_t dO[4] = {(uint32_t)c.N, (uint32_t)c.H, (uint32_t)c.W, (uint32_t)c.Cout};

    ANeuralNetworksOperandType tI = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 4, dI, SA, 0};
    ANeuralNetworksOperandType tF = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 4, dF, SW, 0};
    ANeuralNetworksOperandType tB = {ANEURALNETWORKS_TENSOR_INT32, 1, dB, SA * SW, 0};
    ANeuralNetworksOperandType tO = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 4, dO, SA, 0};
    ANeuralNetworksOperandType tS = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};

    uint32_t iI, iF, iB, ipl, ipr, ipt, ipb, isw, ish, iact, iO;
    if (add_operand(h, &tI, &iI, &why) || add_operand(h, &tF, &iF, &why) ||
        add_operand(h, &tB, &iB, &why) || add_operand(h, &tS, &ipl, &why) ||
        add_operand(h, &tS, &ipr, &why) || add_operand(h, &tS, &ipt, &why) ||
        add_operand(h, &tS, &ipb, &why) || add_operand(h, &tS, &isw, &why) ||
        add_operand(h, &tS, &ish, &why) || add_operand(h, &tS, &iact, &why) ||
        add_operand(h, &tO, &iO, &why)) {
        P("FATAL|conv_add_operand|%s", why); P("DONE"); return 5;
    }

    // weights: deterministic LCG fill. Latency is shape- and layout-determined;
    // the real INT8 VAE artifact does not exist yet (Phase 1).
    const size_t wlen = (size_t)c.Cout * c.Cin * c.k * c.k;
    {
        std::vector<int8_t> W(wlen);
        uint32_t x = 0x9E3779B9u;
        for (size_t i = 0; i < wlen; ++i) { x = x * 1664525u + 1013904223u; W[i] = (int8_t)((x >> 24) & 0xFF); }
        rc = ANeuralNetworksModel_setOperandValue(h.m, iF, W.data(), wlen);
        if (rc) { P("FATAL|conv_set_weights|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 5; }
    }
    {
        std::vector<int32_t> B((size_t)c.Cout, 0);
        rc = ANeuralNetworksModel_setOperandValue(h.m, iB, B.data(), B.size() * sizeof(int32_t));
        if (rc) { P("FATAL|conv_set_bias|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 5; }
    }
    // causal conv 3x3: SAME padding, stride 1
    const int32_t pad = c.k / 2, one = 1;
    int32_t act = ANEURALNETWORKS_FUSED_NONE;
    struct { uint32_t id; const void* p; size_t n; } scal[] = {
        {ipl, &pad, 4}, {ipr, &pad, 4}, {ipt, &pad, 4}, {ipb, &pad, 4},
        {isw, &one, 4}, {ish, &one, 4}, {iact, &act, 4},
    };
    for (auto& s : scal) {
        rc = ANeuralNetworksModel_setOperandValue(h.m, s.id, s.p, s.n);
        if (rc) { P("FATAL|conv_set_scalar|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 5; }
    }

    uint32_t op_in[10] = {iI, iF, iB, ipl, ipr, ipt, ipb, isw, ish, iact};
    uint32_t op_out[1] = {iO};
    rc = ANeuralNetworksModel_addOperation(h.m, ANEURALNETWORKS_CONV_2D, 10, op_in, 1, op_out);
    if (rc) { P("FATAL|conv_addOperation|in=%d,%d,%d|rc=%d(%s)", c.N, c.H, c.W, rc, rcs(rc));
              P("DONE"); return 5; }
    uint32_t ins[1] = {iI}, outs[1] = {iO};
    ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
    rc = ANeuralNetworksModel_finish(h.m);
    P("MODEL_FINISH|mode=conv|N=%d|H=%d|W=%d|Cin=%d|Cout=%d|k=%d|rc=%d(%s)",
      c.N, c.H, c.W, c.Cin, c.Cout, c.k, rc, rcs(rc));
    if (rc) { P("DONE"); return 6; }

    bool sup[1] = {false};
    int src = ANeuralNetworksModel_getSupportedOperationsForDevices(h.m, &tpu, 1, sup);
    P("SUPPORT|mode=conv|rc=%d(%s)|conv2d_delegated=%d", src, rcs(src), src == 0 ? (sup[0] ? 1 : 0) : -1);

    ANeuralNetworksCompilation* comp = nullptr;
    int64_t c0 = now_ns();
    rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
    if (rc) { P("FATAL|conv_comp_create|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 7; }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    rc = ANeuralNetworksCompilation_finish(comp);
    const double compile_ms = (now_ns() - c0) / 1e6;
    P("COMPILE|mode=conv|N=%d|H=%d|W=%d|Cin=%d|Cout=%d|k=%d|ms=%.1f|rc=%d(%s)",
      c.N, c.H, c.W, c.Cin, c.Cout, c.k, compile_ms, rc, rcs(rc));
    if (rc) { P("FATAL|conv_compile|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 7; }
    if (iters <= 0) {
        P("SUMMARY|mode=conv|probe_only=1|compile_ms=%.1f|macs=%lld", compile_ms, (long long)macs);
        ANeuralNetworksCompilation_free(comp); ANeuralNetworksModel_free(h.m); P("DONE"); return 0;
    }

    std::vector<int8_t> A((size_t)c.N * c.H * c.W * c.Cin, 1);
    std::vector<int8_t> O((size_t)c.N * c.H * c.W * c.Cout, 0);
    std::vector<double> ms;
    for (int it = 0; it < iters; ++it) {
        for (int t = 0; t < tiles; ++t) {
            ANeuralNetworksExecution* ex = nullptr;
            if (ANeuralNetworksExecution_create(comp, &ex)) { P("FATAL|conv_exec"); P("DONE"); return 8; }
            ANeuralNetworksExecution_setInput(ex, 0, nullptr, A.data(), A.size());
            ANeuralNetworksExecution_setOutput(ex, 0, nullptr, O.data(), O.size());
            int64_t t0 = now_ns();
            int erc = ANeuralNetworksExecution_compute(ex);
            int64_t t1 = now_ns();
            ANeuralNetworksExecution_free(ex);
            if (erc) { P("FATAL|conv_compute|tile=%d|rc=%d(%s)", t, erc, rcs(erc)); P("DONE"); return 8; }
            ms.push_back((t1 - t0) / 1e6);
            if (it == 0 && t == 0)
                P("TILE|conv|tile=0|ms=%.3f|out0=%d", (t1 - t0) / 1e6, (int)O[0]);
        }
    }
    double sum = 0;
    for (double v : ms) sum += v;
    const double per = sum / (double)ms.size();
    P("OP|conv2d|N=%d|H=%d|W=%d|Cin=%d|Cout=%d|k=%d|compile_ms=%.1f|ms=%.3f|macs=%lld|gops=%.1f",
      c.N, c.H, c.W, c.Cin, c.Cout, c.k, compile_ms, per, (long long)macs,
      2.0 * (double)macs / (per / 1000.0) / 1e9);
    P("SUMMARY|mode=conv|N=%d|H=%d|W=%d|Cin=%d|Cout=%d|k=%d|compile_ms=%.1f|ms=%.3f|gops=%.1f|samples=%zu",
      c.N, c.H, c.W, c.Cin, c.Cout, c.k, compile_ms, per,
      2.0 * (double)macs / (per / 1000.0) / 1e9, ms.size());
    ANeuralNetworksCompilation_free(comp);
    ANeuralNetworksModel_free(h.m);
    P("DONE");
    return 0;
}

static ANeuralNetworksDevice* find_tpu() {
    uint32_t n = 0;
    if (ANeuralNetworks_getDeviceCount(&n) != 0) return nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        ANeuralNetworksDevice* d = nullptr;
        if (ANeuralNetworks_getDevice(i, &d) != 0 || !d) continue;
        const char* name = nullptr;
        if (ANeuralNetworksDevice_getName(d, &name) != 0 || !name) continue;
        int32_t type = -1;
        ANeuralNetworksDevice_getType(d, &type);
        P("DEV|idx=%u|name=%s|type=%d", i, name, type);
        if (strstr(name, "edgetpu")) return d;
    }
    return nullptr;
}

// ================================================ convseq: the VAE decode run
// `--decode 1` in the pipeline prints the 10.640 s hybrid reference and executes
// nothing, because no INT8 VAE weights had ever been written. They exist now
// (phase1_quant/vae_int8.py, artifacts/vae/). This mode executes them:
//
//   * the plan is flattened on the host by tools/npu/vae_seq.py into
//     /data/local/tmp/vae_seq.txt (LAYER/TAP records, byte offsets, not maths)
//   * ONE file-mapped copy of weights_int8.bin (72,751,648 B) is windowed per
//     tap with setOperandValueFromMemory — route B, the discipline proc_b uses
//   * one graph per (layer, tap); one execute per (layer, frame, tap) => the
//     plan's 592 executes for a 3-frame chunk
//   * per-output-channel FP32 scales are offered to CONV_2D as
//     SYMM_PER_CHANNEL. FULLY_CONNECTED rejects per-channel on this HAL; whether
//     CONV_2D accepts it is the fork this mode answers (pc = -1 probes first)
//   * FP32 biases are quantised to INT32 in SA*SW units, per output channel
//   * the activation scale is the MEASURED per-layer one (P3.9 `calibrate.py`
//     -> `phase1_quant/vae_sa.py` -> /data/local/tmp/vae_sa.f32). The nominal
//     1/127 remains only as a fallback, and a run that uses it says `sa=nominal`
//     in its summary. 34 of 36 layers need a larger scale than 1/127 (median
//     5.05x, worst 32.06x), so a nominal walk clips and cannot be a quality claim.
//
// What is NOT wired here: residual adds, the resnet-shortcut addition and the
// SiLU/clamp epilogues. They move no weights and their numerical effect depends
// on SA, so this run measures EXECUTION and LATENCY, not decode quality.
static const char* VAE_W_PATH   = "/data/local/tmp/vae_int8.bin";
static const char* VAE_SC_PATH  = "/data/local/tmp/vae_scales.f32";
static const char* VAE_BI_PATH  = "/data/local/tmp/vae_bias.f32";
static const char* VAE_SEQ_PATH = "/data/local/tmp/vae_seq.txt";
static const char* VAE_SA_PATH  = "/data/local/tmp/vae_sa.f32";   // 36 x f32, layer order
// The nominal is only a fallback for a table-less run. P3.9 measured the real
// per-layer max|x|/127: 34 of 36 layers need a LARGER scale than this (median
// 5.05x, worst 32.06x on up_blocks.1.resnets.0.conv_shortcut), i.e. 1/127 was
// clipping the decode. phase1_quant/vae_sa.py writes the table; a run that
// reports sa=nominal is a mechanism run, not a quality run.
static const float VAE_SA_NOMINAL = 1.0f / 127.0f;

struct SeqTap   { int tap, delta, kh, kw; long long off, bytes; };
struct SeqLayer { std::string name; int frames, fin, cout, cin, h, w, ntaps, wiring;
                  long long sc_off; int ai = -1;          // ai = the ARTIFACT index
                  std::vector<SeqTap> taps; };

static std::vector<uint8_t> slurp(const char* p, const char* what) {
    std::vector<uint8_t> v;
    FILE* f = fopen(p, "rb");
    if (!f) { P("FATAL|open|%s|%s", what, p); return v; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); P("FATAL|size|%s|%ld", what, n); return v; }
    v.resize((size_t)n);
    if (fread(v.data(), 1, (size_t)n, f) != (size_t)n) { P("FATAL|read|%s", what); v.clear(); }
    fclose(f);
    return v;
}

static int load_seq(std::vector<SeqLayer>* out) {
    FILE* f = fopen(VAE_SEQ_PATH, "r");
    if (!f) { P("FATAL|seq_open|%s", VAE_SEQ_PATH); return -1; }
    char line[512];
    SeqLayer* cur = nullptr;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "LAYER ", 6)) {
            SeqLayer L; char nm[192] = {0}; int idx = 0;
            // LAYER <idx> <frames_out> <frames_in> <cout> <cin> <h> <w> <ntaps> <wiring> <sc_off> <name>
            int n = sscanf(line + 6, "%d %d %d %d %d %d %d %d %d %lld %191s",
                           &idx, &L.frames, &L.fin, &L.cout, &L.cin, &L.h, &L.w,
                           &L.ntaps, &L.wiring, &L.sc_off, nm);
            if (n < 11) { P("FATAL|seq_layer_parse|%s", line); fclose(f); return -1; }
            L.name = nm;
            out->push_back(L);
            cur = &out->back();
        } else if (!strncmp(line, "TAP ", 4) && cur) {
            SeqTap t; int n = sscanf(line + 4, "%d %d %lld %lld %d %d",
                                     &t.tap, &t.delta, &t.off, &t.bytes, &t.kh, &t.kw);
            if (n < 6) { P("FATAL|seq_tap_parse|%s", line); fclose(f); return -1; }
            cur->taps.push_back(t);
        }
    }
    fclose(f);
    size_t taps = 0;
    for (auto& L : *out) taps += L.taps.size();
    P("SEQ|layers=%zu|taps=%zu", out->size(), taps);
    return out->empty() ? -1 : 0;
}

// One CONV_2D graph: [1,H,W,Cin] -> [1,H,W,Cout], SAME padding, stride 1.
// `sa` is the INPUT (and bias) scale, `sa_out` the output operand's scale. They
// were the same value until P3.12: quantising the output at the consumer's
// scale threw away resolution whenever the measured SA_out is smaller than
// SA_in, and the decode's per-layer SA_out/SA_in spans 0.275..15.5.
static int add_conv_op(Holder& h, ANeuralNetworksMemory* wmem, int H, int W, int Cin, int Cout,
                       int kh, int kw, long long woff, const float* sc, int pc,
                       const int32_t* bias, float sa, float sa_out, uint32_t in,
                       uint32_t* out, const char** why) {
    const uint32_t dF[4] = {(uint32_t)Cout, (uint32_t)kh, (uint32_t)kw, (uint32_t)Cin};
    const uint32_t dB[1] = {(uint32_t)Cout};
    const uint32_t dO[4] = {1u, (uint32_t)H, (uint32_t)W, (uint32_t)Cout};
    const float sw = sc[0];

    ANeuralNetworksOperandType tF = {pc ? ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL
                                        : ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
                                     4, dF, pc ? 0.0f : sw, 0};
    // Bias is INT32 in SA*SW units. For a per-channel filter NNAPI derives the
    // per-channel bias scales from the input scale and the filter scales, so the
    // values must be scaled per output channel (sc[c]), not by sc[0].
    ANeuralNetworksOperandType tB = {ANEURALNETWORKS_TENSOR_INT32, 1, dB, sa * sw, 0};
    ANeuralNetworksOperandType tO = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 4, dO, sa_out, 0};
    ANeuralNetworksOperandType tS = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};

    uint32_t iF, iB, ipl, ipr, ipt, ipb, isw, ish, iact, io;
    if (add_operand(h, &tF, &iF, why) || add_operand(h, &tB, &iB, why) ||
        add_operand(h, &tS, &ipl, why) || add_operand(h, &tS, &ipr, why) ||
        add_operand(h, &tS, &ipt, why) || add_operand(h, &tS, &ipb, why) ||
        add_operand(h, &tS, &isw, why) || add_operand(h, &tS, &ish, why) ||
        add_operand(h, &tS, &iact, why) || add_operand(h, &tO, &io, why)) return -1;

    int rc = ANeuralNetworksModel_setOperandValueFromMemory(
        h.m, iF, wmem, (size_t)woff, (size_t)Cout * (size_t)kh * (size_t)kw * (size_t)Cin);
    if (rc) { P("FATAL|convseq_wmem|off=%lld|rc=%d(%s)", woff, rc, rcs(rc)); *why = "wmem"; return -1; }
    if (pc) {
        ANeuralNetworksSymmPerChannelQuantParams pch = {0, (uint32_t)Cout, sc};
        rc = ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(h.m, iF, &pch);
        if (rc) { P("FATAL|convseq_per_channel|rc=%d(%s)", rc, rcs(rc)); *why = "pc"; return -1; }
    }
    rc = ANeuralNetworksModel_setOperandValue(h.m, iB, bias, (size_t)Cout * sizeof(int32_t));
    if (rc) { P("FATAL|convseq_bias|rc=%d(%s)", rc, rcs(rc)); *why = "bias"; return -1; }

    const int32_t pad = kh / 2, one = 1;
    int32_t act = ANEURALNETWORKS_FUSED_NONE;
    int32_t pw = kw / 2;
    struct { uint32_t id; const void* p; size_t n; } scal[] = {
        {ipl, &pad, 4}, {ipr, &pw, 4}, {ipt, &pad, 4}, {ipb, &pw, 4},
        {isw, &one, 4}, {ish, &one, 4}, {iact, &act, 4},
    };
    for (auto& s : scal) {
        rc = ANeuralNetworksModel_setOperandValue(h.m, s.id, s.p, s.n);
        if (rc) { P("FATAL|convseq_scalar|rc=%d(%s)", rc, rcs(rc)); *why = "scalar"; return -1; }
    }

    uint32_t op_in[10] = {in, iF, iB, ipl, ipr, ipt, ipb, isw, ish, iact};
    uint32_t op_out[1] = {io};
    rc = ANeuralNetworksModel_addOperation(h.m, ANEURALNETWORKS_CONV_2D, 10, op_in, 1, op_out);
    if (rc) { P("FATAL|convseq_addOperation|H=%d|W=%d|Cin=%d|Cout=%d|k=%dx%d|pc=%d|rc=%d(%s)",
                H, W, Cin, Cout, kh, kw, pc, rc, rcs(rc)); *why = "addOperation"; return -1; }
    *out = io;
    return 0;
}

// Does this HAL take SYMM_PER_CHANNEL on CONV_2D? Small shape first: if the
// answer is DEAD_OBJECT we want to lose seconds, not the whole sweep.
static int probe_conv_pc(ANeuralNetworksDevice* tpu, ANeuralNetworksMemory* wmem) {
    const int H = 8, W = 8, Cin = 16, Cout = 32;
    const char* why = "";
    Holder h;
    if (ANeuralNetworksModel_create(&h.m)) { P("PROBE|conv2d_per_channel|model_create_fail"); return -1; }
    std::vector<float> sc((size_t)Cout, 1.0f / 128.0f);
    std::vector<int32_t> bi((size_t)Cout, 0);
    const uint32_t dI[4] = {1u, (uint32_t)H, (uint32_t)W, (uint32_t)Cin};
    ANeuralNetworksOperandType tI = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 4, dI,
                                     VAE_SA_NOMINAL, 0};
    uint32_t ia, io;
    int rc = add_operand(h, &tI, &ia, &why);
    if (!rc) rc = add_conv_op(h, wmem, H, W, Cin, Cout, 3, 3, 0, sc.data(), 1, bi.data(),
                              VAE_SA_NOMINAL, VAE_SA_NOMINAL, ia, &io, &why);
    if (!rc) { uint32_t ins[1] = {ia}, outs[1] = {io};
               ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs); }
    const int arc = rc;
    int frc = arc ? -arc : ANeuralNetworksModel_finish(h.m);
    bool sup[1] = {false};
    int src = frc ? -1 : ANeuralNetworksModel_getSupportedOperationsForDevices(h.m, &tpu, 1, sup);
    int crc = -1;
    double cms = -1.0;
    if (!frc) {
        ANeuralNetworksCompilation* comp = nullptr;
        int64_t c0 = now_ns();
        int k = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
        if (!k) {
            ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
            crc = ANeuralNetworksCompilation_finish(comp);
            cms = (now_ns() - c0) / 1e6;
            ANeuralNetworksCompilation_free(comp);
        } else { crc = -k; }
    }
    P("PROBE|conv2d_per_channel|add_rc=%d(%s)|finish_rc=%d(%s)|support_rc=%d|delegated=%d|"
      "compile_rc=%d(%s)|compile_ms=%.1f",
      arc, rcs(arc), frc, rcs(frc), src, src == 0 ? (sup[0] ? 1 : 0) : -1, crc, rcs(crc), cms);
    ANeuralNetworksModel_free(h.m);
    return (arc == 0 && frc == 0 && crc == 0) ? 1 : 0;
}

static int run_convseq(ANeuralNetworksDevice* tpu, ANeuralNetworksMemory* wmem,
                       int max_layers, int pcmode) {
    const char* why = "";
    std::vector<SeqLayer> plan;
    if (load_seq(&plan)) { P("FATAL|seq_load"); P("DONE"); return 2; }

    std::vector<uint8_t> scraw = slurp(VAE_SC_PATH, "scales");
    std::vector<uint8_t> biraw = slurp(VAE_BI_PATH, "bias");
    if (scraw.empty() || biraw.empty()) { P("DONE"); return 2; }
    const float* SC = (const float*)scraw.data();
    const float* BI = (const float*)biraw.data();
    const size_t nsc = scraw.size() / 4, nbi = biraw.size() / 4;

    // Per-layer activation scale (phase1_quant/vae_sa.py). Absent or the wrong
    // length is not fatal: the run degrades to the nominal and SAYS SO in the
    // summary, because a nominal run answers execution/latency, not quality.
    std::vector<uint8_t> saraw;
    { FILE* f = fopen(VAE_SA_PATH, "rb");
      if (f) { fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
               if (n > 0) { saraw.resize((size_t)n);
                            if (fread(saraw.data(), 1, (size_t)n, f) != (size_t)n) saraw.clear(); }
               fclose(f); } }
    std::vector<float> SA;
    if (!saraw.empty() && saraw.size() % 4 == 0) {
        const size_t n = saraw.size() / 4;
        const float* p = (const float*)saraw.data();
        for (size_t i = 0; i < n; ++i) SA.push_back(p[i]);
        if ((int)n != (int)plan.size()) {
            P("NOTE|convseq_sa_length_mismatch|table=%zu|layers=%zu|using_nominal", n, plan.size());
            SA.clear();
        }
    }
    const bool sa_table = !SA.empty();
    if (!sa_table) P("NOTE|convseq_sa=nominal|%.8g|no_table_at_%s", (double)VAE_SA_NOMINAL, VAE_SA_PATH);
    else P("NOTE|convseq_sa=table|layers=%zu|min=%.8g|max=%.8g", SA.size(),
           (double)*std::min_element(SA.begin(), SA.end()),
           (double)*std::max_element(SA.begin(), SA.end()));
    int64_t bias_clamps = 0;

    if (pcmode < 0) {
        pcmode = probe_conv_pc(tpu, wmem);
        P("NOTE|convseq_pc_mode=%d|1=SYMM_PER_CHANNEL_accepted|0=per_tensor_fallback", pcmode);
    }

    const int nlay = (max_layers > 0 && max_layers < (int)plan.size()) ? max_layers : (int)plan.size();
    std::vector<int8_t> prev;               // previous layer's output frames
    int pf = 0, pc_ = 0, ph = 0, pw = 0;    // its [frames, ch, h, w]
    int64_t execs = 0, macs = 0, compile_ns = 0, pcfb = 0, tapfails = 0;
    int64_t run0 = now_ns();

    for (int li = 0; li < nlay; ++li) {
        SeqLayer& L = plan[li];
        if (L.ntaps != (int)L.taps.size()) {
            P("FATAL|seq_tap_count|layer=%d|%s|declared=%d|read=%zu", li, L.name.c_str(),
              L.ntaps, L.taps.size());
            P("DONE"); return 5;
        }
        const size_t plane_in  = (size_t)L.h * L.w * L.cin;
        const size_t plane_out = (size_t)L.h * L.w * L.cout;
        if (L.sc_off < 0 || (size_t)L.sc_off + (size_t)L.cout > nsc ||
            (size_t)L.sc_off + (size_t)L.cout > nbi) {
            P("FATAL|seq_scale_range|layer=%d|off=%lld|cout=%d", li, L.sc_off, L.cout);
            P("DONE"); return 5;
        }
        const float* lsc = SC + L.sc_off;
        const float* lbi = BI + L.sc_off;

        // This layer's activation scale (its INPUT tensor), and the scale its
        // OUTPUT buffer must carry — the consumer's input scale, so that the
        // wiring step stays a pure byte copy/permute.
        const float sa_in  = sa_table ? SA[(size_t)li] : VAE_SA_NOMINAL;
        const float sa_out = (sa_table && li + 1 < nlay) ? SA[(size_t)li + 1] : sa_in;

        // ---- build this layer's input frames from the previous output
        std::vector<int8_t> inb((size_t)L.fin * plane_in, 0);
        if (L.wiring == 3 || li == 0) {
            // synthetic latent [fin, H, W, Cin] — the decode's input tensor
            uint32_t x = 0x9E3779B9u;
            for (size_t i = 0; i < inb.size(); ++i) { x = x * 1664525u + 1013904223u;
                                                      inb[i] = (int8_t)((x >> 24) & 0xFF); }
        } else if (L.wiring == 0) {
            if (pf != L.fin || pc_ != L.cin || ph != L.h || pw != L.w) {
                P("FATAL|wiring_mismatch|layer=%d|%s|want fin=%d cin=%d h=%d w=%d|"
                  "prev f=%d c=%d h=%d w=%d", li, L.name.c_str(), L.fin, L.cin, L.h, L.w,
                  pf, pc_, ph, pw);
                P("DONE"); return 5;
            }
            inb.assign(prev.begin(), prev.end());
        } else if (L.wiring == 4) {
            // resnet shortcut branch: its input is the enclosing resnet's entry, not
            // the previous layer's output. Residual adds are not wired in this run, so
            // the conv executes on a zero activation of the manifest's shape — the
            // shape chain is still asserted host-side against the layer that produced
            // exactly this [frames, cin, h, w].
            P("NOTE|convseq_shortcut_zero_input|layer=%d|%s", li, L.name.c_str());
        } else {
            // 1 = temporal unshuffle (2C -> C, 2t-1 frames)
            // 2 = that plus a 2x nearest spatial upsample
            // 5 = 2x nearest spatial upsample only (up_blocks.2's resample: the
            //     temporal expansion is already finished at that stage)
            const int c = L.cin;                       // prev channels == 2*c, or c
            const bool tshuf = (L.wiring != 5);
            const bool spatial = (L.wiring == 2 || L.wiring == 5);
            const int two_f = 2 * pf - 1;
            if ((tshuf ? (pc_ != 2 * c || two_f != L.fin) : (pc_ != c || pf != L.fin)) ||
                (spatial ? (2 * ph != L.h || 2 * pw != L.w) : (ph != L.h || pw != L.w))) {
                P("FATAL|wiring_mismatch|layer=%d|%s|wiring=%d|want fin=%d cin=%d h=%d w=%d|"
                  "prev f=%d c=%d h=%d w=%d", li, L.name.c_str(), L.wiring, L.fin, L.cin,
                  L.h, L.w, pf, pc_, ph, pw);
                P("DONE"); return 5;
            }
            for (int t = 0; t < L.fin; ++t) {
                const int srcf = tshuf ? (t / 2) : t;
                const int half = tshuf ? (t % 2) : 0;  // 0 = first channel half
                for (int hh = 0; hh < L.h; ++hh) {
                    const int sh = spatial ? (hh / 2) : hh;
                    for (int ww = 0; ww < L.w; ++ww) {
                        const int sw = spatial ? (ww / 2) : ww;
                        int8_t* dst = &inb[((size_t)t * L.h * L.w + (size_t)hh * L.w + ww) * c];
                        if (srcf >= pf) { memset(dst, 0, (size_t)c); continue; }
                        const int8_t* src = &prev[((size_t)srcf * ph * pw +
                                                  (size_t)sh * pw + sw) * pc_ + (size_t)half * c];
                        memcpy(dst, src, (size_t)c);
                    }
                }
            }
        }

        // ---- compile one graph per tap, then run frames x taps
        std::vector<ANeuralNetworksCompilation*> comps;
        std::vector<Holder> holders(L.taps.size());
        int use_pc = pcmode;
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool ok = true;
            for (size_t ti = 0; ti < L.taps.size(); ++ti) {
                const SeqTap& T = L.taps[ti];
                Holder& h = holders[ti];
                uint32_t ia, io;
                int64_t c0 = now_ns();
                if (ANeuralNetworksModel_create(&h.m)) { ok = false; break; }
                const uint32_t dI[4] = {1u, (uint32_t)L.h, (uint32_t)L.w, (uint32_t)L.cin};
                ANeuralNetworksOperandType tI = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
                                                 4, dI, sa_in, 0};
                std::vector<int32_t> bi((size_t)L.cout, 0);
                // Bias -> INT32 in SA*SW units, per output channel (the filter
                // scale for that channel, whether or not the op takes
                // per-channel). A bias larger than the int32 range at this SA
                // is counted, not hidden.
                for (int c = 0; c < L.cout; ++c) {
                    const double den = (double)sa_in * (double)lsc[c];
                    if (!(den > 0.0)) continue;
                    const double v = (double)lbi[c] / den;
                    if (v > 2147483000.0 || v < -2147483000.0) { bias_clamps++; }
                    bi[c] = (int32_t)(v > 2147483000.0 ? 2147483000.0
                                     : (v < -2147483000.0 ? -2147483000.0 : v));
                }
                if (add_operand(h, &tI, &ia, &why) ||
                    add_conv_op(h, wmem, L.h, L.w, L.cin, L.cout, T.kh, T.kw, T.off,
                                lsc, use_pc, bi.data(), sa_in, sa_in, ia, &io, &why)) { ok = false; break; }
                uint32_t ins[1] = {ia}, outs[1] = {io};
                ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
                int frc = ANeuralNetworksModel_finish(h.m);
                if (frc) { P("FATAL|convseq_finish|layer=%d|%s|rc=%d(%s)", li, L.name.c_str(),
                             frc, rcs(frc)); ok = false; break; }
                ANeuralNetworksCompilation* comp = nullptr;
                int crc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
                if (!crc) {
                    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
                    crc = ANeuralNetworksCompilation_finish(comp);
                }
                compile_ns += now_ns() - c0;
                if (crc) {
                    P("CSCOMPILE|layer=%d|%s|pc=%d|rc=%d(%s)", li, L.name.c_str(), use_pc,
                      crc, rcs(crc));
                    if (comp) ANeuralNetworksCompilation_free(comp);
                    ok = false; break;
                }
                comps.push_back(comp);
                if ((int)ti < 2)
                    P("CSCOMPILE|layer=%d|%s|tap=%d|pc=%d|ms=%.1f|rc=0", li, L.name.c_str(),
                      T.tap, use_pc, (now_ns() - c0) / 1e6);
            }
            if (ok) break;
            // retry this layer per-tensor before giving up on the layer
            for (auto* c : comps) ANeuralNetworksCompilation_free(c);
            comps.clear();
            for (auto& h : holders) { if (h.m) { ANeuralNetworksModel_free(h.m); h.m = nullptr; } }
            if (use_pc == 0) { tapfails++; ok = false; break; }
            use_pc = 0; pcfb++;
            P("NOTE|convseq_layer_pc_fallback|layer=%d|%s", li, L.name.c_str());
        }
        if ((int)comps.size() != (int)L.taps.size()) {
            for (auto* c : comps) ANeuralNetworksCompilation_free(c);
            P("FATAL|convseq_no_graph|layer=%d|%s|compiled=%zu/%zu", li, L.name.c_str(),
              comps.size(), L.taps.size());
            P("DONE"); return 6;
        }

        std::vector<int8_t> outb((size_t)L.frames * plane_out, 0);
        std::vector<int32_t> acc(plane_out, 0);
        std::vector<int8_t> inp(plane_in, 0), zero(plane_in, 0), opb(plane_out, 0);
        int64_t lay0 = now_ns();
        int64_t layexec = 0, layepi = 0, layclamp = 0;
        for (int f = 0; f < L.frames; ++f) {
            std::fill(acc.begin(), acc.end(), 0);
            for (size_t ti = 0; ti < L.taps.size(); ++ti) {
                const SeqTap& T = L.taps[ti];
                const int src = f - T.delta;
                if (src >= 0 && src < L.fin) memcpy(inp.data(), &inb[(size_t)src * plane_in], plane_in);
                else                         memcpy(inp.data(), zero.data(), plane_in);

                ANeuralNetworksExecution* ex = nullptr;
                int rc = ANeuralNetworksExecution_create(comps[ti], &ex);
                if (rc) { P("FATAL|convseq_exec_create|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 8; }
                ANeuralNetworksExecution_setInput(ex, 0, nullptr, inp.data(), inp.size());
                ANeuralNetworksExecution_setOutput(ex, 0, nullptr, opb.data(), opb.size());
                int64_t t0 = now_ns();
                int erc = ANeuralNetworksExecution_compute(ex);
                int64_t t1 = now_ns();
                ANeuralNetworksExecution_free(ex);
                if (erc) { P("FATAL|convseq_compute|layer=%d|tap=%d|frame=%d|rc=%d(%s)",
                             li, T.tap, f, erc, rcs(erc)); P("DONE"); return 8; }
                P("TS|convseq|l=%d|t=%d|f=%d|t0=%.3f|t1=%.3f|ms=%.3f",
                  li, T.tap, f, (double)t0 / 1e6, (double)t1 / 1e6, (t1 - t0) / 1e6);
                execs++; layexec++;
                for (size_t i = 0; i < plane_out; ++i) acc[i] += (int32_t)opb[i];
            }
            // epilogue: accumulate the per-tap INT8 outputs — each is already at
            // THIS layer's SA, with the weight scale folded in by the HAL — and
            // requantise to the CONSUMER layer's SA, so the next wiring step is a
            // pure byte copy. The pre-P3.10 line here was `g = lsc[c]/VAE_SA`,
            // i.e. it multiplied an *activation* accumulator by a *weight* scale;
            // measured 2026-11-16 and removed, with saturation counted instead of
            // assumed away.
            int8_t* dst = &outb[(size_t)f * plane_out];   // HWC, same as the NNAPI tensor
            const size_t plane = (size_t)L.h * L.w;
            const double g = (double)sa_in / (double)sa_out;
            const int64_t e0 = now_ns();
            for (int c = 0; c < L.cout; ++c) {
                for (size_t i = 0; i < plane; ++i) {
                    const int32_t a = acc[i * (size_t)L.cout + (size_t)c];
                    long v = (long)((double)a * g + (a >= 0 ? 0.5 : -0.5));
                    if (v > 127) { v = 127; layclamp++; }
                    if (v < -127) { v = -127; layclamp++; }
                    dst[i * (size_t)L.cout + (size_t)c] = (int8_t)v;
                }
            }
            layepi += now_ns() - e0;
        }
        const double layms = (now_ns() - lay0) / 1e6;
        int64_t lmacs = 0;
        for (auto& T : L.taps)
            lmacs += (int64_t)L.h * L.w * L.cout * L.cin * T.kh * T.kw * (int64_t)L.frames;
        macs += lmacs;
        P("CSLAYER|layer=%d|%s|frames=%d|taps=%zu|executes=%lld|ms=%.1f|macs=%lld|gops=%.1f|pc=%d|"
          "sa=%.6g|sa_out=%.6g|epi_ms=%.1f|clamp=%.4f",
          li, L.name.c_str(), L.frames, L.taps.size(), (long long)layexec, layms,
          (long long)lmacs, 2.0 * (double)lmacs / (layms / 1000.0) / 1e9, use_pc,
          (double)sa_in, (double)sa_out, layepi / 1e6,
          (double)layclamp / (double)((int64_t)L.frames * (int64_t)plane_out));

        for (auto* c : comps) ANeuralNetworksCompilation_free(c);
        for (auto& h : holders) { if (h.m) { ANeuralNetworksModel_free(h.m); h.m = nullptr; } }

        prev.swap(outb);
        pf = L.frames; pc_ = L.cout; ph = L.h; pw = L.w;
    }

    const double total_s = (now_ns() - run0) / 1e9;
    P("SUMMARY|mode=convseq|layers=%d|executes=%lld|compile_s=%.2f|warm_s=%.3f|wall_s=%.3f|"
      "macs=%lld|gops=%.1f|pc=%d|pc_fallbacks=%lld|tap_failures=%lld|sa=%s|bias_clamps=%lld",
      nlay, (long long)execs, compile_ns / 1e9, total_s - compile_ns / 1e9, total_s,
      (long long)macs, 2.0 * (double)macs / total_s / 1e9, pcmode,
      (long long)pcfb, (long long)tapfails, sa_table ? "table" : "nominal",
      (long long)bias_clamps);
    P("DONE");
    return 0;
}

// ================================================ convwire: the WIRED decode
// `convseq` executed the 592 CONV_2D dispatches and left everything between them
// unwired: no norm, no SiLU, no residual add, and its activation scale table was
// indexed by POSITION in a storage order that `WanResidualBlock.forward` does not
// follow. P3.11 measured what those ops are — a per-pixel `WanRMS_norm`
// (F.normalize over channels * sqrt(C) * gamma) followed by SiLU between the two
// convs of every resnet, 14 residual adds, 2 temporal shuffles, 3 2-D upsamples
// and one mid-block attention — so this mode walks a LINEAR PROGRAM instead:
//
//   OP CONV <li>                              one artifact layer (taps + frames)
//   OP SAVE <slot> / OP LOAD <slot>            int8 activation copies (residuals)
//   OP NORMSILU <goff> <C> <scale> <out_sa>    one fused float pass
//   OP NORM <goff> <C> <scale> <out_sa>        a norm with no SiLU after it
//   OP SILU <out_sa>
//   OP ADD <slot> <out_sa> <max_abs>
//   OP SKIP <what>                             folded into a conv's gather, or the
//                                              mid-block attention (not executed)
//
// The activation is int8 throughout at a declared per-tensor scale, so every
// float op is one streaming pass and never a full float copy (the largest plane
// is 9x96x272x480 = 112.8 MB int8, 4x that in f32). A conv's INPUT operand is at
// the layer's measured SA_in and its OUTPUT operand at its measured SA_out —
// those differ by 0.275..15.5x across the decode, so they are not
// interchangeable, and quantising the output at the consumer's scale (what
// `convseq` did) threw resolution away wherever SA_out < SA_in.
static const char* VAE_WIRE_PATH = "/data/local/tmp/vae_wire.txt";
static const char* VAE_SA2_PATH  = "/data/local/tmp/vae_sa2.f32";    // 2 x 36: SA_in ++ SA_out
static const char* VAE_NM_PATH   = "/data/local/tmp/vae_norms.f32";  // 29 gammas, concatenated
// P3.13: the walk had no way to show its OUTPUT. These two files are the decode's
// evidence — the final activation codes and the latent codes the first conv
// actually consumed — so the host can dequantise both and PSNR the device's
// decode against the fp32 reference from the SAME quantised latent.
static const char* VAE_DUMP_PATH = "/data/local/tmp/vae_dump.bin";   // final act, [f][h][w][c] int8
static const char* VAE_DUMP_META = "/data/local/tmp/vae_dump.txt";
static const char* VAE_LAT_PATH  = "/data/local/tmp/vae_latent.bin"; // the latent codes, same layout
static const char* VAE_LAT_META  = "/data/local/tmp/vae_latent.txt";

struct WireOp { int kind, a, b; long long goff; double p1, p2; };
enum { W_CONV = 0, W_SAVE, W_LOAD, W_NORMSILU, W_NORM, W_SILU, W_ADD, W_SKIP };

static int load_wire(std::vector<SeqLayer>* out, std::vector<WireOp>* prog) {
    FILE* f = fopen(VAE_WIRE_PATH, "r");
    if (!f) { P("FATAL|wire_open|%s", VAE_WIRE_PATH); return -1; }
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == 10 || line[0] == 13) continue;
        if (!strncmp(line, "LAYER ", 6)) {
            SeqLayer L; char nm[192] = {0}; int idx = 0;
            // LAYER <idx> <frames> <frames_in> <cout> <cin> <h> <w> <ntaps>
            //       <wiring> <sc_off> <artifact_index> <name>
            int n = sscanf(line + 6, "%d %d %d %d %d %d %d %d %d %lld %d %191s",
                           &idx, &L.frames, &L.fin, &L.cout, &L.cin, &L.h, &L.w,
                           &L.ntaps, &L.wiring, &L.sc_off, &L.ai, nm);
            if (n < 12) { P("FATAL|wire_layer_parse|%s", line); fclose(f); return -1; }
            L.name = nm;
            out->push_back(L);
        } else if (!strncmp(line, "TAP ", 4)) {
            if (out->empty()) { P("FATAL|wire_tap_without_layer"); fclose(f); return -1; }
            SeqTap t; int n = sscanf(line + 4, "%d %d %lld %lld %d %d",
                                     &t.tap, &t.delta, &t.off, &t.bytes, &t.kh, &t.kw);
            if (n < 6) { P("FATAL|wire_tap_parse|%s", line); fclose(f); return -1; }
            out->back().taps.push_back(t);
        } else if (!strncmp(line, "OP ", 3)) {
            char k[24] = {0};
            WireOp o;
            memset(&o, 0, sizeof o);
            if (sscanf(line + 3, "%23s", k) != 1) continue;
            if (!strcmp(k, "CONV")) {
                if (sscanf(line + 3, "%23s %d", k, &o.a) != 2) { fclose(f); return -1; }
                o.kind = W_CONV;
            } else if (!strcmp(k, "SAVE")) {
                if (sscanf(line + 3, "%23s %d", k, &o.a) != 2) { fclose(f); return -1; }
                o.kind = W_SAVE;
            } else if (!strcmp(k, "LOAD")) {
                if (sscanf(line + 3, "%23s %d", k, &o.a) != 2) { fclose(f); return -1; }
                o.kind = W_LOAD;
            } else if (!strcmp(k, "NORMSILU")) {
                o.kind = W_NORMSILU;
                if (sscanf(line + 3, "%23s %lld %d %lf %lf", k, &o.goff, &o.a, &o.p1, &o.p2) != 5) {
                    fclose(f); return -1;
                }
            } else if (!strcmp(k, "NORM")) {
                o.kind = W_NORM;
                if (sscanf(line + 3, "%23s %lld %d %lf %lf", k, &o.goff, &o.a, &o.p1, &o.p2) != 5) {
                    fclose(f); return -1;
                }
            } else if (!strcmp(k, "SILU")) {
                o.kind = W_SILU;
                if (sscanf(line + 3, "%23s %lf", k, &o.p1) != 2) { fclose(f); return -1; }
            } else if (!strcmp(k, "ADD")) {
                o.kind = W_ADD;
                if (sscanf(line + 3, "%23s %d %lf %lf", k, &o.a, &o.p1, &o.p2) != 4) {
                    fclose(f); return -1;
                }
            } else {
                o.kind = W_SKIP;
            }
            prog->push_back(o);
        }
    }
    fclose(f);
    if (out->empty() || prog->empty()) { P("FATAL|wire_empty"); return -1; }
    size_t taps = 0;
    for (auto& L : *out) taps += L.taps.size();
    P("WIRE|layers=%zu|taps=%zu|ops=%zu", out->size(), taps, prog->size());
    return 0;
}

struct ActBuf {
    std::vector<int8_t> d; int f = 0, c = 0, h = 0, w = 0; float sa = 0.0f; bool live = false;
    void shape(int nf, int nc, int nh, int nw, float s) {
        f = nf; c = nc; h = nh; w = nw; sa = s; live = true;
        d.assign((size_t)f * (size_t)c * (size_t)h * (size_t)w, 0);
    }
};

static int run_convwire(ANeuralNetworksDevice* tpu, ANeuralNetworksMemory* wmem,
                        int max_convs, int pcmode, int dump, int dumplayer) {
    const char* why = "";
    std::vector<SeqLayer> planrec;
    std::vector<WireOp> prog;
    if (load_wire(&planrec, &prog)) { P("FATAL|wire_load"); P("DONE"); return 2; }
    const int nlay = (int)planrec.size();

    std::vector<uint8_t> scraw = slurp(VAE_SC_PATH, "scales");
    std::vector<uint8_t> biraw = slurp(VAE_BI_PATH, "bias");
    std::vector<uint8_t> sa2raw = slurp(VAE_SA2_PATH, "sa2");
    std::vector<uint8_t> nmraw = slurp(VAE_NM_PATH, "norms");
    if (scraw.empty() || biraw.empty() || sa2raw.empty() || nmraw.empty()) { P("DONE"); return 2; }
    const float* SC = (const float*)scraw.data();
    const float* BI = (const float*)biraw.data();
    const float* SA2 = (const float*)sa2raw.data();
    const float* NM = (const float*)nmraw.data();
    const size_t nsc = scraw.size() / 4, nbi = biraw.size() / 4;
    const size_t nsa = sa2raw.size() / 4, ngam = nmraw.size() / 4;
    if (nsa != 2 * (size_t)nlay) {
        P("FATAL|wire_sa2_len|floats=%zu|layers=%d", nsa, nlay); P("DONE"); return 2;
    }
    // The artifact index must be a PERMUTATION: it is what the SA / weight-scale /
    // bias tables are indexed by, so a duplicate would silently apply one layer's
    // calibration to another.
    std::vector<int> seen(nlay, 0);
    for (auto& L : planrec) {
        if (L.ai < 0 || L.ai >= nlay || seen[L.ai]++) {
            P("FATAL|wire_ai_not_permutation|layer=%s|ai=%d", L.name.c_str(), L.ai);
            P("DONE"); return 2;
        }
    }
    P("WIRE|sa2=table|in=%.6g..%.6g|out=%.6g..%.6g|gammas=%zu",
      (double)*std::min_element(SA2, SA2 + nlay), (double)*std::max_element(SA2, SA2 + nlay),
      (double)*std::min_element(SA2 + nlay, SA2 + 2 * nlay),
      (double)*std::max_element(SA2 + nlay, SA2 + 2 * nlay), ngam);

    if (pcmode < 0) {
        pcmode = probe_conv_pc(tpu, wmem);
        P("NOTE|convwire_pc_mode=%d", pcmode);
    }

    ActBuf act, slot[2];
    // The latent the walk actually consumed (wiring 3 builds it once). Captured so
    // the host never has to re-derive the LCG — it dequantises these exact codes.
    std::vector<int8_t> lat;
    int lat_f = 0, lat_c = 0, lat_h = 0, lat_w = 0;
    double lat_sa = 0.0;
    int64_t convs = 0, execs = 0, macs = 0, compile_ns = 0, tapfails = 0, pcfb = 0;
    int64_t prep_ns = 0, epi_ns = 0, silk_ns = 0, add_ns = 0, slot_ns = 0;
    int64_t clamps = 0, el_norm = 0, el_add = 0;
    double conv_ms_total = 0.0;
    int started = 0;
    int64_t run0 = now_ns();

    for (size_t pi = 0; pi < prog.size(); ++pi) {
        const WireOp& op = prog[pi];
        if (max_convs > 0 && started >= max_convs && op.kind == W_CONV) break;
        if (op.kind == W_SKIP) continue;

        if (op.kind == W_SAVE || op.kind == W_LOAD) {
            ActBuf& s = slot[op.a & 1];
            const int64_t t0 = now_ns();
            if (op.kind == W_SAVE) {
                if (!act.live) { P("FATAL|wire_save_no_act"); P("DONE"); return 5; }
                if (s.d.size() != act.d.size()) s.shape(act.f, act.c, act.h, act.w, act.sa);
                else { s.f = act.f; s.c = act.c; s.h = act.h; s.w = act.w; s.sa = act.sa; s.live = true; }
                memcpy(s.d.data(), act.d.data(), act.d.size());
            } else {
                if (!s.live) { P("FATAL|wire_load_empty_slot|slot=%d", op.a); P("DONE"); return 5; }
                act.shape(s.f, s.c, s.h, s.w, s.sa);
                memcpy(act.d.data(), s.d.data(), s.d.size());
            }
            slot_ns += now_ns() - t0;
            continue;
        }

        if (op.kind == W_NORMSILU || op.kind == W_NORM || op.kind == W_SILU) {
            if (!act.live) { P("FATAL|wire_norm_no_act"); P("DONE"); return 5; }
            const int C = (op.kind == W_SILU) ? act.c : op.a;
            if (C != act.c) {
                P("FATAL|wire_norm_C|kind=%d|C=%d|act=%d", op.kind, C, act.c); P("DONE"); return 5;
            }
            const double out_sa = (op.kind == W_SILU) ? op.p1 : op.p2;
            if (!(out_sa > 0.0)) { P("FATAL|wire_norm_sa|%.9g", out_sa); P("DONE"); return 5; }
            const bool do_norm = (op.kind != W_SILU);
            const bool do_silu = (op.kind != W_NORM);
            const float gam_scale = (float)op.p1;          // sqrt(C), as the exporter wrote it
            const float* gam = do_norm ? (NM + op.goff) : nullptr;
            if (do_norm && (size_t)op.goff + (size_t)C > ngam) {
                P("FATAL|wire_norm_gamma_range|goff=%lld|C=%d|ngam=%zu", op.goff, C, ngam);
                P("DONE"); return 5;
            }
            const float sain = act.sa;
            const size_t plane = (size_t)act.h * (size_t)act.w;
            const int64_t t0 = now_ns();
            for (int f = 0; f < act.f; ++f) {
                for (size_t i = 0; i < plane; ++i) {
                    int8_t* p = &act.d[((size_t)f * plane + i) * (size_t)C];
                    float den = 1.0f;
                    if (do_norm) {
                        float ss = 0.0f;
                        for (int c = 0; c < C; ++c) {
                            const float x = (float)p[c] * sain;
                            ss += x * x;
                        }
                        const float nr = sqrtf(ss);
                        den = nr > 1e-12f ? nr : 1e-12f;   // F.normalize's clamp_min(eps)
                    }
                    for (int c = 0; c < C; ++c) {
                        float y = (float)p[c] * sain / den;
                        if (do_norm) y = y * gam_scale * gam[c];
                        if (do_silu) y = y / (1.0f + expf(-y));
                        const long q = (long)lroundf(y / (float)out_sa);
                        if (q > 127) { clamps++; p[c] = 127; }
                        else if (q < -127) { clamps++; p[c] = -127; }
                        else p[c] = (int8_t)q;
                    }
                    el_norm += C;
                }
            }
            act.sa = (float)out_sa;
            silk_ns += now_ns() - t0;
            continue;
        }

        if (op.kind == W_ADD) {
            ActBuf& s = slot[op.a & 1];
            if (!act.live || !s.live || s.d.size() != act.d.size()) {
                P("FATAL|wire_add_shape|slot=%d|act=%zu|slot_n=%zu", op.a, act.d.size(), s.d.size());
                P("DONE"); return 5;
            }
            const double out_sa = op.p1;
            const double ka = (double)act.sa, ks = (double)s.sa;
            const int64_t t0 = now_ns();
            for (size_t i = 0; i < act.d.size(); ++i) {
                const double v = ((double)act.d[i] * ka + (double)s.d[i] * ks) / out_sa;
                const long q = (long)(v >= 0 ? v + 0.5 : v - 0.5);
                if (q > 127) { clamps++; act.d[i] = 127; }
                else if (q < -127) { clamps++; act.d[i] = -127; }
                else act.d[i] = (int8_t)q;
            }
            act.sa = (float)out_sa;
            el_add += (int64_t)act.d.size();
            add_ns += now_ns() - t0;
            continue;
        }

        // ------------------------------------------------------------ W_CONV
        ++started;
        SeqLayer& L = planrec[op.a];
        const int ai = L.ai;
        const float sa_in = SA2[ai], sa_out = SA2[nlay + ai];
        if (!(sa_in > 0.0f) || !(sa_out > 0.0f)) {
            P("FATAL|wire_conv_sa|layer=%d|in=%.9g|out=%.9g", (int)op.a, (double)sa_in, (double)sa_out);
            P("DONE"); return 5;
        }
        if (L.ntaps != (int)L.taps.size()) {
            P("FATAL|wire_tap_count|layer=%d|%s|declared=%d|read=%zu", (int)op.a,
              L.name.c_str(), L.ntaps, L.taps.size());
            P("DONE"); return 5;
        }
        const size_t plane_in = (size_t)L.h * L.w * L.cin;
        const size_t plane_out = (size_t)L.h * L.w * L.cout;
        if (L.sc_off < 0 || (size_t)L.sc_off + (size_t)L.cout > nsc ||
            (size_t)L.sc_off + (size_t)L.cout > nbi) {
            P("FATAL|wire_scale_range|layer=%d|off=%lld|cout=%d", (int)op.a, L.sc_off, L.cout);
            P("DONE"); return 5;
        }
        const float* lsc = SC + L.sc_off;
        const float* lbi = BI + L.sc_off;

        // ---- build this layer's input codes at its own SA_in. The shuffle and
        // the 2x nearest upsample the plan lists as separate ops live here: they
        // are a re-layout of this conv's input, not a new tensor.
        std::vector<int8_t> inb((size_t)L.fin * plane_in, 0);
        const int64_t p0 = now_ns();
        if (L.wiring == 3) {
            uint32_t x = 0x9E3779B9u;                 // the decode's synthetic latent
            for (size_t i = 0; i < inb.size(); ++i) {
                x = x * 1664525u + 1013904223u;
                inb[i] = (int8_t)((x >> 24) & 0xFF);
            }
            if (dump && lat.empty()) {
                lat = inb; lat_f = L.fin; lat_c = L.cin; lat_h = L.h; lat_w = L.w;
                lat_sa = (double)sa_in;
            }
        } else {
            if (!act.live) { P("FATAL|wire_conv_no_act|layer=%d", (int)op.a); P("DONE"); return 5; }
            const int c = L.cin;
            const bool tshuf = (L.wiring == 1 || L.wiring == 2);
            const bool spatial = (L.wiring == 2 || L.wiring == 5);
            const int want_f = tshuf ? (2 * act.f - 1) : act.f;
            const int want_c = tshuf ? 2 * c : c;
            const int want_h = spatial ? 2 * act.h : act.h;
            const int want_w = spatial ? 2 * act.w : act.w;
            if (want_f != L.fin || want_c != act.c || want_h != L.h || want_w != L.w) {
                P("FATAL|wire_wiring_mismatch|layer=%d|%s|wiring=%d|want f=%d c=%d h=%d w=%d|"
                  "act f=%d c=%d h=%d w=%d", (int)op.a, L.name.c_str(), L.wiring,
                  L.fin, want_c, L.h, L.w, act.f, act.c, act.h, act.w);
                P("DONE"); return 5;
            }
            const double g = (double)act.sa / (double)sa_in;
            const bool plain = (g == 1.0) && (L.wiring == 0);
            for (int t = 0; t < L.fin; ++t) {
                const int srcf = tshuf ? (t / 2) : t;
                const int half = tshuf ? (t % 2) : 0;
                for (int hh = 0; hh < L.h; ++hh) {
                    const int sh = spatial ? (hh / 2) : hh;
                    for (int ww = 0; ww < L.w; ++ww) {
                        const int sw = spatial ? (ww / 2) : ww;
                        int8_t* dst = &inb[((size_t)t * L.h * L.w + (size_t)hh * L.w + ww) * (size_t)c];
                        if (srcf >= act.f) { memset(dst, 0, (size_t)c); continue; }
                        const int8_t* src = &act.d[((size_t)srcf * act.h * act.w +
                                                   (size_t)sh * act.w + sw) * (size_t)act.c +
                                                  (size_t)half * c];
                        if (plain) { memcpy(dst, src, (size_t)c); continue; }
                        for (int k = 0; k < c; ++k) {
                            const double v = (double)src[k] * g;
                            long q = (long)(v >= 0 ? v + 0.5 : v - 0.5);
                            if (q > 127) q = 127;
                            if (q < -127) q = -127;
                            dst[k] = (int8_t)q;
                        }
                    }
                }
            }
        }
        const int64_t prep_this = now_ns() - p0;
        prep_ns += prep_this;

        // ---- one graph per tap, then frames x taps
        std::vector<ANeuralNetworksCompilation*> comps;
        std::vector<Holder> holders(L.taps.size());
        int use_pc = pcmode;
        const int64_t c0 = now_ns();
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool ok = true;
            for (size_t ti = 0; ti < L.taps.size(); ++ti) {
                const SeqTap& T = L.taps[ti];
                Holder& h = holders[ti];
                uint32_t ia, io;
                if (ANeuralNetworksModel_create(&h.m)) { ok = false; break; }
                const uint32_t dI[4] = {1u, (uint32_t)L.h, (uint32_t)L.w, (uint32_t)L.cin};
                ANeuralNetworksOperandType tI = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
                                                 4, dI, sa_in, 0};
                std::vector<int32_t> bi((size_t)L.cout, 0);
                for (int c = 0; c < L.cout; ++c) {
                    const double den = (double)sa_in * (double)lsc[c];
                    if (!(den > 0.0)) continue;
                    const double v = (double)lbi[c] / den;
                    bi[c] = (int32_t)(v > 2147483000.0 ? 2147483000.0
                                     : (v < -2147483000.0 ? -2147483000.0 : v));
                }
                if (add_operand(h, &tI, &ia, &why) ||
                    add_conv_op(h, wmem, L.h, L.w, L.cin, L.cout, T.kh, T.kw, T.off,
                                lsc, use_pc, bi.data(), sa_in, sa_out, ia, &io, &why)) {
                    ok = false; break;
                }
                uint32_t ins[1] = {ia}, outs[1] = {io};
                ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
                const int frc = ANeuralNetworksModel_finish(h.m);
                if (frc) {
                    P("FATAL|wire_finish|layer=%d|%s|rc=%d(%s)", (int)op.a, L.name.c_str(),
                      frc, rcs(frc));
                    ok = false; break;
                }
                ANeuralNetworksCompilation* comp = nullptr;
                int crc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
                if (!crc) {
                    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
                    crc = ANeuralNetworksCompilation_finish(comp);
                }
                if (crc) {
                    P("CSCOMPILE|layer=%d|%s|pc=%d|rc=%d(%s)", (int)op.a, L.name.c_str(),
                      use_pc, crc, rcs(crc));
                    if (comp) ANeuralNetworksCompilation_free(comp);
                    ok = false; break;
                }
                comps.push_back(comp);
            }
            if (ok) break;
            for (auto* c : comps) ANeuralNetworksCompilation_free(c);
            comps.clear();
            for (auto& h : holders) { if (h.m) { ANeuralNetworksModel_free(h.m); h.m = nullptr; } }
            if (use_pc == 0) { tapfails++; break; }
            use_pc = 0; pcfb++;
            P("NOTE|convwire_layer_pc_fallback|layer=%d|%s", (int)op.a, L.name.c_str());
        }
        compile_ns += now_ns() - c0;
        if ((int)comps.size() != (int)L.taps.size()) {
            for (auto* c : comps) ANeuralNetworksCompilation_free(c);
            P("FATAL|wire_no_graph|layer=%d|%s|compiled=%zu/%zu", (int)op.a, L.name.c_str(),
              comps.size(), L.taps.size());
            P("DONE"); return 6;
        }

        std::vector<int8_t> outb((size_t)L.frames * plane_out, 0);
        std::vector<int32_t> acc(plane_out, 0);
        std::vector<int8_t> inp(plane_in, 0), zero(plane_in, 0), opb(plane_out, 0);
        const int64_t lay0 = now_ns();
        int64_t layexec = 0, layepi = 0, layclamp = 0;
        for (int f = 0; f < L.frames; ++f) {
            std::fill(acc.begin(), acc.end(), 0);
            for (size_t ti = 0; ti < L.taps.size(); ++ti) {
                const SeqTap& T = L.taps[ti];
                const int src = f - T.delta;
                if (src >= 0 && src < L.fin) memcpy(inp.data(), &inb[(size_t)src * plane_in], plane_in);
                else                          memcpy(inp.data(), zero.data(), plane_in);

                ANeuralNetworksExecution* ex = nullptr;
                int rc = ANeuralNetworksExecution_create(comps[ti], &ex);
                if (rc) { P("FATAL|wire_exec_create|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 8; }
                ANeuralNetworksExecution_setInput(ex, 0, nullptr, inp.data(), inp.size());
                ANeuralNetworksExecution_setOutput(ex, 0, nullptr, opb.data(), opb.size());
                const int64_t t0 = now_ns();
                const int erc = ANeuralNetworksExecution_compute(ex);
                const int64_t t1 = now_ns();
                ANeuralNetworksExecution_free(ex);
                if (erc) { P("FATAL|wire_compute|layer=%d|tap=%d|frame=%d|rc=%d(%s)",
                             (int)op.a, T.tap, f, erc, rcs(erc)); P("DONE"); return 8; }
                P("TS|convwire|l=%d|t=%d|f=%d|t0=%.3f|t1=%.3f|ms=%.3f",
                  (int)op.a, T.tap, f, (double)t0 / 1e6, (double)t1 / 1e6, (t1 - t0) / 1e6);
                execs++; layexec++;
                for (size_t i = 0; i < plane_out; ++i) acc[i] += (int32_t)opb[i];
            }
            // Each tap's int8 output is ALREADY at this layer's SA_out (the output
            // operand carries it), so the summed int32 IS in SA_out units: clamp
            // and store, no rescale. Saturation is counted, never hidden.
            int8_t* dst = &outb[(size_t)f * plane_out];
            const int64_t e0 = now_ns();
            for (size_t i = 0; i < plane_out; ++i) {
                const int32_t a = acc[i];
                if (a > 127) { dst[i] = 127; layclamp++; }
                else if (a < -127) { dst[i] = -127; layclamp++; }
                else dst[i] = (int8_t)a;
            }
            layepi += now_ns() - e0;
        }
        const double layms = (now_ns() - lay0) / 1e6;
        int64_t lmacs = 0;
        for (auto& T : L.taps)
            lmacs += (int64_t)L.h * L.w * L.cout * L.cin * T.kh * T.kw * (int64_t)L.frames;
        macs += lmacs; convs++;
        epi_ns += layepi;
        clamps += layclamp;
        conv_ms_total += layms;
        P("CWLAYER|layer=%d|ai=%d|name=%s|frames=%d|taps=%zu|executes=%lld|conv_ms=%.1f|"
          "prep_ms=%.1f|epi_ms=%.1f|macs=%lld|gops=%.1f|pc=%d|sa_in=%.6g|sa_out=%.6g|clamp=%.4f",
          (int)op.a, ai, L.name.c_str(), L.frames, L.taps.size(), (long long)layexec, layms,
          (double)prep_this / 1e6, layepi / 1e6, (long long)lmacs,
          2.0 * (double)lmacs / (layms / 1000.0) / 1e9, use_pc, (double)sa_in, (double)sa_out,
          (double)layclamp / (double)((int64_t)L.frames * (int64_t)plane_out));

        for (auto* c : comps) ANeuralNetworksCompilation_free(c);
        for (auto& h : holders) { if (h.m) { ANeuralNetworksModel_free(h.m); h.m = nullptr; } }

        act.shape(L.frames, L.cout, L.h, L.w, sa_out);
        memcpy(act.d.data(), outb.data(), outb.size());

        // P3.13 parity probe: dump THIS conv's exact int8 input and output and stop.
        // A single layer with both ends is a self-contained test — no wiring, no
        // norm, nothing assumed: the host can emulate the taps from the same codes.
        if (dump && dumplayer == (int)op.a) {
            char pbo[256], pbi[256];
            snprintf(pbo, sizeof pbo, "/data/local/tmp/vae_dump_out_l%02d.bin", (int)op.a);
            snprintf(pbi, sizeof pbi, "/data/local/tmp/vae_dump_in_l%02d.bin", (int)op.a);
            FILE* f = fopen(pbo, "wb");
            if (!f) { P("FATAL|dump_layer_open|%s", pbo); P("DONE"); return 5; }
            const size_t n1 = fwrite(outb.data(), 1, outb.size(), f); fclose(f);
            f = fopen(pbi, "wb");
            if (!f) { P("FATAL|dump_layer_open|%s", pbi); P("DONE"); return 5; }
            const size_t n2 = fwrite(inb.data(), 1, inb.size(), f); fclose(f);
            char pm[256];
            snprintf(pm, sizeof pm, "/data/local/tmp/vae_dump_out_l%02d.txt", (int)op.a);
            f = fopen(pm, "w");
            if (f) { fprintf(f, "layout=fhwc\nf=%d\nc=%d\nh=%d\nw=%d\nsa=%.9g\nbytes=%zu\n",
                             L.frames, L.cout, L.h, L.w, (double)sa_out, outb.size()); fclose(f); }
            snprintf(pm, sizeof pm, "/data/local/tmp/vae_dump_in_l%02d.txt", (int)op.a);
            f = fopen(pm, "w");
            if (f) { fprintf(f, "layout=fhwc\nf=%d\nc=%d\nh=%d\nw=%d\nsa=%.9g\nbytes=%zu\n",
                             L.fin, L.cin, L.h, L.w, (double)sa_in, inb.size()); fclose(f); }
            P("DUMPLAYER|layer=%d|name=%s|out=%s|out_bytes=%zu|in=%s|in_bytes=%zu|"
              "sa_in=%.9g|sa_out=%.9g|frames=%d|fin=%d|taps=%zu|clamp=%lld/%lld",
              (int)op.a, L.name.c_str(), pbo, n1, pbi, n2, (double)sa_in, (double)sa_out,
              L.frames, L.fin, L.taps.size(), (long long)layclamp,
              (long long)((int64_t)L.frames * (int64_t)plane_out));
            P("DONE");
            return 0;
        }
    }

    // ---- P3.13: dump the final activation and the consumed latent. Without this
    // the walk can only prove it EXECUTED; the codes are what let the host PSNR the
    // device's decode against the fp32 reference from the identical quantised
    // latent. Layout [f][h][w][c] int8 (the harness's internal order), dequantised
    // with each tensor's own declared scale, which is written beside it.
    if (dump) {
        if (!act.live) { P("FATAL|wire_dump_no_act"); P("DONE"); return 5; }
        FILE* d = fopen(VAE_DUMP_PATH, "wb");
        if (!d) { P("FATAL|wire_dump_open|%s", VAE_DUMP_PATH); P("DONE"); return 5; }
        const size_t nw = fwrite(act.d.data(), 1, act.d.size(), d);
        fclose(d);
        if (nw != act.d.size()) { P("FATAL|wire_dump_short|%zu/%zu", nw, act.d.size()); P("DONE"); return 5; }
        FILE* m = fopen(VAE_DUMP_META, "w");
        if (m) {
            fprintf(m, "layout=fhwc\nf=%d\nc=%d\nh=%d\nw=%d\nsa=%.9g\nbytes=%zu\nmacs=%lld\n",
                    act.f, act.c, act.h, act.w, (double)act.sa, act.d.size(), (long long)macs);
            fclose(m);
        }
        P("DUMP|path=%s|bytes=%zu|f=%d|c=%d|h=%d|w=%d|sa=%.9g",
          VAE_DUMP_PATH, act.d.size(), act.f, act.c, act.h, act.w, (double)act.sa);
        if (!lat.empty()) {
            FILE* lf = fopen(VAE_LAT_PATH, "wb");
            if (lf) {
                const size_t lw = fwrite(lat.data(), 1, lat.size(), lf);
                fclose(lf);
                FILE* lm = fopen(VAE_LAT_META, "w");
                if (lm) {
                    fprintf(lm, "layout=fhwc\nf=%d\nc=%d\nh=%d\nw=%d\nsa=%.9g\nbytes=%zu\n",
                            lat_f, lat_c, lat_h, lat_w, lat_sa, lat.size());
                    fclose(lm);
                }
                P("DUMP|path=%s|bytes=%zu|f=%d|c=%d|h=%d|w=%d|sa=%.9g",
                  VAE_LAT_PATH, lat.size(), lat_f, lat_c, lat_h, lat_w, lat_sa);
            }
        }
    }

    const double total_s = (now_ns() - run0) / 1e9;
    P("SUMMARY|mode=convwire|convs=%lld|executes=%lld|compile_s=%.2f|conv_ms=%.1f|"
      "silk_ms=%.1f|add_ms=%.1f|prep_ms=%.1f|epi_ms=%.1f|slot_ms=%.1f|wall_s=%.3f|"
      "macs=%lld|pc=%d|pc_fallbacks=%lld|tap_failures=%lld|clamp=%.4f|el_norm=%lld|el_add=%lld",
      (long long)convs, (long long)execs, compile_ns / 1e9, conv_ms_total,
      silk_ns / 1e6, add_ns / 1e6, prep_ns / 1e6, epi_ns / 1e6, slot_ns / 1e6, total_s,
      (long long)macs, pcmode, (long long)pcfb, (long long)tapfails,
      100.0 * (double)clamps / (double)std::max<int64_t>(1, el_norm + el_add),
      (long long)el_norm, (long long)el_add);
    P("DONE");
    return 0;
}

// -------------------------------------------------------------------- runner
struct RunCfg { int M; int tiles; int iters; };

// Compile `comp` and time `tiles` executions of it, returning per-tile ms.
static int time_tiles(ANeuralNetworksCompilation* comp, const RunCfg& cfg, uint32_t out_id,
                      int out_n, int in_k, std::vector<double>* tile_ms, std::vector<double>* hw_ms,
                      const char* tag, const char** why) {
    std::vector<int8_t> A((size_t)cfg.M * (size_t)in_k, 1);
    std::vector<int8_t> O((size_t)cfg.M * (size_t)out_n, 0);

    for (int it = 0; it < cfg.iters; ++it) {
        for (int t = 0; t < cfg.tiles; ++t) {
            ANeuralNetworksExecution* ex = nullptr;
            int rc = ANeuralNetworksExecution_create(comp, &ex);
            if (rc) { P("FATAL|exec_create|%s|rc=%d(%s)", tag, rc, rcs(rc)); *why = "exec_create"; return -rc; }
            ANeuralNetworksExecution_setMeasureTiming(ex, true);
            rc = ANeuralNetworksExecution_setInput(ex, 0, nullptr, A.data(), A.size());
            if (rc) { P("FATAL|setInput|%s|rc=%d(%s)", tag, rc, rcs(rc)); *why = "setInput";
                      ANeuralNetworksExecution_free(ex); return -rc; }
            rc = ANeuralNetworksExecution_setOutput(ex, 0, nullptr, O.data(), O.size());
            if (rc) { P("FATAL|setOutput|%s|rc=%d(%s)", tag, rc, rcs(rc)); *why = "setOutput";
                      ANeuralNetworksExecution_free(ex); return -rc; }

            int64_t t0 = now_ns();
            rc = ANeuralNetworksExecution_compute(ex);
            int64_t t1 = now_ns();
            if (rc) { P("FATAL|compute|%s|tile=%d|iter=%d|rc=%d(%s)", tag, t, it, rc, rcs(rc));
                      *why = "compute"; ANeuralNetworksExecution_free(ex); return -rc; }

            uint64_t dur = UINT64_MAX;
            ANeuralNetworksExecution_getDuration(ex, ANEURALNETWORKS_DURATION_ON_HARDWARE, &dur);
            ANeuralNetworksExecution_free(ex);

            tile_ms->push_back((t1 - t0) / 1e6);
            hw_ms->push_back(dur == UINT64_MAX ? -1.0 : (double)dur / 1e6);
            // Absolute CLOCK_MONOTONIC stamps: the co-run driver uses these to
            // prove (and quantify) temporal overlap with another process.
            P("TS|npu|%s|tile=%d|iter=%d|t0=%.3f|t1=%.3f|ms=%.3f",
              tag, t, it, (double)t0 / 1e6, (double)t1 / 1e6, (t1 - t0) / 1e6);
            if (it == 0)
                P("TILE|%s|tile=%d|ms=%.3f|ms_hw=%.3f|out0=%d", tag, t,
                  (t1 - t0) / 1e6, dur == UINT64_MAX ? -1.0 : (double)dur / 1e6, (int)O[0]);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "support";
    int tiles = argc > 2 ? atoi(argv[2]) : 4;
    int iters = argc > 3 ? atoi(argv[3]) : 3;
    int M     = argc > 4 ? atoi(argv[4]) : 384;
    if (argc > 5 && strcmp(mode, "conv") != 0 && strcmp(mode, "attn") != 0)
        G_PC = atoi(argv[5]) ? 1 : 0;
    int only = (argc > 6 && strcmp(mode, "conv") != 0) ? atoi(argv[6]) : -1;   // single-mode op filter (-1 = all)
    RunCfg cfg{M, tiles, iters};

    if (M > 384) {
        // Hard guard: M=448 -> OP_FAILED and the HAL stops answering for the session.
        P("FATAL|M_above_measured_ceiling|M=%d|ceiling=384|refusing to submit", M);
        P("DONE");
        return 4;
    }

    const char* why = "";
    // `down2` / `down4` need the chunk-major blob; every other mode uses the
    // block blob. Both are the same page-aligned size maths (checked below).
    const char* wpath = W_PATH;
    int nchunks = 0;
    int64_t wbytes = W_BYTES;
    if (strncmp(mode, "down", 4) == 0) {
        nchunks = (strcmp(mode, "downsep") == 0) ? 7 : atoi(mode + 4);
        if (nchunks != 2 && nchunks != 4 && nchunks != 7) {
            P("FATAL|bad_chunk_count|%s|want=down2|down4|downsep", mode); P("DONE"); return 2; }
        // G_PC defaults to 1 and FULLY_CONNECTED rejects per-channel scales at
        // addOperation (BAD_DATA) before it ever compiles — the same trap `attn` closed
        // below. The 0.2580 s downsep reference (RESULTS_npu_tn.md §4) is per-tensor.
        if (strcmp(mode, "downsep") == 0 && G_PC) {
            G_PC = 0;
            P("NOTE|mode=downsep|pc=0|per_tensor|per_channel_rejected_by_FULLY_CONNECTED");
        }
        if (ensure_down_blob(nchunks, &wpath, &why)) { P("FATAL|down_weights|%s", why); P("DONE"); return 2; }
        wbytes = (int64_t)DOWN_N * (int64_t)DOWN_K;
    } else if (strcmp(mode, "attn") == 0) {
        if (ensure_attn_blob(&why)) { P("FATAL|attn_weights|%s", why); P("DONE"); return 2; }
        wpath = W_ATTN_PATH;
        wbytes = ATTN_BYTES;
    } else if (strcmp(mode, "convseq") == 0 || strcmp(mode, "convwire") == 0) {
        // the VAE decoder's real INT8 artifact — no LCG fill, no second copy
        struct stat st;
        if (stat(VAE_W_PATH, &st) != 0 || st.st_size <= 0) {
            P("FATAL|vae_weights_missing|%s|push weights_int8.bin first", VAE_W_PATH);
            P("DONE"); return 2;
        }
        wpath = VAE_W_PATH;
        wbytes = (int64_t)st.st_size;
        P("WFILE|mode=convseq|path=%s|bytes=%lld", wpath, (long long)wbytes);
    } else if (ensure_weights(&why)) { P("FATAL|weights|%s", why); P("DONE"); return 2; }

    ANeuralNetworksMemory* wmem = nullptr;
    int wfd = open(wpath, O_RDWR);
    if (wfd < 0) { P("FATAL|wfile_reopen|%s", wpath); P("DONE"); return 2; }
    int rc = ANeuralNetworksMemory_createFromFd((size_t)wbytes, PROT_READ | PROT_WRITE, wfd, 0, &wmem);
    if (rc) { P("FATAL|createFromFd|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 2; }

    ANeuralNetworksDevice* tpu = find_tpu();
    if (!tpu) { P("FATAL|no_edgetpu_device"); P("DONE"); return 3; }

    // mode=convseq <max_layers> <pc>: execute the VAE decoder's real INT8
    // artifact through CONV_2D (one graph per layer/tap, one execute per frame).
    if (strcmp(mode, "convseq") == 0) {
        const int max_layers = (argc > 2) ? atoi(argv[2]) : 0;   // 0 = all 36
        const int pcmode     = (argc > 3) ? atoi(argv[3]) : -1;  // -1 = probe first
        P("CONVSEQ|max_layers=%d|pc=%d|wbytes=%lld", max_layers, pcmode, (long long)wbytes);
        return run_convseq(tpu, wmem, max_layers, pcmode);
    }

    // mode=convwire <max_convs> <pc> <dump>: the WIRED decode — the same 592 CONV_2D
    // dispatches, with the P3.11 norm/SiLU/add/shuffle program between them.
    // dump=1 writes the final activation codes + the consumed latent (P3.13).
    if (strcmp(mode, "convwire") == 0) {
        const int max_convs = (argc > 2) ? atoi(argv[2]) : 0;    // 0 = all 36
        const int pcmode    = (argc > 3) ? atoi(argv[3]) : -1;   // -1 = probe first
        const int dmp       = (argc > 4) ? atoi(argv[4]) : 0;    // 1 = write the dump
        const int dmpl      = (argc > 5) ? atoi(argv[5]) : -1;   // >=0 = dump that conv only
        P("CONVWIRE|max_convs=%d|pc=%d|dump=%d|dump_layer=%d|wbytes=%lld", max_convs, pcmode, dmp,
          dmpl, (long long)wbytes);
        return run_convwire(tpu, wmem, max_convs, pcmode, dmp, dmpl);
    }

    // mode=conv <tiles> <iters> <N> <H> <W> <Cin> <Cout> <k> — one CONV_2D op,
    // the T_V unit of measurement (VAE decode temporal-tap decomposition).
    if (strcmp(mode, "conv") == 0) {
        ConvCfg c;
        c.N    = (argc > 4) ? atoi(argv[4]) : 9;
        c.H    = (argc > 5) ? atoi(argv[5]) : 272;
        c.W    = (argc > 6) ? atoi(argv[6]) : 480;
        c.Cin  = (argc > 7) ? atoi(argv[7]) : 96;
        c.Cout = (argc > 8) ? atoi(argv[8]) : 96;
        c.k    = (argc > 9) ? atoi(argv[9]) : 3;
        P("CONVCFG|N=%d|H=%d|W=%d|Cin=%d|Cout=%d|k=%d|tiles=%d|iters=%d",
          c.N, c.H, c.W, c.Cin, c.Cout, c.k, tiles, iters);
        return run_conv(tpu, c, tiles, iters);
    }

    // mode=ksweep: bracket the K dimension the model validator accepts for a
    // single FC, and test whether ADD is accepted at all (signed vs unsigned
    // quant8). Validation only — no compile, so this cannot take the HAL down.
    if (strcmp(mode, "ksweep") == 0) {
        const char* dwhy = "";
        const int Ks[7] = {1536, 1792, 2240, 3072, 3584, 4480, 8960};
        for (int ki = 0; ki < 7; ++ki) {
            const int Kc = Ks[ki];
            if ((int64_t)DOWN_N * Kc > W_BYTES) { P("DIAG|fc_k%d|skipped_blob_too_small", Kc); continue; }
            Holder h; uint32_t in_id = 0, out_id = 0;
            int brc = ANeuralNetworksModel_create(&h.m);
            if (!brc) {
                uint32_t dA[2] = {(uint32_t)M, (uint32_t)Kc};
                ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, SA, 0};
                add_operand(h, &tA, &in_id, &dwhy);
                Gemm g = {DOWN_N, Kc, 0, "downprobe"};
                brc = add_fc(h, wmem, M, g, in_id, &out_id, &dwhy);
                if (!brc) {
                    uint32_t ins[1] = {in_id}, outs[1] = {out_id};
                    ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
                }
            }
            int frc = brc ? -brc : ANeuralNetworksModel_finish(h.m);
            P("DIAG|fc_k%d|build_rc=%d|model_finish_rc=%d(%s)", Kc, brc, frc, rcs(frc));
            ANeuralNetworksModel_free(h.m);
        }
        // ADD, signed vs unsigned quant8, [M, 1536]
        for (int uns = 0; uns < 2; ++uns) {
            Holder h; uint32_t o = 0;
            if (ANeuralNetworksModel_create(&h.m)) { P("DIAG|add_%s|create_fail", uns ? "u8" : "s8"); continue; }
            uint32_t d[2] = {(uint32_t)M, (uint32_t)DOWN_N};
            ANeuralNetworksOperandType t = {uns ? ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
                                               : ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
                                            2, d, SA, 0};
            ANeuralNetworksOperandType tAct = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
            uint32_t i1 = h.next, i2, iact;
            add_operand(h, &t, &i1, &dwhy);
            add_operand(h, &t, &i2, &dwhy);
            add_operand(h, &tAct, &iact, &dwhy);
            add_operand(h, &t, &o, &dwhy);
            ANeuralNetworksModel_setOperandValue(h.m, iact, &h.fused, sizeof(h.fused));
            uint32_t op_in[3] = {i1, i2, iact}, op_out[1] = {o};
            int arc = ANeuralNetworksModel_addOperation(h.m, ANEURALNETWORKS_ADD, 3, op_in, 1, op_out);
            uint32_t ins[2] = {i1, i2}, outs[1] = {o};
            ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 2, ins, 1, outs);
            int frc = ANeuralNetworksModel_finish(h.m);
            P("DIAG|add_%s|addOperation_rc=%d|model_finish_rc=%d(%s)", uns ? "u8" : "s8", arc, frc, rcs(frc));
            ANeuralNetworksModel_free(h.m);
        }
        P("DONE");
        return 0;
    }

    // mode=kcompile: compile ONE FC of shape [N=1536, K=only] (K from argv[6]),
    // to bracket the largest K this compiler will accept. Fresh process per K.
    if (strcmp(mode, "kcompile") == 0) {
        const char* dwhy = "";
        const int Kc = (only > 0) ? only : 2240;
        if ((int64_t)DOWN_N * Kc > W_BYTES) { P("FATAL|k_too_big_for_blob|K=%d", Kc); P("DONE"); return 2; }
        Holder h; uint32_t in_id = 0, out_id = 0;
        int brc = ANeuralNetworksModel_create(&h.m);
        if (!brc) {
            uint32_t dA[2] = {(uint32_t)M, (uint32_t)Kc};
            ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, SA, 0};
            add_operand(h, &tA, &in_id, &dwhy);
            Gemm g = {DOWN_N, Kc, 0, "kc"};
            brc = add_fc(h, wmem, M, g, in_id, &out_id, &dwhy);
            if (!brc) { uint32_t ins[1] = {in_id}, outs[1] = {out_id};
                        ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs); }
        }
        int frc = brc ? -brc : ANeuralNetworksModel_finish(h.m);
        P("MODEL_FINISH|mode=kcompile|K=%d|build_rc=%d|rc=%d(%s)", Kc, brc, frc, rcs(frc));
        if (frc) { P("DONE"); return 6; }
        ANeuralNetworksCompilation* comp = nullptr;
        int64_t c0 = now_ns();
        rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
        if (rc) { P("FATAL|comp_create|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 7; }
        ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        rc = ANeuralNetworksCompilation_finish(comp);
        P("COMPILE|mode=kcompile|K=%d|ms=%.1f|rc=%d(%s)", Kc, (now_ns() - c0) / 1e6, rc, rcs(rc));
        P("DONE");
        return 0;
    }

    // ------------------------------------------- chunked-K ffn.down (the fork)
    // mode=diag: isolate which construct the combination rejects (validation only).
    // G2: two FCs from SEPARATE inputs + ADD   G3: two FCs chained   G4: fan-out
    // (one input feeding two FCs, no ADD)   G5: build_down (fan-out + ADD).
    if (strcmp(mode, "diag") == 0) {
        if (nchunks == 0) nchunks = 2;
        const char* dwhy = "";
        const int Kc = DOWN_K / nchunks;
        auto mk_in = [&](Holder& h, uint32_t* id) {
            uint32_t dA[2] = {(uint32_t)M, (uint32_t)Kc};
            ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, SA, 0};
            return add_operand(h, &tA, id, &dwhy);
        };
        Gemm g0 = {DOWN_N, Kc, 0, "ffn.down_c0"};
        Gemm g1 = {DOWN_N, Kc, (int64_t)DOWN_N * Kc, "ffn.down_c1"};

        {   // G2: two separate inputs, one FC each, then ADD
            Holder h; uint32_t a0 = 0, a1 = 0, o0 = 0, o1 = 0, out = 0;
            int brc = ANeuralNetworksModel_create(&h.m);
            if (!brc) brc = mk_in(h, &a0);
            if (!brc) brc = mk_in(h, &a1);
            if (!brc) brc = add_fc(h, wmem, M, g0, a0, &o0, &dwhy);
            if (!brc) brc = add_fc(h, wmem, M, g1, a1, &o1, &dwhy);
            if (!brc) brc = add_add(h, M, DOWN_N, o0, o1, &out, &dwhy);
            if (!brc) { uint32_t ins[2] = {a0, a1}, outs[1] = {out};
                        ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 2, ins, 1, outs); }
            int frc = brc ? -brc : ANeuralNetworksModel_finish(h.m);
            P("DIAG|G2_two_in_fc_add|build_rc=%d|model_finish_rc=%d(%s)", brc, frc, rcs(frc));
            ANeuralNetworksModel_free(h.m);
        }
        {   // G3: two FCs chained (Kc then Kc again) — no ADD at all
            Holder h; uint32_t a0 = 0, mid = 0, out = 0;
            int brc = ANeuralNetworksModel_create(&h.m);
            if (!brc) brc = mk_in(h, &a0);
            if (!brc) brc = add_fc(h, wmem, M, g0, a0, &mid, &dwhy);
            if (!brc) brc = add_fc(h, wmem, M, g1, mid, &out, &dwhy);
            if (!brc) { uint32_t ins[1] = {a0}, outs[1] = {out};
                        ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs); }
            int frc = brc ? -brc : ANeuralNetworksModel_finish(h.m);
            P("DIAG|G3_two_fc_chain|build_rc=%d|model_finish_rc=%d(%s)", brc, frc, rcs(frc));
            ANeuralNetworksModel_free(h.m);
        }
        {   // G4: one input fanning out to two FCs, model output = the second only
            Holder h; uint32_t a0 = 0, o0 = 0, o1 = 0;
            int brc = ANeuralNetworksModel_create(&h.m);
            if (!brc) brc = mk_in(h, &a0);
            if (!brc) brc = add_fc(h, wmem, M, g0, a0, &o0, &dwhy);
            if (!brc) brc = add_fc(h, wmem, M, g1, a0, &o1, &dwhy);
            if (!brc) { uint32_t ins[1] = {a0}, outs[1] = {o1};
                        ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs); }
            int frc = brc ? -brc : ANeuralNetworksModel_finish(h.m);
            P("DIAG|G4_fanout_two_fc|build_rc=%d|model_finish_rc=%d(%s)", brc, frc, rcs(frc));
            ANeuralNetworksModel_free(h.m);
        }
        {   // G5: build_down exactly as the down2/down4 modes build it
            Holder h; uint32_t in_id = 0, out_id = 0;
            int brc = build_down(wmem, M, nchunks, h, &in_id, &out_id, &dwhy);
            int frc = brc ? -brc : ANeuralNetworksModel_finish(h.m);
            P("DIAG|G5_build_down|build_rc=%d|model_finish_rc=%d(%s)", brc, frc, rcs(frc));
            ANeuralNetworksModel_free(h.m);
        }
        P("DONE");
        return 0;
    }

    if (nchunks) {
        if (strcmp(mode, "downsep") == 0)
            return run_downsep(tpu, wmem, M, nchunks, tiles, iters);
        Holder h; uint32_t in_id = 0, out_id = 0; why = "";
        if (build_down(wmem, M, nchunks, h, &in_id, &out_id, &why)) {
            P("FATAL|build_down|chunks=%d|why=%s", nchunks, why); P("DONE"); return 5; }
        rc = ANeuralNetworksModel_finish(h.m);
        if (rc) { P("FATAL|model_finish_down|chunks=%d|rc=%d(%s)", nchunks, rc, rcs(rc));
                  P("DONE"); return 6; }
        P("MODEL_FINISH|mode=%s|nops=%d|M=%d|rc=%d(%s)", mode, 2 * nchunks - 1, M, rc, rcs(rc));
        bool sup[16] = {false};
        int src = ANeuralNetworksModel_getSupportedOperationsForDevices(h.m, &tpu, 1, sup);
        P("SUPPORT|mode=%s|nops=%d|rc=%d(%s)", mode, 2 * nchunks - 1, src, rcs(src));
        if (src == 0) {
            for (int i = 0; i < 2 * nchunks - 1; ++i)
                P("SUP_OP|pc=0|%s|%d", (i < nchunks) ? "FC" : "ADD", sup[i] ? 1 : 0);
        }

        ANeuralNetworksCompilation* comp = nullptr;
        int64_t c0 = now_ns();
        rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
        if (rc) { P("FATAL|comp_create_down|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 7; }
        ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        rc = ANeuralNetworksCompilation_finish(comp);
        double compile_ms = (now_ns() - c0) / 1e6;
        P("COMPILE|mode=%s|Kc=%d|ms=%.1f|rc=%d(%s)", mode, DOWN_K / nchunks, compile_ms, rc, rcs(rc));
        if (rc) { P("FATAL|comp_finish_down|chunks=%d|ms=%.1f|rc=%d(%s)",
                    nchunks, compile_ms, rc, rcs(rc)); P("DONE"); return 7; }
        if (iters <= 0) {
            // compile-only probe: answers "does this shape compile at all"
            // without heating the device through a timed run.
            P("SUMMARY|mode=%s|M=%d|compile_ms=%.1f|probe_only=1", mode, M, compile_ms);
            ANeuralNetworksCompilation_free(comp); ANeuralNetworksModel_free(h.m);
            P("DONE"); return 0;
        }

        std::vector<double> tm, hw;
        if (time_tiles(comp, cfg, out_id, DOWN_N, DOWN_K / nchunks, &tm, &hw, mode, &why)) {
            P("DONE"); return 8; }
        ANeuralNetworksCompilation_free(comp);
        ANeuralNetworksModel_free(h.m);

        double sum = 0;
        for (double v : tm) sum += v;
        double per_tile = sum / (double)tm.size();
        double block_s = per_tile * tiles / 1000.0;
        double gmacs = (double)tiles * (double)M * (double)DOWN_N * (double)DOWN_K;
        P("OP|ffn.down|chunked=%d|N=%d|K=%d|compile_ms=%.1f|ms_per_tile=%.3f|per_block_s=%.4f|gops=%.1f",
          nchunks, DOWN_N, DOWN_K, compile_ms, per_tile, block_s, 2.0 * gmacs / block_s / 1e9);
        P("SUMMARY|mode=%s|M=%d|tiles=%d|iters=%d|compile_ms=%.1f|ms_per_tile=%.3f|"
          "per_block_s=%.4f|gops=%.1f|samples=%zu",
          mode, M, tiles, iters, compile_ms, per_tile, block_s,
          2.0 * gmacs / block_s / 1e9, tm.size());
        P("DONE");
        return 0;
    }

    // mode=attn: the VAE mid-block attention GEMM — the last unmeasured term of
    // `T_V`. 6120 tokens x 384 ch decomposes into 16 token tiles at M=384:
    //   * score pass  [M, 6120, 384] — this is QK^T and, by symmetry, AV.
    //   * proj  pass  [M, 384, 384]  — q, k, v, o (four of them).
    // Per-tensor scales only: FULLY_CONNECTED rejects per-channel before compile
    // (measured, RESULTS_npu_tn.md finding #2), so the NPU rate exists only for
    // that scheme. `tiles` = 16 reproduces one full pass over all 6120 tokens.
    if (strcmp(mode, "attn") == 0) {
        G_PC = 0;
        P("NOTE|mode=attn|pc=0|per_tensor|M=%d|tiles=%d|iters=%d|score=%dx%d|proj=%dx%d",
          M, tiles, iters, ATTN_SCORE_N, ATTN_SCORE_K, ATTN_PROJ_N, ATTN_PROJ_K);
        Gemm ga[2] = {
            {ATTN_SCORE_N, ATTN_SCORE_K, ATTN_SCORE_OFF, "attn.score"},
            {ATTN_PROJ_N,  ATTN_PROJ_K,  ATTN_PROJ_OFF,  "attn.proj"},
        };
        for (int gi = 0; gi < 2; ++gi) {
            Holder h; uint32_t in_id = 0, out_id = 0; why = "";
            if (build_model(wmem, M, &ga[gi], 1, h, &in_id, &out_id, &why)) {
                P("FATAL|build_attn|%s|why=%s", ga[gi].name, why); P("DONE"); return 5; }
            rc = ANeuralNetworksModel_finish(h.m);
            if (rc) { P("FATAL|model_finish_attn|%s|rc=%d(%s)", ga[gi].name, rc, rcs(rc));
                      ANeuralNetworksModel_free(h.m); P("DONE"); return 6; }
            ANeuralNetworksCompilation* comp = nullptr;
            int64_t c0 = now_ns();
            rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
            if (rc) { P("FATAL|comp_create_attn|%s|rc=%d(%s)", ga[gi].name, rc, rcs(rc));
                      ANeuralNetworksModel_free(h.m); P("DONE"); return 7; }
            ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
            rc = ANeuralNetworksCompilation_finish(comp);
            double cms = (now_ns() - c0) / 1e6;
            if (rc) { P("FATAL|comp_finish_attn|%s|ms=%.1f|rc=%d(%s)", ga[gi].name, cms, rc, rcs(rc));
                      P("DONE"); return 7; }
            std::vector<double> atm, ahw;
            if (time_tiles(comp, cfg, out_id, ga[gi].N, ga[gi].K, &atm, &ahw,
                           ga[gi].name, &why)) { P("DONE"); return 8; }
            double s = 0; for (double v : atm) s += v;
            double mean = atm.empty() ? -1.0 : s / (double)atm.size();
            double pass_s = mean * tiles / 1000.0;
            double gmacs = (double)tiles * (double)M * (double)ga[gi].N * (double)ga[gi].K;
            P("OP|%s|N=%d|K=%d|compile_ms=%.1f|ms_per_tile=%.3f|per_pass_s=%.4f|gops=%.1f",
              ga[gi].name, ga[gi].N, ga[gi].K, cms, mean, pass_s,
              pass_s > 0 ? 2.0 * gmacs / pass_s / 1e9 : -1.0);
            ANeuralNetworksCompilation_free(comp);
            ANeuralNetworksModel_free(h.m);
        }
        P("DONE");
        return 0;
    }

    // ---------------------------------------------------------- support probe
    {
        int nops = (strcmp(mode, "single") == 0) ? 1 : 10;
        int pc_lo = (strcmp(mode, "support") == 0) ? 0 : G_PC;
        int pc_hi = (strcmp(mode, "support") == 0) ? 1 : G_PC;
        for (int pc = pc_lo; pc <= pc_hi; ++pc) {
            G_PC = pc;
            Holder h; uint32_t in_id = 0, out_id = 0; why = "";
            if (build_model(wmem, M, GEMMS, nops, h, &in_id, &out_id, &why)) {
                P("BUILD_FAIL|pc=%d|nops=%d|why=%s", pc, nops, why);
                continue;
            }
            // NNAPI requires a FINISHED model for getSupportedOperationsForDevices
            rc = ANeuralNetworksModel_finish(h.m);
            P("MODEL_FINISH|pc=%d|nops=%d|M=%d|rc=%d(%s)", pc, nops, M, rc, rcs(rc));
            if (rc == 0) {
                bool sup[10] = {false};
                int src = ANeuralNetworksModel_getSupportedOperationsForDevices(h.m, &tpu, 1, sup);
                P("SUPPORT|pc=%d|nops=%d|rc=%d(%s)", pc, nops, src, rcs(src));
                if (src == 0)
                    for (int i = 0; i < nops; ++i)
                        P("SUP_OP|pc=%d|%s|%d", pc, GEMMS[i].name, sup[i] ? 1 : 0);
            }
            ANeuralNetworksModel_free(h.m);
        }
        if (strcmp(mode, "support") == 0) { P("DONE"); return 0; }
    }

    if (strcmp(mode, "chain") == 0) {
        // one model, 10 chained FCs, `tiles` executions
        Holder h; uint32_t in_id = 0, out_id = 0; why = "";
        if (build_model(wmem, M, GEMMS, 10, h, &in_id, &out_id, &why)) { P("DONE"); return 5; }
        rc = ANeuralNetworksModel_finish(h.m);
        if (rc) { P("FATAL|model_finish_chain|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 6; }

        ANeuralNetworksCompilation* comp = nullptr;
        int64_t c0 = now_ns();
        rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
        if (rc) { P("FATAL|comp_create|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 7; }
        ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        rc = ANeuralNetworksCompilation_finish(comp);
        double compile_ms = (now_ns() - c0) / 1e6;
        P("COMPILE|mode=chain|ms=%.1f|rc=%d(%s)", compile_ms, rc, rcs(rc));
        if (rc) { P("FATAL|comp_finish_chain|had_compile_ms=%.1f", compile_ms); P("DONE"); return 7; }

        std::vector<double> tm, hw;
        if (time_tiles(comp, cfg, out_id, 1536, 1536, &tm, &hw, "chain", &why)) { P("DONE"); return 8; }
        ANeuralNetworksCompilation_free(comp);
        ANeuralNetworksModel_free(h.m);

        // per-block rate: `tiles` executions of a 10-GEMM chain
        double sum = 0;
        for (double v : tm) sum += v;
        double per_tile = sum / (double)tm.size();   // mean of (iters*tiles) samples
        double block_s = per_tile * tiles / 1000.0;
        double macs = (double)tiles * (double)M * (double)BLOCK_MACS_PER_TOKEN;
        P("SUMMARY|mode=chain|M=%d|tiles=%d|iters=%d|compile_ms=%.1f|ms_per_tile=%.3f|"
          "per_block_s=%.4f|gops=%.1f|samples=%zu",
          M, tiles, iters, compile_ms, per_tile, block_s, 2.0 * macs / block_s / 1e9, tm.size());
        P("DONE");
        return 0;
    }

    if (strcmp(mode, "corun") == 0) {
        // Sustained NPU load for a co-run measurement: compile the 9 compilable
        // block GEMMs ONCE, hold every compilation, then loop `iters` rounds of
        // `tiles` executes per GEMM round-robin. ffn.down is excluded because it
        // has no compilable graph at all (K=8960 > the 1664 compiler ceiling).
        // Rationale: in `single` mode compiles are interleaved with the timed
        // executes, so a concurrent GPU process would contend with the CPU-side
        // compiler as well as the TPU — that is not a stage rate.
        struct Slot {
            ANeuralNetworksCompilation* comp = nullptr;
            ANeuralNetworksModel* model = nullptr;
            uint32_t out_id = 0;
            int n = 0, k = 0;
            const char* name = "";
            std::vector<double> ms;
        };
        std::vector<Slot> slots;
        double compile_total = 0;
        for (int g = 0; g < 10; ++g) {
            if (g == 9) continue;               // ffn.down: no compilable graph
            Holder h; uint32_t in_id = 0, out_id = 0; why = "";
            int brc = build_model(wmem, M, &GEMMS[g], 1, h, &in_id, &out_id, &why);
            if (brc) { P("FATAL|corun_build|%s|why=%s", GEMMS[g].name, why); P("DONE"); return 5; }
            rc = ANeuralNetworksModel_finish(h.m);
            if (rc) { P("FATAL|corun_finish|%s|rc=%d(%s)", GEMMS[g].name, rc, rcs(rc));
                      ANeuralNetworksModel_free(h.m); P("DONE"); return 6; }
            ANeuralNetworksCompilation* comp = nullptr;
            int64_t c0 = now_ns();
            rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
            if (rc) { P("FATAL|corun_comp_create|%s|rc=%d(%s)", GEMMS[g].name, rc, rcs(rc));
                      ANeuralNetworksModel_free(h.m); P("DONE"); return 7; }
            ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
            rc = ANeuralNetworksCompilation_finish(comp);
            double cms = (now_ns() - c0) / 1e6;
            compile_total += cms;
            // NOTE: the model is NOT freed here. setInput/setOutput resolve operand
            // descriptors through it on this HAL (freeing it -> BAD_DATA at attn1.to_q),
            // so every model is held for the whole loop.
            if (rc) { P("FATAL|corun_comp_finish|%s|ms=%.1f|rc=%d(%s)", GEMMS[g].name, cms, rc, rcs(rc));
                      P("DONE"); return 7; }
            P("CORUN_COMPILE|%s|N=%d|K=%d|ms=%.1f", GEMMS[g].name, GEMMS[g].N, GEMMS[g].K, cms);
            Slot s; s.comp = comp; s.model = h.m; s.out_id = out_id;
            s.n = GEMMS[g].N; s.k = GEMMS[g].K;
            s.name = GEMMS[g].name;
            slots.push_back(s);
        }
        // Everything is compiled: tell the driver the TPU is about to go to work,
        // then spin. All 9 compilations stay live for the whole loop.
        P("CORUN_READY|n=%zu|compile_ms=%.1f|M=%d|tiles=%d|rounds=%d",
          slots.size(), compile_total, M, tiles, iters);

        std::vector<int8_t> A((size_t)M * 1536, 1);
        std::vector<int8_t> O((size_t)M * 8960, 0);
        for (int r = 0; r < iters; ++r) {
            int64_t r0 = now_ns();
            for (auto& s : slots) {
                for (int t = 0; t < tiles; ++t) {
                    ANeuralNetworksExecution* ex = nullptr;
                    int erc = ANeuralNetworksExecution_create(s.comp, &ex);
                    if (erc) { P("FATAL|corun_exec|%s|rc=%d(%s)", s.name, erc, rcs(erc)); P("DONE"); return 8; }
                    erc = ANeuralNetworksExecution_setInput(ex, 0, nullptr, A.data(), (size_t)M * (size_t)s.k);
                    if (!erc) erc = ANeuralNetworksExecution_setOutput(ex, 0, nullptr, O.data(), (size_t)M * (size_t)s.n);
                    if (erc) { P("FATAL|corun_io|%s|rc=%d(%s)", s.name, erc, rcs(erc));
                               ANeuralNetworksExecution_free(ex); P("DONE"); return 8; }
                    int64_t t0 = now_ns();
                    erc = ANeuralNetworksExecution_compute(ex);
                    int64_t t1 = now_ns();
                    ANeuralNetworksExecution_free(ex);
                    if (erc) { P("FATAL|corun_compute|%s|round=%d|rc=%d(%s)", s.name, r, erc, rcs(erc));
                               P("DONE"); return 8; }
                    s.ms.push_back((t1 - t0) / 1e6);
                    P("TS|npu|%s|round=%d|tile=%d|t0=%.3f|t1=%.3f|ms=%.3f",
                      s.name, r, t, (double)t0 / 1e6, (double)t1 / 1e6, (t1 - t0) / 1e6);
                }
            }
            int64_t r1 = now_ns();
            P("CORUN_ROUND|r=%d|t0=%.3f|t1=%.3f|ms=%.3f", r, (double)r0 / 1e6, (double)r1 / 1e6,
              (r1 - r0) / 1e6);
        }
        double round_sum = 0; int round_n = 0;
        for (auto& s : slots) {
            double sum = 0; for (double v : s.ms) sum += v;
            double mean = s.ms.empty() ? -1 : sum / (double)s.ms.size();
            double block_s = mean * tiles / 1000.0;
            double gmacs = (double)tiles * (double)M * (double)s.n * (double)s.k;
            P("OP|%s|N=%d|K=%d|compile_ms=0.0|ms_per_tile=%.3f|per_block_s=%.4f|gops=%.1f",
              s.name, s.n, s.k, mean, block_s, block_s > 0 ? 2.0 * gmacs / block_s / 1e9 : -1.0);
            ANeuralNetworksCompilation_free(s.comp);
            ANeuralNetworksModel_free(s.model);
        }
        (void)round_sum; (void)round_n;
        // Per-block total for the 9-GEMM subset actually loaded on the TPU
        double m9 = 0;
        for (auto& s : slots) {
            double sum = 0; for (double v : s.ms) sum += v;
            double mean = s.ms.empty() ? 0 : sum / (double)s.ms.size();
            m9 += mean * tiles / 1000.0;
        }
        const double macs9 = (double)tiles * (double)M * (double)(BLOCK_MACS_PER_TOKEN - (int64_t)DOWN_N * DOWN_K);
        P("SUMMARY|mode=corun|M=%d|tiles=%d|iters=%d|compile_ms=%.1f|ms_per_tile=%.3f|"
          "per_block_s=%.4f|gops=%.1f",
          M, tiles, iters, compile_total, m9 / tiles * 1000.0, m9,
          m9 > 0 ? 2.0 * macs9 / m9 / 1e9 : -1.0);
        P("DONE");
        return 0;
    }

    if (strcmp(mode, "single") == 0) {
        // fallback shape: one compilation per GEMM -> 10 * tiles dispatches per block
        std::vector<double> tm, hw;
        double compile_total = 0, block_s = 0;
        for (int g = 0; g < 10; ++g) {
            if (only >= 0 && g != only) continue;
            Holder h; uint32_t in_id = 0, out_id = 0; why = "";
            int brc = build_model(wmem, M, &GEMMS[g], 1, h, &in_id, &out_id, &why);
            if (brc) { P("FATAL|build_single|g=%d", g); P("DONE"); return 5; }
            rc = ANeuralNetworksModel_finish(h.m);
            if (rc) { P("FATAL|model_finish_single|%s|rc=%d(%s)", GEMMS[g].name, rc, rcs(rc));
                      P("DONE"); return 6; }
            ANeuralNetworksCompilation* comp = nullptr;
            int64_t c0 = now_ns();
            rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
            if (rc) { P("FATAL|comp_create|%s|rc=%d(%s)", GEMMS[g].name, rc, rcs(rc));
                      P("DONE"); return 7; }
            ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
            rc = ANeuralNetworksCompilation_finish(comp);
            double cms = (now_ns() - c0) / 1e6;
            compile_total += cms;
            if (rc) { P("FATAL|comp_finish|%s|ms=%.1f|rc=%d(%s)", GEMMS[g].name, cms, rc, rcs(rc));
                      P("DONE"); return 7; }
            std::vector<double> gtm, ghw;
            if (time_tiles(comp, cfg, out_id, GEMMS[g].N, GEMMS[g].K, &gtm, &ghw,
                           GEMMS[g].name, &why)) {
                P("DONE"); return 8; }
            double gs = 0; for (double v : gtm) gs += v;
            double gm = gs / (double)gtm.size();    // mean per token tile
            double gmacs = (double)tiles * (double)M * (double)GEMMS[g].N * (double)GEMMS[g].K;
            P("OP|%s|N=%d|K=%d|compile_ms=%.1f|ms_per_tile=%.3f|per_block_s=%.4f|gops=%.1f",
              GEMMS[g].name, GEMMS[g].N, GEMMS[g].K, cms, gm, gm * tiles / 1000.0,
              2.0 * gmacs / (gm * tiles / 1000.0) / 1e9);
            block_s += gm * tiles / 1000.0;
            ANeuralNetworksCompilation_free(comp);
            ANeuralNetworksModel_free(h.m);
        }
        double macs = (double)tiles * (double)M * (double)BLOCK_MACS_PER_TOKEN;
        P("SUMMARY|mode=single|M=%d|tiles=%d|iters=%d|compile_ms=%.1f|ms_per_tile=%.3f|"
          "per_block_s=%.4f|gops=%.1f",
          M, tiles, iters, compile_total, block_s / tiles * 1000.0, block_s,
          2.0 * macs / block_s / 1e9);
        P("DONE");
        return 0;
    }

    P("FATAL|unknown_mode|%s", mode);
    P("DONE");
    return 2;
}
