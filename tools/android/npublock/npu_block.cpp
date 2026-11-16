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
#include <string>
#include <vector>

static const char* W_PATH = "/data/local/tmp/hv_block_int8.bin";
// Chunk-major copy of the same 13,762,560 B of `ffn.down` weights, laid out as
// `nchunks` contiguous [1536, 8960/nchunks] matrices so each chunk is a legal
// constant operand. Synthetic LCG fill, same as W_PATH.
static const char* W_DOWN_PATH = "/data/local/tmp/hv_down_chunks.bin";
static const int DOWN_N = 1536;
static const int DOWN_K = 8960;

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
    if (argc > 5 && strcmp(mode, "conv") != 0) G_PC = atoi(argv[5]) ? 1 : 0;
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
        if (ensure_down_blob(nchunks, &wpath, &why)) { P("FATAL|down_weights|%s", why); P("DONE"); return 2; }
        wbytes = (int64_t)DOWN_N * (int64_t)DOWN_K;
    } else if (ensure_weights(&why)) { P("FATAL|weights|%s", why); P("DONE"); return 2; }

    ANeuralNetworksMemory* wmem = nullptr;
    int wfd = open(wpath, O_RDWR);
    if (wfd < 0) { P("FATAL|wfile_reopen|%s", wpath); P("DONE"); return 2; }
    int rc = ANeuralNetworksMemory_createFromFd((size_t)wbytes, PROT_READ | PROT_WRITE, wfd, 0, &wmem);
    if (rc) { P("FATAL|createFromFd|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 2; }

    ANeuralNetworksDevice* tpu = find_tpu();
    if (!tpu) { P("FATAL|no_edgetpu_device"); P("DONE"); return 3; }

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
