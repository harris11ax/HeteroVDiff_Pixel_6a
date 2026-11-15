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
    if (argc > 5) G_PC = atoi(argv[5]) ? 1 : 0;
    int only = argc > 6 ? atoi(argv[6]) : -1;   // single-mode op filter (-1 = all)
    RunCfg cfg{M, tiles, iters};

    if (M > 384) {
        // Hard guard: M=448 -> OP_FAILED and the HAL stops answering for the session.
        P("FATAL|M_above_measured_ceiling|M=%d|ceiling=384|refusing to submit", M);
        P("DONE");
        return 4;
    }

    const char* why = "";
    if (ensure_weights(&why)) { P("FATAL|weights|%s", why); P("DONE"); return 2; }

    ANeuralNetworksMemory* wmem = nullptr;
    int wfd = open(W_PATH, O_RDWR);
    if (wfd < 0) { P("FATAL|wfile_reopen"); P("DONE"); return 2; }
    int rc = ANeuralNetworksMemory_createFromFd((size_t)W_BYTES, PROT_READ | PROT_WRITE, wfd, 0, &wmem);
    if (rc) { P("FATAL|createFromFd|rc=%d(%s)", rc, rcs(rc)); P("DONE"); return 2; }

    ANeuralNetworksDevice* tpu = find_tpu();
    if (!tpu) { P("FATAL|no_edgetpu_device"); P("DONE"); return 3; }

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
