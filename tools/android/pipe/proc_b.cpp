// tools/android/pipe/proc_b.cpp — P3.3: the NPU stage over the real INT8 artifact.
//
// The Edge-TPU half of the two-process pipeline. It is the consumer side of
// phase3_pipeline/handoff.h: it waits on `seq`, runs the `(180 - k)` blocks the cut
// assigns to the NPU as held-compilation FULLY_CONNECTED graphs (the `corun` code
// path from npu_block), then the decode stub, then publishes `done`.
//
// One INT8 copy of the base weights, by construction:
//
//   open(block_00.bin, O_RDWR)                      one fd, both processes
//   ANeuralNetworksMemory_createFromFd(fd, off=0)    one view of the whole artifact
//   setOperandValueFromMemory(model, W, wmem, off, N*K)
//
// ...so every graph's constant operand is a window into the same page cache proc_a
// imports with clImportMemoryARM. Nothing here copies, unpacks or re-lays out a base
// weight *except* the priced `--down stage` branch, which is reported as a one-copy
// deviation instead of being adopted.
//
// Per-tensor weight scales, not per-output-column: FULLY_CONNECTED rejects
// TENSOR_QUANT8_SYMM_PER_CHANNEL at addOperation on this HAL (measured 2026-11-15,
// RESULTS_npu_tn.md §2). The accuracy cost of that substitution is measured in
// docs/device/RESULTS_scales.md (2.15x mean GEMM error). This binary therefore
// measures LATENCY on the real weight bytes; it is not a parity claim.
//
//   proc_b --block /data/local/tmp/hv_blk.bin --ring /data/local/tmp/hv_handoff.bin \
//          --blocks 3 --nblocks 1 --M 1530 --down skip|stage --source fork|none
//
//   --down skip    ffn.down stays on Mali (default; the measured conclusion). The
//                  47.33 ms/block is cited, not re-derived here.
//   --down stage   price the TPU path: the row-major [1536,8960] matrix cannot be
//                  sliced along K into a legal constant operand (stride 8960), so
//                  the 7 K=1280 slices are gathered into a staging buffer, then
//                  7x FC + 6x ADD. Also gathers each tile's activation slice
//                  (that part is inherent to the decomposition and is printed
//                  separately). While the models live this duplicates
//                  13,762,560 B of down weights -> a one-copy deviation, WARN'd.
//   --source fork  fork a producer that fills the ring with synthetic residual
//                  chunks, so this stage is testable on its own (the mirror of
//                  P3.2's `--drain fork`). `none` expects a real proc_a instead.
//
// Exit: 0 = PASS, 1 = protocol/compute failure, 2 = setup failure.

#include <android/neuralnetworks.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "handoff.h"

// --------------------------------------------------------------- block layout
// Same order and offsets as proc_a's table (manifest.json:gemm_order): the down
// region starts at 32,636,928 = 8 * 1536^2 + 8960 * 1536.
struct Gemm { int N, K; int64_t off; const char* name; };

static const int KD = 1536;
static const int NUP = 8960;
static const int64_t W_BYTES = 46'399'488;
static const int64_t DOWN_OFF = 32'636'928;
static const int TILE = 384;             // measured per-graph token ceiling (Gate 0)
static const float SA = 1.0f / 127.0f;
static const float SW = 1.0f / 127.0f;   // per-tensor; per-channel is rejected

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

static void P(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}

static const char* rcs(int rc) {
    switch (rc) {
        case 0: return "NO_ERROR";
        case 1: return "BAD_DATA";
        case 2: return "BAD_STATE";
        case 3: return "OUT_OF_MEMORY";
        case 4: return "BAD_DATA(4)";
        case 5: return "OP_FAILED";
        case 6: return "UNMAPPABLE";
        case 10: return "DEAD_OBJECT";
        default: return "?";
    }
}

// ------------------------------------------------------------------- fp16
static inline float h2f(std::uint16_t h) {
    const unsigned sign = (h & 0x8000u) ? 0x80000000u : 0u;
    unsigned exp = (h >> 10) & 0x1Fu;
    unsigned mant = h & 0x3FFu;
    unsigned bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {                                  // subnormal
            exp = 127 - 15 + 2;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static inline std::uint16_t f2h(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    const unsigned sign = (u >> 16) & 0x8000u;
    int exp = (int)((u >> 23) & 0xFF) - 127 + 15;
    unsigned mant = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF) return (std::uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp >= 0x1F) return (std::uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (std::uint16_t)sign;
        mant |= 0x800000u;
        const int shift = 14 - exp;
        unsigned h = mant >> shift;
        const unsigned rem = mant & ((1u << shift) - 1);
        const unsigned half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (std::uint16_t)(sign | h);
    }
    unsigned h = ((unsigned)exp << 10) | (mant >> 13);
    const unsigned rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (std::uint16_t)(sign | h);
}

static const std::uint64_t FNV0 = 1469598103934665603ull;
static inline std::uint64_t fnv(std::uint64_t h, const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// Gate 3 evidence: raw int8 bytes to disk, so the host can recompute the same
// product with exact integer arithmetic instead of trusting a hash.
static bool write_blob(const char* path, const void* p, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    const size_t w = fwrite(p, 1, n, f);
    fclose(f);
    return w == n;
}

// ------------------------------------------------------------------- builder
struct Holder {
    ANeuralNetworksModel* m = nullptr;
    uint32_t next = 0;
    std::vector<std::vector<int32_t>> bi;
    int32_t fused = ANEURALNETWORKS_FUSED_NONE;
};

static int add_operand(Holder& h, const ANeuralNetworksOperandType* t, uint32_t* id, const char** why) {
    uint32_t idx = h.next;
    int rc = ANeuralNetworksModel_addOperand(h.m, t);
    if (rc) { P("FATAL|addOperand|idx=%u|rc=%d(%s)", idx, rc, rcs(rc)); *why = "addOperand"; return -1; }
    h.next = idx + 1;
    *id = idx;
    return 0;
}

// Weight source: a window into the shared artifact (route B), or — for the priced
// `--down stage` branch only — a host pointer.
struct WSrc {
    ANeuralNetworksMemory* mem = nullptr;
    int64_t off = 0;
    const int8_t* ptr = nullptr;
};

// One FULLY_CONNECTED. FC takes exactly 4 inputs and a MANDATORY bias on this HAL.
static int add_fc(Holder& h, int M, int N, int K, const WSrc& w, uint32_t in, uint32_t* out,
                  const char** why) {
    h.bi.emplace_back((size_t)N, 0);
    const std::vector<int32_t>& bi = h.bi.back();

    uint32_t dW[2] = {(uint32_t)N, (uint32_t)K};
    uint32_t dB[1] = {(uint32_t)N};
    uint32_t dO[2] = {(uint32_t)M, (uint32_t)N};

    ANeuralNetworksOperandType tW = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dW, SW, 0};
    ANeuralNetworksOperandType tB = {ANEURALNETWORKS_TENSOR_INT32, 1, dB, SA * SW, 0};
    ANeuralNetworksOperandType tAct = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
    ANeuralNetworksOperandType tO = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dO, SA, 0};

    uint32_t iw, ib, iact, io;
    if (add_operand(h, &tW, &iw, why)) return -1;
    if (add_operand(h, &tB, &ib, why)) return -1;
    if (add_operand(h, &tAct, &iact, why)) return -1;
    if (add_operand(h, &tO, &io, why)) return -1;

    const size_t wlen = (size_t)N * (size_t)K;
    int rc;
    if (w.ptr) {
        rc = ANeuralNetworksModel_setOperandValue(h.m, iw, w.ptr, wlen);
        if (rc) { P("FATAL|setWeightsPtr|N=%d|K=%d|rc=%d(%s)", N, K, rc, rcs(rc));
                  *why = "w_ptr"; return -1; }
    } else {
        rc = ANeuralNetworksModel_setOperandValueFromMemory(h.m, iw, w.mem, (size_t)w.off, wlen);
        if (rc) { P("FATAL|setOperandValueFromMemory|N=%d|K=%d|off=%" PRId64 "|rc=%d(%s)",
                    N, K, w.off, rc, rcs(rc));
                  *why = "w_from_memory"; return -1; }
    }
    rc = ANeuralNetworksModel_setOperandValue(h.m, ib, bi.data(), bi.size() * sizeof(int32_t));
    if (rc) { P("FATAL|setBias|rc=%d(%s)", rc, rcs(rc)); *why = "bias"; return -1; }
    rc = ANeuralNetworksModel_setOperandValue(h.m, iact, &h.fused, sizeof(h.fused));
    if (rc) { P("FATAL|setAct|rc=%d(%s)", rc, rcs(rc)); *why = "activation"; return -1; }

    uint32_t op_in[4] = {in, iw, ib, iact};
    uint32_t op_out[1] = {io};
    rc = ANeuralNetworksModel_addOperation(h.m, ANEURALNETWORKS_FULLY_CONNECTED, 4, op_in, 1, op_out);
    if (rc) { P("FATAL|addOperation|N=%d|K=%d|rc=%d(%s)", N, K, rc, rcs(rc));
              *why = "addOperation"; return -1; }
    *out = io;
    return 0;
}

// ADD for the chunked-K down accumulation. Both inputs must be EXISTING operand
// ids: an operand no op writes and no value backs fails model_finish BAD_DATA.
static int add_add(Holder& h, int M, int N, uint32_t a, uint32_t b, uint32_t* out, const char** why) {
    uint32_t d[2] = {(uint32_t)M, (uint32_t)N};
    ANeuralNetworksOperandType t = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, d, SA, 0};
    ANeuralNetworksOperandType tAct = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
    uint32_t iact, io;
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

// One compiled single-op graph, held live for the whole stage.
struct Slot {
    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksCompilation* comp = nullptr;
    uint32_t out_id = 0;
    int n = 0, k = 0;
    const char* name = "";
    double compile_ms = 0;
};

static int compile_fc(ANeuralNetworksDevice* tpu, int M, int N, int K, const WSrc& w,
                      const char* name, int idx, Slot* out, int* last_rc) {
    const char* why = "";
    Holder h;
    int rc = ANeuralNetworksModel_create(&h.m);
    if (rc) { P("FATAL|model_create|%s|rc=%d(%s)", name, rc, rcs(rc)); *last_rc = rc; return 5; }
    uint32_t dA[2] = {(uint32_t)M, (uint32_t)K};
    ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, SA, 0};
    uint32_t in_id = 0, out_id = 0;
    if (add_operand(h, &tA, &in_id, &why) || add_fc(h, M, N, K, w, in_id, &out_id, &why)) {
        ANeuralNetworksModel_free(h.m);
        *last_rc = -1;
        return 5;
    }
    uint32_t ins[1] = {in_id}, outs[1] = {out_id};
    rc = ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 1, ins, 1, outs);
    if (rc) { P("FATAL|identifyIO|%s|rc=%d(%s)", name, rc, rcs(rc));
              ANeuralNetworksModel_free(h.m); *last_rc = rc; return 5; }
    rc = ANeuralNetworksModel_finish(h.m);
    if (rc) { P("FATAL|model_finish|%s|rc=%d(%s)", name, rc, rcs(rc));
              ANeuralNetworksModel_free(h.m); *last_rc = rc; return 6; }

    ANeuralNetworksCompilation* comp = nullptr;
    const std::uint64_t c0 = hv::now_ns();
    rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
    if (rc) { P("FATAL|comp_create|%s|rc=%d(%s)", name, rc, rcs(rc));
              ANeuralNetworksModel_free(h.m); *last_rc = rc; return 7; }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    rc = ANeuralNetworksCompilation_finish(comp);
    const double cms = (hv::now_ns() - c0) / 1e6;
    P("COMPILE|i=%d|%s|N=%d|K=%d|ms=%.1f|rc=%d(%s)", idx, name, N, K, cms, rc, rcs(rc));
    if (rc) { ANeuralNetworksCompilation_free(comp); ANeuralNetworksModel_free(h.m);
              *last_rc = rc; return 7; }
    out->model = h.m;      // NOT freed: setInput/setOutput resolve operand
    out->comp = comp;      // descriptors through the model on this HAL.
    out->out_id = out_id;
    out->n = N; out->k = K;
    out->name = name;
    out->compile_ms = cms;
    return 0;
}

static int compile_add(ANeuralNetworksDevice* tpu, int M, int N, Slot* out, int* last_rc) {
    const char* why = "";
    Holder h;
    int rc = ANeuralNetworksModel_create(&h.m);
    if (rc) { *last_rc = rc; return 5; }
    uint32_t d[2] = {(uint32_t)M, (uint32_t)N};
    ANeuralNetworksOperandType t = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, d, SA, 0};
    uint32_t a0 = 0, a1 = 0, ao = 0;
    if (add_operand(h, &t, &a0, &why) || add_operand(h, &t, &a1, &why) ||
        add_add(h, M, N, a0, a1, &ao, &why)) {
        ANeuralNetworksModel_free(h.m); *last_rc = -1; return 5;
    }
    uint32_t ins[2] = {a0, a1}, outs[1] = {ao};
    ANeuralNetworksModel_identifyInputsAndOutputs(h.m, 2, ins, 1, outs);
    rc = ANeuralNetworksModel_finish(h.m);
    if (rc) { P("FATAL|add_finish|rc=%d(%s)", rc, rcs(rc));
              ANeuralNetworksModel_free(h.m); *last_rc = rc; return 6; }
    ANeuralNetworksCompilation* comp = nullptr;
    const std::uint64_t c0 = hv::now_ns();
    rc = ANeuralNetworksCompilation_createForDevices(h.m, &tpu, 1, &comp);
    if (rc) { P("FATAL|add_comp_create|rc=%d(%s)", rc, rcs(rc));
              ANeuralNetworksModel_free(h.m); *last_rc = rc; return 7; }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    rc = ANeuralNetworksCompilation_finish(comp);
    const double cms = (hv::now_ns() - c0) / 1e6;
    P("COMPILE|ADD|M=%d|N=%d|ms=%.1f|rc=%d(%s)", M, N, cms, rc, rcs(rc));
    if (rc) { ANeuralNetworksCompilation_free(comp); ANeuralNetworksModel_free(h.m);
              *last_rc = rc; return 7; }
    out->model = h.m; out->comp = comp; out->out_id = ao; out->n = N; out->k = 0;
    out->name = "ADD"; out->compile_ms = cms;
    return 0;
}

// One execute of a held compilation. A reused Execution returns BAD_STATE on this
// HAL, so every compute gets a fresh one; wall time around the synchronous
// compute() is the measurement (getDuration reports ~150 TOPS here: unusable).
static int exec_fc(Slot& s, int M, const int8_t* in, int8_t* out, double* ms) {
    ANeuralNetworksExecution* ex = nullptr;
    int rc = ANeuralNetworksExecution_create(s.comp, &ex);
    if (rc) return rc;
    rc = ANeuralNetworksExecution_setInput(ex, 0, nullptr, in, (size_t)M * (size_t)s.k);
    if (!rc) rc = ANeuralNetworksExecution_setOutput(ex, 0, nullptr, out, (size_t)M * (size_t)s.n);
    if (!rc) {
        const std::uint64_t t0 = hv::now_ns();
        rc = ANeuralNetworksExecution_compute(ex);
        *ms = (hv::now_ns() - t0) / 1e6;
    }
    ANeuralNetworksExecution_free(ex);
    return rc;
}

static int exec_add(Slot& s, int M, const int8_t* a0, const int8_t* a1, int8_t* out, double* ms) {
    ANeuralNetworksExecution* ex = nullptr;
    int rc = ANeuralNetworksExecution_create(s.comp, &ex);
    if (rc) return rc;
    rc = ANeuralNetworksExecution_setInput(ex, 0, nullptr, a0, (size_t)M * (size_t)s.n);
    if (!rc) rc = ANeuralNetworksExecution_setInput(ex, 1, nullptr, a1, (size_t)M * (size_t)s.n);
    if (!rc) rc = ANeuralNetworksExecution_setOutput(ex, 0, nullptr, out, (size_t)M * (size_t)s.n);
    if (!rc) {
        const std::uint64_t t0 = hv::now_ns();
        rc = ANeuralNetworksExecution_compute(ex);
        *ms = (hv::now_ns() - t0) / 1e6;
    }
    ANeuralNetworksExecution_free(ex);
    return rc;
}

// ---------------------------------------------------------------------- args
struct Args {
    std::string block, scales, ring = "/data/local/tmp/hv_handoff.bin";
    std::string down = "skip";     // skip | stage
    std::string source = "fork";   // fork | none
    int blocks = 3;                // chunks to consume
    int nblocks = 1;               // NPU blocks per chunk (the cut's 180 - k)
    int M = 1530;                  // tokens per chunk
    int decode = 1;                // emit the decode budget line
    std::string dump_parity;       // DIR: dump tile 0's int8 in/out per GEMM (Gate 3)
    std::string act = "payload";   // payload | basis | window (Gate 3 probe activations)
    int hash_every = 1;            // hash ffn.up's 13.7 MB output every N blocks (0 = never)
                                  // 1 = every block (the P3.3/P3.4 behaviour, ~80 ms dead
                                  // gap per block measured there)
    std::string probe_dir;         // DIR: Gate 3 bisect — single/multi output-block graphs
    int probe_g = 0;               // GEMM index the bisect probes (0 = attn1.to_q)
    std::string probe_plan;        // "M,N,col0;M,N,col0;..." overrides the default sweep
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&](const char* n) -> std::string {
            if (i + 1 >= argc) { P("FATAL|missing_value|%s", n); std::exit(2); }
            return argv[++i];
        };
        if (k == "--block") a.block = val("--block");
        else if (k == "--scales") a.scales = val("--scales");
        else if (k == "--ring") a.ring = val("--ring");
        else if (k == "--down") a.down = val("--down");
        else if (k == "--source") a.source = val("--source");
        else if (k == "--blocks") a.blocks = atoi(val("--blocks").c_str());
        else if (k == "--nblocks") a.nblocks = atoi(val("--nblocks").c_str());
        else if (k == "--M") a.M = atoi(val("--M").c_str());
        else if (k == "--decode") a.decode = atoi(val("--decode").c_str());
        else if (k == "--dump-parity") a.dump_parity = val("--dump-parity");
        else if (k == "--act") a.act = val("--act");
        else if (k == "--hash-every") a.hash_every = atoi(val("--hash-every").c_str());
        else if (k == "--probe-dir") a.probe_dir = val("--probe-dir");
        else if (k == "--probe-g") a.probe_g = atoi(val("--probe-g").c_str());
        else if (k == "--probe-plan") a.probe_plan = val("--probe-plan");
        else if (k == "--help" || k == "-h") {
            P("usage: proc_b --block B [--scales S] [--ring R] [--blocks N] [--nblocks N] [--M 1530]"
              " [--down skip|stage] [--source fork|none] [--decode 0|1] [--hash-every N]"
              " [--dump-parity DIR] [--act payload|basis|window]"
              " [--probe-dir DIR --probe-g 0..8   (Gate 3 bisect, no ring needed)]");
            std::exit(0);
        } else { P("FATAL|unknown_arg|%s", k.c_str()); std::exit(2); }
    }
    if (a.block.empty()) { P("FATAL|need --block"); std::exit(2); }
    if (a.hash_every < 0) a.hash_every = 0;
    if (a.down != "skip" && a.down != "stage") { P("FATAL|bad_down|%s", a.down.c_str()); std::exit(2); }
    if (a.source != "fork" && a.source != "none") { P("FATAL|bad_source|%s", a.source.c_str()); std::exit(2); }
    if (a.blocks < 1) a.blocks = 1;
    if (a.nblocks < 1) a.nblocks = 1;
    return a;
}

// ------------------------------------------------------------ Gate 3 bisect
// P3.7b. The Gate 3 dump showed a FULLY_CONNECTED tile returning the exact
// integer product for only the LAST 384-column output block (every row) plus
// ROW 0 (every column). Two structural explanations fit, and they have
// different fixes:
//
//   (a) the multi-output-block path is broken  -> last-block-only is a
//       copy-back/loop bug, fix = one graph per 384-column block;
//   (b) the M = 384 row path is broken         -> only row 0 is right, fix is
//       in the row handling.
//
// This mode builds exactly those graphs — N = 384 (one block), N = 768 (two),
// N = full, at M = 1/8/128/384 — over the SAME one-memory-view weight window
// (offset + col0*K into the row-major [N_out, K] matrix, so still zero copies),
// runs them, and dumps the int8 activation and output for each. The host
// recomputes the product exactly. Nothing here is a latency claim.
struct ProbeSpec { int M, N, col0; const char* tag; };

static int run_probe(ANeuralNetworksDevice* tpu, ANeuralNetworksMemory* wmem, const Args& a) {
    const int g = a.probe_g;
    if (g < 0 || g >= 9) { P("FATAL|probe_g|%d|range=0..8", g); return 2; }
    const int NOUT = GEMMS[g].N;
    const int K = GEMMS[g].K;
    ::mkdir(a.probe_dir.c_str(), 0777);
    P("PROBE|plan|g=%d|gemm=%s|N_out=%d|K=%d|woff_base=%" PRId64 "|dir=%s",
      g, GEMMS[g].name, NOUT, K, GEMMS[g].off, a.probe_dir.c_str());

    std::vector<ProbeSpec> plan;
    if (!a.probe_plan.empty()) {
        // "M,N,col0;M,N,col0;..." — explicit sweep, no rebuild needed. Tags are
        // generated (and de-duplicated) so the host can key its verification by
        // tag without parsing positions.
        static char tagbuf[128][64];            // stable addresses: ProbeSpec holds const char*
        int ntags = 0;
        size_t pos = 0;
        const std::string& s = a.probe_plan;
        while (pos <= s.size()) {
            size_t sep = s.find(';', pos);
            const std::string item = s.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
            if (!item.empty()) {
                int M = 0, N = 0, c0 = 0;
                if (sscanf(item.c_str(), "%d,%d,%d", &M, &N, &c0) >= 2) {
                    if (ntags >= 128) { P("WARN|probe_plan_too_long"); break; }
                    snprintf(tagbuf[ntags], sizeof tagbuf[ntags], "m%d_n%d_c%d", M, N, c0);
                    bool dup = false;
                    for (int q = 0; q < ntags; ++q)
                        if (strcmp(tagbuf[q], tagbuf[ntags]) == 0) dup = true;
                    if (dup) {
                        char tmp[64];
                        snprintf(tmp, sizeof tmp, "%s_d", tagbuf[ntags]);
                        snprintf(tagbuf[ntags], sizeof tagbuf[ntags], "%s", tmp);
                    }
                    plan.push_back({M, N, c0, tagbuf[ntags]});
                    ++ntags;
                } else {
                    P("WARN|bad_probe_item|%s", item.c_str());
                }
            }
            if (sep == std::string::npos) break;
            pos = sep + 1;
        }
        P("PROBE|plan_explicit|items=%zu", plan.size());
    } else {
    plan.push_back({384, 384, 0,            "m384_n384_c0"});
    plan.push_back({384, 384, NOUT - 384,   "m384_n384_clast"});
    if (NOUT >= 768) {
        plan.push_back({384, 768, 0,          "m384_n768_c0"});
        plan.push_back({384, 768, NOUT - 768, "m384_n768_clast"});
    }
    plan.push_back({384, NOUT, 0,           "m384_nfull"});
    plan.push_back({1,   384, 0,            "m1_n384_c0"});
    plan.push_back({1,   NOUT, 0,           "m1_nfull"});
    plan.push_back({8,   384, 0,            "m8_n384_c0"});
    plan.push_back({128, 384, 0,            "m128_n384_c0"});
    plan.push_back({384, 384, 0,            "m384_n384_c0_rep"});
    }

    int fails = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const ProbeSpec& s = plan[i];
        const int64_t woff = GEMMS[g].off + (int64_t)s.col0 * (int64_t)K;
        Slot slot;
        int last_rc = 0;
        char nm[128];
        snprintf(nm, sizeof nm, "%s/%s", GEMMS[g].name, s.tag);
        WSrc w{wmem, woff, nullptr};
        const int crc = compile_fc(tpu, s.M, s.N, K, w, nm, (int)i, &slot, &last_rc);
        if (crc) { P("PROBE|FAIL|i=%zu|tag=%s|stage=compile|rc=%d(%s)", i, s.tag, last_rc, rcs(last_rc));
                   ++fails; continue; }

        // basis activation over this graph's K: A[r][r % K] = 127, so the exact
        // output is out[r][n] = W_global[col0 + n][r % K] — one stored weight
        // byte per element, no saturation, and the host can recompute it.
        std::vector<int8_t> in((size_t)s.M * (size_t)K, 0);
        for (int r = 0; r < s.M; ++r) in[(size_t)r * (size_t)K + (r % K)] = 127;
        std::vector<int8_t> out((size_t)s.M * (size_t)s.N, 0);

        double ms1 = 0, ms2 = 0;
        int erc = exec_fc(slot, s.M, in.data(), out.data(), &ms1);
        if (!erc) erc = exec_fc(slot, s.M, in.data(), out.data(), &ms2);
        if (erc) { P("PROBE|FAIL|i=%zu|tag=%s|stage=exec|rc=%d(%s)", i, s.tag, erc, rcs(erc));
                   ANeuralNetworksCompilation_free(slot.comp); ANeuralNetworksModel_free(slot.model);
                   ++fails; continue; }

        char fi[512], fo[512];
        snprintf(fi, sizeof fi, "%s/p%02zu_%s_in.bin", a.probe_dir.c_str(), i, s.tag);
        snprintf(fo, sizeof fo, "%s/p%02zu_%s_out.bin", a.probe_dir.c_str(), i, s.tag);
        const size_t inb = in.size(), outb = out.size();
        const bool oki = write_blob(fi, in.data(), inb);
        const bool oko = write_blob(fo, out.data(), outb);
        P("PROBE|i=%zu|tag=%s|g=%d|N=%d|K=%d|M=%d|col0=%d|woff=%" PRId64
          "|in_bytes=%zu|out_bytes=%zu|fnv_in=%016" PRIx64 "|fnv_out=%016" PRIx64
          "|compile_ms=%.1f|exec_ms=%.3f|ms2=%.3f|ok=%d",
          i, s.tag, g, s.N, K, s.M, s.col0, woff, inb, outb,
          fnv(FNV0, in.data(), inb), fnv(FNV0, out.data(), outb),
          slot.compile_ms, ms1, ms2, (oki && oko) ? 1 : 0);
        ANeuralNetworksCompilation_free(slot.comp);
        ANeuralNetworksModel_free(slot.model);
    }
    P("SUMMARY|probe|graphs=%zu|fails=%d", plan.size(), fails);
    return fails ? 1 : 0;
}

// ------------------------------------------------------------- fork producer
// Stands in for proc_a so this stage can be exercised alone: same ring, same
// publish protocol, synthetic fp16 payload, and a hash chain the parent's chain
// must match byte for byte (the handoff integrity check).
static int run_producer(const Args& a, hv::Ring& r) {
    const size_t nwords = hv::CHUNK_BYTES / 2;
    std::uint64_t chain = FNV0;
    for (int chunk = 1; chunk <= a.blocks; ++chunk) {
        const int s = (chunk - 1) % hv::SLOTS;
        std::uint64_t waited = 0;
        if (!hv::wait_free(r, s, (std::uint64_t)chunk, 300ull * 1'000'000'000ull, &waited)) {
            P("FATAL|prod_slot_never_freed|chunk=%d", chunk);
            return 2;
        }
        std::uint16_t* w = reinterpret_cast<std::uint16_t*>(r.payload(s));
        std::uint32_t x = 0x51ED270Bu + (std::uint32_t)chunk * 2654435761u;
        for (size_t i = 0; i < nwords; ++i) {
            x = x * 1664525u + 1013904223u;
            const float f = ((float)((x >> 16) & 0xFFFFu) - 32768.0f) / 32768.0f;   // [-1,1)
            w[i] = f2h(f);
        }
        const std::uint64_t h = fnv(FNV0, w, hv::CHUNK_BYTES);
        chain = fnv(chain, &h, sizeof(h));
        long ms = 0;
        hv::publish(r, s, (std::uint64_t)chunk, (std::uint32_t)hv::CHUNK_BYTES,
                    hv::Msync::FirstPage, &ms);
        P("TS|prod|chunk=%d|slot=%d|hash=%016" PRIx64 "|msync_ns=%ld|slot_wait_us=%" PRIu64,
          chunk, s, h, ms, waited / 1000);
    }
    __atomic_store_n(&r.hdr((a.blocks - 1) % hv::SLOTS)->flags, hv::FLAG_STREAM_DONE, __ATOMIC_RELEASE);
    P("SUMMARY|prod|chunks=%d|hash_chain=%016" PRIx64, a.blocks, chain);
    return 0;
}

// ------------------------------------------------------------------- main
int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const int tiles = (a.M + TILE - 1) / TILE;

    // ---- the artifact: one fd, one memory view, per-GEMM windows (route B)
    int fd = ::open(a.block.c_str(), O_RDWR);
    if (fd < 0) fd = ::open(a.block.c_str(), O_RDONLY);
    if (fd < 0) { P("FATAL|cannot_open_block|%s", a.block.c_str()); return 2; }
    struct stat st {};
    if (fstat(fd, &st) != 0 || (size_t)st.st_size != (size_t)W_BYTES) {
        P("FATAL|block_size|%s|got=%lld|want=%lld", a.block.c_str(),
          (long long)st.st_size, (long long)W_BYTES);
        return 2;
    }
    ANeuralNetworksMemory* wmem = nullptr;
    int rc = ANeuralNetworksMemory_createFromFd((size_t)W_BYTES, PROT_READ, fd, 0, &wmem);
    if (rc) { P("FATAL|createFromFd|rc=%d(%s)", rc, rcs(rc)); return 2; }

    std::string scl_note = "none";
    if (!a.scales.empty()) {
        struct stat ss {};
        if (stat(a.scales.c_str(), &ss) == 0) {
            scl_note = (ss.st_size == 91136) ? "present_91136B" : "unexpected_size";
            P("SCALES|path=%s|bytes=%lld|%s", a.scales.c_str(), (long long)ss.st_size, scl_note.c_str());
        }
    }

    ANeuralNetworksDevice* tpu = find_tpu();
    if (!tpu) { P("FATAL|no_edgetpu_device"); return 3; }

    P("HV|proc_b|block=%s|bytes=%lld|M=%d|tiles=%d|TILE=%d|chunks=%d|nblocks=%d|down=%s|source=%s",
      a.block.c_str(), (long long)W_BYTES, a.M, tiles, TILE, a.blocks, a.nblocks,
      a.down.c_str(), a.source.c_str());
    P("NOTE|scales=%s|per_tensor_graphs(per_column_rejected)|latency_only", scl_note.c_str());

    // ---- Gate 3 bisect: standalone probe run, no ring, no producer, no decode.
    if (!a.probe_dir.empty()) {
        const int prc = run_probe(tpu, wmem, a);
        ANeuralNetworksMemory_free(wmem);
        ::close(fd);
        return prc;
    }

    // ---- 9 held compilations, one graph per GEMM. A whole-block graph reboots
    //      this phone inside Compilation_finish — never retry that here.
    std::vector<Slot> slots;
    slots.reserve(9);
    double compile9_ms = 0;
    int last_rc = 0;
    for (int g = 0; g < 9; ++g) {
        Slot s;
        const int crc = compile_fc(tpu, TILE, GEMMS[g].N, GEMMS[g].K,
                                   WSrc{wmem, GEMMS[g].off, nullptr}, GEMMS[g].name, g, &s, &last_rc);
        if (crc) { P("RESULT|FAIL|compile|%s|rc=%d", GEMMS[g].name, last_rc); return 7; }
        compile9_ms += s.compile_ms;
        slots.push_back(s);
    }
    P("READY|n=%zu|compile_ms=%.1f|M=%d|tiles=%d", slots.size(), compile9_ms, TILE, tiles);

    // ---- ffn.down branch
    const int DCH = 7;
    const int DKC = NUP / DCH;                    // 1280 <= the 1664 compiler ceiling
    std::vector<Slot> down_fc, down_add;
    std::vector<int8_t> stage_w;
    double down_gather_ms = 0, down_compile_ms = 0;
    size_t down_stage_bytes = 0;

    if (a.down == "stage") {
        // The row-major [1536,8960] matrix cannot be sliced along K into a legal
        // constant operand (chunk c is columns c*1280.., stride 8960). Gather it.
        const int8_t* src = static_cast<const int8_t*>(
            mmap(nullptr, (size_t)W_BYTES, PROT_READ, MAP_SHARED, fd, 0));
        if (src == MAP_FAILED) { P("FATAL|down_gather_mmap"); return 2; }
        stage_w.resize((size_t)KD * DKC * DCH);
        const std::uint64_t g0 = hv::now_ns();
        size_t dst = 0;
        for (int c = 0; c < DCH; ++c) {
            for (int n = 0; n < KD; ++n) {
                std::memcpy(&stage_w[dst], src + DOWN_OFF + (size_t)n * NUP + (size_t)c * DKC, DKC);
                dst += DKC;
            }
        }
        down_gather_ms = (hv::now_ns() - g0) / 1e6;
        down_stage_bytes = dst;
        P("DOWN|branch=stage|chunks=%d|Kc=%d|gather_ms=%.1f|staged_bytes=%zu",
          DCH, DKC, down_gather_ms, dst);
        P("WARN|down_staging_copy_bytes=%zu|one_copy_deviation_reported_not_adopted", dst);
        for (int c = 0; c < DCH; ++c) {
            Slot s;
            const int cc = compile_fc(tpu, TILE, KD, DKC,
                                      WSrc{nullptr, 0, &stage_w[(size_t)c * KD * DKC]},
                                      "ffn.down[stage]", 100 + c, &s, &last_rc);
            if (cc) { P("RESULT|FAIL|compile|down_stage|c=%d|rc=%d", c, last_rc); return 7; }
            down_compile_ms += s.compile_ms;
            down_fc.push_back(s);
        }
        Slot sa;
        if (compile_add(tpu, TILE, KD, &sa, &last_rc)) {
            P("RESULT|FAIL|compile|down_add|rc=%d", last_rc);
            return 7;
        }
        down_compile_ms += sa.compile_ms;
        down_add.push_back(sa);
        P("READY|down=stage|fc=%zu|add=%zu|compile_ms=%.1f", down_fc.size(), down_add.size(),
          down_compile_ms);
    } else {
        P("DOWN|branch=skip|mali_ref_ms=47.33|mali_gops=890|src=RESULTS_gemm_mali.md"
          "|note=down_projection_stays_on_the_GPU_stage");
    }

    // ---- ring (consumer side) + optional forked producer
    hv::Ring ring = hv::map_ring(a.ring.c_str(), true);
    if (!ring.ok()) { P("FATAL|cannot_map_ring|%s", a.ring.c_str()); return 2; }
    hv::reset_headers(ring);
    P("RING|path=%s|file_bytes=%zu|slots=%d|slot_stride=%zu|chunk_bytes=%zu",
      a.ring.c_str(), hv::FILE_BYTES, hv::SLOTS, hv::SLOT_STRIDE, hv::CHUNK_BYTES);

    pid_t prod = -1;
    if (a.source == "fork") {
        prod = fork();
        if (prod == 0) {
            const int prc = run_producer(a, ring);
            std::fflush(stdout);
            hv::unmap_ring(ring);
            std::_Exit(prc);
        }
        if (prod < 0) { P("FATAL|fork_producer"); return 2; }
    }

    // ---- work buffers
    std::vector<int8_t> A((size_t)tiles * TILE * KD, 0);          // int8 activation, [tiles*TILE, KD]
    // 4096-byte aligned: testing whether the HAL's output copy-back is sensitive
    // to buffer alignment (the parity dump shows only part of the tile landing).
    static int8_t Obuf_store[(size_t)TILE * NUP] __attribute__((aligned(4096)));
    int8_t* Obuf = Obuf_store;                                    // per-tile GEMM scratch (max N = ffn.up)
    std::vector<int8_t> UpOut((size_t)tiles * TILE * NUP, 0);     // ffn.up output, per tile (down's input)
    std::vector<int8_t> Dacc((size_t)TILE * KD, 0), Dping((size_t)TILE * KD, 0);
    std::vector<std::vector<int8_t>> Dchunk(DCH, std::vector<int8_t>((size_t)TILE * KD, 0));
    std::vector<int8_t> Act_chunk((size_t)TILE * DKC, 0);

    std::vector<double> gsum(9, 0.0);
    std::uint64_t gcalls = 0, dcalls = 0;
    double quant_ms_total = 0, down_ms_total = 0, gather_act_ms_total = 0;
    std::uint64_t lat_min = ~0ull, lat_max = 0, lat_sum = 0;
    std::uint64_t hash_chain = FNV0, out_hash = FNV0;
    std::uint64_t out_hash_calls = 0;
    double out_hash_ms_total = 0.0;
    std::vector<double> chunk_block_ms;
    int proto_err = 0;

    for (int chunk = 1; chunk <= a.blocks && !proto_err; ++chunk) {
        const int sIdx = (chunk - 1) % hv::SLOTS;
        std::uint64_t t_seen = 0;
        if (!hv::wait_publish(ring, sIdx, (std::uint64_t)chunk, 300ull * 1'000'000'000ull, &t_seen)) {
            P("FATAL|chunk_never_published|chunk=%d|slot=%d", chunk, sIdx);
            proto_err = 1;
            break;
        }
        hv::SlotHdr* h = ring.hdr(sIdx);
        const std::uint32_t bytes = __atomic_load_n(&h->bytes, __ATOMIC_RELAXED);
        const std::uint64_t t_prod = __atomic_load_n(&h->t_prod_ns, __ATOMIC_RELAXED);
        const std::uint32_t flags = __atomic_load_n(&h->flags, __ATOMIC_RELAXED);
        const std::uint16_t* w = reinterpret_cast<const std::uint16_t*>(ring.payload(sIdx));
        const size_t nw = bytes / 2;

        // integrity: hash exactly the bytes proc_a (or the forked producer) wrote,
        // byte-wise — the same construction the producer prints, so the chains are
        // comparable. (Hashing per 16-bit word here instead was a real mismatch.)
        std::uint64_t nan = 0, zero = 0;
        for (size_t i = 0; i < nw; ++i) {
            const std::uint16_t v = w[i];
            if (((v & 0x7C00u) == 0x7C00u) && (v & 0x3FFu)) ++nan;
            if (v == 0) ++zero;
        }
        const std::uint64_t hh = fnv(FNV0, w, bytes);
        hash_chain = fnv(hash_chain, &hh, sizeof(hh));
        const std::uint64_t lat = t_seen > t_prod ? t_seen - t_prod : 0;
        lat_sum += lat;
        if (lat < lat_min) lat_min = lat;
        if (lat > lat_max) lat_max = lat;
        if (bytes != hv::CHUNK_BYTES) {
            P("WARN|chunk_bytes|chunk=%d|got=%u|want=%zu", chunk, bytes, hv::CHUNK_BYTES);
        }

        // fp16 residual -> INT8 activation at SA, rows padded to tiles*TILE
        const std::uint64_t q0 = hv::now_ns();
        if (a.act != "payload") {
            // Gate 3 probe activations. The dense residual saturates ~45 % of the
            // int8 outputs at SA = SW = 1/127, which makes an elementwise parity
            // check information-poor; these patterns keep every output in range
            // and make the device's pairing directly checkable on the host:
            //   basis   A[r][r % K] = 127      => out[r][n] = W[n][r % K] exactly
            //   window  A[r][8*(r%192) + j] = 1 => out[r][n] = round(sum of 8 w / 127)
            std::fill(A.begin(), A.end(), (int8_t)0);
            const int rows = tiles * TILE;
            for (int r = 0; r < rows; ++r) {
                int8_t* d = &A[(size_t)r * KD];
                if (a.act == "basis") {
                    d[r % KD] = 127;
                } else {
                    const int off = 8 * (r % 192);
                    for (int j = 0; j < 8 && off + j < KD; ++j) d[off + j] = 1;
                }
            }
            P("ACT|mode=%s|rows=%d|K=%d", a.act.c_str(), rows, KD);
        } else {
        std::fill(A.begin(), A.end(), (int8_t)0);
        for (int t = 0; t < a.M; ++t) {
            int8_t* dst = &A[(size_t)t * KD];
            const std::uint16_t* srow = w + (size_t)t * KD;
            for (int k = 0; k < KD; ++k) {
                long q = std::lroundf(h2f(srow[k]) / SA);
                if (q > 127) q = 127;
                else if (q < -128) q = -128;
                dst[k] = (int8_t)q;
            }
        }
        }
        const double quant_ms = (hv::now_ns() - q0) / 1e6;
        quant_ms_total += quant_ms;

        for (int b = 0; b < a.nblocks && !proto_err; ++b) {
            const std::uint64_t tb0 = hv::now_ns();
            double gsum_block = 0;
            for (int g = 0; g < 9 && !proto_err; ++g) {
                double ms_sum = 0;
                for (int t = 0; t < tiles; ++t) {
                    const int8_t* in = &A[(size_t)t * TILE * KD];
                    // ffn.up keeps its per-tile output (ffn.down consumes it); every
                    // other GEMM writes into the shared scratch tile.
                    int8_t* dst = (g == 8) ? &UpOut[(size_t)t * TILE * NUP] : Obuf;
                    double ms = 0;
                    const int erc = exec_fc(slots[g], TILE, in, dst, &ms);
                    if (erc) {
                        P("FATAL|exec|%s|chunk=%d|block=%d|tile=%d|rc=%d(%s)",
                          GEMMS[g].name, chunk, b + 1, t, erc, rcs(erc));
                        proto_err = 1;
                        break;
                    }
                    ms_sum += ms;
                    gsum[g] += ms;
                    ++gcalls;
                    // Gate 3: dump the first block's first tile, per GEMM — the
                    // exact int8 activation the graph saw and the int8 output it
                    // produced. The host recomputes the product exactly.
                    if (t == 0 && b == 0 && chunk == 1 && !a.dump_parity.empty()) {
                        ::mkdir(a.dump_parity.c_str(), 0777);
                        char fi[512], fo[512];
                        snprintf(fi, sizeof fi, "%s/%02d_%s_in.bin",
                                 a.dump_parity.c_str(), g, GEMMS[g].name);
                        snprintf(fo, sizeof fo, "%s/%02d_%s_out.bin",
                                 a.dump_parity.c_str(), g, GEMMS[g].name);
                        const size_t inb = (size_t)TILE * (size_t)GEMMS[g].K;
                        const size_t outb = (size_t)TILE * (size_t)GEMMS[g].N;
                        const bool oki = write_blob(fi, in, inb);
                        const bool oko = write_blob(fo, dst, outb);
                        P("PARITY|g=%d|gemm=%s|N=%d|K=%d|M=%d|in_bytes=%zu|out_bytes=%zu"
                          "|fnv_in=%016" PRIx64 "|fnv_out=%016" PRIx64 "|ok=%d",
                          g, GEMMS[g].name, GEMMS[g].N, GEMMS[g].K, TILE, inb, outb,
                          fnv(FNV0, in, inb), fnv(FNV0, dst, outb),
                          (oki && oko) ? 1 : 0);
                        // Is the output written asynchronously? Re-dump after a
                        // delay (g=0 only): if the bytes change, compute() returns
                        // before the HAL has finished writing and every dump
                        // raced the writer.
                        if (g == 0) {
                            const std::uint64_t h_before = fnv(FNV0, dst, outb);
                            usleep(150 * 1000);
                            char f2[512];
                            snprintf(f2, sizeof f2, "%s/%02d_%s_out2.bin",
                                     a.dump_parity.c_str(), g, GEMMS[g].name);
                            const std::uint64_t h_after = fnv(FNV0, dst, outb);
                            const bool ok2 = write_blob(f2, dst, outb);
                            P("PARITY2|g=%d|fnv_at_dump=%016" PRIx64 "|fnv_after_150ms=%016" PRIx64
                              "|changed=%d|ok=%d",
                              g, h_before, h_after, h_before == h_after ? 0 : 1, ok2 ? 1 : 0);
                        }
                    }
                }
                if (proto_err) break;
                gsum_block += ms_sum;
                P("BLK|npu|chunk=%d|block=%d|gemm=%s|N=%d|K=%d|tiles=%d|ms=%.3f",
                  chunk, b + 1, GEMMS[g].name, GEMMS[g].N, GEMMS[g].K, tiles, ms_sum);
            }
            if (proto_err) break;

            double down_ms = -1.0;
            if (a.down == "stage") {
                const std::uint64_t d0 = hv::now_ns();
                double act_ms = 0;
                for (int t = 0; t < tiles; ++t) {
                    const int8_t* urow = &UpOut[(size_t)t * TILE * NUP];   // this tile's ffn.up output
                    // gather this tile's K-slices; activation re-layout is inherent
                    // to the decomposition, so it is priced separately from weights
                    double sum_ms = 0;
                    for (int c = 0; c < DCH; ++c) {
                        const std::uint64_t a0 = hv::now_ns();
                        for (int m = 0; m < TILE; ++m) {
                            std::memcpy(&Act_chunk[(size_t)m * DKC],
                                        urow + (size_t)m * NUP + (size_t)c * DKC, DKC);
                        }
                        act_ms += (hv::now_ns() - a0) / 1e6;
                        double ms = 0;
                        const int erc = exec_fc(down_fc[c], TILE, Act_chunk.data(),
                                                Dchunk[c].data(), &ms);
                        if (erc) { P("FATAL|down_exec|c=%d|rc=%d(%s)", c, erc, rcs(erc));
                                   proto_err = 1; break; }
                        sum_ms += ms;
                        ++dcalls;
                    }
                    if (proto_err) break;
                    // 6 ADDs to accumulate the 7 chunk outputs (ping-pong)
                    const int8_t* prev = Dchunk[0].data();
                    for (int c = 1; c < DCH; ++c) {
                        int8_t* dst = (c & 1) ? Dping.data() : Dacc.data();
                        double ms = 0;
                        const int erc = exec_add(down_add[0], TILE, prev, Dchunk[c].data(), dst, &ms);
                        if (erc) { P("FATAL|down_add|c=%d|rc=%d(%s)", c, erc, rcs(erc));
                                   proto_err = 1; break; }
                        sum_ms += ms;
                        prev = dst;
                        ++dcalls;
                    }
                    if (proto_err) break;
                    down_ms = (down_ms < 0 ? 0.0 : down_ms) + sum_ms;
                }
                gather_act_ms_total += act_ms;
                const double wall = (hv::now_ns() - d0) / 1e6;
                down_ms_total += wall;
                P("BLK|npu|chunk=%d|block=%d|gemm=ffn.down|branch=stage|tiles=%d|ms=%.3f"
                  "|fc_add_ms=%.3f|act_gather_ms=%.3f", chunk, b + 1, tiles, wall, down_ms, act_ms);
                down_ms = wall;
            }

            const std::uint64_t tb1 = hv::now_ns();
            const double bms = (tb1 - tb0) / 1e6;
            chunk_block_ms.push_back(bms);
            P("TS|npu|chunk=%d|block=%d|t0=%.3f|t1=%.3f|ms=%.3f|down_ms=%.3f|gemms9_ms=%.3f",
              chunk, b + 1, (double)tb0 / 1e6, (double)tb1 / 1e6, bms, down_ms, gsum_block);
            // hash the last graph's output so the artifact proves compute happened.
            // Strided: FNV over 13,762,560 B costs ~80 ms, which P3.4 measured as a
            // dead gap between blocks (1.62 s per 20-block chunk, 12.8 s per 160) —
            // an artifact of the harness, not of the pipeline. `--hash-every 20`
            // keeps the evidence at 1/20 the cost; 0 disables it entirely.
            if (a.hash_every == 1 || (a.hash_every > 1 && (b + 1) % a.hash_every == 0)) {
                const std::uint64_t hh0 = hv::now_ns();
                out_hash = fnv(out_hash, UpOut.data(), UpOut.size());
                out_hash_ms_total += (hv::now_ns() - hh0) / 1e6;
                ++out_hash_calls;
            }
        }
        if (proto_err) break;

        if (a.decode && chunk == 1) {
            P("DECODE|stub=1|T_V_s=10.640|branch=hybrid|src=RESULTS_vae_tv.md"
              "|note=decode_not_executed_here_Gate3_owns_it|conv2d_class_unsoaked_23.5s");
        }

        hv::release(ring, sIdx, (std::uint64_t)chunk);
        P("TS|npu|chunk=%d|slot=%d|lat_ns=%" PRIu64 "|payload_hash=%016" PRIx64
          "|zero_words=%" PRIu64 "|nan_words=%" PRIu64 "|quant_ms=%.3f|flags=%u",
          chunk, sIdx, lat, hh, zero, nan, quant_ms, flags);
    }

    int prc = 0;
    if (prod > 0) {
        int stp = 0;
        waitpid(prod, &stp, 0);
        prc = WIFEXITED(stp) ? WEXITSTATUS(stp) : 128;
    }

    // ---- summary
    double g9 = 0;
    for (double v : gsum) g9 += v;
    const double per_block_g9 = g9 / (double)(a.blocks * a.nblocks);
    const double block_med = [&] {
        if (chunk_block_ms.empty()) return 0.0;
        std::vector<double> s = chunk_block_ms;
        std::sort(s.begin(), s.end());
        return s[s.size() / 2];
    }();
    const double macs9 = (double)tiles * (double)TILE *
                         (8.0 * KD * KD + (double)KD * NUP);
    P("SUMMARY|npu|chunks=%d|nblocks=%d|tiles=%d|gemm_execs=%" PRIu64 "|down_execs=%" PRIu64
      "|gemms9_ms_per_block=%.3f|block_ms_median=%.3f|quant_ms_total=%.3f|down_ms_total=%.3f"
      "|act_gather_ms_total=%.3f|lat_mean_us=%.3f|lat_max_us=%.3f|hash_chain=%016" PRIx64
      "|out_hash=%016" PRIx64 "|compile9_ms=%.1f|compile_down_ms=%.1f|down_gather_ms=%.1f"
      "|hash_every=%d|out_hash_calls=%" PRIu64 "|out_hash_ms_total=%.3f",
      a.blocks, a.nblocks, tiles, gcalls, dcalls, per_block_g9, block_med, quant_ms_total,
      down_ms_total, gather_act_ms_total,
      chunk_block_ms.empty() ? 0.0 : (double)lat_sum / (double)chunk_block_ms.size() / 1e3,
      lat_max / 1e3, hash_chain, out_hash, compile9_ms, down_compile_ms, down_gather_ms,
      a.hash_every, out_hash_calls, out_hash_ms_total);
    // 9-GEMM subset: 2 * MACs / (s per block). The down projection is excluded
    // when it is skipped (it runs on Mali) — stated, never folded in silently.
    P("THROUGHPUT|gemms9_mac_per_block=%.3f|gemms9_gops=%.1f|down_executed=%d",
      macs9 / 1e9, per_block_g9 > 0 ? 2.0 * macs9 / (per_block_g9 * 1e6) : -1.0,
      (a.down == "stage") ? 1 : 0);

    const bool pass = (proto_err == 0 && prc == 0 && gcalls ==
                       (std::uint64_t)a.blocks * a.nblocks * 9 * tiles);
    P("RESULT|%s|chunks=%d/%d|gemm_execs=%" PRIu64 "/%" PRIu64 "|down=%s|proto_err=%d|prod_rc=%d",
       pass ? "PASS" : "FAIL", a.blocks, a.blocks, gcalls,
       (std::uint64_t)a.blocks * a.nblocks * 9 * tiles, a.down.c_str(), proto_err, prc);

    for (auto& s : slots) {
        ANeuralNetworksCompilation_free(s.comp);
        ANeuralNetworksModel_free(s.model);
    }
    for (auto& s : down_fc) {
        ANeuralNetworksCompilation_free(s.comp);
        ANeuralNetworksModel_free(s.model);
    }
    for (auto& s : down_add) {
        ANeuralNetworksCompilation_free(s.comp);
        ANeuralNetworksModel_free(s.model);
    }
    ANeuralNetworksMemory_free(wmem);
    hv::unmap_ring(ring);
    ::close(fd);
    return pass ? 0 : 1;
}
