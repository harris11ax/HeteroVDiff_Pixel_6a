// tools/android/pipe/proc_a.cpp — P3.2: the GPU stage over the real INT8 artifact.
//
// The Mali half of the two-process pipeline. It does exactly what the cut assigns
// to the GPU — blocks `[0, k)` of the trajectory — and hands each block's residual
// stream to the NPU stage through phase3_pipeline/handoff.h:
//
//   mmap(block_L.bin, MAP_SHARED, O_RDONLY)      <- route B, Gate 2
//     -> clImportMemoryARM(HOST_ARM) per GEMM      zero copy, no second INT8 copy
//   run the block table from the tiled gemm8.cl kernel (64x64x32, tile 8x4)
//   10 GEMMs: 8 x [1536,1536] attn + ffn.up [8960,1536] + ffn.down [1536,8960]
//   dequantised ffn.down output -> fp16 residual -> ring slot (i-1) % 2
//
// Every number printed is measured in this process: per-GEMM ms, per-block ms,
// payload conversion ms, msync ns, and the publish->consume latency from the
// drain consumer (the NPU stage's stand-in until P3.3).
//
//   proc_a --block /data/local/tmp/hv_blk.bin --scales /data/local/tmp/hv_blk.scales.f32 \
//          --cl /data/local/tmp/gemm8.cl --ring /data/local/tmp/hv_handoff.bin --blocks 8
//
//   --drain fork    (default when blocks > 2) fork a consumer that releases slots
//   --drain none    no consumer: only legal for <= SLOTS blocks
//   --route import|copy   import = zero-copy (default); copy forces the pessimistic
//                         path so the cost of NOT being zero-copy is measurable
//
// Exit: 0 = PASS, 1 = verification/protocol failure, 2 = setup failure.

#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "handoff.h"

// ------------------------------------------------------------------ CL glue
typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_mem_flags;
typedef intptr_t cl_context_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_TRUE 1
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_KERNEL_PRIVATE_MEM_SIZE 0x11B4
#define CL_KERNEL_WORK_GROUP_SIZE 0x11B0
#define CL_EVENT_COMMAND_EXECUTION_STATUS 0x11F3
#define CL_IMPORT_TYPE_ARM 0x40B2
#define CL_IMPORT_TYPE_HOST_ARM 0x40B3

static const char* cl_err(int e) {
    switch (e) {
        case 0: return "SUCCESS";
        case -4: return "MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "OUT_OF_RESOURCES";
        case -9: return "BUILD_PROGRAM_FAILURE";
        case -11: return "BUILD_PROGRAM_FAILURE";
        case -14: return "EXEC_STATUS_ERROR";
        case -30: return "INVALID_VALUE";
        case -52: return "INVALID_KERNEL_ARGS";
        case -54: return "INVALID_WORK_GROUP_SIZE";
        case -64: return "INVALID_PROPERTY";
        default: return "?";
    }
}

struct CL {
    void* h = nullptr;
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_context (*CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                void*, void*, cl_int*);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*,
                                cl_uint, const void*, void*);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_kernel (*CreateKernel)(cl_program, const char*, cl_int*);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                                   const size_t*, const size_t*, cl_uint, const void*, void*);
    cl_int (*Finish)(cl_command_queue);
    cl_int (*GetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_int (*WaitForEvents)(cl_uint, const void**);
    cl_int (*GetEventInfo)(void*, cl_uint, size_t, void*, size_t*);
    cl_int (*ReleaseEvent)(void*);
    cl_int (*ReleaseMemObject)(cl_mem);
    cl_int (*ReleaseKernel)(cl_kernel);
    cl_int (*ReleaseProgram)(cl_program);
    cl_int (*ReleaseCommandQueue)(cl_command_queue);
    cl_int (*ReleaseContext)(cl_context);
    // clImportMemoryARM is NOT exported by Android's ICD loader; it is resolved
    // through clGetExtensionFunctionAddressForPlatform (Gate 2, trap #1).
    cl_mem (*ImportMemoryARM)(cl_context, cl_mem_flags, const void*, void*, size_t, cl_int*);

    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
};

static void pl(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}

static bool cl_open(CL* c, const char* import_type_note) {
    (void)import_type_note;
    memset(c, 0, sizeof(*c));
    c->h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!c->h) c->h = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!c->h) { pl("FATAL|libOpenCL_missing"); return false; }
#define S(f, n) *(void**)(&c->f) = dlsym(c->h, n); if (!c->f) { pl("FATAL|missing_%s", n); return false; }
    S(GetPlatformIDs, "clGetPlatformIDs")
    S(GetDeviceIDs, "clGetDeviceIDs")
    S(GetDeviceInfo, "clGetDeviceInfo")
    S(CreateContext, "clCreateContext")
    S(CreateCommandQueue, "clCreateCommandQueue")
    S(CreateBuffer, "clCreateBuffer")
    S(EnqueueReadBuffer, "clEnqueueReadBuffer")
    S(CreateProgramWithSource, "clCreateProgramWithSource")
    S(BuildProgram, "clBuildProgram")
    S(GetProgramBuildInfo, "clGetProgramBuildInfo")
    S(CreateKernel, "clCreateKernel")
    S(SetKernelArg, "clSetKernelArg")
    S(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel")
    S(Finish, "clFinish")
    S(GetKernelWorkGroupInfo, "clGetKernelWorkGroupInfo")
    S(WaitForEvents, "clWaitForEvents")
    S(GetEventInfo, "clGetEventInfo")
    S(ReleaseEvent, "clReleaseEvent")
    S(ReleaseMemObject, "clReleaseMemObject")
    S(ReleaseKernel, "clReleaseKernel")
    S(ReleaseProgram, "clReleaseProgram")
    S(ReleaseCommandQueue, "clReleaseCommandQueue")
    S(ReleaseContext, "clReleaseContext")
#undef S

    cl_uint np = 0;
    if (c->GetPlatformIDs(0, nullptr, &np) || !np) { pl("FATAL|no_platform"); return false; }
    cl_platform_id plats[4];
    c->GetPlatformIDs(np > 4 ? 4 : np, plats, nullptr);
    c->plat = plats[0];
    cl_uint nd = 0;
    c->GetDeviceIDs(c->plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
    if (!nd) { pl("FATAL|no_device"); return false; }
    cl_device_id devs[4];
    c->GetDeviceIDs(c->plat, CL_DEVICE_TYPE_ALL, nd > 4 ? 4 : nd, devs, nullptr);
    c->dev = devs[0];

    cl_int err = 0;
    c->ctx = c->CreateContext(nullptr, 1, &c->dev, nullptr, nullptr, &err);
    if (!c->ctx) { pl("FATAL|ctx=%d", err); return false; }
    c->q = c->CreateCommandQueue(c->ctx, c->dev, 0, &err);
    if (!c->q) { pl("FATAL|queue=%d", err); return false; }

    typedef void* (*gefp_t)(cl_platform_id, const char*);
    gefp_t gefp = (gefp_t)dlsym(c->h, "clGetExtensionFunctionAddressForPlatform");
    if (gefp) *(void**)(&c->ImportMemoryARM) = gefp(c->plat, "clImportMemoryARM");
    if (!c->ImportMemoryARM) *(void**)(&c->ImportMemoryARM) = dlsym(c->h, "clImportMemoryARM");

    char name[256] = {0};
    size_t got = 0;
    c->GetDeviceInfo(c->dev, CL_DEVICE_NAME, sizeof(name) - 1, name, &got);
    cl_uint cus = 0;
    c->GetDeviceInfo(c->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, &got);
    pl("CL_DEVICE|name=%s|cus=%u|import_memory_arm=%d", name, cus, c->ImportMemoryARM ? 1 : 0);
    return true;
}

// ------------------------------------------------------- block GEMM table
// Region order is manifest.json:gemm_order — the device must use it (Gate 1 §"Region
// order"). Offsets are derived cumulatively here and asserted against the sizes the
// manifest reports, so a layout drift fails loudly instead of computing garbage.
struct GemmSpec {
    const char* name;
    int N, K;
    size_t off, nbytes, scale_off;
    int n_scales;
};

static const int KD = 1536;             // model width at Preset A
static const int NUP = 8960;            // ffn.net.0.proj output width

static int build_table(int M, std::vector<GemmSpec>* out) {
    static const char* order[10] = {
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out",
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out",
        "ffn.net.0.proj", "ffn.net.2",
    };
    size_t off = 0, soff = 0;
    out->clear();
    for (int i = 0; i < 10; ++i) {
        GemmSpec g{};
        g.name = order[i];
        const bool up = (i == 8), down = (i == 9);
        g.N = up ? NUP : KD;
        g.K = down ? NUP : KD;
        g.off = off;
        g.nbytes = (size_t)g.N * g.K;
        g.scale_off = soff;
        g.n_scales = g.N;
        off += g.nbytes;
        soff += (size_t)g.n_scales * sizeof(float);
        out->push_back(g);
    }
    (void)M;
    return off == 46'399'488u && soff == 91'136u ? 0 : 1;
}

// --------------------------------------------------------------- fp16 conv
// bit-exact float->half (round-to-nearest-even); no dependency on a half library.
static inline unsigned short f2h(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    const unsigned sign = (u >> 16) & 0x8000u;
    int exp = (int)((u >> 23) & 0xFF) - 127 + 15;
    unsigned mant = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF) return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp >= 0x1F) return (unsigned short)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        const int shift = 14 - exp;
        unsigned h = mant >> shift;
        const unsigned rem = mant & ((1u << shift) - 1);
        const unsigned half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (unsigned short)(sign | h);
    }
    unsigned h = ((unsigned)exp << 10) | (mant >> 13);
    const unsigned rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;   // may carry into exp: correct
    return (unsigned short)(sign | h);
}

// ------------------------------------------------------------------ args
struct Args {
    std::string block;
    std::string scales;
    std::string cl = "/data/local/tmp/gemm8.cl";
    std::string ring = "/data/local/tmp/hv_handoff.bin";
    std::string route = "import";   // import | copy
    int blocks = 8;
    int M = 1530;
    std::string drain = "fork";
    int warmup = 1;
    int pub_every = 1;      // publish one chunk per N blocks (the cut hands off once per 160)
    int once_blocks = -1;   // closure-reference pass length; -1 = all blocks
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&](const char* n) -> std::string {
            if (i + 1 >= argc) { pl("FATAL|missing_value|%s", n); std::exit(2); }
            return argv[++i];
        };
        if (k == "--block") a.block = val("--block");
        else if (k == "--scales") a.scales = val("--scales");
        else if (k == "--cl") a.cl = val("--cl");
        else if (k == "--ring") a.ring = val("--ring");
        else if (k == "--route") a.route = val("--route");
        else if (k == "--blocks") a.blocks = atoi(val("--blocks").c_str());
        else if (k == "--M") a.M = atoi(val("--M").c_str());
        else if (k == "--drain") a.drain = val("--drain");
        else if (k == "--warmup") a.warmup = atoi(val("--warmup").c_str());
        else if (k == "--pub-every") a.pub_every = atoi(val("--pub-every").c_str());
        else if (k == "--once-blocks") a.once_blocks = atoi(val("--once-blocks").c_str());
        else if (k == "--help" || k == "-h") {
            pl("usage: proc_a --block B --scales S [--cl C] [--ring R] [--blocks N] "
               "[--M 1530] [--route import|copy] [--drain fork|none] [--pub-every N] "
               "[--once-blocks N]");
            std::exit(0);
        } else { pl("FATAL|unknown_arg|%s", k.c_str()); std::exit(2); }
    }
    if (a.block.empty() || a.scales.empty()) { pl("FATAL|need --block and --scales"); std::exit(2); }
    if (a.blocks < 1) a.blocks = 1;
    if (a.pub_every < 1) a.pub_every = 1;
    if (a.once_blocks == 0 || a.once_blocks > a.blocks) a.once_blocks = a.blocks;
    if (a.drain == "fork" && a.blocks <= hv::SLOTS) a.drain = "fork";  // harmless
    return a;
}

static std::string read_text(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return std::string();
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s((size_t)n, '\0');
    if (fread(&s[0], 1, (size_t)n, f) != (size_t)n) s.clear();
    fclose(f);
    return s;
}

struct MapF {
    int fd = -1;
    unsigned char* base = nullptr;
    size_t bytes = 0;
    bool ok() const { return base != nullptr; }
};

// O_RDWR + PROT_READ|PROT_WRITE is the combination Gate 2 proved for route B
// (probe2.cpp `FILE_CL`). A read-only mapping imports with CL_OUT_OF_HOST_MEMORY
// (-6) on this driver, measured 2026-11-15 22:23 — which is why this is rw even
// though the stage never writes a weight byte.
static MapF map_ro(const std::string& path) {
    MapF m;
    m.fd = ::open(path.c_str(), O_RDWR);
    if (m.fd < 0) m.fd = ::open(path.c_str(), O_RDONLY);
    if (m.fd < 0) return m;
    struct stat st {};
    if (fstat(m.fd, &st) != 0) { ::close(m.fd); m.fd = -1; return m; }
    m.bytes = (size_t)st.st_size;
    void* p = mmap(nullptr, m.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, 0);
    if (p == MAP_FAILED) { ::close(m.fd); m.fd = -1; return m; }
    m.base = static_cast<unsigned char*>(p);
    return m;
}

// ------------------------------------------------------------------ drain
// Stands in for the NPU stage (P3.3). It exists so the producer's slot-reuse
// back-pressure is exercised by a real second PID, and so publish->consume is
// measured rather than assumed. It does not decode the payload (that is P3.3).
struct DrainStats {
    std::uint64_t chunks = 0, late = 0, zero_words = 0, nan_words = 0;
    std::uint64_t lat_min = ~0ull, lat_max = 0, lat_sum = 0;
    std::uint64_t hash = 1469598103934665603ull;
};

static int run_drain(const Args& a, hv::Ring& r, DrainStats* st) {
    const int npub = (a.blocks + a.pub_every - 1) / a.pub_every;
    for (int chunk = 1; chunk <= npub; ++chunk) {
        const int s = (chunk - 1) % hv::SLOTS;
        std::uint64_t t_seen = 0;
        if (!hv::wait_publish(r, s, (std::uint64_t)chunk, 120ull * 1'000'000'000ull, &t_seen)) {
            pl("FATAL|drain|chunk_%d_never_published", chunk);
            return 2;
        }
        hv::SlotHdr* h = r.hdr(s);
        const std::uint32_t bytes = __atomic_load_n(&h->bytes, __ATOMIC_RELAXED);
        const std::uint64_t t_prod = __atomic_load_n(&h->t_prod_ns, __ATOMIC_RELAXED);
        const std::uint16_t* w = reinterpret_cast<const std::uint16_t*>(r.payload(s));
        const size_t n = bytes / 2;
        std::uint64_t z = 0, nan = 0;
        for (size_t i = 0; i < n; ++i) {
            const std::uint16_t v = w[i];
            if (v == 0) ++z;
            if ((v & 0x7C00u) == 0x7C00u && (v & 0x3FFu)) ++nan;
            st->hash = (st->hash ^ v) * 1099511628211ull;
        }
        const std::uint64_t lat = t_seen > t_prod ? t_seen - t_prod : 0;
        st->lat_sum += lat;
        if (lat < st->lat_min) st->lat_min = lat;
        if (lat > st->lat_max) st->lat_max = lat;
        st->zero_words += z;
        st->nan_words += nan;
        if (t_seen <= t_prod) ++st->late;
        ++st->chunks;
        pl("TS|drain|chunk=%d|slot=%d|lat_ns=%" PRIu64 "|zero_words=%" PRIu64 "|nan_words=%" PRIu64,
           chunk, s, lat, z, nan);
        hv::release(r, s, (std::uint64_t)chunk);
    }
    pl("SUMMARY|drain|chunks=%" PRIu64 "|lat_min_us=%.3f|lat_mean_us=%.3f|lat_max_us=%.3f"
       "|late=%" PRIu64 "|nan_words=%" PRIu64 "|payload_hash=%016" PRIx64,
       st->chunks, (st->lat_min == ~0ull ? 0.0 : st->lat_min / 1e3),
       st->chunks ? (double)st->lat_sum / st->chunks / 1e3 : 0.0, st->lat_max / 1e3,
       st->late, st->nan_words, st->hash);
    return (st->late || st->nan_words) ? 1 : 0;
}

// ------------------------------------------------------------------- main
struct Buf {
    cl_mem A = nullptr, C = nullptr, Craw = nullptr, sw = nullptr, U = nullptr, Bl = nullptr;
    size_t a_bytes = 0, c_bytes = 0;
};

// W for one GEMM: imported view of the mmap (zero copy) or a host copy.
static cl_mem make_w(CL* c, const MapF& map, const GemmSpec& g, const std::string& route,
                     int* imported) {
    cl_int err = 0;
    cl_mem m = nullptr;
    if (route == "import" && c->ImportMemoryARM) {
        intptr_t props[5] = {CL_IMPORT_TYPE_ARM, (intptr_t)CL_IMPORT_TYPE_HOST_ARM, 0};
        // try the Gate-2 flag first, then read-only; report which one the driver took
        int e_rw = 0;
        m = c->ImportMemoryARM(c->ctx, CL_MEM_READ_WRITE, props, map.base + g.off, g.nbytes, &e_rw);
        if (!m) {
            m = c->ImportMemoryARM(c->ctx, CL_MEM_READ_ONLY, props, map.base + g.off, g.nbytes, &err);
            if (m) pl("IMPORT|%s|flags=read_only|rw_err=%d", g.name, e_rw);
        }
        if (m) { *imported = 1; return m; }
        pl("WARN|import_failed|%s|err=%d(%s)|rw_err=%d|falling_back_to_copy", g.name, err, cl_err(err), e_rw);
    }
    *imported = 0;
    m = c->CreateBuffer(c->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.nbytes,
                        map.base + g.off, &err);
    if (!m) pl("FATAL|W_create_failed|%s|err=%d", g.name, err);
    return m;
}

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const std::string src = read_text(a.cl.c_str());
    if (src.empty()) { pl("FATAL|kernel_source_missing|%s", a.cl.c_str()); return 2; }

    MapF blk = map_ro(a.block);
    if (!blk.ok()) { pl("FATAL|cannot_map_block|%s", a.block.c_str()); return 2; }
    MapF scl = map_ro(a.scales);
    if (!scl.ok()) { pl("FATAL|cannot_map_scales|%s", a.scales.c_str()); return 2; }

    std::vector<GemmSpec> tab;
    if (build_table(a.M, &tab)) { pl("FATAL|layout_mismatch|region=%zu|scales=%zu", blk.bytes, scl.bytes); return 2; }
    if (blk.bytes != 46'399'488u || scl.bytes != 91'136u) {
        pl("FATAL|file_size_mismatch|block=%zu_scales=%zu|expected_46399488_91136",
           blk.bytes, scl.bytes);
        return 2;
    }

    pl("HV|proc_a|block=%s|block_bytes=%zu|scales_bytes=%zu|route=%s|M=%d|blocks=%d|chunk_bytes=%zu",
       a.block.c_str(), blk.bytes, scl.bytes, a.route.c_str(), a.M, a.blocks, hv::CHUNK_BYTES);

    // Thermal gate: this stage drives the GPU, so the same rule as every timed run.
    // (Zones are read by the host runner; here we only stamp the block start.)

    CL c;
    if (!cl_open(&c, a.route.c_str())) return 2;

    const int M = a.M;
    const int Mpad = (M + 63) / 64 * 64;
    const int TM = 8, TN = 4, BM = 64, BN = 64;

    const char* opts = "-DBM=64 -DBN=64 -DBK=32 -DTM=8 -DTN=4";
    cl_int err = 0;
    cl_program prog = nullptr;
    {
        const char* s = src.c_str();
        size_t len = src.size();
        prog = c.CreateProgramWithSource(c.ctx, 1, &s, &len, &err);
        if (!prog) { pl("FATAL|program_create=%d", err); return 2; }
        err = c.BuildProgram(prog, 1, &c.dev, opts, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[8192] = {0};
            size_t g = 0;
            c.GetProgramBuildInfo(prog, c.dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, &g);
            pl("BUILD|FAIL|err=%d|%s", err, log);
            return 2;
        }
        cl_kernel k = c.CreateKernel(prog, "gemm8_tiled", &err);
        size_t psz = 0, wgs = 0;
        if (k) {
            c.GetKernelWorkGroupInfo(k, c.dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(psz), &psz, nullptr);
            c.GetKernelWorkGroupInfo(k, c.dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr);
            c.ReleaseKernel(k);
        }
        pl("BUILD|OK|opts=%s|private_mem=%zu|max_wg=%zu", opts, psz, wgs);
    }

    // --- activation + output buffers (the only bytes this stage copies: real
    //     weights never are). A is synthetic int8 for P3.2 — the point of this
    //     tick is the weight path and the handoff, not image quality (Gate 3).
    Buf attn{}, up{}, dn{};
    {
        const size_t a1536 = (size_t)Mpad * KD;
        const size_t a8960 = (size_t)Mpad * NUP;
        std::vector<signed char> hA(a8960, 0);
        unsigned sd = 12345u;
        for (auto& x : hA) { sd = sd * 1664525u + 1013904223u; x = (signed char)((int)((sd >> 16) & 0xFF) - 128); }
        cl_mem A1536 = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, a1536, hA.data(), &err);
        cl_mem A8960 = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, a8960, hA.data(), &err);
        if (!A1536 || !A8960) { pl("FATAL|act_alloc|%d", err); return 2; }
        attn.A = A1536; attn.a_bytes = a1536;
        dn.A = A8960;   dn.a_bytes = a8960;
        up.A = A1536;   up.a_bytes = a1536;     // ffn.up shares the [M,1536] activation

        attn.C = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, (size_t)Mpad * KD * sizeof(float), nullptr, &err);
        up.C = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, (size_t)Mpad * NUP * sizeof(float), nullptr, &err);
        dn.C = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, (size_t)Mpad * KD * sizeof(float), nullptr, &err);
        attn.Craw = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, (size_t)Mpad * KD * sizeof(int), nullptr, &err);
        up.Craw = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, (size_t)Mpad * NUP * sizeof(int), nullptr, &err);
        dn.Craw = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, (size_t)Mpad * KD * sizeof(int), nullptr, &err);
        cl_int dummy = 0;
        attn.U = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, 16, nullptr, &err);
        attn.Bl = c.CreateBuffer(c.ctx, CL_MEM_READ_WRITE, 16, nullptr, &err);
        (void)dummy;
        if (!attn.C || !up.C || !dn.C || !attn.Craw || !up.Craw || !dn.Craw) {
            pl("FATAL|out_alloc|%d", err);
            return 2;
        }
    }

    // --- per-GEMM weight views + scale buffers + kernels
    const int NG = 10;
    std::vector<cl_mem> Wm(NG, nullptr), swm(NG, nullptr);
    std::vector<cl_kernel> kernels(NG, nullptr);
    int imported_count = 0;
    for (int i = 0; i < NG; ++i) {
        const GemmSpec& g = tab[i];
        int imp = 0;
        Wm[i] = make_w(&c, blk, g, a.route, &imp);
        imported_count += imp;
        if (!Wm[i]) return 2;
        swm[i] = c.CreateBuffer(c.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                (size_t)g.n_scales * sizeof(float), scl.base + g.scale_off, &err);
        if (!swm[i]) { pl("FATAL|scale_alloc|%s|%d", g.name, err); return 2; }
        pl("BUF|W|name=%s|N=%d|K=%d|off=%zu|bytes=%zu|route=%s", g.name, g.N, g.K, g.off,
           g.nbytes, imp ? "import" : "copy");
    }
    pl("W_TOTAL|gems=%d|imported=%d|copied=%d", NG, imported_count, NG - imported_count);

    // --- ring (producer side)
    hv::Ring ring = hv::map_ring(a.ring.c_str(), true);
    if (!ring.ok()) { pl("FATAL|cannot_map_ring|%s", a.ring.c_str()); return 2; }
    hv::reset_headers(ring);
    pl("RING|path=%s|file_bytes=%zu|slots=%d|slot_stride=%zu", a.ring.c_str(), hv::FILE_BYTES,
       hv::SLOTS, hv::SLOT_STRIDE);

    // --- optional drain consumer in a second PID
    DrainStats dst{};
    pid_t dpid = -1;
    if (a.drain == "fork") {
        dpid = fork();
        if (dpid == 0) {
            const int rc = run_drain(a, ring, &dst);
            std::fflush(stdout);
            hv::unmap_ring(ring);
            std::_Exit(rc);
        }
        if (dpid < 0) { pl("FATAL|fork_drain"); return 2; }
    }

    // --- kernels (one per GEMM so args are set once and the loop is dispatch-only)
    for (int i = 0; i < NG; ++i) {
        const GemmSpec& g = tab[i];
        Buf* b = (i == 8) ? &up : (i == 9 ? &dn : &attn);
        cl_kernel k = c.CreateKernel(prog, "gemm8_tiled", &err);
        if (!k) { pl("FATAL|kernel_create|%s|%d", g.name, err); return 2; }
        const int K4 = g.K / 4;
        const float sa = 1.0f / 127.0f;
        const int R = 0, raw_out = 0, use_lora = 0;
        cl_mem Ax = b->A, Wx = Wm[i], Cx = b->C, Craw = b->Craw, swx = swm[i], Ux = attn.U, Blx = attn.Bl;
        c.SetKernelArg(k, 0, sizeof(cl_mem), &Ax);
        c.SetKernelArg(k, 1, sizeof(cl_mem), &Wx);
        c.SetKernelArg(k, 2, sizeof(cl_mem), &Cx);
        c.SetKernelArg(k, 3, sizeof(cl_mem), &Craw);
        c.SetKernelArg(k, 4, sizeof(cl_mem), &swx);
        c.SetKernelArg(k, 5, sizeof(float), &sa);
        c.SetKernelArg(k, 6, sizeof(cl_mem), &Ux);
        c.SetKernelArg(k, 7, sizeof(cl_mem), &Blx);
        c.SetKernelArg(k, 8, sizeof(int), &R);
        c.SetKernelArg(k, 9, sizeof(int), &M);
        c.SetKernelArg(k, 10, sizeof(int), &g.N);
        c.SetKernelArg(k, 11, sizeof(int), &K4);
        c.SetKernelArg(k, 12, sizeof(int), &raw_out);
        c.SetKernelArg(k, 13, sizeof(int), &use_lora);
        kernels[i] = k;
    }

    const int lx = BN / TN, ly = BM / TM;
    auto dispatch = [&](int i, void** ev_out) -> cl_int {
        const GemmSpec& g = tab[i];
        size_t lsz[2] = {(size_t)lx, (size_t)ly};
        size_t gsz[2] = {(((size_t)g.N + BN - 1) / BN) * lx, (((size_t)M + BM - 1) / BM) * ly};
        return c.EnqueueNDRangeKernel(c.q, kernels[i], 2, nullptr, gsz, lsz, 0, nullptr, ev_out);
    };

    // --- warm-up: one full block, not timed (kernels + DVFS settle)
    for (int w = 0; w < a.warmup; ++w) {
        for (int i = 0; i < NG; ++i) dispatch(i, nullptr);
        c.Finish(c.q);
        int probe = 0;
        c.EnqueueReadBuffer(c.q, attn.Craw, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
    }

    // --- closure overhead: what one blocking-read completion barrier costs when it
    //     has nothing to wait for. The per-GEMM timings below each carry one, so the
    //     net per-GEMM number is (raw - overhead) and that is a measurement, not a
    //     model. The `once` pass measures the same block with ONE closure, which is
    //     the apples-to-apples form for the 0.1474 s co-run reference.
    double close_us = 0.0;
    {
        const int NC = 20;
        int probe = 0;
        const double t0 = hv::now_ns() / 1e3;
        for (int i = 0; i < NC; ++i)
            c.EnqueueReadBuffer(c.q, attn.Craw, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
        close_us = (hv::now_ns() / 1e3 - t0) / NC;
        pl("CLOSE|overhead_us=%.1f|n=%d|bytes=%zu", close_us, NC, sizeof(int));
    }

    std::vector<double> block_ms;
    std::uint64_t pub_count = 0, msync_max = 0, wait_max = 0, conv_max = 0;
    int rc = 0;
    const int npub = (a.blocks + a.pub_every - 1) / a.pub_every;
    std::uint16_t* hpay = reinterpret_cast<std::uint16_t*>(ring.payload(0));   // placeholder

    for (int blk_i = 1; blk_i <= a.blocks; ++blk_i) {
        // Publish only every `pub_every` blocks, and always the last one. The cut
        // hands off ONCE per 160 GPU blocks, so per-block publishing was the wrong
        // cadence: P3.4 measured the GPU finishing 2x20 blocks in 0.214 s and then
        // idling 18.4 s (overlap 1.1 %). Slot/seq numbering is by PUBLISH ordinal,
        // which keeps the ring protocol (done >= seq - SLOTS) unchanged.
        const bool is_pub = (blk_i % a.pub_every == 0) || (blk_i == a.blocks);
        const int s = (int)(pub_count % (std::uint64_t)hv::SLOTS);
        std::uint64_t waited = 0;
        if (is_pub) {
            if (!hv::wait_free(ring, s, (std::uint64_t)(pub_count + 1),
                               300ull * 1'000'000'000ull, &waited)) {
                pl("FATAL|slot_never_freed|block=%d|slot=%d|seq=%" PRIu64,
                   blk_i, s, (std::uint64_t)(pub_count + 1));
                rc = 2;
                break;
            }
            if (waited > wait_max) wait_max = waited;
        }

        double t0 = 0, t1 = 0;
        double gsum = 0;
        t0 = (double)hv::now_ns() / 1e6;
        for (int i = 0; i < NG; ++i) {
            const GemmSpec& g = tab[i];
            const double g0 = (double)hv::now_ns() / 1e6;
            void* ev = nullptr;
            cl_int e = dispatch(i, &ev);
            c.Finish(c.q);
            if (ev) {
                c.WaitForEvents(1, (const void**)&ev);
                cl_int status = 12345;
                size_t sb = 0;
                c.GetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, &sb);
                if (status < 0) pl("EXEC_ERR|block=%d|%s|status=%d(%s)", blk_i, g.name, status, cl_err(status));
                c.ReleaseEvent(ev);
            }
            // clFinish is not a completion barrier on Mali r54p2: close the timing
            // with a blocking read of the raw accumulator buffer (paid-for trap).
            int probe = 0;
            cl_int rd = c.EnqueueReadBuffer(c.q, ((i == 8) ? up.Craw : (i == 9 ? dn.Craw : attn.Craw)),
                                            CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
            const double g1 = (double)hv::now_ns() / 1e6;
            const double ms = g1 - g0;
            gsum += ms;
            const double net = ms - close_us / 1000.0;
            const double gops = 2.0 * (double)M * g.N * g.K / (ms * 1e6);
            const double gops_net = 2.0 * (double)M * g.N * g.K / (net * 1e6);
            pl("GEMM|block=%d|name=%s|M=%d|N=%d|K=%d|ms=%.3f|ms_net=%.3f|gops=%.2f|gops_net=%.2f"
               "|enq=%d|rd=%d",
               blk_i, g.name, M, g.N, g.K, ms, net, gops, gops_net, e, rd);
            if (e || rd) { pl("FATAL|gemm_failed|%s|enq=%d|rd=%d", g.name, e, rd); rc = 1; }
        }
        t1 = (double)hv::now_ns() / 1e6;

        // Dequantised ffn.down output -> fp16 residual into the slot (the payload
        // the NPU stage consumes). Timed, because it is work the GPU stage owns.
        // Only on a publish block: with pub_every > 1 the chunk boundary is the
        // group's last block, and converting on every block would be work the
        // pipeline never does.
        double payload_ms = 0.0;
        long msync_ns = 0;
        if (is_pub) {
            const double c0 = (double)hv::now_ns() / 1e6;
            std::vector<float> host_c((size_t)M * KD);
            c.EnqueueReadBuffer(c.q, dn.C, CL_TRUE, 0, host_c.size() * sizeof(float), host_c.data(),
                                0, nullptr, nullptr);
            hpay = reinterpret_cast<std::uint16_t*>(ring.payload(s));
            for (size_t i = 0; i < host_c.size(); ++i) hpay[i] = f2h(host_c[i]);
            const double c1 = (double)hv::now_ns() / 1e6;
            payload_ms = c1 - c0;
            const std::uint64_t conv_us = (std::uint64_t)((c1 - c0) * 1000.0);
            if (conv_us > conv_max) conv_max = conv_us;

            hv::publish(ring, s, (std::uint64_t)(pub_count + 1), (std::uint32_t)hv::CHUNK_BYTES,
                        hv::Msync::FirstPage, &msync_ns);
            if (msync_ns > 0 && (std::uint64_t)msync_ns > msync_max) msync_max = (std::uint64_t)msync_ns;
            ++pub_count;
        }

        const double bms = t1 - t0;
        block_ms.push_back(bms);
        if (is_pub) {
            pl("TS|gpu|block=%d|slot=%d|t0_ms=%.3f|t1_ms=%.3f|block_ms=%.3f|gemm_ms_sum=%.3f"
               "|payload_ms=%.3f|msync_ns=%ld|slot_wait_us=%" PRIu64 "|t_pub_ns=%" PRIu64
               "|pub=1|chunk=%" PRIu64,
               blk_i, s, t0, t1, bms, gsum, payload_ms, msync_ns, waited / 1000, hv::now_ns(),
               (std::uint64_t)pub_count);
        } else {
            pl("TS|gpu|block=%d|t0_ms=%.3f|t1_ms=%.3f|block_ms=%.3f|gemm_ms_sum=%.3f|pub=0",
               blk_i, t0, t1, bms, gsum);
        }
        if (rc) break;
    }

    __atomic_store_n(&ring.hdr((npub - 1) % hv::SLOTS)->flags, hv::FLAG_STREAM_DONE,
                     __ATOMIC_RELEASE);

    // --- reference-equivalent pass: the same 10 dispatches with ONE closure per
    //     block (the form the 0.1474 s co-run base block was measured in). No
    //     handoff here — this isolates the compute from the per-GEMM TS stamps.
    std::vector<double> once_ms;
    for (int blk_i = 0; blk_i < a.once_blocks; ++blk_i) {
        int probe = 0;
        const double t0 = hv::now_ns() / 1e6;
        for (int i = 0; i < NG; ++i) dispatch(i, nullptr);
        c.Finish(c.q);
        cl_int rd = c.EnqueueReadBuffer(c.q, attn.Craw, CL_TRUE, 0, sizeof(int), &probe, 0,
                                        nullptr, nullptr);
        const double t1 = hv::now_ns() / 1e6;
        once_ms.push_back(t1 - t0);
        pl("TSC|gpu|block=%d|ms_once=%.3f|rd=%d", blk_i + 1, t1 - t0, rd);
        if (rd) { pl("FATAL|once_closure|rd=%d", rd); rc = 1; }
    }
    {
        std::vector<double> s2 = once_ms;
        std::sort(s2.begin(), s2.end());
        const double med = s2.empty() ? 0.0 : s2[s2.size() / 2];
        pl("SUMMARY|gpu_once|blocks=%d|block_ms_once_min=%.3f|block_ms_once_median=%.3f"
           "|block_ms_once_max=%.3f|gops_once=%.2f|closure_us=%.1f",
           (int)once_ms.size(), s2.empty() ? 0.0 : s2.front(), med,
           s2.empty() ? 0.0 : s2.back(),
           2.0 * ((double)M * (8.0 * KD * KD + 2.0 * KD * NUP)) / (med * 1e6), close_us);
    }

    int drc = 0;
    if (dpid > 0) {
        int stp = 0;
        waitpid(dpid, &stp, 0);
        drc = WIFEXITED(stp) ? WEXITSTATUS(stp) : 128;
    }

    double bmin = 1e30, bsum = 0;
    for (double v : block_ms) { if (v < bmin) bmin = v; bsum += v; }
    std::vector<double> sorted = block_ms;
    std::sort(sorted.begin(), sorted.end());
    const double bmed = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
    const double macs = (double)M * (8.0 * KD * KD + 2.0 * KD * NUP);
    pl("SUMMARY|gpu|blocks=%d|gemms_per_block=%d|block_ms_min=%.3f|block_ms_median=%.3f"
       "|block_ms_mean=%.3f|gops_median=%.2f|payload_bytes=%zu|msync_max_ns=%" PRIu64
       "|payload_conv_max_us=%" PRIu64 "|slot_wait_max_us=%" PRIu64
       "|pub_every=%d|chunks_published=%" PRIu64,
       (int)block_ms.size(), NG, bmin, bmed, block_ms.empty() ? 0.0 : bsum / block_ms.size(),
       2.0 * macs / (bmed * 1e6), hv::CHUNK_BYTES, msync_max, conv_max, wait_max / 1000,
       a.pub_every, pub_count);

    const bool pass = (rc == 0 && drc == 0 && pub_count == (std::uint64_t)npub);
    pl("RESULT|%s|published=%" PRIu64 "/%d|drain_rc=%d|route=%s|imported_weights=%d/%d",
       pass ? "PASS" : "FAIL", pub_count, npub, drc, a.route.c_str(), imported_count, NG);
    for (int i = 0; i < NG; ++i) {
        if (kernels[i]) c.ReleaseKernel(kernels[i]);
        if (Wm[i]) c.ReleaseMemObject(Wm[i]);
        if (swm[i]) c.ReleaseMemObject(swm[i]);
    }
    hv::unmap_ring(ring);
    munmap(blk.base, blk.bytes);
    munmap(scl.base, scl.bytes);
    return pass ? 0 : 1;
}
