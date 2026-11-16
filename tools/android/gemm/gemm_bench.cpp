// gemm_bench — on-device INT8 GEMM benchmark for Mali-G78 MP20 (Pixel 6a).
//
// Everything here is a measurement: no modelled rates. Emits pipe-delimited lines
// between ==JSON_BEGIN== / ==JSON_END== that tools/gemm/gemm_host.py parses.
//
//   gemm_bench verify                 exactness of tiled/direct vs the int32 reference
//   gemm_bench bench  [v1,v2,...]     GOPS sweep over M x GEMM shape x tile config
//   gemm_bench block  [M] [v] [iters] one DiT block's 10 GEMMs back-to-back, timed
//   gemm_bench all
//
// Kernel source is read from /data/local/tmp/gemm8.cl at run time so kernel edits do
// not need a rebuild of this binary.
//
// Correctness model: both the reference and the tiled kernels are exact integer
// arithmetic, so -- for the same A and W -- their int32 outputs must match
// bit-for-bit. That is the check `verify` performs; float output is only used after
// the integer path is proven.

#include <dlfcn.h>
#include <stdarg.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static void p(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap); printf("\n");
}

typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id*   cl_device_id;
typedef struct _cl_context*     cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem*         cl_mem;
typedef struct _cl_program*     cl_program;
typedef struct _cl_kernel*      cl_kernel;
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_mem_flags;
typedef intptr_t cl_context_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_TRUE 1
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_KERNEL_LOCAL_MEM_SIZE 0x11B1
#define CL_KERNEL_PRIVATE_MEM_SIZE 0x11B4
#define CL_KERNEL_WORK_GROUP_SIZE 0x11B0
#define CL_EVENT_COMMAND_EXECUTION_STATUS 0x11F3
static const char* cl_err(int e) {
    switch (e) {
        case 0: return "SUCCESS"; case -1: return "DEVICE_NOT_FOUND";
        case -2: return "DEVICE_NOT_AVAILABLE"; case -4: return "MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "OUT_OF_RESOURCES"; case -6: return "OUT_OF_HOST_MEMORY";
        case -9: return "BUILD_PROGRAM_FAILURE"; case -11: return "BUILD_PROGRAM_FAILURE";
        case -12: return "MAP_FAILURE"; case -14: return "EXEC_STATUS_ERROR";
        case -15: return "COMPILER_NOT_AVAILABLE"; case -16: return "MEM_OBJECT_FAILURE";
        case -17: return "INVALID_OPERATION"; case -18: return "INVALID_GL_OBJECT";
        case -19: return "INVALID_BUFFER_SIZE"; case -30: return "INVALID_VALUE";
        case -52: return "INVALID_KERNEL_ARGS"; case -54: return "INVALID_WORK_GROUP_SIZE";
        default: return "?";
    }
}

struct CL {
    void* h;
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_context (*CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                void*, void*, cl_int*);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
                                 const void*, cl_uint, const void*, void*);
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
                                void*, cl_uint, const void*, void*);
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

    cl_platform_id plat;
    cl_device_id   dev;
    cl_context     ctx;
    cl_command_queue q;
};

static bool cl_open(CL* c) {
    memset(c, 0, sizeof(*c));
    c->h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!c->h) c->h = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!c->h) { p("FATAL libOpenCL_missing"); return false; }
#define S(f, n) *(void**)(&c->f) = dlsym(c->h, n); if (!c->f) { p("FATAL missing %s", n); return false; }
    S(GetPlatformIDs, "clGetPlatformIDs")
    S(GetDeviceIDs, "clGetDeviceIDs")
    S(GetDeviceInfo, "clGetDeviceInfo")
    S(CreateContext, "clCreateContext")
    S(CreateCommandQueue, "clCreateCommandQueue")
    S(CreateBuffer, "clCreateBuffer")
    S(EnqueueWriteBuffer, "clEnqueueWriteBuffer")
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
    if (c->GetPlatformIDs(0, nullptr, &np) || !np) { p("FATAL no_platform"); return false; }
    cl_platform_id plats[4];
    c->GetPlatformIDs(np > 4 ? 4 : np, plats, nullptr);
    c->plat = plats[0];
    cl_uint nd = 0;
    c->GetDeviceIDs(c->plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
    if (!nd) { p("FATAL no_device"); return false; }
    cl_device_id devs[4];
    c->GetDeviceIDs(c->plat, CL_DEVICE_TYPE_ALL, nd > 4 ? 4 : nd, devs, nullptr);
    c->dev = devs[0];

    cl_int err = 0;
    c->ctx = c->CreateContext(nullptr, 1, &c->dev, nullptr, nullptr, &err);
    if (!c->ctx) { p("FATAL ctx=%d", err); return false; }
    c->q = c->CreateCommandQueue(c->ctx, c->dev, 0, &err);
    if (!c->q) { p("FATAL queue=%d", err); return false; }

    char name[256] = {0}; size_t got = 0;
    c->GetDeviceInfo(c->dev, CL_DEVICE_NAME, sizeof(name) - 1, name, &got);
    cl_uint cus = 0; c->GetDeviceInfo(c->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, &got);
    cl_uint mhz = 0; c->GetDeviceInfo(c->dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, &got);
    p("CL_DEVICE name=%s cus=%u max_clock_mhz=%u", name, cus, mhz);
    return true;
}

// ---------------------------------------------------------------- utilities
static std::string read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return std::string();
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s((size_t)n, '\0');
    if (fread(&s[0], 1, (size_t)n, f) != (size_t)n) s.clear();
    fclose(f);
    return s;
}

struct Variant {
    const char* name;
    const char* opts;
    int bm, bn, bk;
    int kind;   // 0 = tiled(LDS), 1 = direct(global), 2 = naive reference
    int tm, tn; // per-thread register tile (drives the work-group shape)
    cl_program prog;
};

// The 8x8 register tile spills on this GPU (private_mem 112-144 B/thread), so the
// sweep includes smaller tiles: fewer live accumulators, no scratch traffic.
static const Variant VARIANTS[] = {
    {"ref_naive",             "", 0, 0, 0, 2, 0, 0, nullptr},
    {"tiled_64x64x16_t8x8",   "-DBM=64  -DBN=64  -DBK=16",  64, 64, 16, 0, 8, 8, nullptr},
    {"tiled_64x64x16_t4x4",   "-DBM=64  -DBN=64  -DBK=16 -DTM=4 -DTN=4", 64, 64, 16, 0, 4, 4, nullptr},
    {"tiled_64x64x16_t8x4",   "-DBM=64  -DBN=64  -DBK=16 -DTM=8 -DTN=4", 64, 64, 16, 0, 8, 4, nullptr},
    {"tiled_64x64x32_t8x4",   "-DBM=64  -DBN=64  -DBK=32 -DTM=8 -DTN=4", 64, 64, 32, 0, 8, 4, nullptr},
    {"tiled_128x64x16_t8x4",  "-DBM=128 -DBN=64  -DBK=16 -DTM=8 -DTN=4", 128, 64, 16, 0, 8, 4, nullptr},
    {"tiled_64x128x16_t8x4",  "-DBM=64  -DBN=128 -DBK=16 -DTM=8 -DTN=4", 64, 128, 16, 0, 8, 4, nullptr},
    {"direct_64x64_t8x8",     "-DBM=64  -DBN=64  -DBK=32",  64, 64, 32, 1, 8, 8, nullptr},
    {"direct_64x64_t4x4",     "-DBM=64  -DBN=64  -DBK=32 -DTM=4 -DTN=4", 64, 64, 32, 1, 4, 4, nullptr},
};
static const int NVARIANTS = (int)(sizeof(VARIANTS) / sizeof(VARIANTS[0]));
static const int REF_IDX = 0;   // VARIANTS[0] is the naive reference
static Variant g_var[NVARIANTS];

static bool build_variant(CL* c, Variant* v, const std::string& src) {
    const char* s = src.c_str(); size_t len = src.size();
    cl_int err = 0;
    v->prog = c->CreateProgramWithSource(c->ctx, 1, &s, &len, &err);
    if (!v->prog) { p("BUILD|%s|FAIL|create=%d", v->name, err); return false; }
    err = c->BuildProgram(v->prog, 1, &c->dev, v->opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[8192] = {0}; size_t g = 0;
        c->GetProgramBuildInfo(v->prog, c->dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, &g);
        std::string L(log);
        for (auto& ch : L) if (ch == '\n') ch = '|';
        p("BUILD|%s|FAIL|err=%d|%s", v->name, err, L.c_str());
        return false;
    }
    {   // the Mali offline compiler puts spill/warning notes in the build log
        char log[8192] = {0}; size_t g = 0;
        c->GetProgramBuildInfo(v->prog, c->dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, &g);
        std::string L(log);
        for (auto& ch : L) if (ch == '\n') ch = '|';
        if (!L.empty()) p("BUILDLOG|%s|%s", v->name, L.c_str());
    }
    // report local/private memory of the kernels that matter (a spill shows as private mem)
    const char* kn = v->kind == 2 ? "gemm8_ref" : (v->kind == 1 ? "gemm8_direct" : "gemm8_tiled");
    cl_kernel k = c->CreateKernel(v->prog, kn, &err);
    unsigned long long lmem = 0; size_t psz = 0; size_t wgs = 0;
    if (k) {
        c->GetKernelWorkGroupInfo(k, c->dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        c->GetKernelWorkGroupInfo(k, c->dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(psz), &psz, nullptr);
        c->GetKernelWorkGroupInfo(k, c->dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr);
        c->ReleaseKernel(k);
    }
    p("BUILD|%s|OK|local_mem=%llu|private_mem=%zu|max_wg=%zu", v->name, lmem, psz, wgs);
    return true;
}

static void fill_i8(std::vector<signed char>& v, unsigned seed) {
    unsigned s = seed;
    for (auto& x : v) { s = s * 1664525u + 1013904223u; x = (signed char)((int)((s >> 16) & 0xFF) - 128); }
}

// ------------------------------------------------------------------ bench
struct Buf {
    cl_mem A, W, C, Craw, sw, U, Bl;
    size_t a_rows = 0, n_rows = 0, k = 0;
};

static std::vector<signed char> gA, gW;   // host shadows for verify

static bool alloc_shape(CL* c, Buf* b, int Mpad, int N, int K, int R) {
    cl_int err = 0;
    gA.assign((size_t)Mpad * K, 0);
    gW.assign((size_t)N * K, 0);
    fill_i8(gA, 12345u);
    fill_i8(gW, 67890u);
    b->A = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gA.size(), gA.data(), &err);
    if (!b->A) { p("BUF_FAIL A err=%d", err); return false; }
    b->W = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gW.size(), gW.data(), &err);
    if (!b->W) { p("BUF_FAIL W err=%d", err); return false; }
    b->C = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, (size_t)Mpad * N * sizeof(float), nullptr, &err);
    b->Craw = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE,
                               (size_t)Mpad * N * sizeof(int), nullptr, &err);
    std::vector<float> sw((size_t)N, 1.0f / 127.0f);
    b->sw = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sw.size() * sizeof(float), sw.data(), &err);
    b->U = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, 16, nullptr, &err);
    b->Bl = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, 16, nullptr, &err);
    b->a_rows = (size_t)Mpad; b->n_rows = (size_t)N; b->k = (size_t)K;
    return b->C && b->Craw && b->sw && b->U && b->Bl;
}

static void free_shape(CL* c, Buf* b) {
    if (b->A) c->ReleaseMemObject(b->A);
    if (b->W) c->ReleaseMemObject(b->W);
    if (b->C) c->ReleaseMemObject(b->C);
    if (b->Craw) c->ReleaseMemObject(b->Craw);
    if (b->sw) c->ReleaseMemObject(b->sw);
    if (b->U) c->ReleaseMemObject(b->U);
    if (b->Bl) c->ReleaseMemObject(b->Bl);
    memset(b, 0, sizeof(*b));
}

// launch one GEMM; returns kernel ms (best of `iters`), or -1
static double launch(CL* c, Variant* v, Buf* b, int M, int N, int K, int iters,
                     int raw_out, int use_lora, cl_mem craw, cl_mem U, cl_mem Bl, int R) {
    cl_int err = 0;
    const char* kn = v->kind == 2 ? "gemm8_ref" : (v->kind == 1 ? "gemm8_direct" : "gemm8_tiled");
    cl_kernel k = c->CreateKernel(v->prog, kn, &err);
    if (!k) { p("KERNEL_FAIL %s err=%d", kn, err); return -1.0; }
    const int K4 = K / 4;
    const float sa = 1.0f / 127.0f;
    cl_mem Ax = b->A, Wx = b->W, Cx = (v->kind == 2) ? craw : b->C, swx = b->sw;
    c->SetKernelArg(k, 0, sizeof(cl_mem), &Ax);
    c->SetKernelArg(k, 1, sizeof(cl_mem), &Wx);
    c->SetKernelArg(k, 2, sizeof(cl_mem), &Cx);
    if (v->kind == 2) {
        c->SetKernelArg(k, 3, sizeof(int), &M);
        c->SetKernelArg(k, 4, sizeof(int), &N);
        c->SetKernelArg(k, 5, sizeof(int), &K4);
        size_t gsz[2] = {(size_t)N, (size_t)M};
        err = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
        c->Finish(c->q);
        int probe = 0;
        c->EnqueueReadBuffer(c->q, craw, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
        if (err) { p("ENQ_FAIL %d", err); c->ReleaseKernel(k); return -1.0; }
        double t0 = now_ms();
        for (int i = 0; i < iters; ++i)
            err = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
        c->Finish(c->q);
        cl_int rerr = c->EnqueueReadBuffer(c->q, craw, CL_TRUE, 0, sizeof(int), &probe,
                                           0, nullptr, nullptr);
        double best = (now_ms() - t0) / (double)iters;
        p("  IT|%s|M=%d|N=%d|iters=%d|ms_per_iter=%.4f|enq=%d|rd=%d|v=%d",
          v->name, M, N, iters, best, err, rerr, probe);
        c->ReleaseKernel(k);
        return err ? -1.0 : best;
    }
    c->SetKernelArg(k, 3, sizeof(cl_mem), &craw);
    c->SetKernelArg(k, 4, sizeof(cl_mem), &swx);
    c->SetKernelArg(k, 5, sizeof(float), &sa);
    cl_mem Ux = U, Blx = Bl;
    c->SetKernelArg(k, 6, sizeof(cl_mem), &Ux);
    c->SetKernelArg(k, 7, sizeof(cl_mem), &Blx);
    c->SetKernelArg(k, 8, sizeof(int), &R);
    c->SetKernelArg(k, 9, sizeof(int), &M);
    c->SetKernelArg(k, 10, sizeof(int), &N);
    c->SetKernelArg(k, 11, sizeof(int), &K4);
    c->SetKernelArg(k, 12, sizeof(int), &raw_out);
    c->SetKernelArg(k, 13, sizeof(int), &use_lora);
    // Work-group = (BN/TN, BM/TM) and each work-group owns exactly one BN x BM
    // output tile, so the grid is ceil(N/BN) x ceil(M/BM) work-groups — NOT
    // round_up(N, lx). The latter dispatched ~8x too many groups and wrote past
    // the output buffer, which the driver reports only as CL_EXEC_STATUS_ERROR
    // (-14) on the next blocking read.
    size_t lx = (size_t)(v->bn / v->tn), ly = (size_t)(v->bm / v->tm);
    size_t gx = (((size_t)N + v->bn - 1) / v->bn) * lx;
    size_t gy = (((size_t)M + v->bm - 1) / v->bm) * ly;
    size_t lsz[2] = {lx, ly};
    size_t gsz[2] = {gx, gy};
    // warm-up dispatch (also proves the kernel is not rejected by the driver)
    err = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, lsz, 0, nullptr, nullptr);
    c->Finish(c->q);
    int probe = 0;
    c->EnqueueReadBuffer(c->q, craw, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
    if (err) { p("ENQ_FAIL %d variant=%s", err, v->name); c->ReleaseKernel(k); return -1.0; }

    // steady state: `iters` dispatches back to back, completion forced by one
    // blocking read at the end (clFinish on this driver returns early).
    double t0 = now_ms();
    void* ev = nullptr;
    for (int i = 0; i < iters; ++i)
        err = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, lsz, 0, nullptr,
                                      (i == iters - 1) ? &ev : nullptr);
    c->Finish(c->q);
    if (ev) {
        c->WaitForEvents(1, (const void**)&ev);
        cl_int status = 12345; size_t sb = 0;
        c->GetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, &sb);
        if (status < 0) p("EXEC_ERR|%s|M=%d|status=%d(%s)", v->name, M, status, cl_err(status));
        c->ReleaseEvent(ev);
    }
    cl_int rerr = c->EnqueueReadBuffer(c->q, craw, CL_TRUE, 0, sizeof(int), &probe,
                                       0, nullptr, nullptr);
    double best = (now_ms() - t0) / (double)iters;
    p("  IT|%s|M=%d|N=%d|iters=%d|ms_per_iter=%.4f|enq=%d|rd=%d|v=%d|gsz=%zux%zu|lsz=%zux%zu",
      v->name, M, N, iters, best, err, rerr, probe, gsz[0], gsz[1], lsz[0], lsz[1]);
    c->ReleaseKernel(k);
    return err ? -1.0 : best;
}

static double gops(int M, int N, int K, double ms) {
    if (ms <= 0) return -1.0;
    return 2.0 * (double)M * (double)N * (double)K / (ms * 1e6);
}

// ------------------------------------------------------------------ verify
static int do_verify(CL* c, const std::string& src) {
    // shape chosen to exercise the M-pad path (1530 -> 1536) and a clean one
    const int K = 1536;
    const int Ms[] = {384, 1530};
    const int N = 256;
    int fails = 0;
    if (!g_var[REF_IDX].prog) build_variant(c, &g_var[REF_IDX], src);
    for (int M : Ms) {
        const int Mpad = (M + 127) / 128 * 128;
        Buf b{}; 
        if (!alloc_shape(c, &b, Mpad, N, K, 128)) { p("VERIFY|setup_fail"); return 1; }
        cl_int err = 0;
        std::vector<int> ref((size_t)M * N, 0), got((size_t)M * N, 0);
        cl_mem craw = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      ref.size() * sizeof(int), ref.data(), &err);
        // reference
        double msr = launch(c, &g_var[REF_IDX], &b, M, N, K, 1, 1, 0, craw, b.U, b.Bl, 128);
        c->EnqueueReadBuffer(c->q, craw, CL_TRUE, 0, ref.size() * sizeof(int), ref.data(), 0, nullptr, nullptr);
        p("VERIFY|ref_naive|M=%d|N=%d|K=%d|ms=%.3f|first=%d", M, N, K, msr, ref[0]);

        for (int vi = 0; vi < NVARIANTS; ++vi) {
            Variant* v = &g_var[vi];
            if (!v->prog || v->kind == 2) continue;
            std::fill(got.begin(), got.end(), 0);
            c->EnqueueWriteBuffer(c->q, craw, CL_TRUE, 0, got.size() * sizeof(int), got.data(), 0, nullptr, nullptr);
            double ms = launch(c, v, &b, M, N, K, 1, 1, 0, craw, b.U, b.Bl, 128);
            c->EnqueueReadBuffer(c->q, craw, CL_TRUE, 0, got.size() * sizeof(int), got.data(), 0, nullptr, nullptr);
            long mism = 0; int first_bad = -1;
            long long sum_ref = 0, sum_got = 0, nz = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                sum_ref += ref[i]; sum_got += got[i];
                if (got[i]) ++nz;
                if (ref[i] != got[i]) { ++mism; if (first_bad < 0) first_bad = (int)i; }
            }
            p("VERIFY|%s|M=%d|N=%d|K=%d|%s|mismatch=%ld|first_bad=%d|ref=%d|got=%d|ms=%.3f"
              "|sum_ref=%lld|sum_got=%lld|nonzero=%lld|ref0=%d|got0=%d",
              v->name, M, N, K, mism == 0 ? "MATCH" : "MISMATCH", mism, first_bad,
              first_bad >= 0 ? ref[first_bad] : 0, first_bad >= 0 ? got[first_bad] : 0, ms,
              sum_ref, sum_got, nz, ref[0], got[0]);
            if (mism) ++fails;
        }
        c->ReleaseMemObject(craw);
        free_shape(c, &b);
    }
    p("VERIFY_SUMMARY|fails=%d", fails);
    return fails ? 1 : 0;
}

// ------------------------------------------------------------------- bench
struct Shape { const char* name; int K, N; };
static const Shape SHAPES[] = {
    {"attn",       1536, 1536},
    {"ffn_up",     1536, 8960},
    {"ffn_down",   8960, 1536},
    // The detail expert's rank-128 delta, if its factors were stored INT8
    // (114.5 MiB) rather than FP16 (229.0 MiB) — the two candidate costs.
    {"lora_up8",   1536, 128},
    {"lora_dn8",   128,  1536},
};
static const int NSHAPES = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

static void do_bench(CL* c, const std::vector<int>& which, int iters) {
    const int Ms[] = {128, 384, 768, 1530};
    for (int si = 0; si < NSHAPES; ++si) {
        const Shape& sh = SHAPES[si];
        const int Mpad = 1536;
        Buf b{};
        if (!alloc_shape(c, &b, Mpad, sh.N, sh.K, 128)) { p("BENCH|setup_fail|%s", sh.name); continue; }
        for (int M : Ms) {
            for (int vi : which) {
                Variant* v = &g_var[vi];
                if (!v->prog) continue;
                // the naive reference is only run at the small M (it is the floor, not the rate)
                if (v->kind == 2 && M > 384) continue;
                const int it = v->kind == 2 ? 2 : iters;
                double ms = launch(c, v, &b, M, sh.N, sh.K, it, 0, 0, b.Craw, b.U, b.Bl, 128);
                p("GEMM|%s|%s|M=%d|N=%d|K=%d|ms=%.4f|gops=%.2f",
                  v->name, sh.name, M, sh.N, sh.K, ms, gops(M, sh.N, sh.K, ms));
            }
        }
        free_shape(c, &b);
    }
}

// ------------------------------------------------------------- block timing
// One DiT block at Preset A = 8 x attn GEMMs (K=N=1536) + FFN up (N=8960) +
// FFN down (K=8960), all at the same M. The kernels are built once and the 10
// dispatches are timed as one sequence, so the number is a measurement of a
// block, not a sum of best-cases and not double-counted warm-up.
static cl_kernel make_gemm_kernel(CL* c, Variant* v, Buf* b, int M, int N, int K,
                                  int raw_out, int use_lora, cl_mem craw,
                                  cl_mem U, cl_mem Bl, int R) {
    cl_int err = 0;
    cl_kernel k = c->CreateKernel(v->prog, "gemm8_tiled", &err);
    if (!k) return nullptr;
    const int K4 = K / 4;
    const float sa = 1.0f / 127.0f;
    cl_mem Ax = b->A, Wx = b->W, Cx = b->C, swx = b->sw, Ux = U, Blx = Bl;
    c->SetKernelArg(k, 0, sizeof(cl_mem), &Ax);
    c->SetKernelArg(k, 1, sizeof(cl_mem), &Wx);
    c->SetKernelArg(k, 2, sizeof(cl_mem), &Cx);
    c->SetKernelArg(k, 3, sizeof(cl_mem), &craw);
    c->SetKernelArg(k, 4, sizeof(cl_mem), &swx);
    c->SetKernelArg(k, 5, sizeof(float), &sa);
    c->SetKernelArg(k, 6, sizeof(cl_mem), &Ux);
    c->SetKernelArg(k, 7, sizeof(cl_mem), &Blx);
    c->SetKernelArg(k, 8, sizeof(int), &R);
    c->SetKernelArg(k, 9, sizeof(int), &M);
    c->SetKernelArg(k, 10, sizeof(int), &N);
    c->SetKernelArg(k, 11, sizeof(int), &K4);
    c->SetKernelArg(k, 12, sizeof(int), &raw_out);
    c->SetKernelArg(k, 13, sizeof(int), &use_lora);
    return k;
}

static cl_int enqueue_tile(CL* c, cl_kernel k, Variant* v, int M, int N) {
    size_t lx = (size_t)(v->bn / v->tn), ly = (size_t)(v->bm / v->tm);
    size_t gx = (((size_t)N + v->bn - 1) / v->bn) * lx;
    size_t gy = (((size_t)M + v->bm - 1) / v->bm) * ly;
    size_t lsz[2] = {lx, ly}, gsz[2] = {gx, gy};
    return c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, lsz, 0, nullptr, nullptr);
}

static void do_block(CL* c, Variant* v, int M, int iters, int lora_mode) {
    const int Mpad = (M + 127) / 128 * 128;
    Buf b_attn{}, b_up{}, b_dn{};
    cl_int err = 0;
    if (!alloc_shape(c, &b_attn, Mpad, 1536, 1536, 128)) { p("BLOCK|setup_fail"); return; }
    if (!alloc_shape(c, &b_up, Mpad, 8960, 1536, 128)) goto done;
    if (!alloc_shape(c, &b_dn, Mpad, 1536, 8960, 128)) goto done;
    {
        cl_mem Dm = nullptr, Ulo = nullptr, Blo = nullptr;
        if (lora_mode == 1)
            Dm = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE,
                                 (size_t)Mpad * 1536 * sizeof(float), nullptr, &err);
        if (lora_mode == 2) {
            Ulo = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, (size_t)Mpad * 128 * sizeof(float), nullptr, &err);
            Blo = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, (size_t)1536 * 128 * sizeof(float), nullptr, &err);
        }
        cl_kernel ka = make_gemm_kernel(c, v, &b_attn, M, 1536, 1536, 0, lora_mode,
                                        b_attn.Craw, lora_mode == 1 ? Dm : Ulo,
                                        lora_mode == 1 ? b_attn.Bl : Blo, 128);
        cl_kernel ku = make_gemm_kernel(c, v, &b_up, M, 8960, 1536, 0, 0,
                                        b_up.Craw, b_up.U, b_up.Bl, 0);
        cl_kernel kd = make_gemm_kernel(c, v, &b_dn, M, 1536, 8960, 0, 0,
                                        b_dn.Craw, b_dn.U, b_dn.Bl, 0);
        if (!ka || !ku || !kd) { p("BLOCK|kernel_fail|%d", err); goto done; }
        // warm-up sequence
        for (int g = 0; g < 8; ++g) enqueue_tile(c, ka, v, M, 1536);
        enqueue_tile(c, ku, v, M, 8960);
        enqueue_tile(c, kd, v, M, 1536);
        c->Finish(c->q);
        int probe = 0;
        c->EnqueueReadBuffer(c->q, b_attn.C, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);

        double best = 1e30;
        for (int it = 0; it < iters; ++it) {
            double t0 = now_ms();
            for (int g = 0; g < 8; ++g) err = enqueue_tile(c, ka, v, M, 1536);
            err = enqueue_tile(c, ku, v, M, 8960);
            err = enqueue_tile(c, kd, v, M, 1536);
            c->Finish(c->q);
            c->EnqueueReadBuffer(c->q, b_attn.C, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
            double t1 = now_ms();
            double dt = t1 - t0;
            if (dt < best) best = dt;
            // Absolute CLOCK_MONOTONIC stamps so a host can prove which samples of
            // this process overlapped an NPU execute in another process (co-run).
            p("TS|gpu|it=%d|lora=%d|t0=%.3f|t1=%.3f|ms=%.3f", it, lora_mode, t0, t1, dt);
        }
        const double macs = (double)M * (8.0 * 1536.0 * 1536.0 + 2.0 * 1536.0 * 8960.0);
        p("BLOCK|%s|M=%d|lora=%d|ms=%.2f|gops=%.2f|gmac=%.3f|enq=%d",
          v->name, M, lora_mode, best, 2.0 * macs / (best * 1e6), macs / 1e9, err);
        c->ReleaseKernel(ka); c->ReleaseKernel(ku); c->ReleaseKernel(kd);
        if (Dm) c->ReleaseMemObject(Dm);
        if (Ulo) c->ReleaseMemObject(Ulo);
        if (Blo) c->ReleaseMemObject(Blo);
    }
done:
    free_shape(c, &b_attn); free_shape(c, &b_up); free_shape(c, &b_dn);
}

// FP16 GEMM rate — the detail expert's LoRA path is FP16 (design lock), so the
// rank-128 delta costs two FP16 GEMMs per attention projection:
//   U  = Xh[M,1536] @ Al[128,1536]^T      (K=1536 -> N=128)
//   delta = U[M,128] @ Bl[1536,128]^T     (K=128  -> N=1536)
// Both shapes are measured; the epilogue add itself is measured separately by
// block mode lora=1 - lora=0.
static void f16_one(CL* c, Variant* v, int M, int N, int K, int iters) {
    cl_int err = 0;
    const int Mpad = (M + 127) / 128 * 128;
    std::vector<short> A((size_t)Mpad * K), W((size_t)N * K);
    unsigned s = 999u;
    for (auto& x : A) { s = s * 1664525u + 1013904223u; x = (short)((int)((s >> 16) & 0x3FF) - 512); }
    for (auto& x : W) { s = s * 1664525u + 1013904223u; x = (short)((int)((s >> 16) & 0x3FF) - 512); }
    cl_mem Am = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, A.size() * 2, A.data(), &err);
    cl_mem Wm = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, W.size() * 2, W.data(), &err);
    cl_mem Cm = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, (size_t)Mpad * N * sizeof(float), nullptr, &err);
    if (!Am || !Wm || !Cm) { p("F16|setup_fail"); return; }
    cl_kernel k = c->CreateKernel(v->prog, "gemm16_tiled", &err);
    if (!k) { p("F16|kernel_fail|%d", err); return; }
    c->SetKernelArg(k, 0, sizeof(cl_mem), &Am);
    c->SetKernelArg(k, 1, sizeof(cl_mem), &Wm);
    c->SetKernelArg(k, 2, sizeof(cl_mem), &Cm);
    int a = M, b = N, kk = K;
    c->SetKernelArg(k, 3, sizeof(int), &a);
    c->SetKernelArg(k, 4, sizeof(int), &b);
    c->SetKernelArg(k, 5, sizeof(int), &kk);
    size_t lx = (size_t)(v->bn / 8), ly = (size_t)(v->bm / 8);
    size_t gsz[2] = {(((size_t)N + v->bn - 1) / v->bn) * lx, (((size_t)M + v->bm - 1) / v->bm) * ly};
    size_t lsz[2] = {lx, ly};
    c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, lsz, 0, nullptr, nullptr);
    c->Finish(c->q);
    int probe = 0;
    c->EnqueueReadBuffer(c->q, Cm, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i)
        err = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, lsz, 0, nullptr, nullptr);
    c->Finish(c->q);
    cl_int rd = c->EnqueueReadBuffer(c->q, Cm, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
    double ms = (now_ms() - t0) / (double)iters;
    p("F16|%s|M=%d|N=%d|K=%d|ms=%.4f|gops=%.2f|rd=%d", v->name, M, N, K, ms,
      gops(M, N, K, ms), rd);
    c->ReleaseKernel(k);
    c->ReleaseMemObject(Am); c->ReleaseMemObject(Wm); c->ReleaseMemObject(Cm);
}

static void do_f16(CL* c, const std::vector<int>& which, int iters) {
    for (int vi : which) {
        Variant* v = &g_var[vi];
        if (!v->prog || v->kind != 0) continue;
        f16_one(c, v, 1530, 128, 1536, iters);   // LoRA up   (K=1536 -> R=128)
        f16_one(c, v, 1530, 1536, 128, iters);   // LoRA down (K=128 -> N=1536)
        break;   // one variant is enough for the FP16 rate
    }
}

// ------------------------------------------------------------------- diag
// Two questions, one run:
//   (1) does arm_dot_acc survive an explicitly supplied LOCAL work-group size,
//       and if so which mapping?
//   (2) how many live int accumulators (DTM*DTN) does it tolerate?
static void do_diag(CL* c, const std::string& src) {
    const int M = 128, N = 128, K = 1536;
    const int Mpad = 128;
    Buf b{};
    if (!alloc_shape(c, &b, Mpad, N, K, 128)) { p("DIAG|setup_fail"); return; }
    const int K4 = K / 4;

    struct Cfg { int tm, tn, lx, ly; };   // lx=ly=0 -> driver-chosen local size
    const Cfg cfg[] = {
        {1, 1, 0, 0}, {1, 1, 8, 8}, {1, 1, 64, 1}, {1, 1, 16, 4},
        {4, 4, 8, 8}, {8, 8, 8, 8}, {8, 8, 16, 4},
    };
    for (const Cfg& cf : cfg) {
        char opts[64];
        snprintf(opts, sizeof(opts), "-DDTM=%d -DDTN=%d", cf.tm, cf.tn);
        const char* s = src.c_str(); size_t len = src.size();
        cl_int err = 0;
        cl_program prog = c->CreateProgramWithSource(c->ctx, 1, &s, &len, &err);
        if (!prog) { p("DIAG|%dx%d|program_fail|%d", cf.tm, cf.tn, err); continue; }
        err = c->BuildProgram(prog, 1, &c->dev, opts, nullptr, nullptr);
        if (err) { p("DIAG|%dx%d|build_fail|%d", cf.tm, cf.tn, err); c->ReleaseProgram(prog); continue; }
        cl_kernel k = c->CreateKernel(prog, "dg_dot", &err);
        if (!k) { p("DIAG|kernel_fail|%d", err); c->ReleaseProgram(prog); continue; }
        size_t psz = 0;
        c->GetKernelWorkGroupInfo(k, c->dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(psz), &psz, nullptr);

        cl_mem Ax = b.A, Wx = b.W;
        c->SetKernelArg(k, 0, sizeof(cl_mem), &Ax);
        c->SetKernelArg(k, 1, sizeof(cl_mem), &Wx);
        c->SetKernelArg(k, 2, sizeof(cl_mem), &b.Craw);
        int a = M, bb = N, kk = K4;
        c->SetKernelArg(k, 3, sizeof(int), &a);
        c->SetKernelArg(k, 4, sizeof(int), &bb);
        c->SetKernelArg(k, 5, sizeof(int), &kk);

        size_t lsz[2] = {(size_t)cf.lx, (size_t)cf.ly};
        size_t gsz[2] = {(size_t)((N + cf.tn - 1) / cf.tn),
                         (size_t)((M + cf.tm - 1) / cf.tm)};
        std::vector<int> zero((size_t)M * N, 0), got((size_t)M * N, -7);
        c->EnqueueWriteBuffer(c->q, b.Craw, CL_TRUE, 0, zero.size() * sizeof(int), zero.data(),
                              0, nullptr, nullptr);
        void* ev = nullptr;
        cl_int e1 = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz,
                                            (cf.lx ? lsz : nullptr), 0, nullptr, &ev);
        c->Finish(c->q);
        cl_int rd = c->EnqueueReadBuffer(c->q, b.Craw, CL_TRUE, 0, got.size() * sizeof(int),
                                         got.data(), 0, nullptr, nullptr);
        if (ev) c->ReleaseEvent(ev);
        p("DIAG|tm=%d|tn=%d|lsz=%dx%d|private_mem=%zu|enq=%d|read=%d|g0=%d|g129=%d|g130=%d",
          cf.tm, cf.tn, cf.lx, cf.ly, psz, e1, rd, got[0], got[129], got[130]);
        c->ReleaseKernel(k);
        c->ReleaseProgram(prog);
    }
    free_shape(c, &b);
}


// ------------------------------------------------------------------- peak
// Pure dot-throughput probes. This is the number that decides whether a tiled
// INT8 GEMM on Mali can ever approach the Edge TPU: if arm_dot_acc compiles to a
// real SDOT, dg_peak_dot must be ~4x dg_peak_mac4 (the same 256 MACs done as 256
// scalar multiply-adds); if the two are equal, the builtin is being emulated and
// the GOPS reported by the GEMM kernels are an instruction-issue limit, not a
// tiling limit.
static void do_peak(CL* c, const std::string& src, int iters) {
    const char* s = src.c_str(); size_t len = src.size();
    cl_int err = 0;
    cl_program prog = c->CreateProgramWithSource(c->ctx, 1, &s, &len, &err);
    if (!prog) { p("PEAK|program_fail|%d", err); return; }
    err = c->BuildProgram(prog, 1, &c->dev, "", nullptr, nullptr);
    if (err) { p("PEAK|build_fail|%d", err); c->ReleaseProgram(prog); return; }

    const size_t nthreads = 5120;                    // 80 work-groups of 64
    std::vector<signed char> host_src(1024 * 4);
    unsigned sd = 7u;
    for (auto& x : host_src) { sd = sd * 1664525u + 1013904223u; x = (signed char)((sd >> 16) & 0x3F); }
    cl_mem srcb = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  host_src.size(), host_src.data(), &err);
    cl_mem out = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, nthreads * sizeof(int), nullptr, &err);
    if (!srcb || !out) { p("PEAK|buf_fail|%d", err); return; }

    struct T { const char* name; int macs; };
    const T tests[] = {{"dg_peak_dot", 256}, {"dg_peak_mac4", 256}, {"dg_peak_mad1", 64}};
    for (const T& t : tests) {
        cl_kernel k = c->CreateKernel(prog, t.name, &err);
        if (!k) { p("PEAK|%s|kernel_fail|%d", t.name, err); continue; }
        cl_mem sb = srcb, ob = out;
        int it = iters;
        c->SetKernelArg(k, 0, sizeof(cl_mem), &sb);
        c->SetKernelArg(k, 1, sizeof(cl_mem), &ob);
        c->SetKernelArg(k, 2, sizeof(int), &it);
        size_t gsz = nthreads;
        // warm-up
        c->EnqueueNDRangeKernel(c->q, k, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
        c->Finish(c->q);
        int probe = 0;
        c->EnqueueReadBuffer(c->q, out, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
        double t0 = now_ms();
        for (int r = 0; r < 3; ++r)
            err = c->EnqueueNDRangeKernel(c->q, k, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
        c->Finish(c->q);
        c->EnqueueReadBuffer(c->q, out, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
        double ms = (now_ms() - t0) / 3.0;
        const double macs = (double)nthreads * (double)iters * (double)t.macs;
        p("PEAK|%s|iters=%d|threads=%zu|ms=%.3f|gops=%.2f|gmac_s=%.2f|v=%d",
          t.name, iters, nthreads, ms, 2.0 * macs / (ms * 1e6), macs / (ms * 1e6), probe);
        c->ReleaseKernel(k);
    }
    c->ReleaseMemObject(srcb);
    c->ReleaseMemObject(out);
    c->ReleaseProgram(prog);
}

// ------------------------------------------------------------ VAE conv (T_V)
// The 13 conv classes of the Wan 2.1 decoder at Preset A — the same classes the
// TPU branch in docs/device/RESULTS_vae_tv.md measured, so the two branches are a
// like-for-like comparison. Ordered heaviest class first: the thermal gate is
// checked when the run starts, so the MACs that actually dominate `T_V` are timed
// while the device is still at the floor (same protocol as the T_N re-run).
//
// Measures ONE dispatch per class = all temporal and spatial taps of one output
// frame accumulated (the fused form the GPU can do, which NNAPI cannot). Layer
// cost = ms x frames; class cost = layer x ops_in_class.
struct ConvClass { const char* name; int H, W, Cin, Cout, taps, s, ops, frames; };
static const ConvClass CONV_CLASSES[] = {
    {"conv3^3_96->96_272x480",    272, 480,  96,  96, 3, 3, 6, 9},
    {"conv3^3_192->192_136x240",  136, 240, 192, 192, 3, 3, 6, 9},
    {"conv3^3_384->384_68x120",    68, 120, 384, 384, 3, 3, 5, 5},
    {"conv3^3_384->384_34x60",     34,  60, 384, 384, 3, 3, 10, 3},
    {"conv3x3_192->96_272x480",   272, 480, 192,  96, 1, 3, 1, 9},
    {"conv3x3_384->192_136x240",  136, 240, 384, 192, 1, 3, 1, 9},
    {"conv3^3_192->384_68x120",    68, 120, 192, 384, 3, 3, 1, 5},
    {"tconv3x1_384->768_68x120",   68, 120, 384, 768, 3, 1, 1, 5},
    {"conv3x3_384->192_68x120",    68, 120, 384, 192, 1, 3, 1, 5},
    {"conv3^3_96->3_272x480",     272, 480,  96,   3, 3, 3, 1, 9},
    {"tconv3x1_384->768_34x60",    34,  60, 384, 768, 3, 1, 1, 3},
    {"conv1x1_192->384_68x120",    68, 120, 192, 384, 1, 1, 1, 5},
    {"conv3^3_16->384_34x60",      34,  60,  16, 384, 3, 3, 1, 3},
};
static const int NCONV = (int)(sizeof(CONV_CLASSES) / sizeof(CONV_CLASSES[0]));

// Cin padded to a multiple of 32 (4 channels x KV=8) so a char4 load never
// straddles a pixel; the extra channels and their weights are zero.
static int pad_cin(int cin) { return (cin + 31) / 32 * 32; }

static void do_conv(CL* c, const std::vector<int>& which, int iters, int do_verify) {
    // first LDS variant that built — the conv kernel wants the 32-accumulator
    // 64x64x32 t8x4 shape for the same reason the GEMM does
    Variant* v = nullptr;
    for (int vi : which) if (g_var[vi].prog && g_var[vi].kind == 0) { v = &g_var[vi]; break; }
    if (!v) { p("CONV|no_variant"); return; }
    p("CONV_VARIANT|%s|bm=%d|bn=%d|bk=%d|tm=%d|tn=%d", v->name, v->bm, v->bn, v->bk, v->tm, v->tn);

    double tv_ms = 0.0;
    double tv_macs = 0.0;
    int ncls = 0;
    for (int ci = 0; ci < NCONV; ++ci) {
        const ConvClass& cc = CONV_CLASSES[ci];
        const int CinP = pad_cin(cc.Cin), Cin4 = CinP / 4;
        const int pad = (cc.s == 3) ? 1 : 0;
        const int Wp = cc.W + 2 * pad, Hp = cc.H + 2 * pad;
        const int nst = cc.taps * cc.s * cc.s;
        const int M = cc.H * cc.W;
        const size_t in_bytes = (size_t)cc.taps * Hp * Wp * CinP;
        const size_t w_bytes = (size_t)cc.Cout * nst * CinP;
        const size_t c_bytes = (size_t)M * cc.Cout * sizeof(int);

        std::vector<signed char> hin(in_bytes, 0), hw(w_bytes, 0);
        {   // random int8 in the real channels, zeros everywhere else
            unsigned sd = 4242u + (unsigned)ci;
            for (int t = 0; t < cc.taps; ++t)
                for (int y = 0; y < Hp; ++y)
                    for (int x = 0; x < Wp; ++x) {
                        signed char* px = &hin[(((size_t)t * Hp + y) * Wp + x) * CinP];
                        for (int ch = 0; ch < cc.Cin; ++ch) {
                            sd = sd * 1664525u + 1013904223u;
                            px[ch] = (signed char)((int)((sd >> 16) & 0xFF) - 128);
                        }
                    }
            for (size_t i = 0; i < hw.size(); ++i) {
                sd = sd * 1664525u + 1013904223u;
                hw[i] = (signed char)((int)((sd >> 16) & 0xFF) - 128);
            }
            for (int n = 0; n < cc.Cout; ++n)
                for (int st = 0; st < nst; ++st)
                    for (int ch = cc.Cin; ch < CinP; ++ch)
                        hw[((size_t)n * nst + st) * CinP + ch] = 0;
        }
        cl_int err = 0;
        cl_mem inm = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     in_bytes, hin.data(), &err);
        if (!inm) { p("CONV|%s|ALLOC_FAIL_in|err=%d|bytes=%zu", cc.name, err, in_bytes); continue; }
        cl_mem wm = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    w_bytes, hw.data(), &err);
        cl_mem outm = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE, c_bytes, nullptr, &err);
        if (!wm || !outm) { p("CONV|%s|ALLOC_FAIL|err=%d", cc.name, err);
            c->ReleaseMemObject(inm); if (wm) c->ReleaseMemObject(wm);
            if (outm) c->ReleaseMemObject(outm); continue; }

        cl_kernel k = c->CreateKernel(v->prog, "conv8_taps", &err);
        if (!k) { p("CONV|%s|KERNEL_FAIL|%d", cc.name, err); continue; }
        int aH = cc.H, aW = cc.W, aCin4 = Cin4, aCout = cc.Cout, as = cc.s, at = cc.taps, aM = M;
        c->SetKernelArg(k, 0, sizeof(cl_mem), &inm);
        c->SetKernelArg(k, 1, sizeof(cl_mem), &wm);
        c->SetKernelArg(k, 2, sizeof(cl_mem), &outm);
        c->SetKernelArg(k, 3, sizeof(int), &aH);
        c->SetKernelArg(k, 4, sizeof(int), &aW);
        c->SetKernelArg(k, 5, sizeof(int), &aCin4);
        c->SetKernelArg(k, 6, sizeof(int), &aCout);
        c->SetKernelArg(k, 7, sizeof(int), &as);
        c->SetKernelArg(k, 8, sizeof(int), &at);
        c->SetKernelArg(k, 9, sizeof(int), &aM);
        const size_t lx = (size_t)(v->bn / v->tn), ly = (size_t)(v->bm / v->tm);
        const size_t gx = (((size_t)cc.Cout + v->bn - 1) / v->bn) * lx;
        const size_t gy = (((size_t)M + v->bm - 1) / v->bm) * ly;
        size_t lsz[2] = {lx, ly}, gsz[2] = {gx, gy};

        err = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, lsz, 0, nullptr, nullptr);
        c->Finish(c->q);
        int probe = 0;
        cl_int r0 = c->EnqueueReadBuffer(c->q, outm, CL_TRUE, 0, sizeof(int), &probe, 0, nullptr, nullptr);
        if (err) { p("CONV|%s|ENQ_FAIL|%d", cc.name, err); c->ReleaseKernel(k); continue; }
        if (r0) {   // a faulted dispatch leaves the queue in error and every later
                    // timing is enqueue overhead only — refuse to report it
            p("CONV|%s|FAULT|read=%d(%s)|warmup", cc.name, r0, cl_err(r0));
            c->ReleaseKernel(k); c->ReleaseMemObject(inm); c->ReleaseMemObject(wm);
            c->ReleaseMemObject(outm); continue;
        }

        double t0 = now_ms();
        void* ev = nullptr;
        for (int i = 0; i < iters; ++i)
            err = c->EnqueueNDRangeKernel(c->q, k, 2, nullptr, gsz, lsz, 0, nullptr,
                                          (i == iters - 1) ? &ev : nullptr);
        c->Finish(c->q);
        if (ev) {
            c->WaitForEvents(1, (const void**)&ev);
            cl_int status = 12345; size_t sb = 0;
            c->GetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, &sb);
            if (status < 0) p("CONV_EXEC_ERR|%s|status=%d(%s)", cc.name, status, cl_err(status));
            c->ReleaseEvent(ev);
        }
        cl_int r1 = c->EnqueueReadBuffer(c->q, outm, CL_TRUE, 0, sizeof(int), &probe,
                                         0, nullptr, nullptr);
        const double ms = (now_ms() - t0) / (double)iters;
        if (r1) { p("CONV|%s|FAULT|read=%d(%s)|timed", cc.name, r1, cl_err(r1));
                  c->ReleaseKernel(k); c->ReleaseMemObject(inm); c->ReleaseMemObject(wm);
                  c->ReleaseMemObject(outm); continue; }

        const double macs = (double)M * cc.Cin * cc.Cout * nst;     // real MACs, unpadded
        const double layer_ms = ms * cc.frames;
        const double cls_ms = layer_ms * cc.ops;
        p("CONV|%s|H=%d|W=%d|Cin=%d|CinP=%d|Cout=%d|taps=%d|s=%d|frames=%d|ops=%d"
          "|ms=%.3f|gops=%.2f|layer_ms=%.1f|cls_ms=%.1f|macs_G=%.3f",
          cc.name, cc.H, cc.W, cc.Cin, CinP, cc.Cout, cc.taps, cc.s, cc.frames, cc.ops,
          ms, macs / (ms * 1e6), layer_ms, cls_ms, macs / 1e9);
        tv_ms += cls_ms; tv_macs += macs * cc.frames * cc.ops;
        ++ncls;

        // Exactness: the tiled kernel vs the scalar int32 reference on the SAME
        // buffers. Run on the cheap classes (or a subset is enough) — a 3-tap 3x3
        // case, a 1x1 case and the tiny-Cout tail.
        if (do_verify && macs / 1e9 <= 10.0) {
            cl_kernel kr = c->CreateKernel(v->prog, "conv8_ref", &err);
            if (kr) {
                std::vector<int> got((size_t)M * cc.Cout, 0), ref((size_t)M * cc.Cout, 0);
                c->EnqueueReadBuffer(c->q, outm, CL_TRUE, 0, c_bytes, got.data(), 0, nullptr, nullptr);
                cl_mem rbuf = c->CreateBuffer(c->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              c_bytes, ref.data(), &err);
                int bCin = cc.Cin, bCinP = CinP;
                c->SetKernelArg(kr, 0, sizeof(cl_mem), &inm);
                c->SetKernelArg(kr, 1, sizeof(cl_mem), &wm);
                c->SetKernelArg(kr, 2, sizeof(cl_mem), &rbuf);
                c->SetKernelArg(kr, 3, sizeof(int), &aH);
                c->SetKernelArg(kr, 4, sizeof(int), &aW);
                c->SetKernelArg(kr, 5, sizeof(int), &bCin);
                c->SetKernelArg(kr, 6, sizeof(int), &bCinP);
                c->SetKernelArg(kr, 7, sizeof(int), &aCout);
                c->SetKernelArg(kr, 8, sizeof(int), &as);
                c->SetKernelArg(kr, 9, sizeof(int), &at);
                c->SetKernelArg(kr, 10, sizeof(int), &aM);
                size_t rsz[2] = {(size_t)cc.Cout, (size_t)M};
                double tr0 = now_ms();
                cl_int rerr = c->EnqueueNDRangeKernel(c->q, kr, 2, nullptr, rsz, nullptr, 0, nullptr, nullptr);
                c->Finish(c->q);
                cl_int rd = c->EnqueueReadBuffer(c->q, rbuf, CL_TRUE, 0, c_bytes, ref.data(),
                                                 0, nullptr, nullptr);
                double rms = now_ms() - tr0;
                long mism = 0; int first_bad = -1; long nzg = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    if (got[i]) ++nzg;
                    if (ref[i] != got[i]) { ++mism; if (first_bad < 0) first_bad = (int)i; }
                }
                // a zero output on random input means nothing executed — do not
                // let "0 mismatches between two zero buffers" pass as MATCH
                const char* verdict = (mism == 0 && nzg > 0) ? "MATCH" : "MISMATCH";
                if (rerr || rd || nzg == 0) verdict = "UNVERIFIED";
                p("CONVV|%s|%s|mismatch=%ld|first_bad=%d|ref=%d|got=%d|ms_ref=%.1f|enq=%d|rd=%d|nz_got=%ld",
                  cc.name, verdict, mism, first_bad,
                  first_bad >= 0 ? ref[first_bad] : 0, first_bad >= 0 ? got[first_bad] : 0,
                  rms, rerr, rd, nzg);
                c->ReleaseMemObject(rbuf);
                c->ReleaseKernel(kr);
            }
        }
        c->ReleaseKernel(k);
        c->ReleaseMemObject(inm); c->ReleaseMemObject(wm); c->ReleaseMemObject(outm);
    }
    p("TV_MALI|classes=%d|total_s=%.3f|gops=%.1f|tmacs=%.1f",
      ncls, tv_ms / 1000.0, tv_macs / (tv_ms * 1e6), tv_macs / 1e9);
}


// ------------------------------------------- arbitrary GEMM (T_V attention unit)
// One dispatch of an arbitrary [M,N,K] INT8 GEMM at the block-table variant.
// The VAE mid-block attention is 6120 tokens x 384 ch: QK^T and AV are
// [M,6120,384] and the four projections are [M,384,384]. Same unit of
// measurement as the NPU `attn` mode, so the two branches stay comparable and no
// number here is modelled. Usage: gemm <cl> <variants> <iters> <M> <N> <K>
static void do_gemm_shape(CL* c, const std::vector<int>& which, int M, int N, int K, int iters) {
    const int Mpad = (M + 127) / 128 * 128;
    Buf b{};
    if (!alloc_shape(c, &b, Mpad, N, K, 128)) { p("ATTN|setup_fail"); return; }
    for (int vi : which) {
        Variant* v = &g_var[vi];
        if (!v->prog || v->kind == 2) continue;      // tiled / direct only
        double ms = launch(c, v, &b, M, N, K, iters, 0, 0, b.Craw, b.U, b.Bl, 128);
        p("ATTN_GEMM|%s|M=%d|N=%d|K=%d|Mpad=%d|iters=%d|ms=%.4f|gops=%.2f|per_pass_16t=%.4f",
          v->name, M, N, K, Mpad, iters, ms, gops(M, N, K, ms), ms * 16.0 / 1000.0);
    }
    free_shape(c, &b);
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "all";
    std::string cl_path = argc > 2 ? argv[2] : "/data/local/tmp/gemm8.cl";
    std::string src = read_file(cl_path.c_str());
    printf("==JSON_BEGIN==\n");
    if (src.empty()) { p("FATAL kernel_source_missing path=%s", cl_path.c_str()); printf("==JSON_END==\n"); return 2; }

    CL c;
    if (!cl_open(&c)) { printf("==JSON_END==\n"); return 3; }

    // variant filter: 4th arg, comma separated names (default = all)
    std::vector<int> which;
    if (argc > 3 && argv[3][0]) {
        std::string sel = argv[3];
        size_t pos = 0;
        while (pos <= sel.size()) {
            size_t nx = sel.find(',', pos);
            std::string one = sel.substr(pos, nx == std::string::npos ? std::string::npos : nx - pos);
            for (int i = 0; i < NVARIANTS; ++i) if (one == VARIANTS[i].name) which.push_back(i);
            if (nx == std::string::npos) break;
            pos = nx + 1;
        }
    }
    if (which.empty()) for (int i = 0; i < NVARIANTS; ++i) which.push_back(i);

    memcpy(g_var, VARIANTS, sizeof(VARIANTS));
    if (mode != "diag" && mode != "peak")
        for (int vi : which) build_variant(&c, &g_var[vi], src);

    int iters = argc > 4 ? atoi(argv[4]) : 5;
    if (iters < 1) iters = 1;

    if (mode == "diag") do_diag(&c, src);
    if (mode == "conv") do_conv(&c, which, iters, argc > 6 ? atoi(argv[6]) : 1);
    if (mode == "peak") do_peak(&c, src, argc > 7 ? atoi(argv[7]) : 2000);
    if (mode == "gemm")
        do_gemm_shape(&c, which, argc > 5 ? atoi(argv[5]) : 6120,
                      argc > 6 ? atoi(argv[6]) : 6120,
                      argc > 7 ? atoi(argv[7]) : 384, iters);
    if (mode == "verify" || mode == "all") do_verify(&c, src);
    if (mode == "bench" || mode == "all") do_bench(&c, which, iters);
    if (mode == "f16" || mode == "all") do_f16(&c, which, iters);
    if (mode == "block" || mode == "all") {
        int M = argc > 5 ? atoi(argv[5]) : 1530;
        for (int vi : which) {
            if (!g_var[vi].prog || g_var[vi].kind == 2) continue;
            do_block(&c, &g_var[vi], M, iters, 0);
            do_block(&c, &g_var[vi], M, iters, 1);
            do_block(&c, &g_var[vi], M, iters, 2);
        }
    }
    printf("==JSON_END==\n");
    return 0;
}
