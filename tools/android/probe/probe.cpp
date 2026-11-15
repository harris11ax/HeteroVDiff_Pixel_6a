// HeteroVDiff_Pixel_6a — device capability probe (Phase 2, Gate 0)
//
// Answers, empirically and on the actual phone, the questions the HeteroVDiff
// blueprint *assumes*:
//   1. Which accelerators exist and are reachable from a plain NDK binary?
//   2. Does the Mali driver expose CL_ARM_integer_dot_product{,_int8}?
//      What is CL_DEVICE_MAX_MEM_ALLOC_SIZE?
//   3. How large an AHardwareBuffer (BLOB) will gralloc actually give us?
//      (paper: Adreno claims 1024 MiB; gralloc dies at exactly 2^29 B)
//   4. Can the same AHB be mapped by NNAPI *and* imported into OpenCL in the
//      same process => one physical copy, two engines? (Gate 2)
//   5. Does NNAPI enumerate a real accelerator, and how fast is an INT8
//      FULLY_CONNECTED 1536x8960 on each device? (T_N estimate)
//
// Build: tools/android/probe/build.sh    Run: tools/probe/probe_host.py
// Output: KEY=VALUE lines + one JSON blob between ==JSON_BEGIN== / ==JSON_END==
//
// NOTE: AHardwareBuffer_getNativeHandle() is NOT in the public NDK header
// (NDK r27 hardware_buffer.h). We resolve it with dlsym() from libandroid.so.
// The blueprint's export_ahb_fd() as written (returning whatever type the
// compiler guesses) does not compile; this is the working form.

#include <android/hardware_buffer.h>
#include <android/neuralnetworks.h>
#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <sstream>

// ---------------------------------------------------------------- util ----
static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static std::string jstr(const char* s) {   // minimal JSON string escape
    std::string o;
    for (const char* p = s; p && *p; ++p) {
        if (*p == '"' || *p == '\\') { o += '\\'; o += *p; }
        else if (*p == '\n') o += "\\n";
        else if ((unsigned char)*p < 0x20) o += ' ';
        else o += *p;
    }
    return o;
}
static void p(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap); printf("\n");
}

// =========================================================== OpenCL =======
typedef struct _cl_platform_id*   cl_platform_id;
typedef struct _cl_device_id*     cl_device_id;
typedef struct _cl_context*       cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem*           cl_mem;
typedef struct _cl_program*       cl_program;
typedef struct _cl_kernel*        cl_kernel;
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_device_type;
#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_ADDRESS_BITS 0x100D
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_MEM_READ_ONLY  (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_USE_HOST_PTR (1 << 3)

typedef int64_t cl_context_properties;

struct CL {
    cl_int  (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int  (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    cl_int  (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_context (*CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                void*, void*, cl_int*);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int  (*BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
    cl_int  (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_kernel (*CreateKernel)(cl_program, const char*, cl_int*);
    cl_int  (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_mem  (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_int  (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
                                  const void*, cl_uint, const void*, void*);
    cl_int  (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                                    const size_t*, const size_t*, cl_uint, const void*, void*);
    cl_int  (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*,
                                 cl_uint, const void*, void*);
    cl_int  (*Finish)(cl_command_queue);
    void*   (*ImportMemoryARM)(cl_context, cl_mem_flags, const void*, void*, size_t, cl_int*);
    void*   handle;
} g_cl;

#define CL_PROGRAM_BUILD_LOG 0x1183

static bool cl_load(CL* c) {
    memset(c, 0, sizeof(*c));
    const char* names[] = {"libOpenCL.so", "libOpenCL.so.1", "/vendor/lib64/libOpenCL.so",
                           "/vendor/lib/libOpenCL.so"};
    for (const char* n : names) {
        c->handle = dlopen(n, RTLD_NOW | RTLD_GLOBAL);
        if (c->handle) { p("CL_LIB=%s", n); break; }
    }
    if (!c->handle) { p("CL_LIB=NONE"); p("CL_DLERROR=%s", dlerror()); return false; }
#define SYM(f, n) do { *(void**)(&c->f) = dlsym(c->handle, n); if(!c->f) p("CL_MISSING_SYM=%s", n); } while(0)
    SYM(GetPlatformIDs, "clGetPlatformIDs");
    SYM(GetDeviceIDs, "clGetDeviceIDs");
    SYM(GetDeviceInfo, "clGetDeviceInfo");
    SYM(CreateContext, "clCreateContext");
    SYM(CreateCommandQueue, "clCreateCommandQueue");
    SYM(CreateProgramWithSource, "clCreateProgramWithSource");
    SYM(BuildProgram, "clBuildProgram");
    SYM(GetProgramBuildInfo, "clGetProgramBuildInfo");
    SYM(CreateKernel, "clCreateKernel");
    SYM(SetKernelArg, "clSetKernelArg");
    SYM(CreateBuffer, "clCreateBuffer");
    SYM(EnqueueWriteBuffer, "clEnqueueWriteBuffer");
    SYM(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
    SYM(EnqueueReadBuffer, "clEnqueueReadBuffer");
    SYM(Finish, "clFinish");
    *(void**)(&c->ImportMemoryARM) = dlsym(c->handle, "clImportMemoryARM");  // optional
#undef SYM
    return c->GetPlatformIDs && c->GetDeviceIDs && c->GetDeviceInfo;
}

static std::string dev_str(cl_device_id d, cl_uint what, CL* c) {
    size_t sz = 0;
    if (c->GetDeviceInfo(d, what, 0, nullptr, &sz) != CL_SUCCESS || !sz) return "";
    std::string s(sz, '\0');
    if (c->GetDeviceInfo(d, what, sz, &s[0], &sz) != CL_SUCCESS) return "";
    while (!s.empty() && (s.back() == '\0' || s.back() == ' ' || s.back() == '\n')) s.pop_back();
    return s;
}

// INT8 dot-product microbenchmark: C[M,N] = A[M,K] x B[N,K]^T with char4 ops.
// Reports whether cl_arm_integer_dot_product_int8 builds and how many GOPS.
static const char* K_ARM_DOT = R"CLC(
#pragma OPENCL EXTENSION cl_arm_integer_dot_product : enable
__kernel void gemm8(__global const char4* A, __global const char4* B, __global float* C,
                    __global const float* scales, const int K_vec) {
    int col = get_global_id(0), row = get_global_id(1);
    int sum = 0, ao = row * K_vec, bo = col * K_vec;
    for (int k = 0; k < K_vec; ++k) {
        sum += arm_dot(A[ao + k], B[bo + k]);   /* 2-arg: cl_arm_integer_dot_product_int8 */
    }
    C[row * get_global_size(0) + col] = (float)sum * scales[col];
}
)CLC";

static const char* K_ARM_DOT8 = R"CLC(
#pragma OPENCL EXTENSION cl_arm_integer_dot_product : enable
__kernel void gemm8(__global const char4* A, __global const char4* B, __global float* C,
                    __global const float* scales, const int K_vec) {
    int col = get_global_id(0), row = get_global_id(1);
    int sum = 0, ao = row * K_vec, bo = col * K_vec;
    for (int k = 0; k < K_vec; ++k) {
        /* 3-arg accumulate: cl_arm_integer_dot_product_accumulate_int8 */
        sum = arm_dot_acc(A[ao + k], B[bo + k], sum);
    }
    C[row * get_global_size(0) + col] = (float)sum * scales[col];
}
)CLC";

static const char* K_SCALAR = R"CLC(
__kernel void gemm8(__global const char4* A, __global const char4* B, __global float* C,
                    __global const float* scales, const int K_vec) {
    int col = get_global_id(0), row = get_global_id(1);
    int sum = 0, ao = row * K_vec, bo = col * K_vec;
    for (int k = 0; k < K_vec; ++k) {
        char4 a = A[ao + k], b = B[bo + k];
        sum += (int)a.x*(int)b.x + (int)a.y*(int)b.y + (int)a.z*(int)b.z + (int)a.w*(int)b.w;
    }
    C[row * get_global_size(0) + col] = (float)sum * scales[col];
}
)CLC";

static bool try_gemm(CL* c, cl_context ctx, cl_command_queue q, cl_device_id d,
                     const char* src, const char* tag, int M, int N, int K,
                     int iters, double* gops_out) {
    cl_int err = 0;
    cl_program pr = c->CreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    if (!pr) { p("GEMM_%s=PROGRAM_FAIL(%d)", tag, err); return false; }
    err = c->BuildProgram(pr, 1, &d, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[4096] = {0}; size_t got = 0;
        c->GetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, &got);
        p("GEMM_%s=BUILD_FAIL(%d)", tag, err);
        p("GEMM_%s_LOG=%s", tag, jstr(log).c_str());
        return false;
    }
    cl_kernel k = c->CreateKernel(pr, "gemm8", &err);
    if (!k) { p("GEMM_%s=KERNEL_FAIL(%d)", tag, err); return false; }

    int K_vec = K / 4;
    size_t abytes = (size_t)M * K_vec * 4;
    size_t bbytes = (size_t)N * K_vec * 4;
    std::vector<char> A(abytes, 1), B(bbytes, 3);
    std::vector<float> scales(N, 0.0078f), C((size_t)M * N, 0.f);

    cl_mem dA = c->CreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, abytes, A.data(), &err);
    cl_mem dB = c->CreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bbytes, B.data(), &err);
    cl_mem dC = c->CreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)M * N * sizeof(float), nullptr, &err);
    cl_mem dS = c->CreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                N * sizeof(float), scales.data(), &err);

    c->SetKernelArg(k, 0, sizeof(cl_mem), &dA);
    c->SetKernelArg(k, 1, sizeof(cl_mem), &dB);
    c->SetKernelArg(k, 2, sizeof(cl_mem), &dC);
    c->SetKernelArg(k, 3, sizeof(cl_mem), &dS);
    c->SetKernelArg(k, 4, sizeof(int), &K_vec);

    size_t gsz[2] = {(size_t)N, (size_t)M};
    err = c->EnqueueNDRangeKernel(q, k, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { p("GEMM_%s=ENQUEUE_FAIL(%d)", tag, err); return false; }
    c->Finish(q);

    double t0 = now_ms();
    for (int i = 0; i < iters; ++i)
        c->EnqueueNDRangeKernel(q, k, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
    c->Finish(q);
    double ms = now_ms() - t0;
    c->EnqueueReadBuffer(q, dC, 1, 0, sizeof(float), &C[0], 0, nullptr, nullptr);
    double ops = 2.0 * M * N * K * iters;
    double gops = ops / (ms * 1e6);
    p("GEMM_%s=OK ms_per_call=%.2f GOPS=%.1f out0=%.1f", tag, ms / iters, gops, C[0]);
    *gops_out = gops;
    return true;
}

static void section_opencl() {
    CL c;
    if (!cl_load(&c)) return;
    cl_uint np = 0;
    if (c.GetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) { p("CL_PLATFORMS=0"); return; }
    std::vector<cl_platform_id> plats(np);
    c.GetPlatformIDs(np, plats.data(), nullptr);
    p("CL_PLATFORMS=%u", np);
    for (cl_uint i = 0; i < np; ++i) {
        cl_uint nd = 0;
        if (c.GetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS) continue;
        std::vector<cl_device_id> devs(nd);
        c.GetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
        for (cl_uint j = 0; j < nd; ++j) {
            cl_device_id d = devs[j];
            std::string name = dev_str(d, CL_DEVICE_NAME, &c);
            std::string ver = dev_str(d, CL_DEVICE_VERSION, &c);
            std::string ext = dev_str(d, CL_DEVICE_EXTENSIONS, &c);
            cl_ulong maxalloc = 0, gmem = 0; cl_uint cu = 0, bits = 0;
            c.GetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc), &maxalloc, nullptr);
            c.GetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
            c.GetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
            c.GetDeviceInfo(d, CL_DEVICE_ADDRESS_BITS, sizeof(bits), &bits, nullptr);
            p("CL_DEV[%u].NAME=%s", j, jstr(name.c_str()).c_str());
            p("CL_DEV[%u].VERSION=%s", j, jstr(ver.c_str()).c_str());
            p("CL_DEV[%u].MAX_COMPUTE_UNITS=%u", j, cu);
            p("CL_DEV[%u].ADDRESS_BITS=%u", j, bits);
            p("CL_DEV[%u].GLOBAL_MEM_MiB=%.1f", j, gmem / 1048576.0);
            p("CL_DEV[%u].MAX_MEM_ALLOC_MiB=%.1f", j, maxalloc / 1048576.0);
            p("CL_DEV[%u].EXTENSIONS=%s", j, jstr(ext.c_str()).c_str());
            p("CL_DEV[%u].HAS_ARM_DOT=%d", j,
              ext.find("cl_arm_integer_dot_product") != std::string::npos ? 1 : 0);
            p("CL_DEV[%u].HAS_ARM_DOT_INT8=%d", j,
              ext.find("cl_arm_integer_dot_product_int8") != std::string::npos ? 1 : 0);
            p("CL_DEV[%u].HAS_ARM_IMPORT_MEMORY=%d", j,
              ext.find("cl_arm_import_memory") != std::string::npos ? 1 : 0);

            if (!c.CreateContext) continue;
            cl_int err = 0;
            cl_context ctx = c.CreateContext(nullptr, 1, &d, nullptr, nullptr, &err);
            if (!ctx) { p("CL_DEV[%u].CONTEXT_FAIL=%d", j, err); continue; }
            cl_command_queue q = c.CreateCommandQueue(ctx, d, 0, &err);
            if (!q) { p("CL_DEV[%u].QUEUE_FAIL=%d", j, err); continue; }

            // M = 128 tokens, N = 8960 (FFN up), K = 1536  -> one FFN GEMM of one DiT block
            double g = 0;
            if (!try_gemm(&c, ctx, q, d, K_ARM_DOT, "ARMDOT", 128, 8960, 1536, 3, &g))
                if (!try_gemm(&c, ctx, q, d, K_ARM_DOT8, "ARMDOT_INT8", 128, 8960, 1536, 3, &g))
                    try_gemm(&c, ctx, q, d, K_SCALAR, "SCALAR", 128, 8960, 1536, 3, &g);
        }
    }
}

// ======================================================= AHardwareBuffer ===
typedef const void* (*getNativeHandle_t)(const AHardwareBuffer*);

static int ahb_dmabuf_fd(AHardwareBuffer* ahb, getNativeHandle_t fn) {
    if (!fn) return -1;
    const void* nh = fn(ahb);
    if (!nh) return -1;
    // native_handle_t { int version; int numFds; int numInts; int data[numFds+numInts]; }
    const int* raw = (const int*)nh;
    if (raw[1] < 1) return -1;
    return raw[3];
}

static long fd_size(int fd) {
    char path[64]; snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", fd);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[256]; long size = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "size:", 5) == 0) { size = atol(line + 5); break; }
    }
    fclose(f);
    return size;
}

static bool ahb_alloc(size_t bytes, AHardwareBuffer** out, getNativeHandle_t fn,
                      int* fd_out, long* fd_size_out) {
    AHardwareBuffer_Desc desc = {};
    desc.width = (uint32_t)bytes;
    desc.height = 1;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                 AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    *fd_out = -1; *fd_size_out = -1;
    if (AHardwareBuffer_allocate(&desc, out) != 0) return false;
    *fd_out = ahb_dmabuf_fd(*out, fn);
    if (*fd_out >= 0) *fd_size_out = fd_size(*fd_out);
    return true;
}

static void section_ahb() {
    getNativeHandle_t gnh = (getNativeHandle_t)dlsym(RTLD_DEFAULT, "AHardwareBuffer_getNativeHandle");
    if (!gnh) gnh = (getNativeHandle_t)dlsym(dlopen("libandroid.so", RTLD_NOW), "AHardwareBuffer_getNativeHandle");
    p("AHB_GETNATIVEHANDLE_SYMBOL=%d", gnh ? 1 : 0);

    // paper §3.2: gralloc refused a BLOB at exactly 2^29 B on Snapdragon; find the Pixel limit.
    const size_t MiB = 1024 * 1024;
    size_t probes[] = {16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768,
                       832, 896, 960, 1024, 1152, 1280, 1536, 1792, 2048};
    size_t max_ok = 0;
    for (size_t m : probes) {
        AHardwareBuffer_Desc desc = {};
        desc.width = (uint32_t)(m * MiB); desc.height = 1; desc.layers = 1;
        desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
        desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                     AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
        int supported = AHardwareBuffer_isSupported(&desc);
        AHardwareBuffer* ahb = nullptr; int fd = -1; long sz = -1;
        bool ok = ahb_alloc(m * MiB, &ahb, gnh, &fd, &sz);
        p("AHB_%zuMiB=supported:%d alloc:%d fd:%d fdsize:%ld", m, supported, ok ? 1 : 0, fd, sz);
        if (ok) {
            if (m > max_ok) max_ok = m;
            AHardwareBuffer_release(ahb);
        } else {
            break;  // first refusal: report it and stop growing
        }
    }
    p("AHB_MAX_OK_MiB=%zu", max_ok);

    // zero-copy gate: same AHB -> NNAPI memory + OpenCL import (Gate 2, partly)
    AHardwareBuffer* ahb = nullptr; int fd = -1; long sz = -1;
    if (ahb_alloc(64 * MiB, &ahb, gnh, &fd, &sz)) {
        ANeuralNetworksMemory* mem = nullptr;
        int rc = ANeuralNetworksMemory_createFromAHardwareBuffer(ahb, &mem);
        p("ZEROCOPY_NNAPI_MEMORY_rc=%d (0=OK)", rc);
        CL c;
        if (cl_load(&c) && c.ImportMemoryARM && c.GetPlatformIDs) {
            cl_uint np = 0; c.GetPlatformIDs(0, nullptr, &np);
            std::vector<cl_platform_id> plats(np); c.GetPlatformIDs(np, plats.data(), nullptr);
            for (cl_uint i = 0; i < np && i < 1; ++i) {
                cl_uint nd = 0; c.GetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
                std::vector<cl_device_id> devs(nd);
                c.GetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
                for (cl_uint j = 0; j < nd; ++j) {
                    cl_int e = 0;
                    cl_context ctx = c.CreateContext(nullptr, 1, &devs[j], nullptr, nullptr, &e);
                    if (!ctx) continue;
                    cl_int ierr = 0;
                    void* m = c.ImportMemoryARM(ctx, CL_MEM_READ_ONLY, nullptr,
                                                (void*)(uintptr_t)fd, 64 * MiB, &ierr);
                    p("ZEROCOPY_CL_IMPORT_dev%u_mem=%s err=%d",
                      j, m ? "OK" : "FAIL", ierr);
                }
            }
        } else {
            p("ZEROCOPY_CL_IMPORT=SKIP (no cl_arm_import_memory / clImportMemoryARM)");
        }
        if (mem) ANeuralNetworksMemory_free(mem);
        AHardwareBuffer_release(ahb);
    }
}

// =============================================================== NNAPI =====
static const char* nn_err(int rc) {
    switch (rc) {
        case ANEURALNETWORKS_NO_ERROR: return "NO_ERROR";
        case ANEURALNETWORKS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case ANEURALNETWORKS_BAD_DATA: return "BAD_DATA";
        case ANEURALNETWORKS_BAD_STATE: return "BAD_STATE";
        case ANEURALNETWORKS_UNMAPPABLE: return "UNMAPPABLE";
        case ANEURALNETWORKS_OP_FAILED: return "OP_FAILED";
        default: return "OTHER";
    }
}

// One INT8 FULLY_CONNECTED: A[M,K] x B[N,K]^T -> C[M,N], per-tensor symmetric.
// NNAPI FC wants B as [N,K] (weights, not transposed input layout).
static void nnapi_fc_on(ANeuralNetworksDevice* dev, int M, int K, int N, int iters) {
    ANeuralNetworksModel* model = nullptr;
    if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) return;

    uint32_t dimA[2] = {(uint32_t)M, (uint32_t)K};
    uint32_t dimB[2] = {(uint32_t)N, (uint32_t)K};
    uint32_t dimC[2] = {(uint32_t)M, (uint32_t)N};
    // symmetric int8: zeroPoint = 0, scale = 1/127 on each tensor
    ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_SYMM, 2, dimA, 1.0f / 127.0f, 0};
    ANeuralNetworksOperandType tB = {ANEURALNETWORKS_TENSOR_QUANT8_SYMM, 2, dimB, 1.0f / 127.0f, 0};
    ANeuralNetworksOperandType tC = {ANEURALNETWORKS_TENSOR_QUANT8_SYMM, 2, dimC, 1.0f / 127.0f, 0};
    ANeuralNetworksOperandType tI = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};

    ANeuralNetworksModel_addOperand(model, &tA);
    ANeuralNetworksModel_addOperand(model, &tB);
    ANeuralNetworksModel_addOperand(model, &tI);   // transpose B flag
    ANeuralNetworksModel_addOperand(model, &tI);   // activation
    ANeuralNetworksModel_addOperand(model, &tI);   // activation params (index into values)
    ANeuralNetworksModel_addOperand(model, &tC);

    int32_t one = 1, act = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 2, &one, sizeof(one));
    ANeuralNetworksModel_setOperandValue(model, 3, &act, sizeof(act));
    ANeuralNetworksModel_setOperandValue(model, 4, nullptr, 0);

    uint32_t in[] = {0, 1}, out[] = {5};
    ANeuralNetworksModel_identifyInputsAndOutputs(model, 2, in, 1, out);

    // quantisation: symmetric int8, same scale both sides
    ANeuralNetworksModel_finish(model);

    // symmetric quantisation is expressed through the operand type at compile time
    ANeuralNetworksCompilation* comp = nullptr;
    int rc = (dev ? ANeuralNetworksCompilation_createForDevices(model, &dev, 1, &comp)
                  : ANeuralNetworksCompilation_create(model, &comp));
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        p("NNAPI_FC[%dx%dx%d]=COMPILE_CREATE_rc=%d(%s)", M, K, N, rc, nn_err(rc));
        ANeuralNetworksModel_free(model);
        return;
    }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    rc = ANeuralNetworksCompilation_finish(comp);
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        p("NNAPI_FC[%dx%dx%d]=COMPILE_FINISH_rc=%d(%s)", M, K, N, rc, nn_err(rc));
        ANeuralNetworksCompilation_free(comp); ANeuralNetworksModel_free(model);
        return;
    }
    p("NNAPI_FC[%dx%dx%d]=COMPILED", M, K, N);

    ANeuralNetworksExecution* ex = nullptr;
    ANeuralNetworksExecution_create(comp, &ex);
    std::vector<int8_t> A((size_t)M * K, 1), B((size_t)N * K, 2), C((size_t)M * N, 0);
    ANeuralNetworksExecution_setInput(ex, 0, nullptr, A.data(), A.size());
    ANeuralNetworksExecution_setInput(ex, 1, nullptr, B.data(), B.size());
    ANeuralNetworksExecution_setOutput(ex, 0, nullptr, C.data(), C.size());

    double t0 = now_ms();
    int okc = 0;
    for (int i = 0; i < iters; ++i) {
        rc = ANeuralNetworksExecution_compute(ex);
        if (rc != ANEURALNETWORKS_NO_ERROR) { p("NNAPI_FC_EXEC_rc=%d(%s)", rc, nn_err(rc)); break; }
        okc++;
    }
    double ms = now_ms() - t0;
    if (okc) p("NNAPI_FC[%dx%dx%d]=OK ms_per_call=%.3f", M, K, N, ms / okc);
    ANeuralNetworksExecution_free(ex);
    ANeuralNetworksCompilation_free(comp);
    ANeuralNetworksModel_free(model);
}

static void section_nnapi() {
    uint32_t n = 0;
    int rc = ANeuralNetworks_getDeviceCount(&n);
    p("NNAPI_DEVICE_COUNT_rc=%d count=%u", rc, n);
    for (uint32_t i = 0; i < n; ++i) {
        ANeuralNetworksDevice* dev = nullptr;
        if (ANeuralNetworks_getDevice(i, &dev) != ANEURALNETWORKS_NO_ERROR) continue;
        const char* name = nullptr; int32_t type = -1; int64_t fl = -1;
        const char* ver = nullptr;
        ANeuralNetworksDevice_getName(dev, &name);
        ANeuralNetworksDevice_getType(dev, &type);
        ANeuralNetworksDevice_getFeatureLevel(dev, &fl);
        ANeuralNetworksDevice_getVersion(dev, &ver);
        p("NNAPI_DEV[%u].NAME=%s", i, name ? name : "?");
        p("NNAPI_DEV[%u].TYPE=%d fl=%lld version=%s", i, type, (long long)fl,
          ver ? ver : "?");
        nnapi_fc_on(dev, 32, 1536, 8960, 3);
    }
    p("NNAPI_FC_DEFAULT_PATH:");
    nnapi_fc_on(nullptr, 32, 1536, 8960, 3);
}

// ================================================================ main =====
int main(int argc, char** argv) {
    std::string what = argc > 1 ? argv[1] : "all";
    printf("==JSON_BEGIN==\n");
    if (what == "opencl" || what == "all") section_opencl();
    if (what == "ahb" || what == "all") section_ahb();
    if (what == "nnapi" || what == "all") section_nnapi();
    printf("==JSON_END==\n");
    return 0;
}
