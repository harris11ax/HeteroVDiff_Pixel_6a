// probe2 — Gate 0, round 2. Fixes the two things probe round 1 got wrong.
//
// 1. NNAPI FULLY_CONNECTED is 4 inputs {input, weights[N_out,K], bias[N_out],
//    activation} — bias is MANDATORY and there is no activationParams operand.
//    Round 1 passed 5 inputs -> model_finish failed -> every compile returned
//    BAD_STATE. Also FC does NOT accept TENSOR_QUANT8_SYMM; signed int8 is
//    TENSOR_QUANT8_ASYMM_SIGNED with zeroPoint = 0 (symmetric).
//    Weights go in as [num_units, input_size] = our row-major [N_out, K].
//    Per-output-column scales use TENSOR_QUANT8_SYMM_PER_CHANNEL, channelDim 0.
//
// 2. clImportMemoryARM is not exported by Android's libOpenCL.so loader, so
//    dlsym() on that handle fails even though the driver advertises
//    cl_arm_import_memory. Resolve it from the vendor driver instead
//    (libGLES_mali.so / libOpenCL-pixel.so) and test BOTH zero-copy routes:
//      A) AHardwareBuffer  -> CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM
//      B) MAP_SHARED file  -> CL_IMPORT_TYPE_HOST_ARM  (one page cache, both
//         engines; also tests ANeuralNetworksMemory_createFromFd)
//
// Build/run: tools/android/probe/build.sh && adb shell /data/local/tmp/hv_probe2 nnapi
//            modes: nnapi | import | all

#include <android/hardware_buffer.h>
#include <android/neuralnetworks.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>
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
static const char* nn_rc(int rc) {
    // NeuralNetworksTypes.h ResultCode
    switch (rc) {
        case 0: return "NO_ERROR";
        case 1: return "OUT_OF_MEMORY";
        case 2: return "INCOMPLETE";
        case 3: return "UNEXPECTED_NULL";
        case 4: return "BAD_DATA";
        case 5: return "BAD_STATE";
        case 6: return "OP_FAILED";
        case 8: return "UNMAPPABLE";
        default: return "OTHER";
    }
}

// ======================================================== NNAPI FC test =====
struct FCResult { int ok; double ms; };

// weights_per_channel: 0 = per-tensor scale, 1 = per-output-column scales
static int nnapi_fc(ANeuralNetworksDevice* dev, const char* tag, int M, int K, int N,
                    int per_channel, int iters, const char** why, int weight_src) {
    *why = "";
    ANeuralNetworksModel* model = nullptr;
    int rc = ANeuralNetworksModel_create(&model);
    if (rc) { *why = "model_create"; return -rc; }

    uint32_t dA[2] = {(uint32_t)M, (uint32_t)K};
    uint32_t dW[2] = {(uint32_t)N, (uint32_t)K};   // [num_units, input_size]
    uint32_t dB[1] = {(uint32_t)N};
    uint32_t dO[2] = {(uint32_t)M, (uint32_t)N};

    const float sa = 1.0f / 127.0f;      // activation scale
    const float sw = 1.0f / 127.0f;      // weight scale (per-tensor) / nominal per-channel
    const float sb = sa * sw;            // bias scale must equal input*weights

    ANeuralNetworksOperandType tA = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dA, sa, 0};
    ANeuralNetworksOperandType tW = {per_channel ? ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL
                                                 : ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
                                     2, dW, per_channel ? 0.0f : sw, 0};
    ANeuralNetworksOperandType tB = {ANEURALNETWORKS_TENSOR_INT32, 1, dB, sb, 0};
    ANeuralNetworksOperandType tAct = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
    ANeuralNetworksOperandType tO = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 2, dO, sa, 0};

    ANeuralNetworksModel_addOperand(model, &tA);    // 0
    ANeuralNetworksModel_addOperand(model, &tW);    // 1
    ANeuralNetworksModel_addOperand(model, &tB);    // 2
    ANeuralNetworksModel_addOperand(model, &tAct);  // 3
    ANeuralNetworksModel_addOperand(model, &tO);    // 4

    int32_t fused = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 3, &fused, sizeof(fused));

    std::vector<int32_t> bias((size_t)N, 0);
    ANeuralNetworksModel_setOperandValue(model, 2, bias.data(), bias.size() * sizeof(int32_t));

    // weights as a CONSTANT operand. Two backings, because vendor NPUs differ:
    //   weight_src 0 = file-backed mmap (shared page cache, no AHB needed)
    //   weight_src 1 = AHardwareBuffer (DMA-BUF; what QNN/edge drivers usually demand)
    const size_t wbytes = (size_t)N * K;
    int wfd = -1;
    AHardwareBuffer* wahb = nullptr;
    ANeuralNetworksMemory* wmem = nullptr;
    if (weight_src == 0) {
        wfd = open("/data/local/tmp/hv_w.bin", O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (wfd < 0) { *why = "wfile"; return -1; }
        if (ftruncate(wfd, (off_t)wbytes) != 0) { *why = "wtrunc"; return -1; }
        rc = ANeuralNetworksMemory_createFromFd(wbytes, PROT_READ | PROT_WRITE, wfd, 0, &wmem);
        if (rc) { p("  %s createFromFd rc=%d(%s)", tag, rc, nn_rc(rc)); *why = "wmemory"; return -1; }
    } else {
        AHardwareBuffer_Desc wd = {};
        wd.width = (uint32_t)wbytes; wd.height = 1; wd.layers = 1;
        wd.format = AHARDWAREBUFFER_FORMAT_BLOB;
        wd.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                   AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
        if (AHardwareBuffer_allocate(&wd, &wahb) != 0) { *why = "wahb_alloc"; return -1; }
        void* wh = nullptr;
        if (AHardwareBuffer_lock(wahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &wh) == 0) {
            memset(wh, 1, wbytes);   // small non-zero weights so out0 is meaningful
            AHardwareBuffer_unlock(wahb, nullptr);
        }
        rc = ANeuralNetworksMemory_createFromAHardwareBuffer(wahb, &wmem);
        if (rc) { p("  %s createFromAHB rc=%d(%s)", tag, rc, nn_rc(rc)); *why = "wahb_mem"; return -1; }
    }
    rc = ANeuralNetworksModel_setOperandValueFromMemory(model, 1, wmem, 0, wbytes);
    if (rc) { p("  %s setOperandValueFromMemory rc=%d(%s)", tag, rc, nn_rc(rc));
              *why = "wfrommemory"; return -1; }

    std::vector<float> scales((size_t)N, sw);
    if (per_channel) {
        ANeuralNetworksSymmPerChannelQuantParams pch = {0, (uint32_t)N, scales.data()};
        rc = ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(model, 1, &pch);
        if (rc) { *why = "set_per_channel"; return -rc; }
    }

    // the operation itself (round 2 forgot this; the validator says
    // "!subgraph.operations.empty()" in logcat when you do)
    uint32_t op_in[4] = {0, 1, 2, 3};
    uint32_t op_out[1] = {4};
    rc = ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED,
                                          4, op_in, 1, op_out);
    if (rc) { p("  %s addOperation rc=%d(%s)", tag, rc, nn_rc(rc)); *why = "addOperation"; return -1; }

    uint32_t ins[1] = {0}, outs[1] = {4};
    ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, ins, 1, outs);

    rc = ANeuralNetworksModel_finish(model);
    if (rc) { p("  %s model_finish rc=%d(%s)", tag, rc, nn_rc(rc)); *why = "model_finish"; return -rc; }
    p("  %s model_finish OK", tag);

    if (dev) {
        bool supported = false;
        rc = ANeuralNetworksModel_getSupportedOperationsForDevices(model, &dev, 1, &supported);
        p("  %s getSupportedOperationsForDevices rc=%d told=%d", tag, rc, supported ? 1 : 0);
        if (rc == 0 && !supported) { *why = "device_reports_unsupported"; return -2; }
    }

    ANeuralNetworksCompilation* comp = nullptr;
    rc = dev ? ANeuralNetworksCompilation_createForDevices(model, &dev, 1, &comp)
             : ANeuralNetworksCompilation_create(model, &comp);
    if (rc) { p("  %s comp_create rc=%d(%s)", tag, rc, nn_rc(rc)); *why = "comp_create"; return -rc; }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    rc = ANeuralNetworksCompilation_finish(comp);
    if (rc) { p("  %s comp_finish rc=%d(%s)", tag, rc, nn_rc(rc)); *why = "comp_finish"; return -rc; }

    ANeuralNetworksExecution* ex = nullptr;
    rc = ANeuralNetworksExecution_create(comp, &ex);
    if (rc) { *why = "exec_create"; return -rc; }

    std::vector<int8_t> A((size_t)M * K, 1), O((size_t)M * N, 0);
    ANeuralNetworksExecution_setInput(ex, 0, nullptr, A.data(), A.size());
    ANeuralNetworksExecution_setOutput(ex, 0, nullptr, O.data(), O.size());

    ANeuralNetworksExecution_setMeasureTiming(ex, true);   // enables getDuration
    int good = 0;
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        rc = ANeuralNetworksExecution_compute(ex);
        if (rc) { p("  %s exec rc=%d(%s)", tag, rc, nn_rc(rc)); *why = "exec_compute"; break; }
        ++good;
    }
    double ms = good ? (now_ms() - t0) / good : -1;
    if (good) {
        uint64_t dur = 0;
        if (ANeuralNetworksExecution_getDuration(ex, ANEURALNETWORKS_DURATION_ON_HARDWARE,
                                                 &dur) == 0)
            p("  %s OK ms_wall=%.3f ms_hw=%.3f out0=%d", tag, ms, dur / 1e6, (int)O[0]);
        else
            p("  %s OK ms_wall=%.3f out0=%d", tag, ms, (int)O[0]);
    }
    ANeuralNetworksExecution_free(ex);
    ANeuralNetworksCompilation_free(comp);
    ANeuralNetworksModel_free(model);
    return good ? 1 : -3;
}

static void section_nnapi() {
    uint32_t n = 0;
    p("NNAPI_DEVICE_COUNT_rc=%d count=%u", ANeuralNetworks_getDeviceCount(&n), n);
    std::vector<ANeuralNetworksDevice*> devs(n);
    for (uint32_t i = 0; i < n; ++i) ANeuralNetworks_getDevice(i, &devs[i]);

    // Preset A block shapes: FFN up 1536->8960 and attention 1536->1536.
    const int Ms[] = {32, 128, 256, 512, 768, 1024, 1530};
    char tag[128];
    for (uint32_t i = 0; i < n; ++i) {
        const char* name = "?";
        ANeuralNetworksDevice_getName(devs[i], &name);
        for (int src = 0; src <= 1; ++src) {
            for (int m : Ms) {
                snprintf(tag, sizeof(tag), "FC_%s_%s_M%d", name,
                         src ? "ahb" : "file", m);
                const char* why = "";
                int r = nnapi_fc(devs[i], tag, m, 1536, 8960, 0, 2, &why, src);
                p("RESULT %s -> %d (%s)", tag, r, r > 0 ? "OK" : why);
            }
        }
        // attention-width GEMM (1536 -> 1536) from an AHB, at Preset A M
        snprintf(tag, sizeof(tag), "FC_%s_ahb_attn_M1530", name);
        const char* why = "";
        int r = nnapi_fc(devs[i], tag, 1530, 1536, 1536, 0, 2, &why, 1);
        p("RESULT %s -> %d (%s)", tag, r, r > 0 ? "OK" : why);
    }
}

// ================================================ OpenCL import / zero-copy ==
typedef struct _cl_context* cl_context;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_mem_flags;
typedef intptr_t cl_context_properties;
#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_MEM_READ_WRITE  (1 << 0)
#define CL_MEM_READ_ONLY   (1 << 2)
#define CL_MEM_WRITE_ONLY  (1 << 1)
#define CL_MEM_SVM_FINE_GRAIN_SYSTEM (1 << 10)
#define CL_IMPORT_TYPE_ARM                        0x40B2
#define CL_IMPORT_TYPE_HOST_ARM                   0x40B3
#define CL_IMPORT_TYPE_DMA_BUF_ARM                0x40B4
#define CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM 0x41E2

struct CLH {
    cl_int (*GetDeviceIDs)(void*, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
    cl_int (*GetPlatformIDs)(cl_uint, void**, cl_uint*);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_context (*CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                void*, void*, cl_int*);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_kernel (*CreateKernel)(cl_program, const char*, cl_int*);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                                   const size_t*, const size_t*, cl_uint, const void*, void*);
    cl_int (*Finish)(cl_command_queue);
    cl_mem (*ImportMemoryARM)(cl_context, cl_mem_flags, const void*, void*, size_t, cl_int*);
    void* loader;
    void* vendor;
};

#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_DEVICE_NAME 0x102B

static bool clh_load(CLH* h) {
    memset(h, 0, sizeof(*h));
    h->loader = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h->loader) h->loader = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h->loader) { p("IMPORT_CL_LOADER=missing"); return false; }
#define S(f, n) *(void**)(&h->f) = dlsym(h->loader, n)
    S(GetPlatformIDs, "clGetPlatformIDs"); S(GetDeviceIDs, "clGetDeviceIDs");
    S(GetDeviceInfo, "clGetDeviceInfo");   S(CreateContext, "clCreateContext");
    S(CreateCommandQueue, "clCreateCommandQueue");
    S(CreateProgramWithSource, "clCreateProgramWithSource");
    S(BuildProgram, "clBuildProgram"); S(GetProgramBuildInfo, "clGetProgramBuildInfo");
    S(CreateKernel, "clCreateKernel"); S(SetKernelArg, "clSetKernelArg");
    S(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel"); S(Finish, "clFinish");
#undef S
    // The ICD loader does not export vendor extension symbols. The portable way is
    // clGetExtensionFunctionAddressForPlatform (CL 1.2+); dlsym on the vendor lib
    // is the fallback.
    {
        typedef void* (*gefp_t)(void*, const char*);
        gefp_t gefp = (gefp_t)dlsym(h->loader, "clGetExtensionFunctionAddressForPlatform");
        if (gefp && h->GetPlatformIDs) {
            cl_uint np = 0; h->GetPlatformIDs(0, nullptr, &np);
            std::vector<void*> plats(np);
            if (np) h->GetPlatformIDs(np, plats.data(), nullptr);
            void* sym = np ? gefp(plats[0], "clImportMemoryARM") : nullptr;
            p("IMPORT_MEMORY_SYM clGetExtensionFunctionAddressForPlatform -> %s",
              sym ? "found" : "absent");
            if (sym) h->ImportMemoryARM = (decltype(h->ImportMemoryARM))sym;
        } else {
            p("IMPORT_MEMORY_SYM clGetExtensionFunctionAddressForPlatform -> symbol absent");
        }
    }
    if (h->ImportMemoryARM) return true;
    const char* cands[] = {"/vendor/lib64/egl/libGLES_mali.so", "/vendor/lib64/libGLES_mali.so",
                           "libGLES_mali.so", "/vendor/lib64/libOpenCL-pixel.so",
                           "libOpenCL-pixel.so"};
    for (const char* c : cands) {
        void* v = dlopen(c, RTLD_NOW | RTLD_GLOBAL);
        if (!v) continue;
        void* sym = dlsym(v, "clImportMemoryARM");
        p("IMPORT_MEMORY_SYM %s -> %s", c, sym ? "found" : "absent");
        if (sym) { h->ImportMemoryARM = (decltype(h->ImportMemoryARM))sym; h->vendor = v; break; }
    }
    return h->GetDeviceIDs && h->CreateContext;
}

static const char* K_FILL = R"CLC(
__kernel void fill(__global int* buf, const int n, const int tag) {
    int g = get_global_id(0);
    if (g < n) buf[g] = tag + g;
}
)CLC";

// Writes a pattern from the GPU into the imported buffer and checks the host
// pointer sees it -> proves the pages are physically shared, not copied.
static int cl_import_probe(CLH* h, void* ptr, size_t bytes, void* import_target,
                           unsigned import_type, const char* tag) {
    cl_int err = 0;
    cl_uint np = 0;
    h->GetPlatformIDs(0, nullptr, &np);
    if (!np) { p("%s platform_missing", tag); return 0; }
    std::vector<void*> plats(np);
    h->GetPlatformIDs(np, plats.data(), nullptr);
    cl_uint nd = 0;
    h->GetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
    std::vector<cl_device_id> devs(nd);
    h->GetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
    cl_context ctx = h->CreateContext(nullptr, 1, &devs[0], nullptr, nullptr, &err);
    if (!ctx) { p("%s ctx_fail=%d", tag, err); return 0; }
    cl_command_queue q = h->CreateCommandQueue(ctx, devs[0], 0, &err);

    unsigned props[3] = {CL_IMPORT_TYPE_ARM, import_type, 0};  // zero-terminated
    cl_int ierr = -1;
    cl_mem m = h->ImportMemoryARM(ctx, CL_MEM_READ_WRITE, props, import_target, bytes, &ierr);
    if (!m) { p("%s import_FAIL err=%d props=0x%X", tag, ierr, import_type); return 0; }
    p("%s import_OK", tag);

    cl_program pr = h->CreateProgramWithSource(ctx, 1, &K_FILL, nullptr, &err);
    err = h->BuildProgram(pr, 1, &devs[0], "", nullptr, nullptr);
    if (err) { char log[2048] = {0}; size_t g = 0;
        h->GetProgramBuildInfo(pr, devs[0], CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, &g);
        p("%s build_FAIL %d %s", tag, err, log); return 0; }
    cl_kernel k = h->CreateKernel(pr, "fill", &err);
    cl_int n = (cl_int)(bytes / sizeof(int)), marker = 0x5A5A;
    h->SetKernelArg(k, 0, sizeof(cl_mem), &m);
    h->SetKernelArg(k, 1, sizeof(cl_int), &n);
    h->SetKernelArg(k, 2, sizeof(cl_int), &marker);
    size_t gsz = (size_t)n;
    err = h->EnqueueNDRangeKernel(q, k, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
    h->Finish(q);
    if (err) { p("%s enqueue_FAIL=%d", tag, err); return 0; }
    int* hostint = (int*)ptr;
    int expect0 = marker, expect1 = marker + 1;
    p("%s ZEROCOPY_VERIFY host[0]=%d(exp %d) host[1]=%d(exp %d) -> %s", tag,
      hostint[0], expect0, hostint[1], expect1,
      (hostint[0] == expect0 && hostint[1] == expect1) ? "SHARED" : "NOT_SHARED");
    return (hostint[0] == expect0 && hostint[1] == expect1) ? 1 : 0;
}

static void section_import() {
    CLH h;
    if (!clh_load(&h)) return;
    const size_t MiB = 1024 * 1024, bytes = 32 * MiB;

    // --- route A: AHardwareBuffer -> OpenCL ---
    AHardwareBuffer_Desc desc = {};
    desc.width = (uint32_t)bytes; desc.height = 1; desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                 AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    AHardwareBuffer* ahb = nullptr;
    if (AHardwareBuffer_allocate(&desc, &ahb) == 0) {
        void* host = nullptr;
        if (AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &host) == 0) {
            if (h.ImportMemoryARM) {
                cl_import_probe(&h, host, bytes, (void*)ahb,
                                CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM, "AHB_CL");
            } else {
                p("AHB_CL=SKIP (clImportMemoryARM not resolvable)");
            }
            AHardwareBuffer_unlock(ahb, nullptr);
        }
        ANeuralNetworksMemory* mem = nullptr;
        int rc = ANeuralNetworksMemory_createFromAHardwareBuffer(ahb, &mem);
        p("AHB_NNAPI_MEMORY_rc=%d(%s)", rc, nn_rc(rc));
        if (mem) ANeuralNetworksMemory_free(mem);
        AHardwareBuffer_release(ahb);
    } else {
        p("AHB_ALLOC_FAIL");
    }

    // --- route B: MAP_SHARED file -> OpenCL + NNAPI (one page cache, both engines) ---
    const char* path = "/data/local/tmp/hv_shared.bin";
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { p("MAPSHARED_FILE_FAIL errno=%d", errno); return; }
    if (ftruncate(fd, (off_t)bytes) != 0) p("MAPSHARED_TRUNC_FAIL");
    void* mp = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mp == MAP_FAILED) { p("MAPSHARED_MMAP_FAIL"); close(fd); return; }
    p("MAPSHARED_OK path=%s bytes=%zu", path, bytes);

    if (h.ImportMemoryARM)
        cl_import_probe(&h, mp, bytes, mp, CL_IMPORT_TYPE_HOST_ARM, "FILE_CL");
    else
        p("FILE_CL=SKIP (clImportMemoryARM not resolvable)");

    ANeuralNetworksMemory* fm = nullptr;
    int rc = ANeuralNetworksMemory_createFromFd(bytes, PROT_READ | PROT_WRITE, fd, 0, &fm);
    p("FILE_NNAPI_MEMORY_rc=%d(%s)", rc, nn_rc(rc));
    if (fm) ANeuralNetworksMemory_free(fm);

    munmap(mp, bytes); close(fd);
}

int main(int argc, char** argv) {
    std::string what = argc > 1 ? argv[1] : "all";
    printf("==JSON_BEGIN==\n");
    if (what == "nnapi" || what == "all") section_nnapi();
    if (what == "import" || what == "all") section_import();
    printf("==JSON_END==\n");
    return 0;
}
