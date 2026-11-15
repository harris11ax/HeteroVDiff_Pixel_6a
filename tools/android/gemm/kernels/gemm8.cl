// gemm8.cl — INT8 GEMM kernels for Mali-G78 MP20 (Valhall), Pixel 6a / Tensor G1.
//
// The artifact layout is fixed by configs/design_locks.json:
//   A : activations, row-major [M, K] int8, K contiguous   (padded to M_pad rows)
//   W : weights,     row-major [N, K] int8, K contiguous   (one scale per output column)
//   s : per-output-column scale  sw[n],  plus one activation scale sa
// so a thread's `char4` operand is exactly 4 consecutive K values of one row —
// the same bytes NNAPI's FULLY_CONNECTED consumes for the NPU side.
//
// C[m,n] = sa * sw[n] * sum_k A[m,k]*W[n,k]   (+ optional rank-R LoRA delta)
//
// Kernels
//   gemm8_ref     one work-item per output, no tiling      (exact int32 reference + floor rate)
//   gemm8_tiled   register blocking TM x TN, local-memory staging, BK deep
//   gemm8_direct  same register blocking, operands read straight from global
//   gemm16_tiled  FP16 register-blocked (the detail-expert LoRA path is FP16)
//
// TM/TN/BM/BN/BK come from the build options (-DBM=.. etc) so one source gives a
// sweep of tile shapes without editing.

#pragma OPENCL EXTENSION cl_arm_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 32
#endif
#define KV (BK / 4)          // char4 operands per BK slice

// ---------------------------------------------------------------- reference
__kernel void gemm8_ref(__global const char4* A, __global const char4* W,
                        __global int* C, const int M, const int N, const int K4)
{
    const int n = get_global_id(0), m = get_global_id(1);
    if (m >= M || n >= N) return;
    int s = 0;
    const int ao = m * K4, wo = n * K4;
    for (int k = 0; k < K4; ++k) s = arm_dot_acc(A[ao + k], W[wo + k], s);
    C[m * N + n] = s;
}

// ------------------------------------------------------- shared epilogue ----
// Kept as a macro so the tiled/direct kernels cannot drift apart. `use_lora`
// is uniform across the work-group (a kernel argument), so the branch is free.
#define EPILOGUE_TAIL(acc)                                                       \
    /* ---- epilogue: dequantise per output column, add LoRA delta ----            \
       Predicated, NOT break-terminated: a `break` in an unrolled loop makes the   \
       accumulator array dynamically indexed, so the compiler spills all TM*TN    \
       accumulators to private memory (measured: 272 B/thread) and the kernel      \
       dies with CL_EXEC_STATUS_ERROR (-14) on Mali. Keep the bounds as ifs. */   \
    _Pragma("unroll")                                                             \
    for (int i = 0; i < TM; ++i) {                                                \
        const int m = m0 + i;                                                     \
        if (m < M) {                                                              \
            _Pragma("unroll")                                                     \
            for (int j = 0; j < TN; ++j) {                                        \
                const int n = n0 + j;                                             \
                if (n < N) {                                                      \
                    if (raw_out) {                                                \
                        Craw[m * N + n] = acc[i][j];                              \
                    } else {                                                      \
                        float v = (float)acc[i][j] * (sa * sw[n]);                \
                        if (use_lora == 1) {                                      \
                            /* delta = U @ Bl^T precomputed by two FP16 GEMMs      \
                               (K=1536->R=128->N=1536); the epilogue only adds it. */\
                            v += U[m * N + n];                                    \
                        } else if (use_lora == 2) {                               \
                            /* the naive fused form: a rank-R dot per output      \
                               element. Measured 8-9x slower than the whole      \
                               INT8 GEMM because the U/Bl reads have no reuse.    \
                               Kept as evidence, not as the design. */             \
                            float d = 0.0f;                                       \
                            for (int r = 0; r < R; ++r)                           \
                                d += Bl[m * R + r] * U[n * R + r];                \
                            v += d;                                               \
                        }                                                         \
                        C[m * N + n] = v;                                         \
                    }                                                             \
                }                                                                 \
            }                                                                     \
        }                                                                         \
    }

// ------------------------------------------------------------ tiled (LDS) ---
__kernel __attribute__((reqd_work_group_size(BN / TN, BM / TM, 1)))
void gemm8_tiled(__global const char4* A, __global const char4* W,
                 __global float* C, __global int* Craw,
                 __global const float* sw, const float sa,
                 __global const float* U, __global const float* Bl, const int R,
                 const int M, const int N, const int K4,
                 const int raw_out, const int use_lora)
{
    const int lx = get_local_id(0), ly = get_local_id(1);
    // The work-group stages ONE BM x BN tile: the staging loops must use the
    // group base, while each thread's own 8x8 sub-tile starts at the local-id
    // offset. Mixing the two (staging from the thread's m0) has every thread in
    // the group stage a different window into the same shared buffers.
    const int nbase = (int)get_group_id(0) * BN;
    const int mbase = (int)get_group_id(1) * BM;
    const int n0 = nbase + lx * TN;
    const int m0 = mbase + ly * TM;

    // +1 char4 of padding: the per-thread strided reads would otherwise all land
    // on the same local-memory bank group (stride 8*16 B = exactly one bank cycle).
    __local char4 As[KV][BM + 1];
    __local char4 Bs[KV][BN + 1];

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

    const int nt = (BN / TN) * (BM / TM);
    const int lid = ly * (BN / TN) + lx;
    const int a_elems = BM * KV, b_elems = BN * KV;

    for (int k0 = 0; k0 < K4; k0 += KV) {
        // ---- stage one BK slice of A and W into local memory ----
        for (int idx = lid; idx < a_elems; idx += nt) {
            const int mi = idx / KV, ki = idx - mi * KV;
            As[ki][mi] = A[(mbase + mi) * K4 + k0 + ki];
        }
        for (int idx = lid; idx < b_elems; idx += nt) {
            const int ni = idx / KV, ki = idx - ni * KV;
            Bs[ki][ni] = W[(nbase + ni) * K4 + k0 + ki];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- TM*TN accumulators against TM+TN local vectors per 4-wide K step ----
#pragma unroll 2
        for (int ki = 0; ki < KV; ++ki) {
            char4 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[ki][ly * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[ki][lx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = arm_dot_acc(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    EPILOGUE_TAIL(acc)
}

// --------------------------------------------------------- direct (no LDS) --
// Same register tile, operands straight from global. A operand is broadcast to
// every thread in a row of the warp; W is strided but L2-resident after the first
// K pass over the tile column. Much shorter critical path than the LDS version
// (no barriers, no LDS issue slots), which is what Valhall tends to like when the
// tile is small relative to the CU count.
__kernel __attribute__((reqd_work_group_size(BN / TN, BM / TM, 1)))
void gemm8_direct(__global const char4* A, __global const char4* W,
                  __global float* C, __global int* Craw,
                  __global const float* sw, const float sa,
                  __global const float* U, __global const float* Bl, const int R,
                  const int M, const int N, const int K4,
                  const int raw_out, const int use_lora)
{
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int n0 = (int)get_group_id(0) * BN + lx * TN;
    const int m0 = (int)get_group_id(1) * BM + ly * TM;

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

    for (int k = 0; k < K4; ++k) {
        char4 a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) a[i] = A[(m0 + i) * K4 + k];
#pragma unroll
        for (int j = 0; j < TN; ++j) b[j] = W[(n0 + j) * K4 + k];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j)
                acc[i][j] = arm_dot_acc(a[i], b[j], acc[i][j]);
    }

    EPILOGUE_TAIL(acc)
}

// ---------------------------------------------------------------- FP16 GEMM --
// The detail expert's LoRA factors are FP16 (design lock), so the delta path is
// an FP16 GEMM: U[M,R] = Xh[M,K] @ Al[R,K]^T, then delta = U @ Bl[N,R]^T (fused
// into the INT8 epilogue above). Same register-blocked shape, float accumulate.
#ifndef TM16
#define TM16 8
#endif
#ifndef TN16
#define TN16 8
#endif

__kernel __attribute__((reqd_work_group_size(BN / TN16, BM / TM16, 1)))
void gemm16_tiled(__global const half* A, __global const half* W,
                  __global float* C,
                  const int M, const int N, const int K)
{
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int nbase = (int)get_group_id(0) * BN;   // group tile base (see gemm8_tiled)
    const int mbase = (int)get_group_id(1) * BM;
    const int n0 = nbase + lx * TN16;
    const int m0 = mbase + ly * TM16;

    __local half As[BK][BM + 1];
    __local half Bs[BK][BN + 1];

    float acc[TM16][TN16];
#pragma unroll
    for (int i = 0; i < TM16; ++i)
#pragma unroll
        for (int j = 0; j < TN16; ++j) acc[i][j] = 0.0f;

    const int nt = (BN / TN16) * (BM / TM16);
    const int lid = ly * (BN / TN16) + lx;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = lid; idx < BM * BK; idx += nt) {
            const int mi = idx / BK, ki = idx - mi * BK;
            As[ki][mi] = A[(mbase + mi) * K + k0 + ki];
        }
        for (int idx = lid; idx < BN * BK; idx += nt) {
            const int ni = idx / BK, ki = idx - ni * BK;
            Bs[ki][ni] = W[(nbase + ni) * K + k0 + ki];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 2
        for (int ki = 0; ki < BK; ++ki) {
            half a[TM16], b[TN16];
#pragma unroll
            for (int i = 0; i < TM16; ++i) a[i] = As[ki][ly * TM16 + i];
#pragma unroll
            for (int j = 0; j < TN16; ++j) b[j] = Bs[ki][lx * TN16 + j];
#pragma unroll
            for (int i = 0; i < TM16; ++i)
#pragma unroll
                for (int j = 0; j < TN16; ++j)
                    acc[i][j] = fma((float)a[i], (float)b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int i = 0; i < TM16; ++i) {
        const int m = m0 + i;
        if (m < M) {
#pragma unroll
            for (int j = 0; j < TN16; ++j) {
                const int n = n0 + j;
                if (n < N) C[m * N + n] = acc[i][j];
            }
        }
    }
}

// ================================================================ diag =======
// Feature ladder + throughput probes used to isolate driver rejections and to
// establish what the dot builtin actually costs on this GPU.

#ifndef DTM
#define DTM 1
#endif
#ifndef DTN
#define DTN 1
#endif

__kernel void dg_2d(__global int* Craw, const int M, const int N)
{
    const int m = (int)get_global_id(1), n = (int)get_global_id(0);
    if (m < M && n < N) Craw[m * N + n] = m * 1000 + n;
}

// DTM*DTN accumulators fed by arm_dot_acc, operands straight from global.
__kernel void dg_dot(__global const char4* A, __global const char4* W, __global int* Craw,
                     const int M, const int N, const int K4)
{
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int nx = get_local_size(0), ny = get_local_size(1);
    const int n0 = ((int)get_group_id(0) * nx + lx) * DTN;
    const int m0 = ((int)get_group_id(1) * ny + ly) * DTM;
    int acc[DTM][DTN];
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int j = 0; j < DTN; ++j) acc[i][j] = 0;
    for (int k = 0; k < K4; ++k) {
        char4 a[DTM], b[DTN];
#pragma unroll
        for (int i = 0; i < DTM; ++i) a[i] = A[(m0 + i) * K4 + k];
#pragma unroll
        for (int j = 0; j < DTN; ++j) b[j] = W[(n0 + j) * K4 + k];
#pragma unroll
        for (int i = 0; i < DTM; ++i)
#pragma unroll
            for (int j = 0; j < DTN; ++j) acc[i][j] = arm_dot_acc(a[i], b[j], acc[i][j]);
    }
#pragma unroll
    for (int i = 0; i < DTM; ++i) {
        const int m = m0 + i;
        if (m < M) {
#pragma unroll
            for (int j = 0; j < DTN; ++j) {
                const int n = n0 + j;
                if (n < N) Craw[m * N + n] = acc[i][j];
            }
        }
    }
}

// ------------------------------------------------ dot-throughput probes -----
// Same shape of inner loop for all three: 8x8 operand pairs -> 8 accumulator
// chains -> 256 MACs per iteration, 1 global load per iteration to keep the
// operands live (the compiler cannot hoist a loop that observes memory).
//   dg_peak_dot   : 64 arm_dot_acc   (= 64 instructions if SDOT is real)
//   dg_peak_mac4  : 256 mad24        (the 4-instruction emulation of one dot)
//   dg_peak_mad1  : 64 mad24         (1 MAC per instruction)
#define PEAK_BODY(OP)                                                            \
    const int gid = (int)get_global_id(0);                                       \
    char4 a[8], b[8];                                                            \
    _Pragma("unroll")                                                            \
    for (int i = 0; i < 8; ++i) { a[i] = src[i]; b[i] = src[8 + i]; }            \
    int acc[8];                                                                  \
    _Pragma("unroll")                                                            \
    for (int i = 0; i < 8; ++i) acc[i] = 0;                                      \
    for (int it = 0; it < iters; ++it) {                                          \
        a[0] = src[it & 63];                                                      \
        _Pragma("unroll")                                                         \
        for (int i = 0; i < 8; ++i)                                              \
            _Pragma("unroll")                                                     \
            for (int j = 0; j < 8; ++j) { OP }                                    \
    }                                                                            \
    int s = 0;                                                                    \
    _Pragma("unroll")                                                            \
    for (int i = 0; i < 8; ++i) s += acc[i];                                      \
    out[gid] = s;

__kernel void dg_peak_dot(__global const char4* src, __global int* out, const int iters)
{
    PEAK_BODY(acc[i] = arm_dot_acc(a[j], b[i], acc[i]);)
}

__kernel void dg_peak_mac4(__global const char4* src, __global int* out, const int iters)
{
    PEAK_BODY(acc[i] = mad24((int)a[j].s0, (int)b[i].s0, acc[i]);
              acc[i] = mad24((int)a[j].s1, (int)b[i].s1, acc[i]);
              acc[i] = mad24((int)a[j].s2, (int)b[i].s2, acc[i]);
              acc[i] = mad24((int)a[j].s3, (int)b[i].s3, acc[i]);)
}

__kernel void dg_peak_mad1(__global const char4* src, __global int* out, const int iters)
{
    PEAK_BODY(acc[i] = mad24((int)a[j].s0, (int)b[i].s0, acc[i]);)
}
