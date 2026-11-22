#include "common.h"
#include "dequantize.h"

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];
constant int32_t  FC_mul_mm_id_map0_ne02  [[function_constant(FC_MUL_MM + 7)]];
constant bool     FC_mul_mm_id_scale      [[function_constant(FC_MUL_MM + 8)]];
constant int32_t  FC_mul_mm_id_map0_ntile [[function_constant(FC_MUL_MM + 9)]];

// each block_q contains 16*nl weights
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    if (!FC_mul_mm_bc_inp) {
        // Compile-time static K = N_MM_NK_TOTAL (64) when K % 64 == 0 eliminates runtime K-loop
        // and clamping overhead inside matmul2d, and hoists row pointers out of the K loop.
        mpp::tensor_ops::matmul2d<
            mpp::tensor_ops::matmul2d_descriptor(
                NRB, NRA, N_MM_NK_TOTAL, false, true, true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

        static_assert(A_WORK_ITEMS == 2 * NUM_THREADS, "expected 2 A-tile chunks per thread");
        const int   row0    = tiitg / N_MM_NK;
        const int   row1    = row0 + (NUM_THREADS / N_MM_NK);
        const int   k_chunk = tiitg % N_MM_NK;
        const short k_base  = k_chunk * 16;
        const bool  v0      = !FC_mul_mm_bc_out || (ra + row0 < M);
        const bool  v1      = !FC_mul_mm_bc_out || (ra + row1 < M);

        device const block_q * pA0 = v0 ? (device const block_q *)(srcA + args.nb01 * (ra + row0) + offset0) : nullptr;
        device const block_q * pA1 = v1 ? (device const block_q *)(srcA + args.nb01 * (ra + row1) + offset0) : nullptr;
        threadgroup vec<SA, 4> * dst_sa0 = (threadgroup vec<SA, 4> *)(sa + row0 * N_MM_NK_TOTAL + k_base);
        threadgroup vec<SA, 4> * dst_sa1 = (threadgroup vec<SA, 4> *)(sa + row1 * N_MM_NK_TOTAL + k_base);

        if (!v0) {
            dst_sa0[0] = vec<SA, 4>(0); dst_sa0[1] = vec<SA, 4>(0);
            dst_sa0[2] = vec<SA, 4>(0); dst_sa0[3] = vec<SA, 4>(0);
        }
        if (!v1) {
            dst_sa1[0] = vec<SA, 4>(0); dst_sa1[1] = vec<SA, 4>(0);
            dst_sa1[2] = vec<SA, 4>(0); dst_sa1[3] = vec<SA, 4>(0);
        }

        const short il_const = k_chunk % nl;
        if constexpr (nl <= N_MM_NK) {
            if (v0) { pA0 += k_chunk / nl; }
            if (v1) { pA1 += k_chunk / nl; }
        }
        constexpr int blk_step = (nl <= N_MM_NK) ? (N_MM_NK / nl) : 0;

        device T1 * ptrB_rb = ptrB + (uint64_t) rb * strideB;
        const int tileN_b = FC_mul_mm_bc_out ? min(NRB, N - rb) : NRB;

        for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
            if constexpr (nl <= N_MM_NK) {
                if (v0) {
                    SA_4x4 t0;
                    dequantize_func(pA0, il_const, t0);
                    dst_sa0[0] = t0[0]; dst_sa0[1] = t0[1]; dst_sa0[2] = t0[2]; dst_sa0[3] = t0[3];
                    pA0 += blk_step;
                }
                if (v1) {
                    SA_4x4 t1;
                    dequantize_func(pA1, il_const, t1);
                    dst_sa1[0] = t1[0]; dst_sa1[1] = t1[1]; dst_sa1[2] = t1[2]; dst_sa1[3] = t1[3];
                    pA1 += blk_step;
                }
            } else {
                const int   k_pos     = loop_k + k_base;
                const int   block_idx = k_pos / (16 * nl);
                const short il        = (k_pos / 16) % nl;
                if (v0) {
                    SA_4x4 t0;
                    dequantize_func(pA0 + block_idx, il, t0);
                    dst_sa0[0] = t0[0]; dst_sa0[1] = t0[1]; dst_sa0[2] = t0[2]; dst_sa0[3] = t0[3];
                }
                if (v1) {
                    SA_4x4 t1;
                    dequantize_func(pA1 + block_idx, il, t1);
                    dst_sa1[0] = t1[0]; dst_sa1[1] = t1[1]; dst_sa1[2] = t1[2]; dst_sa1[3] = t1[3];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto tBv = tensor(ptrB_rb + loop_k, dextents<int32_t, 2>(N_MM_NK_TOTAL, tileN_b), array<int, 2>({1, strideB}));
            mm.run(tBv, tA, cT);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const int tileM = FC_mul_mm_bc_out ? min(NRA, M - ra) : NRA;
        device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;
        auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, tileN_b), array<int, 2>({1, M}));
        cT.store(tD);
        return;
    }

    // Configure matmul operation (dynamic K fallback when K % N_MM_NK_TOTAL != 0)
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    if (k_pos + 16 <= K) {
                        threadgroup vec<SA, 4> * dst_sa = (threadgroup vec<SA, 4> *)(sa + row * N_MM_NK_TOTAL + k_base);
                        dst_sa[0] = temp_a[0];
                        dst_sa[1] = temp_a[1];
                        dst_sa[2] = temp_a[2];
                        dst_sa[3] = temp_a[3];
                    } else {
                        FOR_UNROLL (short i = 0; i < 16; i++) {
                            // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                            sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                        }
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        // Clamp the K extent of both operand tensors to the remaining valid K range so
        // the dynamic-K op never reads past the K extent of src1 (or the staged A tile).
        const int kExt = min(N_MM_NK_TOTAL, K - loop_k);

        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, N_MM_NK_TOTAL}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // a slice is not used because its byte offset is int32 and wraps at 2 GiB
    // ref: https://github.com/ggml-org/llama.cpp/pull/28748
    const int tileM = min(NRA, M - ra);
    const int tileN = min(NRB, N - rb);

    device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;

    auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, tileN), array<int, 2>({1, M}));
    cT.store(tD);
}

// Small-tile M5 tensor matmul (NRA_T x NRB_T, NK_T=64, NSG_T=2) for skinny/small-N dense matrices
// where the default 64x128 tile under-occupies 40-core GPUs or wastes >50% of the N tile.
template<
    typename SA, typename SA_4x4,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1,
    int NRA_T, int NRB_T, int NSG_T, int NK_T>
kernel void kernel_mul_mm_small(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    const int im  = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12 / FC_mul_mm_r2) * args.nb02 + (i13 / FC_mul_mm_r3) * args.nb03;

    const int ra = tgpig.y * NRA_T;
    const int rb = tgpig.x * NRB_T;

    threadgroup SA * sa = (threadgroup SA *)(shmem);

    constexpr int N_NK_CHUNKS  = NK_T / 16;
    constexpr int A_WORK_ITEMS = NRA_T * N_NK_CHUNKS;
    constexpr int NUM_THREADS  = N_SIMDWIDTH * NSG_T;

    device T1 * ptrB = (device T1 *)(srcB + args.nb12 * i12 + args.nb13 * i13);
    const int strideB = args.nb11 / sizeof(T1);

    auto tA = tensor(sa, dextents<int32_t, 2>(NK_T, NRA_T));
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    if (!FC_mul_mm_bc_inp) {
        mpp::tensor_ops::matmul2d<
            mpp::tensor_ops::matmul2d_descriptor(
                NRB_T, NRA_T, NK_T, false, true, true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<NSG_T>> mm;

        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

        static_assert(A_WORK_ITEMS % NUM_THREADS == 0, "A_WORK_ITEMS must be divisible by NUM_THREADS");
        constexpr int N_IT     = A_WORK_ITEMS / NUM_THREADS;
        constexpr int ROW_STEP = NUM_THREADS / N_NK_CHUNKS;
        constexpr int blk_step = (nl <= N_NK_CHUNKS) ? (N_NK_CHUNKS / nl) : 0;

        const int   row_base = tiitg / N_NK_CHUNKS;
        const int   k_chunk  = tiitg % N_NK_CHUNKS;
        const short k_base   = k_chunk * 16;
        const short il_const = k_chunk % nl;

        bool v[N_IT];
        device const block_q * pA[N_IT];
        threadgroup vec<SA, 4> * dst_sa[N_IT];

        FOR_UNROLL (int it = 0; it < N_IT; ++it) {
            const int row = row_base + it * ROW_STEP;
            v[it]      = !FC_mul_mm_bc_out || (ra + row < M);
            dst_sa[it] = (threadgroup vec<SA, 4> *)(sa + row * NK_T + k_base);
            if (v[it]) {
                pA[it] = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);
                if constexpr (nl <= N_NK_CHUNKS) {
                    pA[it] += k_chunk / nl;
                }
            } else {
                pA[it] = nullptr;
                dst_sa[it][0] = vec<SA, 4>(0); dst_sa[it][1] = vec<SA, 4>(0);
                dst_sa[it][2] = vec<SA, 4>(0); dst_sa[it][3] = vec<SA, 4>(0);
            }
        }

        device T1 * ptrB_rb = ptrB + (uint64_t) rb * strideB;
        const int tileN_b = FC_mul_mm_bc_out ? min(NRB_T, N - rb) : NRB_T;

        for (int loop_k = 0; loop_k < K; loop_k += NK_T) {
            if constexpr (nl <= N_NK_CHUNKS) {
                FOR_UNROLL (int it = 0; it < N_IT; ++it) {
                    if (v[it]) {
                        SA_4x4 tmp;
                        dequantize_func(pA[it], il_const, tmp);
                        dst_sa[it][0] = tmp[0]; dst_sa[it][1] = tmp[1];
                        dst_sa[it][2] = tmp[2]; dst_sa[it][3] = tmp[3];
                        pA[it] += blk_step;
                    }
                }
            } else {
                const int   k_pos     = loop_k + k_base;
                const int   block_idx = k_pos / (16 * nl);
                const short il        = (k_pos / 16) % nl;
                FOR_UNROLL (int it = 0; it < N_IT; ++it) {
                    if (v[it]) {
                        SA_4x4 tmp;
                        dequantize_func(pA[it] + block_idx, il, tmp);
                        dst_sa[it][0] = tmp[0]; dst_sa[it][1] = tmp[1];
                        dst_sa[it][2] = tmp[2]; dst_sa[it][3] = tmp[3];
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto tBv = tensor(ptrB_rb + loop_k, dextents<int32_t, 2>(NK_T, tileN_b), array<int, 2>({1, strideB}));
            mm.run(tBv, tA, cT);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const int tileM = FC_mul_mm_bc_out ? min(NRA_T, M - ra) : NRA_T;
        device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;
        auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, tileN_b), array<int, 2>({1, M}));
        cT.store(tD);
        return;
    }

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB_T, NRA_T, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG_T>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += NK_T) {
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int   row     = work / N_NK_CHUNKS;
            const int   k_chunk = work % N_NK_CHUNKS;
            const int   k_pos   = loop_k + k_chunk * 16;
            const short k_base  = k_chunk * 16;

            if (ra + row < M && k_pos < K) {
                if ((is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) || (k_pos + 16 > K)) {
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * NK_T + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA) 0;
                    }
                } else {
                    const int   block_idx = k_pos / (16 * nl);
                    const short il        = (k_pos / 16) % nl;
                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);
                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    threadgroup vec<SA, 4> * dst_sa = (threadgroup vec<SA, 4> *)(sa + row * NK_T + k_base);
                    dst_sa[0] = temp_a[0];
                    dst_sa[1] = temp_a[1];
                    dst_sa[2] = temp_a[2];
                    dst_sa[3] = temp_a[3];
                }
            } else {
                threadgroup vec<SA, 4> * dst_sa = (threadgroup vec<SA, 4> *)(sa + row * NK_T + k_base);
                dst_sa[0] = vec<SA, 4>(0);
                dst_sa[1] = vec<SA, 4>(0);
                dst_sa[2] = vec<SA, 4>(0);
                dst_sa[3] = vec<SA, 4>(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kExt = min(NK_T, K - loop_k);
        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA_T), array<int, 2>({1, NK_T}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int tileM = min(NRA_T, M - ra);
    const int tileN = min(NRB_T, N - rb);

    device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;

    auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, tileN), array<int, 2>({1, M}));
    cT.store(tD);
}

#else

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (uint64_t) (r1 + 16*(sgitg >> 1))*args.ne0 + (uint64_t) im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (uint64_t) (r1 + j)*args.ne0 + (uint64_t) im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

#endif // GGML_METAL_HAS_TENSOR

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short ide = tpitg; // expert id (0 .. ne02 - 1)
    const ushort n_sg = (ntg + N_SIMDWIDTH - 1) / N_SIMDWIDTH;

    threadgroup atomic_uint * cnt  = (threadgroup atomic_uint *) shmem;
    threadgroup atomic_uint * cur  = cnt + FC_mul_mm_id_map0_ne02;
    threadgroup uint32_t    * sg_n = (threadgroup uint32_t *) (cur + FC_mul_mm_id_map0_ne02);
    threadgroup uint32_t    * sg_t = sg_n + 32;

    atomic_store_explicit(&cnt[ide], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 1: parallel count across all tokens (O(ne21*ne20/ntg) per thread)
    for (int i21 = tpitg; i21 < args.ne21; i21 += ntg) {
        device const int32_t * src2_i32 = (device const int32_t *) (src2 + (uint64_t) i21 * args.nb21);
        FOR_UNROLL (short i20 = 0; i20 < ne20; ++i20) {
            const int32_t id = src2_i32[i20];
            if (id >= 0 && id < FC_mul_mm_id_map0_ne02) {
                atomic_fetch_add_explicit(&cnt[id], 1u, memory_order_relaxed);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2-level SIMD exclusive prefix sum for slice bases and tile offsets + parallel tile emission
    const uint32_t NTILE = (uint32_t) FC_mul_mm_id_map0_ntile;
    const uint32_t n     = atomic_load_explicit(&cnt[ide], memory_order_relaxed);
    const uint32_t nt    = (n + NTILE - 1) / NTILE;

    const uint32_t loc_n = simd_prefix_exclusive_sum(n);
    const uint32_t loc_t = simd_prefix_exclusive_sum(nt);
    const uint32_t tot_n = simd_sum(n);
    const uint32_t tot_t = simd_sum(nt);

    if (tiisg == 0) {
        sg_n[sgitg] = tot_n;
        sg_t[sgitg] = tot_t;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint32_t * tpe_u32 = (device uint32_t *) htpe;
    if (tpitg == 0) {
        uint32_t bn = 0;
        uint32_t bt = 0;
        for (ushort s = 0; s < n_sg; ++s) {
            const uint32_t vn = sg_n[s];
            const uint32_t vt = sg_t[s];
            sg_n[s] = bn;
            sg_t[s] = bt;
            bn += vn;
            bt += vt;
        }
        tpe_u32[0] = bt;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint32_t base = sg_n[sgitg] + loc_n;
    const uint32_t t0   = sg_t[sgitg] + loc_t;
    atomic_store_explicit(&cur[ide], base, memory_order_relaxed);

    for (uint32_t k = 0, ti = 0; k < n; k += NTILE, ++ti) {
        const uint32_t idx_t = 1 + 3 * (t0 + ti);
        tpe_u32[idx_t + 0] = (uint32_t) ide;
        tpe_u32[idx_t + 1] = base + k;
        tpe_u32[idx_t + 2] = (n - k < NTILE) ? (n - k) : NTILE;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 2: parallel scatter into compact per-expert slices (zero dummy writes)
    device int32_t * ids_i32 = (device int32_t *) hids;
    for (int i21 = tpitg; i21 < args.ne21; i21 += ntg) {
        device const int32_t * src2_i32 = (device const int32_t *) (src2 + (uint64_t) i21 * args.nb21);
        FOR_UNROLL (short i20 = 0; i20 < ne20; ++i20) {
            const int32_t id = src2_i32[i20];
            if (id >= 0 && id < FC_mul_mm_id_map0_ne02) {
                const uint32_t pos = atomic_fetch_add_explicit(&cur[id], 1u, memory_order_relaxed);
                ids_i32[pos] = i21 * ne20 + i20;
            }
        }
    }
}

// kernel_mul_mm_id only writes the rows that an expert owns, so the skipped rows are zeroed here
kernel void kernel_mul_mm_id_zero(
        constant ggml_metal_kargs_mul_mm_id_zero & args,
        device const char * src2,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i20 = tgpig.x; // slot
    const int i21 = tgpig.y; // token

    const int32_t id = ((device const int32_t *) (src2 + i21*args.nb21))[i20];

    if (id != -1) {
        return;
    }

    device float * dst_row = (device float *) (dst + i20*args.nb1 + i21*args.nb2);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_row[i0] = 0.0f;
    }
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_9" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<9>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

// Extended map0: also emits inverse map hinv[tok*ne20+slot] = pos and expert id hexp[pos] = expert
template<short ne20>
kernel void kernel_mul_mm_id_map0_ext(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char     * src2,
        device        char     * htpe,
        device        char     * hids,
        device        int32_t  * hinv,
        device        uint16_t * hexp,
        threadgroup   char     * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short ide = tpitg;
    const ushort n_sg = (ntg + N_SIMDWIDTH - 1) / N_SIMDWIDTH;

    threadgroup atomic_uint * cnt  = (threadgroup atomic_uint *) shmem;
    threadgroup atomic_uint * cur  = cnt + FC_mul_mm_id_map0_ne02;
    threadgroup uint32_t    * sg_n = (threadgroup uint32_t *) (cur + FC_mul_mm_id_map0_ne02);
    threadgroup uint32_t    * sg_t = sg_n + 32;

    atomic_store_explicit(&cnt[ide], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i21 = tpitg; i21 < args.ne21; i21 += ntg) {
        device const int32_t * src2_i32 = (device const int32_t *) (src2 + (uint64_t) i21 * args.nb21);
        FOR_UNROLL (short i20 = 0; i20 < ne20; ++i20) {
            const int32_t id = src2_i32[i20];
            if (id >= 0 && id < FC_mul_mm_id_map0_ne02) {
                atomic_fetch_add_explicit(&cnt[id], 1u, memory_order_relaxed);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint32_t NTILE = (uint32_t) FC_mul_mm_id_map0_ntile;
    const uint32_t n     = atomic_load_explicit(&cnt[ide], memory_order_relaxed);
    const uint32_t nt    = (n + NTILE - 1) / NTILE;

    const uint32_t loc_n = simd_prefix_exclusive_sum(n);
    const uint32_t loc_t = simd_prefix_exclusive_sum(nt);
    const uint32_t tot_n = simd_sum(n);
    const uint32_t tot_t = simd_sum(nt);

    if (tiisg == 0) {
        sg_n[sgitg] = tot_n;
        sg_t[sgitg] = tot_t;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint32_t * tpe_u32 = (device uint32_t *) htpe;
    if (tpitg == 0) {
        uint32_t bn = 0;
        uint32_t bt = 0;
        for (ushort s = 0; s < n_sg; ++s) {
            const uint32_t vn = sg_n[s];
            const uint32_t vt = sg_t[s];
            sg_n[s] = bn;
            sg_t[s] = bt;
            bn += vn;
            bt += vt;
        }
        tpe_u32[0] = bt;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint32_t base = sg_n[sgitg] + loc_n;
    const uint32_t t0   = sg_t[sgitg] + loc_t;
    atomic_store_explicit(&cur[ide], base, memory_order_relaxed);

    for (uint32_t k = 0, ti = 0; k < n; k += NTILE, ++ti) {
        const uint32_t idx_t = 1 + 3 * (t0 + ti);
        tpe_u32[idx_t + 0] = (uint32_t) ide;
        tpe_u32[idx_t + 1] = base + k;
        tpe_u32[idx_t + 2] = (n - k < NTILE) ? (n - k) : NTILE;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device int32_t * ids_i32 = (device int32_t *) hids;
    for (int i21 = tpitg; i21 < args.ne21; i21 += ntg) {
        device const int32_t * src2_i32 = (device const int32_t *) (src2 + (uint64_t) i21 * args.nb21);
        FOR_UNROLL (short i20 = 0; i20 < ne20; ++i20) {
            const int32_t id = src2_i32[i20];
            if (id >= 0 && id < FC_mul_mm_id_map0_ne02) {
                const uint32_t pos  = atomic_fetch_add_explicit(&cur[id], 1u, memory_order_relaxed);
                const int32_t  flat = i21 * ne20 + i20;
                ids_i32[pos] = flat;
                hinv[flat]   = (int32_t) pos;
                hexp[pos]    = (uint16_t) id;
            }
        }
    }
}

typedef decltype(kernel_mul_mm_id_map0_ext<1>) kernel_mul_mm_id_map0_ext_t;

template [[host_name("kernel_mul_mm_id_map0_ext_ne20_1" )]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<1>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_2" )]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<2>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_4" )]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<4>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_5" )]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<5>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_6" )]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<6>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_8" )]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<8>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_9" )]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<9>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_10")]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<10>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_16")]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<16>;
template [[host_name("kernel_mul_mm_id_map0_ext_ne20_22")]] kernel kernel_mul_mm_id_map0_ext_t kernel_mul_mm_id_map0_ext<22>;

// Gather f32 src1 into expert-contiguous f16 [K_pad, n_rows] once per MoE layer.
[[host_name("kernel_mul_mm_id_gather_f32_f16")]]
kernel void kernel_mul_mm_id_gather_f32_f16(
        constant ggml_metal_kargs_mul_mm_id_gather & args,
        device const char    * src1,
        device const int32_t * hids,
        device       half    * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int row = (int) tgpig.x;
    if (row >= args.n_rows) {
        return;
    }

    const int id  = hids[row];
    const int i11 = (id % args.ne20) % args.ne11;
    const int i12 = id / args.ne20;

    device const float4 * s4 = (device const float4 *) (src1 + (uint64_t) i12 * args.nb12 + (uint64_t) i11 * args.nb11);
    device       half4  * d4 = (device       half4  *) (dst  + (uint64_t) row * args.K_pad);

    const int k4 = args.K >> 2;
    for (int i = tpitg.x; i < k4; i += ntg.x) {
        d4[i] = half4(s4[i]);
    }
    if (args.K_pad > args.K) {
        device half * d = dst + (uint64_t) row * args.K_pad;
        for (int i = args.K + tpitg.x; i < args.K_pad; i += ntg.x) {
            d[i] = (half) 0;
        }
    }
}

// Expert-contiguous SITU activation: folds optional gate_up w_scale and writes f16 directly for ffn_down.
[[host_name("kernel_situ_expert_f16")]]
kernel void kernel_situ_expert_f16(
        constant ggml_metal_kargs_situ_expert & args,
        device const float4   * gate_up4,
        device const uint16_t * hexp,
        device const float    * w_scale,
        device       half4    * dst4,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int row = (int) tgpig.x;
    if (row >= args.n_rows) {
        return;
    }

    const float sc = args.has_scale ? w_scale[hexp[row]] : 1.0f;
    const int n_ff4 = args.n_ff4;

    device const float4 * gx = gate_up4 + (uint64_t) row * (2 * n_ff4);
    device const float4 * ux = gx + n_ff4;
    device       half4  * y  = dst4 + (uint64_t) row * n_ff4;

    for (int i = tpitg.x; i < n_ff4; i += ntg.x) {
        const float4 g = args.has_scale ? (gx[i] * sc) : gx[i];
        const float4 u = args.has_scale ? (ux[i] * sc) : ux[i];
        const float4 a = (args.beta * tanh(g * args.inv_beta)) / (1.0f + exp(-g));
        const float4 b = args.lb * tanh(u * args.inv_lb);
        y[i] = half4(a * b);
    }
}

// TODO: refactor into two seperate functions to avoid ifdef branching: with and without tensor support
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4, int NR1 = 32>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        device const float * w_scale,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

#ifdef GGML_METAL_HAS_TENSOR
    threadgroup float * sc = (threadgroup float *)(shmem);
#endif

    constexpr int NR0  = 64;
    constexpr int NK   = 32;
    constexpr int NL0  = NK/16;
    constexpr int NL1  = NK/8;

    const int tile = tgpig.y; // entry tile, described by the prepass (slow axis -> contiguous expert streaming & src1 L2 reuse)
    const int r0   = tgpig.x*NR0;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    // the prepass emits one (expert, first entry, valid rows) descriptor per tile that has
    // work - the grid is sized by a static upper bound, so the tail early exits
    if (tile >= (int) tpe_u32[0]) {
        return;
    }

    const int im  = tpe_u32[1 + 3*tile];     // expert
    const int idb = tpe_u32[1 + 3*tile + 1]; // first entry of this tile in hids

    // number of entries in this tile, always <= NR1
    const short nr1 = (short) ((tpe_u32[1 + 3*tile + 2] < NR1) ? tpe_u32[1 + 3*tile + 2] : NR1);

    // if this block is of 64xNR1 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63

    const short il0 = (tiitg % NL0);

    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    const short iy = 8*(tiitg % NL1);

    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

#ifndef GGML_METAL_HAS_TENSOR
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];
#else
    auto tA  = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));

    // sb is [NR1][NK] row-major (4 KB for NR1=64, 2 KB for NR1=32; sa+sb <= 8 KB)
    auto tB0 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb,             dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + NR1H*NK,   dextents<int32_t, 2>(NK, NR1H));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

#endif

    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    short il = il0;

    const int id = ids_i32[idb + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const int   i12 = (id / args.ne20);

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    const bool has_hi = nr1 > NR1H;

#ifdef GGML_METAL_HAS_TENSOR
    device const T1 * y1 = y;
    if (NR1 == 64 && has_hi) {
        const short lr1_hi = ((short)(32 + tiitg/NL1) < nr1) ? (short)(32 + tiitg/NL1) : (nr1 - 1);
        const int   id_hi  = ids_i32[idb + lr1_hi];
        const short i11_hi = (id_hi % args.ne20) % args.ne11;
        const int   i12_hi = (id_hi / args.ne20);
        y1 = (device const T1 *)(src1
            + args.nb13*i13
            + args.nb12*i12_hi
            + args.nb11*i11_hi
            + args.nb10*iy);
    }
#endif

#ifndef GGML_METAL_HAS_TENSOR
    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // simdgroups 2,3 own rows NR1H..NR1-1
    const bool sg_active = has_hi || sgitg < 2;
#else
    auto cT0 = mm.template get_destination_cooperative_tensor<decltype(tB0), decltype(tA), float>();
    auto cT1 = mm.template get_destination_cooperative_tensor<decltype(tB1), decltype(tA), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup vec<S0, 4> * dst_sa = (threadgroup vec<S0, 4> *)(sa + NK * (tiitg / NL0) + 16 * il0);
            dst_sa[0] = temp_a[0];
            dst_sa[1] = temp_a[1];
            dst_sa[2] = temp_a[2];
            dst_sa[3] = temp_a[3];
        }

        if (FC_mul_mm_bc_inp) {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            for (short i = 0; i < 8; ++i) {
                *(sb + NK*(8*sy + ly) + 8*sx + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
            if (NR1 == 64 && has_hi) {
                for (short i = 0; i < 8; ++i) {
                    *(sb + NK*(32 + 8*sy + ly) + 8*sx + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y1 + i) : 0;
                }
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));
            if (NR1 == 64 && has_hi) {
                *(threadgroup S1_2x4 *)(sb + NK*(32 + 8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y1));
            }
        }
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;
#ifdef GGML_METAL_HAS_TENSOR
        if (NR1 == 64) {
            y1 += NK;
        }
#endif

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        if (sg_active) {
            // load matrices from threadgroup memory and conduct outer products
            threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
            threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }
#else
        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);

        mm.run(sB0, sA, cT0);

        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);

            mm.run(sB1, sA, cT1);
        }
#endif
    }

    // block is smaller than 64xNR1, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float ex_scale = FC_mul_mm_id_scale ? w_scale[im] : 1.0f;

#ifdef GGML_METAL_HAS_TENSOR
    if (NR1 == 64) {
        // Two-pass coalesced float4 epilogue through the same 8 KB shmem (32 x 64 x 4 B)
        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1H));
        cT0.store(tC);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const short nr1_lo = nr1 < NR1H ? nr1 : NR1H;
        for (short j = sgitg; j < nr1_lo; j += 4) {
            const int id = ids_i32[idb + j];
            const short ide = id % args.ne20;
            const int   idt = id / args.ne20;

            device float  * D  = (device float  *) dst + r0 + (uint64_t) ide*args.ne0 + (uint64_t) idt*args.ne1*args.ne0;
            device float4 * D4 = (device float4 *) D;

            threadgroup float  * C  = sc + j*NR0;
            threadgroup float4 * C4 = (threadgroup float4 *) C;

            for (int i = tiisg; i < nr0/4; i += 32) {
                *(D4 + i) = FC_mul_mm_id_scale ? (*(C4 + i) * ex_scale) : *(C4 + i);
            }
            for (int i = (4 * (nr0 / 4)) + tiisg; i < nr0; i += 32) {
                *(D + i) = FC_mul_mm_id_scale ? (*(C + i) * ex_scale) : *(C + i);
            }
        }

        if (has_hi) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            cT1.store(tC);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short nr1_hi = nr1 - NR1H;
            for (short j = sgitg; j < nr1_hi; j += 4) {
                const int id = ids_i32[idb + NR1H + j];
                const short ide = id % args.ne20;
                const int   idt = id / args.ne20;

                device float  * D  = (device float  *) dst + r0 + (uint64_t) ide*args.ne0 + (uint64_t) idt*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = sc + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                for (int i = tiisg; i < nr0/4; i += 32) {
                    *(D4 + i) = FC_mul_mm_id_scale ? (*(C4 + i) * ex_scale) : *(C4 + i);
                }
                for (int i = (4 * (nr0 / 4)) + tiisg; i < nr0; i += 32) {
                    *(D + i) = FC_mul_mm_id_scale ? (*(C + i) * ex_scale) : *(C + i);
                }
            }
        }
        return;
    }

    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc,             dextents<int32_t, 2>(NR0, NR1H));
    cT0.store(tC0);

    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR1H*NR0, dextents<int32_t, 2>(NR0, NR1H));
        cT1.store(tC1);
    }
#else
    if (sg_active) {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
    }
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[idb + j];

        const short ide = id % args.ne20;
        const int   idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + (uint64_t) ide*args.ne0 + (uint64_t) idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = FC_mul_mm_id_scale ? (*(C4 + i) * ex_scale) : *(C4 + i);
        }

        i = (4 * (nr0 / 4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = FC_mul_mm_id_scale ? (*(C + i) * ex_scale) : *(C + i);
        }
    }
}

// Exact FP32 64x64 tiled matmul for K=16 broadcast batched projections (e.g. attn_rel_bias
// f32[16, 1024] x f32[16, 24, T]): eliminates 12,288x src0 reloads with 1 barrier and coalesced float4 stores.
[[host_name("kernel_mul_mm_f32_f32_k16")]]
kernel void kernel_mul_mm_f32_f32_k16(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  float4 * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    const int M      = args.ne0;
    const int ne11   = args.ne1;
    const int N_flat = ne11 * args.ne12;

    const int ra = (int) tgpig.y * 64;
    const int rb = (int) tgpig.x * 64;

    threadgroup float4 * sA4 = shmem;           // [64][4] float4 = 4096 B
    threadgroup float4 * sB4 = shmem + 64 * 4;  // [64][4] float4 = 4096 B

    const int row_l = tiitg >> 1; // 0..63
    const int half_l = tiitg & 1; // 0..1

    if (ra + row_l < M) {
        device const float4 * ap = (device const float4 *)(src0 + (uint64_t)(ra + row_l) * args.nb01) + half_l * 2;
        sA4[row_l * 4 + half_l * 2 + 0] = ap[0];
        sA4[row_l * 4 + half_l * 2 + 1] = ap[1];
    } else {
        sA4[row_l * 4 + half_l * 2 + 0] = float4(0.0f);
        sA4[row_l * 4 + half_l * 2 + 1] = float4(0.0f);
    }

    const int g_col = rb + row_l;
    if (g_col < N_flat) {
        const int i12 = g_col / ne11;
        const int i11 = g_col - i12 * ne11;
        device const float4 * bp = (device const float4 *)(src1 + (uint64_t) i12 * args.nb12 + (uint64_t) i11 * args.nb11) + half_l * 2;
        sB4[row_l * 4 + half_l * 2 + 0] = bp[0];
        sB4[row_l * 4 + half_l * 2 + 1] = bp[1];
    } else {
        sB4[row_l * 4 + half_l * 2 + 0] = float4(0.0f);
        sB4[row_l * 4 + half_l * 2 + 1] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 16 threads along M (4 consecutive rows each -> coalesced float4 writes) x 8 threads along N (8 cols each)
    const int tm = (tiitg & 15) * 4;
    const int tn = (tiitg >> 4) * 8;

    float4 acc[8];
    FOR_UNROLL (int n = 0; n < 8; ++n) {
        acc[n] = float4(0.0f);
    }

    FOR_UNROLL (int k4 = 0; k4 < 4; ++k4) {
        const float4 a0 = sA4[(tm + 0) * 4 + k4];
        const float4 a1 = sA4[(tm + 1) * 4 + k4];
        const float4 a2 = sA4[(tm + 2) * 4 + k4];
        const float4 a3 = sA4[(tm + 3) * 4 + k4];

        FOR_UNROLL (int n = 0; n < 8; ++n) {
            const float4 bv = sB4[(tn + n) * 4 + k4];
            acc[n].x += dot(a0, bv);
            acc[n].y += dot(a1, bv);
            acc[n].z += dot(a2, bv);
            acc[n].w += dot(a3, bv);
        }
    }

    if (ra + tm < M) {
        FOR_UNROLL (int n = 0; n < 8; ++n) {
            const int col = rb + tn + n;
            if (col < N_flat) {
                device float * dptr = (device float *) dst + (uint64_t) col * M + (ra + tm);
                if (ra + tm + 4 <= M) {
                    *(device float4 *) dptr = acc[n];
                } else {
                    if (ra + tm + 0 < M) dptr[0] = acc[n].x;
                    if (ra + tm + 1 < M) dptr[1] = acc[n].y;
                    if (ra + tm + 2 < M) dptr[2] = acc[n].z;
                    if (ra + tm + 3 < M) dptr[3] = acc[n].w;
                }
            }
        }
    }
}

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float,  float2x4>;
template [[host_name("kernel_mul_mm_bf16_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, bfloat, bfloat2x4>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_nvfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_nvfp4,   4,     dequantize_nvfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_nvfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_nvfp4,   4,     dequantize_nvfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

#ifdef GGML_METAL_HAS_TENSOR
typedef decltype(kernel_mul_mm_small<half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float, 16, 32, 2, 64>) mul_mm_small_t;

template [[host_name("kernel_mul_mm_s16x32_f32_f32")]]   kernel mul_mm_small_t kernel_mul_mm_small<half,   half4x4,   float4x4,   1, dequantize_f32,  float,  float4x4,  float,  16, 32, 2, 64>;
template [[host_name("kernel_mul_mm_s32x32_f32_f32")]]   kernel mul_mm_small_t kernel_mul_mm_small<half,   half4x4,   float4x4,   1, dequantize_f32,  float,  float4x4,  float,  32, 32, 2, 64>;
template [[host_name("kernel_mul_mm_s16x32_q8_0_f32")]]  kernel mul_mm_small_t kernel_mul_mm_small<half,   half4x4,   block_q8_0, 2, dequantize_q8_0, float,  float4x4,  float,  16, 32, 2, 64>;
template [[host_name("kernel_mul_mm_s32x32_q8_0_f32")]]  kernel mul_mm_small_t kernel_mul_mm_small<half,   half4x4,   block_q8_0, 2, dequantize_q8_0, float,  float4x4,  float,  32, 32, 2, 64>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_s16x32_bf16_f32")]]  kernel mul_mm_small_t kernel_mul_mm_small<bfloat, bfloat4x4, bfloat4x4, 1, dequantize_bf16, bfloat, bfloat4x4, float,  16, 32, 2, 64>;
template [[host_name("kernel_mul_mm_s32x32_bf16_f32")]]  kernel mul_mm_small_t kernel_mul_mm_small<bfloat, bfloat4x4, bfloat4x4, 1, dequantize_bf16, bfloat, bfloat4x4, float,  32, 32, 2, 64>;
#endif
#endif

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_nvfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_nvfp4,   4,     dequantize_nvfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#endif
template [[host_name("kernel_mul_mm_id_nvfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_nvfp4,   4,     dequantize_nvfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

#ifdef GGML_METAL_HAS_TENSOR
#define INST_MUL_MM_ID64(NAME, S0, S0_4x4, S0_8x8, S1, S1_2x4, S1_8x8, BQ, NL, DQ, T0, T0_4x4) \
    template [[host_name("kernel_mul_mm_id64_" NAME "_f32")]] kernel mul_mm_id kernel_mul_mm_id<S0, S0_4x4, S0_8x8, S1, S1_2x4, S1_8x8, BQ, NL, DQ, T0, T0_4x4, float, float2x4, 64>; \
    template [[host_name("kernel_mul_mm_id64_" NAME "_f16")]] kernel mul_mm_id kernel_mul_mm_id<S0, S0_4x4, S0_8x8, S1, S1_2x4, S1_8x8, BQ, NL, DQ, T0, T0_4x4, half,  half2x4,  64>;

INST_MUL_MM_ID64("f32",     half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4)
INST_MUL_MM_ID64("f16",     half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4)
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id64_bf16_f32")]] kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4, 1, dequantize_bf16, bfloat, bfloat4x4, float, float2x4, 64>;
INST_MUL_MM_ID64("mxfp4",   bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4)
#else
INST_MUL_MM_ID64("mxfp4",   half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4)
#endif
INST_MUL_MM_ID64("q1_0",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4)
INST_MUL_MM_ID64("q2_0",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4)
INST_MUL_MM_ID64("q4_0",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4)
INST_MUL_MM_ID64("q4_1",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4)
INST_MUL_MM_ID64("q5_0",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4)
INST_MUL_MM_ID64("q5_1",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4)
INST_MUL_MM_ID64("q8_0",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4)
INST_MUL_MM_ID64("nvfp4",   half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_nvfp4,   4,     dequantize_nvfp4,   float,  float4x4)
INST_MUL_MM_ID64("q2_K",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4)
INST_MUL_MM_ID64("q3_K",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4)
INST_MUL_MM_ID64("q4_K",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4)
INST_MUL_MM_ID64("q5_K",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4)
INST_MUL_MM_ID64("q6_K",    half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4)
INST_MUL_MM_ID64("iq2_xxs", half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4)
INST_MUL_MM_ID64("iq2_xs",  half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4)
INST_MUL_MM_ID64("iq3_xxs", half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4)
INST_MUL_MM_ID64("iq3_s",   half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4)
INST_MUL_MM_ID64("iq2_s",   half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4)
INST_MUL_MM_ID64("iq1_s",   half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4)
INST_MUL_MM_ID64("iq1_m",   half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4)
INST_MUL_MM_ID64("iq4_nl",  half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4)
INST_MUL_MM_ID64("iq4_xs",  half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4)
INST_MUL_MM_ID64("tq2_0",   half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4)
#undef INST_MUL_MM_ID64

// Direct-device Metal 4 tensor matmul for expert-contiguous MoE pipelines:
// Matches dense kernel_mul_mm (NRA=64, NK=64, 8 KB sa-only shmem, direct-device f16 tB, native cT.store).
template<
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &),
    typename T0, typename T0_4x4, int NRB>
static inline void mul_mm_id_contig_run(
        constant ggml_metal_kargs_mul_mm_id_contig & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem,
        int ra, int im, int idb, int nr1,
        ushort tiitg) {
    constexpr int NRA          = 64;
    constexpr int NK_T         = 64;
    constexpr int N_NK_CHUNKS  = NK_T / 16;
    constexpr int A_WORK_ITEMS = NRA * N_NK_CHUNKS;
    constexpr int NUM_THREADS  = 128;

    const int K = args.K;
    const int M = args.M;
    const int strideB = args.stride_b;
    const uint64_t offset0 = (uint64_t) im * args.nb02;

    threadgroup half * sa = (threadgroup half *) shmem;

    auto tA = tensor(sa, dextents<int32_t, 2>(NK_T, NRA));
    device half * ptrB = (device half *) srcB + (uint64_t) idb * strideB;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, nr1), array<int, 2>({1, strideB}));

    if (!FC_mul_mm_bc_inp) {
        mpp::tensor_ops::matmul2d<
            mpp::tensor_ops::matmul2d_descriptor(
                NRB, NRA, NK_T, false, true, true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<4>> mm;

        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

        static_assert(A_WORK_ITEMS == 2 * NUM_THREADS, "expected 2 A-tile chunks per thread");
        const int   row0    = tiitg / N_NK_CHUNKS;
        const int   row1    = row0 + (NUM_THREADS / N_NK_CHUNKS);
        const int   k_chunk = tiitg % N_NK_CHUNKS;
        const short k_base  = k_chunk * 16;
        const bool  v0      = !FC_mul_mm_bc_out || (ra + row0 < M);
        const bool  v1      = !FC_mul_mm_bc_out || (ra + row1 < M);

        device const block_q * pA0 = v0 ? (device const block_q *)(srcA + args.nb01 * (ra + row0) + offset0) : nullptr;
        device const block_q * pA1 = v1 ? (device const block_q *)(srcA + args.nb01 * (ra + row1) + offset0) : nullptr;
        threadgroup half4 * dst_sa0 = (threadgroup half4 *)(sa + row0 * NK_T + k_base);
        threadgroup half4 * dst_sa1 = (threadgroup half4 *)(sa + row1 * NK_T + k_base);

        if (!v0) {
            dst_sa0[0] = half4(0); dst_sa0[1] = half4(0); dst_sa0[2] = half4(0); dst_sa0[3] = half4(0);
        }
        if (!v1) {
            dst_sa1[0] = half4(0); dst_sa1[1] = half4(0); dst_sa1[2] = half4(0); dst_sa1[3] = half4(0);
        }

        const short il_const = k_chunk % nl;
        if constexpr (nl <= N_NK_CHUNKS) {
            if (v0) { pA0 += k_chunk / nl; }
            if (v1) { pA1 += k_chunk / nl; }
        }
        constexpr int blk_step = (nl <= N_NK_CHUNKS) ? (N_NK_CHUNKS / nl) : 0;

        for (int loop_k = 0; loop_k < K; loop_k += NK_T) {
            if constexpr (nl <= N_NK_CHUNKS) {
                if (v0) {
                    half4x4 t0;
                    dequantize_func(pA0, il_const, t0);
                    dst_sa0[0] = t0[0]; dst_sa0[1] = t0[1]; dst_sa0[2] = t0[2]; dst_sa0[3] = t0[3];
                    pA0 += blk_step;
                }
                if (v1) {
                    half4x4 t1;
                    dequantize_func(pA1, il_const, t1);
                    dst_sa1[0] = t1[0]; dst_sa1[1] = t1[1]; dst_sa1[2] = t1[2]; dst_sa1[3] = t1[3];
                    pA1 += blk_step;
                }
            } else {
                const int   k_pos     = loop_k + k_base;
                const int   block_idx = k_pos / (16 * nl);
                const short il        = (k_pos / 16) % nl;
                if (v0) {
                    half4x4 t0;
                    dequantize_func(pA0 + block_idx, il, t0);
                    dst_sa0[0] = t0[0]; dst_sa0[1] = t0[1]; dst_sa0[2] = t0[2]; dst_sa0[3] = t0[3];
                }
                if (v1) {
                    half4x4 t1;
                    dequantize_func(pA1 + block_idx, il, t1);
                    dst_sa1[0] = t1[0]; dst_sa1[1] = t1[1]; dst_sa1[2] = t1[2]; dst_sa1[3] = t1[3];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto tBv = tensor(ptrB + loop_k, dextents<int32_t, 2>(NK_T, nr1), array<int, 2>({1, strideB}));
            mm.run(tBv, tA, cT);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const int tileM = FC_mul_mm_bc_out ? min(NRA, M - ra) : NRA;
        device float * dstTile = (device float *) dst + (uint64_t) idb * M + ra;
        auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, nr1), array<int, 2>({1, M}));
        cT.store(tD);
        return;
    }

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += NK_T) {
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int   row     = work / N_NK_CHUNKS;
            const int   k_chunk = work % N_NK_CHUNKS;
            const int   k_pos   = loop_k + k_chunk * 16;
            const short k_base  = k_chunk * 16;

            if (ra + row < M && k_pos < K) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * NK_T + (k_base + i)] = (k_pos + i < K) ? (half) row_ptr[k_pos + i] : (half) 0;
                    }
                } else {
                    const int   block_idx = k_pos / (16 * nl);
                    const short il        = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    half4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    if (!FC_mul_mm_bc_inp || k_pos + 16 <= K) {
                        threadgroup half4 * dst_sa = (threadgroup half4 *)(sa + row * NK_T + k_base);
                        dst_sa[0] = temp_a[0];
                        dst_sa[1] = temp_a[1];
                        dst_sa[2] = temp_a[2];
                        dst_sa[3] = temp_a[3];
                    } else {
                        FOR_UNROLL (short i = 0; i < 16; i++) {
                            sa[row * NK_T + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (half) 0;
                        }
                    }
                }
            } else {
                threadgroup half4 * dst_sa = (threadgroup half4 *)(sa + row * NK_T + k_base);
                dst_sa[0] = half4(0);
                dst_sa[1] = half4(0);
                dst_sa[2] = half4(0);
                dst_sa[3] = half4(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kExt = min(NK_T, K - loop_k);
        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, NK_T}));
        auto tBv = tensor(ptrB + loop_k, dextents<int32_t, 2>(kExt, nr1), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int tileM = min(NRA, M - ra);
    device float * dstTile = (device float *) dst + (uint64_t) idb * M + ra;
    auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, nr1), array<int, 2>({1, M}));
    cT.store(tD);
}

// Per-tile adaptive NRB (ntile=64): 3-tier 16/32/64 loads/dequantizes weights once per <=64-row tile
// while keeping register pressure low (zero spills) across medium and large prefill.
template<
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &),
    typename T0, typename T0_4x4>
kernel void kernel_mul_mm_id_tensor_contig_adapt64(
        constant ggml_metal_kargs_mul_mm_id_contig & args,
        device const char * srcA,
        device const char * srcB,
        device const char * htpe,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    const int tile = (int) tgpig.y;
    device const uint32_t * tpe_u32 = (device const uint32_t *) htpe;
    if (tile >= (int) tpe_u32[0]) {
        return;
    }

    const int im  = (int) tpe_u32[1 + 3*tile + 0];
    const int idb = (int) tpe_u32[1 + 3*tile + 1];
    const int nr1 = min((int) tpe_u32[1 + 3*tile + 2], 64);
    const int ra  = (int) tgpig.x * 64;

    if (nr1 <= 16) {
        mul_mm_id_contig_run<block_q, nl, dequantize_func, T0, T0_4x4, 16>(
            args, srcA, srcB, dst, shmem, ra, im, idb, nr1, tiitg);
    } else if (nr1 <= 32) {
        mul_mm_id_contig_run<block_q, nl, dequantize_func, T0, T0_4x4, 32>(
            args, srcA, srcB, dst, shmem, ra, im, idb, nr1, tiitg);
    } else {
        mul_mm_id_contig_run<block_q, nl, dequantize_func, T0, T0_4x4, 64>(
            args, srcA, srcB, dst, shmem, ra, im, idb, nr1, tiitg);
    }
}

typedef decltype(kernel_mul_mm_id_tensor_contig_adapt64<block_nvfp4, 4, dequantize_nvfp4, float, float4x4>) mul_mm_id_contig_adapt64_t;

#define INST_MUL_MM_ID_CONTIG(NAME, BQ, NL, DQ, T0, T0_4x4) \
    template [[host_name("kernel_mul_mm_id_contigAdapt64_" NAME "_f16")]] kernel mul_mm_id_contig_adapt64_t kernel_mul_mm_id_tensor_contig_adapt64<BQ, NL, DQ, T0, T0_4x4>;

INST_MUL_MM_ID_CONTIG("f32",   float4x4,    1,     dequantize_f32,   float, float4x4)
INST_MUL_MM_ID_CONTIG("f16",   half4x4,     1,     dequantize_f16,   half,  half4x4)
INST_MUL_MM_ID_CONTIG("q4_0",  block_q4_0,  2,     dequantize_q4_0,  float, float4x4)
INST_MUL_MM_ID_CONTIG("q8_0",  block_q8_0,  2,     dequantize_q8_0,  float, float4x4)
INST_MUL_MM_ID_CONTIG("nvfp4", block_nvfp4, 4,     dequantize_nvfp4, float, float4x4)
INST_MUL_MM_ID_CONTIG("q4_K",  block_q4_K,  QK_NL, dequantize_q4_K,  float, float4x4)
#undef INST_MUL_MM_ID_CONTIG
#endif
