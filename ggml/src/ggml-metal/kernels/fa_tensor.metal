#include "common.h"
#include "dequantize.h"

#ifdef GGML_METAL_HAS_TENSOR

constant bool FC_flash_attn_ext_tensor_has_mask   [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 0)]];
constant bool FC_flash_attn_ext_tensor_has_sinks  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 1)]];
constant bool FC_flash_attn_ext_tensor_has_bias   [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 2)]];
constant bool FC_flash_attn_ext_tensor_has_scap   [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 3)]];
constant bool FC_flash_attn_ext_tensor_has_banded  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 4)]];
constant bool FC_flash_attn_ext_tensor_has_kvpad   [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 5)]];
constant bool FC_flash_attn_ext_tensor_has_rel_pos [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 6)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename T,                                 // Q, K, V and P types
    typename T4,
    short DK,                                   // K head size
    short DV,                                   // V head size
    short Q   = OP_FLASH_ATTN_EXT_TENSOR_NQPSG, // queries per threadgroup
    short C   = OP_FLASH_ATTN_EXT_TENSOR_NCPSG, // cache items per threadgroup
    short NSG = OP_FLASH_ATTN_EXT_TENSOR_NSG>   // number of simd groups
kernel void kernel_flash_attn_ext_tensor(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * blk,
        device       char * dst,
        device const char * rel,
        device const char * pad,
        device const char * k_pos,
        device const char * q_pos,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup T     * sq = (threadgroup T     *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup T     * sp = (threadgroup T     *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last iteration (ic0 + 1) that rescaled O

    q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

    {
        const int ikv2 = iq2/(args.ne02/args.ne_12_2);
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // with softcap the scale is small (scale/softcap), so it is applied to the scores to keep the precision of Q
    const float qscale = FC_flash_attn_ext_tensor_has_scap ? 1.0f : args.scale;

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);

        float4 q4 = 0.0f;
        if (iq1 + j < args.ne01) {
            q4 = ((device const float4 *) (q + j*args.nb01))[i%(DK/4)];
        }

        ((threadgroup T4 *) sq)[i] = (T4) (q4*qscale);
    }

    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm[jj] = (device const half *) (mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    device const float * pr[NQ];
    int qp[NQ];
    int qp_min = INT_MAX;
    int qp_max = INT_MIN;
    if (FC_flash_attn_ext_tensor_has_banded) {
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;
            const int iq1_r = min(iq1 + j, args.ne01 - 1);

            pr[jj] = (device const float *) (rel + iq1_r*args.nb62 + iq2*args.nb61 + (iq3%args.ne63)*args.nb63);
            if (FC_flash_attn_ext_tensor_has_rel_pos) {
                qp[jj] = ((device const int *) q_pos)[iq3*args.ne01 + iq1_r];
            } else {
                qp[jj] = iq1_r + (args.ne11 - args.ne01);
            }
            if (iq1 + j < args.ne01) {
                qp_min = min(qp_min, qp[jj]);
                qp_max = max(qp_max, qp[jj]);
            }
        }
    }

    {
        const int nblk1 = (args.ne01 + Q - 1)/Q;
        const int nblk0 = (args.ne11 + C - 1)/C;

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    float M[NQ];
    float S[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    float slope = 1.0f;

    // ALiBi
    if (FC_flash_attn_ext_tensor_has_bias) {
        const short h = iq2;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exph);
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device T *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    // the O matrix from the paper
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ic0 = 0, ic = 0; ic < args.ne11; ++ic0, ic += C) {
        char blk_cur = 1;

        if (FC_flash_attn_ext_tensor_has_mask) {
            blk_cur = blk[ic0];

            if (blk_cur == 0) {
                continue;
            }
        }

        const bool is_pad = FC_flash_attn_ext_tensor_has_kvpad && (ic + C > args.ne11);
        device const char * k_tile = k + (uint64_t) ic*args.nb11;
        device const char * v_tile = v + (uint64_t) ic*args.nb21;

        if (is_pad) {
            const int ikv2 = iq2/(args.ne02/args.ne_12_2);
            const int ikv3 = iq3/(args.ne03/args.ne_12_3);
            k_tile = pad + (uint64_t)(ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
            v_tile = pad + args.nb11*C*args.ne_12_2*args.ne_12_3 + (uint64_t)(ikv2 + ikv3*args.ne_12_2)*args.nb21*C;
        }

        // Q*K^T
        {
            auto tk = tensor((device T *) k_tile, dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // hoist tile-level k_pos range check for banded attention (skips 99% of tiles when rel_extent << ne11)
        int kp_reg[NC];
        bool tile_has_rel = false;
        if (FC_flash_attn_ext_tensor_has_banded) {
            int kp_min = ic;
            int kp_max = min(ic + C - 1, args.ne11 - 1);
            if (FC_flash_attn_ext_tensor_has_rel_pos) {
                device const int * kp = (device const int *) k_pos + (uint64_t) iq3*args.ne11 + ic;
                int t_min = INT_MAX;
                int t_max = INT_MIN;
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    const int col = ii*NW + tiisg;
                    if (ic + col < args.ne11) {
                        const int kv = kp[col];
                        kp_reg[ii] = kv;
                        t_min = min(t_min, kv);
                        t_max = max(t_max, kv);
                    } else {
                        kp_reg[ii] = 0;
                    }
                }
                kp_min = simd_min(t_min);
                kp_max = simd_max(t_max);
            }
            tile_has_rel = (qp_max >= kp_min) && (qp_min - kp_max < args.rel_extent);
        }

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (FC_flash_attn_ext_tensor_has_scap) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = args.logit_softcap*precise::tanh(s[ii]*args.scale);
                }
            }

            if (is_pad) {
                if (FC_flash_attn_ext_tensor_has_mask && iq1 + j < args.ne31) {
                    device const half * pm_pad = (device const half *) (pad + args.nb11*C*args.ne_12_2*args.ne_12_3 + args.nb21*C*args.ne_12_2*args.ne_12_3)
                        + (iq1 + j)*C + (iq2%args.ne32)*(C*args.ne31) + (iq3%args.ne33)*(C*args.ne31*args.ne32);
                    FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                        s[ii] += slope*(float) pm_pad[ii*NW + tiisg];
                    }
                } else {
                    FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                        if (ic + ii*NW + tiisg >= args.ne11) {
                            s[ii] = -FLT_MAX/2;
                        }
                    }
                }
            } else if (FC_flash_attn_ext_tensor_has_mask && blk_cur != 2 && iq1 + j < args.ne31) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] += slope*(float) pm[jj][ic + ii*NW + tiisg];
                }
            }

            if (FC_flash_attn_ext_tensor_has_banded && tile_has_rel && iq1 + j < args.ne01) {
                if (FC_flash_attn_ext_tensor_has_rel_pos) {
                    FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                        const int col = ii*NW + tiisg;
                        if (ic + col < args.ne11) {
                            const int rel_dist = qp[jj] - kp_reg[ii];
                            if (rel_dist >= 0 && rel_dist < args.rel_extent) {
                                s[ii] += pr[jj][rel_dist];
                            }
                        }
                    }
                } else {
                    const int rel_base = (iq1 + j) + (args.ne11 - args.ne01) - ic;
                    if (rel_base >= 0 && rel_base - (C - 1) < args.rel_extent) {
                        FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                            const int rel_dist = rel_base - (ii*NW + tiisg);
                            if (rel_dist >= 0 && rel_dist < args.rel_extent) {
                                s[ii] += pr[jj][rel_dist];
                            }
                        }
                    }
                }
            }

            float m = M[jj];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const T p = (T) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device T *) v_tile, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        // the sink only adds to the denominator - its rescale of O is folded into the final scale
        float ms = 1.0f;

        if (FC_flash_attn_ext_tensor_has_sinks) {
            const int iq1_s = min(iq1 + j, args.ne01 - 1);
            const float s = ((device const float *) sinks)[iq2 + iq1_s*args.sinks_s1 + iq3*args.sinks_s2];
            const float m = max(M[jj], s);

            ms = exp(M[jj] - m);

            S[jj] = S[jj]*ms + exp(s - m);
        }

        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : ms/S[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    // store to global memory - rows past ne01 are clipped by the tensor extents
    device float * pdst = (device float *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + iq2 + (uint64_t) iq1*args.ne1)*DV;

    auto td = tensor(pdst, dextents<int32_t, 2>(DV, args.ne01 - iq1), array<int, 2>({1, args.ne1*DV}));

    co.store(td);
}

typedef decltype(kernel_flash_attn_ext_tensor<half, half4, 64, 64>) flash_attn_ext_tensor_t;

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk64_dv64"  )]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk192_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk256_dv256")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk512_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk576_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk64_dv64"  )]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk128_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk192_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk256_dv256")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk512_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk576_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

constant int32_t FC_flash_attn_ext_tensor_nwg [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 7)]];

// High-throughput F16 Metal 4 tensor FlashAttention with pre-packed device Q (zero Q shmem),
// hardware exp2 softmax, tile-level k_pos skip, Split-KV (NWG), and wide Q=32xC=128 MMA tiles.
template<
    short DK,
    short DV,
    short Q,
    short C,
    short NSG>
kernel void kernel_flash_attn_ext_tensor_f16_wide(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q_f16,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * blk,
        device       char * dst,
        device const char * rel,
        device const char * pad,
        device const char * k_pos,
        device const char * q_pos,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = Q / NSG;
    constexpr short NC = C / NW;
    constexpr short DV4 = DV / 4;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;

    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW  == 0, "C must be divisible by NW");

    const int NWG     = FC_flash_attn_ext_tensor_nwg;
    const int iwg     = tgpig[2] % NWG;
    const int iq3     = tgpig[2] / NWG;
    const int iq2     = tgpig[1];
    const int iq1_blk = tgpig[0];
    const int iq1     = iq1_blk * Q;
    const int nq_blk  = (args.ne01 + Q - 1) / Q;

    const short tiitg = sgitg * NW + tiisg;

    threadgroup float * ss = (threadgroup float *) shmem;        // [Q, C] scores
    threadgroup half  * sp = (threadgroup half  *) (ss + Q * C); // [Q, C] probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q * C); // [Q]    per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);     // [1]    last iteration (ic0+1) that rescaled O

    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    k += (uint64_t) ikv2 * args.nb12 + (uint64_t) ikv3 * args.nb13;
    v += (uint64_t) ikv2 * args.nb22 + (uint64_t) ikv3 * args.nb23;

    device const half * pm[NQ];
    if (FC_flash_attn_ext_tensor_has_mask) {
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int iq1_m = min(iq1 + j, args.ne31 - 1);
            pm[jj] = (device const half *) (mask + (uint64_t) iq1_m * args.nb31 + (uint64_t) (iq2 % args.ne32) * args.nb32 + (uint64_t) (iq3 % args.ne33) * args.nb33);
        }
    }

    int qp[NQ];
    int qp_min = INT_MAX;
    int qp_max = INT_MIN;
    if (FC_flash_attn_ext_tensor_has_banded) {
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int iq1_r = min(iq1 + j, args.ne01 - 1);
            if (FC_flash_attn_ext_tensor_has_rel_pos) {
                qp[jj] = ((device const int *) q_pos)[(uint64_t) iq3 * args.ne01 + iq1_r];
            } else {
                qp[jj] = iq1_r + (args.ne11 - args.ne01);
            }
            if (iq1 + j < args.ne01) {
                qp_min = min(qp_min, qp[jj]);
                qp_max = max(qp_max, qp[jj]);
            }
        }
    }

    {
        const int nblk1 = (args.ne01 + Q - 1) / Q;
        const int nblk0 = (args.ne11 + C - 1) / C;
        blk += (((uint64_t) (iq3 % args.ne33) * args.ne32 + (iq2 % args.ne32)) * nblk1 + iq1_blk) * nblk0;
    }

    float M[NQ];
    float S[NQ];
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX / 2;
        S[jj] = 0.0f;
    }

    float slope_log2 = LOG2E;
    if (FC_flash_attn_ext_tensor_has_bias) {
        const short h = iq2;
        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2 * (h - args.n_head_log2) + 1;
        slope_log2 = pow(base, exph) * LOG2E;
    }

    const float scap_log2 = args.logit_softcap * LOG2E;
    const int sk = args.ns10;
    const int sv = args.ns20;

    const uint64_t tg_idx = ((uint64_t) iq3 * args.ne02 + iq2) * nq_blk + iq1_blk;
    device half * q_tile = (device half *) q_f16 + tg_idx * ((uint64_t) Q * DK);

    auto tq  = tensor(q_tile, dextents<int32_t, 2>(DK, Q));
    auto ts  = tensor(ss,     dextents<int32_t, 2>(C,  Q));
    auto tp  = tensor(sp,     dextents<int32_t, 2>(C,  Q));
    auto tv0 = tensor((device half *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg < Q) {
        sr[tiitg] = 1.0f;
    }
    if (tiitg == 0) {
        sf[0] = 0;
    }

    device const char * rel_head = rel + (uint64_t) iq2 * args.nb61 + (uint64_t) (iq3 % args.ne63) * args.nb63;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ic0 = iwg; ; ic0 += NWG) {
        const int ic = ic0 * C;
        if (ic >= args.ne11) {
            break;
        }
        char blk_cur = 1;
        if (FC_flash_attn_ext_tensor_has_mask) {
            blk_cur = blk[ic0];
            if (blk_cur == 0) {
                continue;
            }
        }

        const bool is_pad = FC_flash_attn_ext_tensor_has_kvpad && (ic + C > args.ne11);
        device const char * k_tile = k + (uint64_t) ic * args.nb11;
        device const char * v_tile = v + (uint64_t) ic * args.nb21;

        if (is_pad) {
            k_tile = pad + (uint64_t) (ikv2 + ikv3 * args.ne_12_2) * args.nb11 * C;
            v_tile = pad + (uint64_t) args.nb11 * C * args.ne_12_2 * args.ne_12_3
                         + (uint64_t) (ikv2 + ikv3 * args.ne_12_2) * args.nb21 * C;
        }

        // Q * K^T (note: when !has_scap, q_tile is already scaled by scale * LOG2E!)
        {
            auto tk = tensor((device half *) k_tile, dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));
            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Vectorized float4/half4 online softmax for C=128 (NC=4) in log2 space using single-cycle hardware exp2
        static_assert(NC == 4, "kernel_flash_attn_ext_tensor_f16_wide requires C == 128 (NC == 4)");
        {
            int4 kp4 = int4(ic + 4 * tiisg + 0, ic + 4 * tiisg + 1, ic + 4 * tiisg + 2, ic + 4 * tiisg + 3);
            bool tile_has_rel = false;
            if (FC_flash_attn_ext_tensor_has_banded) {
                int kp_min = ic;
                int kp_max = min(ic + C - 1, args.ne11 - 1);
                if (FC_flash_attn_ext_tensor_has_rel_pos) {
                    device const int * kp = (device const int *) k_pos + (uint64_t) iq3 * args.ne11 + ic;
                    const int base_col = 4 * tiisg;
                    int t_min = INT_MAX;
                    int t_max = INT_MIN;
                    if (ic + base_col + 3 < args.ne11) {
                        kp4 = (((ulong) kp & 15ul) == 0)
                            ? ((device const int4 *) kp)[tiisg]
                            : int4(*(device const packed_int4 *) (kp + base_col));
                        t_min = min(min(kp4[0], kp4[1]), min(kp4[2], kp4[3]));
                        t_max = max(max(kp4[0], kp4[1]), max(kp4[2], kp4[3]));
                    } else {
                        FOR_UNROLL (short k = 0; k < 4; ++k) {
                            if (ic + base_col + k < args.ne11) {
                                const int kv = kp[base_col + k];
                                kp4[k] = kv;
                                t_min = min(t_min, kv);
                                t_max = max(t_max, kv);
                            } else {
                                kp4[k] = 0;
                            }
                        }
                    }
                    kp_min = simd_min(t_min);
                    kp_max = simd_max(t_max);
                }
                tile_has_rel = (qp_max >= kp_min) && (qp_min - kp_max < args.rel_extent);
            }

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj * NSG + sgitg;

                float4 s4 = ((threadgroup float4 *) (ss + j * C))[tiisg];

                if (FC_flash_attn_ext_tensor_has_scap) {
                    s4 = scap_log2 * precise::tanh(s4 * args.scale);
                }

                if (is_pad) {
                    if (FC_flash_attn_ext_tensor_has_mask && iq1 + j < args.ne31) {
                        device const half4 * pm_pad4 = (device const half4 *) (
                            (device const half *) (pad + (uint64_t) (args.nb11 + args.nb21) * C * args.ne_12_2 * args.ne_12_3)
                            + (iq1 + j) * C + (iq2 % args.ne32) * (C * args.ne31) + (iq3 % args.ne33) * (C * args.ne31 * args.ne32));
                        s4 += slope_log2 * (float4) pm_pad4[tiisg];
                    } else {
                        const int base_col = ic + 4 * tiisg;
                        FOR_UNROLL (short k = 0; k < 4; ++k) {
                            if (base_col + k >= args.ne11) {
                                s4[k] = -FLT_MAX / 2;
                            }
                        }
                    }
                } else if (FC_flash_attn_ext_tensor_has_mask && blk_cur != 2 && iq1 + j < args.ne31) {
                    device const half * pm_row = pm[jj] + ic;
                    const half4 m4 = (((ulong) pm_row & 7ul) == 0)
                        ? ((device const half4 *) pm_row)[tiisg]
                        : half4(*(device const packed_half4 *) (pm_row + 4 * tiisg));
                    s4 += slope_log2 * (float4) m4;
                }

                if (FC_flash_attn_ext_tensor_has_banded && tile_has_rel && iq1 + j < args.ne01) {
                    device const float * pr_row = (device const float *) (rel_head + (uint64_t) (iq1 + j) * args.nb62);
                    const int base_col = ic + 4 * tiisg;
                    FOR_UNROLL (short k = 0; k < 4; ++k) {
                        if (base_col + k < args.ne11) {
                            const int rel_dist = qp[jj] - kp4[k];
                            if (rel_dist >= 0 && rel_dist < args.rel_extent) {
                                s4[k] += pr_row[rel_dist] * LOG2E;
                            }
                        }
                    }
                }

                float m = max(M[jj], max(max(s4[0], s4[1]), max(s4[2], s4[3])));
                m = simd_max(m);

                // lazy rescaling: move running max only when it grows by > 11 (2^11 = 2048 fits easily in half)
                float ms = 1.0f;
                if (m > M[jj] + 11.0f) {
                    ms    = exp2(M[jj] - m);
                    M[jj] = m;

                    if (tiisg == 0) {
                        sf[0] = ic0 + 1;
                    }
                }

                const half4 p4 = (half4) exp2(s4 - M[jj]);
                ((threadgroup half4 *) (sp + j * C))[tiisg] = p4;

                const float4 pf = (float4) p4;
                S[jj] = S[jj] * ms + simd_sum((pf[0] + pf[1]) + (pf[2] + pf[3]));

                if (tiisg == 0) {
                    sr[j] = ms;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device half *) v_tile, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));
            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        float ms = 1.0f;

        if (FC_flash_attn_ext_tensor_has_sinks && iwg == 0) {
            const int iq1_s = min(iq1 + j, args.ne01 - 1);
            const float s_val = ((device const float *) sinks)[iq2 + iq1_s * args.sinks_s1 + iq3 * args.sinks_s2] * LOG2E;
            const float m = max(M[jj], s_val);

            ms    = exp2(M[jj] - m);
            M[jj] = m;
            S[jj] = S[jj] * ms + exp2(s_val - m);
        }

        if (tiisg == 0) {
            sr[j] = (NWG == 1) ? (S[jj] == 0.0f ? 0.0f : ms / S[jj]) : ms;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (NWG == 1) {
        FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
            if (co.is_valid_element(i)) {
                co[i] *= sr[co.get_multidimensional_index(i)[1]];
            }
        }

        device float * pdst = (device float *) dst + ((uint64_t) iq3 * args.ne2 * args.ne1 + iq2 + (uint64_t) iq1 * args.ne1) * DV;
        auto td = tensor(pdst, dextents<int32_t, 2>(DV, args.ne01 - iq1), array<int, 2>({1, args.ne1 * DV}));
        co.store(td);
    } else {
        const uint64_t nrows = (uint64_t) args.ne3 * args.ne2 * args.ne1;
        device float * dst_f = (device float *) dst;
        device float * dst1  = dst_f + nrows * DV * NWG;
        FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
            if (co.is_valid_element(i)) {
                const auto idx = co.get_multidimensional_index(i);
                const short d  = idx[0];
                const short r  = idx[1];
                const int iq1_o = iq1 + r;
                if (iq1_o < args.ne01) {
                    const uint64_t rid   = (uint64_t) iq3 * args.ne2 * args.ne1 + iq2 + (uint64_t) iq1_o * args.ne1;
                    const uint64_t f_off = ((rid * DV4 * NWG + (uint64_t) (d >> 2) * NWG + (uint64_t) iwg) << 2) + (d & 3);
                    dst_f[f_off] = co[i] * sr[r];
                }
            }
        }
        if (tiisg == 0) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j   = jj * NSG + sgitg;
                const int iq1_o = iq1 + j;
                if (iq1_o < args.ne01) {
                    const uint64_t rid = (uint64_t) iq3 * args.ne2 * args.ne1 + iq2 + (uint64_t) iq1_o * args.ne1;
                    dst1[rid * (2 * NWG) + 2 * iwg + 0] = S[jj];
                    dst1[rid * (2 * NWG) + 2 * iwg + 1] = M[jj] * LN2;
                }
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_tensor_f16_wide<128, 128, 32, 128, 4>) flash_attn_ext_tensor_f16_wide_t;

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128_q32_c128_nsg4")]] kernel flash_attn_ext_tensor_f16_wide_t kernel_flash_attn_ext_tensor_f16_wide<128, 128, 32, 128, 4>;

// Split-KV Metal 4 Tensor inline-Q8_0/Q4_0 GQA FlashAttention for speculative verification & small-batch decode
// (ne01 = 2..19, DK = DV = 128, Q_TOK = 4, H = 2/3/4 -> padded to M = 16, C = 64, NSG = 4, adaptive NWG).
// Runs Q*K^T and P*V on M5 Neural Accelerators while reading Q8_0/Q4_0 KV inline once (zero kv_f16 / q_f16 scratch)
// and writing partial outputs to bid_tmp for the existing kernel_flash_attn_ext_vec_reduce.
template<short Q_TOK, short H, short M_PAD, bool IS_Q4_0 = false>
kernel void kernel_flash_attn_ext_vec_tensor_q8_0_gqa(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        device const char * idx,
        device const char * rel,
        device const char * k_pos,
        device const char * q_pos,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) idx;

    constexpr short DK    = 128;
    constexpr short DV    = 128;
    constexpr short DK4   = DK / 4;
    constexpr short DV4   = DV / 4;
    constexpr short DK16  = DK / 16; // 8
    constexpr short C     = 64;
    constexpr short NSG   = 4;
    constexpr short NW    = N_SIMDWIDTH; // 32
    constexpr short NT    = NSG * NW;    // 128
    constexpr short QH    = Q_TOK * H;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;

    static_assert(Q_TOK == NSG, "Q_TOK must equal NSG (4)");
    static_assert(QH <= M_PAD,  "Q_TOK*H must fit in M_PAD");

    const int NWG      = FC_flash_attn_ext_tensor_nwg;
    const int iwg      = tgpig[2] % NWG;
    const int iq3      = tgpig[2] / NWG;
    const int iq2_blk  = tgpig[1];
    const int iq1_blk  = tgpig[0];
    const int iq1      = iq1_blk * Q_TOK;
    const int iq2_base = iq2_blk * H;

    const short tiitg = sgitg * NW + tiisg;

    // Shared memory layout (26.1 KB <= 32 KB):
    //   sq   : [M_PAD, DK] half   = 16 * 128 * 2 =  4,096 B
    //   skv  : [C, DK]     half   = 64 * 128 * 2 = 16,384 B (reused for K, V, and final 8 KB float O store)
    //   ss   : [M_PAD, C]  float  = 16 *  64 * 4 =  4,096 B
    //   sp   : [M_PAD, C]  half   = 16 *  64 * 2 =  2,048 B
    //   sr   : [M_PAD]     float  = 16 *       4 =     64 B
    //   s_ctl: [8]         int    = 32 B (s_ctl[0..3] = mask fin flags, s_ctl[4] = rescale flag sf)
    threadgroup half  * sq    = (threadgroup half  *) shmem;
    threadgroup half  * skv   = sq + M_PAD * DK;
    threadgroup float * ss    = (threadgroup float *) (skv + C * DK);
    threadgroup half  * sp    = (threadgroup half  *) (ss  + M_PAD * C);
    threadgroup float * sr    = (threadgroup float *) (sp  + M_PAD * C);
    threadgroup int   * s_ctl = (threadgroup int   *) (sr  + M_PAD);

    const int ikv2 = iq2_base / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3      / (args.ne03 / args.ne_12_3);

    device const char * k_base = k + (uint64_t) ikv2 * args.nb12 + (uint64_t) ikv3 * args.nb13;
    device const char * v_base = v + (uint64_t) ikv2 * args.nb22 + (uint64_t) ikv3 * args.nb23;

    const float qscale    = FC_flash_attn_ext_tensor_has_scap ? 1.0f : args.scale * LOG2E;
    const float scap_log2 = args.logit_softcap * LOG2E;

    // Load Q [QH, DK] into sq (rows QH..M_PAD-1 zero-padded)
    for (short i = tiitg; i < M_PAD * DK4; i += NT) {
        const short r  = i / DK4;
        const short i4 = i % DK4;
        if (r < QH) {
            const short qq    = r / H;
            const short hh    = r % H;
            const int   iq1_q = iq1 + qq;
            if (iq1_q < args.ne01) {
                device const float4 * q4 = (device const float4 *) (
                    q + (uint64_t) iq1_q * args.nb01 + (uint64_t) (iq2_base + hh) * args.nb02 + (uint64_t) iq3 * args.nb03);
                ((threadgroup half4 *) sq)[i] = (half4) (q4[i4] * qscale);
            } else {
                ((threadgroup half4 *) sq)[i] = half4(0.0h);
            }
        } else {
            ((threadgroup half4 *) sq)[i] = half4(0.0h);
        }
    }

    // Zero padded probability rows QH..M_PAD-1 once
    for (short r = QH; r < M_PAD; ++r) {
        for (short c = tiitg; c < C; c += NT) {
            sp[r * C + c] = 0.0h;
        }
    }

    if (tiitg < M_PAD) {
        sr[tiitg] = 1.0f;
    }
    if (tiitg == 0) {
        s_ctl[4] = 0;
    }

    // Each simdgroup sgitg owns query token qq = sgitg across all H heads (rows r = sgitg*H + hh)
    const int iq1_sg = iq1 + sgitg;
    const bool sg_q_valid = iq1_sg < args.ne01;

    int qp_sg = 0;
    if (FC_flash_attn_ext_tensor_has_banded) {
        const int iq1_r = min(iq1_sg, args.ne01 - 1);
        if (FC_flash_attn_ext_tensor_has_rel_pos) {
            qp_sg = ((device const int *) q_pos)[(uint64_t) iq3 * args.ne01 + iq1_r];
        } else {
            qp_sg = iq1_r + (args.ne11 - args.ne01);
        }
    }

    float M[H];
    float S[H];
    FOR_UNROLL (short hh = 0; hh < H; ++hh) {
        M[hh] = -FLT_MAX / 2;
        S[hh] = 0.0f;
    }

    device const char * pm_stream = mask + (uint64_t) (iq3 % args.ne33) * args.nb33;
    device const char * rel_base  = rel  + (uint64_t) (iq3 % args.ne63) * args.nb63 + (uint64_t) iq2_base * args.nb61;

    auto tq = tensor(sq,  dextents<int32_t, 2>(DK, M_PAD));
    auto tk = tensor(skv, dextents<int32_t, 2>(DK, C));
    auto tv = tensor(skv, dextents<int32_t, 2>(DV, C));
    auto ts = tensor(ss,  dextents<int32_t, 2>(C,  M_PAD));
    auto tp = tensor(sp,  dextents<int32_t, 2>(C,  M_PAD));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M_PAD, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M_PAD, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ic0 = iwg; ; ic0 += NWG) {
        const int ic = ic0 * C;
        if (ic >= args.ne11) {
            break;
        }

        const bool is_pad = FC_flash_attn_ext_tensor_has_kvpad && (ic + C > args.ne11);

        // Load 2 mask columns per lane (C=64) for qq=sgitg and skip all-masked tiles (critical for multi-slot KV!)
        float m0 = 0.0f;
        float m1 = 0.0f;
        bool force_ninf0 = false;
        bool force_ninf1 = false;

        if (FC_flash_attn_ext_tensor_has_mask) {
            if (sg_q_valid) {
                if (is_pad) {
                    device const half * pm_pad = (device const half *) (
                        pad + (uint64_t) (args.nb11 + args.nb21) * C * args.ne_12_2 * args.ne_12_3)
                        + (uint64_t) (iq3 % args.ne33) * (C * args.ne31) + iq1_sg * C;
                    m0 = (float) pm_pad[tiisg];
                    m1 = (float) pm_pad[tiisg + NW];
                } else {
                    device const half * pm_row = (device const half *) (pm_stream + (uint64_t) iq1_sg * args.nb31);
                    m0 = (float) pm_row[ic + tiisg];
                    m1 = (float) pm_row[ic + tiisg + NW];
                }
            } else {
                m0 = (float) (-MAXHALF);
                m1 = (float) (-MAXHALF);
            }

            const bool fin = simd_max(max(m0, m1)) > (float) (-MAXHALF);
            if (tiisg == 0) {
                s_ctl[sgitg] = fin ? 1 : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const bool any_fin = (s_ctl[0] | s_ctl[1] | s_ctl[2] | s_ctl[3]) != 0;
            if (!any_fin) {
                // Barrier before continue so all simdgroups finish reading s_ctl before the next iteration overwrites it.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                continue;
            }
        } else if (is_pad) {
            force_ninf0 = (ic + tiisg) >= args.ne11;
            force_ninf1 = (ic + tiisg + NW) >= args.ne11;
        }

        device const char * k_tile = k_base + (uint64_t) ic * args.nb11;
        device const char * v_tile = v_base + (uint64_t) ic * args.nb21;

        if (is_pad) {
            k_tile = pad + (uint64_t) (ikv2 + ikv3 * args.ne_12_2) * args.nb11 * C;
            v_tile = pad + (uint64_t) args.nb11 * C * args.ne_12_2 * args.ne_12_3
                         + (uint64_t) (ikv2 + ikv3 * args.ne_12_2) * args.nb21 * C;
        }

        // Inline Q8_0/Q4_0 K dequantize into skv (64*8 = 512 half-blocks / 128 threads = 4/thread)
        FOR_UNROLL (short kit = 0; kit < 4; ++kit) {
            const short work = kit * NT + tiitg;
            const short row  = work / DK16;
            const short i16  = work % DK16;
            half4x4 tmp;
            if constexpr (IS_Q4_0) {
                device const block_q4_0 * kb = (device const block_q4_0 *) (k_tile + (uint64_t) row * args.nb11) + (i16 / 2);
                dequantize_q4_0(kb, i16 % 2, tmp);
            } else {
                device const block_q8_0 * kb = (device const block_q8_0 *) (k_tile + (uint64_t) row * args.nb11) + (i16 / 2);
                dequantize_q8_0(kb, i16 % 2, tmp);
            }
            threadgroup half4 * dst_k = (threadgroup half4 *) (skv + row * DK + i16 * 16);
            dst_k[0] = tmp[0];
            dst_k[1] = tmp[1];
            dst_k[2] = tmp[2];
            dst_k[3] = tmp[3];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Q * K^T on M5 Neural Accelerator
        mm_qk.run(tq, tk, ts);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tile-level k_pos range check for qq=sgitg
        const bool v0 = (ic + tiisg) < args.ne11;
        const bool v1 = (ic + tiisg + NW) < args.ne11;
        int kp0 = ic + tiisg;
        int kp1 = ic + tiisg + NW;
        bool tile_has_rel = false;

        if (FC_flash_attn_ext_tensor_has_banded && sg_q_valid) {
            int kp_min = ic;
            int kp_max = min(ic + C - 1, args.ne11 - 1);
            if (FC_flash_attn_ext_tensor_has_rel_pos) {
                device const int * kp = (device const int *) k_pos + (uint64_t) iq3 * args.ne11 + ic;
                kp0 = v0 ? kp[tiisg]      : 0;
                kp1 = v1 ? kp[tiisg + NW] : 0;
                const int t_min = min(v0 ? kp0 : INT_MAX, v1 ? kp1 : INT_MAX);
                const int t_max = max(v0 ? kp0 : INT_MIN, v1 ? kp1 : INT_MIN);
                kp_min = simd_min(t_min);
                kp_max = simd_max(t_max);
            }
            tile_has_rel = (qp_sg >= kp_min) && (qp_sg - kp_max < args.rel_extent);
        }

        const float m0_log2 = m0 * LOG2E;
        const float m1_log2 = m1 * LOG2E;
        const int rd0 = (FC_flash_attn_ext_tensor_has_banded && tile_has_rel && v0) ? (qp_sg - kp0) : -1;
        const int rd1 = (FC_flash_attn_ext_tensor_has_banded && tile_has_rel && v1) ? (qp_sg - kp1) : -1;
        const bool ib0 = rd0 >= 0 && rd0 < args.rel_extent;
        const bool ib1 = rd1 >= 0 && rd1 < args.rel_extent;

        // Online softmax for H heads of token qq = sgitg
        FOR_UNROLL (short hh = 0; hh < H; ++hh) {
            const short r = sgitg * H + hh;
            float s0 = ss[r * C + tiisg];
            float s1 = ss[r * C + tiisg + NW];

            if (FC_flash_attn_ext_tensor_has_scap) {
                s0 = scap_log2 * precise::tanh(s0 * args.scale);
                s1 = scap_log2 * precise::tanh(s1 * args.scale);
            }

            s0 = force_ninf0 ? (-FLT_MAX / 2) : (s0 + m0_log2);
            s1 = force_ninf1 ? (-FLT_MAX / 2) : (s1 + m1_log2);

            if (FC_flash_attn_ext_tensor_has_banded && (ib0 || ib1)) {
                device const float * pr_h = (device const float *) (
                    rel_base + (uint64_t) iq1_sg * args.nb62 + (uint64_t) hh * args.nb61);
                if (ib0) {
                    s0 += pr_h[rd0] * LOG2E;
                }
                if (ib1) {
                    s1 += pr_h[rd1] * LOG2E;
                }
            }

            const float m = simd_max(max(M[hh], max(s0, s1)));
            float ms = 1.0f;
            if (m > M[hh] + 11.0f) {
                ms    = exp2(M[hh] - m);
                M[hh] = m;
                if (tiisg == 0) {
                    s_ctl[4] = ic0 + 1;
                }
            }

            const half p0 = (half) exp2(s0 - M[hh]);
            const half p1 = (half) exp2(s1 - M[hh]);
            sp[r * C + tiisg]      = p0;
            sp[r * C + tiisg + NW] = p1;

            S[hh] = S[hh] * ms + simd_sum((float) p0 + (float) p1);
            if (tiisg == 0) {
                sr[r] = ms;
            }
        }

        // Inline Q8_0/Q4_0 V dequantize into skv (free after mm_qk barrier)
        FOR_UNROLL (short vit = 0; vit < 4; ++vit) {
            const short work = vit * NT + tiitg;
            const short row  = work / DK16;
            const short i16  = work % DK16;
            half4x4 tmp;
            if constexpr (IS_Q4_0) {
                device const block_q4_0 * vb = (device const block_q4_0 *) (v_tile + (uint64_t) row * args.nb21) + (i16 / 2);
                dequantize_q4_0(vb, i16 % 2, tmp);
            } else {
                device const block_q8_0 * vb = (device const block_q8_0 *) (v_tile + (uint64_t) row * args.nb21) + (i16 / 2);
                dequantize_q8_0(vb, i16 % 2, tmp);
            }
            threadgroup half4 * dst_v = (threadgroup half4 *) (skv + row * DV + i16 * 16);
            dst_v[0] = tmp[0];
            dst_v[1] = tmp[1];
            dst_v[2] = tmp[2];
            dst_v[3] = tmp[3];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (s_ctl[4] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        mm_pv.run(tp, tv, co);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply sinks on iwg == 0 and prepare final per-row scale sr[r]
    FOR_UNROLL (short hh = 0; hh < H; ++hh) {
        const short r = sgitg * H + hh;
        float ms = 1.0f;
        if (FC_flash_attn_ext_tensor_has_sinks && iwg == 0) {
            const int iq1_s = min(iq1_sg, args.ne01 - 1);
            const float s_val = ((device const float *) sinks)[(iq2_base + hh) + iq1_s * args.sinks_s1 + iq3 * args.sinks_s2] * LOG2E;
            const float m = max(M[hh], s_val);
            ms    = exp2(M[hh] - m);
            M[hh] = m;
            S[hh] = S[hh] * ms + exp2(s_val - m);
        }
        if (tiisg == 0) {
            sr[r] = (NWG == 1) ? (S[hh] == 0.0f ? 0.0f : ms / S[hh]) : ms;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage cooperative tensor O [M_PAD, DV] (8 KB) into free skv (16 KB), then write bid_tmp layout
    threadgroup float * so = (threadgroup float *) skv;
    auto to = tensor(so, dextents<int32_t, 2>(DV, M_PAD));
    co.store(to);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int64_t nrows = (int64_t) args.ne3 * args.ne2 * args.ne1;
    device float4 * dst4 = (device float4 *) dst;
    device float  * dst1 = (device float  *) dst + nrows * DV * NWG;

    if (sg_q_valid) {
        FOR_UNROLL (short hh = 0; hh < H; ++hh) {
            const short r = sgitg * H + hh;
            const int64_t rid = (int64_t) iq3 * args.ne2 * args.ne1 + (iq2_base + hh) + (int64_t) iq1_sg * args.ne1;
            const float scale_o = sr[r];
            dst4[rid * DV4 * NWG + NWG * tiisg + iwg] = ((threadgroup float4 *) (so + r * DV))[tiisg] * scale_o;
            if (NWG > 1 && tiisg == 0) {
                dst1[rid * (2 * NWG) + 2 * iwg + 0] = S[hh];
                dst1[rid * (2 * NWG) + 2 * iwg + 1] = M[hh] * LN2;
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_vec_tensor_q8_0_gqa<4, 3, 16, false>) flash_attn_ext_vec_tensor_q8_0_gqa_t;

template [[host_name("kernel_flash_attn_ext_vec_tensor_q8_0_gqa_dk128_dv128_q4_h3")]] kernel flash_attn_ext_vec_tensor_q8_0_gqa_t kernel_flash_attn_ext_vec_tensor_q8_0_gqa<4, 3, 16, false>;
template [[host_name("kernel_flash_attn_ext_vec_tensor_q8_0_gqa_dk128_dv128_q4_h4")]] kernel flash_attn_ext_vec_tensor_q8_0_gqa_t kernel_flash_attn_ext_vec_tensor_q8_0_gqa<4, 4, 16, false>;
template [[host_name("kernel_flash_attn_ext_vec_tensor_q8_0_gqa_dk128_dv128_q4_h2")]] kernel flash_attn_ext_vec_tensor_q8_0_gqa_t kernel_flash_attn_ext_vec_tensor_q8_0_gqa<4, 2, 16, false>;
template [[host_name("kernel_flash_attn_ext_vec_tensor_q4_0_gqa_dk128_dv128_q4_h3")]] kernel flash_attn_ext_vec_tensor_q8_0_gqa_t kernel_flash_attn_ext_vec_tensor_q8_0_gqa<4, 3, 16, true>;
template [[host_name("kernel_flash_attn_ext_vec_tensor_q4_0_gqa_dk128_dv128_q4_h4")]] kernel flash_attn_ext_vec_tensor_q8_0_gqa_t kernel_flash_attn_ext_vec_tensor_q8_0_gqa<4, 4, 16, true>;
template [[host_name("kernel_flash_attn_ext_vec_tensor_q4_0_gqa_dk128_dv128_q4_h2")]] kernel flash_attn_ext_vec_tensor_q8_0_gqa_t kernel_flash_attn_ext_vec_tensor_q8_0_gqa<4, 2, 16, true>;

#endif // GGML_METAL_HAS_TENSOR
