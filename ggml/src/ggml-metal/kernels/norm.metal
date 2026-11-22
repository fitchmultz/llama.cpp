#include "common.h"

constant bool FC_norm_use_scale [[function_constant(FC_NORM + 0)]];

// F == 1 : norm (no fuse)
// F == 2 : norm + mul
// F == 3 : norm + mul + add
template <typename T, short F>
kernel void kernel_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    T sumft(0.0f);

    float sumf = 0.0f;

    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumft += x[i00];
    }
    sumf = dot(sumft, T(1.0f));
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean = sumf/args.ne00;

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);

    sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        y[i00] = x[i00] - mean;
        sumf += dot(y[i00], y[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float variance = sumf/args.ne00;

    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (y[i00]*scale);
        }
        if (F == 2) {
            if (FC_norm_use_scale) {
                y[i00] = (y[i00]*scale) * args.scale;
            } else {
                y[i00] = (y[i00]*scale)*f0[i00];
            }
        }
        if (F == 3) {
            y[i00] = (y[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_norm_fuse_impl<float4, 1>) kernel_norm_fuse_t;

template [[host_name("kernel_norm_f32")]]         kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 1>;
template [[host_name("kernel_norm_mul_f32")]]     kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 2>;
template [[host_name("kernel_norm_mul_add_f32")]] kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float, 3>;

template [[host_name("kernel_norm_f32_4")]]         kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_norm_mul_f32_4")]]     kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 2>;
template [[host_name("kernel_norm_mul_add_f32_4")]] kernel kernel_norm_fuse_t kernel_norm_fuse_impl<float4, 3>;

// F == 1 : rms_norm (no fuse)
// F == 2 : rms_norm + mul
// F == 3 : rms_norm + mul + add
template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            if (FC_norm_use_scale) {
                y[i00] = (x[i00]*scale) * args.scale;
            } else {
                y[i00] = (x[i00]*scale)*f0[i00];
            }
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;

template [[host_name("kernel_rms_norm_f32")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 1>;
template [[host_name("kernel_rms_norm_mul_f32")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;
template [[host_name("kernel_rms_norm_mul_add_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 3>;

template [[host_name("kernel_rms_norm_f32_4")]]         kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_rms_norm_mul_f32_4")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 2>;
template [[host_name("kernel_rms_norm_mul_add_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 3>;

template <typename T0, typename T>
kernel void kernel_l2_norm_impl(
        constant ggml_metal_kargs_l2_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const T0 * x = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T  * y = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f/max(sqrt(sumf), args.eps);

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        y[i00] = x[i00] * scale;
    }
}

typedef decltype(kernel_l2_norm_impl<float, float>) kernel_l2_norm_t;

template [[host_name("kernel_l2_norm_f32_f32")]]   kernel kernel_l2_norm_t kernel_l2_norm_impl<float,  float>;
template [[host_name("kernel_l2_norm_f32_f32_4")]] kernel kernel_l2_norm_t kernel_l2_norm_impl<float4, float4>;

template <typename T, bool HAS_SCALE>
kernel void kernel_l2_clip_impl(
        constant ggml_metal_kargs_l2_clip & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const T * s = HAS_SCALE ? (device const T *) (src1 + (i03%args.ne13)*args.nb13 + (i02%args.ne12)*args.nb12 + (i01%args.ne11)*args.nb11) : nullptr;
    device       T * y = (device       T *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    if (args.ne00 <= 32) {
        T v = T(0.0f);
        if ((int) tiisg < args.ne00) {
            v = HAS_SCALE ? (x[tiisg] * s[tiisg]) : x[tiisg];
        }
        const float sumf = simd_sum(dot(v, v));
        if ((int) tiisg < args.ne00) {
            const float factor = clamp(args.threshold / (sqrt(sumf) + args.eps), 0.0f, 1.0f);
            y[tiisg] = v * factor;
        }
        return;
    }

    float sumf = 0.0f;
    for (int i00 = tiisg; i00 < args.ne00; i00 += 32) {
        T v = HAS_SCALE ? (x[i00] * s[i00]) : x[i00];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    const float factor = clamp(args.threshold / (sqrt(sumf) + args.eps), 0.0f, 1.0f);

    for (int i00 = tiisg; i00 < args.ne00; i00 += 32) {
        T v = HAS_SCALE ? (x[i00] * s[i00]) : x[i00];
        y[i00] = v * factor;
    }
}

typedef decltype(kernel_l2_clip_impl<float, false>) kernel_l2_clip_t;

template [[host_name("kernel_l2_clip_f32")]]         kernel kernel_l2_clip_t kernel_l2_clip_impl<float,  false>;
template [[host_name("kernel_l2_clip_scale_f32")]]   kernel kernel_l2_clip_t kernel_l2_clip_impl<float,  true>;
template [[host_name("kernel_l2_clip_f32_4")]]       kernel kernel_l2_clip_t kernel_l2_clip_impl<float4, false>;
template [[host_name("kernel_l2_clip_scale_f32_4")]] kernel kernel_l2_clip_t kernel_l2_clip_impl<float4, true>;

kernel void kernel_loop_control_f32_4(
        constant int64_t & n4,
        device const float4 * f_lin,
        device const float4 * g_lin,
        device const float4 * cur,
        device const float4 * msg,
        device       float4 * dst,
        uint gid [[thread_position_in_grid]]) {
    if ((int64_t) gid < n4) {
        dst[gid] = (1.0f - tanh(f_lin[gid])) * cur[gid] + tanh(g_lin[gid]) * msg[gid];
    }
}

kernel void kernel_mul_sigmoid_f32_4(
        constant int64_t & n4,
        device const float4 * x,
        device const float4 * g,
        device       float4 * dst,
        uint gid [[thread_position_in_grid]]) {
    if ((int64_t) gid < n4) {
        dst[gid] = x[gid] / (1.0f + exp(-g[gid]));
    }
}

kernel void kernel_situ_f32_4(
        constant ggml_metal_kargs_situ & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float4 * gx = (device const float4 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const float4 * ux = (device const float4 *) (src1 + i03*args.nb13 + i02*args.nb12 + i01*args.nb11);
    device       float4 * y  = (device       float4 *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const float4 g = gx[i0];
        const float4 u = ux[i0];
        const float4 a = (args.beta * tanh(g * args.inv_beta)) / (1.0f + exp(-g));
        const float4 b = args.lb * tanh(u * args.inv_lb);
        y[i0] = a * b;
    }
}

template <bool HAS_NORM>
kernel void kernel_depth_mix_impl(
        constant ggml_metal_kargs_depth_mix & args,
        device const char * blocks,
        device const char * cur,
        device const char * query,
        device       char * dst,
        device const char * norm_w,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    const int tok    = tgpig.x;
    const int j      = tgpig.y;
    const int k_prev = args.k_prev;
    const int K      = k_prev + 1;

    device const float4 * cur_row = (device const float4 *) (cur   + tok * args.nb11);
    device const float4 * q_row   = (device const float4 *) (query + j   * args.nb21);

    // 1. Single-pass: accumulate ||cur||^2, dot(cur, q), and dot(blocks[k], q) before one barrier
    float ss = 0.0f;
    float s_cur = 0.0f;
    float4 cn[4];
    int cidx = 0;
    for (int i = tpitg.x; i < args.ne00; i += 256) {
        const float4 v = cur_row[i];
        const float4 q = q_row[i];
        ss    += dot(v, v);
        s_cur += dot(v, q);
        if (cidx < 4) {
            cn[cidx] = v;
        }
        cidx++;
    }

    float sc[16];
    sc[k_prev] = simd_sum(s_cur);
    ss         = simd_sum(ss);

    for (int k = 0; k < k_prev; ++k) {
        device const float4 * b_row = (device const float4 *) (blocks + tok * args.nb02 + k * args.nb01);
        float sk = 0.0f;
        for (int i = tpitg.x; i < args.ne00; i += 256) {
            sk += dot(b_row[i], q_row[i]);
        }
        sc[k] = simd_sum(sk);
    }

    // shmem[sg * 17 + 0..K-1] = simdgroup dot partials; shmem[sg * 17 + 16] = simdgroup ss partial
    // shmem[8 * 17 + 0..7] = optional HAS_NORM output sum-of-squares partials (separate slots -> no extra pre-barrier!)
    threadgroup float shmem[8 * 17 + 8];
    if (tiisg == 0) {
        for (int k = 0; k < K; ++k) {
            shmem[sgitg * 17 + k] = sc[k];
        }
        shmem[sgitg * 17 + 16] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float tot_ss =
        shmem[0*17 + 16] + shmem[1*17 + 16] + shmem[2*17 + 16] + shmem[3*17 + 16] +
        shmem[4*17 + 16] + shmem[5*17 + 16] + shmem[6*17 + 16] + shmem[7*17 + 16];
    const float inv_rms = rsqrt(tot_ss / float(args.ne00 * 4) + args.eps);

    float w[16];
    float smax = -INFINITY;
    for (int k = 0; k < K; ++k) {
        float tot =
            shmem[0*17 + k] + shmem[1*17 + k] + shmem[2*17 + k] + shmem[3*17 + k] +
            shmem[4*17 + k] + shmem[5*17 + k] + shmem[6*17 + k] + shmem[7*17 + k];
        if (k == k_prev) {
            tot *= inv_rms;
        }
        w[k] = tot * args.scale;
        smax = max(smax, w[k]);
    }
    float wsum = 0.0f;
    for (int k = 0; k < K; ++k) {
        w[k] = exp(w[k] - smax);
        wsum += w[k];
    }
    const float inv_wsum = 1.0f / wsum;
    for (int k = 0; k < K; ++k) {
        w[k] *= inv_wsum;
    }
    const float w_cur = w[k_prev] * inv_rms;

    device float4 * out_row = (device float4 *) (dst + tok * args.nb2 + j * args.nb1);
    if (HAS_NORM) {
        float4 acc_reg[4];
        float out_ss = 0.0f;
        int ci = 0;
        for (int i = tpitg.x; i < args.ne00; i += 256) {
            float4 acc = w_cur * ((ci < 4) ? cn[ci] : cur_row[i]);
            for (int k = 0; k < k_prev; ++k) {
                device const float4 * b_row = (device const float4 *) (blocks + tok * args.nb02 + k * args.nb01);
                acc += w[k] * b_row[i];
            }
            acc_reg[ci] = acc;
            out_ss += dot(acc, acc);
            ci++;
        }
        out_ss = simd_sum(out_ss);
        if (tiisg == 0) {
            shmem[8 * 17 + sgitg] = out_ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tot_out_ss =
            shmem[8*17 + 0] + shmem[8*17 + 1] + shmem[8*17 + 2] + shmem[8*17 + 3] +
            shmem[8*17 + 4] + shmem[8*17 + 5] + shmem[8*17 + 6] + shmem[8*17 + 7];
        const float out_inv_rms = rsqrt(tot_out_ss / float(args.ne00 * 4) + args.norm_eps);
        device const float4 * nw_row = (device const float4 *) norm_w;
        ci = 0;
        for (int i = tpitg.x; i < args.ne00; i += 256) {
            out_row[i] = acc_reg[ci] * out_inv_rms * nw_row[i];
            ci++;
        }
    } else {
        int ci = 0;
        for (int i = tpitg.x; i < args.ne00; i += 256) {
            float4 acc = w_cur * ((ci < 4) ? cn[ci] : cur_row[i]);
            for (int k = 0; k < k_prev; ++k) {
                device const float4 * b_row = (device const float4 *) (blocks + tok * args.nb02 + k * args.nb01);
                acc += w[k] * b_row[i];
            }
            out_row[i] = acc;
            ci++;
        }
    }
}

typedef decltype(kernel_depth_mix_impl<false>) kernel_depth_mix_t;
template [[host_name("kernel_depth_mix_f32_4")]]      kernel kernel_depth_mix_t kernel_depth_mix_impl<false>;
template [[host_name("kernel_depth_mix_norm_f32_4")]] kernel kernel_depth_mix_t kernel_depth_mix_impl<true>;

kernel void kernel_concat_rms_norm_f32_4(
        constant ggml_metal_kargs_concat_rms_norm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    const int half_idx = tgpig.x;
    const int tok      = tgpig.y;

    device const float4 * x = (device const float4 *) ((half_idx == 0) ? (src0 + tok * args.nb02) : (src1 + tok * args.nb12));
    device       float4 * y = (device       float4 *) (dst + tok * args.nb2 + half_idx * args.nb1);

    float ss = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += 256) {
        const float4 v = x[i];
        ss += dot(v, v);
    }
    ss = simd_sum(ss);

    threadgroup float shmem[9];
    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float t = (tiisg < 8) ? shmem[tiisg] : 0.0f;
        t = simd_sum(t);
        if (tiisg == 0) {
            shmem[8] = rsqrt(t / float(args.ne00 * 4) + args.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = shmem[8];

    for (int i = tpitg.x; i < args.ne00; i += 256) {
        y[i] = x[i] * inv_rms;
    }
}

#if defined(GGML_METAL_HAS_BF16)
template <bool MULTI_TOK>
kernel void kernel_loop_down_norm_bf16_f32_impl(
        constant ggml_metal_kargs_loop_down_norm & args,
        device const char * src_a,
        device const char * src_b,
        device const char * w_down,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    const int r = (int) tgpig.x;
    if (r >= args.ne01) {
        return;
    }
    const int n4 = args.ne00;
    device const bfloat4 * wa4 = (device const bfloat4 *) (w_down + (uint64_t) r * args.nb_w1);
    device const bfloat4 * wb4 = wa4 + n4;

    if (!MULTI_TOK) {
        device const float4 * a4 = (device const float4 *) src_a;
        device const float4 * b4 = (device const float4 *) src_b;

        float ss_a = 0.0f;
        float ss_b = 0.0f;
        float da   = 0.0f;
        float db   = 0.0f;

        for (int i = tpitg.x; i < n4; i += 128) {
            const float4 av = a4[i];
            const float4 bv = b4[i];
            ss_a += dot(av, av);
            ss_b += dot(bv, bv);
            da   += dot(float4(wa4[i]), av);
            db   += dot(float4(wb4[i]), bv);
        }

        ss_a = simd_sum(ss_a);
        ss_b = simd_sum(ss_b);
        da   = simd_sum(da);
        db   = simd_sum(db);

        threadgroup float shmem[4 * 4];
        if (tiisg == 0) {
            shmem[sgitg * 4 + 0] = ss_a;
            shmem[sgitg * 4 + 1] = ss_b;
            shmem[sgitg * 4 + 2] = da;
            shmem[sgitg * 4 + 3] = db;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg == 0) {
            const float tot_ssa = shmem[0] + shmem[4] + shmem[8]  + shmem[12];
            const float tot_ssb = shmem[1] + shmem[5] + shmem[9]  + shmem[13];
            const float tot_da  = shmem[2] + shmem[6] + shmem[10] + shmem[14];
            const float tot_db  = shmem[3] + shmem[7] + shmem[11] + shmem[15];

            const float inv_d = 1.0f / float(n4 * 4);
            const float inv_a = rsqrt(tot_ssa * inv_d + args.eps);
            const float inv_b = rsqrt(tot_ssb * inv_d + args.eps);

            ((device float *) dst)[r] = tot_da * inv_a + tot_db * inv_b;
        }
    } else {
        constexpr int BT = 4;
        const int t0 = (int) tgpig.y * BT;
        const int t_cnt = min(BT, args.n_tokens - t0);
        if (t_cnt <= 0) {
            return;
        }

        device const float4 * a4[BT];
        device const float4 * b4[BT];
        FOR_UNROLL (int k = 0; k < BT; ++k) {
            const int tk = min(t0 + k, args.n_tokens - 1);
            a4[k] = (device const float4 *) (src_a + (uint64_t) tk * args.nb_a2);
            b4[k] = (device const float4 *) (src_b + (uint64_t) tk * args.nb_b2);
        }

        float ss_a[BT] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float ss_b[BT] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float da[BT]   = { 0.0f, 0.0f, 0.0f, 0.0f };
        float db[BT]   = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (int i = tpitg.x; i < n4; i += 128) {
            const float4 wva = float4(wa4[i]);
            const float4 wvb = float4(wb4[i]);
            FOR_UNROLL (int k = 0; k < BT; ++k) {
                if (k < t_cnt) {
                    const float4 av = a4[k][i];
                    const float4 bv = b4[k][i];
                    ss_a[k] += dot(av, av);
                    ss_b[k] += dot(bv, bv);
                    da[k]   += dot(wva, av);
                    db[k]   += dot(wvb, bv);
                }
            }
        }

        threadgroup float shmem[4 * BT * 4];
        FOR_UNROLL (int k = 0; k < BT; ++k) {
            if (k < t_cnt) {
                const float sa = simd_sum(ss_a[k]);
                const float sb = simd_sum(ss_b[k]);
                const float xa = simd_sum(da[k]);
                const float xb = simd_sum(db[k]);
                if (tiisg == 0) {
                    const int base = (k * 4 + sgitg) * 4;
                    shmem[base + 0] = sa;
                    shmem[base + 1] = sb;
                    shmem[base + 2] = xa;
                    shmem[base + 3] = xb;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0 && tiisg < (ushort) t_cnt) {
            const int k = tiisg;
            const int base = k * 16;
            const float tot_ssa = shmem[base + 0] + shmem[base + 4] + shmem[base + 8]  + shmem[base + 12];
            const float tot_ssb = shmem[base + 1] + shmem[base + 5] + shmem[base + 9]  + shmem[base + 13];
            const float tot_da  = shmem[base + 2] + shmem[base + 6] + shmem[base + 10] + shmem[base + 14];
            const float tot_db  = shmem[base + 3] + shmem[base + 7] + shmem[base + 11] + shmem[base + 15];

            const float inv_d = 1.0f / float(n4 * 4);
            const float inv_a = rsqrt(tot_ssa * inv_d + args.eps);
            const float inv_b = rsqrt(tot_ssb * inv_d + args.eps);

            ((device float *) (dst + (uint64_t) (t0 + k) * args.nb_d1))[r] = tot_da * inv_a + tot_db * inv_b;
        }
    }
}

typedef decltype(kernel_loop_down_norm_bf16_f32_impl<false>) kernel_loop_down_norm_t;
template [[host_name("kernel_loop_down_norm_bf16_f32_t1")]] kernel kernel_loop_down_norm_t kernel_loop_down_norm_bf16_f32_impl<false>;
template [[host_name("kernel_loop_down_norm_bf16_f32_tn")]] kernel kernel_loop_down_norm_t kernel_loop_down_norm_bf16_f32_impl<true>;

kernel void kernel_loop_control_mm_bf16_f32(
        constant ggml_metal_kargs_loop_control_mm & args,
        device const char * w_gain,
        device const char * w_forget,
        device const char * h,
        device const char * cur,
        device const char * msg,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int tok     = tgpig.y;
    const int half_id = tiisg / 16;
    const int lane    = tiisg % 16;
    const int row0    = (tgpig.x * 8 + sgitg * 2 + half_id) * 2;

    device const float4 * h4   = (device const float4 *) (h   + tok * args.nb_h1);
    device const float  * crow = (device const float  *) (cur + tok * args.nb_c1);
    device const float  * mrow = (device const float  *) (msg + tok * args.nb_m1);
    device       float  * drow = (device       float  *) (dst + tok * args.nb_d1);

    const float4 hv = h4[lane];

    FOR_UNROLL (int r = 0; r < 2; ++r) {
        const int row = row0 + r;
        if (row < args.ne01) {
            device const bfloat4 * wg4 = (device const bfloat4 *) (w_gain   + row * args.nb_g1);
            device const bfloat4 * wf4 = (device const bfloat4 *) (w_forget + row * args.nb_f1);

            float dg = dot(float4(wg4[lane]), hv);
            float df = dot(float4(wf4[lane]), hv);

            dg += simd_shuffle_xor(dg, 8);
            dg += simd_shuffle_xor(dg, 4);
            dg += simd_shuffle_xor(dg, 2);
            dg += simd_shuffle_xor(dg, 1);

            df += simd_shuffle_xor(df, 8);
            df += simd_shuffle_xor(df, 4);
            df += simd_shuffle_xor(df, 2);
            df += simd_shuffle_xor(df, 1);

            if (lane == 0) {
                drow[row] = (1.0f - tanh(df)) * crow[row] + tanh(dg) * mrow[row];
            }
        }
    }
}
#endif

constant bool FC_group_norm_flat [[function_constant(FC_GROUP_NORM + 0)]];

kernel void kernel_group_norm_f32(
        constant ggml_metal_kargs_group_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float  * buf [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t ne00 = args.ne00;
    const int64_t ne01 = args.ne01;
    const int64_t ne02 = args.ne02;

    const int64_t ngrp = args.ngrp;

    const int64_t i03 = tgpig.z; // batch index
    const int64_t ig  = tgpig.x; // group index

    const int64_t cpg = (ne02 + ngrp - 1)/ngrp;

    const int64_t c0 =  ig*cpg;
    const int64_t c1 = (c0 + cpg < ne02) ? c0 + cpg : ne02;

    const int64_t npf  = ne00*ne01;      // elements per channel
    const int64_t gs   = npf*(c1 - c0);  // elements in this group

    device const char * x03 = src0 + i03*args.nb03 + c0*args.nb02;
    device       char * y03 = dst  + i03*args.nb3  + c0*args.nb2;

    float tmp = 0.0f; // partial sum for thread in warp

    if (FC_group_norm_flat) {
        device const float * x = (device const float *) x03;
        for (int64_t e = tpitg.x; e < gs; e += ntg.x) {
            tmp += x[e];
        }
    } else {
        for (int64_t e = tpitg.x; e < gs; e += ntg.x) {
            const int64_t i02 = c0 + e/npf;
            const int64_t ir  = e - (i02 - c0)*npf;
            const int64_t i01 = ir/ne00;
            const int64_t i00 = ir - i01*ne00;

            tmp += *(device const float *)(x03 + (i02 - c0)*args.nb02 + i01*args.nb01 + i00*sizeof(float));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    if (FC_group_norm_flat) {
        device const float * x = (device const float *) x03;
        device       float * y = (device       float *) y03;
        for (int64_t e = tpitg.x; e < gs; e += ntg.x) {
            const float xi = x[e] - mean;
            y[e] = xi;
            tmp += xi*xi;
        }
    } else {
        for (int64_t e = tpitg.x; e < gs; e += ntg.x) {
            const int64_t i02 = c0 + e/npf;
            const int64_t ir  = e - (i02 - c0)*npf;
            const int64_t i01 = ir/ne00;
            const int64_t i00 = ir - i01*ne00;

            const float xi = *(device const float *)(x03 + (i02 - c0)*args.nb02 + i01*args.nb01 + i00*sizeof(float)) - mean;

            *(device float *)(y03 + (i02 - c0)*args.nb2 + i01*args.nb1 + i00*sizeof(float)) = xi;

            tmp += xi*xi;
        }
    }

    tmp = simd_sum(tmp);
    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + args.eps);

    if (FC_group_norm_flat) {
        device float * y = (device float *) y03;
        for (int64_t e = tpitg.x; e < gs; e += ntg.x) {
            y[e] *= scale;
        }
    } else {
        for (int64_t e = tpitg.x; e < gs; e += ntg.x) {
            const int64_t i02 = c0 + e/npf;
            const int64_t ir  = e - (i02 - c0)*npf;
            const int64_t i01 = ir/ne00;
            const int64_t i00 = ir - i01*ne00;

            *(device float *)(y03 + (i02 - c0)*args.nb2 + i01*args.nb1 + i00*sizeof(float)) *= scale;
        }
    }
}
