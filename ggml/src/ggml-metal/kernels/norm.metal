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

kernel void kernel_depth_mix_f32_4(
        constant ggml_metal_kargs_depth_mix & args,
        device const char * blocks,
        device const char * cur,
        device const char * query,
        device       char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int tok = tgpig;
    device const float4 * cur_row = (device const float4 *) (cur + tok * args.nb11);

    // 1. RMS-normalize cur in registers (up to 4 float4s/thread = D <= 4096)
    float ss = 0.0f;
    float4 cn[4];
    int cidx = 0;
    for (int i = tpitg; i < args.ne00; i += 256) {
        const float4 v = cur_row[i];
        ss += dot(v, v);
        if (cidx < 4) {
            cn[cidx] = v;
        }
        cidx++;
    }
    ss = simd_sum(ss);

    // shmem[0..127] = 8 simdgroups x 16 depth scores; shmem[128..135] = RMS reduction; shmem[136] = inv_rms
    threadgroup float shmem[137];
    if (tiisg == 0) {
        shmem[128 + sgitg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float t = (tiisg < 8) ? shmem[128 + tiisg] : 0.0f;
        t = simd_sum(t);
        if (tiisg == 0) {
            shmem[136] = rsqrt(t / float(args.ne00 * 4) + args.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = shmem[136];
    for (int c = 0; c < cidx && c < 4; ++c) {
        cn[c] *= inv_rms;
    }

    const int k_prev = args.k_prev;
    const int K      = k_prev + 1;

    for (int j = 0; j < args.n_q; ++j) {
        device const float4 * q_row = (device const float4 *) (query + j * args.nb21);
        float sc[16];

        float s_cur = 0.0f;
        int ci = 0;
        for (int i = tpitg; i < args.ne00; i += 256) {
            const float4 v = (ci < 4) ? cn[ci] : (cur_row[i] * inv_rms);
            s_cur += dot(v, q_row[i]);
            ci++;
        }
        sc[k_prev] = simd_sum(s_cur);

        for (int k = 0; k < k_prev; ++k) {
            device const float4 * b_row = (device const float4 *) (blocks + tok * args.nb02 + k * args.nb01);
            float sk = 0.0f;
            for (int i = tpitg; i < args.ne00; i += 256) {
                sk += dot(b_row[i], q_row[i]);
            }
            sc[k] = simd_sum(sk);
        }

        if (tiisg == 0) {
            for (int k = 0; k < K; ++k) {
                shmem[sgitg * 16 + k] = sc[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float w[16];
        float smax = -INFINITY;
        for (int k = 0; k < K; ++k) {
            const float tot =
                shmem[0*16 + k] + shmem[1*16 + k] + shmem[2*16 + k] + shmem[3*16 + k] +
                shmem[4*16 + k] + shmem[5*16 + k] + shmem[6*16 + k] + shmem[7*16 + k];
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

        device float4 * out_row = (device float4 *) (dst + tok * args.nb2 + j * args.nb1);
        ci = 0;
        for (int i = tpitg; i < args.ne00; i += 256) {
            float4 acc = w[k_prev] * ((ci < 4) ? cn[ci] : (cur_row[i] * inv_rms));
            for (int k = 0; k < k_prev; ++k) {
                device const float4 * b_row = (device const float4 *) (blocks + tok * args.nb02 + k * args.nb01);
                acc += w[k] * b_row[i];
            }
            out_row[i] = acc;
            ci++;
        }

        if (j + 1 < args.n_q) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

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
