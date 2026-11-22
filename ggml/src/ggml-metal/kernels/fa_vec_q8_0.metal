#include "common.h"
#include "dequantize.h"
#include "fa_vec_common.metal"

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 4>;

// Register-resident, deferred-reduction inline-Q8_0 FlashAttention vector kernel for DK=128, DV=128.
// Processes Q query tokens x H query heads sharing one KV head per threadgroup:
// - Q is staged once in threadgroup memory (shared across all NSG simdgroups)
// - Q*K^T uses NE=4 (4 parallel K rows per simdgroup, 8 lanes per 32-elem Q8_0 block, 3-stage shuffle)
// - Softmax uses hardware exp2 (LOG2E folded into scale/mask/rel/sinks; M converted back to ln for vec_reduce)
// - P*V uses NE_V in {1, 2, 4} parallel V rows with O kept entirely in registers and NE_V reduction
//   deferred ONCE to after the KV loop (zero shmem O traffic, zero P*V shuffles inside the loop).
template<short Q, short H, short NE_V = 1>
kernel void kernel_flash_attn_ext_vec_q8_0_gqa(
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
        device const char * rel_q,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    (void) idx;

#define NWG (FC_flash_attn_ext_vec_nwg)
#define NSG (FC_flash_attn_ext_vec_nsg)

    constexpr short DV    = 128;
    constexpr short DV4   = DV / 4; // 32 == N_SIMDWIDTH
    constexpr short NL_V  = DV4 / NE_V;
    constexpr short NV4   = NE_V; // DV4 / NL_V
    constexpr short C     = OP_FLASH_ATTN_EXT_VEC_NCPSG; // 32
    constexpr short QH    = Q * H;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;

    static_assert(NE_V == 1 || NE_V == 2 || NE_V == 4, "NE_V must be 1, 2, or 4");

    const short  iwg      = tgpig[2] % NWG;
    const ushort iq3      = tgpig[2] / NWG;
    const ushort iq2_blk  = tgpig[1];
    const ushort iq1      = tgpig[0];
    const int    iq2_base = iq2_blk * H;

    const int ikv2 = iq2_base / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3      / (args.ne03 / args.ne_12_3);

    device const char * k_base = k + (uint64_t) ikv2 * args.nb12 + (uint64_t) ikv3 * args.nb13;
    device const char * v_base = v + (uint64_t) ikv2 * args.nb22 + (uint64_t) ikv3 * args.nb23;

    // Shared memory: sq [QH, 32] half4 (shared by all NSG) + ss [NSG, QH, C] float
    threadgroup half4 * sq      = (threadgroup half4 *) shmem_f16;
    threadgroup float * ss_base = (threadgroup float *) (sq + QH * DV4);
    threadgroup float * ss      = ss_base + sgitg * QH * C;

    // Stage Q*H query vectors into shared memory cooperatively once
    {
        const short tiitg = sgitg * 32 + tiisg;
        const short ntg   = NSG * 32;
        for (short idx_q = tiitg; idx_q < QH * DV4; idx_q += ntg) {
            const short m_idx = idx_q / DV4;
            const short i4    = idx_q % DV4;
            const short qq    = m_idx / H;
            const short hh    = m_idx % H;
            const int   iq1_q = iq1 * Q + qq;
            if (iq1_q < args.ne01) {
                device const float4 * q4 = (device const float4 *) (
                    q + (uint64_t) iq1_q * args.nb01 + (uint64_t) (iq2_base + hh) * args.nb02 + (uint64_t) iq3 * args.nb03);
                sq[idx_q] = (half4) q4[i4];
            } else {
                sq[idx_q] = half4(0.0h);
            }
        }
    }

    float4 o_reg[QH][NV4];
    float  M[QH];
    float  S[QH];

    FOR_UNROLL (short m = 0; m < QH; ++m) {
        FOR_UNROLL (short ii = 0; ii < NV4; ++ii) {
            o_reg[m][ii] = float4(0.0f);
        }
        M[m] = -FLT_MAX / 2;
        S[m] = 0.0f;
    }

    // Hoist only compact int q_pos outside the KV loop (recompute rel pointers only on the ~1% in-band tiles)
    int qp_val[Q];
    int qp_min = INT_MAX;
    int qp_max = INT_MIN;

    if (FC_flash_attn_ext_vec_has_banded) {
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1 * Q + qq;
            const int iq1_r = min(iq1_q, args.ne01 - 1);
            if (FC_flash_attn_ext_vec_has_rel_pos) {
                qp_val[qq] = ((device const int *) q_pos)[(uint64_t) iq3 * args.ne01 + iq1_r];
            } else {
                qp_val[qq] = iq1_r + (args.ne11 - args.ne01);
            }
            if (iq1_q < args.ne01) {
                qp_min = min(qp_min, qp_val[qq]);
                qp_max = max(qp_max, qp_val[qq]);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Mask base (when H > 1, host guarantees ne32 == 1; when H == 1, handle arbitrary ne32)
    device const char * pm_stream  = mask  + (uint64_t) (iq2_base % args.ne32) * args.nb32 + (uint64_t) (iq3 % args.ne33) * args.nb33;
    device const char * rel_base   = rel   + (uint64_t) (iq3 % args.ne63) * args.nb63 + (uint64_t) iq2_base * args.nb61;
    device const char * rel_q_base = rel_q + (uint64_t) (iq3 % args.ne63) * args.nb63 + (uint64_t) iq2_base * args.nb61;

    const float scale_log2 = args.scale * LOG2E;
    const float scap_log2  = args.logit_softcap * LOG2E;

    const short tx   = tiisg & 7;       // 0..7 within Q8_0 block (8 x char4 = 32 elems)
    const short ty   = tiisg >> 3;      // 0..3 (4 K rows processed in parallel)
    const short tx_v = tiisg % NL_V;
    const short ty_v = tiisg / NL_V;

    // Main KV cache loop: each simdgroup independently walks strided C-row tiles
    for (int ic0 = iwg * NSG + sgitg; ; ic0 += NWG * NSG) {
        const int ic_orig = ic0 * C;
        if (ic_orig >= args.ne11) {
            break;
        }

        const bool is_pad = FC_flash_attn_ext_vec_has_kvpad && (ic_orig + C > args.ne11);
        device const char * k_cur = k_base;
        device const char * v_cur = v_base;
        int ic_cur = ic_orig;

        if (is_pad) {
            k_cur  = pad + (uint64_t) (ikv2 + ikv3 * args.ne_12_2) * args.nb11 * C;
            v_cur  = pad + (uint64_t) args.nb11 * C * args.ne_12_2 * args.ne_12_3
                         + (uint64_t) (ikv2 + ikv3 * args.ne_12_2) * args.nb21 * C;
            ic_cur = 0;
        }

        // Load per-query mask values directly into registers (shared across all H heads)
        float m_val[Q];
        bool any_finite = false;

        if (FC_flash_attn_ext_vec_has_mask) {
            if (is_pad) {
                device const half * mask_pad = (device const half *) (
                    pad + (uint64_t) (args.nb11 + args.nb21) * C * args.ne_12_2 * args.ne_12_3)
                    + (uint64_t) (iq2_base % args.ne32) * (C * args.ne31)
                    + (uint64_t) (iq3 % args.ne33) * (C * args.ne31 * args.ne32);
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    const int iq1_q = iq1 * Q + qq;
                    m_val[qq] = (iq1_q < args.ne01) ? (float) mask_pad[iq1_q * C + tiisg] : (float) (-MAXHALF);
                }
            } else {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    const int iq1_q = iq1 * Q + qq;
                    if (iq1_q < args.ne01) {
                        device const half * pm_row = (device const half *) (pm_stream + (uint64_t) iq1_q * args.nb31);
                        m_val[qq] = (float) pm_row[ic_orig + tiisg];
                    } else {
                        m_val[qq] = (float) (-MAXHALF);
                    }
                }
            }
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                if (simd_max(m_val[qq]) > (float) (-MAXHALF)) {
                    any_finite = true;
                }
            }
        } else {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const int iq1_q = iq1 * Q + qq;
                if (iq1_q >= args.ne01 || (is_pad && (ic_orig + tiisg >= args.ne11))) {
                    m_val[qq] = (float) (-MAXHALF);
                } else {
                    m_val[qq] = 0.0f;
                    any_finite = true;
                }
            }
            any_finite = simd_any(any_finite);
        }

        if (!any_finite) {
            continue;
        }

        // Q * K^T with NE=4 (4 parallel K rows per iteration, 8 iterations total, 3-stage shuffle)
        FOR_UNROLL (short cc = 0; cc < C / 4; ++cc) {
            device const block_q8_0 * pk = (device const block_q8_0 *) (k_cur + (uint64_t) (ic_cur + 4 * cc + ty) * args.nb11);

            float acc[QH];
            FOR_UNROLL (short m = 0; m < QH; ++m) {
                acc[m] = 0.0f;
            }

            FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                device const block_q8_0 * pb = pk + ii;
                const float  d  = (float) pb->d;
                const float4 kf = (float4) (((device const packed_char4 *) pb->qs)[tx]) * d;
                const short  q_off = ii * 8 + tx;

                FOR_UNROLL (short m = 0; m < QH; ++m) {
                    acc[m] += dot(kf, (float4) sq[m * DV4 + q_off]);
                }
            }

            FOR_UNROLL (short m = 0; m < QH; ++m) {
                acc[m] += simd_shuffle_down(acc[m], 4);
                acc[m] += simd_shuffle_down(acc[m], 2);
                acc[m] += simd_shuffle_down(acc[m], 1);
                if (tx == 0) {
                    ss[m * C + 4 * cc + ty] = acc[m];
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Tile-level relative position range check (skips 99% of tiles when rel_extent << ne11)
        const bool col_valid = (ic_orig + tiisg) < args.ne11;
        int kp_val = ic_orig + tiisg;
        bool tile_has_rel = false;
        if (FC_flash_attn_ext_vec_has_banded) {
            int kp_min = ic_orig;
            int kp_max = min(ic_orig + C - 1, args.ne11 - 1);
            if (FC_flash_attn_ext_vec_has_rel_pos) {
                kp_val = col_valid ? ((device const int *) k_pos)[(uint64_t) iq3 * args.ne11 + ic_orig + tiisg] : 0;
                kp_min = simd_min(col_valid ? kp_val : INT_MAX);
                kp_max = simd_max(col_valid ? kp_val : INT_MIN);
            }
            tile_has_rel = (qp_max >= kp_min) && (qp_min - kp_max < args.rel_extent);
        }

        // Online softmax in log2 space using single-cycle hardware exp2
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const float m_log2 = m_val[qq] * LOG2E;
            const int iq1_q = iq1 * Q + qq;
            int rel_dist = -1;
            bool in_band = false;
            if (FC_flash_attn_ext_vec_has_banded && tile_has_rel && iq1_q < args.ne01 && col_valid) {
                rel_dist = qp_val[qq] - kp_val;
                in_band  = (rel_dist >= 0 && rel_dist < args.rel_extent);
            }

            FOR_UNROLL (short hh = 0; hh < H; ++hh) {
                const short m_idx = qq * H + hh;
                float s = ss[m_idx * C + tiisg];

                if (FC_flash_attn_ext_vec_has_scap) {
                    s = scap_log2 * precise::tanh(s * args.scale) + m_log2;
                } else {
                    s = fma(s, scale_log2, m_log2);
                }

                if (FC_flash_attn_ext_vec_has_banded && in_band) {
                    if (FC_flash_attn_ext_vec_has_rel_mm) {
                        device const float4 * rq = (device const float4 *) (
                            rel_q_base + (uint64_t) iq1_q * args.nb62 + (uint64_t) hh * args.nb61);
                        device const float4 * rw = (device const float4 *) rel + (uint64_t) rel_dist * 4;
                        s += (dot(rq[0], rw[0]) + dot(rq[1], rw[1]) + dot(rq[2], rw[2]) + dot(rq[3], rw[3])) * LOG2E;
                    } else {
                        device const float * pr_h = (device const float *) (
                            rel_base + (uint64_t) iq1_q * args.nb62 + (uint64_t) hh * args.nb61);
                        s += pr_h[rel_dist] * LOG2E;
                    }
                }

                const float m_old = M[m_idx];
                const float m_new = simd_max(max(m_old, s));
                M[m_idx] = m_new;

                const float ms = exp2(m_old - m_new);
                const float vs = exp2(s - m_new);

                S[m_idx] = S[m_idx] * ms + simd_sum(vs);
                ss[m_idx * C + tiisg] = vs;

                if (ms != 1.0f) {
                    FOR_UNROLL (short ii = 0; ii < NV4; ++ii) {
                        o_reg[m_idx][ii] *= ms;
                    }
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P * V: NE_V parallel V rows in registers, reduction deferred until after KV loop!
        FOR_UNROLL (short cc = 0; cc < C / NE_V; ++cc) {
            const short row = NE_V * cc + ty_v;
            device const block_q8_0 * pv = (device const block_q8_0 *) (v_cur + (uint64_t) (ic_cur + row) * args.nb21);

            FOR_UNROLL (short ii = 0; ii < NV4; ++ii) {
                const short elem4 = ii * NL_V + tx_v;
                float4 vf;
                dequantize_q8_0_t4(pv + elem4 / 8, elem4 % 8, vf);

                FOR_UNROLL (short m = 0; m < QH; ++m) {
                    o_reg[m][ii] = fma(vf, ss[m * C + row], o_reg[m][ii]);
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Attention sinks (in log2 space, only once per query row: sgitg == 0 && iwg == 0)
    if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const int iq1_s = min((int) (iq1 * Q + qq), args.ne01 - 1);
            FOR_UNROLL (short hh = 0; hh < H; ++hh) {
                const short m_idx = qq * H + hh;
                const float m_old = M[m_idx];
                const float s_val = ((device const float *) sinks)[(iq2_base + hh) + iq1_s * args.sinks_s1 + iq3 * args.sinks_s2] * LOG2E;
                const float m_new = max(m_old, s_val);
                M[m_idx] = m_new;

                const float ms = exp2(m_old - m_new);
                const float vs = exp2(s_val - m_new);

                S[m_idx] = S[m_idx] * ms + vs;
                if (ms != 1.0f) {
                    FOR_UNROLL (short ii = 0; ii < NV4; ++ii) {
                        o_reg[m_idx][ii] *= ms;
                    }
                }
            }
        }
    }

    // Cross-simdgroup reduction one (qq, hh) at a time reusing shmem_f16 (exact bid_tmp layout for vec_reduce)
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float4 * red_o  = (threadgroup float4 *) shmem_f16;
    threadgroup float  * red_sm = (threadgroup float  *) (red_o + NSG * DV4);

    const int64_t nrows = (int64_t) args.ne3 * args.ne2 * args.ne1;
    device float4 * dst4 = (device float4 *) dst;
    device float  * dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
        const int iq1_q = iq1 * Q + qq;
        if (iq1_q >= args.ne01) {
            break;
        }
        FOR_UNROLL (short hh = 0; hh < H; ++hh) {
            const short m_idx = qq * H + hh;

            // Deferred NE_V reduction across ty_v (only executed ONCE after the entire KV loop!)
            FOR_UNROLL (short ii = 0; ii < NV4; ++ii) {
                float4 v = o_reg[m_idx][ii];
                if (NE_V >= 2) {
                    v[0] += simd_shuffle_down(v[0], 16);
                    v[1] += simd_shuffle_down(v[1], 16);
                    v[2] += simd_shuffle_down(v[2], 16);
                    v[3] += simd_shuffle_down(v[3], 16);
                }
                if (NE_V >= 4) {
                    v[0] += simd_shuffle_down(v[0], 8);
                    v[1] += simd_shuffle_down(v[1], 8);
                    v[2] += simd_shuffle_down(v[2], 8);
                    v[3] += simd_shuffle_down(v[3], 8);
                }
                if (ty_v == 0) {
                    red_o[sgitg * DV4 + ii * NL_V + tx_v] = v;
                }
            }

            if (tiisg == 0) {
                red_sm[2 * sgitg + 0] = S[m_idx];
                red_sm[2 * sgitg + 1] = M[m_idx];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                float Mx = red_sm[1];
                for (short s = 1; s < NSG; ++s) {
                    Mx = max(Mx, red_sm[2 * s + 1]);
                }

                float  Sx = 0.0f;
                float4 Ox = float4(0.0f);
                for (short s = 0; s < NSG; ++s) {
                    const float ms = exp2(red_sm[2 * s + 1] - Mx);
                    Sx += red_sm[2 * s + 0] * ms;
                    Ox = fma(red_o[s * DV4 + tiisg], ms, Ox);
                }

                const int64_t rid  = (int64_t) iq3 * args.ne2 * args.ne1 + (iq2_base + hh) + (int64_t) iq1_q * args.ne1;
                const float   Sval = (NWG == 1) ? (Sx == 0.0f ? 0.0f : 1.0f / Sx) : 1.0f;

                dst4[rid * DV4 * NWG + NWG * tiisg + iwg] = Ox * Sval;

                if (NWG > 1 && tiisg == 0) {
                    dst1[rid * (2 * NWG) + 2 * iwg + 0] = Sx;
                    // Convert log2 max back to natural log for kernel_flash_attn_ext_vec_reduce
                    dst1[rid * (2 * NWG) + 2 * iwg + 1] = Mx * LN2;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

#undef NWG
#undef NSG
}

typedef decltype(kernel_flash_attn_ext_vec_q8_0_gqa<1, 3, 4>) flash_attn_ext_vec_q8_0_gqa_t;

template [[host_name("kernel_flash_attn_ext_vec_q8_0_gqa_dk128_dv128_q1_h2")]] kernel flash_attn_ext_vec_q8_0_gqa_t kernel_flash_attn_ext_vec_q8_0_gqa<1, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_gqa_dk128_dv128_q1_h3")]] kernel flash_attn_ext_vec_q8_0_gqa_t kernel_flash_attn_ext_vec_q8_0_gqa<1, 3, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_gqa_dk128_dv128_q1_h4")]] kernel flash_attn_ext_vec_q8_0_gqa_t kernel_flash_attn_ext_vec_q8_0_gqa<1, 4, 4>;

#undef FA_TYPES
#undef FA_TYPES_F32

