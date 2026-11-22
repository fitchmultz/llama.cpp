// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#pragma once

#include "ggml-impl.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#define GGML_METAL_FUSION_MAX 48

typedef enum ggml_metal_fusion_mode {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_FULL,
} ggml_metal_fusion_mode;

// identifier of each fusion pattern so the op encoders know which kernel to use
typedef enum ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_ADD, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    GGML_METAL_FUSION_ADD_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
    GGML_METAL_FUSION_SSM_CONV_RES,  // SSM_CONV + RESHAPE + ADD (residual) + SCALE
    GGML_METAL_FUSION_SSM_CONV_FULL, // CONCAT + CPY/SET_ROWS + SSM_CONV + RESHAPE + ADD + SCALE
    GGML_METAL_FUSION_L2_CLIP,       // [MUL +] SQR + SUM_ROWS + SQRT + SCALE + FILL + DIV + CLAMP + MUL
    GGML_METAL_FUSION_LOOP_CONTROL,  // TANH + SCALE(-1,1) + MUL + TANH + MUL + ADD
    GGML_METAL_FUSION_LOOP_CONTROL_MM, // MUL_MAT(gain) + MUL_MAT(forget) + LOOP_CONTROL
    GGML_METAL_FUSION_SITU,          // SCALE + TANH + SCALE + SIGMOID + MUL + [VIEW] + SCALE + TANH + SCALE + MUL
    GGML_METAL_FUSION_MUL_SIGMOID,   // SIGMOID + MUL
    GGML_METAL_FUSION_DEPTH_MIX,     // RMS_NORM + CONCAT + TRANSPOSE + CONT + [REPEAT] + MUL_MAT + SOFT_MAX + MUL_MAT [+ PERMUTE + CONT]
    GGML_METAL_FUSION_CONCAT_RMS_NORM, // CONCAT(dim=1, ne01=1, ne11=1) + RMS_NORM
    GGML_METAL_FUSION_LOOP_DOWN_NORM,  // CONCAT(dim=1) + RMS_NORM + RESHAPE + MUL_MAT(bf16[2D, R])
    GGML_METAL_FUSION_MUL_MAT_ID_SCALE, // MUL_MAT_ID + REPEAT + GET_ROWS + MUL (per-expert scalar weight scale)
    GGML_METAL_FUSION_MUL_MAT_ADD,   // MUL_MAT + ADD (residual added in the few-row MMA store)
    GGML_METAL_FUSION_MUL_SOFT_MAX,  // MUL(x, per-row scalar) + SOFT_MAX (large-vocab parallel softmax)
    GGML_METAL_FUSION_SOFTCAP,       // SCALE(1/cap) + TANH + SCALE(cap) (logit softcapping)
    GGML_METAL_FUSION_MOE_REDUCE_CONV_ADD, // MOE_REDUCE + SSM_CONV_FULL + ADD (register-resident FFN epilogue)
} ggml_metal_fusion_id;

struct ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

typedef struct ggml_metal_fusion ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
ggml_metal_fusion_id ggml_metal_fusion_get_id(const struct ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, struct ggml_tensor *, struct ggml_tensor *),
        struct ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
struct ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug);
void ggml_metal_fusion_info_free(struct ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(const struct ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (const struct ggml_metal_fusion_info * finfo);
int  ggml_metal_fusion_info_debug  (const struct ggml_metal_fusion_info * finfo);

int          ggml_metal_fusion_info_n_fusions(const struct ggml_metal_fusion_info * finfo);
const char * ggml_metal_fusion_info_label    (const struct ggml_metal_fusion_info * finfo, int idx);
uint64_t     ggml_metal_fusion_info_count    (const struct ggml_metal_fusion_info * finfo, int idx);

void ggml_metal_fusion_info_count_fusion(struct ggml_metal_fusion_info * finfo, const struct ggml_metal_fusion * fusion);
void ggml_metal_fusion_info_set_enabled (struct ggml_metal_fusion_info * finfo, bool enabled);

void ggml_metal_fusion_info_stats_init (      struct ggml_metal_fusion_info * finfo);
void ggml_metal_fusion_info_stats_reset(      struct ggml_metal_fusion_info * finfo);
int  ggml_metal_fusion_info_stats_get  (const struct ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n);
void ggml_metal_fusion_info_labels_init(      struct ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
const ggml_metal_fusion * ggml_metal_fusion_next(
        const struct ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
int ggml_metal_fusion_max(const struct ggml_cgraph * gf, int idx);

#ifdef __cplusplus
}
#endif
