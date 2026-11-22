#include "ggml-metal-fusion.h"

#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    for (ggml_op op : ops_all) {
        if (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    return ops;
}

struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                  const struct ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct ggml_metal_fusion   * fusion,
                          const struct ggml_tensor * const * nodes,
                          const struct ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion) {
    return fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
static bool ggml_metal_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    const ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    if (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        return false;
    }

    return bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    if (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        const ggml_tensor * scale = nodes[1];
        if (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// SSM_CONV + UNARY (silu)
static bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * un   = nodes[1];

    if (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        return false;
    }

    if (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

// SSM_CONV + RESHAPE + ADD (residual) + SCALE
static bool ggml_metal_fusion_check_ssm_conv_res(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);
    GGML_UNUSED(mode);

    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + 2];
    if (raw_end - raw_start != 3) {
        return false;
    }

    const ggml_tensor * conv = gf->nodes[raw_start + 0];
    const ggml_tensor * rsh  = gf->nodes[raw_start + 1];
    const ggml_tensor * add  = gf->nodes[raw_start + 2];
    const ggml_tensor * sc   = gf->nodes[raw_start + 3];

    if (conv->op != GGML_OP_SSM_CONV || rsh->op != GGML_OP_RESHAPE || add->op != GGML_OP_ADD || sc->op != GGML_OP_SCALE) {
        return false;
    }
    if (rsh->src[0] != conv || sc->src[0] != add) {
        return false;
    }
    if (ggml_get_op_params_f32(sc, 1) != 0.0f) {
        return false;
    }

    const ggml_tensor * cur = nullptr;
    if (add->src[0] == rsh) {
        cur = add->src[1];
    } else if (add->src[1] == rsh) {
        cur = add->src[0];
    } else {
        return false;
    }

    // prove s[nc - 1] == cur: conv->src[0] == concat(past[nc-1], transpose(reshape(cur)), dim=0)
    const ggml_tensor * cat = conv->src[0];
    if (!cat || cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0) {
        return false;
    }
    if (!cat->src[0] || !cat->src[1] || cat->src[0]->ne[0] != conv->src[1]->ne[0] - 1) {
        return false;
    }
    const ggml_tensor * tr = cat->src[1];
    if (tr->op != GGML_OP_TRANSPOSE || !tr->src[0]) {
        return false;
    }
    const ggml_tensor * cur_rsh = tr->src[0];
    if (cur_rsh != cur && !(cur_rsh->op == GGML_OP_RESHAPE && cur_rsh->src[0] == cur)) {
        return false;
    }

    if (conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(conv) || !ggml_is_contiguous(sc) || ggml_nelements(conv) != ggml_nelements(sc)) {
        return false;
    }

    const int raw_idxs[4] = { raw_start, raw_start + 1, raw_start + 2, raw_start + 3 };
    const int outputs[1]  = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, 4, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    // ssm_conv_res reads transposed cat [T+3, C] while writing sc [C, T], so sc must never alias cat
    if (mode == GGML_METAL_FUSION_FULL) {
        if (!sc->data || !cat->data || ggml_metal_fusion_overlap(sc, cat)) {
            return false;
        }
    }

    return true;
}

// [MUL +] SQR + SUM_ROWS + SQRT + SCALE(1,eps) + FILL(thresh) + DIV + CLAMP(0,1) + MUL
static bool ggml_metal_fusion_check_l2_clip(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const int n_ops = (int) fusion->ops.size();
    const bool has_scale = (n_ops == 9);
    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + n_ops - 1];
    if (raw_end - raw_start != n_ops - 1) {
        return false;
    }

    const int off = has_scale ? 1 : 0;
    const ggml_tensor * cur   = has_scale ? nodes[0] : nodes[0]->src[0];
    const ggml_tensor * x     = has_scale ? nodes[0]->src[0] : cur;
    const ggml_tensor * scale = has_scale ? nodes[0]->src[1] : nullptr;

    const ggml_tensor * sqr = nodes[off + 0];
    const ggml_tensor * sum = nodes[off + 1];
    const ggml_tensor * sq  = nodes[off + 2];
    const ggml_tensor * sc  = nodes[off + 3];
    const ggml_tensor * fil = nodes[off + 4];
    const ggml_tensor * dv  = nodes[off + 5];
    const ggml_tensor * clp = nodes[off + 6];
    const ggml_tensor * out = nodes[off + 7];

    if (!x || sqr->src[0] != cur || sum->src[0] != sqr || sq->src[0] != sum ||
        sc->src[0] != sq || fil->src[0] != sc || dv->src[0] != fil || dv->src[1] != sc ||
        clp->src[0] != dv || out->src[0] != cur || out->src[1] != clp) {
        return false;
    }

    if (ggml_get_op_params_f32(sc, 0) != 1.0f) {
        return false;
    }
    if (ggml_get_op_params_f32(clp, 0) != 0.0f || ggml_get_op_params_f32(clp, 1) != 1.0f) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous_rows(x) || !ggml_is_contiguous_rows(out) ||
        !ggml_are_same_shape(x, out)) {
        return false;
    }
    if (has_scale) {
        if (!scale || scale->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(scale) ||
            scale->ne[0] != x->ne[0] || x->ne[1] % scale->ne[1] != 0 ||
            x->ne[2] % scale->ne[2] != 0 || x->ne[3] % scale->ne[3] != 0) {
            return false;
        }
    }

    int raw_idxs[9];
    for (int i = 0; i < n_ops; ++i) {
        raw_idxs[i] = raw_start + i;
    }
    const int outputs[1] = { raw_end };
    return ggml_can_fuse_subgraph_ext(gf, raw_idxs, n_ops, fusion->ops_all.data(), outputs, 1);
}

// TANH(f_lin) + SCALE(-1,+1) + MUL(cur) + TANH(g_lin) + MUL(msg) + ADD
static bool ggml_metal_fusion_check_loop_control(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + 5];
    if (raw_end - raw_start != 5) {
        return false;
    }

    const ggml_tensor * t0  = nodes[0];
    const ggml_tensor * sc  = nodes[1];
    const ggml_tensor * m0  = nodes[2];
    const ggml_tensor * t1  = nodes[3];
    const ggml_tensor * m1  = nodes[4];
    const ggml_tensor * add = nodes[5];

    if (ggml_get_unary_op(t0) != GGML_UNARY_OP_TANH || ggml_get_unary_op(t1) != GGML_UNARY_OP_TANH) {
        return false;
    }
    if (sc->src[0] != t0 || ggml_get_op_params_f32(sc, 0) != -1.0f || ggml_get_op_params_f32(sc, 1) != 1.0f) {
        return false;
    }
    if ((m0->src[0] != sc && m0->src[1] != sc) || (m1->src[0] != t1 && m1->src[1] != t1)) {
        return false;
    }
    if (!((add->src[0] == m0 && add->src[1] == m1) || (add->src[0] == m1 && add->src[1] == m0))) {
        return false;
    }

    const ggml_tensor * f_lin = t0->src[0];
    const ggml_tensor * g_lin = t1->src[0];
    const ggml_tensor * cur   = (m0->src[0] == sc) ? m0->src[1] : m0->src[0];
    const ggml_tensor * msg   = (m1->src[0] == t1) ? m1->src[1] : m1->src[0];

    if (!f_lin || !g_lin || !cur || !msg) {
        return false;
    }
    if (f_lin->type != GGML_TYPE_F32 || g_lin->type != GGML_TYPE_F32 ||
        cur->type != GGML_TYPE_F32 || msg->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(f_lin) || !ggml_is_contiguous(g_lin) ||
        !ggml_is_contiguous(cur) || !ggml_is_contiguous(msg) || !ggml_is_contiguous(add)) {
        return false;
    }
    if (!ggml_are_same_shape(f_lin, add) || !ggml_are_same_shape(g_lin, add) ||
        !ggml_are_same_shape(cur, add) || !ggml_are_same_shape(msg, add)) {
        return false;
    }
    if (ggml_nelements(add) % 4 != 0) {
        return false;
    }

    const int raw_idxs[6] = { raw_start, raw_start + 1, raw_start + 2, raw_start + 3, raw_start + 4, raw_start + 5 };
    const int outputs[1]  = { raw_end };
    return ggml_can_fuse_subgraph_ext(gf, raw_idxs, 6, fusion->ops_all.data(), outputs, 1);
}

// SITU: SCALE(1/b) + TANH + SCALE(b) + SIGMOID + MUL + [VIEW] + SCALE(1/lb) + TANH + SCALE(lb) + MUL
static bool ggml_metal_fusion_check_situ(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + 8];
    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) fusion->ops_all.size()) {
        return false;
    }

    int raw_idxs[10];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * s0  = nodes[0];
    const ggml_tensor * t0  = nodes[1];
    const ggml_tensor * s1  = nodes[2];
    const ggml_tensor * sig = nodes[3];
    const ggml_tensor * m0  = nodes[4];
    const ggml_tensor * s2  = nodes[5];
    const ggml_tensor * t1  = nodes[6];
    const ggml_tensor * s3  = nodes[7];
    const ggml_tensor * m1  = nodes[8];

    const ggml_tensor * gate = s0->src[0];
    const ggml_tensor * up   = s2->src[0];
    if (!gate || !up) {
        return false;
    }
    if (ggml_get_op_params_f32(s0, 1) != 0.0f || ggml_get_op_params_f32(s1, 1) != 0.0f ||
        ggml_get_op_params_f32(s2, 1) != 0.0f || ggml_get_op_params_f32(s3, 1) != 0.0f) {
        return false;
    }
    if (ggml_get_unary_op(t0) != GGML_UNARY_OP_TANH || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID ||
        ggml_get_unary_op(t1) != GGML_UNARY_OP_TANH) {
        return false;
    }
    if (t0->src[0] != s0 || s1->src[0] != t0 || sig->src[0] != gate ||
        !((m0->src[0] == s1 && m0->src[1] == sig) || (m0->src[0] == sig && m0->src[1] == s1)) ||
        t1->src[0] != s2 || s3->src[0] != t1 ||
        !((m1->src[0] == m0 && m1->src[1] == s3) || (m1->src[0] == s3 && m1->src[1] == m0))) {
        return false;
    }
    if (gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || m1->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous_rows(gate) || !ggml_is_contiguous_rows(up) || !ggml_is_contiguous_rows(m1) ||
        !ggml_are_same_shape(gate, up) || !ggml_are_same_shape(gate, m1) ||
        gate->ne[0] % 4 != 0) {
        return false;
    }

    const int outputs[1] = { raw_end };
    return ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, fusion->ops_all.data(), outputs, 1);
}

// SIGMOID + MUL (e.g. attn_gate)
static bool ggml_metal_fusion_check_mul_sigmoid(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * sig = nodes[0];
    const ggml_tensor * mul = nodes[1];
    if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const ggml_tensor * g = sig->src[0];
    const ggml_tensor * x = (mul->src[0] == sig) ? mul->src[1] : mul->src[0];
    if (!g || !x) {
        return false;
    }
    if (g->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(x) || !ggml_is_contiguous(mul) ||
        !ggml_are_same_shape(g, mul) || !ggml_are_same_shape(x, mul) ||
        ggml_nelements(mul) % 4 != 0) {
        return false;
    }
    return true;
}

// DEPTH_MIX: RMS_NORM + CONCAT + TRANSPOSE + CONT + [REPEAT] + MUL_MAT + SOFT_MAX + MUL_MAT [+ PERMUTE + CONT | + RESHAPE + RMS_NORM + MUL]
static bool ggml_metal_fusion_check_depth_mix(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int n_ops      = (int) fusion->ops.size();
    const bool has_norm  = (nodes[n_ops - 1]->op == GGML_OP_MUL);
    const bool has_tr    = (nodes[n_ops - 1]->op == GGML_OP_CONT);
    const bool has_rep   = (fusion->ops[3] == GGML_OP_REPEAT);
    const int raw_start  = node_idxs[idx];
    const int raw_end    = node_idxs[idx + n_ops - 1];
    const int raw_count  = raw_end - raw_start + 1;
    if (raw_count != (int) fusion->ops_all.size()) {
        return false;
    }

    int raw_idxs[12];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * rms = nodes[0];
    const ggml_tensor * cat = nodes[1];
    const ggml_tensor * tr  = gf->nodes[raw_start + 2];
    const ggml_tensor * cnt = nodes[2];
    const ggml_tensor * rep = has_rep ? nodes[3] : nullptr;
    const ggml_tensor * mm0 = nodes[has_rep ? 4 : 3];
    const ggml_tensor * sm  = nodes[has_rep ? 5 : 4];
    const ggml_tensor * mm1 = nodes[has_rep ? 6 : 5];
    const ggml_tensor * dst = nodes[n_ops - 1];

    const ggml_tensor * cur    = rms->src[0];
    const ggml_tensor * blocks = cat->src[0];
    const ggml_tensor * query  = has_rep ? rep->src[0] : mm0->src[1];
    if (!cur || !blocks || !query) {
        return false;
    }
    if (ggml_get_op_params_i32(cat, 0) != 1 || cat->src[1] != rms || tr->src[0] != cat || cnt->src[0] != tr) {
        return false;
    }
    if (has_rep && (mm0->src[0] != cat || mm0->src[1] != rep)) {
        return false;
    }
    if (!has_rep && mm0->src[0] != cat) {
        return false;
    }
    if (sm->src[0] != mm0 || sm->src[1] || sm->src[2] || ggml_get_op_params_f32(sm, 1) != 0.0f) {
        return false;
    }
    if (mm1->src[0] != cnt || mm1->src[1] != sm) {
        return false;
    }

    const int64_t D = cur->ne[0];
    const int64_t T = cur->ne[2];
    const int64_t K_prev = blocks->ne[1];
    const int64_t n_q    = query->ne[1];
    if (D <= 0 || D % 4 != 0 || cur->ne[1] != 1 || cur->ne[3] != 1 ||
        blocks->ne[0] != D || K_prev < 1 || K_prev > 15 || blocks->ne[2] != T || blocks->ne[3] != 1 ||
        query->ne[0] != D || n_q < 1 || n_q > 64 || query->ne[2] != 1 || query->ne[3] != 1 ||
        mm1->ne[0] != D || mm1->ne[1] != n_q || mm1->ne[2] != T || mm1->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * norm_w = nullptr;
    if (has_tr) {
        const ggml_tensor * perm = gf->nodes[raw_start + 8];
        if (perm->op != GGML_OP_PERMUTE || perm->src[0] != mm1 || dst->op != GGML_OP_CONT || dst->src[0] != perm) {
            return false;
        }
        if (perm->ne[0] != D || perm->ne[1] != T || perm->ne[2] != n_q || perm->ne[3] != 1 ||
            perm->nb[0] != mm1->nb[0] || perm->nb[1] != mm1->nb[2] || perm->nb[2] != mm1->nb[1] || perm->nb[3] != mm1->nb[3] ||
            dst->ne[0] != D || dst->ne[1] != T || dst->ne[2] != n_q || dst->ne[3] != 1) {
            return false;
        }
    } else if (has_norm) {
        if (n_q != 1 || D > 4096) {
            return false;
        }
        const ggml_tensor * rsh     = gf->nodes[raw_end - 2];
        const ggml_tensor * out_rms = nodes[n_ops - 2];
        const ggml_tensor * out_mul = nodes[n_ops - 1];
        if (rsh->op != GGML_OP_RESHAPE || rsh->src[0] != mm1 ||
            rsh->ne[0] != D || rsh->ne[1] != T || rsh->ne[2] != 1 || rsh->ne[3] != 1) {
            return false;
        }
        if (out_rms->op != GGML_OP_RMS_NORM || out_rms->src[0] != rsh || out_rms->type != GGML_TYPE_F32) {
            return false;
        }
        if (out_mul->op != GGML_OP_MUL || (out_mul->src[0] != out_rms && out_mul->src[1] != out_rms)) {
            return false;
        }
        norm_w = (out_mul->src[0] == out_rms) ? out_mul->src[1] : out_mul->src[0];
        if (!norm_w || norm_w->type != GGML_TYPE_F32 || !ggml_is_contiguous(norm_w) ||
            norm_w->ne[0] != D || norm_w->ne[1] != 1 || norm_w->ne[2] != 1 || norm_w->ne[3] != 1) {
            return false;
        }
        if (dst->ne[0] != D || dst->ne[1] != T || dst->ne[2] != 1 || dst->ne[3] != 1) {
            return false;
        }
    }
    if (blocks->type != GGML_TYPE_F32 || cur->type != GGML_TYPE_F32 ||
        query->type != GGML_TYPE_F32 || mm1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(blocks) || !ggml_is_contiguous(cur) ||
        !ggml_is_contiguous(query) || !ggml_is_contiguous(mm1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int outputs[1] = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!dst->data || !blocks->data || !cur->data || !query->data ||
            ggml_metal_fusion_overlap(dst, blocks) || ggml_metal_fusion_overlap(dst, cur) ||
            ggml_metal_fusion_overlap(dst, query)) {
            return false;
        }
        if (has_norm && (!norm_w->data || ggml_metal_fusion_overlap(dst, norm_w))) {
            return false;
        }
    }

    return true;
}

// CONCAT(dim=1, ne01=1, ne11=1) + RMS_NORM (e.g. loop control)
static bool ggml_metal_fusion_check_concat_rms_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * cat = nodes[0];
    const ggml_tensor * rms = nodes[1];
    if (ggml_get_op_params_i32(cat, 0) != 1 || rms->src[0] != cat) {
        return false;
    }
    const ggml_tensor * a = cat->src[0];
    const ggml_tensor * b = cat->src[1];
    if (!a || !b) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(rms)) {
        return false;
    }
    if (a->ne[0] % 4 != 0 || a->ne[0] != b->ne[0] || a->ne[1] != 1 || b->ne[1] != 1 ||
        a->ne[2] != b->ne[2] || a->ne[3] != 1 || b->ne[3] != 1 || rms->ne[1] != 2) {
        return false;
    }
    return true;
}

// CONCAT(dim=1) + RMS_NORM + RESHAPE + MUL_MAT(bf16[2*D, R]) (loop controller down-projection)
static bool ggml_metal_fusion_check_loop_down_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + 2];
    if (raw_end - raw_start != 3) {
        return false;
    }
    int raw_idxs[4] = { raw_start, raw_start + 1, raw_start + 2, raw_start + 3 };
    for (int i = 0; i < 4; ++i) {
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * cat = nodes[0];
    const ggml_tensor * rms = nodes[1];
    const ggml_tensor * rsh = gf->nodes[raw_start + 2];
    const ggml_tensor * mm  = nodes[2];

    if (ggml_get_op_params_i32(cat, 0) != 1 || rms->src[0] != cat || rsh->src[0] != rms || mm->src[1] != rsh) {
        return false;
    }
    const ggml_tensor * a = cat->src[0];
    const ggml_tensor * b = cat->src[1];
    const ggml_tensor * w = mm->src[0];
    if (!a || !b || !w) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        w->type != GGML_TYPE_BF16 || mm->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(w) || !ggml_is_contiguous(mm)) {
        return false;
    }
    const int64_t D = a->ne[0];
    const int64_t T = a->ne[2];
    if (D <= 0 || D % 4 != 0 || b->ne[0] != D || a->ne[1] != 1 || b->ne[1] != 1 ||
        b->ne[2] != T || a->ne[3] != 1 || b->ne[3] != 1 || rms->ne[1] != 2) {
        return false;
    }
    if (rsh->ne[0] != 2 * D || rsh->ne[1] != T || rsh->ne[2] != 1 || rsh->ne[3] != 1 ||
        w->ne[0] != 2 * D || w->ne[2] != 1 || w->ne[3] != 1 ||
        mm->ne[0] != w->ne[1] || mm->ne[1] != T || mm->ne[2] != 1 || mm->ne[3] != 1) {
        return false;
    }

    const int outputs[1] = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, 4, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!a->data || !b->data || !w->data || !mm->data ||
            ggml_metal_fusion_overlap(mm, a) || ggml_metal_fusion_overlap(mm, b)) {
            return false;
        }
    }

    return true;
}

// MUL_MAT(w_gain, h) + MUL_MAT(w_forget, h) + TANH + SCALE(-1,+1) + MUL(cur) + TANH + MUL(msg) + ADD
static bool ggml_metal_fusion_check_loop_control_mm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + 7];
    if (raw_end - raw_start != 7) {
        return false;
    }

    const ggml_tensor * mm_g = nodes[0];
    const ggml_tensor * mm_f = nodes[1];
    const ggml_tensor * t0   = nodes[2];
    const ggml_tensor * sc   = nodes[3];
    const ggml_tensor * m0   = nodes[4];
    const ggml_tensor * t1   = nodes[5];
    const ggml_tensor * m1   = nodes[6];
    const ggml_tensor * add  = nodes[7];

    if (ggml_get_unary_op(t0) != GGML_UNARY_OP_TANH || ggml_get_unary_op(t1) != GGML_UNARY_OP_TANH) {
        return false;
    }
    if (sc->src[0] != t0 || ggml_get_op_params_f32(sc, 0) != -1.0f || ggml_get_op_params_f32(sc, 1) != 1.0f) {
        return false;
    }
    if ((m0->src[0] != sc && m0->src[1] != sc) || (m1->src[0] != t1 && m1->src[1] != t1)) {
        return false;
    }
    if (!((add->src[0] == m0 && add->src[1] == m1) || (add->src[0] == m1 && add->src[1] == m0))) {
        return false;
    }

    // t0 = tanh(f_lin), t1 = tanh(g_lin) where f_lin/g_lin are mm_f/mm_g in either order
    const ggml_tensor * f_lin = t0->src[0];
    const ggml_tensor * g_lin = t1->src[0];
    if (!((g_lin == mm_g && f_lin == mm_f) || (g_lin == mm_f && f_lin == mm_g))) {
        return false;
    }

    const ggml_tensor * w_gain   = g_lin->src[0];
    const ggml_tensor * w_forget = f_lin->src[0];
    const ggml_tensor * h        = g_lin->src[1];
    if (!w_gain || !w_forget || !h || f_lin->src[1] != h) {
        return false;
    }

    const ggml_tensor * cur = (m0->src[0] == sc) ? m0->src[1] : m0->src[0];
    const ggml_tensor * msg = (m1->src[0] == t1) ? m1->src[1] : m1->src[0];
    if (!cur || !msg) {
        return false;
    }

    if (w_gain->type != GGML_TYPE_BF16 || w_forget->type != GGML_TYPE_BF16 ||
        h->type != GGML_TYPE_F32 || cur->type != GGML_TYPE_F32 ||
        msg->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
        return false;
    }
    if (w_gain->ne[0] != 64 || w_forget->ne[0] != 64 || h->ne[0] != 64 ||
        w_gain->ne[1] != w_forget->ne[1] || w_gain->ne[2] != 1 || w_gain->ne[3] != 1 ||
        w_forget->ne[2] != 1 || w_forget->ne[3] != 1 || h->ne[2] != 1 || h->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(w_gain) || !ggml_is_contiguous(w_forget) || !ggml_is_contiguous(h) ||
        !ggml_is_contiguous(cur) || !ggml_is_contiguous(msg) || !ggml_is_contiguous(add)) {
        return false;
    }
    if (!ggml_are_same_shape(cur, add) || !ggml_are_same_shape(msg, add) ||
        add->ne[0] != w_gain->ne[1] || add->ne[1] != h->ne[1] || add->ne[2] != 1 || add->ne[3] != 1) {
        return false;
    }

    const int raw_idxs[8] = {
        raw_start, raw_start + 1, raw_start + 2, raw_start + 3,
        raw_start + 4, raw_start + 5, raw_start + 6, raw_start + 7
    };
    const int outputs[1] = { raw_end };
    return ggml_can_fuse_subgraph_ext(gf, raw_idxs, 8, fusion->ops_all.data(), outputs, 1);
}

// CONCAT(past, tr(cur)) + CPY/SET_ROWS(state) + SSM_CONV + RESHAPE [+ dynamic 4-tap lags] + ADD(cur) + SCALE [+ ADD(residual)]
static bool ggml_metal_fusion_check_ssm_conv_full(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int  n_ops   = (int) fusion->ops.size();
    const bool has_dyn = (n_ops == 16);
    const bool has_add = (n_ops == 6);

    const ggml_tensor * cat   = nodes[0];
    const ggml_tensor * store = nodes[1];
    const ggml_tensor * conv  = nodes[2];
    const ggml_tensor * add   = nodes[has_dyn ? 14 : 3];
    const ggml_tensor * sc    = nodes[has_dyn ? 15 : 4];
    const ggml_tensor * out   = has_add ? nodes[5] : sc;

    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0) {
        return false;
    }
    const bool is_set = (store->op == GGML_OP_SET_ROWS);
    const bool is_cpy = (store->op == GGML_OP_CPY);
    if (!is_set && !is_cpy) {
        return false;
    }
    if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != cat || !conv->src[1]) {
        return false;
    }
    if (conv->src[1]->ne[0] != 4 || conv->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(conv->src[1])) {
        return false;
    }
    if (sc->op != GGML_OP_SCALE || sc->src[0] != add || ggml_get_op_params_f32(sc, 1) != 0.0f) {
        return false;
    }

    const ggml_tensor * res_in = nullptr;
    if (has_add) {
        if (out->op != GGML_OP_ADD || (out->src[0] != sc && out->src[1] != sc)) {
            return false;
        }
        res_in = (out->src[0] == sc) ? out->src[1] : out->src[0];
        if (!res_in || res_in->type != GGML_TYPE_F32 || !ggml_is_contiguous(res_in) ||
            !ggml_are_same_shape(res_in, sc) || out->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(out) || !ggml_are_same_shape(out, sc)) {
            return false;
        }
    }

    const ggml_tensor * past = cat->src[0];
    const ggml_tensor * tr   = cat->src[1];
    if (!past || !tr || past->ne[0] != 3 || past->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(past)) {
        return false;
    }
    if (tr->op != GGML_OP_TRANSPOSE || !tr->src[0]) {
        return false;
    }
    const ggml_tensor * cur_rsh = tr->src[0];
    const ggml_tensor * cur = (cur_rsh->op == GGML_OP_RESHAPE && cur_rsh->src[0]) ? cur_rsh->src[0] : cur_rsh;

    if (cur->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 || store->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(cur) || !ggml_is_contiguous(sc) || !ggml_are_same_shape(cur, sc)) {
        return false;
    }

    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seqs       = past->ne[2];
    const int64_t n_seq_tokens = cat->ne[0] - 3;
    const int64_t n_tokens     = n_seq_tokens * n_seqs;
    if (n_channels <= 0 || n_seqs <= 0 || n_seq_tokens <= 0 ||
        past->ne[1] != n_channels || cur->ne[1] != n_tokens ||
        cur->ne[2] != 1 || cur->ne[3] != 1 || past->ne[3] != 1) {
        return false;
    }

    const ggml_tensor * rsh     = nullptr;
    const ggml_tensor * dynamic = nullptr;

    if (!has_dyn) {
        if (add->src[0] == cur && add->src[1] && add->src[1]->op == GGML_OP_RESHAPE && add->src[1]->src[0] == conv) {
            rsh = add->src[1];
        } else if (add->src[1] == cur && add->src[0] && add->src[0]->op == GGML_OP_RESHAPE && add->src[0]->src[0] == conv) {
            rsh = add->src[0];
        } else {
            return false;
        }
    } else {
        const ggml_tensor * mul0 = nodes[3];
        const ggml_tensor * add0 = nodes[4];
        if (!add0->src[0] || add0->src[0]->op != GGML_OP_RESHAPE || add0->src[0]->src[0] != conv ||
            !add0->src[1] || add0->src[1]->op != GGML_OP_RESHAPE || add0->src[1]->src[0] != mul0) {
            return false;
        }
        rsh = add0->src[0];
        if (!mul0->src[0] || mul0->src[0]->op != GGML_OP_RESHAPE || mul0->src[0]->src[0] != cur ||
            !mul0->src[1] || mul0->src[1]->op != GGML_OP_VIEW || !mul0->src[1]->src[0] ||
            mul0->src[1]->view_offs != 0) {
            return false;
        }
        const ggml_tensor * dyn_rsh = mul0->src[1]->src[0];
        if (dyn_rsh->op != GGML_OP_RESHAPE || !dyn_rsh->src[0] || dyn_rsh->ne[0] != 4 || dyn_rsh->ne[2] != n_tokens) {
            return false;
        }
        dynamic = dyn_rsh->src[0];
        const int64_t n_heads = dyn_rsh->ne[1];
        if (n_heads <= 0 || n_channels % n_heads != 0 ||
            dynamic->type != GGML_TYPE_F32 || !ggml_is_contiguous(dynamic) ||
            dynamic->ne[0] != 4 * n_heads || dynamic->ne[1] != n_tokens ||
            dynamic->ne[2] != 1 || dynamic->ne[3] != 1) {
            return false;
        }
        if (mul0->src[1]->ne[0] != 1 || mul0->src[1]->ne[1] != n_heads || mul0->src[1]->ne[2] != n_tokens) {
            return false;
        }

        for (int lag = 1; lag < 4; ++lag) {
            const ggml_tensor * prev_add = nodes[4 + (lag - 1) * 3];
            const ggml_tensor * cnt_l    = nodes[5 + (lag - 1) * 3];
            const ggml_tensor * mul_l    = nodes[6 + (lag - 1) * 3];
            const ggml_tensor * add_l    = nodes[7 + (lag - 1) * 3];

            if (add_l->src[0] != prev_add || !add_l->src[1] ||
                add_l->src[1]->op != GGML_OP_RESHAPE || add_l->src[1]->src[0] != mul_l) {
                return false;
            }
            if (!mul_l->src[0] || mul_l->src[0]->op != GGML_OP_RESHAPE || mul_l->src[0]->src[0] != cnt_l) {
                return false;
            }
            if (!cnt_l->src[0] || cnt_l->src[0]->op != GGML_OP_TRANSPOSE ||
                !cnt_l->src[0]->src[0] || cnt_l->src[0]->src[0]->op != GGML_OP_VIEW) {
                return false;
            }
            const ggml_tensor * win_v = cnt_l->src[0]->src[0];
            if (win_v->view_src != cat || win_v->ne[0] != n_seq_tokens || win_v->ne[1] != n_channels ||
                win_v->ne[2] != n_seqs || win_v->view_offs != (size_t) (3 - lag) * cat->nb[0]) {
                return false;
            }
            const ggml_tensor * tap = mul_l->src[1];
            if (!tap || tap->op != GGML_OP_VIEW || tap->src[0] != dyn_rsh ||
                tap->ne[0] != 1 || tap->ne[1] != n_heads || tap->ne[2] != n_tokens ||
                tap->view_offs != (size_t) lag * dyn_rsh->nb[0]) {
                return false;
            }
        }

        const ggml_tensor * add3 = nodes[13];
        if (!((add->src[0] == cur && add->src[1] == add3) || (add->src[0] == add3 && add->src[1] == cur))) {
            return false;
        }
    }

    if (is_cpy) {
        const ggml_tensor * last = store->src[0];
        if (!last || last->op != GGML_OP_VIEW || last->view_src != cat ||
            last->ne[0] != 3 || last->ne[1] != n_channels || last->ne[2] != n_seqs ||
            last->view_offs != (size_t) n_seq_tokens * sizeof(float)) {
            return false;
        }
        if (store->ne[0] != 3 * n_channels || store->ne[1] != n_seqs || store->ne[2] != 1 || store->ne[3] != 1 ||
            store->nb[0] != sizeof(float)) {
            return false;
        }
    } else {
        const ggml_tensor * windows    = store->src[0];
        const ggml_tensor * slot_order = store->src[1];
        if (!windows || !slot_order || windows->op != GGML_OP_VIEW || windows->view_src != cat ||
            slot_order->type != GGML_TYPE_I32 || !ggml_is_contiguous(slot_order)) {
            return false;
        }
        const int64_t kw = windows->ne[1];
        if (kw <= 0 || kw > n_seq_tokens || slot_order->ne[0] != kw ||
            windows->ne[0] != 3 || windows->ne[2] != n_channels || windows->ne[3] != n_seqs ||
            windows->view_offs != (size_t) (n_seq_tokens - (kw - 1)) * sizeof(float)) {
            return false;
        }
        if (store->ne[0] != 3 || store->ne[2] != n_channels || store->ne[3] != n_seqs ||
            store->nb[0] != sizeof(float)) {
            return false;
        }
    }

    // verify intermediates have no external consumers and are not graph outputs
    if ((cat->flags & GGML_TENSOR_FLAG_OUTPUT) || (rsh->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (ggml_node_get_use_count(gf, node_idxs[idx + 0]) != (has_dyn ? 5 : 2)) {
        return false;
    }
    for (int j = 2; j < n_ops - 1; ++j) {
        if ((nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + j]) != 1) {
            return false;
        }
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!past->data || !cur->data || !conv->src[1]->data || !store->data || !out->data) {
            return false;
        }
        if (has_dyn && !dynamic->data) {
            return false;
        }
        if (has_add && !res_in->data) {
            return false;
        }
        if (is_set && !store->src[1]->data) {
            return false;
        }
        if (ggml_metal_fusion_overlap(out, cur) || ggml_metal_fusion_overlap(out, past)) {
            return false;
        }
        // for T > 16, multiple token tiles run concurrently so state_dst cannot alias past
        if (n_seq_tokens > 16 && ggml_metal_fusion_overlap(store, past)) {
            return false;
        }
    }

    return true;
}

// MUL_MAT_ID + REPEAT(w_s) + GET_ROWS(ids) + MUL (per-expert scalar weight scale for NVFP4 MoE)
static bool ggml_metal_fusion_check_mul_mat_id_scale(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * rep = nodes[1];
    const ggml_tensor * gr  = nodes[2];
    const ggml_tensor * mul = nodes[3];

    if (mm->op != GGML_OP_MUL_MAT_ID || mm->src[0]->type != GGML_TYPE_NVFP4 || mm->type != GGML_TYPE_F32) {
        return false;
    }
    if (rep->op != GGML_OP_REPEAT || gr->op != GGML_OP_GET_ROWS || mul->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * w_s = rep->src[0];
    if (!w_s || w_s->type != GGML_TYPE_F32 || !ggml_is_contiguous(w_s) ||
        w_s->ne[0] != 1 || w_s->ne[1] != mm->src[0]->ne[2] || w_s->ne[2] != 1 || w_s->ne[3] != 1) {
        return false;
    }
    if (gr->src[0] != rep || gr->src[1] != mm->src[2]) {
        return false;
    }
    if (!((mul->src[0] == mm && mul->src[1] == gr) || (mul->src[0] == gr && mul->src[1] == mm))) {
        return false;
    }
    if (mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || !ggml_are_same_shape(mm, mul)) {
        return false;
    }

    const int outputs[1] = { node_idxs[idx + 3] };
    return ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 4, fusion->ops.data(), outputs, 1);
}
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

static const std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sp = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sp_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sig = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sig_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static bool ggml_metal_fusion_check_topk_moe(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    const int n_ops = (int) fusion->ops.size();

    bool with_norm = false;
    for (int j = 0; j < n_ops; ++j) {
        with_norm |= fusion->ops[j] == GGML_OP_SUM_ROWS;
    }
    const bool with_scale = fusion->ops[n_ops - 1] == GGML_OP_SCALE;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const std::vector<ggml_op> & ops_all = fusion->ops_all;

    const int raw_start = node_idxs[idx];
    int       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm && !with_scale) {
        if (raw_end + 1 >= gf->n_nodes) {
            return false;
        }
        const ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        if (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            return false;
        }
        raw_end++;
    }

    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) ops_all.size()) {
        return false;
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * first = gf->nodes[raw_start];
    const ggml_tensor * out   = gf->nodes[raw_end];

    // the unary chains start at the sigmoid or the softplus node and rank biased scores
    const bool with_unary = first->op == GGML_OP_UNARY;
    const bool with_sp    = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS;
    const bool with_sig   = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SIGMOID;

    if (with_unary && !with_sp && !with_sig) {
        return false;
    }

    const ggml_tensor * probs;          // the scoring output
    const ggml_tensor * scores;         // the node the argsort ranks (adds the bias for the unary chains)
    const ggml_tensor * probs_reshaped;
    const ggml_tensor * argsort;
    const ggml_tensor * ids;
    const ggml_tensor * get_rows;
    const ggml_tensor * logits = first->src[0];
    int ids_off;

    if (with_unary) {
        // the softplus chains add a SQRT node, the sigmoid chains start at the sigmoid node
        const int base = with_sp ? 1 : 0;

        probs          = gf->nodes[raw_start + base + 0];
        probs_reshaped = gf->nodes[raw_start + base + 1];
        scores         = gf->nodes[raw_start + base + 2];
        argsort        = gf->nodes[raw_start + base + 3];
        ids            = gf->nodes[raw_start + base + 4];
        get_rows       = gf->nodes[raw_start + base + 5];
        ids_off        = base + 4;

        if (scores->op != GGML_OP_ADD || scores->src[0] != probs ||
            (with_sp && (probs->op != GGML_OP_SQRT || probs->src[0] != first))) {
            return false;
        }

        // the selection bias is a per-expert vector shared by all rows
        const ggml_tensor * bias = scores->src[1];
        if (bias->type != GGML_TYPE_F32 || scores->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            bias->ne[0] != probs->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
            return false;
        }
    } else {
        probs          = first;
        scores         = first;
        probs_reshaped = gf->nodes[raw_start + 1];
        argsort        = gf->nodes[raw_start + 2];
        ids            = gf->nodes[raw_start + 3];
        get_rows       = gf->nodes[raw_start + 4];
        ids_off        = 3;

        // the fused kernel implements plain softmax only
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale,    ((const int32_t *) first->op_params) + 0, sizeof(scale));
        memcpy(&max_bias, ((const int32_t *) first->op_params) + 1, sizeof(max_bias));
        if (scale != 1.0f || max_bias != 0.0f || first->src[1] || first->src[2]) {
            return false;
        }
    }

    if (logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        return false;
    }

    if (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        return false;
    }

    // the top-k gathers the unbiased scores, the argsort ranks the biased ones
    if (probs_reshaped->src[0] != probs || argsort->src[0] != scores ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        return false;
    }

    if (with_norm) {
        const int norm_off = with_unary ? 6 + (with_sp ? 1 : 0) : 5;

        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + norm_off + 0];
        const ggml_tensor * sum_rows         = gf->nodes[raw_start + norm_off + 1];
        const ggml_tensor * denom            = gf->nodes[raw_start + norm_off + 2];
        const ggml_tensor * div              = gf->nodes[raw_start + norm_off + 3];
        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + norm_off + 4];

        if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            denom->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != denom ||
            out_reshaped->src[0] != div) {
            return false;
        }

        // the kernel applies the sum adjustment as a lower clamp or an additive bias, never a scale
        if (denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) != 1.0f) {
            return false;
        }

        if (with_scale && out->src[0] != out_reshaped) {
            return false;
        }
    } else if (with_scale && out->src[0] != get_rows) {
        return false;
    }

    const int outputs[2] = { raw_start + ids_off, raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!logits->data || !out->data || !ids->data) {
            return false;
        }
    }

    return true;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

struct ggml_metal_moe_reduce_match {
    const ggml_tensor * experts;
    const ggml_tensor * weights;
    const ggml_tensor * dst;
    int node_count;
};

static bool ggml_metal_fusion_match_moe_reduce(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    if (match == nullptr || node_idx < 0 || node_idx + (int) ops_all.size() > gf->n_nodes) {
        return false;
    }

    const ggml_tensor * mul = gf->nodes[node_idx];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        return false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const int raw_count     = (int) ops_all.size();
    const int n_expert_used = raw_count / 2;

    if (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        return false;
    }

    int n_views = 0;
    while (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    if (n_views != n_expert_used) {
        return false;
    }

    for (int i = n_expert_used + 1; i < raw_count; ++i) {
        if (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            return false;
        }
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        if (gf->nodes[node_idx + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    const ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        return false;
    }

    for (int i = 1; i <= n_expert_used; ++i) {
        const ggml_tensor * view = gf->nodes[node_idx + i];
        if (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            return false;
        }
    }

    const ggml_tensor * prev_add = nullptr;
    for (int j = 1; j < n_expert_used; ++j) {
        const ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        if (add->src[0] != lhs || add->src[1] != rhs) {
            return false;
        }
        prev_add = add;
    }

    const int outputs[1] = { node_idx + raw_count - 1 };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        return false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    return true;
}

static bool ggml_metal_fusion_check_moe_reduce(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    if (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        return false;
    }

    if ((int) fusion->ops.size() != match.experts->ne[1]) {
        return false;
    }

    const int raw_end = node_idxs[idx] + match.node_count - 1;
    if (node_idxs[idx + (int) fusion->ops.size() - 1] != raw_end) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!match.experts->data || !match.weights->data || !match.dst->data) {
            return false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        if (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            return false;
        }
    }

    return true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool ggml_metal_tensor_is_weight(const struct ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    return base->buffer != NULL && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_operand(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    if (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        return NULL;
    }

    const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    return ok ? other : NULL;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_residual(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    const ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    return ok ? res : NULL;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool ggml_metal_fusion_check_mul_mat_add(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * add = nodes[1];

    if (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_STRUCTURAL) {
        return true;
    }

    const ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    if (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        return false;
    }

    return !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

static const std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

static const std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };
static const std::vector<ggml_op> ops_ssm_conv_res  = { GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_ssm_conv_full_cpy = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_ssm_conv_full_cpy_add = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_ssm_conv_full_set = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_SET_ROWS, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_ssm_conv_full_set_add = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_SET_ROWS, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_ssm_conv_full_dyn_cpy = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY,
    GGML_OP_SSM_CONV, GGML_OP_RESHAPE,
    GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_ADD, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_ssm_conv_full_dyn_set = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_SET_ROWS,
    GGML_OP_SSM_CONV, GGML_OP_RESHAPE,
    GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_ADD, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_l2_clip_scale = {
    GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_l2_clip = {
    GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_loop_control = {
    GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_loop_control_mm = {
    GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_situ_view = {
    GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_situ = {
    GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
    GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_mul_sigmoid = {
    GGML_OP_UNARY, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_depth_mix_1 = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT,
    GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT
};
static const std::vector<ggml_op> ops_depth_mix_1_norm = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT,
    GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_depth_mix_n = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_REPEAT,
    GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT
};
static const std::vector<ggml_op> ops_depth_mix_n_norm = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_REPEAT,
    GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_depth_mix_n_tr = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_REPEAT,
    GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_PERMUTE, GGML_OP_CONT
};
static const std::vector<ggml_op> ops_concat_rms_norm = {
    GGML_OP_CONCAT, GGML_OP_RMS_NORM
};
static const std::vector<ggml_op> ops_loop_down_norm = {
    GGML_OP_CONCAT, GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_MUL_MAT
};
static const std::vector<ggml_op> ops_mul_mat_id_scale = {
    GGML_OP_MUL_MAT_ID, GGML_OP_REPEAT, GGML_OP_GET_ROWS, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_mul_mat_id_scale_rsh = {
    GGML_OP_MUL_MAT_ID, GGML_OP_RESHAPE, GGML_OP_REPEAT, GGML_OP_GET_ROWS, GGML_OP_MUL
};

static const std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_moe_reduce_8_conv_cpy_add = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_RESHAPE, GGML_OP_TRANSPOSE,
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_moe_reduce_8_conv_set_add = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_RESHAPE, GGML_OP_TRANSPOSE,
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_SET_ROWS, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD
};

// MOE_REDUCE(8) + SSM_CONV_FULL(non-dyn) + ADD(residual)
static bool ggml_metal_fusion_check_moe_reduce_conv_add(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int n_ops     = (int) fusion->ops.size();
    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + n_ops - 1];
    const int raw_count = raw_end - raw_start + 1;
    if (n_ops != 14 || raw_count != (int) fusion->ops_all.size() || raw_count > GGML_METAL_FUSION_MAX) {
        return false;
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    ggml_metal_moe_reduce_match mr;
    if (!ggml_metal_fusion_match_moe_reduce(gf, raw_start, ops_moe_reduce_8, &mr)) {
        return false;
    }

    const ggml_tensor * past    = gf->nodes[raw_start + 16];
    const ggml_tensor * cur_rsh = gf->nodes[raw_start + 17];
    const ggml_tensor * tr      = gf->nodes[raw_start + 18];
    const ggml_tensor * cat     = nodes[8];
    const ggml_tensor * store   = nodes[9];
    const ggml_tensor * conv    = nodes[10];
    const ggml_tensor * c_add   = nodes[11];
    const ggml_tensor * sc      = nodes[12];
    const ggml_tensor * out     = nodes[13];

    if (cat != gf->nodes[raw_start + 19] || cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0) {
        return false;
    }
    if (cat->src[0] != past || cat->src[1] != tr || tr->src[0] != cur_rsh || cur_rsh->src[0] != mr.dst) {
        return false;
    }

    const bool is_set = (store->op == GGML_OP_SET_ROWS);
    const bool is_cpy = (store->op == GGML_OP_CPY);
    if (!is_set && !is_cpy) {
        return false;
    }
    if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != cat || !conv->src[1] ||
        conv->src[1]->ne[0] != 4 || conv->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(conv->src[1])) {
        return false;
    }
    if (sc->op != GGML_OP_SCALE || sc->src[0] != c_add || ggml_get_op_params_f32(sc, 1) != 0.0f) {
        return false;
    }
    if (!((c_add->src[0] == mr.dst && c_add->src[1] && c_add->src[1]->op == GGML_OP_RESHAPE && c_add->src[1]->src[0] == conv) ||
          (c_add->src[1] == mr.dst && c_add->src[0] && c_add->src[0]->op == GGML_OP_RESHAPE && c_add->src[0]->src[0] == conv))) {
        return false;
    }
    if (out->op != GGML_OP_ADD || (out->src[0] != sc && out->src[1] != sc)) {
        return false;
    }
    const ggml_tensor * res_in = (out->src[0] == sc) ? out->src[1] : out->src[0];
    if (!res_in || res_in->type != GGML_TYPE_F32 || !ggml_is_contiguous(res_in) ||
        !ggml_are_same_shape(res_in, sc) || out->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(out) || !ggml_are_same_shape(out, sc) || store->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_channels   = mr.dst->ne[0];
    const int64_t n_seqs       = past->ne[2];
    const int64_t n_seq_tokens = cat->ne[0] - 3;
    const int64_t n_tokens     = n_seq_tokens * n_seqs;
    if (past->ne[0] != 3 || past->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(past) ||
        n_channels <= 0 || n_seqs <= 0 || n_seq_tokens <= 0 ||
        past->ne[1] != n_channels || mr.dst->ne[1] != n_tokens ||
        mr.dst->ne[2] != 1 || mr.dst->ne[3] != 1 || past->ne[3] != 1 ||
        sc->type != GGML_TYPE_F32 || !ggml_is_contiguous(sc) || !ggml_are_same_shape(mr.dst, sc)) {
        return false;
    }

    if (is_cpy) {
        const ggml_tensor * last = store->src[0];
        if (!last || last->op != GGML_OP_VIEW || last->view_src != cat ||
            last->ne[0] != 3 || last->ne[1] != n_channels || last->ne[2] != n_seqs ||
            last->view_offs != (size_t) n_seq_tokens * sizeof(float) ||
            store->ne[0] != 3 * n_channels || store->ne[1] != n_seqs || store->ne[2] != 1 || store->ne[3] != 1 ||
            store->nb[0] != sizeof(float)) {
            return false;
        }
    } else {
        const ggml_tensor * windows    = store->src[0];
        const ggml_tensor * slot_order = store->src[1];
        if (!windows || !slot_order || windows->op != GGML_OP_VIEW || windows->view_src != cat ||
            slot_order->type != GGML_TYPE_I32 || !ggml_is_contiguous(slot_order)) {
            return false;
        }
        const int64_t kw = windows->ne[1];
        if (kw <= 0 || kw > n_seq_tokens || slot_order->ne[0] != kw ||
            windows->ne[0] != 3 || windows->ne[2] != n_channels || windows->ne[3] != n_seqs ||
            windows->view_offs != (size_t) (n_seq_tokens - (kw - 1)) * sizeof(float) ||
            store->ne[0] != 3 || store->ne[2] != n_channels || store->ne[3] != n_seqs ||
            store->nb[0] != sizeof(float)) {
            return false;
        }
    }

    const ggml_tensor * conv_rsh = (c_add->src[0] == mr.dst) ? c_add->src[1] : c_add->src[0];
    if ((mr.dst->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, raw_start + 15) != 2) {
        return false;
    }
    if ((cat->flags & GGML_TENSOR_FLAG_OUTPUT) || (conv_rsh->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (ggml_node_get_use_count(gf, node_idxs[idx + 8]) != 2) {
        return false;
    }
    for (int j = 10; j < n_ops - 1; ++j) {
        if ((nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + j]) != 1) {
            return false;
        }
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!mr.experts->data || !mr.weights->data || !past->data || !conv->src[1]->data ||
            !store->data || !res_in->data || !out->data) {
            return false;
        }
        if (is_set && !store->src[1]->data) {
            return false;
        }
        if (ggml_metal_fusion_overlap(out, mr.experts) || ggml_metal_fusion_overlap(out, mr.weights) ||
            ggml_metal_fusion_overlap(out, past)) {
            return false;
        }
        if (n_seq_tokens > 16 && ggml_metal_fusion_overlap(store, past)) {
            return false;
        }
    }

    return true;
}

static const std::vector<ggml_op> ops_mul_mat_add     = { GGML_OP_MUL_MAT, GGML_OP_ADD };
static const std::vector<ggml_op> ops_mul_soft_max    = { GGML_OP_MUL, GGML_OP_SOFT_MAX };
static const std::vector<ggml_op> ops_softcap         = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_fa_banded_rel_rsh = { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_FLASH_ATTN_EXT_BANDED };
static const std::vector<ggml_op> ops_fa_banded_rel     = { GGML_OP_MUL_MAT, GGML_OP_FLASH_ATTN_EXT_BANDED };

// MUL_MAT(rel_w[16, E], rel_q[16, H, T]) [+ RESHAPE] + FLASH_ATTN_EXT_BANDED
static bool ggml_metal_fusion_check_fa_banded_rel(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const ggml_tensor * mm = nodes[0];
    const ggml_tensor * fa = nodes[1];

    const bool has_rsh   = fusion->ops_all.size() == 3;
    const int  raw_start = node_idxs[idx];
    const int  raw_end   = node_idxs[idx + 1];
    if (raw_end - raw_start != (has_rsh ? 2 : 1)) {
        return false;
    }

    const ggml_tensor * rel = fa->src[6];
    if (!rel) {
        return false;
    }
    if (has_rsh) {
        if (gf->nodes[raw_start + 1] != rel || rel->op != GGML_OP_RESHAPE || rel->src[0] != mm) {
            return false;
        }
    } else {
        if (rel != mm) {
            return false;
        }
    }

    const ggml_tensor * rel_w = mm->src[0];
    const ggml_tensor * rel_q = mm->src[1];
    if (!rel_w || !rel_q || mm->type != GGML_TYPE_F32 ||
        rel_w->type != GGML_TYPE_F32 || rel_q->type != GGML_TYPE_F32) {
        return false;
    }
    if (rel_w->ne[0] != 16 || rel_q->ne[0] != 16 ||
        rel_w->ne[2] != 1  || rel_w->ne[3] != 1  || rel_q->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(rel_w) || !ggml_is_contiguous(rel_q)) {
        return false;
    }

    const ggml_tensor * q = fa->src[0];
    const ggml_tensor * k = fa->src[1];
    const ggml_tensor * v = fa->src[2];
    if (!q || !k || !v || q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128) {
        return false;
    }
    if (fa->src[5] != nullptr) {
        return false;
    }
    int32_t n_kv_max = 0;
    memcpy(&n_kv_max, ((const int32_t *) fa->op_params) + 4, sizeof(n_kv_max));
    if (n_kv_max > 0) {
        return false;
    }
    // Supported on wide Metal 4 tensor path (ne01 >= 32) and all vec paths (ne01 < 20)
    if (q->ne[1] >= 20 && q->ne[1] < 32) {
        return false;
    }
    switch (k->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            if (q->ne[1] >= 20) {
                return false;
            }
            break;
        default:
            return false;
    }

    if (rel->ne[0] != rel_w->ne[1] ||
        rel->ne[1] != q->ne[2]     || rel_q->ne[1] != q->ne[2] ||
        rel->ne[2] != q->ne[1]     || rel->ne[3]   != q->ne[3] ||
        rel_q->ne[2] != q->ne[1] * q->ne[3]) {
        return false;
    }

    int raw_idxs[3] = { raw_start, raw_start + 1, raw_start + 2 };
    const int raw_n = has_rsh ? 3 : 2;
    const int outputs[1] = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_n, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!rel_w->data || !rel_q->data || !fa->data ||
            ggml_metal_fusion_overlap(fa, rel_w) || ggml_metal_fusion_overlap(fa, rel_q)) {
            return false;
        }
    }

    return true;
}

static const std::vector<ggml_op> ops_l2_clip_fwht = {
    GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL,
    GGML_OP_RESHAPE, GGML_OP_MUL_MAT
};
static const std::vector<ggml_op> ops_l2_clip_tau_fwht = {
    GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL,
    GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT
};
static const std::vector<ggml_op> ops_l2_clip_fwht_set_rows = {
    GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL,
    GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SET_ROWS
};
static const std::vector<ggml_op> ops_l2_clip_fwht_set_rows_rsh = {
    GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL,
    GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_RESHAPE, GGML_OP_SET_ROWS
};
static const std::vector<ggml_op> ops_rms_norm_fwht_set_rows = {
    GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SET_ROWS
};
static const std::vector<ggml_op> ops_rms_norm_fwht_set_rows_rsh = {
    GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_RESHAPE, GGML_OP_SET_ROWS
};
static const std::vector<ggml_op> ops_fwht_mul_sigmoid = {
    GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY, GGML_OP_MUL
};

static bool ggml_metal_fusion_is_fwht_64_128(const ggml_tensor * fwht) {
    if (!fwht || fwht->op != GGML_OP_MUL_MAT || !fwht->src[0] || !fwht->src[1]) {
        return false;
    }
    if (ggml_get_op_params_i32(fwht, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        return false;
    }
    const int64_t n = fwht->src[1]->ne[0];
    return (n == 64 || n == 128) && fwht->type == GGML_TYPE_F32 &&
           fwht->src[1]->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(fwht->src[1]) && ggml_is_contiguous(fwht) &&
           ggml_are_same_shape(fwht->src[1], fwht);
}

// L2_CLIP_SCALE [+ MUL(tau)] + RESHAPE + FWHT(64|128) on 128-dim heads (Q head)
static bool ggml_metal_fusion_check_l2_clip_fwht(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int n_ops     = (int) fusion->ops.size();
    const bool has_tau  = (n_ops == 11);
    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + n_ops - 1];
    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) fusion->ops_all.size()) {
        return false;
    }
    int raw_idxs[16];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * cur   = nodes[0];
    const ggml_tensor * x     = cur->src[0];
    const ggml_tensor * scale = cur->src[1];
    const ggml_tensor * sqr   = nodes[1];
    const ggml_tensor * sum   = nodes[2];
    const ggml_tensor * sq    = nodes[3];
    const ggml_tensor * sc    = nodes[4];
    const ggml_tensor * fil   = nodes[5];
    const ggml_tensor * dv    = nodes[6];
    const ggml_tensor * clp   = nodes[7];
    const ggml_tensor * l2out = nodes[8];

    if (!x || !scale || sqr->src[0] != cur || sum->src[0] != sqr || sq->src[0] != sum ||
        sc->src[0] != sq || fil->src[0] != sc || dv->src[0] != fil || dv->src[1] != sc ||
        clp->src[0] != dv || l2out->src[0] != cur || l2out->src[1] != clp) {
        return false;
    }
    if (ggml_get_op_params_f32(sc, 0) != 1.0f ||
        ggml_get_op_params_f32(clp, 0) != 0.0f || ggml_get_op_params_f32(clp, 1) != 1.0f) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != 128 || x->ne[3] != 1) {
        return false;
    }
    if (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) ||
        scale->ne[0] != 128 || scale->ne[1] != x->ne[1] || scale->ne[2] != 1 || scale->ne[3] != 1) {
        return false;
    }

    const ggml_tensor * prev = l2out;
    const ggml_tensor * tau  = nullptr;
    if (has_tau) {
        const ggml_tensor * mtau = nodes[9];
        if (mtau->src[0] == l2out) {
            tau = mtau->src[1];
        } else if (mtau->src[1] == l2out) {
            tau = mtau->src[0];
        } else {
            return false;
        }
        if (!tau || tau->type != GGML_TYPE_F32 || !ggml_is_contiguous(tau) ||
            tau->ne[0] != 1 || tau->ne[1] != 1 || tau->ne[2] != x->ne[2] || tau->ne[3] != 1) {
            return false;
        }
        prev = mtau;
    }

    const ggml_tensor * rsh  = gf->nodes[raw_end - 1];
    const ggml_tensor * fwht = nodes[n_ops - 1];
    if (rsh->op != GGML_OP_RESHAPE || rsh->src[0] != prev || fwht->src[1] != rsh ||
        !ggml_metal_fusion_is_fwht_64_128(fwht) || ggml_nelements(fwht) != ggml_nelements(x)) {
        return false;
    }

    const int outputs[1] = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!x->data || !scale->data || !fwht->data || (tau && !tau->data)) {
            return false;
        }
        if (ggml_metal_fusion_overlap(fwht, x) || ggml_metal_fusion_overlap(fwht, scale) ||
            (tau && ggml_metal_fusion_overlap(fwht, tau))) {
            return false;
        }
    }

    return true;
}

// L2_CLIP + RESHAPE + FWHT(64|128) + RESHAPE + VIEW + [RESHAPE] + SET_ROWS(q8_0, i64) on 128-dim heads (K head)
static bool ggml_metal_fusion_check_l2_clip_fwht_set_rows(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int n_ops     = (int) fusion->ops.size();
    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + n_ops - 1];
    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) fusion->ops_all.size()) {
        return false;
    }
    int raw_idxs[16];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * sqr   = nodes[0];
    const ggml_tensor * x     = sqr->src[0];
    const ggml_tensor * sum   = nodes[1];
    const ggml_tensor * sq    = nodes[2];
    const ggml_tensor * sc    = nodes[3];
    const ggml_tensor * fil   = nodes[4];
    const ggml_tensor * dv    = nodes[5];
    const ggml_tensor * clp   = nodes[6];
    const ggml_tensor * l2out = nodes[7];
    const ggml_tensor * fwht  = nodes[8];
    const ggml_tensor * srows = nodes[9];

    if (!x || sum->src[0] != sqr || sq->src[0] != sum || sc->src[0] != sq ||
        fil->src[0] != sc || dv->src[0] != fil || dv->src[1] != sc ||
        clp->src[0] != dv || l2out->src[0] != x || l2out->src[1] != clp) {
        return false;
    }
    if (ggml_get_op_params_f32(sc, 0) != 1.0f ||
        ggml_get_op_params_f32(clp, 0) != 0.0f || ggml_get_op_params_f32(clp, 1) != 1.0f) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != 128 || x->ne[3] != 1) {
        return false;
    }

    const ggml_tensor * rsh_in  = gf->nodes[raw_start + 8];
    const ggml_tensor * rsh_out = gf->nodes[raw_start + 10];
    const ggml_tensor * view    = gf->nodes[raw_start + 11];
    if (rsh_in->src[0] != l2out || fwht->src[1] != rsh_in || !ggml_metal_fusion_is_fwht_64_128(fwht) ||
        ggml_nelements(fwht) != ggml_nelements(x)) {
        return false;
    }
    if (rsh_out->src[0] != fwht || view->src[0] != rsh_out || view->view_offs != 0 ||
        !ggml_is_contiguous(view) || view->ne[0] != 128 * x->ne[1] || view->ne[1] != x->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1) {
        return false;
    }
    if (raw_count == 14 && srows->src[2] != gf->nodes[raw_start + 12]) {
        return false;
    }
    const ggml_tensor * idxs = srows->src[1];
    if (srows->src[0] != view || !idxs || idxs->type != GGML_TYPE_I64 || !ggml_is_contiguous(idxs) ||
        idxs->ne[0] != x->ne[2] || idxs->ne[1] != 1 || idxs->ne[2] != 1 || idxs->ne[3] != 1 ||
        srows->type != GGML_TYPE_Q8_0 || srows->ne[0] != 128 * x->ne[1] ||
        srows->ne[2] != 1 || srows->ne[3] != 1) {
        return false;
    }

    const int outputs[1] = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!x->data || !idxs->data || !srows->data) {
            return false;
        }
        if (ggml_metal_fusion_overlap(srows, x) || ggml_metal_fusion_overlap(srows, idxs)) {
            return false;
        }
    }

    return true;
}

// RMS_NORM + RESHAPE + FWHT(64|128) + RESHAPE + VIEW + [RESHAPE] + SET_ROWS(q8_0, i64) on 128-dim heads (V head)
static bool ggml_metal_fusion_check_rms_norm_fwht_set_rows(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int n_ops     = (int) fusion->ops.size();
    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + n_ops - 1];
    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) fusion->ops_all.size()) {
        return false;
    }
    int raw_idxs[8];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * rms     = nodes[0];
    const ggml_tensor * x       = rms->src[0];
    const ggml_tensor * rsh_in  = gf->nodes[raw_start + 1];
    const ggml_tensor * fwht    = nodes[1];
    const ggml_tensor * rsh_out = gf->nodes[raw_start + 3];
    const ggml_tensor * view    = gf->nodes[raw_start + 4];
    const ggml_tensor * srows   = nodes[2];

    if (!x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != 128 || x->ne[3] != 1) {
        return false;
    }
    if (rsh_in->src[0] != rms || fwht->src[1] != rsh_in || !ggml_metal_fusion_is_fwht_64_128(fwht) ||
        ggml_nelements(fwht) != ggml_nelements(x)) {
        return false;
    }
    if (rsh_out->src[0] != fwht || view->src[0] != rsh_out || view->view_offs != 0 ||
        !ggml_is_contiguous(view) || view->ne[0] != 128 * x->ne[1] || view->ne[1] != x->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1) {
        return false;
    }
    if (raw_count == 7 && srows->src[2] != gf->nodes[raw_start + 5]) {
        return false;
    }
    const ggml_tensor * idxs = srows->src[1];
    if (srows->src[0] != view || !idxs || idxs->type != GGML_TYPE_I64 || !ggml_is_contiguous(idxs) ||
        idxs->ne[0] != x->ne[2] || idxs->ne[1] != 1 || idxs->ne[2] != 1 || idxs->ne[3] != 1 ||
        srows->type != GGML_TYPE_Q8_0 || srows->ne[0] != 128 * x->ne[1] ||
        srows->ne[2] != 1 || srows->ne[3] != 1) {
        return false;
    }

    const int outputs[1] = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!x->data || !idxs->data || !srows->data) {
            return false;
        }
        if (ggml_metal_fusion_overlap(srows, x) || ggml_metal_fusion_overlap(srows, idxs)) {
            return false;
        }
    }

    return true;
}

// FWHT(64|128) + RESHAPE + SIGMOID(gate) + MUL (kqv_out gating)
static bool ggml_metal_fusion_check_fwht_mul_sigmoid(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int raw_start = node_idxs[idx];
    const int raw_end   = node_idxs[idx + 2];
    if (raw_end - raw_start != 3) {
        return false;
    }
    int raw_idxs[4];
    for (int i = 0; i < 4; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != fusion->ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * fwht = nodes[0];
    const ggml_tensor * rsh  = gf->nodes[raw_start + 1];
    const ggml_tensor * sig  = nodes[1];
    const ggml_tensor * mul  = nodes[2];

    if (!ggml_metal_fusion_is_fwht_64_128(fwht) || rsh->src[0] != fwht) {
        return false;
    }
    if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    if (!((mul->src[0] == rsh && mul->src[1] == sig) || (mul->src[0] == sig && mul->src[1] == rsh))) {
        return false;
    }
    const ggml_tensor * gate = sig->src[0];
    if (!gate || gate->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gate) || !ggml_is_contiguous(mul) ||
        !ggml_are_same_shape(gate, mul) || ggml_nelements(mul) != ggml_nelements(fwht)) {
        return false;
    }

    const int outputs[1] = { raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, 4, fusion->ops_all.data(), outputs, 1)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!fwht->src[1]->data || !gate->data || !mul->data) {
            return false;
        }
        // Kernel reads entire N_FWHT row into registers before writing same row -> exact in-place is safe
        if ((mul->data != fwht->src[1]->data && ggml_metal_fusion_overlap(mul, fwht->src[1])) ||
            (mul->data != gate->data         && ggml_metal_fusion_overlap(mul, gate))) {
            return false;
        }
    }

    return true;
}

static bool ggml_metal_fusion_check_softcap(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * sc0 = nodes[0];
    const ggml_tensor * th  = nodes[1];
    const ggml_tensor * sc1 = nodes[2];

    if (!sc0->src[0] || th->src[0] != sc0 || sc1->src[0] != th) {
        return false;
    }
    if (ggml_get_unary_op(th) != GGML_UNARY_OP_TANH) {
        return false;
    }
    if (ggml_get_op_params_f32(sc0, 1) != 0.0f || ggml_get_op_params_f32(sc1, 1) != 0.0f) {
        return false;
    }
    if (sc0->src[0]->type != GGML_TYPE_F32 || sc0->type != GGML_TYPE_F32 ||
        th->type != GGML_TYPE_F32 || sc1->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous_rows(sc0->src[0]) || !ggml_is_contiguous_rows(sc1)) {
        return false;
    }

    return true;
}

static bool ggml_metal_fusion_check_mul_soft_max(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul = nodes[0];
    const ggml_tensor * sm  = nodes[1];

    const ggml_tensor * x  = mul->src[0];
    const ggml_tensor * sc = mul->src[1];

    if (!x || !sc || x->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || sm->type != GGML_TYPE_F32) {
        return false;
    }

    if (mul->ne[0] < 16384 || mul->ne[1] > 16 || mul->ne[2] != 1 || mul->ne[3] != 1) {
        return false;
    }

    if (x->ne[0] != mul->ne[0] || x->ne[1] != mul->ne[1] || x->ne[2] != 1 || x->ne[3] != 1 ||
        sc->ne[0] != 1 || sc->ne[1] != mul->ne[1] || sc->ne[2] != 1 || sc->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous_rows(x) || !ggml_is_contiguous_rows(sc) ||
        !ggml_is_contiguous_rows(mul) || !ggml_is_contiguous_rows(sm)) {
        return false;
    }

    if (sm->src[0] != mul || sm->src[1] != nullptr || sm->src[2] != nullptr) {
        return false;
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    ((const int32_t *) sm->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) sm->op_params) + 1, sizeof(max_bias));
    if (scale != 1.0f || max_bias != 0.0f) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (ggml_metal_fusion_overlap(sm, x) || ggml_metal_fusion_overlap(sm, sc)) {
            return false;
        }
    }

    return true;
}

static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_norm_scale,       {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_rms_norm_scale,   {}, false, ggml_metal_fusion_check_norm },

    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_2, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_3, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_4, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_5, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_6, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_7, {}, false, ggml_metal_fusion_check_add_chain },

    { GGML_METAL_FUSION_SNAKE, ops_snake, {}, false, ggml_metal_fusion_check_snake },

    { GGML_METAL_FUSION_GDN_CACHE, ops_gdn_cache, {}, true, ggml_metal_fusion_check_gdn_cache },

    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe,                     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_scale,               {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm,                {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale,          {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_bias,           {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale_bias,     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp,                  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_scale,            {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm,             {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale,       {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_bias,        {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale_bias,  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig,                 {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_scale,           {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm,            {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale,      {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_bias,       {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale_bias, {2}, true, ggml_metal_fusion_check_topk_moe },

    { GGML_METAL_FUSION_MOE_REDUCE_CONV_ADD, ops_moe_reduce_8_conv_cpy_add, {9}, true, ggml_metal_fusion_check_moe_reduce_conv_add },
    { GGML_METAL_FUSION_MOE_REDUCE_CONV_ADD, ops_moe_reduce_8_conv_set_add, {9}, true, ggml_metal_fusion_check_moe_reduce_conv_add },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_2, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_3, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_4, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_5, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_6, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_7, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_8, {}, true, ggml_metal_fusion_check_moe_reduce },

    { GGML_METAL_FUSION_SSM_CONV_SILU,   ops_ssm_conv_silu,         {},  false, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_SSM_CONV_RES,    ops_ssm_conv_res,          {},  true,  ggml_metal_fusion_check_ssm_conv_res },
    { GGML_METAL_FUSION_SSM_CONV_FULL,   ops_ssm_conv_full_dyn_cpy, {1}, true,  ggml_metal_fusion_check_ssm_conv_full },
    { GGML_METAL_FUSION_SSM_CONV_FULL,   ops_ssm_conv_full_dyn_set, {1}, true,  ggml_metal_fusion_check_ssm_conv_full },
    { GGML_METAL_FUSION_SSM_CONV_FULL,   ops_ssm_conv_full_cpy_add, {1}, true,  ggml_metal_fusion_check_ssm_conv_full },
    { GGML_METAL_FUSION_SSM_CONV_FULL,   ops_ssm_conv_full_set_add, {1}, true,  ggml_metal_fusion_check_ssm_conv_full },
    { GGML_METAL_FUSION_SSM_CONV_FULL,   ops_ssm_conv_full_cpy,     {1}, true,  ggml_metal_fusion_check_ssm_conv_full },
    { GGML_METAL_FUSION_SSM_CONV_FULL,   ops_ssm_conv_full_set,     {1}, true,  ggml_metal_fusion_check_ssm_conv_full },
    { GGML_METAL_FUSION_L2_CLIP,         ops_l2_clip_scale,         {},  true,  ggml_metal_fusion_check_l2_clip },
    { GGML_METAL_FUSION_L2_CLIP,         ops_l2_clip,               {},  true,  ggml_metal_fusion_check_l2_clip },
    { GGML_METAL_FUSION_LOOP_CONTROL_MM, ops_loop_control_mm,       {},  true,  ggml_metal_fusion_check_loop_control_mm },
    { GGML_METAL_FUSION_LOOP_CONTROL,    ops_loop_control,          {},  true,  ggml_metal_fusion_check_loop_control },
    { GGML_METAL_FUSION_SITU,          ops_situ_view,     {}, true,  ggml_metal_fusion_check_situ },
    { GGML_METAL_FUSION_SITU,          ops_situ,          {}, true,  ggml_metal_fusion_check_situ },
    { GGML_METAL_FUSION_MUL_SIGMOID,   ops_mul_sigmoid,   {}, false, ggml_metal_fusion_check_mul_sigmoid },
    { GGML_METAL_FUSION_DEPTH_MIX,       ops_depth_mix_n_norm, {}, true,  ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX,       ops_depth_mix_1_norm, {}, true,  ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX,       ops_depth_mix_n_tr,   {}, true,  ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX,       ops_depth_mix_n,      {}, true,  ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX,       ops_depth_mix_1,      {}, true,  ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_LOOP_DOWN_NORM,   ops_loop_down_norm,   {}, true,  ggml_metal_fusion_check_loop_down_norm },
    { GGML_METAL_FUSION_CONCAT_RMS_NORM,  ops_concat_rms_norm,  {}, false, ggml_metal_fusion_check_concat_rms_norm },
    { GGML_METAL_FUSION_MUL_MAT_ID_SCALE, ops_mul_mat_id_scale_rsh, {}, true,  ggml_metal_fusion_check_mul_mat_id_scale },
    { GGML_METAL_FUSION_MUL_MAT_ID_SCALE, ops_mul_mat_id_scale,     {}, true,  ggml_metal_fusion_check_mul_mat_id_scale },

    { GGML_METAL_FUSION_MUL_MAT_ADD,  ops_mul_mat_add,  {}, false, ggml_metal_fusion_check_mul_mat_add },
    { GGML_METAL_FUSION_MUL_SOFT_MAX, ops_mul_soft_max, {}, false, ggml_metal_fusion_check_mul_soft_max },
    { GGML_METAL_FUSION_SOFTCAP,      ops_softcap,      {}, false, ggml_metal_fusion_check_softcap },
    { GGML_METAL_FUSION_FA_BANDED_REL, ops_fa_banded_rel_rsh, {}, true, ggml_metal_fusion_check_fa_banded_rel },
    { GGML_METAL_FUSION_FA_BANDED_REL, ops_fa_banded_rel,     {}, true, ggml_metal_fusion_check_fa_banded_rel },
    { GGML_METAL_FUSION_L2_CLIP_FWHT,           ops_l2_clip_tau_fwht,          {}, true, ggml_metal_fusion_check_l2_clip_fwht },
    { GGML_METAL_FUSION_L2_CLIP_FWHT,           ops_l2_clip_fwht,              {}, true, ggml_metal_fusion_check_l2_clip_fwht },
    { GGML_METAL_FUSION_L2_CLIP_FWHT_SET_ROWS,  ops_l2_clip_fwht_set_rows_rsh, {}, true, ggml_metal_fusion_check_l2_clip_fwht_set_rows },
    { GGML_METAL_FUSION_L2_CLIP_FWHT_SET_ROWS,  ops_l2_clip_fwht_set_rows,     {}, true, ggml_metal_fusion_check_l2_clip_fwht_set_rows },
    { GGML_METAL_FUSION_RMS_NORM_FWHT_SET_ROWS, ops_rms_norm_fwht_set_rows_rsh,{}, true, ggml_metal_fusion_check_rms_norm_fwht_set_rows },
    { GGML_METAL_FUSION_RMS_NORM_FWHT_SET_ROWS, ops_rms_norm_fwht_set_rows,    {}, true, ggml_metal_fusion_check_rms_norm_fwht_set_rows },
    { GGML_METAL_FUSION_FWHT_MUL_SIGMOID,       ops_fwht_mul_sigmoid,          {}, true, ggml_metal_fusion_check_fwht_mul_sigmoid },
};

// ---- alloc deps -----------------------------------------------------------

static bool ggml_metal_fusion_match_raw_pattern(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops) {
    if (node_idx < 0 || node_idx + (int) ops.size() > gf->n_nodes) {
        return false;
    }

    for (int i = 0; i < (int) ops.size(); ++i) {
        if (gf->nodes[node_idx + i]->op != ops[i]) {
            return false;
        }
    }

    return true;
}

static void ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        const ggml_metal_fusion * fusion,
        int node_idx) {
    const int last_node = node_idx + (int) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    for (int j = 0; j < (int) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_metal_fusion * best = nullptr;
        int best_raw = 0;

        for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
            if ((int) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            if (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (int) fusion.ops_all.size();
            }
        }

        if (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < (int) fusion->ops.size(); j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(ggml_metal_fusion_info * finfo, const ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    const ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    if (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool ggml_metal_fusion_check_memory_ranges(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        int                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data && b->data && a->data == b->data) {
            return true;
        }
        for (const ggml_tensor * v = a; v; v = v->view_src) {
            if (v == b) {
                return true;
            }
        }
        for (const ggml_tensor * v = b; v; v = v->view_src) {
            if (v == a) {
                return true;
            }
        }
        return false;
    };

    auto nodes_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            return false;
        }

        if (a->buffer != b->buffer) {
            return false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        return (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const ggml_tensor * src, const ggml_tensor * const * nodes, int j) {
        for (int k = 0; k < j; ++k) {
            if (src == nodes[k]) {
                return true;
            }
            for (const ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                if (view_src == nodes[k]) {
                    return true;
                }
            }
        }
        return false;
    };

    auto check_dst = [&](const ggml_tensor * dst) {
        for (int j = 0; j < node_count; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nodes[j]->src[s];
                if (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!check_dst(nodes[node_count - 1])) {
        return false;
    }

    for (int offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        if (!check_dst(nodes[offset])) {
            return false;
        }
    }

    return true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        const int n_ops = (int) fusion.ops.size();

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    // a view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op)) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX || idxs[i_f + len - 1] - idx + 1 > GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
