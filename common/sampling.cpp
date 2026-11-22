#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;
    std::vector<llama_token_data> cand_grmr; // scratch buffer for grammar-masked rejection sampling
    std::vector<float>            p_buf;     // persistent contiguous target probabilities for fast rejection sampling
    std::vector<float>            resid_buf; // persistent contiguous max(0, p - q) buffer for fast rejection sampling

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    // true when the configured chain reduces to pure softmax(logits / temp) (no truncators, penalties, biases, suppress tokens, or logprobs)
    bool simple_softmax_chain = false;
    // true when the configured chain reduces to pure greedy argmax(logits)
    bool simple_greedy_chain  = false;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!common_grammar_value(params.grammar).empty()) {
        for (const auto & b : params.logit_bias) {
            if (b.bias == -INFINITY && llama_vocab_is_eog(vocab, b.token)) {
                throw std::invalid_argument("EOG tokens cannot be banned when a grammar is set");
            }
        }
    }
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    throw std::runtime_error(
                        "unknown grammar trigger type: " + std::to_string(trigger.type));
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
            samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    // Determine if the sampling chain reduces mathematically to pure softmax(logits / temp) or pure greedy argmax(logits).
    bool simple_softmax_chain = false;
    bool simple_greedy_chain  = false;
    {
        int32_t n_suppress = 0;
        (void) llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (params.mirostat == 0 &&
            params.n_probs == 0 &&
            params.dynatemp_range == 0.0f &&
            params.logit_bias.empty() &&
            n_suppress == 0) {
            bool simple_chain = true;
            for (const auto & cnstr : params.samplers) {
                switch (cnstr) {
                    case COMMON_SAMPLER_TYPE_TEMPERATURE:
                        break;
                    case COMMON_SAMPLER_TYPE_TOP_K:
                        if (params.top_k > 0 && params.top_k < n_vocab) {
                            simple_chain = false;
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_TOP_P:
                        if (params.top_p < 1.0f) {
                            simple_chain = false;
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_MIN_P:
                        if (params.min_p > 0.0f) {
                            simple_chain = false;
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_TYPICAL_P:
                        if (params.typ_p < 1.0f) {
                            simple_chain = false;
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_XTC:
                        if (params.xtc_probability > 0.0f && params.xtc_threshold <= 0.5f) {
                            simple_chain = false;
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                        if (params.top_n_sigma > 0.0f) {
                            simple_chain = false;
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_DRY:
                        if (params.dry_multiplier != 0.0f) {
                            simple_chain = false;
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_PENALTIES:
                        if (params.penalty_last_n != 0 &&
                            (params.penalty_repeat != 1.0f || params.penalty_freq != 0.0f || params.penalty_present != 0.0f)) {
                            simple_chain = false;
                        }
                        break;
                    default:
                        simple_chain = false;
                        break;
                }
                if (!simple_chain) {
                    break;
                }
            }
            if (simple_chain) {
                simple_softmax_chain = params.temp > 0.0f;
                simple_greedy_chain  = params.temp <= 0.0f;
            }
        }
    }

    auto * result = new common_sampler {
        /* .params               = */ params,
        /* .grmr                 = */ grmr,
        /* .rbudget              = */ rbudget,
        /* .chain                = */ chain,
        /* .prev                 = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur                  = */ {},
        /* .cand_grmr            = */ {},
        /* .p_buf                = */ {},
        /* .resid_buf            = */ {},
        /* .cur_p                = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng                  = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
        /* .simple_softmax_chain = */ simple_softmax_chain,
        /* .simple_greedy_chain  = */ simple_greedy_chain,
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params               = */ gsmpl->params,
        /* .grmr                 = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget              = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain                = */ llama_sampler_clone(gsmpl->chain),
        /* .prev                 = */ gsmpl->prev,
        /* .cur                  = */ gsmpl->cur,
        /* .cand_grmr            = */ {},
        /* .p_buf                = */ {},
        /* .resid_buf            = */ {},
        /* .cur_p                = */ gsmpl->cur_p,
        /* .rng                  = */ gsmpl->rng,
        /* .simple_softmax_chain = */ gsmpl->simple_softmax_chain,
        /* .simple_greedy_chain  = */ gsmpl->simple_greedy_chain,
    };
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params               = src->params;
    dst->prev                 = src->prev;
    dst->cur                  = src->cur;
    dst->cur_p                = src->cur_p;
    dst->cur_p.data           = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->rng                  = src->rng;
    dst->simple_softmax_chain = src->simple_softmax_chain;
    dst->simple_greedy_chain  = src->simple_greedy_chain;
    dst->t_total_us           = src->t_total_us;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

static bool grammar_allows_token(struct llama_sampler * grmr, llama_token id);

#if defined(__APPLE__)
static size_t apple_pcore_count() {
    static const size_t n_pcores = []() -> size_t {
        int32_t v = 0;
        size_t sz = sizeof(v);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &sz, nullptr, 0) == 0 && v >= 2) {
            return (size_t) std::min<int32_t>(v, 16);
        }
        return 6;
    }();
    return n_pcores;
}
#endif

static llama_token try_fast_greedy_argmax(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first, int n_vocab) {
    if (!gsmpl->simple_greedy_chain || grammar_first || n_vocab <= 0) {
        return LLAMA_TOKEN_NULL;
    }
    if (gsmpl->rbudget && common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_FORCING) {
        return LLAMA_TOKEN_NULL;
    }
    if (llama_get_sampled_token_ith(ctx, idx) != LLAMA_TOKEN_NULL ||
        llama_get_sampled_probs_ith(ctx, idx) != nullptr ||
        llama_get_sampled_logits_ith(ctx, idx) != nullptr) {
        return LLAMA_TOKEN_NULL;
    }
    const float * logits = llama_get_logits_ith(ctx, idx);
    if (!logits) {
        return LLAMA_TOKEN_NULL;
    }

    float max_val = logits[0];
    llama_token best_id = 0;
#if defined(__APPLE__)
    if (n_vocab >= 1024 && n_vocab <= INT_MAX) {
        vDSP_Length max_idx = 0;
        vDSP_maxvi(logits, 1, &max_val, &max_idx, (vDSP_Length) n_vocab);
        best_id = (llama_token) max_idx;
    } else
#endif
    {
        for (int k = 1; k < n_vocab; ++k) {
            if (logits[k] > max_val) {
                max_val = logits[k];
                best_id = (llama_token) k;
            }
        }
    }
    if (!std::isfinite(max_val)) {
        return LLAMA_TOKEN_NULL;
    }
    if (grammar_should_apply(gsmpl) && !grammar_allows_token(gsmpl->grmr, best_id)) {
        return LLAMA_TOKEN_NULL;
    }
    return best_id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    llama_synchronize(ctx);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int n_vocab = llama_vocab_n_tokens(vocab);

    size_t i = 0;
    for (; i < draft.size(); i++) {
        llama_token id = try_fast_greedy_argmax(gsmpl, ctx, idxs[i], grammar_first, n_vocab);
        if (id == LLAMA_TOKEN_NULL) {
            id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        }

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // do not accept draft tokens after an EOG - they are not output but would stay in the context
        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        llama_token id = try_fast_greedy_argmax(gsmpl, ctx, idxs[i], grammar_first, n_vocab);
        if (id == LLAMA_TOKEN_NULL) {
            id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        }

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    if (id >= 0 && (size_t) id < n && data[id].id == id) {
        return data[id].p;
    }
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

static llama_token common_sampler_eval_chain(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool apply_grammar) {
    llama_synchronize(ctx);

    const auto tm = gsmpl->tm();

    auto & grmr    = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain   = gsmpl->chain;
    auto & cur_p   = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    {
        const llama_token id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    llama_sampler_apply(rbudget, &cur_p);

    if (apply_grammar && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}

static bool grammar_allows_token(struct llama_sampler * grmr, llama_token id) {
    llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
    llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };
    llama_sampler_apply(grmr, &single_token_data_array);
    return single_token_data_array.data[0].logit != -INFINITY;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    auto & cand = gsmpl->cand_grmr; // persistent scratch buffer for grammar masking

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const bool masked = !grammar_first && grammar_should_apply(gsmpl);

        // Compute the target post-chain distribution once without running a full-vocab grammar pass
        // when masked == true; the exact single-token fast path below avoids full-vocab masking
        // whenever the drafted token is accepted.
        llama_token id_tgt = common_sampler_eval_chain(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, false);
        const auto & q     = draft_q[i];

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        auto lookup_raw_p = [&](const llama_token_data_array * arr, llama_token tok) -> float {
            if (tok >= 0 && (size_t) tok < arr->size && arr->data[tok].id == tok) {
                return arr->data[tok].p;
            }
            for (size_t k = 0; k < arr->size; ++k) {
                if (arr->data[k].id == tok) {
                    return arr->data[k].p;
                }
            }
            return 0.0f;
        };

        float p_raw_x = lookup_raw_p(cur_p, draft[i]);

        if (!masked) {
            if (q_x > 0.0f && (p_raw_x >= q_x || uni(gsmpl->rng) < p_raw_x / q_x)) {
                common_sampler_accept(gsmpl, draft[i], true);
                result.push_back(draft[i]);
                continue;
            }
        } else {
            // Exact lossless single-token grammar fast path:
            // Masking invalid tokens only shrinks the partition sum S = sum_{y in V} p(y) <= 1,
            // so for any grammar-valid token x in V, p_V(x) = p(x)/S >= p(x).
            // Conservatively leave a 1e-4 band for FP32 softmax sum rounding; if u * q_x < p_raw_x * k_safe,
            // then u * q_x < p_V(x) is guaranteed and we accept immediately without masking the full vocab.
            // Otherwise we compute exact S and re-test u * q_x < p_V(x) with the SAME u, giving exact
            // marginal acceptance probability min(1, p_V(x) / q(x)).
            constexpr float k_safe = 0.9999f;

            const bool draft_valid = grammar_allows_token(gsmpl->grmr, draft[i]);
            bool drew_u = false;
            float u = 0.0f;

            if (draft_valid && q_x > 0.0f) {
                if (p_raw_x * k_safe >= q_x) {
                    common_sampler_accept(gsmpl, draft[i], true);
                    result.push_back(draft[i]);
                    continue;
                }
                u = uni(gsmpl->rng);
                drew_u = true;
                if (u * q_x < p_raw_x * k_safe) {
                    common_sampler_accept(gsmpl, draft[i], true);
                    result.push_back(draft[i]);
                    continue;
                }
            }

            // Slow path: draft is invalid or lies outside the conservative lower-bound accept region.
            // Mask the full candidate array once and compute the exact renormalized distribution p_V.
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);

            float p_sum = 0.0f;
            for (size_t k = 0; k < cur_p->size; ++k) {
                if (cand[k].logit != -INFINITY) {
                    p_sum += cur_p->data[k].p;
                }
            }

            bool apply_cand_mask = true;
            float p_norm = 1.0f;
            if (p_sum > 0.0f) {
                p_norm = 1.0f / p_sum;
            } else {
                // Truncation samplers dropped all grammar-valid tokens; resample with grammar first.
                id_tgt = common_sampler_eval_chain(gsmpl, ctx, idxs[i], true);
                cur_p = common_sampler_get_candidates(gsmpl, false);
                p_raw_x = lookup_raw_p(cur_p, draft[i]);
                apply_cand_mask = false;
            }

            const float p_x = draft_valid ? p_raw_x * p_norm : 0.0f;
            if (draft_valid && q_x > 0.0f && (p_x >= q_x || (drew_u ? (u * q_x < p_x) : (uni(gsmpl->rng) * q_x < p_x)))) {
                common_sampler_accept(gsmpl, draft[i], true);
                result.push_back(draft[i]);
                continue;
            }

            auto p_of = [&](size_t k) {
                return (apply_cand_mask && cand[k].logit == -INFINITY) ? 0.0f : cur_p->data[k].p * p_norm;
            };

            const llama_token_data * q_data = q.data();
            const size_t q_size = q.size();
            float sum = 0.0f;
            llama_token fallback_id = (apply_cand_mask && !grammar_allows_token(gsmpl->grmr, id_tgt)) ? LLAMA_TOKEN_NULL : id_tgt;
            float best_valid_p = -1.0f;
            llama_token best_valid_id = LLAMA_TOKEN_NULL;
            llama_token last_pos_id = LLAMA_TOKEN_NULL;
            for (size_t k = 0; k < cur_p->size; ++k) {
                const float pk = p_of(k);
                if (pk > best_valid_p && (!apply_cand_mask || cand[k].logit != -INFINITY)) {
                    best_valid_p = pk;
                    best_valid_id = cur_p->data[k].id;
                }
                const float r = pk - prob_of(q_data, q_size, cur_p->data[k].id);
                if (r > 0.0f) {
                    sum += r;
                    last_pos_id = cur_p->data[k].id;
                }
            }

            llama_token id = (fallback_id != LLAMA_TOKEN_NULL) ? fallback_id : best_valid_id;
            if (sum > 0.0f) {
                float u_res = uni(gsmpl->rng) * sum;
                id = last_pos_id;
                for (size_t k = 0; k < cur_p->size; ++k) {
                    const float r = p_of(k) - prob_of(q_data, q_size, cur_p->data[k].id);
                    if (r > 0.0f) {
                        u_res -= r;
                        if (u_res <= 0.0f) {
                            id = cur_p->data[k].id;
                            break;
                        }
                    }
                }
            }

            common_sampler_accept(gsmpl, id, true);
            result.push_back(id);

            break;
        }

        // rejected (unmasked path): tokens outside q's support keep all of p
        const llama_token_data * q_data = q.data();
        const size_t q_size = q.size();
        float sum = 0.0f;
        llama_token last_pos_id = id_tgt;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = cur_p->data[k].p - prob_of(q_data, q_size, cur_p->data[k].id);
            if (r > 0.0f) {
                sum += r;
                last_pos_id = cur_p->data[k].id;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u_res = uni(gsmpl->rng) * sum;
            id = last_pos_id;
            for (size_t k = 0; k < cur_p->size; ++k) {
                const float r = cur_p->data[k].p - prob_of(q_data, q_size, cur_p->data[k].id);
                if (r > 0.0f) {
                    u_res -= r;
                    if (u_res <= 0.0f) {
                        id = cur_p->data[k].id;
                        break;
                    }
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

static bool compute_exp_sum_contig(const float * logits, int n_vocab, float temp, std::vector<float> & p_out, float & sum_out) {
    if (!logits || n_vocab <= 0 || !(temp > 0.0f)) {
        return false;
    }
    p_out.resize((size_t) n_vocab);
    float * p = p_out.data();

#if defined(__APPLE__)
    if (n_vocab >= 1024 && n_vocab <= INT_MAX) {
        const vDSP_Length n = (vDSP_Length) n_vocab;
        float max_l = -INFINITY;
        const float inv_t = (temp == 1.0f) ? 1.0f : (1.0f / temp);
        vDSP_maxv(logits, 1, &max_l, n);
        max_l *= inv_t;
        if (!std::isfinite(max_l)) {
            return false;
        }

        if (n_vocab >= 32768) {
            const size_t n_chunks = apple_pcore_count();
            const size_t chunk_sz = ((size_t) n_vocab + n_chunks - 1) / n_chunks;
            float partial_sums[16] = {};
            struct exp_ctx {
                const float * logits;
                float       * p;
                float       * partial_sums;
                size_t        n_total;
                size_t        chunk_sz;
                float         inv_t;
                float         neg_max;
                bool          unit_temp;
            } ectx = { logits, p, partial_sums, (size_t) n_vocab, chunk_sz, inv_t, -max_l, temp == 1.0f };

            dispatch_apply_f(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), &ectx, [](void * ptr, size_t c) {
                const auto * ctx = (const exp_ctx *) ptr;
                const size_t beg = c * ctx->chunk_sz;
                const size_t end = std::min(beg + ctx->chunk_sz, ctx->n_total);
                if (beg >= end) {
                    return;
                }
                const vDSP_Length len = (vDSP_Length) (end - beg);
                const float * src = ctx->logits + beg;
                float       * dst = ctx->p + beg;
                if (ctx->unit_temp) {
                    vDSP_vsadd(src, 1, &ctx->neg_max, dst, 1, len);
                } else {
                    vDSP_vsmsa(src, 1, &ctx->inv_t, &ctx->neg_max, dst, 1, len);
                }
                int li = (int) len;
                vvexpf(dst, dst, &li);
                float s = 0.0f;
                vDSP_sve(dst, 1, &s, len);
                ctx->partial_sums[c] = s;
            });

            double sum_d = 0.0;
            for (size_t c = 0; c < n_chunks; ++c) {
                sum_d += (double) partial_sums[c];
            }
            const float sum_p = (float) sum_d;
            if (!(sum_p > 0.0f) || !std::isfinite(sum_p)) {
                return false;
            }
            sum_out = sum_p;
            return true;
        }

        const float neg_max = -max_l;
        if (temp == 1.0f) {
            vDSP_vsadd(logits, 1, &neg_max, p, 1, n);
        } else {
            vDSP_vsmsa(logits, 1, &inv_t, &neg_max, p, 1, n);
        }
        int ni = n_vocab;
        vvexpf(p, p, &ni);
        float sum_p = 0.0f;
        vDSP_sve(p, 1, &sum_p, n);
        if (!(sum_p > 0.0f) || !std::isfinite(sum_p)) {
            return false;
        }
        sum_out = sum_p;
        return true;
    }
#endif

    const float inv_t = (temp == 1.0f) ? 1.0f : (1.0f / temp);
    float max_l = logits[0] * inv_t;
    for (int i = 1; i < n_vocab; ++i) {
        const float v = logits[i] * inv_t;
        if (v > max_l) {
            max_l = v;
        }
    }
    if (!std::isfinite(max_l)) {
        return false;
    }
    double sum_p = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const float v = expf(logits[i] * inv_t - max_l);
        p[i] = v;
        sum_p += v;
    }
    if (!(sum_p > 0.0) || !std::isfinite(sum_p)) {
        return false;
    }
    sum_out = (float) sum_p;
    return true;
}

static llama_token sample_from_probs_contig(const float * p, int n_vocab, std::mt19937 & rng, double total_mass = 1.0) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double u = uni(rng) * total_mass;
    llama_token last_pos = n_vocab - 1;
    for (int i = 0; i < n_vocab; ++i) {
        const float pi = p[i];
        if (pi > 0.0f) {
            last_pos = (llama_token) i;
            u -= (double) pi;
            if (u <= 0.0) {
                return (llama_token) i;
            }
        }
    }
    return last_pos;
}

static llama_token sample_residual_contig(
        const float * p,
        int n_vocab,
        const std::vector<float> & q,
        std::vector<float> & resid_buf,
        std::mt19937 & rng,
        const llama_token * draft_d2t = nullptr,
        int32_t draft_n_d2t = 0) {
    resid_buf.resize((size_t) n_vocab);
    float * r = resid_buf.data();
    float sum = 0.0f;

    if (draft_d2t != nullptr && draft_n_d2t > 0) {
        std::memcpy(r, p, (size_t) n_vocab * sizeof(float));
        const int n_q = (int) std::min<size_t>((size_t) draft_n_d2t, q.size());
        const float * q_data = q.data();
        for (int j = 0; j < n_q; ++j) {
            const llama_token tid = draft_d2t[j];
            if (tid >= 0 && tid < n_vocab) {
                r[tid] -= q_data[j];
            }
        }
#if defined(__APPLE__)
        if (n_vocab >= 1024 && n_vocab <= INT_MAX) {
            const float zero = 0.0f;
            vDSP_vthres(r, 1, &zero, r, 1, (vDSP_Length) n_vocab);
            vDSP_sve(r, 1, &sum, (vDSP_Length) n_vocab);
        } else
#endif
        {
            for (int i = 0; i < n_vocab; ++i) {
                if (r[i] > 0.0f) {
                    sum += r[i];
                } else {
                    r[i] = 0.0f;
                }
            }
        }
    } else {
        const int n_common = (int) std::min<size_t>((size_t) n_vocab, q.size());
#if defined(__APPLE__)
        if (n_vocab >= 1024 && n_vocab <= INT_MAX) {
            if (n_common > 0) {
                // vDSP_vsub(B, 1, A, 1, C, 1, N) computes C = A - B
                vDSP_vsub(q.data(), 1, p, 1, r, 1, (vDSP_Length) n_common);
            }
            if (n_vocab > n_common) {
                std::memcpy(r + n_common, p + n_common, (size_t) (n_vocab - n_common) * sizeof(float));
            }
            const float zero = 0.0f;
            vDSP_vthres(r, 1, &zero, r, 1, (vDSP_Length) n_vocab);
            vDSP_sve(r, 1, &sum, (vDSP_Length) n_vocab);
        } else
#endif
        {
            for (int i = 0; i < n_common; ++i) {
                const float d = p[i] - q[i];
                if (d > 0.0f) {
                    r[i] = d;
                    sum += d;
                } else {
                    r[i] = 0.0f;
                }
            }
            for (int i = n_common; i < n_vocab; ++i) {
                const float d = p[i];
                if (d > 0.0f) {
                    r[i] = d;
                    sum += d;
                } else {
                    r[i] = 0.0f;
                }
            }
        }
    }

    if (sum > 0.0f) {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        double u_res = uni(rng) * (double) sum;
        llama_token last_pos = 0;
        for (int i = 0; i < n_vocab; ++i) {
            const float ri = r[i];
            if (ri > 0.0f) {
                last_pos = (llama_token) i;
                u_res -= (double) ri;
                if (u_res <= 0.0) {
                    return (llama_token) i;
                }
            }
        }
        return last_pos;
    }

    return sample_from_probs_contig(p, n_vocab, rng);
}

static bool can_use_fast_softmax(const common_sampler * gsmpl, llama_context * ctx, int idx, bool grammar_first) {
    if (!gsmpl->simple_softmax_chain || grammar_first) {
        return false;
    }
    if (gsmpl->rbudget && common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_FORCING) {
        return false;
    }
    if (llama_get_sampled_token_ith(ctx, idx) != LLAMA_TOKEN_NULL ||
        llama_get_sampled_probs_ith(ctx, idx) != nullptr ||
        llama_get_sampled_logits_ith(ctx, idx) != nullptr) {
        return false;
    }
    return llama_get_logits_ith(ctx, idx) != nullptr;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const std::vector<int> & idxs,
        const llama_tokens & draft,
        const std::vector<std::vector<float>> & draft_q,
        bool grammar_first,
        const llama_token * draft_d2t,
        int32_t draft_n_d2t,
        const int32_t * draft_t2d) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    llama_synchronize(ctx);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const bool pruned_q = (draft_d2t != nullptr && draft_n_d2t > 0 && draft_t2d != nullptr);

    auto & cand = gsmpl->cand_grmr;

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        const auto & q = draft_q[i];
        const float * q_data = q.data();
        const size_t  q_size = q.size();
        auto lookup_q = [&](llama_token tok) -> float {
            if (tok < 0) {
                return 0.0f;
            }
            if (pruned_q) {
                if (tok >= n_vocab) {
                    return 0.0f;
                }
                const int32_t j = draft_t2d[tok];
                return (j >= 0 && (size_t) j < q_size) ? q_data[j] : 0.0f;
            }
            return ((size_t) tok < q_size) ? q_data[tok] : 0.0f;
        };

        const llama_token x = draft[i];
        const float q_x = lookup_q(x);

        bool fast_ok = false;
        const float * logits_raw = nullptr;
        float exp_sum = 0.0f;
        if (can_use_fast_softmax(gsmpl, ctx, idxs[i], grammar_first)) {
            const auto tm = gsmpl->tm();
            logits_raw = llama_get_logits_ith(ctx, idxs[i]);
            fast_ok = compute_exp_sum_contig(logits_raw, n_vocab, gsmpl->params.temp, gsmpl->p_buf, exp_sum);
        }

        if (fast_ok) {
            float * p_exp = gsmpl->p_buf.data();
            const float inv_exp_sum = 1.0f / exp_sum;
            const float p_raw_x = (x >= 0 && x < n_vocab) ? (p_exp[x] * inv_exp_sum) : 0.0f;

            if (!masked) {
                if (q_x > 0.0f && (p_raw_x >= q_x || uni(gsmpl->rng) < p_raw_x / q_x)) {
                    common_sampler_accept(gsmpl, x, true);
                    result.push_back(x);
                    continue;
                }

                const llama_token id = [&]() {
                    const auto tm = gsmpl->tm();
#if defined(__APPLE__)
                    if (n_vocab >= 1024 && n_vocab <= INT_MAX) {
                        vDSP_vsmul(p_exp, 1, &inv_exp_sum, p_exp, 1, (vDSP_Length) n_vocab);
                    } else
#endif
                    {
                        for (int k = 0; k < n_vocab; ++k) {
                            p_exp[k] *= inv_exp_sum;
                        }
                    }
                    return sample_residual_contig(p_exp, n_vocab, q, gsmpl->resid_buf, gsmpl->rng,
                            pruned_q ? draft_d2t : nullptr, pruned_q ? draft_n_d2t : 0);
                }();
                common_sampler_accept(gsmpl, id, true);
                result.push_back(id);
                break;
            }

            // Exact lossless single-token grammar fast path on contiguous probabilities
            constexpr float k_safe = 0.9999f;

            const bool draft_valid = grammar_allows_token(gsmpl->grmr, x);
            bool drew_u = false;
            float u = 0.0f;

            if (draft_valid && q_x > 0.0f) {
                if (p_raw_x * k_safe >= q_x) {
                    common_sampler_accept(gsmpl, x, true);
                    result.push_back(x);
                    continue;
                }
                u = uni(gsmpl->rng);
                drew_u = true;
                if (u * q_x < p_raw_x * k_safe) {
                    common_sampler_accept(gsmpl, x, true);
                    result.push_back(x);
                    continue;
                }
            }

            // Slow path: mask full vocabulary once and compute exact renormalized p_V directly from p_exp
            {
                const auto tm = gsmpl->tm();
                cand.resize((size_t) n_vocab);
#if defined(__APPLE__)
                if (n_vocab >= 32768) {
                    const size_t n_chunks = apple_pcore_count();
                    const size_t chunk_sz = ((size_t) n_vocab + n_chunks - 1) / n_chunks;
                    struct init_ctx {
                        llama_token_data * cand;
                        const float      * logits;
                        size_t             n_total;
                        size_t             chunk_sz;
                    } ictx = { cand.data(), logits_raw, (size_t) n_vocab, chunk_sz };
                    dispatch_apply_f(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), &ictx, [](void * ptr, size_t c) {
                        const auto * ctx = (const init_ctx *) ptr;
                        const size_t beg = c * ctx->chunk_sz;
                        const size_t end = std::min(beg + ctx->chunk_sz, ctx->n_total);
                        for (size_t k = beg; k < end; ++k) {
                            ctx->cand[k] = { (llama_token) k, ctx->logits[k], 0.0f };
                        }
                    });
                } else
#endif
                {
                    for (int k = 0; k < n_vocab; ++k) {
                        cand[k] = { (llama_token) k, logits_raw[k], 0.0f };
                    }
                }
                llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
                llama_sampler_apply(gsmpl->grmr, &arr);
            }

            float p_sum = 0.0f;
            float best_valid_exp = -1.0f;
            llama_token best_valid_id = LLAMA_TOKEN_NULL;
            for (int k = 0; k < n_vocab; ++k) {
                if (cand[k].logit != -INFINITY) {
                    const float ek = p_exp[k];
                    p_sum += ek;
                    if (ek > best_valid_exp) {
                        best_valid_exp = ek;
                        best_valid_id = (llama_token) k;
                    }
                }
            }

            if (p_sum > 0.0f) {
                const float p_norm = 1.0f / p_sum;
                const float p_x = (draft_valid && x >= 0 && x < n_vocab) ? (p_exp[x] * p_norm) : 0.0f;
                if (draft_valid && q_x > 0.0f && (p_x >= q_x || (drew_u ? (u * q_x < p_x) : (uni(gsmpl->rng) * q_x < p_x)))) {
                    common_sampler_accept(gsmpl, x, true);
                    result.push_back(x);
                    continue;
                }

                float sum = 0.0f;
                llama_token last_pos_id = LLAMA_TOKEN_NULL;
                for (int k = 0; k < n_vocab; ++k) {
                    if (cand[k].logit == -INFINITY) {
                        continue;
                    }
                    const float r = p_exp[k] * p_norm - lookup_q((llama_token) k);
                    if (r > 0.0f) {
                        cand[k].p = r;
                        sum += r;
                        last_pos_id = (llama_token) k;
                    } else {
                        cand[k].p = 0.0f;
                    }
                }

                llama_token id = best_valid_id;
                if (sum > 0.0f) {
                    float u_res = uni(gsmpl->rng) * sum;
                    id = last_pos_id;
                    for (int k = 0; k < n_vocab; ++k) {
                        if (cand[k].logit == -INFINITY) {
                            continue;
                        }
                        const float r = cand[k].p;
                        if (r > 0.0f) {
                            u_res -= r;
                            if (u_res <= 0.0f) {
                                id = (llama_token) k;
                                break;
                            }
                        }
                    }
                }

                common_sampler_accept(gsmpl, id, true);
                result.push_back(id);
                break;
            }
            // Fall through to full chain with grammar_first if p_sum == 0 (all valid tokens underflowed FP32)
        }

        // General path: evaluate full sampler chain, still using O(1) contiguous q lookup
        llama_token id_tgt = common_sampler_eval_chain(gsmpl, ctx, idxs[i], grammar_first || (fast_ok && masked));

        const auto * cur_p = common_sampler_get_candidates(gsmpl, false);

        auto lookup_raw_p = [&](const llama_token_data_array * arr, llama_token tok) -> float {
            if (tok >= 0 && (size_t) tok < arr->size && arr->data[tok].id == tok) {
                return arr->data[tok].p;
            }
            for (size_t k = 0; k < arr->size; ++k) {
                if (arr->data[k].id == tok) {
                    return arr->data[k].p;
                }
            }
            return 0.0f;
        };

        float p_raw_x = lookup_raw_p(cur_p, x);

        if (!masked || fast_ok) {
            if (q_x > 0.0f && (p_raw_x >= q_x || uni(gsmpl->rng) < p_raw_x / q_x)) {
                common_sampler_accept(gsmpl, x, true);
                result.push_back(x);
                continue;
            }

            float sum = 0.0f;
            llama_token last_pos_id = id_tgt;
            for (size_t k = 0; k < cur_p->size; ++k) {
                const float r = cur_p->data[k].p - lookup_q(cur_p->data[k].id);
                if (r > 0.0f) {
                    sum += r;
                    last_pos_id = cur_p->data[k].id;
                }
            }

            llama_token id = id_tgt;
            if (sum > 0.0f) {
                float u_res = uni(gsmpl->rng) * sum;
                id = last_pos_id;
                for (size_t k = 0; k < cur_p->size; ++k) {
                    const float r = cur_p->data[k].p - lookup_q(cur_p->data[k].id);
                    if (r > 0.0f) {
                        u_res -= r;
                        if (u_res <= 0.0f) {
                            id = cur_p->data[k].id;
                            break;
                        }
                    }
                }
            }

            common_sampler_accept(gsmpl, id, true);
            result.push_back(id);
            break;
        }

        // General masked path with single-token grammar fast path
        constexpr float k_safe = 0.9999f;

        const bool draft_valid = grammar_allows_token(gsmpl->grmr, x);
        bool drew_u = false;
        float u = 0.0f;

        if (draft_valid && q_x > 0.0f) {
            if (p_raw_x * k_safe >= q_x) {
                common_sampler_accept(gsmpl, x, true);
                result.push_back(x);
                continue;
            }
            u = uni(gsmpl->rng);
            drew_u = true;
            if (u * q_x < p_raw_x * k_safe) {
                common_sampler_accept(gsmpl, x, true);
                result.push_back(x);
                continue;
            }
        }

        cand.assign(cur_p->data, cur_p->data + cur_p->size);
        llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
        llama_sampler_apply(gsmpl->grmr, &arr);

        float p_sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cand[k].logit != -INFINITY) {
                p_sum += cur_p->data[k].p;
            }
        }

        bool apply_cand_mask = true;
        float p_norm = 1.0f;
        if (p_sum > 0.0f) {
            p_norm = 1.0f / p_sum;
        } else {
            id_tgt = common_sampler_eval_chain(gsmpl, ctx, idxs[i], true);
            cur_p = common_sampler_get_candidates(gsmpl, false);
            p_raw_x = lookup_raw_p(cur_p, x);
            apply_cand_mask = false;
        }

        const float p_x = draft_valid ? p_raw_x * p_norm : 0.0f;
        if (draft_valid && q_x > 0.0f && (p_x >= q_x || (drew_u ? (u * q_x < p_x) : (uni(gsmpl->rng) * q_x < p_x)))) {
            common_sampler_accept(gsmpl, x, true);
            result.push_back(x);
            continue;
        }

        auto p_of = [&](size_t k) {
            return (apply_cand_mask && cand[k].logit == -INFINITY) ? 0.0f : cur_p->data[k].p * p_norm;
        };

        float sum = 0.0f;
        llama_token fallback_id = (apply_cand_mask && !grammar_allows_token(gsmpl->grmr, id_tgt)) ? LLAMA_TOKEN_NULL : id_tgt;
        float best_valid_p = -1.0f;
        llama_token best_valid_id = LLAMA_TOKEN_NULL;
        llama_token last_pos_id = LLAMA_TOKEN_NULL;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float pk = p_of(k);
            if (pk > best_valid_p && (!apply_cand_mask || cand[k].logit != -INFINITY)) {
                best_valid_p = pk;
                best_valid_id = cur_p->data[k].id;
            }
            const float r = pk - lookup_q(cur_p->data[k].id);
            if (r > 0.0f) {
                sum += r;
                last_pos_id = cur_p->data[k].id;
            }
        }

        llama_token id = (fallback_id != LLAMA_TOKEN_NULL) ? fallback_id : best_valid_id;
        if (sum > 0.0f) {
            float u_res = uni(gsmpl->rng) * sum;
            id = last_pos_id;
            for (size_t k = 0; k < cur_p->size; ++k) {
                const float r = p_of(k) - lookup_q(cur_p->data[k].id);
                if (r > 0.0f) {
                    u_res -= r;
                    if (u_res <= 0.0f) {
                        id = cur_p->data[k].id;
                        break;
                    }
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        break;
    }

    if (i == draft.size()) {
        llama_token id = LLAMA_TOKEN_NULL;
        if (can_use_fast_softmax(gsmpl, ctx, idxs[i], grammar_first)) {
            const auto tm = gsmpl->tm();
            const float * logits_raw = llama_get_logits_ith(ctx, idxs[i]);
            float bonus_sum = 0.0f;
            if (compute_exp_sum_contig(logits_raw, n_vocab, gsmpl->params.temp, gsmpl->p_buf, bonus_sum)) {
                const llama_token cand_id = sample_from_probs_contig(gsmpl->p_buf.data(), n_vocab, gsmpl->rng, (double) bonus_sum);
                if (!grammar_should_apply(gsmpl) || grammar_allows_token(gsmpl->grmr, cand_id)) {
                    id = cand_id;
                }
            }
        }
        if (id == LLAMA_TOKEN_NULL) {
            id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first || grammar_should_apply(gsmpl));
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
