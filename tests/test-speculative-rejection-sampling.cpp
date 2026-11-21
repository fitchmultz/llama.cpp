#include "sampling.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <unordered_map>

// Tests that common_sampler_sample_and_accept_n_rejection preserves the target
// distribution regardless of what the draft distribution Q looks like.

static int check_rejection_distribution(
        const llama_model * model_tgt,
        llama_context     * ctx_tgt,
        common_params_sampling sampling,
        bool                grammar_first_ref,
        bool                grammar_first_test,
        int                 n_trials,
        const char        * label) {
    // Reference sampler computes the exact target distribution.
    common_sampler_ptr smpl_ref(common_sampler_init(model_tgt, sampling));
    if (!smpl_ref) {
        LOG_ERR("%s: failed to initialize reference sampler\n", label);
        return 1;
    }

    common_sampler_sample(smpl_ref.get(), ctx_tgt, -1, grammar_first_ref);
    const llama_token_data_array * dist_tgt = common_sampler_get_candidates(smpl_ref.get(), true);

    // Verification sampler uses the same sampling config (without grammar_first when testing the fast path).
    common_sampler_ptr smpl_verify(common_sampler_init(model_tgt, sampling));
    if (!smpl_verify) {
        LOG_ERR("%s: failed to initialize verification sampler\n", label);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::mt19937 rng(18);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::unordered_map<llama_token, int> token_counts;

    std::vector<llama_token_data> q(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        q[i].id = i;
        q[i].logit = 0.0f;
    }

    // Build a lookup of top target tokens to occasionally bias q near p (exercising both fast-accept and slow-path).
    std::vector<llama_token> hot_tokens;
    for (size_t i = 0; i < std::min<size_t>(dist_tgt->size, 16); ++i) {
        if (dist_tgt->data[i].p > 0.0f) {
            hot_tokens.push_back(dist_tgt->data[i].id);
        }
    }

    for (int trial = 0; trial < n_trials; ++trial) {
        float q_sum = 0.0f;

        if ((trial & 1) == 0 && !hot_tokens.empty()) {
            // Concentrated draft Q around hot valid tokens + a few random tokens to exercise fast-accept heavily.
            for (auto & td : q) {
                td.p = 1e-6f;
                q_sum += td.p;
            }
            for (llama_token id : hot_tokens) {
                const float w = 0.05f + uni(rng);
                q[id].p += w;
                q_sum += w;
            }
            const llama_token noise = (llama_token) (rng() % (uint32_t) n_vocab);
            q[noise].p += 0.5f;
            q_sum += 0.5f;
        } else {
            for (auto & td : q) {
                td.p = uni(rng);
                q_sum += td.p;
            }
        }

        const float inv_sum = 1.0f / q_sum;
        for (auto & td : q) {
            td.p *= inv_sum;
        }

        float u = uni(rng);

        // Fallback to the last bucket to absorb residual probability from FP precision loss
        llama_token draft_tok = q.back().id;
        for (const auto & e : q) {
            u -= e.p;
            if (u <= 0.0f) {
                draft_tok = e.id;
                break;
            }
        }

        // Both indices reuse the prompt's last output logits. The second entry
        // is the bonus draw slot; it is always stale and its result is ignored.
        const llama_tokens draft = { draft_tok };
        const std::vector<std::vector<llama_token_data>> draft_q = { q };
        const std::vector<int> idxs = { -1, -1 };

        const std::vector<llama_token> result =
            common_sampler_sample_and_accept_n_rejection(smpl_verify.get(), ctx_tgt, idxs, draft, draft_q, grammar_first_test);

        token_counts[result[0]]++;
    }

    LOG("\n[%s] Empirical vs target distribution after %d trials:\n", label, n_trials);

    bool   ok          = true;
    size_t covered     = 0;
    float  tail_p_tgt  = 0.0f;
    size_t tail_count  = 0;
    const float min_bin_p = 5.0f / n_trials; // normal-approx validity threshold for per-token 6-sigma check

    auto get_count_for = [&](llama_token id) -> int {
        const auto it = token_counts.find(id);
        return it != token_counts.end() ? it->second : 0;
    };

    for (size_t i = 0; i < dist_tgt->size; ++i) {
        const llama_token id  = dist_tgt->data[i].id;
        const float       p_t = dist_tgt->data[i].p;

        const int   count = get_count_for(id);
        const float p_e   = (float) count / n_trials;
        const float diff  = std::fabs(p_t - p_e);

        covered += count;

        if (p_t == 0.0f) {
            if (count > 0) {
                LOG_ERR("[%s] token %d: forbidden token sampled %d times (target = 0)\n", label, id, count);
                ok = false;
            }
            continue;
        }

        if (p_t < min_bin_p) {
            tail_p_tgt += p_t;
            tail_count += count;
            continue;
        }

        const float standard_error = std::sqrt(p_t * (1.0f - p_t) / n_trials);
        const float tolerance      = 6.0f * standard_error;

        if (p_t >= 0.005f || diff > tolerance) {
            LOG("token %5d : target = %f, empirical = %f, diff = %f, tol = %f\n", id, p_t, p_e, diff, tolerance);
        }

        if (diff > tolerance) {
            LOG_ERR("[%s] token %d: empirical probability %f too far from target %f (diff %f > tol %f)\n",
                    label, id, p_e, p_t, diff, tolerance);
            ok = false;
        }
    }

    if (tail_p_tgt > 0.0f) {
        const float tail_p_e  = (float) tail_count / n_trials;
        const float tail_diff = std::fabs(tail_p_tgt - tail_p_e);
        const float tail_tol  = 6.0f * std::sqrt(tail_p_tgt * (1.0f - tail_p_tgt) / n_trials);
        LOG("tail  bucket: target = %f, empirical = %f, diff = %f, tol = %f\n", tail_p_tgt, tail_p_e, tail_diff, tail_tol);
        if (tail_diff > tail_tol) {
            LOG_ERR("[%s] tail bucket: empirical probability %f too far from target %f (diff %f > tol %f)\n",
                    label, tail_p_e, tail_p_tgt, tail_diff, tail_tol);
            ok = false;
        }
    }

    if (covered != (size_t) n_trials) {
        LOG_ERR("[%s] %zu of %d trials produced a token outside dist_tgt's candidates\n",
                label, (size_t) n_trials - covered, n_trials);
        ok = false;
    }

    return ok ? 0 : 1;
}

static int test_speculative_rejection_sampling(common_params & params, int n_trials) {
    params.sampling.backend_sampling = false;
    auto llama_init = common_init_from_params(params);

    auto * model_tgt = llama_init->model();
    auto * ctx_tgt   = llama_init->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("failed to initialize target model and context\n");
        return 1;
    }

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };

    // Process the prompt with the target model to get its true distribution for
    // the position right after the prompt.
    common_batch batch = common_batch_get_one(ctx_tgt, prompt);
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("failed to decode the test prompt\n");
        return 1;
    }

    common_params_sampling sampling;
    sampling.seed     = 18;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    sampling.temp     = 1.0f;

    if (check_rejection_distribution(model_tgt, ctx_tgt, sampling, false, false, n_trials, "unconstrained") != 0) {
        return 1;
    }

    // Grammar-masked rejection sampling: stationary grammar (accepts [a-zA-Z0-9 ] or EOG at every step)
    // verifies the single-token fast path preserves the exact grammar-masked target distribution and
    // never emits grammar-rejected tokens.
    common_params_sampling sampling_grmr = sampling;
    sampling_grmr.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= [a-zA-Z0-9 ]*");
    if (check_rejection_distribution(model_tgt, ctx_tgt, sampling_grmr, true, false, n_trials, "grammar-masked") != 0) {
        return 1;
    }

    return 0;
}

struct parsed_args {
    int n_trials;
    std::vector<char *> argv;
};

static std::optional<parsed_args> parse_n_trials_arg(int argc, char ** argv, int default_n_trials) {
    parsed_args result;
    result.n_trials = default_n_trials;
    result.argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-trials") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --n-trials requires a number argument\n", __func__);
                return std::nullopt;
            }
            int value = std::atoi(argv[i + 1]);
            if (value <= 0) {
                LOG_ERR("%s: --n-trials requires a positive number argument\n", __func__);
                return std::nullopt;
            }
            result.n_trials = value;
            i++;
        } else {
            result.argv.push_back(argv[i]);
        }
    }
    result.argv.push_back(nullptr);

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 256;

    common_init();

    auto parsed = parse_n_trials_arg(argc, argv, 5000);
    if (!parsed) {
        return 1;
    }
    const int fargc = (int) parsed->argv.size() - 1;

    if (!common_params_parse(fargc, parsed->argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("Using target model '%s'\n", params.model.path.c_str());

    llama_backend_init();

    const int result = test_speculative_rejection_sampling(params, parsed->n_trials);
    if (result == 0) {
        LOG("\nAll tests passed.\n");
    } else {
        LOG("\nSome tests failed.\n");
    }

    llama_backend_free();
    return result;
}
