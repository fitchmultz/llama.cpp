# Inkling 2 Mini on Apple M5 Max (Metal 4 + Lossless Hybrid DSpark)

Reproducible build, benchmark, and server guide for running Thinking Machines Lab's **Inkling 2 Mini** (`26.07B` total parameters, 128 MoE experts / 8 active, 48 virtual layers, 1M native context) on Apple Silicon with Metal 4 tensor operations (`fitchmultz/llama.cpp@inkling-opt`).

## Quick start (Apple Silicon / macOS)

### 1. Build

```sh
git clone -b inkling-opt https://github.com/fitchmultz/llama.cpp.git
cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
ninja -C build llama-bench llama-server llama-perplexity llama-tokenize
```

### 2. Get weights

Download an Inkling 2 Mini GGUF and its DSpark draft model (for example from `unsloth/Inkling-2-Mini-GGUF`):

- Target model: `Inkling-2-Mini-Native8-Full-Q8_0.gguf` (`16.76 GiB`, native NVFP4 experts on layers `0..19` + `Q8_0` dense/layers `20..23`) or `Inkling-2-Mini-UD-Q8_K_XL.gguf` (`21.29 GiB`).
- Draft model: `dspark-Inkling-2-Mini-Q8_0.gguf` (`384 MB`).
- Vision projector (optional): `mmproj-F16.gguf` (`610 MB`).

### 3. Run the `llama-bench` suite

```sh
./scripts/bench-inkling-m5-max.sh /path/to/Inkling-2-Mini-Native8-Full-Q8_0.gguf
```

### 4. Run `llama-server` (100% lossless probabilistic DSpark + adaptive prompt-lookup)

For a single 256K-400K slot (works on 32-64 GB Macs too by adjusting `--ctx-size`):

```sh
./build/bin/llama-server \
  -m /path/to/Inkling-2-Mini-Native8-Full-Q8_0.gguf \
  -md /path/to/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-type ngram-simple,draft-dspark \
  --spec-ngram-simple-size-n 8 \
  --spec-ngram-simple-size-m 47 \
  --spec-draft-sampling probabilistic \
  --spec-draft-n-max 7 \
  --spec-draft-adaptive \
  --spec-draft-n-ubatch 64 \
  -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 \
  --ubatch-size 1024 -t 4 --jinja \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --ctx-size 400000 --port 8000
```

On an M5 Max 128 GB (with `sudo sysctl iogpu.wired_limit_mb=114688`), add `--parallel 6 --no-kv-unified --no-cache-idle-slots --decode-share 0.5 --ctx-size 2400000` for **6 independent 400K native contexts** (`2.4M` total KV tokens in `~81.5 GiB` wired RAM).

## Measured M5 Max 128 GB results

Hardware: Apple M5 Max (40 GPU cores, 128 GB unified memory, Metal 4 tensor ops, `iogpu.wired_limit_mb=114688`), `Inkling-2-Mini-Native8-Full-Q8_0.gguf` (`16.76 GiB`), `q8_0` KV cache, FlashAttention on.

### Raw `llama-bench` (no speculative decoding)

| Test | `-ub` | Upstream `b12588` | Prior `inkling-opt` (`b171b04c`) | **Current `inkling-opt`** | Gain vs `b171b04c` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tg128` | `1024` | `33.77 tok/s` | `68.34 tok/s` | **`69.93 tok/s`** | **`+2.3%`** (`2.07x` vs `b12588`) |
| `pp4` | `1024` | — | `186.8 tok/s` | **`191.7 tok/s`** | **`+2.6%`** |
| `pp8` | `1024` | — | `295.4 tok/s` | **`299.5 tok/s`** | **`+1.4%`** |
| `pp16` | `1024` | — | `408.6 tok/s` | **`416.8 tok/s`** | **`+2.0%`** |
| `pp512` | `1024` | `968.7 tok/s` | `2,705 tok/s` | **`2,902 tok/s`** | **`+7.3%`** (`3.00x` vs `b12588`) |
| `pp1024` | `1024` | — | `3,095 tok/s` | **`3,321 tok/s`** | **`+7.3%`** |
| `pp2048` | `1024` | `834.9 tok/s` | `2,998 tok/s` | **`3,229 tok/s`** | **`+7.7%`** (`3.87x` vs `b12588`) |
| `pp512 @ d65536` | `1024` | — | `1,364 tok/s` | **`1,452 tok/s`** | **`+6.5%`** |

### Live `llama-server` (`6x400K` slots, 100% lossless at `temp=1.0`, interleaved `CTRL -> CAND -> CTRL`)

| Workload | Prior (`b171b04c`, mean) | **Current (`ngram-simple` + DSpark)** | Delta |
| --- | ---: | ---: | ---: |
| `draft-dspark` draft latency (graph dispatches/barriers) | `2.74 ms/call` (`243` / `179`) | **`2.42 ms/call`** (`125` / `90`) | **`-11.8%`** (`-48.6%` / `-49.7%`) |
| Prose `0K` (`temp=1.0`, lossless) | `78.1 tok/s` | **`87.4 tok/s`** (peak `94.8`) | **`+11.9%`** |
| Tool call (`temp=1.0`, lossless) | `128.7 tok/s` | **`141.9 tok/s`** | **`+10.3%`** |
| Python file rewrite (`temp=1.0`, lossless) | `92.9 tok/s` | **`100.3 tok/s`** | **`+8.0%`** |
| Greedy JSON `1..120` (`exact120=true`) | `275.6 tok/s` | **`265.8–276.9 tok/s`** (`exact120=true`) | Parity |
| Host wired RAM (`6x400K` + vision + draft) | `82.1 GiB` | **`81.9 GiB`** | `-0.2 GiB` (`26.9 GiB` below `108.8` cap) |

## What's in the optimization stack

1. **Expert-contiguous direct-device Metal 4 MoE pipeline (`adapt64`) with static `K=64` specialization:** Packs active tokens into contiguous per-expert FP16 slices (`gather_f32_f16`), runs a dense-equivalent 8 KB shmem-`sa` / direct-device `tB` Metal 4 `matmul2d` (`NRA=64`, compile-time static `NK=64` + hoisted A-tile pointers when `K % 64 == 0`, register-spill-free 3-tier adaptive `NRB` `16/32/64`), fuses `SITU` into FP16, and streams directly into vectorized `float4` `BT=32` `SSM_CONV` + expert reduction.
2. **Metal 4 tensor fusions for prefill loop control, barrier-free small-batch `LOOP_DOWN_NORM`, & in-place `FWHT_MUL_SIGMOID`:** Eliminates intermediate loop-control and gating buffers across all 48 layers; runs 1 simdgroup per token (zero shmem, zero barriers) for `T = 2..32` `LOOP_DOWN_NORM`; lowers `F32`/`BF16` `mul_mv_mma` `rows_min` to `3` so `pp4` MoE routing hits MMA instead of `mul_mv_ext`.
3. **Fused rank-16 relative bias projection in both vector & wide Metal 4 banded FlashAttention + small-batch wave concurrency:** Fuses `MUL_MAT(rel_w, rel_q)` directly into `fa_vec_q8_0_gqa` and `fa_tensor` (fixing FP16 rounding on `ub=1`) and enables lockstep wave concurrency for `n_tokens <= 32`.
4. **Fused DSpark Markov draft pipeline (`-48.6%` dispatches, `-49.7%` barriers):** Factors static `dspark_joint_proj` out of the 7-step draft loop (`55 -> 12.6 MB`), adds dual-output `DSPARK_MARKOV_GRU` (`13 ops -> 1` dispatch) and `DSPARK_MARKOV_LOGITS` (`6 ops -> 1+1` dispatches, 100% simdgroup utilization on `Q4_0` 49K vocab + Gumbel argmax), tunes `N_ARGMAX_CHUNK = 512`, and batches all 7 draft softmaxes into one parallel `MUL_SOFT_MAX`.
5. **In-register 128-dim Q/K/V/`kqv_out` head fusions + SIMD `q8_0` `SET_ROWS`:** Keeps 128-dim head vectors in 32-lane simdgroup registers (4 floats/lane) through `L2_CLIP`/`RMS_NORM` -> `FWHT(64|128)` -> coalesced 32-lane `q8_0` quantization and `FWHT + SIGMOID + MUL`, cutting 296 dispatches and barriers per step across all 48 layers.
6. **100% lossless hybrid O(1) adaptive prompt-lookup + probabilistic DSpark:** Replaces `ngram-simple`'s `O(N)` scan with a per-slot 65536x4-way incremental hash index (~30 ns lookup at 400K context), aligns draft lengths to `{15 -> 47}` Metal 4 MMA tile sizes (`N=16` and `N=48`) with miss backoff, adds Apple `vDSP`/P-core fast lossless softmax verification (including exact grammar-masked `p_V`), sizes recurrent rollback snapshots and DSpark anchor history across combined draft lengths, and syncs `spec_prompt` incrementally in `O(delta)`.
7. **Inkling 2 tokenizer fix:** Enables `ignore_merges = true` for `tokenizer_pre == "inkling2"` so whole-word tokens like `kwargs` stay intact.

## Verification

- `test-backend-ops -b MTL0`: `MUL_MAT`, `MUL_MAT_ID`, `FLASH_ATTN_EXT_BANDED`, `GET_ROWS`, `ARGMAX`, `SOFT_MAX`, `SSM_CONV` all pass.
- Pipeline occupancy audit: `68/68` unique Metal pipelines hit `th_max = 1024` (zero register spilling).
- `test-speculative-rejection-sampling`: all distribution-preservation suites pass (`5,000` trials); `test-sampling`, `test-grammar-parser`, `test-grammar-integration`, and `test-chat` all pass.
- `llama-perplexity --kl-divergence` vs `b171b04c` (`ub=512`): `Mean KLD = 0.000669 ± 0.000039`, `Mean Δp = -0.002 ± 0.010%`, `Same top p = 98.83%`.
