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

Hardware: Apple M5 Max (40 GPU cores, 128 GB unified memory, Metal 4 tensor ops, `iogpu.wired_limit_mb=114688`), `Inkling-2-Mini-Native8-Full-Q8_0.gguf` (`16.76 GiB`) / `Inkling-2-Mini-Unbound-v2-Native8-Full-Q8_0.gguf` (`16.37 GiB`), `q8_0` KV cache, FlashAttention on.

### Raw `llama-bench` (no speculative decoding, interleaved `91991c93` vs Sprint 12)

| Test | `-ub` | Prior `inkling-opt` (`91991c93`) | **Current `inkling-opt` (Sprint 12)** | Gain vs `91991c93` |
| --- | ---: | ---: | ---: | ---: |
| `tg128` | `512` | `66.84 tok/s` | **`67.92 tok/s`** | **`+1.6%`** |
| `pp2` | `512` | `103.82 tok/s` | **`107.83 tok/s`** | **`+3.9%`** |
| `pp3` | `512` | `141.37 tok/s` | **`149.35 tok/s`** | **`+5.7%`** |
| `pp4` | `512` | `169.96 tok/s` | **`182.94 tok/s`** | **`+7.6%`** |
| `pp6` | `512` | `238.22 tok/s` | **`248.09 tok/s`** | **`+4.2%`** |
| `pp7` (verify) | `512` | `254.63 tok/s` | **`263.90 tok/s`** | **`+3.6%`** |
| `pp8` (verify) | `512` | `243.83 tok/s` | **`271.24 tok/s`** | **`+11.2%`** |
| `pp9` (verify) | `512` | `198.06 tok/s` | **`256.12 tok/s`** | **`+29.3%`** |
| `pp10` (verify) | `512` | `167.51 tok/s` | **`238.99 tok/s`** | **`+42.7%`** |
| `pp512` | `512` | `2,846.1 tok/s` | **`2,860.2 tok/s`** | **`+0.5%`** |
| `pp512 @ d65536` | `512` | `1,362.7 tok/s` | **`1,445.3 tok/s`** | **`+6.1%`** |

### Live `llama-server` (100% lossless at `temp=1.0`, `--spec-draft-n-max 7` + `ngram-simple`)

| Configuration | Aggregate (`prose + code + edit`) | Accept rate | Delta vs `91991c93` |
| --- | ---: | ---: | ---: |
| Prior `91991c93` + Stock + `DSpark-Q8_0` | `78.32 tok/s` | `36.3%` (`420/1158`) | baseline |
| **Sprint 12 + Stock + `DSpark-Q8_0`** | **`86.88 tok/s`** | `37.7%` (`422/1118`) | **`+10.9%`** |
| **Sprint 12 + Unbound v2 + `DSpark-Q8_0`** | **`87.42 tok/s`** | `36.5%` (`423/1159`) | **`+11.6%`** |
| Sprint 12 + Stock + `DSpark-Q4_K_M` (rejected) | `85.08 tok/s` | `37.4%` (`446/1194`) | `-2.1%` vs `Q8_0` draft |
| Sprint 12 + Stock + `DSpark-Q4_0` (rejected) | `82.37 tok/s` | `34.2%` (`412/1203`) | `-5.2%` vs `Q8_0` draft |

## What's in the optimization stack

1. **Small-batch fused MoE `gate_up + SITU` (`T=1..23`, NVFP4 & Q8_0):** Pairs `gate` row `r` with `up` row `r + n_ff` in the same 16-lane sub-warp (`NVFP4`) or 32-lane simdgroup (`Q8_0`) for both `mul_mv_id` (`T=1..3`) and grouped expert-deduplicating `mul_mv_id_grouped` (`T=4..23`), applying `SITU` in registers and writing `situ_out` directly (`th_max=1024`, zero register spills, eliminating `kernel_situ_f32_4` dispatches, barriers, and `dst_gu_op` DRAM traffic during decode and DSpark verification).
2. **Optimized `DEPTH_MIX` (`192`-thread exact tiling + multi-query prefill coalescing + `BF16`/`F16` queries):** Uses `192` threads (`6` simdgroups, `576/192 = 3.0` exact iterations, `100%` lane utilization) for `D=2304`, tiles up to `4` queries per threadgroup during prefill (`n_tokens >= 32`) so `cur` and `blocks[]` are read once per `3–4` queries, and accepts `F16`/`BF16` query tensors.
3. **`inkling2_loop` GGUF compatibility shim + upstream parser/grammar fix (`#32833`):** Loads upstream `inkling2_loop` GGUFs (expanding physical layers to virtual loop visits and mapping `loop_msg_query` / `loop_{q,k,v,ffn}_ctrl_*`) through the full Metal 4 + DSpark pipeline, and backports Inkling research-model tool-call parser/grammar improvements while preserving mid-conversation `system.tools`.
4. **Expert-contiguous direct-device Metal 4 MoE pipeline (`adapt64`) with static `K=64` specialization:** Packs active tokens into contiguous per-expert FP16 slices (`gather_f32_f16`), runs a dense-equivalent 8 KB shmem-`sa` / direct-device `tB` Metal 4 `matmul2d` (`NRA=64`, compile-time static `NK=64` + hoisted A-tile pointers when `K % 64 == 0`, register-spill-free 3-tier adaptive `NRB` `16/32/64`), fuses `SITU` into FP16, and streams directly into vectorized `float4` `BT=32` `SSM_CONV` + expert reduction.
5. **Metal 4 tensor fusions for prefill loop control, barrier-free small-batch `LOOP_DOWN_NORM`, & in-place `FWHT_MUL_SIGMOID`:** Eliminates intermediate loop-control and gating buffers across all 48 layers; runs 1 simdgroup per token (zero shmem, zero barriers) for `T = 2..32` `LOOP_DOWN_NORM`; lowers `F32`/`BF16` `mul_mv_mma` `rows_min` to `3` so `pp4` MoE routing hits MMA instead of `mul_mv_ext`.
6. **Fused rank-16 relative bias projection in both vector & wide Metal 4 banded FlashAttention + small-batch wave concurrency:** Fuses `MUL_MAT(rel_w, rel_q)` directly into `fa_vec_q8_0_gqa` and `fa_tensor` (fixing FP16 rounding on `ub=1`) and enables lockstep wave concurrency for `n_tokens <= 32`.
7. **Fused DSpark Markov draft pipeline (`-48.6%` dispatches, `-49.7%` barriers):** Factors static `dspark_joint_proj` out of the 7-step draft loop (`55 -> 12.6 MB`), adds dual-output `DSPARK_MARKOV_GRU` (`13 ops -> 1` dispatch) and `DSPARK_MARKOV_LOGITS` (`6 ops -> 1+1` dispatches, 100% simdgroup utilization on `Q4_0` 49K vocab + Gumbel argmax), tunes `N_ARGMAX_CHUNK = 512`, and batches all 7 draft softmaxes into one parallel `MUL_SOFT_MAX`.
8. **In-register 128-dim Q/K/V/`kqv_out` head fusions + SIMD `q8_0` `SET_ROWS`:** Keeps 128-dim head vectors in 32-lane simdgroup registers (4 floats/lane) through `L2_CLIP`/`RMS_NORM` -> `FWHT(64|128)` -> coalesced 32-lane `q8_0` quantization and `FWHT + SIGMOID + MUL`, cutting 296 dispatches and barriers per step across all 48 layers.
9. **100% lossless hybrid O(1) adaptive prompt-lookup + probabilistic DSpark:** Replaces `ngram-simple`'s `O(N)` scan with a per-slot 65536x4-way incremental hash index (~30 ns lookup at 400K context), aligns draft lengths to `{15 -> 47}` Metal 4 MMA tile sizes (`N=16` and `N=48`) with miss backoff, adds Apple `vDSP`/P-core fast lossless softmax verification (including exact grammar-masked `p_V`), sizes recurrent rollback snapshots and DSpark anchor history across combined draft lengths, and syncs `spec_prompt` incrementally in `O(delta)`.
10. **Inkling 2 tokenizer fix:** Enables `ignore_merges = true` for `tokenizer_pre == "inkling2"` so whole-word tokens like `kwargs` stay intact.

## Verification

- `test-backend-ops -b MTL0 -o MUL_MAT_ID`: `1192/1192` tests pass.
- Pipeline occupancy audit: `67/67` unique Metal pipelines hit `th_max = 1024` (zero register spilling).
- `test-chat`, `test-sampling`, `test-grammar-parser`, `test-grammar-integration`, and `test-speculative-rejection-sampling` all pass.
- `llama-perplexity --kl-divergence` vs `GGML_METAL_FUSION_DISABLE=1` ground truth (`Inkling-2-Mini-Native8-Full-Q8_0.gguf`):
  - `ub=512`: `Mean KLD = 0.00270 ± 0.00030`, `Same top p = 97.84%` (closer to ground truth than `91991c93`'s `0.00306` / `96.08%`).
  - `ub=8` (verify): `Mean KLD = 0.00219 ± 0.00022`, `Same top p = 99.21%` (vs `91991c93`'s `98.03%`).
  - `ub=1` (decode): `Mean KLD = 0.00119 ± 0.00014`, `Same top p = 100.00%` (vs `91991c93`'s `98.41%`).
