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

| Test | `-ub` | Upstream `b12588` | Prior `inkling-opt` (`ba485cd4`) | **Current `inkling-opt`** | Gain vs `ba485cd4` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tg128` | `1024` | `33.77 tok/s` | `67.11 tok/s` | **`68.34 tok/s`** | `+1.8%` (`2.02x` vs `b12588`) |
| `pp24` | `1024` | — | `392.1 tok/s` | **`536.3 tok/s`** | **`+36.8%`** |
| `pp512` | `1024` | `968.7 tok/s` | `2,348 tok/s` | **`2,705 tok/s`** | **`+15.2%`** (`2.79x` vs `b12588`) |
| `pp1024` | `1024` | — | `2,512 tok/s` | **`3,095 tok/s`** | **`+23.2%`** |
| `pp2048` | `2048` | `834.9 tok/s` | `2,386 tok/s` | **`3,212 tok/s`** | **`+34.6%`** (`3.85x` vs `b12588`) |
| `pp4096` | `2048` | — | `2,272 tok/s` | **`3,014 tok/s`** | **`+32.7%`** |

### Live `llama-server` (`6x400K` slots, 100% lossless at `temp=1.0`, interleaved `CTRL -> CAND -> CTRL`)

| Workload | Prior (`ba485cd4`, mean) | **Current (`ngram-simple` + DSpark)** | Delta |
| --- | ---: | ---: | ---: |
| Prose `0K` (`temp=1.0`, lossless) | `76.7 tok/s` | **`81.7 tok/s`** | **`+6.5%`** |
| Greedy JSON `1..120` (`exact120=true`) | `218.2 tok/s` | **`286.5 tok/s`** | **`+31.3%`** |
| Python file rewrite (`temp=1.0`, lossless) | `198.0 tok/s` | **`389.0 tok/s`** | **`+96.5%` (`1.96x`)** |
| Python file rewrite (`temp=0.0`) | `208.0 tok/s` | **`486.8 tok/s`** | **`+134.1%` (`2.34x`)** |
| `195K`-token fresh prefill (`-ub 1024` / `-ub 2048`) | `1,074 tok/s` | **`1,203` / `1,257 tok/s`** | **`+12.0%` / `+17.0%`** |
| Concurrent 3-slot aggregate (`temp=1.0`) | `120.1 tok/s` | **`132.1 tok/s`** | **`+9.9%`** |
| Concurrent 6-slot aggregate (`temp=1.0`) | `191.0 tok/s` | **`199.2 tok/s`** | **`+4.3%`** |
| Host wired RAM (`6x400K` + vision + draft) | `80.8 GiB` | **`81.5 GiB`** | `+0.7 GiB` |

## What's in the optimization stack

1. **Expert-contiguous direct-device Metal 4 MoE pipeline (`adapt64`):** Packs active tokens into contiguous per-expert FP16 slices (`gather_f32_f16`), runs a dense-equivalent 8 KB shmem-`sa` / direct-device `tB` Metal 4 `matmul2d` (`NRA=64, NK=64`, register-spill-free 3-tier adaptive `NRB` `16/32/64`), fuses `SITU` into FP16, and streams directly into vectorized `float4` `BT=32` `SSM_CONV` + expert reduction.
2. **Few-row Metal 4 MMA (`rt=3`, `N=17..24`) & lowered `mul_mat_id_use_mm` (`>= 24`):** Eliminates the small-batch `pp24` cliff (`392 -> 536 tok/s`, `+36.8%`).
3. **In-register 128-dim Q/K/V/`kqv_out` head fusions + SIMD `q8_0` `SET_ROWS`:** Keeps 128-dim head vectors in 32-lane simdgroup registers (4 floats/lane) through `L2_CLIP`/`RMS_NORM` -> `FWHT(64|128)` -> coalesced 32-lane `q8_0` quantization and `FWHT + SIGMOID + MUL`, cutting 296 dispatches and barriers per step across all 48 layers.
4. **Fused rank-16 relative bias projection in wide Metal 4 banded FlashAttention:** Eliminates the intermediate `[E, H, T]` FP32 bias tensor during long-context prefill.
5. **100% lossless hybrid O(1) adaptive prompt-lookup + probabilistic DSpark:** Replaces `ngram-simple`'s `O(N)` scan with a per-slot 65536x4-way incremental hash index (~30 ns lookup at 400K context), aligns draft lengths to `{15 -> 47}` Metal 4 MMA tile sizes (`N=16` and `N=48`) with miss backoff, adds Apple `vDSP`/P-core fast lossless softmax verification (including exact grammar-masked `p_V`), sizes recurrent rollback snapshots and DSpark anchor history across combined draft lengths, and syncs `spec_prompt` incrementally in `O(delta)`.
6. **Inkling 2 tokenizer fix:** Enables `ignore_merges = true` for `tokenizer_pre == "inkling2"` so whole-word tokens like `kwargs` stay intact.

## Verification

- `test-backend-ops -b MTL0`: `MUL_MAT`, `MUL_MAT_ID`, `FLASH_ATTN_EXT_BANDED`, `FLASH_ATTN_EXT`, `SSM_CONV`, `RMS_NORM`, `SOFT_MAX`, `CPY`, `SET_ROWS`.
- `test-speculative-rejection-sampling --n-trials 2000`: all 8 suites pass (unconstrained + grammar-masked, full/f32/pruned-d2t rejection and fast-softmax sample-and-accept).
- `llama-perplexity --kl-divergence` vs unfused reference: `Median KLD = 0.000964`, `Same top p = 98.14%`.
