#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/Inkling-2-Mini.gguf [build_dir]" >&2
  exit 1
fi

MODEL="$1"
BUILD_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)/build}"
BENCH="$BUILD_DIR/bin/llama-bench"

if [[ ! -x "$BENCH" ]]; then
  echo "llama-bench not found at $BENCH. Build first with:" >&2
  echo "  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON && ninja -C build llama-bench" >&2
  exit 1
fi

echo "=== Short-to-medium prefill & raw decode (-ub 1024) ==="
"$BENCH" -m "$MODEL" -fa 1 -ctk q8_0 -ctv q8_0 -t 4 -b 2048 -ub 1024 \
  -p 24,256,512,1024 -n 128 -r 3

echo "=== Peak wide prefill (-ub 2048) ==="
"$BENCH" -m "$MODEL" -fa 1 -ctk q8_0 -ctv q8_0 -t 4 -b 2048 -ub 2048 \
  -p 2048,4096 -n 0 -r 3
