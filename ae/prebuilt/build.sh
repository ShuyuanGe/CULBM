#!/usr/bin/env bash
set -euo pipefail
artifact_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
build_root=$(realpath -m -- "$1")
cmake -S "$artifact_root/code" -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Release -DCULBM_STANDALONE=ON \
  '-DCMAKE_CUDA_ARCHITECTURES=80-real;86-real;87-real;89-real;90-real;100-real;101-real;103-real;120-real;121-real'
cmake --build "$build_root" --parallel 2 --target culbm-single culbm-tune
