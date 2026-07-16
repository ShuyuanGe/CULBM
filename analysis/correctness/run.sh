#!/usr/bin/env bash
set -euo pipefail

solver=$1
case_dir=$2
dump_dir=$3

test -x "$solver"
test -f "$case_dir/flag.dat"
test -f "$case_dir/vx.dat"
mkdir -p "$dump_dir"

common=(
    --devId 0
    --domainDim '[768,384,384]'
    --invTau 0.2857142857142857
    --dstep 1539
    --nstep 1539
    --initStateFolder "$case_dir"
    --dumpRho --dumpVx --dumpVy --dumpVz
)

"$solver" "${common[@]}" \
    --dumpFolder "$dump_dir/plain" \
    --blockDim '[32,4,2]' \
    --gridDim '[24,96,192]' \
    --streamPolicy 0 \
    --optPolicy 0

"$solver" "${common[@]}" \
    --dumpFolder "$dump_dir/temporal" \
    --blockDim '[96,1,1]' \
    --gridDim '[1,94,86]' \
    --innerLoop 9 \
    --streamPolicy 1 \
    --optPolicy 3 \
    --useRingDDFBuf
