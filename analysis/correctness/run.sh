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
    --device 0
    --domain '[768,384,384]'
    --inv-tau 0.2857142857142857
    --batch-steps 1539
    --steps 1539
    --init "$case_dir"
    --dump-rho --dump-vx --dump-vy --dump-vz
)

"$solver" "${common[@]}" \
    --output "$dump_dir/plain" \
    --block '[32,4,2]' \
    --grid '[24,96,192]' \
    --stream-policy pull \
    --opt-policy plain

"$solver" "${common[@]}" \
    --output "$dump_dir/temporal" \
    --block '[96,1,1]' \
    --grid '[1,94,86]' \
    --inner-steps 9 \
    --stream-policy esotwist \
    --opt-policy blocking \
    --ddf-buffer ring
