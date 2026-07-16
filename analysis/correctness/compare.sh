#!/usr/bin/env bash
set -euo pipefail

dump_dir=$1

cmp "$dump_dir/plain/rho_1539.dat" "$dump_dir/temporal/rho_1539.dat"
cmp "$dump_dir/plain/vx_1539.dat" "$dump_dir/temporal/vx_1539.dat"
cmp "$dump_dir/plain/vy_1539.dat" "$dump_dir/temporal/vy_1539.dat"
cmp "$dump_dir/plain/vz_1539.dat" "$dump_dir/temporal/vz_1539.dat"
