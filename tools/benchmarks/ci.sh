#!/bin/bash
# The audit workflow's nightly CPU benchmark (.github/workflows/audit.yml, the suite's template): writes the
# default-size timings to $1 as a JSON list of {name, dtype, size, ms: [five times]}. The full table with Python beside
# it is README.md here; this run only compares the kit with its own previous night.
set -euo pipefail
out="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
IMAGOKIT_BENCH=1 IMAGOKIT_BENCH_OUT="$out" swift test -c release --filter ImagoKitParityTests.benchmarks
