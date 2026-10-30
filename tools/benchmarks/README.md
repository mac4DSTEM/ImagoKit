# Benchmarks

The kit and the Python packages on the same functions, dtypes and sizes; the results table is `docs/benchmarks.md`.

```sh
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
IMAGOKIT_BENCH=1 IMAGOKIT_BENCH_OUT=/tmp/swift.json swift test -c release --filter ImagoKitParityTests.benchmarks
.venv/bin/python -I -B tools/benchmarks/bench.py /tmp/python.json
python3 tools/benchmarks/report.py /tmp/swift.json /tmp/python.json
```

- The Swift side is `Tests/ImagoKitParityTests/Benchmarks/Benchmarks.swift`; its one test returns at once unless
  `IMAGOKIT_BENCH=1`. `IMAGOKIT_BENCH_LARGE=1` adds 2048², 4096² and 8192²; `IMAGOKIT_BENCH_ONLY=rfft2,fft2` runs only the
  cases whose name starts with one of the prefixes.
- `bench.py` is the Python side (`--large` adds the large sizes); `report.py` joins the two JSON files into the Markdown table.
- Each case: one warm-up call, then five timed calls, in milliseconds. Run on a quiet machine, one benchmark at a time.
