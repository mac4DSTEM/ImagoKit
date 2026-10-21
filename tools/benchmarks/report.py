#!/usr/bin/env python3
"""Join swift.json and python.json into the table of docs/benchmarks.md (stdout).
  python3 tools/benchmarks/report.py swift.json python.json
"""
import json
import statistics
import sys

sw = json.load(open(sys.argv[1]))
py = json.load(open(sys.argv[2]))["results"]
pyd = {(r["name"], r["dtype"], r["size"]): r["ms"] for r in py}


def fmt(t):
    return f"{statistics.median(t):.3g} ({min(t):.3g}–{max(t):.3g})"


print("| function | dtype | size | kit CPU ms: median (min–max) | Python ms: median (min–max) | kit / Python |")
print("|---|---|---|---|---|---|")
for r in sorted(sw, key=lambda r: (r["name"], r["dtype"], r["size"])):
    p = pyd.get((r["name"], r["dtype"], r["size"]))
    ratio = f"{statistics.median(r['ms']) / statistics.median(p):.2f}" if p else "–"
    print(f"| {r['name']} | {r['dtype']} | {r['size']}² | {fmt(r['ms'])} | {fmt(p) if p else '–'} | {ratio} |")
