#!/usr/bin/env python3
"""
Plot `result2.csv` as line plots: avg_seconds and avg_speedup vs size for each algorithm.
Saves `build/result2_lines.png`.
"""
import os
import csv
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(__file__), 'result2.csv')
OUT_PNG = os.path.join(os.path.dirname(__file__), 'result2_lines.png')

if not os.path.exists(CSV_PATH):
    print('CSV not found:', CSV_PATH)
    sys.exit(1)

# parse into alg -> size -> value
seconds = defaultdict(dict)
speedup = defaultdict(dict)
sizes_set = set()

with open(CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or row[0].strip().lower() == 'algorithm':
            continue
        try:
            alg = row[0].strip()
            size = int(row[1])
            sec = float(row[2])
            sp = float(row[3])
        except Exception:
            continue
        seconds[alg][size] = sec
        speedup[alg][size] = sp
        sizes_set.add(size)

if not sizes_set:
    print('No data in result2.csv')
    sys.exit(1)

sizes = sorted(sizes_set)
algs = sorted(seconds.keys())

fig, axes = plt.subplots(1,2, figsize=(12,6))
# seconds plot
ax = axes[0]
for alg in algs:
    y = [seconds[alg].get(s, np.nan) for s in sizes]
    ax.plot(sizes, y, marker='o', label=alg)
ax.set_xscale('log')
ax.set_xlabel('size')
ax.set_ylabel('avg_seconds (s)')
ax.set_title('Average seconds vs size')
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend()

# speedup plot
ax = axes[1]
for alg in algs:
    y = [speedup[alg].get(s, np.nan) for s in sizes]
    ax.plot(sizes, y, marker='o', label=alg)
ax.set_xscale('log')
ax.set_xlabel('size')
ax.set_ylabel('avg_speedup')
ax.set_title('Average speedup vs size')
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print('Saved', OUT_PNG)
