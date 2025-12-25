#!/usr/bin/env python3
"""
Generate one PNG per `size` from `result.csv`. Each PNG contains two subplots:
- left: boxplot of `seconds` (log scale) grouped by algorithm
- right: boxplot of `speedup` grouped by algorithm

Saves files as `build/result_size_<size>.png`.
"""
import os
import csv
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(__file__), 'result.csv')
OUT_DIR = os.path.dirname(__file__)

if not os.path.exists(CSV_PATH):
    print('CSV not found:', CSV_PATH)
    sys.exit(1)

# collect rows per size
data_by_size = defaultdict(lambda: {'seconds': defaultdict(list), 'speedup': defaultdict(list)})

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
        data_by_size[size]['seconds'][alg].append(sec)
        data_by_size[size]['speedup'][alg].append(sp)

if not data_by_size:
    print('No data found in CSV.')
    sys.exit(1)

for size in sorted(data_by_size.keys()):
    seconds_dict = data_by_size[size]['seconds']
    speedup_dict = data_by_size[size]['speedup']
    algs = sorted(seconds_dict.keys(), key=lambda a: np.median(seconds_dict[a]))
    sec_data = [seconds_dict[a] for a in algs]
    sp_data = [speedup_dict[a] for a in algs]

    fig, axes = plt.subplots(1,2, figsize=(12,6))
    ax = axes[0]
    ax.boxplot(sec_data, labels=algs, patch_artist=True)
    ax.set_title('Seconds by Algorithm')
    ax.set_ylabel('Seconds (s)')
    ax.set_yscale('log')
    # readable log ticks
    all_seconds = [v for vals in sec_data for v in vals]
    if all_seconds:
        smin = max(min(all_seconds), 1e-9)
        smax = max(all_seconds)
        yticks = np.logspace(np.log10(smin), np.log10(smax), num=6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:.6f}" if t < 1 else f"{t:.3f}" for t in yticks])
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)

    ax = axes[1]
    ax.boxplot(sp_data, labels=algs, patch_artist=True)
    ax.set_title('Speedup by Algorithm')
    ax.set_ylabel('Speedup')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.suptitle(f'size = {size}')
    plt.subplots_adjust(top=0.88)
    out = os.path.join(OUT_DIR, f'result_size_{size}.png')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print('Saved', out)
