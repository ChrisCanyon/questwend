#!/usr/bin/env python3
"""Diff two QWEN_STEP_LOGITS files step by step, up to where the sampled tokens part.

    python compare_steps.py a.txt b.txt

Each line is "<sampled token> <logit 0> <logit 1> ...". While both runs have
sampled the same tokens their inputs are identical, so their logits should
agree to rounding; the first step whose tokens differ is still compared (same
history), then the comparison stops.
"""
import sys
import numpy as np


def load(path):
    rows = []
    for line in open(path):
        v = line.split()
        rows.append((int(v[0]), np.array(v[1:], dtype=np.float64)))
    return rows


a, b = load(sys.argv[1]), load(sys.argv[2])
worst = 0.0
for i, ((ta, la), (tb, lb)) in enumerate(zip(a, b)):
    d = np.abs(la - lb).max()
    span = la.max() - la.min()
    worst = max(worst, d / span)
    flag = "" if ta == tb else "   <- sampled tokens part here"
    print("step %3d  tok %6d / %6d  max|dlogit| %.3e  (%.2e of the logit range)%s"
          % (i, ta, tb, d, d / span, flag))
    if ta != tb:
        break
print("worst relative difference while the histories matched: %.2e" % worst)
