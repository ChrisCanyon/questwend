#!/usr/bin/env python3
"""Compare qw-cli --dump-logits ("i: value" lines) with llama-debug --save-logits (raw f32).

    python cmp_logits.py data/llamacpp-<model>.bin <qw-dump.txt>
"""
import sys
import numpy as np

ref = np.fromfile(sys.argv[1], dtype=np.float32)
qw = np.zeros_like(ref)
for line in open(sys.argv[2]):
    i, v = line.split(":")
    qw[int(i)] = float(v)


def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


lr, lq = log_softmax(ref), log_softmax(qw)
print("max |dlogit| %.4f   corr %.6f" % (np.abs(ref - qw).max(), np.corrcoef(ref, qw)[0, 1]))
print("KL(ref||qw)  %.6f" % float((np.exp(lr) * (lr - lq)).sum()))
print("top5 ref", np.argsort(-ref)[:5].tolist(), " qw", np.argsort(-qw)[:5].tolist())
