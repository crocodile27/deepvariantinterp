"""
Cosine distance + channel activation analysis for GIAB TP vs UNK embeddings.

Outputs:
  results/giab_validation/embedding_analysis.json   — distances, variance, top channels
  results/giab_validation/channel_diff_top50.png    — bar chart of top 50 channels by |mean diff|
"""

import numpy as np
import json
import subprocess
import sys
from pathlib import Path
from itertools import combinations

# ── Paths ────────────────────────────────────────────────────────────────────
NPZ_PATH    = "data/embeddings/NA12878_mixed5/activation_cache/activations_00000000.npz"
CVO_PATH    = "data/embeddings/NA12878_mixed5/intermediate_results_dir/call_variants_output-00000-of-00001.tfrecord.gz"
HAPPY_VCF   = "results/giab_validation/happy_results.vcf.gz"
OUT_JSON    = "results/giab_validation/embedding_analysis.json"
OUT_PLOT    = "results/giab_validation/channel_diff_top50.png"

# ── Step 1: parse CVO tfrecord → ordered list of (chrom, pos) ────────────────
print("Parsing CVO tfrecord for position order...", flush=True)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from deepvariant.protos import deepvariant_pb2

cvo_positions = []  # list of (chrom, pos) in batch order
dataset = tf.data.TFRecordDataset(CVO_PATH, compression_type="GZIP")
for raw in dataset:
    cvo = deepvariant_pb2.CallVariantsOutput()
    cvo.ParseFromString(raw.numpy())
    v = cvo.variant
    cvo_positions.append((v.reference_name, v.start))  # 0-based start

print(f"  {len(cvo_positions)} examples in CVO (matches npz dim 0: {np.load(NPZ_PATH)['mixed5'].shape[0]})", flush=True)

# ── Step 2: get BD labels from hap.py VCF (bcftools query) ──────────────────
print("Extracting labels from hap.py VCF...", flush=True)

result = subprocess.run(
    ["bcftools", "query", "-s", "QUERY", "-f", "%CHROM\t%POS\t[%BD]\n", HAPPY_VCF],
    capture_output=True, text=True, check=True
)

priority = {"TP": 3, "FP": 2, "FN": 2, "UNK": 1, ".": 0}
pos_label = {}  # (chrom, 0-based-pos) → label
def norm_chrom(c):
    return c.removeprefix("chr")

for line in result.stdout.strip().splitlines():
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    chrom, pos1, bd = parts[0], int(parts[1]), parts[2]
    key = (norm_chrom(chrom), pos1 - 1)  # hap.py VCF is 1-based; CVO is 0-based
    if priority.get(bd, 0) > priority.get(pos_label.get(key, "."), 0):
        pos_label[key] = bd

print(f"  {len(pos_label)} labeled positions from hap.py VCF", flush=True)

# ── Step 3: assign label to each npz example ────────────────────────────────
labels = []
for chrom, pos in cvo_positions:
    labels.append(pos_label.get((norm_chrom(chrom), pos), "UNK"))
labels = np.array(labels)

counts = {lb: int((labels == lb).sum()) for lb in ["TP", "FP", "FN", "UNK", "."]}
print(f"  Label counts: {counts}", flush=True)

# ── Step 4: load npz, spatial-pool to (N, 768) ──────────────────────────────
print("Loading and pooling activations...", flush=True)
acts = np.load(NPZ_PATH)["mixed5"]          # (N, 4, 12, 768)
X = acts.mean(axis=(1, 2)).astype(np.float32)  # (N, 768)
print(f"  Pooled shape: {X.shape}", flush=True)

tp_mask  = labels == "TP"
unk_mask = labels == "UNK"
X_tp  = X[tp_mask]
X_unk = X[unk_mask]
print(f"  TP: {len(X_tp)}, UNK: {len(X_unk)}", flush=True)

# ── Step 5: cosine distance helpers ─────────────────────────────────────────
def cosine_distances(A, B=None):
    """Mean pairwise cosine distance within A (if B is None) or between A and B."""
    def norm(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(n, 1e-10)
    An = norm(A)
    if B is None:
        # intra: mean of upper-triangle pairwise distances
        sims = An @ An.T                     # (n, n) cosine similarity
        n = len(A)
        idx = np.triu_indices(n, k=1)
        return float(1 - sims[idx].mean())
    else:
        Bn = norm(B)
        sims = An @ Bn.T                     # (n_a, n_b)
        return float(1 - sims.mean())

print("Computing cosine distances (may take a moment for large sets)...", flush=True)
intra_tp  = cosine_distances(X_tp)
intra_unk = cosine_distances(X_unk)
inter     = cosine_distances(X_tp, X_unk)

print(f"  Intra-TP  cosine dist: {intra_tp:.4f}")
print(f"  Intra-UNK cosine dist: {intra_unk:.4f}")
print(f"  Inter TP-UNK cosine dist: {inter:.4f}")

# ── Step 6: intra-cluster variance (in embedding space) ─────────────────────
var_tp  = float(X_tp.var(axis=0).mean())
var_unk = float(X_unk.var(axis=0).mean())
print(f"  Mean per-channel variance  TP: {var_tp:.4f}  UNK: {var_unk:.4f}")

# ── Step 7: per-channel activation differences ───────────────────────────────
mean_tp  = X_tp.mean(axis=0)   # (768,)
mean_unk = X_unk.mean(axis=0)  # (768,)
diff     = mean_tp - mean_unk  # positive → higher in TP
abs_diff = np.abs(diff)

top_k = 50
top_idx = np.argsort(abs_diff)[::-1][:top_k]
top_channels = [
    {"channel": int(i), "tp_mean": float(mean_tp[i]),
     "unk_mean": float(mean_unk[i]), "diff": float(diff[i])}
    for i in top_idx
]

# Cohen's d per channel (effect size)
pooled_std = np.sqrt((X_tp.var(axis=0) + X_unk.var(axis=0)) / 2)
cohens_d   = diff / np.maximum(pooled_std, 1e-8)
top_effect = [
    {"channel": int(i), "cohens_d": float(cohens_d[i]),
     "tp_mean": float(mean_tp[i]), "unk_mean": float(mean_unk[i])}
    for i in np.argsort(np.abs(cohens_d))[::-1][:top_k]
]

print(f"  Top channel by |diff|: ch {top_idx[0]}  diff={diff[top_idx[0]]:.4f}")
print(f"  Top channel by Cohen's d: ch {top_effect[0]['channel']}  d={top_effect[0]['cohens_d']:.3f}")

# ── Step 8: save results ─────────────────────────────────────────────────────
results = {
    "n_tp": int(tp_mask.sum()),
    "n_unk": int(unk_mask.sum()),
    "cosine_distances": {
        "intra_tp":  round(intra_tp,  4),
        "intra_unk": round(intra_unk, 4),
        "inter_tp_unk": round(inter, 4),
        "unk_more_dispersed": intra_unk > intra_tp,
    },
    "mean_per_channel_variance": {
        "tp":  round(var_tp,  4),
        "unk": round(var_unk, 4),
        "unk_higher": var_unk > var_tp,
    },
    "top50_channels_by_abs_diff": top_channels,
    "top50_channels_by_cohens_d": top_effect,
    "interpretation": (
        "intra_unk > intra_tp → UNK embeddings are more dispersed (heterogeneous). "
        "High inter vs intra ratio → mixed5 separates TP from UNK geometrically. "
        "Top channels by Cohen's d are the best candidates for interpretability probing."
    ),
}
Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_JSON).write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {OUT_JSON}", flush=True)

# ── Step 9: channel diff bar chart ──────────────────────────────────────────
print("Plotting top-50 channel differences...", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Top 50 by absolute mean diff
ax = axes[0]
chans = [c["channel"] for c in top_channels]
diffs = [c["diff"] for c in top_channels]
colors = ["#1D9E75" if d > 0 else "#E24B4A" for d in diffs]
ax.bar(range(top_k), diffs, color=colors)
ax.set_xticks(range(top_k))
ax.set_xticklabels([str(c) for c in chans], rotation=90, fontsize=7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Top 50 mixed5 channels by |mean(TP) − mean(UNK)|\n"
             "(green = higher in TP, red = higher in UNK)")
ax.set_ylabel("mean activation difference")
ax.set_xlabel("channel index")

# Top 50 by Cohen's d
ax = axes[1]
chans_d = [c["channel"] for c in top_effect]
ds      = [c["cohens_d"] for c in top_effect]
colors_d = ["#1D9E75" if d > 0 else "#E24B4A" for d in ds]
ax.bar(range(top_k), ds, color=colors_d)
ax.set_xticks(range(top_k))
ax.set_xticklabels([str(c) for c in chans_d], rotation=90, fontsize=7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Top 50 mixed5 channels by Cohen's d (TP vs UNK)\n"
             "(controls for within-group variance — better effect size estimate)")
ax.set_ylabel("Cohen's d")
ax.set_xlabel("channel index")

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
plt.close()
print(f"Plot saved to {OUT_PLOT}", flush=True)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print(f"  TP examples:    {int(tp_mask.sum())}")
print(f"  UNK examples:   {int(unk_mask.sum())}")
print(f"  Intra-TP  cosine dist:    {intra_tp:.4f}")
print(f"  Intra-UNK cosine dist:    {intra_unk:.4f}  ({'MORE' if intra_unk > intra_tp else 'LESS'} dispersed than TP)")
print(f"  Inter TP-UNK cosine dist: {inter:.4f}")
print(f"  Var(TP):  {var_tp:.4f}   Var(UNK): {var_unk:.4f}")
print(f"  Top channel |diff|:   ch {top_idx[0]}  Δ={diff[top_idx[0]]:.4f}")
print(f"  Top channel Cohen's d: ch {top_effect[0]['channel']}  d={top_effect[0]['cohens_d']:.3f}")
print("="*60)
