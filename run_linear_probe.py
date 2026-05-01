"""
Linear probe: logistic regression on NA12878 mixed5 embeddings (TP vs UNK).

Data layout (actual on disk):
  Bulk NPZ  : data/embeddings/NA12878_mixed5/activation_cache/activations_00000000.npz
              shape (337, 4, 12, 768)  — layer key 'mixed5'
  Site order: data/embeddings/NA12878_mixed5/intermediate_results_dir/
                make_examples.tfrecord-00000-of-00001.gz
              (same row order as NPZ)
  Labels    : results/giab_validation/site_labels.json
              keys like 'chr20_10000117'

Output dir : results/giab_validation/linear_probe/

Reproduces the prescribed script step-for-step; only the I/O shim differs.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path

# ── Env setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Paths (actual data layout) ────────────────────────────────────────────────
BULK_NPZ    = Path("data/embeddings/NA12878_mixed5/activation_cache/activations_00000000.npz")
TFRECORD    = Path("data/embeddings/NA12878_mixed5/intermediate_results_dir/"
                   "make_examples.tfrecord-00000-of-00001.gz")
labels_path = Path("results/giab_validation/site_labels.json")
out_dir     = Path("results/giab_validation/linear_probe")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Step 0: recover ordered site keys from make_examples tfrecord ────────────
print("Recovering site order from make_examples tfrecord...", flush=True)
import tensorflow as tf

def _parse_locus(raw: bytes) -> str:
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    feat = ex.features.feature
    if "locus" in feat:
        locus_bytes = feat["locus"].bytes_list.value[0].decode()
        chrom, rest = locus_bytes.split(":", 1)
        start_str = rest.split("-")[0]
        pos = int(start_str)
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        return f"{chrom}_{pos}"
    return "UNKNOWN"

site_keys_ordered: list[str] = []
dataset = tf.data.TFRecordDataset(str(TFRECORD), compression_type="GZIP")
for raw_record in dataset:
    site_keys_ordered.append(_parse_locus(raw_record.numpy()))

print(f"  {len(site_keys_ordered)} sites in tfrecord", flush=True)

# ── 1. Load embeddings ────────────────────────────────────────────────────────
print("Loading activations...", flush=True)
with open(labels_path) as f:
    label_data = json.load(f)
site_labels = label_data["site_labels"]

d_bulk = np.load(BULK_NPZ)
key = next(
    (k for k in d_bulk.files if "mixed" in k.lower() or "concat" in k.lower()),
    d_bulk.files[0],
)
acts_bulk = d_bulk[key]                         # (337, 4, 12, 768)
print(f"  Bulk activation shape: {acts_bulk.shape}", flush=True)

n = min(len(site_keys_ordered), acts_bulk.shape[0])

embeddings, labels, positions = [], [], []
for i in range(n):
    stem = site_keys_ordered[i]
    if stem == "UNKNOWN":
        continue
    lbl = site_labels.get(stem, "UNK")
    if lbl not in ("TP", "UNK"):
        continue                                 # skip FP/FN if any

    act = acts_bulk[i]                           # (4, 12, 768)
    pooled = act.mean(axis=tuple(range(act.ndim - 1)))  # → (768,)
    embeddings.append(pooled)
    labels.append(1 if lbl == "TP" else 0)       # 1=TP, 0=UNK
    positions.append(stem)

X = np.vstack(embeddings)
y = np.array(labels)
print(f"Loaded: {len(X)} sites  |  TP={y.sum()}  UNK={(y==0).sum()}")

# ── 2. Cohen's d per channel ─────────────────────────────────────────────────
tp_mask  = y == 1
unk_mask = y == 0
tp_mean  = X[tp_mask].mean(axis=0)
unk_mean = X[unk_mask].mean(axis=0)
pooled_std = np.sqrt(
    (X[tp_mask].var(axis=0) + X[unk_mask].var(axis=0)) / 2
)
pooled_std = np.where(pooled_std < 1e-9, 1e-9, pooled_std)
cohens_d = (tp_mean - unk_mean) / pooled_std

top20_idx = np.argsort(np.abs(cohens_d))[::-1][:20]
top20 = [
    {"channel": int(i), "cohens_d": float(cohens_d[i]),
      "tp_mean": float(tp_mean[i]), "unk_mean": float(unk_mean[i])}
    for i in top20_idx
]
print(f"Top channel by |Cohen's d|: ch{top20[0]['channel']}  d={top20[0]['cohens_d']:.3f}")

cohens_d_out = {
    "top20_channels": top20,
    "channel_604_d": float(cohens_d[604]) if 604 < len(cohens_d) else None,
    "channel_734_d": float(cohens_d[734]) if 734 < len(cohens_d) else None,
}
(out_dir / "cohens_d_channels.json").write_text(json.dumps(cohens_d_out, indent=2))

# ── 3. Full 768-d probe, stratified 5-fold CV ─────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auroc_full = cross_val_score(
    clf_full, X_scaled, y, cv=cv, scoring="roc_auc"
)
print(f"Full 768-d  AUROC: {auroc_full.mean():.4f} ± {auroc_full.std():.4f}")

# ── 4. Top-20 channel probe ───────────────────────────────────────────────────
X_top20  = X_scaled[:, top20_idx]
clf_top20 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

auroc_top20 = cross_val_score(
    clf_top20, X_top20, y, cv=cv, scoring="roc_auc"
)
print(f"Top-20 ch   AUROC: {auroc_top20.mean():.4f} ± {auroc_top20.std():.4f}")

# ── 5. Permutation test (100 shuffles) ───────────────────────────────────────
rng = np.random.default_rng(42)
null_aurocs = []
for _ in range(100):
    y_perm = rng.permutation(y)
    perm_scores = cross_val_score(
        LogisticRegression(C=1.0, max_iter=500, random_state=42),
        X_scaled, y_perm, cv=cv, scoring="roc_auc",
    )
    null_aurocs.append(float(perm_scores.mean()))

null_mean = float(np.mean(null_aurocs))
null_std  = float(np.std(null_aurocs))
p_val     = float(np.mean(np.array(null_aurocs) >= auroc_full.mean()))
print(f"Null AUROC: {null_mean:.4f} ± {null_std:.4f}  |  p = {p_val:.4f}")

# ── 6. Fit final probe on all data, extract weights ───────────────────────────
clf_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf_final.fit(X_scaled, y)
weights = clf_final.coef_[0]                    # shape (768,)

top20_probe_idx = np.argsort(np.abs(weights))[::-1][:20]
top20_weights = [
    {"channel": int(i), "weight": float(weights[i])}
    for i in top20_probe_idx
]

# Overlap between Cohen's d top-20 and probe weight top-20
cohens_set = set(top20_idx.tolist())
probe_set  = set(top20_probe_idx.tolist())
overlap    = cohens_set & probe_set
print(f"Channel overlap (Cohen's d top-20 ∩ probe top-20): {len(overlap)}/20")
print(f"Overlapping channels: {sorted(overlap)}")

# ── 7. Project all embeddings onto probe weight vector ────────────────────────
complexity_scores = X_scaled @ weights          # shape (n_sites,)
np.save(out_dir / "complexity_scores.npy", complexity_scores)
np.save(out_dir / "labels.npy", y)
np.save(out_dir / "positions.npy", np.array(positions))
np.save(out_dir / "X_scaled.npy", X_scaled)

# ── 8. Write full results JSON ────────────────────────────────────────────────
results = {
    "n_sites": int(len(X)),
    "n_tp": int(y.sum()),
    "n_unk": int((y == 0).sum()),
    "full_768d_probe": {
        "auroc_mean":  float(auroc_full.mean()),
        "auroc_std":   float(auroc_full.std()),
        "auroc_folds": auroc_full.tolist(),
    },
    "top20_channel_probe": {
        "auroc_mean":  float(auroc_top20.mean()),
        "auroc_std":   float(auroc_top20.std()),
        "auroc_folds": auroc_top20.tolist(),
        "channels":    top20_idx.tolist(),
    },
    "permutation_test": {
        "n_permutations": 100,
        "null_auroc_mean": null_mean,
        "null_auroc_std":  null_std,
        "p_value":         p_val,
        "significant":     p_val < 0.05,
    },
    "top20_probe_weights": top20_weights,
    "channel_overlap_cohens_vs_probe": sorted(list(overlap)),
    "n_overlap": len(overlap),
    "channel_604_weight": float(weights[604]) if 604 < len(weights) else None,
    "channel_734_weight": float(weights[734]) if 734 < len(weights) else None,
}
(out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))

print("\n── Results ──────────────────────────────────────────────────")
print(json.dumps({k: v for k, v in results.items()
                  if k not in ("top20_probe_weights",)}, indent=2))
print(f"\nAll outputs written to {out_dir}/")
