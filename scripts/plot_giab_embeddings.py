"""
PCA + UMAP of NA12878 mixed5 embeddings coloured by GIAB truth labels.

Adapted from the standard per-site-npz template to handle the actual on-disk
layout where all 337 sites live in a single NPZ (shape: 337 × 4 × 12 × 768)
and site positions are recovered from the accompanying make_examples TFRecord.
"""

import numpy as np
import json
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
ACT_DIR      = BASE / "data/embeddings/NA12878_mixed5/activation_cache"
TFRECORD     = (BASE / "data/embeddings/NA12878_mixed5/intermediate_results_dir"
                     / "make_examples.tfrecord-00000-of-00001.gz")
LABELS_PATH  = BASE / "results/giab_validation/site_labels.json"
OUT_DIR      = BASE / "results/giab_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load site labels ──────────────────────────────────────────────────────────
with open(LABELS_PATH) as f:
    label_data = json.load(f)
site_labels = label_data["site_labels"]   # e.g. {"chr20_10000117": "TP", ...}

# ── Load positions from TFRecord (same order as NPZ batch dim) ────────────────
import tensorflow as tf  # noqa: E402  (imported after matplotlib backend set)

positions = []
ds = tf.data.TFRecordDataset([str(TFRECORD)], compression_type='GZIP')
for rec in ds:
    ex = tf.train.Example()
    ex.ParseFromString(rec.numpy())
    locus = ex.features.feature['locus'].bytes_list.value[0].decode()
    # "20:10000117-10000117"  →  "chr20_10000117"
    chrom, rest = locus.split(':', 1)
    start = rest.split('-')[0]
    positions.append(f"chr{chrom}_{start}")

print(f"TFRecord positions loaded: {len(positions)}")

# ── Load the single NPZ ───────────────────────────────────────────────────────
npz_path = ACT_DIR / "activations_00000000.npz"
d = np.load(npz_path)
# Prefer a key containing 'mixed'; fall back to first key
key = next((k for k in d.files if 'mixed' in k.lower() or 'concat' in k.lower()),
           d.files[0])
act = d[key]   # (337, 4, 12, 768)
print(f"Loaded activations: key='{key}', shape={act.shape}")

assert act.shape[0] == len(positions), (
    f"Mismatch: {act.shape[0]} activations vs {len(positions)} positions")

# Pool spatial dims → (N, C)
embeddings = act.mean(axis=tuple(range(1, act.ndim - 1)))   # → (337, 768)

# Attach labels
labels = np.array([site_labels.get(p, "UNK") for p in positions])

print(f"Embeddings: {embeddings.shape}")
label_counts = {lb: int((labels == lb).sum()) for lb in ["TP", "FP", "FN", "UNK", "."]}
print("Label counts:", label_counts)

# ── Colour scheme ─────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "TP":  "#1D9E75",
    "FP":  "#E24B4A",
    "FN":  "#EF9F27",
    "UNK": "#B4B2A9",
    ".":   "#B4B2A9",
}

def scatter_by_label(ax, coords, labels, title, xlabel, ylabel):
    draw_order = ["UNK", ".", "FN", "FP", "TP"]
    for lb in draw_order:
        mask = labels == lb
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=LABEL_COLORS.get(lb, "#888"),
            label=f"{lb} (n={mask.sum()})",
            alpha=0.75 if lb in ("TP", "FP", "FN") else 0.35,
            s=30  if lb in ("TP", "FP", "FN") else 12,
            linewidths=0,
            zorder=3 if lb in ("TP", "FP", "FN") else 1,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9, markerscale=1.5, framealpha=0.8)

# ── Standardise ───────────────────────────────────────────────────────────────
X_scaled = StandardScaler().fit_transform(embeddings)

# ── PCA ───────────────────────────────────────────────────────────────────────
n_components = min(20, len(embeddings) - 1)
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scatter_by_label(
    axes[0], X_pca, labels,
    "PCA — NA12878 mixed5, coloured by GIAB label",
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
)
scatter_by_label(
    axes[1], X_pca[:, [0, 2]], labels,
    "PCA PC1 vs PC3",
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
)
plt.tight_layout()
pca_png = OUT_DIR / "NA12878_pca_by_giab_label.png"
plt.savefig(pca_png, dpi=150)
plt.close()
print(f"Saved {pca_png}")

# ── UMAP ──────────────────────────────────────────────────────────────────────
umap_png = None
try:
    import umap  # noqa: E402
    print("Running UMAP…")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    X_umap = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_by_label(
        ax, X_umap, labels,
        "UMAP — NA12878 mixed5, coloured by GIAB label\n"
        "(TP=correct calls  UNK=outside high-conf regions)",
        "UMAP 1", "UMAP 2",
    )
    plt.tight_layout()
    umap_png = OUT_DIR / "NA12878_umap_by_giab_label.png"
    plt.savefig(umap_png, dpi=150)
    plt.close()
    print(f"Saved {umap_png}")
except ImportError:
    print("umap-learn not installed — skipping UMAP. "
          "Install with: conda install -n deepVariant -c conda-forge umap-learn")

# ── Centroid distances in PCA space ───────────────────────────────────────────
unique_labels = [lb for lb in ["TP", "FP", "FN"] if (labels == lb).sum() > 0]
centroids = {lb: X_pca[labels == lb].mean(axis=0) for lb in unique_labels}

print("\nCentroid distances in PCA space:")
dist_results = {}
for l1, l2 in itertools.combinations(unique_labels, 2):
    d_val = float(np.linalg.norm(centroids[l1] - centroids[l2]))
    dist_results[f"{l1}-{l2}"] = round(d_val, 4)
    print(f"  {l1} vs {l2}: {d_val:.4f}")

pc1_by_label = {lb: X_pca[labels == lb, 0].tolist() for lb in unique_labels}
pc1_means    = {lb: round(float(np.mean(v)), 4) for lb, v in pc1_by_label.items()}
print(f"PC1 means by label: {pc1_means}")

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    "n_sites": int(len(embeddings)),
    "label_counts": label_counts,
    "pca_variance_explained_pct": [
        round(float(v) * 100, 2) for v in pca.explained_variance_ratio_[:5]
    ],
    "centroid_distances_pca": dist_results,
    "pc1_means_by_label": pc1_means,
    "plots": [
        str(pca_png),
        str(umap_png) if umap_png else "skipped (umap-learn not installed)",
    ],
    "interpretation_note": (
        "If TP and FP centroids are well-separated, mixed5 encodes call confidence "
        "geometrically. If not, confidence information lives in deeper layers."
    ),
}

summary_path = OUT_DIR / "giab_validation_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"\nSummary saved to {summary_path}")
print(json.dumps({k: v for k, v in summary.items() if k != "plots"}, indent=2))
