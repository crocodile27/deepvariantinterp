#!/usr/bin/env python3
"""PCA + UMAP descriptive analysis on ancestry embeddings."""

import json
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap

BASE = str(Path(__file__).parent)

# ── Step 1: Read status files ──
status_files = sorted(glob.glob(os.path.join(BASE, "data/metadata/*_embedding_status.json")))
samples = []
for sf in status_files:
    with open(sf) as f:
        meta = json.load(f)
    if meta["status"] == "success":
        samples.append(meta)
    else:
        print(f"WARNING: {meta['sample']} status={meta['status']}, skipping")

superpops = set(s["population"] for s in samples)
print(f"Successful samples: {len(samples)}, superpopulations: {sorted(superpops)}")
if len(samples) < 5 or len(superpops) < 3:
    print("WARNING: fewer than 5 samples across 3+ superpopulations!")

# ── Step 2: Load embeddings ──
all_embeddings = []
labels = []
sample_ids = []

for s in samples:
    path = os.path.join(BASE, s["embedding_path"])
    fmt = s["embedding_format"]
    print(f"  Loading {s['sample']} ({s['population']}) from {os.path.basename(path)} ...")

    if fmt == ".npz":
        data = np.load(path)
        # Use first key or 'mixed5'
        key = "mixed5" if "mixed5" in data else list(data.keys())[0]
        arr = data[key]  # shape: (n_sites, 4, 12, 768)
    elif fmt == ".npy":
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Global average pooling over spatial dims (4, 12) -> (n_sites, 768)
    if arr.ndim == 4:
        arr = arr.mean(axis=(1, 2))  # (n_sites, 768)

    n_sites = arr.shape[0]
    all_embeddings.append(arr)
    labels.extend([s["population"]] * n_sites)
    sample_ids.extend([s["sample"]] * n_sites)
    print(f"    -> {n_sites} sites, embedding shape after pooling: {arr.shape}")

X = np.vstack(all_embeddings).astype(np.float32)
labels = np.array(labels)
sample_ids = np.array(sample_ids)
print(f"\nCombined X shape: {X.shape}")
print(f"Labels: {len(labels)}, Sample IDs: {len(sample_ids)}")

# Save combined
np.savez(os.path.join(BASE, "data/metadata/combined_embeddings.npz"),
         X=X, labels=labels, sample_ids=sample_ids)
print("Saved combined_embeddings.npz")

# ── Step 3: Standardize ──
X_scaled = StandardScaler().fit_transform(X)

# ── Step 4: PCA ──
os.makedirs(os.path.join(BASE, "results/pca"), exist_ok=True)
n_components = min(50, len(samples) - 1, X.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Variance explained
var_exp = pca.explained_variance_ratio_
print(f"\nPCA variance explained (top 5):")
for i in range(min(5, len(var_exp))):
    print(f"  PC{i+1}: {var_exp[i]:.4f} ({var_exp[i]*100:.2f}%)")
print(f"  Cumulative (top 5): {sum(var_exp[:5]):.4f} ({sum(var_exp[:5])*100:.2f}%)")

# Color map for superpopulations
unique_superpops = sorted(set(labels))
cmap = plt.cm.tab10
colors = {sp: cmap(i) for i, sp in enumerate(unique_superpops)}
c_array = [colors[l] for l in labels]

# Scree plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, len(var_exp)+1), var_exp, alpha=0.7, label='Individual')
ax.plot(range(1, len(var_exp)+1), np.cumsum(var_exp), 'ro-', markersize=4, label='Cumulative')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained')
ax.set_title('PCA Scree Plot')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(BASE, "results/pca/scree.png"), dpi=150)
plt.close(fig)
print("Saved results/pca/scree.png")

# PC1 vs PC2
fig, ax = plt.subplots(figsize=(10, 8))
for sp in unique_superpops:
    mask = labels == sp
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[sp]], label=sp, alpha=0.5, s=10)
ax.set_xlabel(f'PC1 ({var_exp[0]*100:.2f}%)')
ax.set_ylabel(f'PC2 ({var_exp[1]*100:.2f}%)')
ax.set_title('PCA: PC1 vs PC2 by Superpopulation')
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "results/pca/pc1_pc2.png"), dpi=150)
plt.close(fig)
print("Saved results/pca/pc1_pc2.png")

# PC1 vs PC3
fig, ax = plt.subplots(figsize=(10, 8))
for sp in unique_superpops:
    mask = labels == sp
    ax.scatter(X_pca[mask, 0], X_pca[mask, 2], c=[colors[sp]], label=sp, alpha=0.5, s=10)
ax.set_xlabel(f'PC1 ({var_exp[0]*100:.2f}%)')
ax.set_ylabel(f'PC3 ({var_exp[2]*100:.2f}%)')
ax.set_title('PCA: PC1 vs PC3 by Superpopulation')
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "results/pca/pc1_pc3.png"), dpi=150)
plt.close(fig)
print("Saved results/pca/pc1_pc3.png")

# ── Step 5: UMAP ──
os.makedirs(os.path.join(BASE, "results/umap"), exist_ok=True)
n_neighbors = min(15, len(X) // 2)
print(f"\nRunning UMAP (n_neighbors={n_neighbors}, metric=cosine)...")
reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                    min_dist=0.1, random_state=42, metric='cosine')
X_umap = reducer.fit_transform(X_scaled)
print("UMAP complete.")

# UMAP colored by superpopulation
fig, ax = plt.subplots(figsize=(10, 8))
for sp in unique_superpops:
    mask = labels == sp
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[colors[sp]], label=sp, alpha=0.5, s=10)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP by Superpopulation')
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "results/umap/umap_superpop.png"), dpi=150)
plt.close(fig)
print("Saved results/umap/umap_superpop.png")

# UMAP colored by sample_id
unique_samples = sorted(set(sample_ids))
sample_cmap = plt.cm.tab20
sample_colors = {s: sample_cmap(i % 20) for i, s in enumerate(unique_samples)}

fig, ax = plt.subplots(figsize=(12, 8))
for s in unique_samples:
    mask = sample_ids == s
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[sample_colors[s]], label=s, alpha=0.5, s=10)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP by Sample')
ax.legend(markerscale=3, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "results/umap/umap_sample.png"), dpi=150)
plt.close(fig)
print("Saved results/umap/umap_sample.png")

# ── Step 6: Clustering metrics ──
print("\n=== Clustering Metrics ===")

# Pairwise centroid distances
centroids = {}
for sp in unique_superpops:
    mask = labels == sp
    centroids[sp] = X_umap[mask].mean(axis=0)

print("\nPairwise centroid distances (UMAP space):")
header = "          " + "  ".join(f"{sp:>8s}" for sp in unique_superpops)
print(header)
for sp1 in unique_superpops:
    row = f"{sp1:>8s}  "
    for sp2 in unique_superpops:
        dist = np.linalg.norm(centroids[sp1] - centroids[sp2])
        row += f"{dist:8.3f}  "
    print(row)

# Silhouette score
sil_score = silhouette_score(X_umap, labels)
clustering_observed = sil_score > 0.3
print(f"\nSilhouette score: {sil_score:.4f}")
print(f"Clustering observed (score > 0.3): {clustering_observed}")

# ── Step 8: Write summary JSON ──
summary = {
    "n_samples": len(samples),
    "superpops_present": sorted(list(superpops)),
    "n_sites_total": int(X.shape[0]),
    "embedding_dim": int(X.shape[1]),
    "pca_variance_explained_top5": [round(float(v), 6) for v in var_exp[:5]],
    "umap_silhouette_score": round(float(sil_score), 4),
    "clustering_observed": clustering_observed
}

with open(os.path.join(BASE, "results/pca/analysis_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved results/pca/analysis_summary.json")

# ── Summary table ──
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print(f"  Samples:              {summary['n_samples']}")
print(f"  Superpopulations:     {', '.join(summary['superpops_present'])}")
print(f"  Total sites:          {summary['n_sites_total']}")
print(f"  Embedding dim:        {summary['embedding_dim']}")
print(f"  PCA var explained:")
for i, v in enumerate(summary['pca_variance_explained_top5']):
    print(f"    PC{i+1}: {v*100:.2f}%")
cumul = sum(summary['pca_variance_explained_top5'])
print(f"    Cumulative: {cumul*100:.2f}%")
print(f"  UMAP silhouette:      {summary['umap_silhouette_score']:.4f}")
print(f"  Clustering observed:  {summary['clustering_observed']}")
print("="*60)
