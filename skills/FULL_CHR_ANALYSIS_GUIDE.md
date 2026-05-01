# Full-Chromosome Embedding Analysis Guide

This guide covers two analyses for full-chromosome embeddings:

1. **Clustering** — do embeddings from different superpopulations separate in PCA/UMAP space?
2. **Complexity probe** — do the mixed5 "complexity channels" carry the same population signal at a later layer?

It assumes you have full-chromosome embeddings in the same layout as the chr20 slice:
```
data/embeddings/<sample>_<layer>/activation_cache/activations_00000000.npz
```
with one batched NPZ per sample, shape `(n_sites, H, W, channels)`.

---

## Before You Start — Check Your Data

```python
import numpy as np
from pathlib import Path

layer  = "mixed10"          # replace with your layer
sample = "NA12878"

d = np.load(
    f"data/embeddings/{sample}_{layer}/activation_cache/activations_00000000.npz",
    allow_pickle=True,
)
print(d.files)           # e.g. ['mixed10']
print(d[layer].shape)    # (n_sites, H, W, channels)
```

Note the number of **channels** (last dimension). This matters for Part 2:
- `mixed5`–`mixed7`: 768 channels
- `mixed8`: 1280 channels
- `mixed9`–`mixed10`: 2048 channels

---

## Part 1 — Clustering: Do Samples Separate by Superpopulation?

**Goal:** Visualise all 13 samples in a shared embedding space. If the model encodes
population structure, samples should cluster by superpopulation even without any
supervision.

### Step 1a — Load and pool all embeddings

```python
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

LAYER      = "mixed10"       # your layer
EMBED_BASE = Path("data/embeddings")

SAMPLES = {
    "NA12878": ("EUR", "CEU"), "HG00731": ("EUR", "GBR"), "NA20502": ("EUR", "TSI"),
    "HG01985": ("AFR", "LWK"), "HG02922": ("AFR", "LWK"),
    "HG01048": ("AMR", "CLM"), "HG01197": ("AMR", "CLM"), "HG01565": ("AMR", "PEL"),
    "HG00759": ("EAS", "CHS"), "NA18939": ("EAS", "JPT"), "HG00864": ("EAS", "CDX"),
    "HG03009": ("SAS", "GIH"), "NA20847": ("SAS", "GIH"),
}

def load_embeddings(sample_id, layer):
    path = EMBED_BASE / f"{sample_id}_{layer}" / "activation_cache" / "activations_00000000.npz"
    d    = np.load(path, allow_pickle=True)
    key  = next((k for k in d.files if layer in k.lower()), d.files[0])
    act  = d[key]                   # (n_sites, H, W, channels)
    return act.mean(axis=(1, 2))    # → (n_sites, channels)

# Pool all sites across all samples
X_list, spop_list, sample_list = [], [], []
for sid, (spop, pop) in SAMPLES.items():
    X = load_embeddings(sid, LAYER)
    X_list.append(X)
    spop_list.extend([spop] * len(X))
    sample_list.extend([sid] * len(X))
    print(f"  {sid:12s}  {spop}  n={len(X)}")

X_all    = np.vstack(X_list)              # (total_sites, channels)
spops    = np.array(spop_list)
samples  = np.array(sample_list)

# Standardise using NA12878 as reference (keeps the coordinate system consistent)
na_idx = np.where(samples == "NA12878")[0]
scaler = StandardScaler().fit(X_all[na_idx])
X_sc   = scaler.transform(X_all)
```

### Step 1b — PCA

PCA is fast and interpretable. Run it first.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

COLORS = {"AFR":"#E24B4A","AMR":"#EF9F27","EAS":"#1D9E75","EUR":"#378ADD","SAS":"#7F77DD"}

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_sc)

fig, ax = plt.subplots(figsize=(8, 6))
for sp in ["AFR","AMR","EAS","EUR","SAS"]:
    mask = spops == sp
    ax.scatter(pcs[mask, 0], pcs[mask, 1], s=5, alpha=0.3,
               color=COLORS[sp], label=sp, rasterized=True)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"PCA of {LAYER} embeddings — all 13 samples")
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout()
plt.savefig(f"results/pca_{LAYER}_superpop.png", dpi=150)
plt.close()
```

### Step 1c — UMAP

UMAP reveals non-linear cluster structure that PCA misses. Run after PCA since it is slower.

```python
import umap

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_sc)     # may take a few minutes on full chr

fig, ax = plt.subplots(figsize=(8, 6))
for sp in ["AFR","AMR","EAS","EUR","SAS"]:
    mask = spops == sp
    ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.3,
               color=COLORS[sp], label=sp, rasterized=True)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"UMAP of {LAYER} embeddings — all 13 samples")
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout()
plt.savefig(f"results/umap_{LAYER}_superpop.png", dpi=150)
plt.close()
```

**Tip:** If UMAP is too slow on the full chromosome, subsample uniformly first:

```python
rng  = np.random.default_rng(42)
idx  = rng.choice(len(X_sc), size=50_000, replace=False)
embedding = reducer.fit_transform(X_sc[idx])
spops_sub = spops[idx]
```

### Step 1d — Silhouette score (optional but useful)

This quantifies how well samples cluster by superpopulation on a −1 to 1 scale.
A score above ~0.1 indicates meaningful separation; above 0.3 is strong.

```python
from sklearn.metrics import silhouette_score

# Use PCA-reduced space (faster than full-dimensional)
sil = silhouette_score(pcs, spops, sample_size=10_000, random_state=42)
print(f"Silhouette score (superpopulation): {sil:.4f}")
```

---

## Part 2 — Complexity Probe: Does the Mixed5 Signal Transfer?

**Goal:** The mixed5 complexity probe separates GIAB high-confidence (TP) sites from
uncertain (UNK) sites with AUROC 0.994. Apply that same probe to a later layer and
see if (a) the TP/UNK axis still holds, and (b) whether non-EUR samples still score
lower on it.

### Which script to use

**`scripts/population_complexity.py`** is the main script. It runs all six phases
end-to-end: disk audit → probe fit → project all samples → stats → plots → report.

The script is written for mixed5 and needs **three edits** to run on a new layer.
Open the script and change:

```python
# Line 41 — embedding path pattern (change 'mixed5' to your layer)
npz_path = EMBED_BASE / f"{sample_id}_mixed5" / ...
# → change to your layer, e.g. f"{sample_id}_mixed10"

# Line 52 — same path in the audit loop
npz_path = EMBED_BASE / f"{sid}_mixed5" / ...

# Line 19 — output directory (keep results separate per layer)
OUT = BASE / "results/population_complexity"
# → change to e.g. BASE / "results/population_complexity_mixed10"
```

Then run:

```bash
conda run -n deepVariant python3 scripts/population_complexity.py
```

This produces everything in the output directory: scores per sample, KDE/violin/strip/effect-size plots, Kruskal-Wallis + pairwise stats, and a `FINDINGS.md` report.

### If your layer has a different number of channels

The probe was fit on 768-channel mixed5 embeddings. If your layer has 1280 or 2048
channels, the probe cannot be transferred directly — it must be refit on the new
layer using NA12878's embeddings and the same GIAB labels.

**`run_linear_probe.py`** does exactly this: it loads NA12878 embeddings, assigns
GIAB TP/UNK labels, runs 5-fold cross-validation, and saves probe weights.

Edit the path at lines 35–39:
```python
BULK_NPZ = Path("data/embeddings/NA12878_mixed5/activation_cache/activations_00000000.npz")
# → change 'mixed5' to your layer

TFRECORD = Path("data/embeddings/NA12878_mixed5/intermediate_results_dir/
                 make_examples.tfrecord-00000-of-00001.gz")
# → change 'mixed5' to your layer (you need the tfrecord to recover site labels)
```

Then run:

```bash
conda run -n deepVariant python3 run_linear_probe.py
```

Outputs go to `results/giab_validation/linear_probe/`:
- `probe_results.json` — AUROC (5-fold CV + top-20 channel probe + permutation test)
- `probe_weights.npy` — the weight vector, shape `(channels,)`
- `cohens_d_channels.json` — top-20 channels by Cohen's d (TP vs UNK mean difference)

Once you have new probe weights, point `population_complexity.py` at them by
updating `PROBE_DIR` (line 17):
```python
PROBE_DIR = BASE / "results/giab_validation/linear_probe"
# → change to wherever run_linear_probe.py wrote its output
```

---

## Part 3 — Top-20 Channel PCA: Do the Mixed5 Channels Still Carry the Signal?

**Goal:** The top channels in mixed5 are (by both probe weight and Cohen's d):

```
604, 130, 189, 660, 661, 91, 594, 230, 40, 140,
142, 102, 765, 191, 507, 121, 43, 416, 26, 599
```

Channel 604 is the strongest by Cohen's d (d=2.05 between TP and UNK sites in
NA12878). If these same channel indices carry the signal in a later 768-channel
layer, it suggests the model has a stable internal structure across depth.
If the channels are completely different, the probe has reorganised.

**This only applies to layers with 768 channels (mixed5–mixed7).** For mixed8+
(1280 or 2048 channels), identify the new layer's top channels by running
`run_linear_probe.py` on that layer and reading `cohens_d_channels.json`.

### Running the top-channel PCA

Use the script from `skills/LAYER_ANALYSIS_GUIDE.md`, which is already set up for this:

```bash
conda run -n deepVariant python3 scripts/top_channel_pca.py \
    --layer mixed7 \
    --results_dir results/population_complexity_mixed7
```

The `--results_dir` must contain `probe_weights.npy`, `scaler_mean.npy`, and
`scaler_scale.npy` (written by `population_complexity.py`).

This produces three plots:
- `top20_channel_pca_superpop.png` — PCA of just those 20 channels, coloured by superpop
- `top20_channel_pca_population.png` — same at population level (CEU, LWK, CHS, …)
- `top20_channel_pc1_vs_score.png` — PC1 vs complexity score; if PC1 ≈ score the
  probe channels dominate the variance structure

### Comparing across layers

| What to look at | Where to find it |
|---|---|
| Does TP/UNK separation hold? | `probe_results.json` → `full_768d_probe.auroc_mean` |
| Do the same 20 channels matter? | `probe_results.json` → `channel_overlap_cohens_vs_probe`; compare channel lists across layers |
| Is the EAS effect still present? | `stats.json` → `pairwise_vs_eur.EAS.p_bonferroni` and `cohens_d` |
| Do populations cluster visually? | `pca_<layer>_superpop.png`, `umap_<layer>_superpop.png` |
| Is the complexity axis aligned with the main variance axis? | `top20_channel_pc1_vs_score.png` |

---

## Quick-Start Checklist

```
[ ] Confirm embeddings exist: ls data/embeddings/ | grep <layer>
[ ] Check shape: python3 -c "import numpy as np; d=np.load(...); print(d[...].shape)"
[ ] Part 1: Run PCA and UMAP (Step 1a–1c above)
[ ] Part 1: Compute silhouette score (Step 1d)
[ ] Part 2a: Edit scripts/population_complexity.py (3 lines), run it
[ ] Part 2b (if different channels): Edit and run run_linear_probe.py first
[ ] Part 3: Run scripts/top_channel_pca.py (768-channel layers only)
[ ] Compare probe_results.json and stats.json across layers
```
