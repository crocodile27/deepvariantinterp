import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

probe_dir = Path("results/giab_validation/linear_probe")
out_dir   = probe_dir / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

X_scaled   = np.load(probe_dir / "X_scaled.npy")
y          = np.load(probe_dir / "labels.npy")
scores     = np.load(probe_dir / "complexity_scores.npy")

with open(probe_dir / "probe_results.json") as f:
    results = json.load(f)
with open(probe_dir / "cohens_d_channels.json") as f:
    cohens_data = json.load(f)

COLORS = {"TP": "#1D9E75", "UNK": "#B4B2A9"}
label_names = {1: "TP", 0: "UNK"}

# ── Plot 1: ROC curve (fit on all data, display CV AUROC in title) ────────
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(X_scaled, y)
probs = clf.predict_proba(X_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)
cv_auroc = results["full_768d_probe"]["auroc_mean"]
cv_std   = results["full_768d_probe"]["auroc_std"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#378ADD", lw=2,
        label=f"ROC (train AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"Linear probe ROC — mixed5 TP vs UNK\n"
             f"5-fold CV AUROC = {cv_auroc:.3f} ± {cv_std:.3f}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "01_roc_curve.png", dpi=150); plt.close()
print("Saved 01_roc_curve.png")

# ── Plot 2: Complexity score distributions (TP vs UNK) ────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for lbl_int, lbl_name in label_names.items():
    mask = y == lbl_int
    ax.hist(scores[mask], bins=30, alpha=0.65,
            color=COLORS[lbl_name], label=f"{lbl_name} (n={mask.sum()})",
            density=True, edgecolor="none")
ax.set_xlabel("Complexity score (probe projection)")
ax.set_ylabel("Density")
ax.set_title("Distribution of complexity scores\nTP vs UNK in mixed5 space")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "02_complexity_score_dist.png", dpi=150); plt.close()
print("Saved 02_complexity_score_dist.png")

# ── Plot 3: Top-20 Cohen's d channels ─────────────────────────────────────
top20 = cohens_data["top20_channels"]
ch_ids = [f"ch{c['channel']}" for c in top20]
d_vals = [c["cohens_d"] for c in top20]
colors = ["#E24B4A" if d < 0 else "#1D9E75" for d in d_vals]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(range(len(d_vals)), d_vals, color=colors, alpha=0.85)
ax.set_xticks(range(len(ch_ids))); ax.set_xticklabels(ch_ids, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Cohen's d  (positive = higher in TP)")
ax.set_title("Top 20 channels by |Cohen's d| — mixed5 TP vs UNK")
patch_tp  = mpatches.Patch(color="#1D9E75", label="Higher in TP")
patch_unk = mpatches.Patch(color="#E24B4A", label="Higher in UNK")
ax.legend(handles=[patch_tp, patch_unk], fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "03_cohens_d_top20.png", dpi=150); plt.close()
print("Saved 03_cohens_d_top20.png")

# ── Plot 4: Top-20 probe weights ──────────────────────────────────────────
top20_w = results["top20_probe_weights"]
w_ch  = [f"ch{c['channel']}" for c in top20_w]
w_val = [c["weight"] for c in top20_w]
w_col = ["#1D9E75" if w > 0 else "#E24B4A" for w in w_val]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(w_val)), w_val, color=w_col, alpha=0.85)
ax.set_xticks(range(len(w_ch))); ax.set_xticklabels(w_ch, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Logistic regression weight  (positive → TP)")
ax.set_title("Top 20 probe weight channels — linear probe on mixed5")
overlap = set(results["channel_overlap_cohens_vs_probe"])
for i, c in enumerate(top20_w):
    if c["channel"] in overlap:
        ax.get_children()[i].set_edgecolor("black")
        ax.get_children()[i].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(out_dir / "04_probe_weights_top20.png", dpi=150); plt.close()
print("Saved 04_probe_weights_top20.png")

# ── Plot 5: PCA coloured by complexity score (continuous) ─────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: colour by TP/UNK label
for lbl_int, lbl_name in label_names.items():
    mask = y == lbl_int
    axes[0].scatter(X_pca[mask,0], X_pca[mask,1],
                    c=COLORS[lbl_name], label=lbl_name,
                    alpha=0.6, s=18, linewidths=0)
axes[0].set_title("PCA — coloured by GIAB label")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].legend(fontsize=9, markerscale=1.5)

# Right: colour by continuous complexity score
sc = axes[1].scatter(X_pca[:,0], X_pca[:,1],
                     c=scores, cmap="RdYlGn_r",
                     alpha=0.7, s=18, linewidths=0)
plt.colorbar(sc, ax=axes[1], label="Complexity score")
axes[1].set_title("PCA — coloured by complexity score (probe projection)")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

plt.tight_layout()
plt.savefig(out_dir / "05_pca_complexity_score.png", dpi=150); plt.close()
print("Saved 05_pca_complexity_score.png")

# ── Plot 6: Permutation test null distribution ────────────────────────────
perm = results["permutation_test"]
rng  = np.random.default_rng(42)
null_aurocs = []
from sklearn.model_selection import cross_val_score
cv_sk = __import__("sklearn.model_selection", fromlist=["StratifiedKFold"]).StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42)
for _ in range(100):
    yp = rng.permutation(y)
    s  = cross_val_score(
        LogisticRegression(C=1.0, max_iter=500, random_state=42),
        X_scaled, yp, cv=cv_sk, scoring="roc_auc")
    null_aurocs.append(float(s.mean()))

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(null_aurocs, bins=20, color="#B4B2A9", alpha=0.8,
        edgecolor="none", label="Null AUROC (permuted labels)")
ax.axvline(cv_auroc, color="#378ADD", lw=2,
           label=f"Observed AUROC = {cv_auroc:.3f}")
ax.set_xlabel("AUROC"); ax.set_ylabel("Count")
ax.set_title(f"Permutation test  (p = {perm['p_value']:.4f})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "06_permutation_test.png", dpi=150); plt.close()
print("Saved 06_permutation_test.png")

print(f"\nAll 6 plots saved to {out_dir}/")
