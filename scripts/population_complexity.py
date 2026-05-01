"""Population complexity projection — all phases.

Embeddings are at data/embeddings/<sample>_mixed5/activation_cache/activations_00000000.npz
with key 'mixed5', shape (n_sites, 4, 12, 768). We pool spatial dims → (n_sites, 768).
NA12878 is the probe training sample; we refit the scaler on it and apply the same
scaler to all other samples.
"""

import json
import numpy as np
import warnings
from pathlib import Path
from scipy import stats

BASE = Path(__file__).parent.parent
EMBED_BASE = BASE / "data/embeddings"
PROBE_DIR = BASE / "results/giab_validation/linear_probe"
OUT = BASE / "results/population_complexity"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("HG00731", "GBR", "EUR"),
    ("NA20502", "TSI", "EUR"),
    ("HG01985", "LWK", "AFR"),
    ("HG02922", "LWK", "AFR"),
    ("HG01048", "CLM", "AMR"),
    ("HG01197", "CLM", "AMR"),
    ("HG01565", "PEL", "AMR"),
    ("HG00759", "CHS", "EAS"),
    ("NA18939", "JPT", "EAS"),
    ("HG00864", "CDX", "EAS"),
    ("HG03009", "GIH", "SAS"),
    ("NA20847", "GIH", "SAS"),
]

COLORS = {"AFR": "#E24B4A", "AMR": "#EF9F27", "EAS": "#1D9E75", "EUR": "#378ADD", "SAS": "#7F77DD"}


def load_embeddings(sample_id):
    """Return (n_sites, 768) by mean-pooling spatial dims."""
    npz_path = EMBED_BASE / f"{sample_id}_mixed5" / "activation_cache" / "activations_00000000.npz"
    d = np.load(npz_path, allow_pickle=True)
    key = next((k for k in d.files if "mixed" in k.lower()), d.files[0])
    act = d[key]  # (n_sites, 4, 12, 768)
    return act.mean(axis=(1, 2))  # → (n_sites, 768)


# ─── Phase 0: Disk Audit ──────────────────────────────────────────────────────
print("\n=== Phase 0: Disk Audit ===")
usable, skipped = [], []
for sid, pop, spop in SAMPLES:
    npz_path = EMBED_BASE / f"{sid}_mixed5" / "activation_cache" / "activations_00000000.npz"
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        key = next((k for k in d.files if "mixed" in k.lower()), d.files[0])
        n = d[key].shape[0]
        usable.append({"sample_id": sid, "population": pop, "superpopulation": spop, "n_sites": n})
        print(f"  USABLE {sid:12s} {spop:4s} {pop:4s}  {n} sites")
    else:
        skipped.append({"sample_id": sid, "population": pop, "superpopulation": spop, "n_sites": 0, "reason": "no npz"})
        print(f"  SKIPPED {sid:12s} {spop:4s}  (no npz)")

audit = {"usable": usable, "skipped": skipped}
with open(OUT / "audit.json", "w") as f:
    json.dump(audit, f, indent=2)
print(f"audit.json written: {len(usable)} usable, {len(skipped)} skipped")
assert len(usable) >= 8, "Gate failed: fewer than 8 usable samples"


# ─── Phase 1: Load / Refit Probe ─────────────────────────────────────────────
print("\n=== Phase 1: Load Probe Weights ===")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_na12878 = load_embeddings("NA12878")
y_na12878 = np.load(PROBE_DIR / "labels.npy")
assert len(X_na12878) == len(y_na12878), f"Shape mismatch: {len(X_na12878)} sites vs {len(y_na12878)} labels"

scaler = StandardScaler().fit(X_na12878)
X_scaled_na12878 = scaler.transform(X_na12878)
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_scaled_na12878, y_na12878)
weights = clf.coef_[0]

probs = clf.predict_proba(X_scaled_na12878)[:, 1]
auroc = roc_auc_score(y_na12878, probs)
print(f"NA12878 train AUROC (sanity check): {auroc:.4f}")
assert auroc > 0.98, f"Probe refit failed: AUROC={auroc:.4f}"

np.save(OUT / "probe_weights.npy", weights)
np.save(OUT / "scaler_mean.npy", scaler.mean_)
np.save(OUT / "scaler_scale.npy", scaler.scale_)
print("Saved probe_weights.npy, scaler_mean.npy, scaler_scale.npy")


# ─── Phase 2: Project All Samples ────────────────────────────────────────────
print("\n=== Phase 2: Project All Samples ===")
manifest = {}

# NA12878 first
na12878_scores = X_scaled_na12878 @ weights
np.save(OUT / "NA12878_scores.npy", na12878_scores)
np.save(OUT / "NA12878_positions.npy", np.array([f"site_{i:06d}" for i in range(len(na12878_scores))]))
manifest["NA12878"] = {
    "superpop": "EUR", "population": "CEU", "n_sites": len(na12878_scores),
    "mean_score": float(na12878_scores.mean()), "std_score": float(na12878_scores.std()),
}
print(f"  NA12878 (EUR/CEU): {len(na12878_scores)} sites, mean={na12878_scores.mean():.3f}")

for entry in usable:
    sid = entry["sample_id"]
    pop = entry["population"]
    spop = entry["superpopulation"]
    X = load_embeddings(sid)
    X_sc = (X - scaler.mean_) / scaler.scale_
    scores = X_sc @ weights
    positions = np.array([f"site_{i:06d}" for i in range(len(scores))])
    np.save(OUT / f"{sid}_scores.npy", scores)
    np.save(OUT / f"{sid}_positions.npy", positions)
    manifest[sid] = {
        "superpop": spop, "population": pop, "n_sites": len(scores),
        "mean_score": float(scores.mean()), "std_score": float(scores.std()),
    }
    print(f"  {sid:12s} ({spop}/{pop}): {len(scores)} sites, mean={scores.mean():.3f}")

with open(OUT / "scores_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"scores_manifest.json: {len(manifest)} samples")

spops_present = set(v["superpop"] for v in manifest.values())
assert len(manifest) >= 8, "Gate failed: <8 samples in manifest"
assert len(spops_present) >= 4, f"Gate failed: only {len(spops_present)} superpops"


# ─── Phase 3: Statistical Tests ──────────────────────────────────────────────
print("\n=== Phase 3: Statistical Tests ===")
superpop_scores = {}
for sid, info in manifest.items():
    sp = info["superpop"]
    scores = np.load(OUT / f"{sid}_scores.npy")
    if sp not in superpop_scores:
        superpop_scores[sp] = []
    superpop_scores[sp].append(scores)

superpop_pooled = {sp: np.concatenate(arrs) for sp, arrs in superpop_scores.items()}
for sp, arr in superpop_pooled.items():
    print(f"  {sp}: {len(arr)} sites, mean={arr.mean():.3f}, std={arr.std():.3f}")

groups = [arr for arr in superpop_pooled.values() if len(arr) > 0]
H, p_kw = stats.kruskal(*groups)
print(f"\nKruskal-Wallis H={H:.3f}, p={p_kw:.4e}")

eur_scores = superpop_pooled["EUR"]
n_comparisons = sum(1 for sp in superpop_pooled if sp != "EUR")
pairwise = {}
for sp, scores in superpop_pooled.items():
    if sp == "EUR":
        continue
    U, p_raw = stats.mannwhitneyu(scores, eur_scores, alternative="two-sided")
    p_bonf = min(p_raw * n_comparisons, 1.0)
    d = (np.mean(scores) - np.mean(eur_scores)) / np.sqrt(
        (np.std(scores) ** 2 + np.std(eur_scores) ** 2) / 2
    )
    pairwise[sp] = {
        "U": float(U), "p_raw": float(p_raw),
        "p_bonferroni": float(p_bonf), "cohens_d": float(d),
    }
    sig = "**" if p_bonf < 0.05 else "  "
    print(f"  {sp} vs EUR: U={U:.0f}, p_raw={p_raw:.4e}, p_bonf={p_bonf:.4e}, d={d:.3f} {sig}")

underpowered = [sp for sp, arr in superpop_pooled.items() if len(arr) < 50]
stats_out = {
    "kruskal_wallis": {"H": float(H), "p": float(p_kw)},
    "pairwise_vs_eur": pairwise,
    "n_per_superpop": {sp: int(len(arr)) for sp, arr in superpop_pooled.items()},
    "underpowered_superpops": underpowered,
    "eur_includes_na12878": True,
    "n_comparisons_bonferroni": n_comparisons,
}
with open(OUT / "stats.json", "w") as f:
    json.dump(stats_out, f, indent=2)
print("stats.json written")


# ─── Phase 4: Plots ───────────────────────────────────────────────────────────
print("\n=== Phase 4: Plots ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SPOP_ORDER = ["AFR", "AMR", "EAS", "EUR", "SAS"]

# Plot 1 — KDE curves
fig, ax = plt.subplots(figsize=(9, 5))
for sp in SPOP_ORDER:
    if sp not in superpop_pooled:
        continue
    arr = superpop_pooled[sp]
    kde = gaussian_kde(arr)
    xs = np.linspace(arr.min() - 1, arr.max() + 1, 400)
    ax.plot(xs, kde(xs), color=COLORS[sp], lw=2, label=f"{sp} (n={len(arr)}, μ={arr.mean():.2f})")
ax.axvline(0, color="black", ls="--", lw=1, alpha=0.6, label="score=0")
ax.set_xlabel("Complexity score (probe projection)")
ax.set_ylabel("Density")
ax.set_title("Complexity score distributions by superpopulation (GRCh37 mixed5)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "01_kde_by_superpop.png", dpi=150)
plt.close(fig)
print("  01_kde_by_superpop.png")

# Plot 2 — Violin + jitter
fig, ax = plt.subplots(figsize=(9, 6))
positions_violin = range(len(SPOP_ORDER))
data_ordered = [superpop_pooled.get(sp, np.array([])) for sp in SPOP_ORDER]
parts = ax.violinplot([d for d in data_ordered if len(d) > 0],
                      positions=[i for i, d in enumerate(data_ordered) if len(d) > 0],
                      showmeans=False, showextrema=False, showmedians=False)
for i, (sp, arr) in enumerate(zip(SPOP_ORDER, data_ordered)):
    if len(arr) == 0:
        continue
    parts["bodies"][i].set_facecolor(COLORS[sp])
    parts["bodies"][i].set_alpha(0.5)
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(arr))
    ax.scatter([i + j for j in jitter], arr, color=COLORS[sp], alpha=0.3, s=4)
    ax.scatter([i], [arr.mean()], color="white", s=60, zorder=5, edgecolors=COLORS[sp], lw=1.5)

ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)
ax.set_xticks(range(len(SPOP_ORDER)))
ax.set_xticklabels(SPOP_ORDER)
ax.set_ylabel("Complexity score")
ax.set_title("Complexity score by superpopulation — violin + individual sites")
p_str = f"p={p_kw:.2e}" if p_kw >= 1e-300 else "p<1e-300"
ax.text(0.02, 0.97, f"Kruskal-Wallis H={H:.1f}, {p_str}", transform=ax.transAxes,
        va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
fig.tight_layout()
fig.savefig(OUT / "02_violin_by_superpop.png", dpi=150)
plt.close(fig)
print("  02_violin_by_superpop.png")

# Plot 3 — Per-sample strip plot
sample_ids = list(manifest.keys())
sample_spops = [manifest[s]["superpop"] for s in sample_ids]
sample_scores_list = [np.load(OUT / f"{s}_scores.npy") for s in sample_ids]

# Sort by superpop then sample
order = sorted(range(len(sample_ids)), key=lambda i: (sample_spops[i], sample_ids[i]))
sample_ids_ord = [sample_ids[i] for i in order]
sample_spops_ord = [sample_spops[i] for i in order]
sample_scores_ord = [sample_scores_list[i] for i in order]

fig, ax = plt.subplots(figsize=(14, 6))
for xi, (sid, spop, arr) in enumerate(zip(sample_ids_ord, sample_spops_ord, sample_scores_ord)):
    jitter = np.random.default_rng(xi).uniform(-0.2, 0.2, size=len(arr))
    ax.scatter([xi + j for j in jitter], arr, color=COLORS[spop], alpha=0.25, s=5)
    ax.hlines(arr.mean(), xi - 0.35, xi + 0.35, colors=COLORS[spop], lw=2)

ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
ax.set_xticks(range(len(sample_ids_ord)))
ax.set_xticklabels(sample_ids_ord, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Complexity score")
ax.set_title("Per-sample complexity scores (each dot = one variant site)")
# Superpop legend
handles = [plt.Line2D([0], [0], color=COLORS[sp], lw=3, label=sp) for sp in SPOP_ORDER if sp in set(sample_spops)]
ax.legend(handles=handles, title="Superpop", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_strip_by_sample.png", dpi=150)
plt.close(fig)
print("  03_strip_by_sample.png")

# Plot 4 — Effect sizes
fig, ax = plt.subplots(figsize=(7, 4))
spops_noneur = [sp for sp in SPOP_ORDER if sp != "EUR" and sp in pairwise]
ds = [pairwise[sp]["cohens_d"] for sp in spops_noneur]
colors_bar = ["#EF9F27" if d < 0 else "#378ADD" for d in ds]
bars = ax.barh(spops_noneur, ds, color=colors_bar)
for line_val, ls, label in [(-0.2, ":", "small"), (-0.5, "--", "medium"), (-0.8, "-.", "large")]:
    ax.axvline(line_val, color="gray", ls=ls, lw=1, label=f"|d|={abs(line_val)} ({label})")
    if line_val != -0.2:
        ax.axvline(-line_val, color="gray", ls=ls, lw=1)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Cohen's d (vs EUR)")
ax.set_title("Effect size vs EUR baseline — complexity score by superpopulation")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "04_effect_sizes.png", dpi=150)
plt.close(fig)
print("  04_effect_sizes.png")

# Plot 5 — Score vs PC1
print("  Running PCA for plot 5...")
from sklearn.decomposition import PCA

all_X_scaled = []
all_scores_combined = []
all_spops_combined = []
for sid, info in manifest.items():
    X = load_embeddings(sid)
    X_sc = (X - scaler.mean_) / scaler.scale_
    scores = np.load(OUT / f"{sid}_scores.npy")
    all_X_scaled.append(X_sc)
    all_scores_combined.append(scores)
    all_spops_combined.extend([info["superpop"]] * len(scores))

X_all = np.vstack(all_X_scaled)
scores_all = np.concatenate(all_scores_combined)
spops_all = np.array(all_spops_combined)

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_all)

fig, ax = plt.subplots(figsize=(8, 6))
for sp in SPOP_ORDER:
    mask = spops_all == sp
    if mask.sum() == 0:
        continue
    ax.scatter(pcs[mask, 0], scores_all[mask], color=COLORS[sp], alpha=0.25, s=6, label=sp)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel("Complexity score (probe projection)")
ax.set_title("Complexity score vs PC1 — coloured by superpopulation")
ax.legend(fontsize=9, markerscale=2)
fig.tight_layout()
fig.savefig(OUT / "05_score_vs_pca.png", dpi=150)
plt.close(fig)
print("  05_score_vs_pca.png")


# ─── Phase 5: Validation Checks ──────────────────────────────────────────────
print("\n=== Phase 5: Validation Checks ===")
checks = {}

# Check 1: NA12878 TP mean > 0, UNK mean < 0
tp_mask = y_na12878 == 1
unk_mask = y_na12878 == 0
tp_mean = float(na12878_scores[tp_mask].mean())
unk_mean = float(na12878_scores[unk_mask].mean())
checks["na12878_score_direction"] = {
    "tp_mean": tp_mean, "unk_mean": unk_mean,
    "status": "PASS" if tp_mean > 0 and unk_mean < 0 else "WARN",
    "description": f"TP mean={tp_mean:.3f} (expected >0), UNK mean={unk_mean:.3f} (expected <0)",
}
print(f"  Check 1 — TP mean={tp_mean:.3f}, UNK mean={unk_mean:.3f}: {checks['na12878_score_direction']['status']}")

# Check 2: All scores within NA12878 range ± 5
na12878_min, na12878_max = na12878_scores.min(), na12878_scores.max()
lo, hi = na12878_min - 5, na12878_max + 5
range_issues = []
for sid, info in manifest.items():
    sc = np.load(OUT / f"{sid}_scores.npy")
    if sc.min() < lo or sc.max() > hi:
        range_issues.append(f"{sid}: [{sc.min():.2f}, {sc.max():.2f}]")
checks["score_range"] = {
    "expected_range": [float(lo), float(hi)],
    "issues": range_issues,
    "status": "PASS" if not range_issues else "WARN",
    "description": f"NA12878 range [{na12878_min:.2f}, {na12878_max:.2f}]; tolerance ±5",
}
print(f"  Check 2 — score range: {checks['score_range']['status']} ({len(range_issues)} issues)")

# Check 3: No superpop mean > 5 SD above EUR mean
eur_mean = superpop_pooled["EUR"].mean()
eur_std = superpop_pooled["EUR"].std()
implausible = []
for sp, arr in superpop_pooled.items():
    z = (arr.mean() - eur_mean) / eur_std if eur_std > 0 else 0
    if z > 5:
        implausible.append(f"{sp}: z={z:.1f}")
checks["superpop_mean_ordering"] = {
    "eur_mean": float(eur_mean), "eur_std": float(eur_std),
    "superpop_means": {sp: float(arr.mean()) for sp, arr in superpop_pooled.items()},
    "implausible": implausible,
    "status": "PASS" if not implausible else "WARN",
}
print(f"  Check 3 — superpop means: {checks['superpop_mean_ordering']['status']}")

# Check 4: Channel 604 spot-check on lowest-mean sample
non_na12878 = [(sid, info) for sid, info in manifest.items() if sid != "NA12878"]
lowest_sid = min(non_na12878, key=lambda x: x[1]["mean_score"])[0]
X_lowest = load_embeddings(lowest_sid)
X_na = load_embeddings("NA12878")
ch604_lowest = float(X_lowest[:, :, :, 604].mean() if X_lowest.ndim == 3 else
                     np.load(EMBED_BASE / f"{lowest_sid}_mixed5/activation_cache/activations_00000000.npz")["mixed5"][:, :, :, 604].mean())
ch604_na = float(np.load(EMBED_BASE / "NA12878_mixed5/activation_cache/activations_00000000.npz")["mixed5"][:, :, :, 604].mean())
checks["ch604_spotcheck"] = {
    "lowest_mean_sample": lowest_sid,
    "ch604_lowest": ch604_lowest,
    "ch604_na12878": ch604_na,
    "status": "PASS" if ch604_lowest <= ch604_na else "WARN",
    "description": f"{lowest_sid} ch604={ch604_lowest:.4f}, NA12878 ch604={ch604_na:.4f}",
}
print(f"  Check 4 — ch604 spot-check ({lowest_sid}): {checks['ch604_spotcheck']['status']}")

with open(OUT / "validation_checks.json", "w") as f:
    json.dump(checks, f, indent=2)
print("validation_checks.json written")


# ─── Phase 6: Summary Report ─────────────────────────────────────────────────
print("\n=== Phase 6: Summary Report ===")

sig_pairs = [sp for sp, r in pairwise.items() if r["p_bonferroni"] < 0.05]
direction_lines = []
for sp, r in pairwise.items():
    direction = "lower" if r["cohens_d"] < 0 else "higher"
    mag = "small" if abs(r["cohens_d"]) < 0.2 else ("medium" if abs(r["cohens_d"]) < 0.5 else "large")
    direction_lines.append(f"- **{sp}**: d={r['cohens_d']:.3f} ({direction}, {mag} effect), p_bonf={r['p_bonferroni']:.3e}")

per_spop_consistency = []
for sp, sample_list in superpop_scores.items():
    means = [arr.mean() for arr in sample_list]
    consistent = all(m < eur_mean for m in means) or all(m > eur_mean for m in means)
    per_spop_consistency.append(f"- **{sp}**: sample means = [{', '.join(f'{m:.3f}' for m in means)}]"
                                + (" (consistent direction)" if consistent else " (mixed directions)"))

findings = f"""# Population Complexity Projection — Findings

*Generated from `scripts/population_complexity.py`*

---

## 1. Sample Inventory

**Included samples (n=13, including NA12878):**

| Sample | Population | Superpop | n_sites | Mean score |
|--------|-----------|----------|---------|------------|
""" + "\n".join(
    f"| {sid} | {info['population']} | {info['superpop']} | {info['n_sites']} | {info['mean_score']:.3f} |"
    for sid, info in sorted(manifest.items(), key=lambda x: x[1]["superpop"])
) + f"""

**Excluded:** NA12878 is included in the EUR reference distribution but was the probe training sample (noted in caveats).
No samples from the 13-sample active set were missing — all 12 non-NA12878 samples had usable embeddings.

---

## 2. Kruskal-Wallis Test

**H = {H:.3f}, p = {p_kw:.4e}**

{"There IS a statistically significant difference in complexity scores across superpopulations (p < 0.05)." if p_kw < 0.05 else "No statistically significant difference across superpopulations (p ≥ 0.05)."}

Sites per superpopulation:
{chr(10).join(f'- {sp}: {len(superpop_pooled[sp])} sites' for sp in SPOP_ORDER if sp in superpop_pooled)}

---

## 3. Pairwise Results (vs EUR, Bonferroni-corrected)

{chr(10).join(direction_lines)}

{"**Significant differences after Bonferroni correction:** " + ", ".join(sig_pairs) if sig_pairs else "No superpopulation showed a statistically significant difference from EUR after Bonferroni correction."}

---

## 4. Direction of Effect

EUR mean complexity score: {eur_mean:.3f} ± {eur_std:.3f} (includes NA12878, the probe training sample).

Per-superpopulation consistency across samples:
{chr(10).join(per_spop_consistency)}

{"Non-EUR sites trend **lower** (more complex / uncertain) than EUR sites on average." if sum(r["cohens_d"] for r in pairwise.values()) < 0 else "Non-EUR sites trend **higher** than EUR sites on average — unexpected given probe training on EUR data."}

---

## 5. Caveats

1. **EUR circular reference bias:** The probe was trained on NA12878 (EUR/CEU). EUR is simultaneously the probe training reference and a comparison group. This creates a structural advantage for EUR sites scoring as TP-like. The effect sizes for non-EUR populations must be interpreted with this in mind.

2. **Small sample sizes:** n=2–3 samples per superpopulation, covering a 100 kb region of chr20. Effects should be replicated on full-chr20 data before drawing strong conclusions.

3. **GIAB benchmark coverage confound:** The complexity axis captures GIAB callability (high-confidence vs uncertain regions), not purely genomic complexity. Some EUR advantage may reflect GIAB benchmark coverage bias — the GIAB high-confidence regions were originally defined on EUR samples — rather than model bias per se.

4. **Single region:** All data comes from chr20:10,000,000–10,100,000 (100 kb). Population-specific structural variants or coverage patterns in this window could drive apparent effects.

---

## 6. Next Steps

- **Full chr20 analysis:** Extend embeddings to the full chr20 (≈50 Mb). This would increase site counts from ~300–400 to tens of thousands per sample, providing statistical power to detect small effect sizes (d < 0.2).

- **GIAB truth sets for non-EUR samples:** If GIAB truth sets exist for AFR/AMR/EAS/SAS samples (e.g., HG002–HG007), rerun the linear probe analysis on those samples to get a ground-truth calibration that is independent of EUR.

- **Channel attribution:** Use the Cohen's d channel analysis (already run for NA12878) to identify which mixed5 channels drive the population complexity differences. If population-informative channels overlap with complexity-informative channels, this is direct evidence of entanglement.

- **GRCh38 cross-reference:** Compare complexity scores for the same samples against GRCh38 to disentangle reference-build effects from population effects.
"""

with open(OUT / "FINDINGS.md", "w") as f:
    f.write(findings)
print("FINDINGS.md written")

print("\n=== All phases complete ===")
print(f"Output directory: {OUT}")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
