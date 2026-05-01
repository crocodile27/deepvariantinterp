# Population Complexity Projection
## Claude Code Orchestration Document

> **Read this entire document before writing any code.**
> Follow the phases in order. Check each gate condition before proceeding.
> Resume from the last incomplete phase if re-invoked.

---

## Context & Scientific Goal

We have a linear probe trained on NA12878 (EUR/CEU) mixed5 embeddings that
separates GIAB high-confidence sites (TP) from uncertain sites (UNK) with
AUROC 0.994. The probe weight vector defines a **complexity axis** in 768-d
embedding space: high score = clean, callable variant; low score = complex,
uncertain genomic context.

**Question:** Do variant sites from non-European superpopulations score
systematically lower on this axis — meaning the model represents them as
more ambiguous, even before any variant call is made?

This is a proxy for population bias in DeepVariant's internal representations.

---

## Repo State (what already exists)

```
results/giab_validation/linear_probe/
  probe_results.json          ← probe weights, AUROC, channel overlap
  complexity_scores.npy       ← NA12878 scores (337 sites)
  labels.npy                  ← NA12878 binary labels (1=TP, 0=UNK)
  X_scaled.npy                ← NA12878 standardised embeddings (337 × 768)
  cohens_d_channels.json      ← top-20 channels by Cohen's d
  validator_report.json       ← PASS verdict, delta=0.0

data/embeddings/grch37/
  NA12878/activations/        ← 337 × .npz (EUR, already used for probe)
  HG01985/activations/        ← AFR (LWK)   — n_sites varies per sample
  HG02922/activations/        ← AFR (LWK)
  HG01048/activations/        ← AMR (CLM)
  HG01197/activations/        ← AMR (CLM)
  HG00759/activations/        ← EAS (CHS)
  NA18939/activations/        ← EAS (JPT)
  HG00731/activations/        ← EUR (GBR)   — may have 0 sites, check first
  NA20502/activations/        ← EUR (TSI)
  HG03009/activations/        ← SAS (GIH)
  NA20847/activations/        ← SAS (GIH)

  # These two samples have uncertain status — check disk before including:
  HG01565/activations/        ← AMR (PEL)   — had Docker segfault previously
  HG00864/activations/        ← EAS (CDX)   — had "no .npz files" error
```

**Sample registry** (authoritative — use this, not the directory listing):

| Sample   | Population | Superpop | Notes                        |
|----------|-----------|----------|------------------------------|
| NA12878  | CEU       | EUR      | Probe training sample, skip  |
| HG00731  | GBR       | EUR      | May have 0 sites — check     |
| NA20502  | TSI       | EUR      | —                            |
| HG01985  | LWK       | AFR      | —                            |
| HG02922  | LWK       | AFR      | —                            |
| HG01048  | CLM       | AMR      | —                            |
| HG01197  | CLM       | AMR      | —                            |
| HG01565  | PEL       | AMR      | Uncertain — audit first      |
| HG00759  | CHS       | EAS      | —                            |
| NA18939  | JPT       | EAS      | —                            |
| HG00864  | CDX       | EAS      | Uncertain — audit first      |
| HG03009  | GIH       | SAS      | —                            |
| NA20847  | GIH       | SAS      | —                            |

---

## Phases

### Phase 0 — Disk Audit
**Purpose:** Determine exactly which samples have usable .npz files.
**Do not skip even if you think you know the answer.**

```python
# Run this audit script first
import numpy as np
from pathlib import Path

SAMPLES = [
    ("HG00731", "GBR", "EUR"), ("NA20502", "TSI", "EUR"),
    ("HG01985", "LWK", "AFR"), ("HG02922", "LWK", "AFR"),
    ("HG01048", "CLM", "AMR"), ("HG01197", "CLM", "AMR"), ("HG01565", "PEL", "AMR"),
    ("HG00759", "CHS", "EAS"), ("NA18939", "JPT", "EAS"), ("HG00864", "CDX", "EAS"),
    ("HG03009", "GIH", "SAS"), ("NA20847", "GIH", "SAS"),
]
base = Path("data/embeddings/grch37")
usable, skipped = [], []
for sid, pop, spop in SAMPLES:
    npzs = list((base / sid / "activations").glob("*.npz")) if (base / sid / "activations").exists() else []
    if len(npzs) > 0:
        usable.append((sid, pop, spop, len(npzs)))
    else:
        skipped.append((sid, pop, spop, 0))

print("USABLE:")
for s in usable: print(f"  {s[0]:12s} {s[2]:4s}  {s[3]} sites")
print("\nSKIPPED (0 sites):")
for s in skipped: print(f"  {s[0]:12s} {s[2]:4s}")
```

**Gate:** Write `results/population_complexity/audit.json` with the usable
and skipped lists. Do not proceed to Phase 1 until this file exists.

---

### Phase 1 — Load Probe Weights
**Purpose:** Extract the logistic regression weight vector from the trained probe.

```python
import json, numpy as np
from pathlib import Path

with open("results/giab_validation/linear_probe/probe_results.json") as f:
    probe = json.load(f)

# The probe was fit on StandardScaler-transformed embeddings.
# We need to re-fit the scaler on NA12878 data to transform other samples.
# Load NA12878 raw embeddings (before scaling) to fit the scaler.
acts_dir = Path("data/embeddings/grch37/NA12878/activations")
na12878_vecs = []
for npz in sorted(acts_dir.glob("*.npz")):
    d = np.load(npz)
    key = next((k for k in d.files if "mixed" in k.lower() or "concat" in k.lower()), d.files[0])
    act = d[key]
    na12878_vecs.append(act.mean(axis=tuple(range(act.ndim - 1))))
X_na12878 = np.vstack(na12878_vecs)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

scaler = StandardScaler().fit(X_na12878)

# Refit probe on NA12878 to recover weight vector
y_na12878 = np.load("results/giab_validation/linear_probe/labels.npy")
X_scaled_na12878 = scaler.transform(X_na12878)
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_scaled_na12878, y_na12878)
weights = clf.coef_[0]  # (768,)

# Verify: AUROC on NA12878 should be ~1.0
from sklearn.metrics import roc_auc_score
probs = clf.predict_proba(X_scaled_na12878)[:, 1]
auroc = roc_auc_score(y_na12878, probs)
print(f"NA12878 train AUROC (sanity check, should be ~1.0): {auroc:.4f}")
assert auroc > 0.98, f"Probe refit failed: AUROC={auroc:.4f}"
```

**Important:** The scaler was fit on NA12878. We apply the **same** scaler to
all other samples. This is intentional — we want to project into the same
coordinate system as the probe training data. Do not refit the scaler per sample.

**Gate:** Save `scaler` and `weights` as:
```
results/population_complexity/probe_weights.npy   ← shape (768,)
results/population_complexity/scaler_mean.npy      ← shape (768,)
results/population_complexity/scaler_scale.npy     ← shape (768,)
```

---

### Phase 2 — Project All Samples
**Purpose:** For each usable sample from the audit, load embeddings,
apply the NA12878 scaler, project onto the probe weight vector,
and record one complexity score per variant site.

```python
# Loop over usable samples from audit.json
# For each sample:
#   1. Load all .npz files in activations/
#   2. Pool spatial dims → (n_sites, 768)
#   3. Apply scaler: X_scaled = (X - scaler_mean) / scaler_scale
#   4. Project: scores = X_scaled @ weights   → (n_sites,)
#   5. Save to results/population_complexity/<sample_id>_scores.npy
#   6. Write a per-sample metadata dict

# Key: one .npz = one variant site, filename = genomic position
# e.g. chr20_10001436.npz → position chr20:10001436
```

**Per-sample output contract:**
```
results/population_complexity/
  <sample_id>_scores.npy     ← (n_sites,) float array
  <sample_id>_positions.npy  ← (n_sites,) string array of chr20_XXXXXXX
  scores_manifest.json       ← {sample_id: {superpop, pop, n_sites, mean_score, std_score}}
```

**Gate:** `scores_manifest.json` must exist and contain at least 8 samples
across at least 4 superpopulations before proceeding.

---

### Phase 3 — Statistical Tests
**Purpose:** Test whether complexity score distributions differ significantly
between superpopulations. Run before plotting so we report honest p-values.

Tests to run:
1. **Kruskal-Wallis** across all 5 superpopulations (non-parametric ANOVA).
   Report H-statistic and p-value.
2. **Pairwise Mann-Whitney U** for each superpopulation pair vs EUR.
   Apply Bonferroni correction (multiply p by number of comparisons).
3. **Effect sizes:** Cohen's d for each non-EUR superpop vs EUR (pooled).

```python
# Pool all sites per superpopulation
# superpop_scores = {"AFR": [...], "AMR": [...], "EAS": [...], "EUR": [...], "SAS": [...]}
# EUR includes NA12878 (the probe training sample) — include it for the
# reference distribution but note this in the output.

from scipy import stats

# Kruskal-Wallis
groups = [scores for scores in superpop_scores.values() if len(scores) > 0]
H, p_kw = stats.kruskal(*groups)

# Pairwise vs EUR
eur_scores = superpop_scores["EUR"]
pairwise = {}
for spop, scores in superpop_scores.items():
    if spop == "EUR" or len(scores) == 0: continue
    U, p_raw = stats.mannwhitneyu(scores, eur_scores, alternative="two-sided")
    p_bonf = min(p_raw * 4, 1.0)   # 4 comparisons
    d = (np.mean(scores) - np.mean(eur_scores)) / np.sqrt(
        (np.std(scores)**2 + np.std(eur_scores)**2) / 2)
    pairwise[spop] = {"U": U, "p_raw": p_raw, "p_bonferroni": p_bonf, "cohens_d": d}
```

**Gate:** Write `results/population_complexity/stats.json`.
Flag in the JSON if any superpopulation has fewer than 50 sites total
(underpowered — interpret with caution).

---

### Phase 4 — Plots
**Purpose:** Produce all figures. Run only after stats.json exists.

**Plot 1 — KDE curves per superpopulation** (`01_kde_by_superpop.png`)
- One curve per superpopulation, coloured:
  AFR=#E24B4A, AMR=#EF9F27, EAS=#1D9E75, EUR=#378ADD, SAS=#7F77DD
- X-axis: complexity score. Y-axis: density.
- Vertical dashed line at score=0 (boundary between TP-like and UNK-like).
- Annotate each curve with n_sites and mean score.
- Title: "Complexity score distributions by superpopulation (GRCh37 mixed5)"

**Plot 2 — Violin plot** (`02_violin_by_superpop.png`)
- One violin per superpopulation, same colour scheme.
- Overlay individual points (jittered, alpha=0.3, size=4).
- Mark the mean with a white dot.
- Add a horizontal dashed line at score=0.
- Annotate Kruskal-Wallis p-value in the top-left corner.
- X-axis ordered: AFR, AMR, EAS, EUR, SAS.

**Plot 3 — Per-sample strip plot** (`03_strip_by_sample.png`)
- X-axis: individual sample IDs, grouped and coloured by superpop.
- Y-axis: complexity score (each dot = one variant site).
- Show sample means as horizontal lines within each strip.
- Useful for identifying whether population effects are driven by one outlier sample.

**Plot 4 — Effect size summary** (`04_effect_sizes.png`)
- Horizontal bar chart of Cohen's d for each non-EUR superpop vs EUR.
- Colour bars by sign: negative (lower than EUR) = orange, positive = blue.
- Add reference lines at d=−0.2 (small), d=−0.5 (medium), d=−0.8 (large).
- Title: "Effect size vs EUR baseline — complexity score by superpopulation"

**Plot 5 — Score vs PC1 scatter, coloured by superpop** (`05_score_vs_pca.png`)
- Run PCA on the combined (all samples) scaled embedding matrix.
- Scatter PC1 vs complexity score, colour by superpopulation.
- This shows whether the complexity axis is aligned with or orthogonal to
  the main axis of variance across populations.

---

### Phase 5 — Validation Check
**Purpose:** Sanity-check the projection before interpreting results.

Run these checks and write results to `validation_checks.json`:

1. **NA12878 score distribution check:** Mean score for NA12878 TPs should
   be positive (>0), UNKs should be negative (<0). If not, the scaler
   was applied incorrectly.

2. **Score range check:** All sample scores should fall within
   [min(NA12878_scores) − 5, max(NA12878_scores) + 5]. Extreme outliers
   suggest a loading or pooling error.

3. **Superpop mean ordering:** No strong prior expectation, but AFR should
   not have an *implausibly* high score (e.g. >5 SD above EUR mean), which
   would indicate a data error rather than a real biological effect.

4. **Channel 604 spot-check:** For the sample with the lowest mean
   complexity score, verify that ch604 activation mean is lower than for
   NA12878. This confirms the embedding direction is consistent.

Flag any failed check in `validation_checks.json` with
`"status": "WARN"` and a description. Do not delete plots — report the
warning alongside them.

---

### Phase 6 — Summary Report
**Purpose:** Write a plain-English summary of findings.

Write `results/population_complexity/FINDINGS.md` containing:

1. **Sample inventory:** Which samples were included/excluded and why.
2. **Kruskal-Wallis result:** Is there a statistically significant difference
   across superpopulations? Quote H and p.
3. **Pairwise results:** Which superpops differ from EUR? Quote Bonferroni-
   corrected p-values and Cohen's d.
4. **Direction of effect:** Are non-EUR sites scoring lower (more complex)
   or higher? Is the effect consistent across samples within each superpop?
5. **Caveats:**
   - The probe was trained on NA12878 (EUR). EUR is both the reference
     sample for the probe and a comparison group — this creates a bias
     toward finding EUR sites as more TP-like.
   - Sample sizes are small (n=2–3 per superpop). Effects should be
     replicated on full chr20 before strong conclusions are drawn.
   - The complexity axis captures GIAB callability, not just genomic
     complexity. Some of the EUR advantage may reflect GIAB benchmark
     coverage bias rather than model bias.
6. **Next steps:** What would the full chr20 analysis add?

---

## Output Directory Contract

When complete, `results/population_complexity/` must contain:

```
audit.json                        Phase 0
probe_weights.npy                 Phase 1
scaler_mean.npy                   Phase 1
scaler_scale.npy                  Phase 1
<sample>_scores.npy  (×N)        Phase 2
<sample>_positions.npy (×N)      Phase 2
scores_manifest.json              Phase 2
stats.json                        Phase 3
01_kde_by_superpop.png            Phase 4
02_violin_by_superpop.png         Phase 4
03_strip_by_sample.png            Phase 4
04_effect_sizes.png               Phase 4
05_score_vs_pca.png               Phase 4
validation_checks.json            Phase 5
FINDINGS.md                       Phase 6
```

---

## How to Invoke

Paste the following into Claude Code:

```
Read skills/POPULATION_COMPLEXITY.md and execute all phases in order.
Check gate conditions before each phase transition.
Resume from the last incomplete phase if outputs already exist.
```

> The skills/ directory is the conventional location in this repo for
> orchestration docs. Copy this file there before invoking:
> `cp POPULATION_COMPLEXITY.md skills/POPULATION_COMPLEXITY.md`
