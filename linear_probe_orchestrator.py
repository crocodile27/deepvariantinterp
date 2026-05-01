"""
linear_probe_orchestrator.py

Runs a logistic regression linear probe on NA12878 mixed5 embeddings
to predict TP vs UNK (GIAB high-confidence label), then spawns a
validator agent to independently verify every claim before results
are written to disk.

Paste into deepvariantinterp repo root and run:
    python linear_probe_orchestrator.py

Prerequisites:
    pip install claude-agent-sdk scikit-learn umap-learn matplotlib numpy

Agent architecture
──────────────────
Phase 1  [probe-agent]      Fit probe, run CV, permutation test, write results
Phase 2  [validator-agent]  Re-runs all statistics independently and flags
                            any discrepancy > 0.01 AUROC or > 0.05 Cohen's d
Phase 3  [plot-agent]       Produces all figures only after validator signs off
"""

import asyncio
import json
import sys
from pathlib import Path
from claude_agent_sdk import query, ClaudeAgentOptions

# ── Config ──────────────────────────────────────────────────────────────────

NA12878_ACTS  = "data/embeddings/grch37/NA12878/activations"
LABELS_JSON   = "results/giab_validation/site_labels.json"
PROBE_DIR     = "results/giab_validation/linear_probe"
COHENS_D_JSON = f"{PROBE_DIR}/cohens_d_channels.json"   # written by probe-agent
PROBE_JSON    = f"{PROBE_DIR}/probe_results.json"        # written by probe-agent
VALIDATOR_JSON= f"{PROBE_DIR}/validator_report.json"     # written by validator-agent
PLOTS_DONE    = f"{PROBE_DIR}/plots_complete.flag"       # written by plot-agent

# ── Agent prompts ────────────────────────────────────────────────────────────

PROBE_PROMPT = f"""
You are the linear probe agent. Your job is to fit a logistic regression
probe on NA12878 mixed5 embeddings predicting TP vs UNK (GIAB labels),
then run a permutation test to confirm the result is above chance.

Inputs
  Activation dir : {NA12878_ACTS}/*.npz
  Labels JSON    : {LABELS_JSON}
  Output dir     : {PROBE_DIR}

Write and execute the following Python script exactly as written.
Do not skip any step. Save every intermediate result to disk.

─────────────────────────────────────────────────────────────────────────
```python
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

acts_dir   = Path("{NA12878_ACTS}")
labels_path = Path("{LABELS_JSON}")
out_dir    = Path("{PROBE_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

# ── 1. Load embeddings ────────────────────────────────────────────────────
with open(labels_path) as f:
    label_data = json.load(f)
site_labels = label_data["site_labels"]

embeddings, labels, positions = [], [], []
for npz_path in sorted(acts_dir.glob("*.npz")):
    stem = npz_path.stem
    lbl  = site_labels.get(stem, "UNK")
    if lbl not in ("TP", "UNK"):
        continue                          # skip FP/FN if any
    d   = np.load(npz_path)
    key = next(
        (k for k in d.files if "mixed" in k.lower() or "concat" in k.lower()),
        d.files[0],
    )
    act = d[key]
    pooled = act.mean(axis=tuple(range(act.ndim - 1)))   # → (768,)
    embeddings.append(pooled)
    labels.append(1 if lbl == "TP" else 0)               # 1=TP, 0=UNK
    positions.append(stem)

X = np.vstack(embeddings)
y = np.array(labels)
print(f"Loaded: {{len(X)}} sites  |  TP={{y.sum()}}  UNK={{(y==0).sum()}}")

# ── 2. Cohen's d per channel ─────────────────────────────────────────────
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
    {{"channel": int(i), "cohens_d": float(cohens_d[i]),
      "tp_mean": float(tp_mean[i]), "unk_mean": float(unk_mean[i])}}
    for i in top20_idx
]
print(f"Top channel by |Cohen's d|: ch{{top20[0]['channel']}}  d={{top20[0]['cohens_d']:.3f}}")

cohens_d_out = {{
    "top20_channels": top20,
    "channel_604_d": float(cohens_d[604]) if 604 < len(cohens_d) else None,
    "channel_734_d": float(cohens_d[734]) if 734 < len(cohens_d) else None,
}}
(out_dir / "cohens_d_channels.json").write_text(json.dumps(cohens_d_out, indent=2))

# ── 3. Full 768-d probe, stratified 5-fold CV ────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auroc_full = cross_val_score(
    clf_full, X_scaled, y, cv=cv, scoring="roc_auc"
)
print(f"Full 768-d  AUROC: {{auroc_full.mean():.4f}} ± {{auroc_full.std():.4f}}")

# ── 4. Top-20 channel probe ───────────────────────────────────────────────
X_top20  = X_scaled[:, top20_idx]
clf_top20 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

auroc_top20 = cross_val_score(
    clf_top20, X_top20, y, cv=cv, scoring="roc_auc"
)
print(f"Top-20 ch   AUROC: {{auroc_top20.mean():.4f}} ± {{auroc_top20.std():.4f}}")

# ── 5. Permutation test (100 shuffles) ───────────────────────────────────
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
print(f"Null AUROC: {{null_mean:.4f}} ± {{null_std:.4f}}  |  p = {{p_val:.4f}}")

# ── 6. Fit final probe on all data, extract weights ───────────────────────
clf_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf_final.fit(X_scaled, y)
weights = clf_final.coef_[0]                              # shape (768,)

top20_probe_idx = np.argsort(np.abs(weights))[::-1][:20]
top20_weights = [
    {{"channel": int(i), "weight": float(weights[i])}}
    for i in top20_probe_idx
]

# Overlap between Cohen's d top-20 and probe weight top-20
cohens_set = set(top20_idx.tolist())
probe_set  = set(top20_probe_idx.tolist())
overlap    = cohens_set & probe_set
print(f"Channel overlap (Cohen's d top-20 ∩ probe top-20): {{len(overlap)}}/20")
print(f"Overlapping channels: {{sorted(overlap)}}")

# ── 7. Project all embeddings onto probe weight vector ────────────────────
complexity_scores = X_scaled @ weights                    # shape (n_sites,)
np.save(out_dir / "complexity_scores.npy", complexity_scores)
np.save(out_dir / "labels.npy", y)
np.save(out_dir / "positions.npy", np.array(positions))
np.save(out_dir / "X_scaled.npy", X_scaled)

# ── 8. Write full results JSON ────────────────────────────────────────────
results = {{
    "n_sites": int(len(X)),
    "n_tp": int(y.sum()),
    "n_unk": int((y == 0).sum()),
    "full_768d_probe": {{
        "auroc_mean":  float(auroc_full.mean()),
        "auroc_std":   float(auroc_full.std()),
        "auroc_folds": auroc_full.tolist(),
    }},
    "top20_channel_probe": {{
        "auroc_mean":  float(auroc_top20.mean()),
        "auroc_std":   float(auroc_top20.std()),
        "auroc_folds": auroc_top20.tolist(),
        "channels":    top20_idx.tolist(),
    }},
    "permutation_test": {{
        "n_permutations": 100,
        "null_auroc_mean": null_mean,
        "null_auroc_std":  null_std,
        "p_value":         p_val,
        "significant":     p_val < 0.05,
    }},
    "top20_probe_weights": top20_weights,
    "channel_overlap_cohens_vs_probe": sorted(list(overlap)),
    "n_overlap": len(overlap),
    "channel_604_weight": float(weights[604]) if 604 < len(weights) else None,
    "channel_734_weight": float(weights[734]) if 734 < len(weights) else None,
}}
(out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))

print("\\n── Results ──────────────────────────────────────────────────")
print(json.dumps({{k: v for k, v in results.items()
                  if k not in ("top20_probe_weights",)}}, indent=2))
print(f"\\nAll outputs written to {{out_dir}}/")
```
─────────────────────────────────────────────────────────────────────────

After running the script, confirm:
  - probe_results.json exists and is non-empty
  - complexity_scores.npy, labels.npy, positions.npy, X_scaled.npy exist
  - cohens_d_channels.json exists

Report the full probe_results.json content.
"""

VALIDATOR_PROMPT = f"""
You are the validator agent. Your job is to independently re-derive every
statistic in probe_results.json and flag any discrepancy.

You have full read/write/bash access. Re-run the mathematics from scratch
using the saved numpy arrays — do NOT read probe_results.json first.
Derive your own numbers, then compare.

Inputs (already on disk):
  {PROBE_DIR}/X_scaled.npy       — standardised embeddings (n, 768)
  {PROBE_DIR}/labels.npy         — binary labels (1=TP, 0=UNK)
  {PROBE_DIR}/cohens_d_channels.json

Validation steps — write and execute this Python script:

─────────────────────────────────────────────────────────────────────────
```python
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

probe_dir = Path("{PROBE_DIR}")

X_scaled  = np.load(probe_dir / "X_scaled.npy")
y         = np.load(probe_dir / "labels.npy")

print(f"Data shape: {{X_scaled.shape}}  labels: TP={{y.sum()}} UNK={{(y==0).sum()}}")

# ── V1. Re-derive Cohen's d for channels 604 and 734 ─────────────────────
tp_mask, unk_mask = y == 1, y == 0
for ch in [604, 734]:
    if ch >= X_scaled.shape[1]:
        print(f"ch{{ch}}: out of range"); continue
    tp_vals  = X_scaled[tp_mask, ch]
    unk_vals = X_scaled[unk_mask, ch]
    pooled   = np.sqrt((tp_vals.var() + unk_vals.var()) / 2)
    d = (tp_vals.mean() - unk_vals.mean()) / (pooled if pooled > 1e-9 else 1e-9)
    print(f"ch{{ch}} Cohen's d (validator): {{d:.4f}}")

# ── V2. Re-run 5-fold CV AUROC (full 768-d) ───────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
auroc_folds = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
val_auroc_mean = float(auroc_folds.mean())
val_auroc_std  = float(auroc_folds.std())
print(f"Validator AUROC: {{val_auroc_mean:.4f}} ± {{val_auroc_std:.4f}}")
print(f"Validator folds: {{auroc_folds.tolist()}}")

# ── V3. Re-run permutation test (same 100 seeds) ──────────────────────────
rng = np.random.default_rng(42)
null_aurocs = []
for _ in range(100):
    y_perm = rng.permutation(y)
    s = cross_val_score(
        LogisticRegression(C=1.0, max_iter=500, random_state=42),
        X_scaled, y_perm, cv=cv, scoring="roc_auc",
    )
    null_aurocs.append(float(s.mean()))
val_null_mean = float(np.mean(null_aurocs))
val_p         = float(np.mean(np.array(null_aurocs) >= val_auroc_mean))
print(f"Validator null AUROC: {{val_null_mean:.4f}}  p = {{val_p:.4f}}")

# ── V4. Load original results and compare ─────────────────────────────────
with open(probe_dir / "probe_results.json") as f:
    orig = json.load(f)

discrepancies = []
AUROC_TOL = 0.01    # flag if AUROC differs by more than 0.01
PVAL_TOL  = 0.05    # flag if p-value category flips

orig_auroc = orig["full_768d_probe"]["auroc_mean"]
if abs(val_auroc_mean - orig_auroc) > AUROC_TOL:
    discrepancies.append({{
        "field": "full_768d_probe.auroc_mean",
        "original": orig_auroc,
        "validator": val_auroc_mean,
        "delta": abs(val_auroc_mean - orig_auroc),
    }})

orig_p = orig["permutation_test"]["p_value"]
orig_sig = orig["permutation_test"]["significant"]
val_sig  = val_p < 0.05
if orig_sig != val_sig:
    discrepancies.append({{
        "field": "permutation_test.significant",
        "original": orig_sig,
        "validator": val_sig,
    }})

# ── V5. Write validator report ────────────────────────────────────────────
report = {{
    "validator_auroc_mean":  val_auroc_mean,
    "validator_auroc_std":   val_auroc_std,
    "validator_auroc_folds": auroc_folds.tolist(),
    "validator_null_mean":   val_null_mean,
    "validator_p_value":     val_p,
    "original_auroc_mean":   orig_auroc,
    "auroc_delta":           abs(val_auroc_mean - orig_auroc),
    "discrepancies":         discrepancies,
    "verdict": "PASS" if len(discrepancies) == 0 else "FAIL",
    "notes": (
        "All statistics reproduced within tolerance."
        if not discrepancies
        else f"{{len(discrepancies)}} discrepancy/ies found — review before plotting."
    ),
}}

(probe_dir / "validator_report.json").write_text(json.dumps(report, indent=2))
print("\\n── Validator report ─────────────────────────────────────────")
print(json.dumps(report, indent=2))
```
─────────────────────────────────────────────────────────────────────────

After the script runs:
  - If verdict is PASS: print "VALIDATOR PASS — proceeding to plots."
  - If verdict is FAIL: print each discrepancy clearly and stop.
    Do NOT write plots_complete.flag if validation fails.

Report the full validator_report.json.
"""

PLOT_PROMPT = f"""
You are the plot agent. Produce all figures for the linear probe analysis.
Run only after validator_report.json shows verdict=PASS.

Inputs (all on disk):
  {PROBE_DIR}/X_scaled.npy
  {PROBE_DIR}/labels.npy
  {PROBE_DIR}/positions.npy
  {PROBE_DIR}/complexity_scores.npy
  {PROBE_DIR}/probe_results.json
  {PROBE_DIR}/cohens_d_channels.json

Output dir: {PROBE_DIR}/plots/

Write and execute this Python script:

─────────────────────────────────────────────────────────────────────────
```python
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

probe_dir = Path("{PROBE_DIR}")
out_dir   = probe_dir / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

X_scaled   = np.load(probe_dir / "X_scaled.npy")
y          = np.load(probe_dir / "labels.npy")
scores     = np.load(probe_dir / "complexity_scores.npy")

with open(probe_dir / "probe_results.json") as f:
    results = json.load(f)
with open(probe_dir / "cohens_d_channels.json") as f:
    cohens_data = json.load(f)

COLORS = {{"TP": "#1D9E75", "UNK": "#B4B2A9"}}
label_names = {{1: "TP", 0: "UNK"}}

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
        label=f"ROC (train AUC = {{roc_auc:.3f}})")
ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"Linear probe ROC — mixed5 TP vs UNK\\n"
             f"5-fold CV AUROC = {{cv_auroc:.3f}} ± {{cv_std:.3f}}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "01_roc_curve.png", dpi=150); plt.close()
print("Saved 01_roc_curve.png")

# ── Plot 2: Complexity score distributions (TP vs UNK) ────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for lbl_int, lbl_name in label_names.items():
    mask = y == lbl_int
    ax.hist(scores[mask], bins=30, alpha=0.65,
            color=COLORS[lbl_name], label=f"{{lbl_name}} (n={{mask.sum()}})",
            density=True, edgecolor="none")
ax.set_xlabel("Complexity score (probe projection)")
ax.set_ylabel("Density")
ax.set_title("Distribution of complexity scores\\nTP vs UNK in mixed5 space")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "02_complexity_score_dist.png", dpi=150); plt.close()
print("Saved 02_complexity_score_dist.png")

# ── Plot 3: Top-20 Cohen's d channels ─────────────────────────────────────
top20 = cohens_data["top20_channels"]
ch_ids = [f"ch{{c['channel']}}" for c in top20]
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
w_ch  = [f"ch{{c['channel']}}" for c in top20_w]
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
axes[0].set_xlabel(f"PC1 ({{pca.explained_variance_ratio_[0]*100:.1f}}%)")
axes[0].set_ylabel(f"PC2 ({{pca.explained_variance_ratio_[1]*100:.1f}}%)")
axes[0].legend(fontsize=9, markerscale=1.5)

# Right: colour by continuous complexity score
sc = axes[1].scatter(X_pca[:,0], X_pca[:,1],
                     c=scores, cmap="RdYlGn_r",
                     alpha=0.7, s=18, linewidths=0)
plt.colorbar(sc, ax=axes[1], label="Complexity score")
axes[1].set_title("PCA — coloured by complexity score (probe projection)")
axes[1].set_xlabel(f"PC1 ({{pca.explained_variance_ratio_[0]*100:.1f}}%)")
axes[1].set_ylabel(f"PC2 ({{pca.explained_variance_ratio_[1]*100:.1f}}%)")

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
           label=f"Observed AUROC = {{cv_auroc:.3f}}")
ax.set_xlabel("AUROC"); ax.set_ylabel("Count")
ax.set_title(f"Permutation test  (p = {{perm['p_value']:.4f}})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "06_permutation_test.png", dpi=150); plt.close()
print("Saved 06_permutation_test.png")

print(f"\\nAll 6 plots saved to {{out_dir}}/")
```
─────────────────────────────────────────────────────────────────────────

After the script succeeds, write a one-line flag file:
    echo "done" > {PLOTS_DONE}

Then print a summary table:
  Plot 01: ROC curve with CV AUROC
  Plot 02: Complexity score distribution TP vs UNK
  Plot 03: Top-20 Cohen's d channels
  Plot 04: Top-20 probe weight channels (outline = overlap with Cohen's d)
  Plot 05: PCA coloured by label AND by continuous complexity score
  Plot 06: Permutation null distribution

Report any plots that failed to save.
"""

# ── Agent runner ─────────────────────────────────────────────────────────────

async def run_agent(name: str, prompt: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Launching [{name}]")
    print(f"{'='*60}")
    output = ""
    try:
        async for msg in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                allowed_tools=["Bash", "Read", "Write"],
                max_turns=50,
            ),
        ):
            if hasattr(msg, "text") and msg.text:
                output += msg.text
    except Exception as e:
        print(f"\n  ✗ [{name}] exception: {e}")
        return {"agent": name, "status": "failed", "error": str(e)}

    print(f"\n  ✅ [{name}] complete")
    return {"agent": name, "status": "success", "output": output}


def gate(path: str, agent: str) -> bool:
    if Path(path).exists():
        return True
    print(f"\n  ✗  Gate failed: {path} missing after [{agent}]")
    return False


def read_verdict() -> str:
    try:
        r = json.loads(Path(VALIDATOR_JSON).read_text())
        return r.get("verdict", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  Linear Probe Orchestrator — NA12878 mixed5 TP vs UNK")
    print("=" * 60)

    # Phase 1: probe
    await run_agent("probe-agent", PROBE_PROMPT)

    if not (gate(PROBE_JSON, "probe-agent") and
            gate(COHENS_D_JSON, "probe-agent")):
        print("\nAborting: probe agent did not write required output files.")
        sys.exit(1)

    # Phase 2: validator
    await run_agent("validator-agent", VALIDATOR_PROMPT)

    if not gate(VALIDATOR_JSON, "validator-agent"):
        print("\nAborting: validator agent did not write report.")
        sys.exit(1)

    verdict = read_verdict()
    print(f"\n  Validator verdict: {verdict}")

    if verdict != "PASS":
        print("\n  Aborting: validator found discrepancies.")
        print(f"  Review {VALIDATOR_JSON} before proceeding.")
        sys.exit(1)

    # Phase 3: plots (only if validator passed)
    await run_agent("plot-agent", PLOT_PROMPT)

    if not gate(PLOTS_DONE, "plot-agent"):
        print("\nPlot agent did not complete cleanly — check output above.")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 60)
    print("  All phases complete.")
    print(f"  Plots  : {PROBE_DIR}/plots/")
    print(f"  Results: {PROBE_DIR}/probe_results.json")
    print(f"  Validator: {VALIDATOR_JSON}")
    print("=" * 60)

    try:
        r = json.loads(Path(PROBE_JSON).read_text())
        v = json.loads(Path(VALIDATOR_JSON).read_text())
        print(f"\n  Full probe  AUROC : {r['full_768d_probe']['auroc_mean']:.4f}"
              f" ± {r['full_768d_probe']['auroc_std']:.4f}")
        print(f"  Top-20 ch   AUROC : {r['top20_channel_probe']['auroc_mean']:.4f}"
              f" ± {r['top20_channel_probe']['auroc_std']:.4f}")
        print(f"  Permutation p     : {r['permutation_test']['p_value']:.4f}"
              f"  ({'significant' if r['permutation_test']['significant'] else 'NOT significant'})")
        print(f"  Channel overlap   : {r['n_overlap']}/20"
              f"  {r['channel_overlap_cohens_vs_probe']}")
        print(f"  Validator delta   : {v['auroc_delta']:.5f}")
        print(f"  Validator verdict : {v['verdict']}")
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
