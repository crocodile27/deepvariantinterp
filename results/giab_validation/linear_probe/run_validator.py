import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

probe_dir = Path("results/giab_validation/linear_probe")

X_scaled  = np.load(probe_dir / "X_scaled.npy")
y         = np.load(probe_dir / "labels.npy")

print(f"Data shape: {X_scaled.shape}  labels: TP={y.sum()} UNK={(y==0).sum()}")

# ── V1. Re-derive Cohen's d for channels 604 and 734 ─────────────────────
tp_mask, unk_mask = y == 1, y == 0
for ch in [604, 734]:
    if ch >= X_scaled.shape[1]:
        print(f"ch{ch}: out of range"); continue
    tp_vals  = X_scaled[tp_mask, ch]
    unk_vals = X_scaled[unk_mask, ch]
    pooled   = np.sqrt((tp_vals.var() + unk_vals.var()) / 2)
    d = (tp_vals.mean() - unk_vals.mean()) / (pooled if pooled > 1e-9 else 1e-9)
    print(f"ch{ch} Cohen's d (validator): {d:.4f}")

# ── V2. Re-run 5-fold CV AUROC (full 768-d) ───────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
auroc_folds = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
val_auroc_mean = float(auroc_folds.mean())
val_auroc_std  = float(auroc_folds.std())
print(f"Validator AUROC: {val_auroc_mean:.4f} ± {val_auroc_std:.4f}")
print(f"Validator folds: {auroc_folds.tolist()}")

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
print(f"Validator null AUROC: {val_null_mean:.4f}  p = {val_p:.4f}")

# ── V4. Load original results and compare ─────────────────────────────────
with open(probe_dir / "probe_results.json") as f:
    orig = json.load(f)

discrepancies = []
AUROC_TOL = 0.01    # flag if AUROC differs by more than 0.01
PVAL_TOL  = 0.05    # flag if p-value category flips

orig_auroc = orig["full_768d_probe"]["auroc_mean"]
if abs(val_auroc_mean - orig_auroc) > AUROC_TOL:
    discrepancies.append({
        "field": "full_768d_probe.auroc_mean",
        "original": orig_auroc,
        "validator": val_auroc_mean,
        "delta": abs(val_auroc_mean - orig_auroc),
    })

orig_p = orig["permutation_test"]["p_value"]
orig_sig = orig["permutation_test"]["significant"]
val_sig  = val_p < 0.05
if orig_sig != val_sig:
    discrepancies.append({
        "field": "permutation_test.significant",
        "original": orig_sig,
        "validator": val_sig,
    })

# ── V5. Write validator report ────────────────────────────────────────────
report = {
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
        else f"{len(discrepancies)} discrepancy/ies found — review before plotting."
    ),
}

(probe_dir / "validator_report.json").write_text(json.dumps(report, indent=2))
print("\n── Validator report ─────────────────────────────────────────")
print(json.dumps(report, indent=2))
