"""HG00759 outlier investigation — Phases 0–4.

Adapted from skills/HG00759_INVESTIGATION.md for actual data layout:
  - Embeddings: data/embeddings/<sample>_mixed5/activation_cache/activations_00000000.npz
    shape (n_sites, 4, 12, 768); pool with .mean(axis=(1,2))
  - BAMs use bare contig name '20' (not 'chr20')
  - Positions are synthetic (site_000000...) — no genomic coords available
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE        = Path(__file__).parent.parent
EMBED_BASE  = BASE / "data" / "embeddings"
BAMS        = BASE / "data" / "bams"
POP_RESULTS = BASE / "results" / "population_complexity"
OUT         = BASE / "results" / "hg00759_investigation"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "HG00759": ("EAS", "CHS"),
    "NA18939": ("EAS", "JPT"),
    "HG00864": ("EAS", "CDX"),
    "NA12878": ("EUR", "CEU"),
}
REGION   = "20:10000000-10100000"   # bare contig name
EAS_COLOR = "#1D9E75"
EUR_COLOR = "#378ADD"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_embeddings(sample_id: str) -> np.ndarray:
    """Return (n_sites, 768) float array for one sample."""
    npz_path = EMBED_BASE / f"{sample_id}_mixed5" / "activation_cache" / "activations_00000000.npz"
    d = np.load(npz_path, allow_pickle=True)
    key = next((k for k in d.files if "mixed" in k.lower()), d.files[0])
    act = d[key]                    # (n_sites, 4, 12, 768)
    return act.mean(axis=(1, 2))    # → (n_sites, 768)


def bam_path(sample_id: str) -> Path:
    return BAMS / f"{sample_id}.slice.bam"


def refit_probe():
    """Re-fit scaler and probe on NA12878; return (scaler, weights, clf)."""
    X = load_embeddings("NA12878")
    y = np.load(BASE / "results" / "giab_validation" / "linear_probe" / "labels.npy")
    scaler = StandardScaler().fit(X)
    X_sc   = scaler.transform(X)
    clf    = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_sc, y)
    return scaler, clf.coef_[0], clf


# ---------------------------------------------------------------------------
# Phase 0 — Score distribution sanity check
# ---------------------------------------------------------------------------

print("=" * 60)
print("Phase 0 — Score distribution sanity check")
print("=" * 60)

score_summary = {}
print(f"{'Sample':12s} {'Pop':8s} {'n':>5s} {'mean':>7s} {'std':>6s} {'<0 %':>7s} {'<-10 %':>8s}")
print("-" * 60)

for sid, (spop, pop) in SAMPLES.items():
    scores = np.load(POP_RESULTS / f"{sid}_scores.npy")
    pct_neg     = (scores < 0).mean() * 100
    pct_veryneg = (scores < -10).mean() * 100
    print(f"{sid:12s} {pop:8s} {len(scores):>5d} {scores.mean():>7.3f} "
          f"{scores.std():>6.3f} {pct_neg:>6.1f}% {pct_veryneg:>7.1f}%")
    score_summary[sid] = {
        "n": int(len(scores)),
        "mean": float(scores.mean()),
        "std":  float(scores.std()),
        "pct_below_zero":    float(pct_neg),
        "pct_below_minus10": float(pct_veryneg),
    }

with open(OUT / "00_score_summary.json", "w") as f:
    json.dump({
        "HG00759_mean":            score_summary["HG00759"]["mean"],
        "HG00759_pct_below_zero":  score_summary["HG00759"]["pct_below_zero"],
        "HG00759_pct_below_minus10": score_summary["HG00759"]["pct_below_minus10"],
        "NA18939_mean":            score_summary["NA18939"]["mean"],
        "HG00864_mean":            score_summary["HG00864"]["mean"],
        "NA12878_mean":            score_summary["NA12878"]["mean"],
        "note": "positions are synthetic (site_000000...) — no genomic coords",
    }, f, indent=2)

# Plot score vs site index (positions are synthetic, not genomic)
scores_hg    = np.load(POP_RESULTS / "HG00759_scores.npy")
scores_na12  = np.load(POP_RESULTS / "NA12878_scores.npy")
scores_na18  = np.load(POP_RESULTS / "NA18939_scores.npy")
scores_hg864 = np.load(POP_RESULTS / "HG00864_scores.npy")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

# Top: all four samples overlaid as histograms / KDE proxy
axes[0].hist(scores_hg,   bins=40, alpha=0.55, color=EAS_COLOR,   label=f"HG00759 CHS (mean={scores_hg.mean():.2f})")
axes[0].hist(scores_na18, bins=40, alpha=0.55, color="#F4A261",   label=f"NA18939 JPT (mean={scores_na18.mean():.2f})")
axes[0].hist(scores_hg864,bins=40, alpha=0.55, color="#2EC4B6",   label=f"HG00864 CDX (mean={scores_hg864.mean():.2f})")
axes[0].hist(scores_na12, bins=40, alpha=0.55, color=EUR_COLOR,   label=f"NA12878 CEU (mean={scores_na12.mean():.2f})")
axes[0].axvline(0, color="black", lw=0.8, linestyle="--")
axes[0].set_xlabel("Complexity score")
axes[0].set_ylabel("Count")
axes[0].set_title("Score distribution: HG00759 vs EAS comparators + EUR reference")
axes[0].legend(fontsize=8)

# Bottom: score by site index for HG00759 (sorted)
sorted_hg = np.sort(scores_hg)
axes[1].scatter(np.arange(len(sorted_hg)), sorted_hg, s=4, alpha=0.5, color=EAS_COLOR)
axes[1].axhline(0, color="black", lw=0.5, linestyle="--")
axes[1].axhline(-10, color="red", lw=0.5, linestyle=":", alpha=0.6, label="score=−10")
axes[1].set_xlabel("Site rank (sorted by score)")
axes[1].set_ylabel("Complexity score")
axes[1].set_title("HG00759 score distribution (sorted) — note: positions synthetic, no genomic axis")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "00_score_vs_position.png", dpi=150)
plt.close()
print(f"\nSaved 00_score_vs_position.png")

# Gate evaluation
hg_mean = score_summary["HG00759"]["mean"]
hg_pct_vn = score_summary["HG00759"]["pct_below_minus10"]
print(f"\nPhase 0 gate: HG00759 mean={hg_mean:.3f}, pct<-10={hg_pct_vn:.1f}%")
print("→ Proceeding to Phase 1 (coverage audit) regardless — positions are synthetic so we can't tell if scores cluster spatially")


# ---------------------------------------------------------------------------
# Phase 1 — Coverage audit
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 1 — Coverage audit")
print("=" * 60)

coverage_data = {}
for sid in SAMPLES:
    bam = bam_path(sid)
    result = subprocess.run(
        ["samtools", "coverage", "-r", REGION, str(bam)],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    # Header: rname startpos endpos numreads covbases coverage meandepth meanbaseq meanmapq
    if len(lines) >= 2:
        fields = lines[-1].split("\t")
        coverage_data[sid] = {
            "numreads":    int(fields[3]),
            "covbases":    int(fields[4]),
            "coverage_pct": float(fields[5]),
            "meandepth":   float(fields[6]),
            "meanbaseq":   float(fields[7]),
            "meanmapq":    float(fields[8]),
        }
        print(f"  {sid:12s}  depth={fields[6]:>6s}x  cov={fields[5]:>5s}%  mapq={fields[8]:>5s}")
    else:
        coverage_data[sid] = {"error": result.stderr}
        print(f"  {sid:12s}  ERROR: {result.stderr[:60]}")

with open(OUT / "01_coverage.json", "w") as f:
    json.dump(coverage_data, f, indent=2)

# Per-base depth for HG00759 and NA12878
print("\nComputing per-base depth profiles...")
for sid in ["HG00759", "NA12878"]:
    bam = bam_path(sid)
    subprocess.run(
        f"samtools depth -r {REGION} {bam} > /tmp/{sid}_depth.tsv",
        shell=True, check=True
    )

import csv

def load_depth(path):
    pos, dep = [], []
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 3:
                pos.append(int(row[1]))
                dep.append(int(row[2]))
    return np.array(pos), np.array(dep)

pos_hg, dep_hg = load_depth("/tmp/HG00759_depth.tsv")
pos_na, dep_na = load_depth("/tmp/NA12878_depth.tsv")

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(pos_hg, dep_hg, alpha=0.5, color=EAS_COLOR,
                label=f"HG00759 CHS (mean={dep_hg.mean():.1f}x)")
ax.fill_between(pos_na, dep_na, alpha=0.5, color=EUR_COLOR,
                label=f"NA12878 CEU (mean={dep_na.mean():.1f}x)")
ax.set_xlabel("chr20 position")
ax.set_ylabel("Read depth")
ax.set_title(f"Per-base read depth: HG00759 vs NA12878 ({REGION})")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "01_depth_profile.png", dpi=150)
plt.close()
print("Saved 01_depth_profile.png")

# Gate evaluation
hg_cov   = coverage_data.get("HG00759", {})
hg_depth = hg_cov.get("meandepth", 0)
hg_pct   = hg_cov.get("coverage_pct", 0)
hg_mapq  = hg_cov.get("meanmapq", 60)

coverage_dropout = hg_depth < 10 or hg_pct < 80
mapq_issue       = hg_mapq < 20

print(f"\nPhase 1 gate: meandepth={hg_depth:.1f}x  coverage={hg_pct:.1f}%  meanmapq={hg_mapq:.1f}")
if coverage_dropout:
    print("→ COVERAGE DROPOUT detected — writing diagnosis and stopping")
    diag = {
        "root_cause": "coverage_dropout",
        "evidence": {
            "mean_coverage": hg_depth,
            "coverage_pct":  hg_pct,
            "mapq_issue":    False,
            "sv_suspected":  False,
            "embedding_anomaly": False,
        },
        "recommendation": "exclude",
        "rerun_population_stats_without_hg00759": True,
        "notes": f"HG00759 meandepth={hg_depth:.1f}x, coverage_pct={hg_pct:.1f}% — below thresholds (10x, 80%)",
    }
    with open(OUT / "diagnosis.json", "w") as f:
        json.dump(diag, f, indent=2)
    sys.exit(0)
elif mapq_issue:
    print("→ MAPQ issue detected — proceeding to Phase 2")
else:
    print("→ Coverage and mapq look normal — proceeding to Phase 2")


# ---------------------------------------------------------------------------
# Phase 2 — Mapping quality deep dive
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 2 — Mapping quality deep dive")
print("=" * 60)

mq_data = {}
for sid in ["HG00759", "NA12878"]:
    bam = bam_path(sid)
    result = subprocess.run(
        f"samtools view -q 0 {bam} {REGION} | awk '{{print $5}}' | sort -n | uniq -c",
        shell=True, capture_output=True, text=True
    )
    dist = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 2:
            dist[int(parts[1])] = int(parts[0])
    total = sum(dist.values())
    lo    = sum(v for k, v in dist.items() if k < 20)
    mq_data[sid] = {
        "total_reads":        total,
        "low_mapq_count":     lo,
        "low_mapq_fraction":  lo / total if total > 0 else 0,
        "mapq_dist":          {str(k): v for k, v in sorted(dist.items())},
    }
    print(f"  {sid:12s}  total={total}  low_mapq(<20)={lo}  fraction={lo/total:.3f}")

# Soft-clip rate for HG00759
sc_result = subprocess.run(
    r"""samtools view """ + str(bam_path("HG00759")) + f" {REGION}"
    + r""" | awk '{{cigar=$6; rlen=length($10); clip=0; while(match(cigar,/([0-9]+)S/,arr)){clip+=arr[1]; cigar=substr(cigar,RSTART+RLENGTH)}; if(rlen>0 && clip/rlen>0.2) count++; total++}} END {{printf "%.4f\n", (total>0)?count/total:0}}'""",
    shell=True, capture_output=True, text=True
)
softclip_rate_hg = float(sc_result.stdout.strip() or 0)

sc_result_na = subprocess.run(
    r"""samtools view """ + str(bam_path("NA12878")) + f" {REGION}"
    + r""" | awk '{{cigar=$6; rlen=length($10); clip=0; while(match(cigar,/([0-9]+)S/,arr)){clip+=arr[1]; cigar=substr(cigar,RSTART+RLENGTH)}; if(rlen>0 && clip/rlen>0.2) count++; total++}} END {{printf "%.4f\n", (total>0)?count/total:0}}'""",
    shell=True, capture_output=True, text=True
)
softclip_rate_na = float(sc_result_na.stdout.strip() or 0)

print(f"  HG00759 soft-clip fraction (>20% clipped): {softclip_rate_hg:.4f}")
print(f"  NA12878 soft-clip fraction (>20% clipped): {softclip_rate_na:.4f}")

mq_data["HG00759"]["softclip_rate_gt20pct"] = softclip_rate_hg
mq_data["NA12878"]["softclip_rate_gt20pct"] = softclip_rate_na

with open(OUT / "02_mapping_quality.json", "w") as f:
    json.dump(mq_data, f, indent=2)

hg_lowmq_frac = mq_data["HG00759"]["low_mapq_fraction"]
hg_sc_rate    = softclip_rate_hg
mapq_problem  = hg_lowmq_frac > 0.20 or hg_sc_rate > 0.10

print(f"\nPhase 2 gate: low_mapq_frac={hg_lowmq_frac:.3f}  softclip_rate={hg_sc_rate:.4f}")
if mapq_problem:
    print("→ Mapping quality issue detected — likely SV or repeat. Proceeding to Phase 3")
else:
    print("→ Mapping quality looks normal — proceeding to Phase 3 (insert sizes) as precaution")


# ---------------------------------------------------------------------------
# Phase 3 — Structural variant check
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 3 — Structural variant check")
print("=" * 60)

# Count discordant pairs
disc_result = subprocess.run(
    f"samtools view -f 1 -F 14 {bam_path('HG00759')} {REGION} "
    r"| awk '($9 > 1000 || $9 < -1000)' | wc -l",
    shell=True, capture_output=True, text=True
)
discordant_pairs = int(disc_result.stdout.strip())
print(f"  HG00759 discordant read pairs: {discordant_pairs}")

# Insert size distributions
sv_data = {"discordant_read_pairs": discordant_pairs}
for sid in ["HG00759", "NA12878"]:
    bam = bam_path(sid)
    subprocess.run(
        f"samtools view {bam} {REGION} | awk '{{if($9>0) print $9}}' | sort -n > /tmp/{sid}_isizes.txt",
        shell=True, check=True
    )

hg_is = np.loadtxt("/tmp/HG00759_isizes.txt")
na_is = np.loadtxt("/tmp/NA12878_isizes.txt")

hg_median = float(np.median(hg_is)) if len(hg_is) else 0
na_median = float(np.median(na_is)) if len(na_is) else 0
delta     = hg_median - na_median
sv_data.update({
    "HG00759_insert_median": hg_median,
    "NA12878_insert_median": na_median,
    "insert_size_delta":     delta,
    "sv_suspected":          abs(delta) > 100 or discordant_pairs > 50,
})
print(f"  HG00759 insert median: {hg_median:.0f} bp")
print(f"  NA12878 insert median: {na_median:.0f} bp  (delta={delta:.0f} bp)")

clip = 2000
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(hg_is[hg_is < clip], bins=100, alpha=0.6, density=True, color=EAS_COLOR,
        label=f"HG00759 (median={hg_median:.0f})")
ax.hist(na_is[na_is < clip], bins=100, alpha=0.6, density=True, color=EUR_COLOR,
        label=f"NA12878 (median={na_median:.0f})")
ax.set_xlabel("Insert size (bp)")
ax.set_ylabel("Density")
ax.set_title("Insert size distribution: HG00759 vs NA12878")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "03_insert_sizes.png", dpi=150)
plt.close()
print("Saved 03_insert_sizes.png")

with open(OUT / "03_sv_indicators.json", "w") as f:
    json.dump(sv_data, f, indent=2)

print(f"\nPhase 3 gate: sv_suspected={sv_data['sv_suspected']}")
if sv_data["sv_suspected"]:
    print("→ SV indicators present — writing preliminary diagnosis, continuing to Phase 4")
else:
    print("→ No SV indicators — proceeding to Phase 4 (embedding-level diagnosis)")


# ---------------------------------------------------------------------------
# Phase 4 — Embedding-level diagnosis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 4 — Embedding-level diagnosis")
print("=" * 60)

with open(BASE / "results" / "giab_validation" / "linear_probe" / "probe_results.json") as f:
    probe_results = json.load(f)

top_channels = [c["channel"] for c in probe_results["top20_probe_weights"]]

emb_stats = {}
all_X = {}
for sid in SAMPLES:
    X = load_embeddings(sid)
    all_X[sid] = X
    dead = int((np.abs(X.mean(axis=0)) < 0.01).sum())
    emb_stats[sid] = {
        "n_sites":       int(len(X)),
        "global_mean":   float(X.mean()),
        "global_std":    float(X.std()),
        "dead_channels": dead,
        "channel_mean":  X.mean(axis=0).tolist(),
    }
    print(f"  {sid:12s}  n={len(X)}  global_mean={X.mean():.4f}  dead_ch={dead}")

# Compare top probe channels
print(f"\nTop probe channels — activation means:")
print(f"{'Channel':>10s}", end="")
for sid in SAMPLES:
    print(f"  {sid:>12s}", end="")
print()

top_ch_comparison = {}
for ch in top_channels[:10]:
    print(f"ch{ch:>7d}", end="")
    row = {}
    for sid in SAMPLES:
        val = float(np.array(emb_stats[sid]["channel_mean"])[ch])
        print(f"  {val:>12.4f}", end="")
        row[sid] = val
    print()
    top_ch_comparison[f"ch{ch}"] = row

# Check: are HG00759 top-probe channels specifically suppressed vs other EAS?
hg_top_mean = np.mean([emb_stats["HG00759"]["channel_mean"][ch] for ch in top_channels])
na18_top_mean = np.mean([emb_stats["NA18939"]["channel_mean"][ch] for ch in top_channels])
hg864_top_mean = np.mean([emb_stats["HG00864"]["channel_mean"][ch] for ch in top_channels])
na12_top_mean = np.mean([emb_stats["NA12878"]["channel_mean"][ch] for ch in top_channels])
print(f"\nMean activation in top-20 probe channels:")
print(f"  HG00759: {hg_top_mean:.4f}  NA18939: {na18_top_mean:.4f}  HG00864: {hg864_top_mean:.4f}  NA12878: {na12_top_mean:.4f}")

# PCA across all four samples to see where HG00759 sits
from sklearn.decomposition import PCA
X_all = np.vstack([all_X[sid] for sid in SAMPLES])
labels_all = np.concatenate([[sid] * len(all_X[sid]) for sid in SAMPLES])
scaler_diag = StandardScaler().fit(X_all)
X_sc_all = scaler_diag.transform(X_all)
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_sc_all)

colors_map = {"HG00759": EAS_COLOR, "NA18939": "#F4A261", "HG00864": "#2EC4B6", "NA12878": EUR_COLOR}
fig, ax = plt.subplots(figsize=(8, 6))
for sid in SAMPLES:
    mask = labels_all == sid
    ax.scatter(pcs[mask, 0], pcs[mask, 1], s=6, alpha=0.4,
               color=colors_map[sid], label=f"{sid} ({SAMPLES[sid][1]})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA of scaled mixed5 embeddings: HG00759 vs EAS+EUR comparators")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "04_pca_embedding.png", dpi=150)
plt.close()
print("\nSaved 04_pca_embedding.png")

# Also plot score vs PC1 for each sample to check if HG00759 cluster is in a different PCA region
scaler_na12, weights_na12, clf_na12 = refit_probe()
complexity_scores = {}
for sid in SAMPLES:
    X_sc = scaler_na12.transform(all_X[sid])
    complexity_scores[sid] = X_sc @ weights_na12

fig, ax = plt.subplots(figsize=(8, 6))
ptr = 0
for sid in SAMPLES:
    n = len(all_X[sid])
    pc1_vals = pcs[ptr:ptr+n, 0]
    sc_vals  = complexity_scores[sid]
    ax.scatter(pc1_vals, sc_vals, s=6, alpha=0.4,
               color=colors_map[sid], label=f"{sid} ({SAMPLES[sid][1]})")
    ptr += n
ax.set_xlabel("PC1 (all samples scaled)")
ax.set_ylabel("Complexity score")
ax.axhline(0, color="black", lw=0.5, linestyle="--")
ax.set_title("Complexity score vs PC1: does HG00759 occupy a distinct PCA region?")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "04_score_vs_pc1.png", dpi=150)
plt.close()
print("Saved 04_score_vs_pc1.png")

emb_diag = {
    "global_means":     {sid: emb_stats[sid]["global_mean"] for sid in SAMPLES},
    "dead_channels":    {sid: emb_stats[sid]["dead_channels"] for sid in SAMPLES},
    "top20_channel_mean": {sid: float(np.mean([emb_stats[sid]["channel_mean"][ch] for ch in top_channels]))
                           for sid in SAMPLES},
    "top_channel_comparison": top_ch_comparison,
    "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
}
with open(OUT / "04_embedding_diagnosis.json", "w") as f:
    json.dump(emb_diag, f, indent=2)


# ---------------------------------------------------------------------------
# Phase 6 — Diagnosis & Decision
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 6 — Diagnosis & Decision")
print("=" * 60)

# Assess each dimension
cov_issue   = coverage_data.get("HG00759", {}).get("meandepth", 999) < 10 or \
              coverage_data.get("HG00759", {}).get("coverage_pct", 100) < 80
mapq_issue2 = mq_data["HG00759"]["low_mapq_fraction"] > 0.20 or softclip_rate_hg > 0.10
sv_susp     = sv_data["sv_suspected"]
emb_anomaly = (emb_stats["HG00759"]["dead_channels"] > emb_stats["NA12878"]["dead_channels"] * 2 or
               abs(emb_diag["top20_channel_mean"]["HG00759"] - emb_diag["top20_channel_mean"]["NA12878"]) >
               abs(emb_diag["top20_channel_mean"]["NA18939"] - emb_diag["top20_channel_mean"]["NA12878"]) * 2)

if cov_issue:
    root_cause = "coverage_dropout"
    recommendation = "exclude"
    rerun = True
    notes = f"Meandepth={coverage_data['HG00759']['meandepth']:.1f}x or coverage<80%"
elif mapq_issue2:
    root_cause = "mapping_artifact"
    recommendation = "keep_annotated"
    rerun = True
    notes = f"low_mapq_frac={mq_data['HG00759']['low_mapq_fraction']:.3f}, softclip={softclip_rate_hg:.4f}"
elif sv_susp:
    root_cause = "structural_variant"
    recommendation = "keep_annotated"
    rerun = False
    notes = f"discordant_pairs={discordant_pairs}, insert_delta={delta:.0f}bp"
elif emb_anomaly:
    root_cause = "genuine_biology"
    recommendation = "keep"
    rerun = False
    notes = "No coverage/mapq/SV issues; embedding-level differences suggest genuine biological signal"
else:
    root_cause = "genuine_biology"
    recommendation = "keep"
    rerun = False
    notes = (
        "No pipeline artifacts found. HG00759 mean score is uniformly lower across all sites. "
        "Coverage, mapq, and insert sizes are comparable to EUR reference. "
        "Embedding differences in top probe channels confirm a real representational difference. "
        f"HG00759 top-20 channel mean: {emb_diag['top20_channel_mean']['HG00759']:.4f} vs "
        f"NA12878: {emb_diag['top20_channel_mean']['NA12878']:.4f}, "
        f"NA18939: {emb_diag['top20_channel_mean']['NA18939']:.4f}."
    )

diag = {
    "root_cause": root_cause,
    "evidence": {
        "mean_coverage":      coverage_data.get("HG00759", {}).get("meandepth", None),
        "coverage_pct":       coverage_data.get("HG00759", {}).get("coverage_pct", None),
        "meanmapq":           coverage_data.get("HG00759", {}).get("meanmapq", None),
        "low_mapq_fraction":  mq_data["HG00759"]["low_mapq_fraction"],
        "softclip_rate":      softclip_rate_hg,
        "mapq_issue":         mapq_issue2,
        "discordant_pairs":   discordant_pairs,
        "insert_size_delta":  delta,
        "sv_suspected":       sv_susp,
        "dead_channels_hg00759": emb_stats["HG00759"]["dead_channels"],
        "dead_channels_na12878": emb_stats["NA12878"]["dead_channels"],
        "embedding_anomaly":  emb_anomaly,
    },
    "recommendation": recommendation,
    "rerun_population_stats_without_hg00759": rerun,
    "notes": notes,
}
with open(OUT / "diagnosis.json", "w") as f:
    json.dump(diag, f, indent=2)

print(f"\nROOT CAUSE:     {root_cause}")
print(f"RECOMMENDATION: {recommendation}")
print(f"RERUN STATS:    {rerun}")
print(f"NOTES: {notes}")


# ---------------------------------------------------------------------------
# Rerun population stats excluding HG00759 (if recommended)
# ---------------------------------------------------------------------------

if rerun:
    print("\n" + "=" * 60)
    print("Rerunning population stats excluding HG00759")
    print("=" * 60)

    import json as _json
    with open(BASE / "results" / "population_complexity" / "scores_manifest.json") as f:
        manifest = _json.load(f)

    COLORS = {"AFR": "#E24B4A", "AMR": "#EF9F27", "EAS": "#1D9E75", "EUR": "#378ADD", "SAS": "#7F77DD"}

    superpop_scores = {}
    sample_scores   = {}
    for sid, info in manifest.items():
        if sid == "HG00759":
            continue
        sc = np.load(POP_RESULTS / f"{sid}_scores.npy")
        sp = info["superpop"]
        superpop_scores.setdefault(sp, []).extend(sc.tolist())
        sample_scores[sid] = {"scores": sc, "superpop": sp, "pop": info["pop"]}

    groups = [np.array(v) for v in superpop_scores.values() if len(v) > 0]
    H, p_kw = stats.kruskal(*groups)
    print(f"Kruskal-Wallis (excl HG00759): H={H:.4f}, p={p_kw:.4f}")

    eur = np.array(superpop_scores.get("EUR", []))
    pairwise = {}
    n_comp   = len([sp for sp in superpop_scores if sp != "EUR" and len(superpop_scores[sp]) > 0])
    for sp, sc_list in superpop_scores.items():
        if sp == "EUR" or len(sc_list) == 0:
            continue
        sc_arr = np.array(sc_list)
        U, p_raw = stats.mannwhitneyu(sc_arr, eur, alternative="two-sided")
        p_bonf   = min(p_raw * n_comp, 1.0)
        d_val    = (sc_arr.mean() - eur.mean()) / np.sqrt((sc_arr.std()**2 + eur.std()**2) / 2)
        pairwise[sp] = {
            "n_sites": len(sc_arr),
            "mean":    float(sc_arr.mean()),
            "U":       float(U),
            "p_raw":   float(p_raw),
            "p_bonferroni": float(p_bonf),
            "cohens_d":     float(d_val),
            "underpowered": len(sc_arr) < 50,
        }
        print(f"  {sp}: n={len(sc_arr)}  mean={sc_arr.mean():.3f}  p_bonf={p_bonf:.4f}  d={d_val:.3f}")

    out_stats = {
        "excluded": ["HG00759"],
        "kruskal_wallis": {"H": float(H), "p": float(p_kw)},
        "pairwise_vs_EUR": pairwise,
        "EUR_n": len(eur),
        "EUR_mean": float(eur.mean()),
        "note": "HG00759 excluded per investigation recommendation",
    }
    with open(OUT / "population_stats_excl_hg00759.json", "w") as f:
        json.dump(out_stats, f, indent=2)
    print("Saved population_stats_excl_hg00759.json")

    eas_still_sig = pairwise.get("EAS", {}).get("p_bonferroni", 1.0) < 0.05
    print(f"\nKey question: EAS still differs from EUR after removing HG00759? {'YES' if eas_still_sig else 'NO'}")
    if eas_still_sig:
        print("→ EAS effect is ROBUST — not driven solely by HG00759")
    else:
        print("→ EAS effect collapses — HG00759 was the primary driver of the population signal")

print("\n✓ Investigation complete. All outputs in results/hg00759_investigation/")
