# HG00759 Outlier Investigation
## Claude Code Orchestration Document

> Read this entire document before writing any code.
> Work through phases in order. Each phase produces a finding
> that determines whether subsequent phases are needed.
> Stop early if the root cause is identified conclusively.

---

## Context

HG00759 (CHS/EAS) has a mean complexity score of −2.332, the lowest of
all 13 samples. All other EAS samples (NA18939 −0.380, HG00864 −1.227)
and all other superpopulations sit near zero. This drives the only
statistically significant pairwise difference in the population complexity
analysis (EAS vs EUR, Bonferroni p=0.020, d=−0.135).

The complexity score is a projection of mixed5 embeddings onto the linear
probe weight vector trained on NA12878 TP vs UNK. Low score = embedding
resembles GIAB uncertain/uncallable sites.

**Scientific question:** Is HG00759's low score a pipeline artifact,
a coverage issue, a structural variant, or genuine biological signal?

**Decision rule:**
- If artifact or coverage → exclude from population analysis, note as
  QC failure, re-run population stats without HG00759
- If structural variant → keep but annotate separately, investigate
  whether the SV is population-informative
- If genuine biology → keep, expand to full chr20 to confirm

---

## Repo State

```
data/bams/HG00759.slice.bam                    ← original slice (pre-reheader)
data/embeddings/grch37/HG00759/activations/    ← 384 .npz files
data/embeddings/grch37/HG00759/intermediate/
  make_examples.tfrecord-00000-of-00001.gz     ← pileup images
logs/HG00759_deepvariant.log                   ← make_examples log
logs/HG00759_samtools.log                      ← samtools log
results/population_complexity/
  HG00759_scores.npy                           ← (384,) complexity scores
  HG00759_positions.npy                        ← (384,) position strings
```

Reference samples for comparison:
```
data/embeddings/grch37/NA12878/activations/    ← EUR, mean score ≈ 0.0
data/embeddings/grch37/NA18939/activations/    ← EAS/JPT, mean score −0.380
data/embeddings/grch37/HG00864/activations/    ← EAS/CDX, mean score −1.227
```

---

## Phase 0 — Score Distribution Sanity Check

Before running any new tools, characterise exactly what the score
distribution looks like for HG00759 vs comparators.

```python
import numpy as np
from pathlib import Path

samples = {
    "HG00759": ("EAS", "CHS"),
    "NA18939": ("EAS", "JPT"),
    "HG00864": ("EAS", "CDX"),
    "NA12878": ("EUR", "CEU"),
}

base = Path("results/population_complexity")
print(f"{'Sample':12s} {'Superpop':8s} {'n':>5s} {'mean':>7s} "
      f"{'std':>6s} {'<0 %':>7s} {'<-10 %':>8s}")
print("-" * 60)

for sid, (spop, pop) in samples.items():
    scores = np.load(base / f"{sid}_scores.npy")
    pct_neg    = (scores < 0).mean() * 100
    pct_veryneg = (scores < -10).mean() * 100
    print(f"{sid:12s} {pop:8s} {len(scores):>5d} {scores.mean():>7.3f} "
          f"{scores.std():>6.3f} {pct_neg:>6.1f}% {pct_veryneg:>7.1f}%")
```

**Write finding to** `results/hg00759_investigation/00_score_summary.json`:
```json
{
  "HG00759_mean": ...,
  "HG00759_pct_below_zero": ...,
  "HG00759_pct_below_minus10": ...,
  "NA18939_mean": ...,
  "HG00864_mean": ...,
  "NA12878_mean": ...
}
```

**Key question answered here:** Is HG00759 uniformly shifted (all sites
lower) or does it have a specific subregion with very negative scores?
Plot score vs genomic position:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

positions = np.load(base / "HG00759_positions.npy", allow_pickle=True)
scores    = np.load(base / "HG00759_scores.npy")

# Parse position from filename e.g. "chr20_10045231" → 10045231
coords = np.array([int(p.split("_")[1]) for p in positions])
order  = np.argsort(coords)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Top: HG00759 score vs position
axes[0].scatter(coords[order], scores[order],
                s=6, alpha=0.5, color="#1D9E75", label="HG00759")
axes[0].axhline(0, color="black", lw=0.5, linestyle="--")
axes[0].axhline(-10, color="red", lw=0.5, linestyle=":", alpha=0.5,
                label="score=−10")
axes[0].set_ylabel("Complexity score")
axes[0].set_title("HG00759 complexity score vs genomic position")
axes[0].legend(fontsize=8)

# Bottom: overlay NA12878 for comparison
na12878_pos   = np.load(base / "NA12878_positions.npy", allow_pickle=True)
na12878_scores = np.load(base / "NA12878_scores.npy")
na12878_coords = np.array([int(p.split("_")[1]) for p in na12878_pos])
na12878_order  = np.argsort(na12878_coords)

axes[1].scatter(na12878_coords[na12878_order], na12878_scores[na12878_order],
                s=6, alpha=0.5, color="#378ADD", label="NA12878 (EUR)")
axes[1].axhline(0, color="black", lw=0.5, linestyle="--")
axes[1].set_xlabel("chr20 position"); axes[1].set_ylabel("Complexity score")
axes[1].set_title("NA12878 complexity score vs genomic position (reference)")
axes[1].legend(fontsize=8)

plt.tight_layout()
out = Path("results/hg00759_investigation")
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "00_score_vs_position.png", dpi=150)
plt.close()
print("Saved 00_score_vs_position.png")
```

**Gate:** If scores are uniformly negative across all positions → likely
coverage or model issue (proceed to Phase 1). If scores cluster in a
specific subregion → likely structural variant (jump to Phase 3).

---

## Phase 1 — Coverage Audit

Check read depth at the region for HG00759 vs comparator samples.

```bash
REGION="chr20:10000000-10100000"
REF="quickstart-testdata/ucsc.hg19.chr20.unittest.fasta"

# For each sample, compute mean depth across the region
# Use the reheadered BAM (in /tmp after pipeline run) or re-reheader now

for SAMPLE in HG00759 NA18939 HG00864 NA12878; do
  BAM="data/bams/${SAMPLE}.slice.bam"
  # Reheader on the fly
  samtools view -H "$BAM" \
    | sed 's/SN:20\t/SN:chr20\t/' \
    | samtools reheader - "$BAM" > /tmp/${SAMPLE}_rh.bam
  samtools index /tmp/${SAMPLE}_rh.bam

  # Compute depth stats
  samtools coverage \
    -r "$REGION" \
    /tmp/${SAMPLE}_rh.bam \
    | tail -1
done
```

`samtools coverage` outputs: rname, startpos, endpos, numreads, covbases,
coverage, meandepth, meanbaseq, meanmapq

**Write to** `results/hg00759_investigation/01_coverage.json`:
```json
{
  "HG00759": {"meandepth": ..., "coverage_pct": ..., "meanmapq": ...},
  "NA18939": {"meandepth": ..., "coverage_pct": ..., "meanmapq": ...},
  "HG00864": {"meandepth": ..., "coverage_pct": ..., "meanmapq": ...},
  "NA12878": {"meandepth": ..., "coverage_pct": ..., "meanmapq": ...}
}
```

Also plot per-base depth across the region for HG00759 vs NA12878:

```bash
samtools depth -r "$REGION" /tmp/HG00759_rh.bam > /tmp/HG00759_depth.tsv
samtools depth -r "$REGION" /tmp/NA12878_rh.bam > /tmp/NA12878_depth.tsv
```

```python
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

hg = pd.read_csv("/tmp/HG00759_depth.tsv", sep="\t",
                 names=["chrom","pos","depth"])
na = pd.read_csv("/tmp/NA12878_depth.tsv", sep="\t",
                 names=["chrom","pos","depth"])

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(hg["pos"], hg["depth"], alpha=0.5,
                color="#1D9E75", label=f"HG00759 (mean={hg['depth'].mean():.1f}x)")
ax.fill_between(na["pos"], na["depth"], alpha=0.5,
                color="#378ADD", label=f"NA12878 (mean={na['depth'].mean():.1f}x)")
ax.set_xlabel("chr20 position")
ax.set_ylabel("Read depth")
ax.set_title("Per-base read depth: HG00759 vs NA12878")
ax.legend()
plt.tight_layout()
plt.savefig("results/hg00759_investigation/01_depth_profile.png", dpi=150)
plt.close()
print("Saved 01_depth_profile.png")
```

**Decision logic:**
- If HG00759 meandepth < 10x OR coverage_pct < 80% → **coverage dropout**
  is the likely cause. Write `root_cause: "coverage_dropout"` to
  `results/hg00759_investigation/diagnosis.json` and stop.
- If HG00759 meanmapq < 20 (much lower than others) → **mapping quality
  issue**, possibly a structural variant causing multi-mapping reads.
  Proceed to Phase 2.
- If coverage and mapq are comparable to NA12878 → proceed to Phase 2.

---

## Phase 2 — Mapping Quality Deep Dive

If coverage is fine but mapq is low, reads are mapping ambiguously.
This points to a repeat or SV.

```bash
# Distribution of mapping qualities for HG00759 in the region
samtools view -q 0 /tmp/HG00759_rh.bam chr20:10000000-10100000 \
  | awk '{print $5}' \
  | sort -n | uniq -c > /tmp/HG00759_mapq_dist.tsv

samtools view -q 0 /tmp/NA12878_rh.bam chr20:10000000-10100000 \
  | awk '{print $5}' \
  | sort -n | uniq -c > /tmp/NA12878_mapq_dist.tsv

# Fraction of reads with mapq < 20 (multi-mapping)
echo "HG00759 low-mapq fraction:"
awk 'BEGIN{lo=0;tot=0} {tot+=$1; if($2<20) lo+=$1} END{printf "%.3f\n", lo/tot}' \
  /tmp/HG00759_mapq_dist.tsv

echo "NA12878 low-mapq fraction:"
awk 'BEGIN{lo=0;tot=0} {tot+=$1; if($2<20) lo+=$1} END{printf "%.3f\n", lo/tot}' \
  /tmp/NA12878_mapq_dist.tsv
```

Also check soft-clipping rate (reads with large soft clips suggest
sequence that doesn't match the reference — SV signature):

```bash
# Count reads with >20% of bases soft-clipped
samtools view /tmp/HG00759_rh.bam chr20:10000000-10100000 \
  | awk '{
    cigar=$6; rlen=length($10);
    clip=0;
    while(match(cigar, /([0-9]+)S/, arr)) {
      clip+=arr[1]; cigar=substr(cigar, RSTART+RLENGTH)
    }
    if(clip/rlen > 0.2) count++; total++
  } END {printf "HG00759 high-clip fraction: %.3f\n", count/total}'
```

**Write to** `results/hg00759_investigation/02_mapping_quality.json`.

**Decision logic:**
- High low-mapq fraction (>20%) or high soft-clip rate (>10%) → likely
  **structural variant or repeat expansion**. Proceed to Phase 3.
- Normal mapq and soft-clip rate → proceed to Phase 4 (embedding-level
  diagnosis).

---

## Phase 3 — Structural Variant Check

If Phase 1 or 2 points to an SV, characterise it.

```bash
# Check for discordant read pairs (SV signature)
samtools view -f 1 -F 14 /tmp/HG00759_rh.bam chr20:10000000-10100000 \
  | awk '($9 > 1000 || $9 < -1000)' \
  | wc -l

# Check insert size distribution
samtools view /tmp/HG00759_rh.bam chr20:10000000-10100000 \
  | awk '{if($9>0) print $9}' \
  | sort -n > /tmp/HG00759_isizes.txt

samtools view /tmp/NA12878_rh.bam chr20:10000000-10100000 \
  | awk '{if($9>0) print $9}' \
  | sort -n > /tmp/NA12878_isizes.txt
```

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

hg_is = np.loadtxt("/tmp/HG00759_isizes.txt")
na_is = np.loadtxt("/tmp/NA12878_isizes.txt")

# Clip extreme outliers for display
clip = 2000
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(hg_is[hg_is < clip], bins=100, alpha=0.6, density=True,
        color="#1D9E75", label=f"HG00759 (median={np.median(hg_is):.0f})")
ax.hist(na_is[na_is < clip], bins=100, alpha=0.6, density=True,
        color="#378ADD", label=f"NA12878 (median={np.median(na_is):.0f})")
ax.set_xlabel("Insert size (bp)")
ax.set_ylabel("Density")
ax.set_title("Insert size distribution: HG00759 vs NA12878")
ax.legend()
plt.tight_layout()
plt.savefig("results/hg00759_investigation/03_insert_sizes.png", dpi=150)
plt.close()
```

**What to look for:**
- Insert size median shifted by >100bp from NA12878 → insertion or deletion
- Long tail of very large inserts (>1000bp) → likely insertion
- Bimodal insert size distribution → inversion or translocation
- Normal insert sizes → SV is unlikely, proceed to Phase 4

**Write to** `results/hg00759_investigation/03_sv_indicators.json`:
```json
{
  "discordant_read_pairs": ...,
  "HG00759_insert_median": ...,
  "NA12878_insert_median": ...,
  "insert_size_delta": ...,
  "sv_suspected": true/false
}
```

---

## Phase 4 — Embedding-Level Diagnosis

If Phases 1–3 find nothing anomalous (coverage, mapq, and insert sizes
are all normal), the issue is in the model representations themselves.
This phase compares the raw activation distributions.

```python
import numpy as np
from pathlib import Path

acts_base = Path("data/embeddings/grch37")
samples = ["HG00759", "NA18939", "HG00864", "NA12878"]

# For each sample, load all activations and compute:
# 1. Mean activation per channel (768,)
# 2. Std per channel
# 3. Fraction of channels near-zero (|act| < 0.1)

stats = {}
for sid in samples:
    act_dir = acts_base / sid / "activations"
    vecs = []
    for npz in sorted(act_dir.glob("*.npz")):
        d = np.load(npz)
        key = next(
            (k for k in d.files if "mixed" in k.lower() or "concat" in k.lower()),
            d.files[0]
        )
        act = d[key]
        pooled = act.mean(axis=tuple(range(act.ndim - 1)))
        vecs.append(pooled)
    X = np.vstack(vecs)   # (n_sites, 768)
    stats[sid] = {
        "n_sites":        int(len(X)),
        "channel_mean":   X.mean(axis=0).tolist(),   # per-channel mean
        "global_mean":    float(X.mean()),
        "global_std":     float(X.std()),
        "dead_channels":  int((np.abs(X.mean(axis=0)) < 0.01).sum()),
    }
    print(f"{sid}: n={len(X)}  global_mean={stats[sid]['global_mean']:.4f}"
          f"  dead_channels={stats[sid]['dead_channels']}")
```

Then compare the top-20 complexity channels (from probe_results.json)
specifically — do those channels activate differently in HG00759?

```python
import json

with open("results/giab_validation/linear_probe/probe_results.json") as f:
    probe = json.load(f)

# Channels with highest |weight| in probe
top_channels = [c["channel"] for c in probe["top20_probe_weights"]]

print(f"\nTop probe channels activation means:")
print(f"{'Channel':>10s}", end="")
for sid in samples:
    print(f"  {sid:>12s}", end="")
print()

for ch in top_channels[:10]:
    print(f"ch{ch:>7d}", end="")
    for sid in samples:
        mean_ch = np.array(stats[sid]["channel_mean"])[ch]
        print(f"  {mean_ch:>12.4f}", end="")
    print()
```

**Write to** `results/hg00759_investigation/04_embedding_diagnosis.json`.

**What to look for:**
- If HG00759 has many more dead channels → activation collapse,
  possibly from very low coverage causing degenerate pileup images
- If the top probe channels specifically are suppressed in HG00759 →
  the model is encoding something genuinely different about this sample
- If HG00759 looks similar to other EAS samples on all channels →
  the outlier status may be driven by a small number of extreme sites
  (check Phase 0 position plot for clusters of very negative scores)

---

## Phase 5 — Pileup Image Spot Check

If Phases 1–4 are inconclusive, visually inspect the pileup images for
HG00759's lowest-scoring sites.

```python
import numpy as np
from pathlib import Path

scores    = np.load("results/population_complexity/HG00759_scores.npy")
positions = np.load("results/population_complexity/HG00759_positions.npy",
                    allow_pickle=True)

# Find 5 lowest-scoring sites
lowest_idx = np.argsort(scores)[:5]
print("5 lowest-scoring sites for HG00759:")
for i in lowest_idx:
    print(f"  {positions[i]}  score={scores[i]:.3f}")
```

Use DeepVariant's built-in `show_examples` tool to render the pileup
images for these sites:

```bash
# show_examples is in the DeepVariant Docker image
# The tfrecord is at data/embeddings/grch37/HG00759/intermediate/

docker run --rm \
  -v $(pwd):$(pwd) -w $(pwd) \
  google/deepvariant:1.9.0 \
  python3 /opt/deepvariant/bin/show_examples.py \
  --examples data/embeddings/grch37/HG00759/intermediate/make_examples.tfrecord@1.gz \
  --output results/hg00759_investigation/pileup_images/ \
  --num_records 20
```

Visually inspect the rendered images. Look for:
- Very sparse reads (coverage dropout confirmed)
- Reads all mapping to one strand (mapping artifact)
- Lots of soft-clipped reads (SV boundary)
- Normal-looking pileups (no obvious artifact → genuine biology)

---

## Phase 6 — Diagnosis & Decision

Write `results/hg00759_investigation/diagnosis.json`:

```json
{
  "root_cause": "coverage_dropout" | "mapping_artifact" |
                "structural_variant" | "genuine_biology" | "inconclusive",
  "evidence": {
    "mean_coverage":     ...,
    "mapq_issue":        true/false,
    "sv_suspected":      true/false,
    "embedding_anomaly": true/false
  },
  "recommendation": "exclude" | "keep_annotated" | "keep",
  "rerun_population_stats_without_hg00759": true/false,
  "notes": "..."
}
```

**If recommendation is "exclude" or "keep_annotated":**

Rerun the population complexity statistics excluding HG00759 and write
updated results:

```python
# Reload all sample scores except HG00759
# Rerun Kruskal-Wallis and pairwise Mann-Whitney
# Compare new p-values and effect sizes to original
# Key question: does EAS still differ from EUR after removing HG00759?
# If yes → the EAS effect is robust
# If no  → the entire EAS signal was HG00759
```

Write updated stats to:
`results/hg00759_investigation/population_stats_excl_hg00759.json`

---

## Output Directory Contract

```
results/hg00759_investigation/
  00_score_summary.json
  00_score_vs_position.png
  01_coverage.json
  01_depth_profile.png
  02_mapping_quality.json          (if Phase 2 needed)
  03_insert_sizes.png              (if Phase 3 needed)
  03_sv_indicators.json            (if Phase 3 needed)
  04_embedding_diagnosis.json      (if Phase 4 needed)
  pileup_images/                   (if Phase 5 needed)
  diagnosis.json                   ← always written
  population_stats_excl_hg00759.json  ← if exclude recommended
```

---

## How to Invoke

Copy this file to skills/ and paste into Claude Code:

```
Read skills/HG00759_INVESTIGATION.md and execute all phases
in order, stopping early if the root cause is identified
conclusively. Write diagnosis.json regardless of which phase
you stop at.
```
