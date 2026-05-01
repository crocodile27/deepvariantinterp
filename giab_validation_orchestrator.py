"""
giab_validation_orchestrator.py

GIAB truth-set validation of the hooked DeepVariant pipeline,
implemented as a Claude Agent SDK multi-agent workflow.

Paste this file into your deepvariantinterp repo root and run:

    python giab_validation_orchestrator.py

Prerequisites:
    pip install claude-agent-sdk          # or: pip install claude_agent_sdk
    export ANTHROPIC_API_KEY=...

    # Tools that must be on PATH:
    docker, samtools, wget, bcftools, hap.py (or pkrusche/hap.py Docker image)
    python packages: numpy, matplotlib, scikit-learn, umap-learn

Pipeline phases
───────────────
Phase 1  [download-agent]         Download GIAB truth VCF + BED for chr20 region
Phase 2  [postprocess-agent]      Run postprocess_variants on NA12878 → VCF
Phase 3  [benchmark-agent]        Run hap.py → TP/FP/FN labels per site
Phase 4  [label-agent]            Cross-reference .npz filenames with hap.py output
Phase 5  [plot-agent]             PCA + UMAP coloured by TP/FP/FN

Phases 1 and 2 are independent and run concurrently.
Phases 3-5 are sequential, each gated on the previous phase's output file.
"""

import asyncio
import sys
from pathlib import Path
from claude_agent_sdk import query, ClaudeAgentOptions

# ── Config ─────────────────────────────────────────────────────────────────

REGION        = "chr20:10000000-10100000"
REF_FASTA     = "quickstart-testdata/ucsc.hg19.chr20.unittest.fasta"
CHECKPOINT    = "model/wgs"
NA12878_CVO   = "data/embeddings/grch37/NA12878/intermediate/call_variants_output.tfrecord.gz"
NA12878_ACTS  = "data/embeddings/grch37/NA12878/activations"
GIAB_DIR      = "giab_truth"
RESULTS_DIR   = "results/giab_validation"

GIAB_VCF_URL  = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
    "NA12878_HG001/NISTv3.3.2/GRCh37/"
    "HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2"
    "_highconf_PGandRTGphasetransfer.vcf.gz"
)
GIAB_TBI_URL  = GIAB_VCF_URL + ".tbi"
GIAB_BED_URL  = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
    "NA12878_HG001/NISTv3.3.2/GRCh37/"
    "HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2"
    "_highconf_nosomaticdel.bed"
)

GIAB_VCF_LOCAL = f"{GIAB_DIR}/HG001_GRCh37_highconf.vcf.gz"
GIAB_BED_LOCAL = f"{GIAB_DIR}/HG001_GRCh37_highconf.bed"
HOOKED_VCF     = f"{GIAB_DIR}/NA12878_hooked_chr20.vcf.gz"
HAPPY_PREFIX   = f"{RESULTS_DIR}/happy_results"
LABELS_JSON    = f"{RESULTS_DIR}/site_labels.json"
PLOT_DIR       = RESULTS_DIR

# ── Agent definitions ───────────────────────────────────────────────────────

SUBAGENT_DEFS = {
    "download-agent": {
        "description": (
            "Downloads GIAB truth VCF, TBI index, and high-confidence BED file "
            "for NA12878/HG001 from the NCBI FTP site. Use this agent for any "
            "task involving fetching GIAB benchmark files."
        ),
        "prompt": f"""
You are the GIAB download agent. Your job is to fetch three files from the NCBI
FTP site if they don't already exist locally.

Working directory: the deepvariantinterp repo root.

Files to download (skip any that already exist):
  1. Truth VCF:   {GIAB_VCF_URL}
     → save as:   {GIAB_VCF_LOCAL}
  2. TBI index:   {GIAB_TBI_URL}
     → save as:   {GIAB_VCF_LOCAL}.tbi
  3. High-conf BED: {GIAB_BED_URL}
     → save as:   {GIAB_BED_LOCAL}

Steps:
  mkdir -p {GIAB_DIR}

  For each file:
    - Check if it already exists with: ls -lh <path>
    - If missing, download with wget:
        wget -q --show-progress -O <dest> <url>
    - Verify the download completed (non-zero file size)

After all three files are present, verify the VCF is readable:
    bcftools stats {GIAB_VCF_LOCAL} | head -20

Write a status file at {GIAB_DIR}/download_status.json:
{{
  "status": "success" | "failed",
  "vcf": "{GIAB_VCF_LOCAL}",
  "bed": "{GIAB_BED_LOCAL}",
  "error": null | "<message>"
}}

Report the final status JSON.
""",
        "allowed_tools": ["Bash", "Read", "Write"],
    },
    "postprocess-agent": {
        "description": (
            "Runs DeepVariant postprocess_variants in Docker to convert the "
            "NA12878 call_variants output TFRecord into a VCF file. Use this "
            "agent for the postprocess step of the DeepVariant pipeline."
        ),
        "prompt": f"""
You are the postprocess agent. Convert the hooked call_variants output for
NA12878 into a VCF using DeepVariant's postprocess_variants tool in Docker.

Input:  {NA12878_CVO}
Output: {HOOKED_VCF}
Ref:    {REF_FASTA}

Steps:

1. Check the input file exists and is non-empty:
     ls -lh {NA12878_CVO}

2. Create output directory:
     mkdir -p {GIAB_DIR}

3. Run postprocess_variants in Docker (use absolute paths — Docker requires them):
     docker run --rm \\
       -v $(pwd):$(pwd) \\
       -w $(pwd) \\
       google/deepvariant:1.9.0 \\
       /opt/deepvariant/bin/postprocess_variants \\
       --ref $(pwd)/{REF_FASTA} \\
       --infile $(pwd)/{NA12878_CVO} \\
       --outfile $(pwd)/{HOOKED_VCF}

4. Verify the VCF was created:
     ls -lh {HOOKED_VCF}
     bcftools stats {HOOKED_VCF} | grep "^SN"

5. Write status to {GIAB_DIR}/postprocess_status.json:
{{
  "status": "success" | "failed",
  "vcf_path": "{HOOKED_VCF}",
  "variant_count": <int from bcftools stats>,
  "error": null | "<message>"
}}

Report the final status JSON.
""",
        "allowed_tools": ["Bash", "Read", "Write"],
    },
    "benchmark-agent": {
        "description": (
            "Runs hap.py (via Docker) to benchmark the hooked DeepVariant VCF "
            "against the GIAB truth set, producing TP/FP/FN labels per variant "
            "site. Use this agent for benchmarking and precision/recall analysis."
        ),
        "prompt": f"""
You are the benchmarking agent. Run hap.py to compare the hooked DeepVariant
VCF against the GIAB truth set for NA12878.

Inputs:
  Truth VCF:  {GIAB_VCF_LOCAL}
  Truth BED:  {GIAB_BED_LOCAL}
  Query VCF:  {HOOKED_VCF}
  Reference:  {REF_FASTA}
  Region:     {REGION}

Output prefix: {HAPPY_PREFIX}

Steps:

1. Verify all inputs exist:
     ls -lh {GIAB_VCF_LOCAL} {GIAB_BED_LOCAL} {HOOKED_VCF} {REF_FASTA}

2. Create results directory:
     mkdir -p {RESULTS_DIR}

3. Run hap.py via Docker:
     docker run --rm \\
       -v $(pwd):$(pwd) \\
       -w $(pwd) \\
       pkrusche/hap.py \\
       /opt/hap.py/bin/hap.py \\
       $(pwd)/{GIAB_VCF_LOCAL} \\
       $(pwd)/{HOOKED_VCF} \\
       -f $(pwd)/{GIAB_BED_LOCAL} \\
       -r $(pwd)/{REF_FASTA} \\
       -o $(pwd)/{HAPPY_PREFIX} \\
       --regions {REGION} \\
       --engine=vcfeval \\
       --threads 4

   If pkrusche/hap.py fails to pull, try: jmcdani20/hap.py:v0.3.12

4. Check outputs were created:
     ls -lh {HAPPY_PREFIX}.*
     
   hap.py produces:
     {HAPPY_PREFIX}.summary.csv       — precision/recall/F1 by type
     {HAPPY_PREFIX}.vcf.gz            — per-site TP/FP/FN labels (BD field)
     {HAPPY_PREFIX}.extended.csv      — stratified metrics

5. Print the summary CSV:
     cat {HAPPY_PREFIX}.summary.csv

6. Write status to {RESULTS_DIR}/benchmark_status.json:
{{
  "status": "success" | "failed",
  "summary_csv": "{HAPPY_PREFIX}.summary.csv",
  "labeled_vcf": "{HAPPY_PREFIX}.vcf.gz",
  "error": null | "<message>"
}}

Report the status JSON and the full summary CSV content.
""",
        "allowed_tools": ["Bash", "Read", "Write"],
    },
    "label-agent": {
        "description": (
            "Cross-references hap.py output VCF with activation .npz filenames "
            "to assign TP/FP/FN labels to each embedding. Use this agent for "
            "merging variant truth labels with embedding files."
        ),
        "prompt": f"""
You are the label-assignment agent. Cross-reference the hap.py labeled VCF
with the NA12878 activation .npz files to produce a JSON mapping each site to
its truth label (TP, FP, FN, or UNK).

Inputs:
  hap.py labeled VCF:  {HAPPY_PREFIX}.vcf.gz
  Activation dir:      {NA12878_ACTS}/
  Output:              {LABELS_JSON}

The .npz files are named like: chr20_10001436.npz
  → position = chr20:10001436

The hap.py VCF has a BD FORMAT field with values:
  TP  = true positive (called correctly)
  FP  = false positive (called, not in truth)
  FN  = false negative (in truth, not called)  — only in truth VCF rows
  UNK = outside high-confidence regions

Write and execute this Python script:

```python
import json
import subprocess
import re
from pathlib import Path

acts_dir = Path("{NA12878_ACTS}")
output_path = Path("{LABELS_JSON}")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Parse hap.py VCF for BD labels
# bcftools query extracts: CHROM POS BD (FORMAT field)
cmd = [
    "bcftools", "query",
    "-f", "%CHROM\\t%POS\\t[%BD]\\n",
    "{HAPPY_PREFIX}.vcf.gz"
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f"bcftools failed: {{result.stderr}}")

# Build position → label map
pos_to_label = {{}}
for line in result.stdout.strip().split("\\n"):
    if not line:
        continue
    parts = line.split("\\t")
    if len(parts) >= 3:
        chrom, pos, bd = parts[0], parts[1], parts[2]
        key = f"{{chrom}}_{{pos}}"
        # hap.py may emit multiple rows per position; TP > FP > UNK priority
        priority = {{"TP": 3, "FP": 2, "FN": 2, "UNK": 1, ".": 0}}
        existing = pos_to_label.get(key, {{"label": ".", "priority": 0}})
        p = priority.get(bd, 0)
        if p > existing["priority"]:
            pos_to_label[key] = {{"label": bd, "priority": p}}

# Match against .npz files
# filename format: chr20_10001436.npz → key: chr20_10001436
site_labels = {{}}
unmatched = []

for npz in sorted(acts_dir.glob("*.npz")):
    stem = npz.stem  # e.g. chr20_10001436
    if stem in pos_to_label:
        site_labels[stem] = pos_to_label[stem]["label"]
    else:
        # Also try without leading zeros / with offset ±1
        site_labels[stem] = "UNK"
        unmatched.append(stem)

counts = {{lb: sum(1 for v in site_labels.values() if v == lb)
          for lb in ["TP", "FP", "FN", "UNK", "."]}}

output = {{
    "n_sites": len(site_labels),
    "label_counts": counts,
    "n_unmatched": len(unmatched),
    "site_labels": site_labels,
}}

output_path.write_text(json.dumps(output, indent=2))
print(json.dumps({{k: v for k, v in output.items() if k != "site_labels"}}, indent=2))
print(f"\\nLabels written to {{output_path}}")
```

Run the script and report the label counts summary.
""",
        "allowed_tools": ["Bash", "Read", "Write"],
    },
    "plot-agent": {
        "description": (
            "Generates PCA and UMAP plots of NA12878 mixed5 embeddings coloured "
            "by GIAB truth labels (TP/FP/FN/UNK). Use this agent for all "
            "embedding visualisation and interpretability plotting."
        ),
        "prompt": f"""
You are the visualisation agent. Produce PCA and UMAP plots of NA12878 mixed5
embeddings coloured by GIAB truth labels.

Inputs:
  Activation dir:   {NA12878_ACTS}/  (*.npz files)
  Labels JSON:      {LABELS_JSON}
  Output dir:       {PLOT_DIR}/

Write and execute this Python script:

```python
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

acts_dir  = Path("{NA12878_ACTS}")
labels_path = Path("{LABELS_JSON}")
out_dir   = Path("{PLOT_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

# Load labels
with open(labels_path) as f:
    label_data = json.load(f)
site_labels = label_data["site_labels"]

# Load embeddings
LABEL_COLORS = {{
    "TP":  "#1D9E75",   # teal  — correct calls
    "FP":  "#E24B4A",   # red   — called, not in truth
    "FN":  "#EF9F27",   # amber — missed, in truth
    "UNK": "#B4B2A9",   # gray  — outside high-conf regions
    ".":   "#B4B2A9",   # gray  — unresolved
}}

embeddings, labels, positions = [], [], []

for npz_path in sorted(acts_dir.glob("*.npz")):
    stem = npz_path.stem
    d = np.load(npz_path)
    # Resolve layer key: mixed5 → Mixed_5d fallback
    key = next(
        (k for k in d.files if "mixed" in k.lower() or "concat" in k.lower()),
        d.files[0]
    )
    act = d[key]                        # (1, 4, 12, 768) or (batch, H, W, C)
    pooled = act.mean(axis=tuple(range(act.ndim - 1)))  # → (768,)
    embeddings.append(pooled)
    labels.append(site_labels.get(stem, "UNK"))
    positions.append(stem)

X = np.vstack(embeddings)
labels = np.array(labels)
print(f"Loaded {{len(X)}} sites × {{X.shape[1]}} dims")
print("Label counts:", {{lb: int((labels == lb).sum()) for lb in ["TP","FP","FN","UNK","."]}})

# Standardise
X_scaled = StandardScaler().fit_transform(X)

# ── PCA ──────────────────────────────────────────────────────────────────
pca = PCA(n_components=min(20, len(X)-1), random_state=42)
X_pca = pca.fit_transform(X_scaled)

def scatter_by_label(ax, coords, labels, title, xlabel, ylabel):
    # Draw UNK/. first (background), then TP, FP, FN on top
    draw_order = ["UNK", ".", "FN", "FP", "TP"]
    for lb in draw_order:
        mask = labels == lb
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=LABEL_COLORS.get(lb, "#888"),
            label=f"{{lb}} (n={{mask.sum()}})",
            alpha=0.75 if lb in ("TP","FP","FN") else 0.35,
            s=30 if lb in ("TP","FP","FN") else 12,
            linewidths=0,
            zorder=3 if lb in ("TP","FP","FN") else 1,
        )
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9, markerscale=1.5, framealpha=0.8)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scatter_by_label(
    axes[0], X_pca, labels,
    "PCA — NA12878 mixed5, coloured by GIAB label",
    f"PC1 ({{pca.explained_variance_ratio_[0]*100:.1f}}%)",
    f"PC2 ({{pca.explained_variance_ratio_[1]*100:.1f}}%)",
)
scatter_by_label(
    axes[1], X_pca[:, [0,2]], labels,
    "PCA PC1 vs PC3",
    f"PC1 ({{pca.explained_variance_ratio_[0]*100:.1f}}%)",
    f"PC3 ({{pca.explained_variance_ratio_[2]*100:.1f}}%)",
)
plt.tight_layout()
plt.savefig(out_dir / "NA12878_pca_by_giab_label.png", dpi=150)
plt.close()
print("Saved NA12878_pca_by_giab_label.png")

# ── UMAP ─────────────────────────────────────────────────────────────────
try:
    import umap
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(X)-1),
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    X_umap = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_by_label(
        ax, X_umap, labels,
        "UMAP — NA12878 mixed5, coloured by GIAB label\\n"
        "(TP=correct calls, FP=false positives, FN=missed variants, UNK=outside high-conf)",
        "UMAP 1", "UMAP 2",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "NA12878_umap_by_giab_label.png", dpi=150)
    plt.close()
    print("Saved NA12878_umap_by_giab_label.png")
except ImportError:
    print("umap-learn not installed, skipping UMAP. Run: pip install umap-learn")

# ── Per-label distance analysis ───────────────────────────────────────────
# Centroid distances in PCA space — how separated are TP vs FP embeddings?
unique_labels = [lb for lb in ["TP","FP","FN"] if (labels == lb).sum() > 0]
centroids = {{lb: X_pca[labels == lb].mean(axis=0) for lb in unique_labels}}

print("\\nCentroid distances in PCA space:")
import itertools
dist_results = {{}}
for l1, l2 in itertools.combinations(unique_labels, 2):
    d = float(np.linalg.norm(centroids[l1] - centroids[l2]))
    dist_results[f"{{l1}}-{{l2}}"] = round(d, 4)
    print(f"  {{l1}} vs {{l2}}: {{d:.4f}}")

# Variance explained by label in PC1 (quick proxy for separability)
pc1_by_label = {{lb: X_pca[labels == lb, 0].tolist() for lb in unique_labels}}
pc1_means    = {{lb: float(np.mean(v)) for lb, v in pc1_by_label.items()}}
print(f"\\nPC1 means by label: {{pc1_means}}")

# Save summary
summary = {{
    "n_sites": int(len(X)),
    "label_counts": {{lb: int((labels == lb).sum()) for lb in ["TP","FP","FN","UNK","."]}},
    "pca_variance_explained_pct": [round(float(v)*100,2) for v in pca.explained_variance_ratio_[:5]],
    "centroid_distances_pca": dist_results,
    "pc1_means_by_label": pc1_means,
    "plots": [
        "{PLOT_DIR}/NA12878_pca_by_giab_label.png",
        "{PLOT_DIR}/NA12878_umap_by_giab_label.png",
    ],
    "interpretation_note": (
        "If TP and FP centroids are well-separated, mixed5 encodes call confidence "
        "geometrically. If not, confidence information lives in deeper layers."
    ),
}}
import json
Path("{PLOT_DIR}/giab_validation_summary.json").write_text(json.dumps(summary, indent=2))
print("\\nSummary saved to {PLOT_DIR}/giab_validation_summary.json")
print(json.dumps({{k:v for k,v in summary.items() if k != "plots"}}, indent=2))
```

Execute the script and report the centroid distances and label counts.
""",
        "allowed_tools": ["Bash", "Read", "Write"],
    },
}


# ── Orchestrator ────────────────────────────────────────────────────────────

async def run_subagent(name: str, defn: dict) -> dict:
    """Spawn one subagent and collect its final text output."""
    print(f"\n  → [{name}] starting")
    result_text = ""
    try:
        async for message in query(
            prompt=defn["prompt"],
            options=ClaudeAgentOptions(
                allowed_tools=defn.get("allowed_tools", ["Bash", "Read", "Write"]),
                max_turns=40,
            ),
        ):
            # Collect final assistant text
            if hasattr(message, "text"):
                result_text += message.text
    except Exception as e:
        print(f"  ✗  [{name}] FAILED: {e}")
        return {"agent": name, "status": "failed", "error": str(e), "output": ""}

    print(f"  ✅ [{name}] done")
    return {"agent": name, "status": "success", "output": result_text}


async def gate(path: str, agent_name: str) -> bool:
    """Check a sentinel file exists before proceeding."""
    if Path(path).exists():
        return True
    print(f"\n  ✗  Gate failed: {path} not found after [{agent_name}]")
    print(f"     Check {agent_name} output above for errors.")
    return False


async def main():
    print("=" * 68)
    print(" GIAB Validation Orchestrator")
    print(f" Sample:  NA12878 (HG001/CEU/EUR) — GRCh37")
    print(f" Region:  {REGION}")
    print(f" Truth:   GIAB v3.3.2 high-confidence")
    print("=" * 68)

    # ── Phase 1+2: download GIAB files AND postprocess in parallel ──────────
    print("\n[Phase 1+2] Downloading GIAB truth files + running postprocess_variants (parallel)...")
    download_result, postprocess_result = await asyncio.gather(
        run_subagent("download-agent",     SUBAGENT_DEFS["download-agent"]),
        run_subagent("postprocess-agent",  SUBAGENT_DEFS["postprocess-agent"]),
    )

    # Gate: both outputs must exist
    dl_ok  = await gate(f"{GIAB_DIR}/download_status.json",     "download-agent")
    pp_ok  = await gate(f"{GIAB_DIR}/postprocess_status.json",  "postprocess-agent")
    vcf_ok = await gate(HOOKED_VCF,                              "postprocess-agent")

    if not (dl_ok and pp_ok and vcf_ok):
        print("\nAborting: one or more Phase 1/2 agents did not complete.")
        print("Check the agent output above, fix the issue, then rerun.")
        sys.exit(1)

    # ── Phase 3: benchmark with hap.py ─────────────────────────────────────
    print("\n[Phase 3] Running hap.py benchmark...")
    bench_result = await run_subagent("benchmark-agent", SUBAGENT_DEFS["benchmark-agent"])

    if not await gate(f"{HAPPY_PREFIX}.summary.csv", "benchmark-agent"):
        print("\nAborting: hap.py did not produce summary CSV.")
        print("Common cause: pkrusche/hap.py image not available.")
        print("Try: docker pull jmcdani20/hap.py:v0.3.12")
        print("Then update the image name in SUBAGENT_DEFS['benchmark-agent']['prompt'].")
        sys.exit(1)

    # ── Phase 4: label embeddings ───────────────────────────────────────────
    print("\n[Phase 4] Cross-referencing embeddings with truth labels...")
    label_result = await run_subagent("label-agent", SUBAGENT_DEFS["label-agent"])

    if not await gate(LABELS_JSON, "label-agent"):
        print("\nAborting: site_labels.json was not written.")
        sys.exit(1)

    # ── Phase 5: plots ──────────────────────────────────────────────────────
    print("\n[Phase 5] Generating PCA + UMAP plots coloured by TP/FP/FN...")
    plot_result = await run_subagent("plot-agent", SUBAGENT_DEFS["plot-agent"])

    # ── Summary ─────────────────────────────────────────────────────────────
    summary_path = Path(f"{RESULTS_DIR}/giab_validation_summary.json")
    print("\n" + "=" * 68)
    print(" Pipeline complete!")
    print(f" Plots:   {RESULTS_DIR}/NA12878_pca_by_giab_label.png")
    print(f"          {RESULTS_DIR}/NA12878_umap_by_giab_label.png")
    print(f" Summary: {summary_path}")
    print(f" Labels:  {LABELS_JSON}")
    print("=" * 68)

    if summary_path.exists():
        import json
        s = json.loads(summary_path.read_text())
        print(f"\n Key results:")
        print(f"   Label counts:      {s.get('label_counts')}")
        print(f"   Centroid dists:    {s.get('centroid_distances_pca')}")
        print(f"   PC1 means:         {s.get('pc1_means_by_label')}")
        print(f"\n Interpretation:")
        print(f"   {s.get('interpretation_note')}")


if __name__ == "__main__":
    asyncio.run(main())
