#!/usr/bin/env python3
"""
orchestrate_joint_analysis.py

Joint GRCh37 / GRCh38 Analysis 1 orchestrator.
Mirrors the pattern in orchestrate_ancestry_embeddings.py.

Architecture:
  - Phase A: 13 GRCh37 embedding agents run in parallel (your local BAMs)
  - Phase B: waits for friend's GRCh38 .npz files to appear in data/embeddings/grch38/
  - Barrier: validates all 26 files present + shapes consistent
  - Phase C: aggregate agent concatenates everything, tags ref + pop
  - Phase D: joint analysis agent runs PCA + UMAP, writes plots + summary

Usage:
  # Full pipeline
  conda run -n deepVariant python orchestrate_joint_analysis.py

  # Skip GRCh37 embedding (already done), go straight to joint analysis
  conda run -n deepVariant python orchestrate_joint_analysis.py --skip-grch37

  # Only run GRCh37 embedding agents
  conda run -n deepVariant python orchestrate_joint_analysis.py --grch37-only

  # Point to a different GRCh38 embedding dir (e.g. mounted from friend's machine)
  conda run -n deepVariant python orchestrate_joint_analysis.py --grch38-dir /mnt/friend/embeddings
"""

import anthropic
import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

REGION = "chr20:10000000-10100000"   # after reheader, BAMs use chr prefix
REF_FASTA = "quickstart-testdata/ucsc.hg19.chr20.unittest.fasta"
CHECKPOINT = "model/wgs"
HOOK_LAYER = "mixed5"
BATCH_SIZE = 512

# 13 unrelated samples (NA19238 mother + NA19240 child excluded per friend's decision)
GRCH37_SAMPLES = [
    {"id": "HG01985", "pop": "LWK", "superpop": "AFR"},
    {"id": "HG02922", "pop": "LWK", "superpop": "AFR"},
    {"id": "HG01048", "pop": "CLM", "superpop": "AMR"},
    {"id": "HG01197", "pop": "CLM", "superpop": "AMR"},
    {"id": "HG01565", "pop": "PEL", "superpop": "AMR"},
    {"id": "HG00759", "pop": "CHS", "superpop": "EAS"},
    {"id": "NA18939", "pop": "JPT", "superpop": "EAS"},
    {"id": "HG00864", "pop": "CDX", "superpop": "EAS"},
    {"id": "NA12878", "pop": "CEU", "superpop": "EUR"},
    {"id": "HG00731", "pop": "GBR", "superpop": "EUR"},
    {"id": "NA20502", "pop": "TSI", "superpop": "EUR"},
    {"id": "HG03009", "pop": "GIH", "superpop": "SAS"},
    {"id": "NA20847", "pop": "GIH", "superpop": "SAS"},
]

# GRCh38 samples from friend — same IDs, different reference
# NA19238 + NA19240 excluded to match GRCh37 set
GRCH38_SAMPLES = [
    {"id": "HG00731", "pop": "GBR",  "superpop": "EUR"},
    {"id": "HG01048", "pop": "CLM",  "superpop": "AMR"},
    {"id": "HG01197", "pop": "CLM",  "superpop": "AMR"},
    {"id": "HG02922", "pop": "LWK",  "superpop": "AFR"},
    {"id": "NA18939", "pop": "JPT",  "superpop": "EAS"},
    {"id": "NA20502", "pop": "TSI",  "superpop": "EUR"},
    {"id": "NA20847", "pop": "GIH",  "superpop": "SAS"},
    {"id": "HG01565", "pop": "PEL",  "superpop": "AMR"},
    {"id": "HG00513", "pop": "CHS",  "superpop": "EAS"},
    {"id": "NA18525", "pop": "CHB",  "superpop": "EAS"},
    {"id": "NA12878", "pop": "CEU",  "superpop": "EUR"},
    {"id": "NA20845", "pop": "GIH",  "superpop": "SAS"},
    {"id": "NA20846", "pop": "GIH",  "superpop": "SAS"},
]

GRCH37_EMB_DIR = Path("data/embeddings/grch37")
GRCH38_EMB_DIR = Path("data/embeddings/grch38")
JOINT_OUT_DIR  = Path("results/joint_analysis")


# ── Agent prompts ─────────────────────────────────────────────────────────────

def grch37_agent_prompt(sample: dict) -> str:
    bam = f"data/bams/{sample['id']}.slice.bam"
    out_dir = f"data/embeddings/grch37/{sample['id']}"
    return f"""
You are an embedding extraction agent for sample {sample['id']} ({sample['superpop']}) aligned to GRCh37/hg19.

Your BAM is at: {bam}
It uses chromosome naming WITHOUT 'chr' prefix (SN:1, SN:2...).
You must reheader it before running make_examples.

Execute these steps IN ORDER using bash:

STEP 1 — Reheader the BAM (20 → chr20):
  samtools view -H {bam} | sed 's/SN:20\\t/SN:chr20\\t/' | samtools reheader - {bam} > /tmp/{sample['id']}_rh.bam
  samtools index /tmp/{sample['id']}_rh.bam

STEP 2 — make_examples (Docker):
  mkdir -p {out_dir}/intermediate
  docker run --rm \\
    -v $(pwd):$(pwd) -v /tmp:/tmp \\
    -w $(pwd) \\
    google/deepvariant:1.9.0 \\
    /opt/deepvariant/bin/make_examples \\
    --mode calling \\
    --ref {REF_FASTA} \\
    --reads /tmp/{sample['id']}_rh.bam \\
    --regions "{REGION}" \\
    --examples {out_dir}/intermediate/make_examples.tfrecord@1.gz \\
    --channel_list "read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref,insert_size"

STEP 3 — call_variants_hooked (native Python):
  python3 scripts/call_variants_hooked.py \\
    --examples {out_dir}/intermediate/make_examples.tfrecord@1.gz \\
    --checkpoint {CHECKPOINT} \\
    --outfile {out_dir}/intermediate/call_variants_output.tfrecord.gz \\
    --activation_cache_dir {out_dir}/activations \\
    --hook_layers {HOOK_LAYER} \\
    --batch_size {BATCH_SIZE} \\
    --max_cache_entries 10000

STEP 4 — Write status JSON:
  Write a file to {out_dir}/status.json with:
  {{
    "sample": "{sample['id']}",
    "population": "{sample['pop']}",
    "superpop": "{sample['superpop']}",
    "reference": "GRCh37",
    "status": "success" or "failed",
    "n_sites": <count of .npz files in {out_dir}/activations/>,
    "activation_dir": "{out_dir}/activations",
    "error": null or "<message>"
  }}

Report back with the status JSON content when done.
"""


def aggregate_agent_prompt(grch37_dirs: list, grch38_dirs: list) -> str:
    grch37_json = json.dumps(GRCH37_SAMPLES, indent=2)
    grch38_json = json.dumps(GRCH38_SAMPLES, indent=2)
    return f"""
You are an aggregation agent. Concatenate GRCh37 and GRCh38 mixed5 embeddings into a single combined file.

GRCh37 activation dirs:
{json.dumps(grch37_dirs, indent=2)}

GRCh38 activation dirs:
{json.dumps(grch38_dirs, indent=2)}

GRCh37 sample metadata:
{grch37_json}

GRCh38 sample metadata:
{grch38_json}

Write a Python script and execute it:

```python
import numpy as np
import json
from pathlib import Path

grch37_meta = {json.dumps(GRCH37_SAMPLES)}
grch38_meta = {json.dumps(GRCH38_SAMPLES)}

all_embeddings = []
all_labels = []   # superpop
all_samples = []
all_refs = []

def load_activations(act_dir, sample_id, superpop, ref):
    act_dir = Path(act_dir)
    npz_files = sorted(act_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {{act_dir}}")
    vecs = []
    for f in npz_files:
        d = np.load(f)
        # mixed5 shape: (n, 4, 12, 768) — global avg pool to 768-d
        act = d['mixed5'] if 'mixed5' in d else d[list(d.keys())[0]]
        pooled = act.reshape(len(act), -1).mean(axis=1, keepdims=True)
        # Actually: mean over spatial dims, keep channel dim
        flat = act.reshape(len(act), -1)  # (n, 4*12*768)
        # Use global avg pool to 768 to match existing pipeline
        pooled = act.mean(axis=(1, 2))    # (n, 768)
        vecs.append(pooled)
    return np.vstack(vecs), len(npz_files)

for s in grch37_meta:
    act_dir = f"data/embeddings/grch37/{{s['id']}}/activations"
    emb, n = load_activations(act_dir, s['id'], s['superpop'], 'GRCh37')
    all_embeddings.append(emb)
    all_labels.extend([s['superpop']] * len(emb))
    all_samples.extend([s['id']] * len(emb))
    all_refs.extend(['GRCh37'] * len(emb))

for s in grch38_meta:
    act_dir = f"data/embeddings/grch38/{{s['id']}}/activations"
    emb, n = load_activations(act_dir, s['id'], s['superpop'], 'GRCh38')
    all_embeddings.append(emb)
    all_labels.extend([s['superpop']] * len(emb))
    all_samples.extend([s['id']] * len(emb))
    all_refs.extend(['GRCh38'] * len(emb))

X = np.vstack(all_embeddings)
out = Path("results/joint_analysis")
out.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    out / "combined_embeddings.npz",
    embeddings=X,
    labels=np.array(all_labels),
    samples=np.array(all_samples),
    refs=np.array(all_refs),
)
print(f"Saved {{len(X)}} sites x {{X.shape[1]}} dims")
print(f"GRCh37: {{all_refs.count('GRCh37')}} sites")
print(f"GRCh38: {{all_refs.count('GRCh38')}} sites")
```

Run it and report the output. If any activation dir is missing, report which sample failed.
"""


def joint_analysis_agent_prompt() -> str:
    return """
You are the joint analysis agent. Run PCA + UMAP on the combined GRCh37/GRCh38 embeddings.

Input: results/joint_analysis/combined_embeddings.npz
  - embeddings: (N, 768) float32
  - labels: (N,) superpopulation strings  [AFR, AMR, EAS, EUR, SAS]
  - samples: (N,) sample ID strings
  - refs: (N,) reference strings  [GRCh37, GRCh38]

Write and execute this analysis script:

```python
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
from pathlib import Path
from itertools import product

out = Path("results/joint_analysis")
out.mkdir(parents=True, exist_ok=True)

d = np.load("results/joint_analysis/combined_embeddings.npz", allow_pickle=True)
X = d['embeddings']
labels = d['labels']
samples = d['samples']
refs = d['refs']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Color scheme: superpop by color, reference by marker
POP_COLORS = {
    'AFR': '#E24B4A', 'AMR': '#EF9F27',
    'EAS': '#1D9E75', 'EUR': '#378ADD', 'SAS': '#7F77DD'
}
REF_MARKERS = {'GRCh37': 'o', 'GRCh38': '^'}

# ── Plot 1: UMAP colored by superpop × reference ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for pop, ref in product(POP_COLORS, REF_MARKERS):
    mask = (labels == pop) & (refs == ref)
    ax.scatter(
        X_umap[mask, 0], X_umap[mask, 1],
        c=POP_COLORS[pop], marker=REF_MARKERS[ref],
        alpha=0.5, s=12, linewidths=0,
        label=f'{pop} {ref}'
    )
# Legend: pops by color, refs by shape
pop_patches = [mpatches.Patch(color=c, label=p) for p, c in POP_COLORS.items()]
ref_lines = [
    plt.Line2D([0],[0], marker='o', color='gray', linestyle='', label='GRCh37'),
    plt.Line2D([0],[0], marker='^', color='gray', linestyle='', label='GRCh38'),
]
leg1 = ax.legend(handles=pop_patches, loc='upper left', fontsize=8, title='Superpop')
ax.add_artist(leg1)
ax.legend(handles=ref_lines, loc='upper right', fontsize=8, title='Reference')
ax.set_title('UMAP: mixed5 embeddings — GRCh37 (circles) vs GRCh38 (triangles)')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
plt.tight_layout()
plt.savefig(out / 'umap_pop_x_ref.png', dpi=150)
plt.close()
print("Saved umap_pop_x_ref.png")

# ── Plot 2: PCA PC1 vs PC2 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for pop, ref in product(POP_COLORS, REF_MARKERS):
    mask = (labels == pop) & (refs == ref)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=POP_COLORS[pop], marker=REF_MARKERS[ref],
               alpha=0.5, s=12, linewidths=0)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA: GRCh37 (circles) vs GRCh38 (triangles)')
plt.tight_layout()
plt.savefig(out / 'pca_pc1_pc2_joint.png', dpi=150)
plt.close()
print("Saved pca_pc1_pc2_joint.png")

# ── Silhouette scores ─────────────────────────────────────────────────────
sil_pop = silhouette_score(X_umap, labels, sample_size=5000, random_state=42)
sil_ref = silhouette_score(X_umap, refs,   sample_size=5000, random_state=42)

# ── Per-reference centroid distances ─────────────────────────────────────
POPS = sorted(POP_COLORS.keys())
centroids = {}
for pop in POPS:
    for ref in ('GRCh37', 'GRCh38'):
        mask = (labels == pop) & (refs == ref)
        if mask.sum() > 0:
            centroids[(pop, ref)] = X_umap[mask].mean(axis=0)

# Matched-sample reference shift per population
ref_shifts = {}
for pop in POPS:
    if (pop, 'GRCh37') in centroids and (pop, 'GRCh38') in centroids:
        delta = np.linalg.norm(centroids[(pop,'GRCh37')] - centroids[(pop,'GRCh38')])
        ref_shifts[pop] = float(delta)

# Within-reference between-pop distances (GRCh37)
within_37 = {}
for p1 in POPS:
    for p2 in POPS:
        if p1 < p2 and (p1,'GRCh37') in centroids and (p2,'GRCh37') in centroids:
            within_37[f'{p1}-{p2}'] = float(
                np.linalg.norm(centroids[(p1,'GRCh37')] - centroids[(p2,'GRCh37')])
            )

summary = {
    "timestamp": __import__('datetime').datetime.now().isoformat(),
    "n_sites_total": int(len(X)),
    "n_sites_grch37": int((refs == 'GRCh37').sum()),
    "n_sites_grch38": int((refs == 'GRCh38').sum()),
    "embedding_dim": int(X.shape[1]),
    "pca_variance_explained": [float(v) for v in pca.explained_variance_ratio_[:5]],
    "silhouette_superpop": float(sil_pop),
    "silhouette_reference": float(sil_ref),
    "reference_shift_by_pop_umap": ref_shifts,
    "between_pop_dist_grch37_umap": within_37,
    "key_diagnostic": (
        "reference_dominates" if sil_ref > sil_pop else "population_dominates"
    )
}
with open(out / 'joint_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
```

Execute the script and return the full summary JSON output plus paths to the saved plots.
"""


# ── Agent runner ──────────────────────────────────────────────────────────────

async def run_agent(client: anthropic.Anthropic, prompt: str, label: str) -> dict:
    """Run a single Claude agent and return structured result."""
    print(f"  → Launching {label}")
    try:
        result_text = ""
        with client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                result_text += text

        # Try to extract JSON status if present
        try:
            # Look for the last JSON block in the response
            import re
            json_blocks = re.findall(r'\{[^{}]+\}', result_text, re.DOTALL)
            if json_blocks:
                status = json.loads(json_blocks[-1])
            else:
                status = {"status": "success", "raw": result_text[:500]}
        except Exception:
            status = {"status": "success", "raw": result_text[:500]}

        print(f"  ✅ {label} — done")
        return {"label": label, "result": status, "raw": result_text}

    except Exception as e:
        print(f"  ✗  {label} — FAILED: {e}")
        return {"label": label, "result": {"status": "failed", "error": str(e)}, "raw": ""}


def validate_barrier(grch37_results: list, grch38_dir: Path, grch38_samples: list) -> tuple[bool, list]:
    """Check all 26 embedding dirs exist before proceeding."""
    missing = []

    for r in grch37_results:
        s_id = r["label"].replace("grch37-", "")
        act_dir = Path(f"data/embeddings/grch37/{s_id}/activations")
        if not act_dir.exists() or not list(act_dir.glob("*.npz")):
            missing.append(f"GRCh37/{s_id}")

    for s in grch38_samples:
        act_dir = grch38_dir / s["id"] / "activations"
        if not act_dir.exists() or not list(act_dir.glob("*.npz")):
            missing.append(f"GRCh38/{s['id']}")

    return len(missing) == 0, missing


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args):
    client = anthropic.Anthropic()

    grch38_dir = Path(args.grch38_dir)
    print("=" * 70)
    print(" Joint Analysis 1 Orchestrator")
    print(f" GRCh37 samples: {len(GRCH37_SAMPLES)}")
    print(f" GRCh38 samples: {len(GRCH38_SAMPLES)} (from {grch38_dir})")
    print(f" Region: {REGION}")
    print("=" * 70)

    # ── Phase A: GRCh37 embedding agents ────────────────────────────────────
    grch37_results = []

    if not args.skip_grch37:
        print("\n[Phase A] Launching 13 GRCh37 embedding agents in parallel...")
        GRCH37_EMB_DIR.mkdir(parents=True, exist_ok=True)

        tasks = [
            run_agent(client, grch37_agent_prompt(s), f"grch37-{s['id']}")
            for s in GRCH37_SAMPLES
        ]
        grch37_results = await asyncio.gather(*tasks)

        # Save phase A results
        with open(GRCH37_EMB_DIR / "phase_a_results.json", "w") as f:
            json.dump(grch37_results, f, indent=2, default=str)
        print(f"[Phase A] Complete. Results → {GRCH37_EMB_DIR}/phase_a_results.json")
    else:
        print("[Phase A] Skipped — using existing GRCh37 embeddings")
        grch37_results = [{"label": f"grch37-{s['id']}", "result": {"status": "success"}}
                          for s in GRCH37_SAMPLES]

    if args.grch37_only:
        print("\nStopping after Phase A (--grch37-only set).")
        return

    # ── Phase B: Wait for GRCh38 files ──────────────────────────────────────
    print(f"\n[Phase B] Waiting for GRCh38 embeddings in {grch38_dir} ...")
    print("          Your friend needs to run their embedding pipeline first.")
    print("          Checking every 60s... (Ctrl-C to abort)")

    while True:
        ok, missing = validate_barrier(grch37_results, grch38_dir, GRCH38_SAMPLES)
        if ok:
            print("[Barrier] All 26 embedding dirs present. Proceeding.")
            break
        print(f"[Barrier] Still waiting for: {missing}")
        if args.skip_grch38_wait:
            print("[Barrier] --skip-grch38-wait set, proceeding anyway.")
            break
        await asyncio.sleep(60)

    # ── Phase C: Aggregate agent ─────────────────────────────────────────────
    print("\n[Phase C] Running aggregate agent...")
    JOINT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    grch37_dirs = [f"data/embeddings/grch37/{s['id']}/activations" for s in GRCH37_SAMPLES]
    grch38_dirs = [str(grch38_dir / s["id"] / "activations") for s in GRCH38_SAMPLES]

    agg_result = await run_agent(
        client,
        aggregate_agent_prompt(grch37_dirs, grch38_dirs),
        "aggregate"
    )
    print(f"[Phase C] Complete.")

    # ── Phase D: Joint analysis agent ────────────────────────────────────────
    print("\n[Phase D] Running joint PCA/UMAP analysis agent...")
    analysis_result = await run_agent(
        client,
        joint_analysis_agent_prompt(),
        "joint-analysis"
    )

    print("\n" + "=" * 70)
    print(" Pipeline complete!")
    print(f" Plots:   {JOINT_OUT_DIR}/umap_pop_x_ref.png")
    print(f"          {JOINT_OUT_DIR}/pca_pc1_pc2_joint.png")
    print(f" Summary: {JOINT_OUT_DIR}/joint_summary.json")
    print("=" * 70)

    # Print key diagnostic
    try:
        with open(JOINT_OUT_DIR / "joint_summary.json") as f:
            summary = json.load(f)
        print(f"\n Key diagnostic: {summary.get('key_diagnostic', 'unknown')}")
        print(f" Silhouette (superpop): {summary.get('silhouette_superpop', 'N/A'):.3f}")
        print(f" Silhouette (reference): {summary.get('silhouette_reference', 'N/A'):.3f}")
        shifts = summary.get("reference_shift_by_pop_umap", {})
        if shifts:
            print(f" Reference shift by pop: {shifts}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint GRCh37/GRCh38 Analysis 1 orchestrator")
    parser.add_argument("--skip-grch37", action="store_true",
                        help="Skip GRCh37 embedding agents (already done)")
    parser.add_argument("--grch37-only", action="store_true",
                        help="Only run GRCh37 embedding agents, then stop")
    parser.add_argument("--grch38-dir", default=str(GRCH38_EMB_DIR),
                        help=f"Path to GRCh38 embedding dir (default: {GRCH38_EMB_DIR})")
    parser.add_argument("--skip-grch38-wait", action="store_true",
                        help="Don't wait for GRCh38 files — run analysis on whatever is present")
    args = parser.parse_args()
    asyncio.run(main(args))
