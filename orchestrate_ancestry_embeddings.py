#!/usr/bin/env python3
"""
orchestrate_ancestry_embeddings.py

Usage:
    python orchestrate_ancestry_embeddings.py
    python orchestrate_ancestry_embeddings.py --region 20:10000000-10100000
    python orchestrate_ancestry_embeddings.py --samples-only
    python orchestrate_ancestry_embeddings.py --skip-download
    python orchestrate_ancestry_embeddings.py --list-samples

Data source: gs://brain-genomics-public/research/cohort/1KGP/grch37_bams/
  - 2,504 samples from the NYGC 30x 1KGP phase 3 cohort, remapped to GRCh37
  - All samples at matched ~30x depth (coverage confound eliminated vs. GRCh38)
  - Reference: hs37d5 (bare contig names, e.g. "20" not "chr20")
  - Paper: Yun et al., Bioinformatics 2021 (DeepVariant + GLnexus cohort calls)

Sample selection (15 unrelated individuals, 3 per superpopulation):
  - All verified unrelated via 1KGP pedigree (20130606_g1k.ped)
  - Original samples NA19240 (AFR/YRI, child of NA19238), HG00514 (EAS),
    HG03732 (SAS), HG04157 (SAS) were absent from or related-within this
    cohort and were replaced with unrelated equivalents (see SAMPLES below).
"""

import anyio
import argparse
import json
import sys
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_REGION = "20:10000000-10100000"  # bare contig — GRCh37 uses "20" not "chr20"

# 15 unrelated individuals, 3 per superpopulation.
# All drawn from gs://brain-genomics-public/research/cohort/1KGP/grch37_bams/
# (NYGC 30x reads remapped to GRCh37/hs37d5 — balanced coverage across builds).
# Verified unrelated via 1KGP pedigree; no parent-child or sibling pairs.
SAMPLES = {
    # AFR — HG01985 replaces NA19240 (child of NA19238, which is kept)
    "HG01985": "AFR",  # ACB  ERR3242296
    "HG02922": "AFR",  # ESN  ERR3242482
    "NA19238": "AFR",  # YRI  ERR3239453
    # AMR
    "HG01048": "AMR",  # PUR  ERR3241761
    "HG01197": "AMR",  # PUR  ERR3241854
    "HG01565": "AMR",  # PEL  ERR3241984
    # EAS — HG00759 replaces HG00514 (absent from cohort)
    "HG00759": "EAS",  # CDX  ERR3242128
    "NA18939": "EAS",  # JPT  ERR3239557
    "HG00864": "EAS",  # CDX  ERR3242132
    # EUR
    "NA12878": "EUR",  # CEU  ERR3239334
    "HG00731": "EUR",  # PUR  ERR3241754
    "NA20502": "EUR",  # TSI  ERR3239785
    # SAS — HG03009/HG03022 replace HG03732/HG04157 (absent from cohort)
    "HG03009": "SAS",  # BEB  ERR3242842
    "NA20847": "SAS",  # GIH  ERR3239999
    "HG03022": "SAS",  # PJL  ERR3243053
}

GCS_BASE = "gs://brain-genomics-public/research/cohort/1KGP/grch37_bams"

# GCS filename pattern: ERR<run>_<sample>.grch37.bam
# ERR accession numbers for direct access (avoids gsutil ls per sample):
GCS_RUN_IDS = {
    "HG01985": "ERR3242296",
    "HG02922": "ERR3242482",
    "NA19238": "ERR3239453",
    "HG01048": "ERR3241761",
    "HG01197": "ERR3241854",
    "HG01565": "ERR3241984",
    "HG00759": "ERR3242128",
    "NA18939": "ERR3239557",
    "HG00864": "ERR3242132",
    "NA12878": "ERR3239334",
    "HG00731": "ERR3241754",
    "NA20502": "ERR3239785",
    "HG03009": "ERR3242842",
    "NA20847": "ERR3239999",
    "HG03022": "ERR3243053",
}

SUBDIRS = [
    "data/bams",
    "data/embeddings",
    "data/metadata",
    "results/pca",
    "results/umap",
    "results/probing",
    "results/figures",
    "logs",
]

# ── Subagent prompts ────────────────────────────────────────────────────────────

PREFLIGHT_PROMPT = f"""
You are the preflight agent. Verify the environment is ready for the ancestry
embedding pipeline. Run each check in order and report pass/fail.

IMPORTANT: samtools lives in the conda environment "deepVariant".
Use `conda run -n deepVariant samtools` for all samtools commands.

Checks to perform:
1. GCS auth: `gcloud auth application-default print-access-token` — confirm it succeeds
2. samtools installed: `conda run -n deepVariant samtools --version`
3. GCS BAM streaming (brain-genomics bucket, GRCh37):
   `conda run -n deepVariant samtools view -H "gs://brain-genomics-public/research/cohort/1KGP/grch37_bams/ERR3239334_NA12878.grch37.bam" 2>&1 | head -10`
   — parse the contig names. These BAMs use bare contig names ("20" not "chr20").
   chr_prefix should be "20".
4. Hook script exists and is executable: `test -x scripts/run_deepvariant_hooked.sh && echo OK`
5. Docker is running: `docker ps`
6. Python deps: `conda run -n deepVariant python3 -c "import numpy, pandas, sklearn, umap, matplotlib; print('OK')"`
7. Create all pipeline directories using `mkdir -p` for each:
   {json.dumps(SUBDIRS, indent=2)}

Write a JSON status file to data/metadata/preflight.json:
{{
  "gcs_auth": true/false,
  "samtools": true/false,
  "gcs_streaming": true/false,
  "hook_script": true/false,
  "docker": true/false,
  "python_deps": true/false,
  "dirs_ready": true/false,
  "chr_prefix": "20",
  "blockers": ["list any hard blockers"],
  "warnings": ["list soft warnings"]
}}

Finish by printing: PREFLIGHT COMPLETE — N blockers found.
"""


def make_bam_pull_prompt(sample_id: str, population: str, region: str) -> str:
    run_id = GCS_RUN_IDS[sample_id]
    gcs_path = f"{GCS_BASE}/{run_id}_{sample_id}.grch37.bam"
    return f"""
You are the BAM pull agent for sample {sample_id} (superpopulation: {population}).

Data source: NYGC 30x 1KGP phase 3 cohort, remapped to GRCh37/hs37d5.
Reference uses bare contig names (e.g. "20", not "chr20").

IMPORTANT: samtools lives in the conda environment "deepVariant".
Use `conda run -n deepVariant samtools` for ALL samtools commands.

Steps:
1. Pull the regional slice directly — the GCS path is known:
   conda run -n deepVariant samtools view -b \\
     "{gcs_path}" \\
     "{region}" \\
     -o data/bams/{sample_id}.slice.bam \\
     2>logs/{sample_id}_samtools.log
   conda run -n deepVariant samtools index data/bams/{sample_id}.slice.bam

2. Verify read count and contig names:
   conda run -n deepVariant samtools flagstat data/bams/{sample_id}.slice.bam
   conda run -n deepVariant samtools view -H data/bams/{sample_id}.slice.bam | grep "^@SQ" | head -3

3. Write data/metadata/{sample_id}_bam_status.json:
   {{
     "sample": "{sample_id}",
     "population": "{population}",
     "status": "success" or "failed",
     "read_count": <int or null>,
     "bam_path": "data/bams/{sample_id}.slice.bam",
     "gcs_source": "{gcs_path}",
     "region_used": "{region}",
     "reference": "GRCh37",
     "error": null or "<error message>"
   }}

Do not retry more than twice. Finish by printing: BAM {sample_id} — <status> (<read_count> reads).
"""


def make_embedding_prompt(sample_id: str, population: str, region: str) -> str:
    return f"""
You are the embedding extraction agent for sample {sample_id} ({population}).

Steps:
1. Check data/bams/{sample_id}.slice.bam exists and is non-empty.
   If missing, read data/metadata/{sample_id}_bam_status.json — if status is "failed",
   write data/metadata/{sample_id}_embedding_status.json with status "skipped"
   and reason "bam_pull_failed", then stop.

2. Read scripts/run_deepvariant_hooked.sh and CLAUDE.md to understand the exact CLI
   flags, output format, and where embedding files are written.

3. Run the hooked pipeline:
   - Input BAM: data/bams/{sample_id}.slice.bam
   - Output: data/embeddings/{sample_id}_mixed5
   - Region: {region}
   - Stderr log: logs/{sample_id}_deepvariant.log
   Adapt flags to match what the script actually accepts.

4. Find and load the output file. Detect format (.npy/.npz/.pkl/.h5).
   Report shape: number of candidate sites and embedding dimension.

5. Write data/metadata/{sample_id}_embedding_status.json:
   {{
     "sample": "{sample_id}",
     "population": "{population}",
     "status": "success" or "failed",
     "embedding_path": "<actual output path>",
     "embedding_format": "<extension>",
     "n_sites": <int or null>,
     "embedding_dim": <int or null>,
     "error": null or "<error message>"
   }}

Finish by printing: EMBEDDING {sample_id} — <status> (<n_sites> sites, dim=<dim>).
"""


ANALYSIS_PROMPT = """
You are the analysis agent. Run PCA + UMAP descriptive analysis on all embeddings.

Steps:

1. Read all data/metadata/*_embedding_status.json files.
   Collect samples with status == "success".
   Warn if fewer than 5 samples across 3+ superpopulations.

2. Load embeddings (detect format per embedding_format field: .npy → np.load,
   .npz → np.load then select array, .pkl → pickle.load, .h5 → h5py).
   Stack into X: shape (total_sites, embedding_dim).
   Build parallel arrays: labels (superpopulation), sample_ids.
   Save: np.savez("data/metadata/combined_embeddings.npz",
                  X=X, labels=np.array(labels), sample_ids=np.array(sample_ids))

3. Standardize: X_scaled = StandardScaler().fit_transform(X)

4. PCA:
   n_components = min(50, len(successful_samples) - 1, X.shape[1])
   X_pca = PCA(n_components=n_components).fit_transform(X_scaled)
   - Scree plot → results/pca/scree.png
   - PC1 vs PC2 colored by superpopulation → results/pca/pc1_pc2.png
   - PC1 vs PC3 → results/pca/pc1_pc3.png
   - Print variance explained by PCs 1–5

5. UMAP:
   reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(X)//2),
                       min_dist=0.1, random_state=42, metric='cosine')
   X_umap = reducer.fit_transform(X_scaled)
   - Colored by superpopulation (tab10) → results/umap/umap_superpop.png
   - Colored by sample_id → results/umap/umap_sample.png
   All plots: title, axis labels, legend, tight_layout(), dpi=150.

6. Clustering metrics:
   - Pairwise centroid distances between superpop clusters in UMAP space (print as matrix)
   - Silhouette score: sklearn.metrics.silhouette_score(X_umap, labels)
   - Score > 0.3 = clustering observed

7. Use Agg backend: import matplotlib; matplotlib.use('Agg')

8. Write results/pca/analysis_summary.json:
   {
     "n_samples": <int>,
     "superpops_present": [...],
     "n_sites_total": <int>,
     "embedding_dim": <int>,
     "pca_variance_explained_top5": [...],
     "umap_silhouette_score": <float>,
     "clustering_observed": <bool>
   }

Finish by printing a summary table.
"""


# ── Coordinator prompt ──────────────────────────────────────────────────────────

def build_coordinator_prompt(region: str, bam_prompts: dict, embedding_prompts: dict) -> str:
    return f"""
You are the coordinator for the ancestry embedding pipeline.
Use the Agent tool to spawn specialist subagents for each phase.

PHASE 1 — Preflight (sequential)
Spawn the "preflight-agent" subagent.
After it completes, read data/metadata/preflight.json.
If "blockers" is non-empty, print each blocker and STOP.

PHASE 2 — BAM pulls (parallel)
Spawn one "bam-pull-agent" per sample simultaneously using the Agent tool.
Pass each sample's specific task prompt (from the list below) as the Agent tool input.
Wait for all to complete. Report success/failure per superpopulation.
Warn if any superpopulation has 0 successful pulls.

BAM pull tasks (pass each value as the task to the bam-pull-agent):
{json.dumps(bam_prompts, indent=2)}

PHASE 3 — Embedding extraction (parallel)
Spawn one "embedding-agent" per sample simultaneously.
Wait for all to complete. Report how many embeddings succeeded.

Embedding tasks:
{json.dumps(embedding_prompts, indent=2)}

PHASE 4 — Analysis (sequential)
Spawn the "analysis-agent" subagent.

After all phases, print final summary:
- BAM pulls: N / {len(SAMPLES)} succeeded
- Embeddings: N / {len(SAMPLES)} succeeded
- Superpopulations with data: [list]
- Clustering observed: yes/no (from results/pca/analysis_summary.json)
- Key outputs: results/pca/pc1_pc2.png, results/umap/umap_superpop.png,
  results/pca/analysis_summary.json, data/metadata/combined_embeddings.npz
"""


# ── Subagent definitions ────────────────────────────────────────────────────────

def build_agents() -> dict:
    return {
        "preflight-agent": AgentDefinition(
            description="Runs environment preflight checks: GCS auth, samtools, Docker, Python deps, directory setup. Use for Phase 1.",
            prompt=PREFLIGHT_PROMPT,
            tools=["Bash", "Read", "Write"],
        ),
        "bam-pull-agent": AgentDefinition(
            description="Pulls a regional BAM slice from GCS for one 1KG sample. Use for each sample in Phase 2. Pass the sample-specific task as the Agent tool input.",
            prompt="You are a BAM pull specialist. Execute the task given to you by the coordinator exactly as specified.",
            tools=["Bash", "Read", "Write"],
        ),
        "embedding-agent": AgentDefinition(
            description="Runs the hooked DeepVariant pipeline to extract mixed5 embeddings for one sample. Use for each sample in Phase 3.",
            prompt="You are an embedding extraction specialist. Execute the task given to you by the coordinator exactly as specified.",
            tools=["Bash", "Read", "Write"],
        ),
        "analysis-agent": AgentDefinition(
            description="Runs PCA, UMAP, and clustering analysis on all extracted embeddings. Use for Phase 4.",
            prompt=ANALYSIS_PROMPT,
            tools=["Bash", "Read", "Write"],
        ),
    }


# ── Runner ──────────────────────────────────────────────────────────────────────

async def run_pipeline(region: str, samples_only: bool, skip_download: bool):
    bam_prompts = {
        sid: make_bam_pull_prompt(sid, pop, region)
        for sid, pop in SAMPLES.items()
    }
    embedding_prompts = {
        sid: make_embedding_prompt(sid, pop, region)
        for sid, pop in SAMPLES.items()
    }

    if skip_download:
        print("⏭  Skipping BAM pulls — embedding extraction + analysis only")
        prompt = f"""
You are the coordinator. Skip Phase 1 and Phase 2.

Phase 3 — spawn one "embedding-agent" per sample in parallel:
{json.dumps(embedding_prompts, indent=2)}

Phase 4 — spawn "analysis-agent":
{ANALYSIS_PROMPT}
"""
    elif samples_only:
        print("⏭  Running preflight + BAM pulls only")
        prompt = f"""
You are the coordinator. Run Phase 1 then Phase 2 only.

Phase 1 — spawn "preflight-agent":
{PREFLIGHT_PROMPT}

If preflight has blockers, stop.

Phase 2 — spawn one "bam-pull-agent" per sample in parallel:
{json.dumps(bam_prompts, indent=2)}

Print a summary of BAM pull results per superpopulation.
"""
    else:
        prompt = build_coordinator_prompt(region, bam_prompts, embedding_prompts)

    print(f"🚀  Launching coordinator | region={region} | samples={len(SAMPLES)}")
    print("=" * 72)

    options = ClaudeAgentOptions(
        cwd=str(Path.cwd()),
        model="claude-opus-4-6",
        thinking={"type": "adaptive"},
        allowed_tools=["Bash", "Read", "Write", "Agent"],
        permission_mode="acceptEdits",
        max_turns=200,
        agents=build_agents(),
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            print(f"\n\n{'='*72}")
            print("✅  Pipeline complete.")
            if message.result:
                print(message.result)

    # Summary from analysis output
    summary_path = Path("results/pca/analysis_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print("\n📊  Analysis summary:")
        print(f"   Superpopulations:    {summary.get('superpops_present')}")
        print(f"   Total sites:         {summary.get('n_sites_total')}")
        print(f"   Embedding dim:       {summary.get('embedding_dim')}")
        sil = summary.get('umap_silhouette_score')
        print(f"   Silhouette score:    {f'{sil:.3f}' if sil is not None else 'n/a'}")
        print(f"   Clustering observed: {summary.get('clustering_observed')}")
        print("\n📁  Key outputs:")
        for p in [
            "results/pca/pc1_pc2.png",
            "results/pca/pc1_pc3.png",
            "results/umap/umap_superpop.png",
            "results/umap/umap_sample.png",
            "results/pca/analysis_summary.json",
            "data/metadata/combined_embeddings.npz",
        ]:
            marker = "✓" if Path(p).exists() else "✗"
            print(f"   {marker}  {p}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate ancestry embedding analysis with Claude Agent SDK"
    )
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help=f"Genomic region (default: {DEFAULT_REGION})")
    parser.add_argument("--samples-only", action="store_true",
                        help="Only run preflight + BAM pulls")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip BAM pulls; run embedding + analysis only")
    parser.add_argument("--list-samples", action="store_true",
                        help="Print sample list and exit")
    args = parser.parse_args()

    if args.list_samples:
        print(f"{'Sample':<12} {'Superpop':<8}")
        print("-" * 22)
        for sid, pop in SAMPLES.items():
            print(f"{sid:<12} {pop:<8}")
        sys.exit(0)

    anyio.run(run_pipeline, args.region, args.samples_only, args.skip_download)


if __name__ == "__main__":
    main()
