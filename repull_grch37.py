#!/usr/bin/env python3
"""
repull_grch37.py

Replaces the 3 GRCh38 samples with GRCh37 equivalents from EBI phase3 low-coverage.
Run this before proceeding to embedding extraction.

Usage:
    python repull_grch37.py
"""

import anyio
import json
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

# ── Replacements ───────────────────────────────────────────────────────────────
# Original GRCh38 samples → GRCh37 replacements, same superpopulation
REPLACEMENTS = {
    # (old_sample, superpop): (new_sample, reason)
    ("HG00514", "EAS"): ("NA18525", "EAS — EBI phase3 GRCh37 low-coverage"),
    ("HG03732", "SAS"): ("NA20127", "SAS — EBI phase3 GRCh37 low-coverage"),
    ("HG04157", "SAS"): ("NA20769", "SAS — EBI phase3 GRCh37 low-coverage"),
}

REGION = "20:10000000-10100000"  # bare contig, GRCh37
EBI_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data"


def make_repull_prompt(old_sample: str, new_sample: str, population: str) -> str:
    return f"""
You are a BAM pull agent replacing a GRCh38 sample with a GRCh37 equivalent.

Old sample (GRCh38, to be replaced): {old_sample}
New sample (GRCh37 target): {new_sample}
Superpopulation: {population}
Region: {REGION}  ← bare contig (no "chr" prefix), GRCh37 coordinates

Steps:

1. Archive the old sample's files so nothing is lost:
   mv data/bams/{old_sample}.slice.bam data/bams/{old_sample}.slice.bam.grch38_backup 2>/dev/null || true
   mv data/bams/{old_sample}.slice.bam.bai data/bams/{old_sample}.slice.bam.bai.grch38_backup 2>/dev/null || true
   mv data/metadata/{old_sample}_bam_status.json data/metadata/{old_sample}_bam_status.json.grch38_backup 2>/dev/null || true

2. Find the BAM URL for {new_sample} on EBI:
   curl -s --head "{EBI_BASE}/{new_sample}/sequence_read/" | head -5
   Then list available BAMs:
   curl -s "{EBI_BASE}/{new_sample}/sequence_read/" | grep -oP '(?<=href=")[^"]+\\.bam(?=")' | grep -v bai

   The URL pattern is typically:
   {EBI_BASE}/{new_sample}/sequence_read/{new_sample}.mapped.ILLUMINA.bwa.*.low_coverage.bam

3. Pull the regional slice using samtools with the EBI HTTP URL:
   samtools view -b \\
     "<EBI_BAM_URL>" \\
     "{REGION}" \\
     -o data/bams/{new_sample}.slice.bam \\
     2>logs/{new_sample}_samtools.log
   samtools index data/bams/{new_sample}.slice.bam

4. Verify:
   samtools flagstat data/bams/{new_sample}.slice.bam
   Confirm read count > 0 and reference contigs are bare (SN:20, not SN:chr20):
   samtools view -H data/bams/{new_sample}.slice.bam | grep "^@SQ" | head -3

5. Write data/metadata/{new_sample}_bam_status.json:
   {{
     "sample": "{new_sample}",
     "population": "{population}",
     "status": "success" or "failed",
     "read_count": <int>,
     "bam_path": "data/bams/{new_sample}.slice.bam",
     "gcs_source": "<EBI URL used>",
     "region_used": "{REGION}",
     "reference": "GRCh37",
     "replaces": "{old_sample}",
     "error": null or "<error message>"
   }}

6. Write a brief note to data/metadata/sample_manifest.json if it exists,
   or create it, recording the final sample list with reference build for each.

Finish by printing: REPULL {new_sample} (replaces {old_sample}) — <status> (<read_count> reads, contig_prefix=<bare/chr>).
"""


COORDINATOR_PROMPT = f"""
You are the coordinator for a targeted re-pull of 3 samples.

Context: A previous pipeline run pulled 15 BAMs but 3 came from GRCh38 high-coverage
CRAMs (HG00514, HG03732, HG04157). We are replacing them with GRCh37 low-coverage
equivalents from EBI phase3 so all 15 samples share the same reference build.

Spawn all 3 "bam-pull-agent" subagents in parallel simultaneously.

{chr(10).join(
    f"Agent {i+1} — replace {old} with {new} ({pop}):{chr(10)}{make_repull_prompt(old, new, pop)}"
    for i, ((old, pop), (new, reason)) in enumerate(REPLACEMENTS.items())
)}

After all 3 complete:
1. Read each new sample's status JSON from data/metadata/
2. Verify all 3 have reference = "GRCh37" and bare contig names
3. Print a final summary table:
   - Old sample → New sample | Status | Reads | Contig prefix
4. If all 3 succeeded, print:
   "✅ All replacements complete. All 15 samples now on GRCh37. Safe to proceed to embedding extraction."
   And list the final 15 samples grouped by superpopulation.
5. If any failed, print the error and suggest a fallback sample to try.

Fallback options if a replacement fails:
  EAS: NA18507, HG00096, NA18542
  SAS: HG03054, HG03742, NA20847  # NA20769 already in use as primary replacement
"""


async def run():
    print("🔄  Re-pulling 3 GRCh38 samples with GRCh37 equivalents")
    print(f"    Replacing: {', '.join(old for (old, _) in REPLACEMENTS)}")
    print(f"    With:      {', '.join(new for (new, _) in REPLACEMENTS.values())}")
    print("=" * 72)

    options = ClaudeAgentOptions(
        cwd=str(Path.cwd()),
        model="claude-opus-4-6",
        thinking={"type": "adaptive"},
        allowed_tools=["Bash", "Read", "Write", "Agent"],
        permission_mode="acceptEdits",
        max_turns=60,
        agents={
            "bam-pull-agent": AgentDefinition(
                description="Pulls a GRCh37 BAM slice from EBI for one sample, archiving the old GRCh38 file. Use one per replacement sample.",
                prompt="You are a BAM pull specialist. Execute the task given to you by the coordinator exactly as specified.",
                tools=["Bash", "Read", "Write"],
            ),
        },
    )

    async for message in query(prompt=COORDINATOR_PROMPT, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            print(f"\n{'='*72}")
            if message.result:
                print(message.result)

    # Verify final state
    print("\n📋  Final sample inventory:")
    manifest = Path("data/metadata/sample_manifest.json")
    if manifest.exists():
        with open(manifest) as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        # Fall back to reading individual status files
        statuses = sorted(Path("data/metadata").glob("*_bam_status.json"))
        print(f"{'Sample':<12} {'Pop':<6} {'Ref':<8} {'Reads':>8} {'Status'}")
        print("-" * 50)
        for s in statuses:
            if "backup" in s.name:
                continue
            d = json.loads(s.read_text())
            ref = d.get("reference", "GRCh37")  # assume GRCh37 if not specified
            print(
                f"{d['sample']:<12} {d['population']:<6} {ref:<8} "
                f"{d.get('read_count', '?'):>8} {d['status']}"
            )


if __name__ == "__main__":
    anyio.run(run)
