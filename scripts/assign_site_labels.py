"""
Cross-reference hap.py BD labels with NA12878 activation sites.

The activations are stored as a bulk matrix (N_sites, ...) in:
  data/embeddings/NA12878_mixed5/activation_cache/activations_00000000.npz

Site positions come from the make_examples tfrecord (same order as activations):
  data/embeddings/NA12878_mixed5/intermediate_results_dir/
    make_examples.tfrecord-00000-of-00001.gz

Output: results/giab_validation/site_labels.json
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── Paths ────────────────────────────────────────────────────────────────────
TFRECORD = Path(
    "data/embeddings/NA12878_mixed5/intermediate_results_dir/"
    "make_examples.tfrecord-00000-of-00001.gz"
)
ACTIVATIONS_NPZ = Path(
    "data/embeddings/NA12878_mixed5/activation_cache/activations_00000000.npz"
)
HAPPY_VCF = Path("results/giab_validation/happy_results.vcf.gz")
OUTPUT_PATH = Path("results/giab_validation/site_labels.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BCFTOOLS = (
    Path("/Users/antheaguo/anaconda3/envs/deepVariant/bin/bcftools")
    if Path("/Users/antheaguo/anaconda3/envs/deepVariant/bin/bcftools").exists()
    else "bcftools"
)

# ── 1. Extract site positions from make_examples tfrecord ────────────────────
print("Reading make_examples tfrecord for site positions…")

def _parse_locus_from_example(raw: bytes) -> str | None:
    """Return 'chr20_<pos>' string from a serialized tf.train.Example.

    DeepVariant locus format: '20:10000117-10000117'  (bare contig, start == VCF POS).
    The hap.py VCF uses 'chr20' contig names, so we add the 'chr' prefix here.
    """
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    feat = ex.features.feature

    if "locus" in feat:
        locus_bytes = feat["locus"].bytes_list.value[0].decode()
        # Format: chrom:start-end  (start value matches VCF POS directly)
        chrom, rest = locus_bytes.split(":", 1)
        start_str = rest.split("-")[0]
        pos = int(start_str)
        # Add chr prefix if missing (GRCh37 uses bare contig names)
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        return f"{chrom}_{pos}"

    return None

site_keys: list[str] = []
dataset = tf.data.TFRecordDataset(str(TFRECORD), compression_type="GZIP")
for raw_record in dataset:
    key = _parse_locus_from_example(raw_record.numpy())
    site_keys.append(key if key else "UNKNOWN")

print(f"  Found {len(site_keys)} candidate sites in tfrecord")
print(f"  First 5 site keys: {site_keys[:5]}")

# Verify alignment with activations
npz = np.load(str(ACTIVATIONS_NPZ))
layer_key = list(npz.keys())[0]
n_acts = npz[layer_key].shape[0]
print(f"  Activation matrix shape: {npz[layer_key].shape}  ({n_acts} rows)")
if n_acts != len(site_keys):
    print(
        f"  WARNING: activation rows ({n_acts}) ≠ tfrecord examples ({len(site_keys)}). "
        "Will use min of the two."
    )
n = min(n_acts, len(site_keys))
site_keys = site_keys[:n]

# ── 2. Query hap.py VCF for BD labels (QUERY sample only) ────────────────────
print(f"\nQuerying {HAPPY_VCF} for BD labels via bcftools…")

# Use -s QUERY to get only the DeepVariant call sample; use \t separator between samples
cmd = [
    str(BCFTOOLS), "query",
    "-s", "QUERY",
    "-f", "%CHROM\t%POS\t[%BD]\n",
    str(HAPPY_VCF),
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f"bcftools failed:\n{result.stderr}")

priority_map = {"TP": 3, "FP": 2, "FN": 2, "UNK": 1, "N": 1, ".": 0}
pos_to_label: dict[str, dict] = {}
for line in result.stdout.strip().splitlines():
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    chrom, pos, bd = parts[0], parts[1], parts[2].strip()
    key = f"{chrom}_{pos}"
    existing = pos_to_label.get(key, {"label": ".", "priority": -1})
    p = priority_map.get(bd, 0)
    if p > existing["priority"]:
        pos_to_label[key] = {"label": bd, "priority": p}

print(f"  Parsed {len(pos_to_label)} unique positions from VCF")
print(f"  Sample VCF entries: {list(pos_to_label.items())[:5]}")

# ── 3. Build site_labels dict ─────────────────────────────────────────────────
print("\nCross-referencing sites…")
site_labels: dict[str, str] = {}
unmatched: list[str] = []

for key in site_keys:
    if key in pos_to_label:
        site_labels[key] = pos_to_label[key]["label"]
    else:
        site_labels[key] = "UNK"
        if key != "UNKNOWN":
            unmatched.append(key)

counts = {
    lb: sum(1 for v in site_labels.values() if v == lb)
    for lb in ["TP", "FP", "FN", "UNK", "."]
}

output = {
    "n_sites": len(site_labels),
    "label_counts": counts,
    "n_unmatched_to_vcf": len(unmatched),
    "notes": (
        "site keys are chr_pos (1-based VCF POS); labels from hap.py QUERY sample BD field; "
        "UNK = site not present in hap.py VCF (outside high-conf region or not genotyped)"
    ),
    "site_labels": site_labels,
}

OUTPUT_PATH.write_text(json.dumps(output, indent=2))

summary = {k: v for k, v in output.items() if k != "site_labels"}
print(json.dumps(summary, indent=2))
print(f"\nLabels written to {OUTPUT_PATH}")
