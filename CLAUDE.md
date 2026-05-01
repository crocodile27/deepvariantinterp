# DeepVariant Interp — Project Notes

## Pipeline Overview

Three-step pipeline defined in `scripts/run_deepvariant_hooked.sh`:

1. **make_examples** — runs inside Docker (`google/deepvariant:1.9.0`); requires C++ extensions
2. **call_variants** — runs natively via `scripts/call_variants_hooked.py`; captures activations
3. **postprocess_variants** — runs inside Docker; produces final VCF

Activations are written to `<output_dir>/activation_cache/` as `.npz` files.

## Running the Pipeline

```bash
cd ~/Documents/Code/deepvariantinterp
export INPUT_DIR="${PWD}/quickstart-testdata"

bash scripts/run_deepvariant_hooked.sh \
  --ref "${INPUT_DIR}/ucsc.hg19.chr20.unittest.fasta" \
  --reads "${INPUT_DIR}/NA12878_S1.chr20.10_10p1mb.bam" \
  --output_dir "${PWD}/quickstart-output" \
  --checkpoint "${PWD}/model/wgs/deepvariant.wgs.ckpt" \
  --regions "chr20:10,000,000-10,010,000" \
  --hook_layers mixed5
```

Always `cd` to the repo root first and derive `INPUT_DIR` from `$PWD` to avoid path case issues (see bugs below).

## Conda Environment

Python deps live in the `deepVariant` conda env. Activate with `conda activate deepVariant`.

Required packages beyond the defaults:
- `tensorflow==2.13.0` (2.12 is missing `tf.keras.metrics.F1Score`)
- `ml_collections`
- `etils`
- `umap-learn`, `matplotlib`, `scikit-learn` (for `plotting/umap_activations.py`)

Install pipeline deps:
```bash
conda activate deepVariant
pip install "tensorflow==2.13.0" ml_collections etils
```

Install UMAP/plotting deps (must use conda — pip fails to build `llvmlite` from source):
```bash
conda install -n deepVariant -c conda-forge umap-learn matplotlib scikit-learn -y
```

## Proto Compilation

The repo contains `.proto` source files but not the compiled `_pb2.py` Python bindings. These must be generated once before running the native Python steps. `protoc` is available inside the `deepVariant` conda env.

```bash
cd ~/Documents/Code/deepvariantinterp
conda run -n deepVariant protoc --proto_path=. --python_out=. \
  deepvariant/protos/deepvariant.proto \
  deepvariant/protos/realigner.proto \
  deepvariant/protos/resources.proto \
  third_party/nucleus/protos/position.proto \
  third_party/nucleus/protos/reads.proto \
  third_party/nucleus/protos/variants.proto \
  third_party/nucleus/protos/range.proto \
  third_party/nucleus/protos/struct.proto \
  third_party/nucleus/protos/cigar.proto \
  third_party/nucleus/protos/reference.proto \
  third_party/nucleus/protos/bed.proto \
  third_party/nucleus/protos/bedgraph.proto \
  third_party/nucleus/protos/fasta.proto \
  third_party/nucleus/protos/fastq.proto \
  third_party/nucleus/protos/feature.proto \
  third_party/nucleus/protos/gff.proto \
  third_party/nucleus/protos/example.proto
```

Re-run if you pull proto changes from upstream.

## Bugs Encountered and Fixed

### 1. Path case sensitivity (`/Code` vs `/code`)

macOS is case-insensitive by default, so `~/Documents/code` and `~/Documents/Code` both resolve locally. Docker volume mounts are case-sensitive and will silently mount a non-existent path. Always use the exact case (`Code`) and derive all paths from `$PWD` after `cd`-ing to the repo root.

### 2. Missing Docker volume mounts (Steps 1 and 3)

The original script only mounted `/usr/lib/locale/`. The Docker container cannot see host filesystem paths unless they are explicitly mounted with `-v`. Both `make_examples` (Step 1) and `postprocess_variants` (Step 3) need mounts for the input data directory and output directory.

Fix: added `-v "${INPUT_DIR}:${INPUT_DIR}" -v "${OUTPUT_DIR}:${OUTPUT_DIR}"` to both docker run commands.

### 3. Typo: `docker run -` instead of `docker run -v` (Step 3)

The original Step 3 had `docker run - /usr/lib/locale/:/usr/lib/locale/` which is invalid and produces "invalid reference format". Fixed to `docker run -v /usr/lib/locale/:/usr/lib/locale/`.

### 4. CVO output filename must use sharded format

`postprocess_variants` 1.9.0 does not accept a plain `call_variants_output.tfrecord.gz`. It globs for `call_variants_output*`, counts the matches, infers an `@N` sharded spec, and then checks that the glob results exactly match what `@N` expands to (`call_variants_output-00000-of-00001.tfrecord.gz`). If the actual filename is `call_variants_output.tfrecord.gz` the check fails with:

```
ValueError: Found multiple file patterns in input filename space
```

Fix:
- `call_variants_hooked.py` (or the caller) must write to `call_variants_output-00000-of-00001.tfrecord.gz`
- Pass `--infile call_variants_output@1.tfrecord.gz` (sharded spec) to `postprocess_variants`

The script now uses `CVO_OUTPUT` for the actual filename and `CVO_SPEC` (with `@${NUM_SHARDS}`) for the `--infile` argument.

### 5. `--gvcf_outfile` requires `make_examples` to have been run with `--gvcf`

Passing `--gvcf_outfile` to `postprocess_variants` without also passing `--nonvariant_site_tfrecord_path` causes an immediate error. The nonvariant site tfrecords are produced by `make_examples` only when `--gvcf` is specified. The pipeline currently does not generate gVCFs; `--gvcf_outfile` has been removed from Step 3.

### 6. Missing Python dependencies in conda env

`ml_collections` and `etils` are not installed by default. The import chain fails at module load time, not at call time, so the error appears before any inference runs.

### 7. TensorFlow version: 2.12 → 2.13 required

`tf.keras.metrics.F1Score` was added in TF 2.13. The `deepvariant/metrics.py` file defines a subclass of it at class body level, causing an `AttributeError` on import with TF 2.12.

### 8. Proto `_pb2.py` files not committed

The repo only ships `.proto` source files. Running native Python code that imports `deepvariant.protos.deepvariant_pb2` (or any nucleus proto) will fail with `ImportError: cannot import name '..._pb2'` until the protos are compiled with `protoc`. See the Proto Compilation section above.

### 9. Malformed `~/.docker/config.json`

Missing comma after `"auths": {}` caused Docker to print a parse warning on every run. Not fatal but noisy. Fixed by adding the comma.

### 10. `conda run` strips environment variables

`conda run -n deepVariant python script.py` does **not** inherit `ANTHROPIC_API_KEY` (or other env vars) from the parent shell. All API calls fail immediately with "Could not resolve authentication method."

Fix — pass the key explicitly:
```bash
conda run -n deepVariant env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" python script.py
```

Or set it permanently in the env (survives across shell sessions):
```bash
conda env config vars set -n deepVariant ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
# reactivate to pick it up
conda activate deepVariant
```

### 11. Hardcoded absolute path in `run_analysis.py`

`run_analysis.py` originally set `BASE = "/Users/antheaguo/Documents/Code/deepvariantinterp"`. Fixed to:
```python
BASE = str(Path(__file__).parent)
```

## Ancestry Embedding Dataset

### Data source
`gs://brain-genomics-public/research/cohort/1KGP/grch37_bams/`

NYGC 30x 1KGP phase 3 cohort (2,504 samples), all remapped to **GRCh37/hs37d5**.
Coverage-matched across builds — eliminates the 4x (GRCh37) vs 30x (GRCh38) confound.
Reference uses **bare contig names** (`20`, not `chr20`). Region args must match.

Paper: Yun et al., Bioinformatics 2021 (DeepVariant + GLnexus cohort calls).

### 15 planned samples → 13 active samples

The original 15-sample plan (3 per superpop) was reduced to **13 active samples** for the joint analysis. NA19238 and HG03022 were excluded per the collaborating lab's decision to keep the GRCh37 and GRCh38 sets fully matched in sample count and avoid close relatedness.

**Active 13 samples** (used in `orchestrate_joint_analysis.py`):

| Sample | Superpop | Pop | ERR accession | Notes |
|--------|----------|-----|---------------|-------|
| HG01985 | AFR | ACB | ERR3242296 | replaces NA19240 (child of NA19238) |
| HG02922 | AFR | ESN | ERR3242482 | |
| HG01048 | AMR | PUR | ERR3241761 | |
| HG01197 | AMR | PUR | ERR3241854 | |
| HG01565 | AMR | PEL | ERR3241984 | |
| HG00759 | EAS | CDX | ERR3242128 | replaces HG00514 (absent from cohort) |
| NA18939 | EAS | JPT | ERR3239557 | |
| HG00864 | EAS | CDX | ERR3242132 | |
| NA12878 | EUR | CEU | ERR3239334 | |
| HG00731 | EUR | PUR | ERR3241754 | |
| NA20502 | EUR | TSI | ERR3239785 | |
| HG03009 | SAS | BEB | ERR3242842 | replaces HG03732 (absent from cohort) |
| NA20847 | SAS | GIH | ERR3239999 | |

Excluded: NA19238 (AFR/YRI, ERR3239453) and HG03022 (SAS/PJL, ERR3243053).

Relatedness verified via `20130606_g1k.ped`. All are `rel=unrel` with no shared family IDs.

GCS path pattern: `gs://brain-genomics-public/research/cohort/1KGP/grch37_bams/ERR<run>_<sample>.grch37.bam`

## Joint Analysis (GRCh37 + GRCh38)

`orchestrate_joint_analysis.py` orchestrates the full cross-reference analysis using the Anthropic API to run parallel embedding agents.

### Architecture

```
Phase A: 13 GRCh37 embedding agents (parallel)
  → reheader BAM (20 → chr20), make_examples, call_variants_hooked, write status.json

Phase B: poll data/embeddings/grch38/ every 60s for collaborator's GRCh38 .npz files

Barrier: validate all 26 embedding dirs present

Phase C: aggregate agent — concatenate embeddings, tag ref + pop
  → results/joint_analysis/combined_embeddings.npz

Phase D: joint analysis agent — PCA + UMAP, silhouette scores, centroid distances
  → results/joint_analysis/umap_pop_x_ref.png
  → results/joint_analysis/pca_pc1_pc2_joint.png
  → results/joint_analysis/joint_summary.json
```

### Usage

```bash
cd ~/Documents/Code/deepvariantinterp

# Run only GRCh37 embedding phase (Phase A)
conda run -n deepVariant env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  python orchestrate_joint_analysis.py --grch37-only

# Skip GRCh37 (already done), run joint analysis after GRCh38 arrives
conda run -n deepVariant env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  python orchestrate_joint_analysis.py --skip-grch37

# Full pipeline (skip GRCh38 wait if files not yet present)
conda run -n deepVariant env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  python orchestrate_joint_analysis.py --skip-grch38-wait
```

### Input BAM convention

Phase A agents expect BAMs at `data/bams/{sample_id}.slice.bam` with **bare contig names** (SN:20, not SN:chr20). The agent reheaders to chr-prefixed before running make_examples against the hg19 reference.

### Key output: joint_summary.json

Includes `key_diagnostic`: `"population_dominates"` if superpopulation silhouette > reference silhouette (embeddings cluster by ancestry across references), or `"reference_dominates"` if the opposite (reference build is the primary signal).

## Docker General Lessons

- **Always mount every host path the container needs.** Docker containers have no access to the host filesystem by default; a missing `-v` silently results in file-not-found errors inside the container.
- **Mount paths must match exactly**, including case, because the mount is used as-is inside the container. On macOS, rely on `$PWD` (after `cd`-ing to the correct directory) rather than typing paths by hand.
- **Check `docker system prune`** periodically. Accumulated images and stopped containers consume significant disk space and can cause `ENOSPC` errors.
- **DeepVariant binaries are Bazel-compiled.** The `.py` source files inside the container are only accessible at runtime under `/tmp/Bazel.runfiles_*/`. You cannot `find` them in a fresh container invocation; run the binary and inspect `/tmp/Bazel.runfiles_*` from the same shell to read source.
- **Output filename conventions matter.** Tools like `postprocess_variants` that handle sharded file specs have strict expectations about filename patterns. Always check what format a downstream tool expects before writing output files from a custom step.
