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

Install all at once:
```bash
conda activate deepVariant
pip install "tensorflow==2.13.0" ml_collections etils
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

## Docker General Lessons

- **Always mount every host path the container needs.** Docker containers have no access to the host filesystem by default; a missing `-v` silently results in file-not-found errors inside the container.
- **Mount paths must match exactly**, including case, because the mount is used as-is inside the container. On macOS, rely on `$PWD` (after `cd`-ing to the correct directory) rather than typing paths by hand.
- **Check `docker system prune`** periodically. Accumulated images and stopped containers consume significant disk space and can cause `ENOSPC` errors.
- **DeepVariant binaries are Bazel-compiled.** The `.py` source files inside the container are only accessible at runtime under `/tmp/Bazel.runfiles_*/`. You cannot `find` them in a fresh container invocation; run the binary and inspect `/tmp/Bazel.runfiles_*` from the same shell to read source.
- **Output filename conventions matter.** Tools like `postprocess_variants` that handle sharded file specs have strict expectations about filename patterns. Always check what format a downstream tool expects before writing output files from a custom step.
