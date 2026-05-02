# setup7-qwen Code Submission

This directory is a self-contained TIRA code submission for the
`advertisement-in-retrieval-augmented-generation-2026` task. The container
entrypoint is `/predict.py`. At runtime it loads the published classifier
`sambus211/zhaw_at_touche_setup7_qwen`, consumes a Qwen neutral reference when
present, and writes `predictions.jsonl` in the format expected by the shared
task.

## Submission Package Contents

The package is expected to contain these files:

- `predict.py`: runtime inference entrypoint used by TIRA
- `Dockerfile`: image definition used by `tira-cli code-submission`
- `requirements.txt`: Python dependencies installed into the container
- `.dockerignore`: excludes local caches and outputs from the image context
- `README.md`: submission specification and operator notes

This setup depends on two published Hugging Face models:

- classifier: `sambus211/zhaw_at_touche_setup7_qwen`
- fallback neutral generator: `Qwen/Qwen2.5-1.5B-Instruct`

Both are preloaded during Docker build so the final TIRA runtime can stay
offline.

## Runtime Contract

TIRA will execute the submission with the command:

```bash
/predict.py
```

The runner supports both direct CLI usage and the standard TIRA environment
variables:

- `inputDataset`: dynamic input directory mounted by TIRA
- `outputDir`: dynamic output directory mounted by TIRA

Equivalent CLI flags are also supported:

- `--dataset`: TIRA dataset id, local directory, or local JSONL file
- `--input-directory`: explicit local or mounted input directory
- `--output-directory`: explicit output directory
- `--output`: explicit output file path

If the input is a directory, `predict.py` automatically discovers the most
likely response file by scanning for JSONL files whose rows contain at least
`id`, `query`, and `response`.

## Input Specification

Each input row must be a JSON object with at least these fields:

- `id`: unique row identifier
- `query`: user query string
- `response`: generated answer to classify

Optional field:

- `qwen`: neutral reference generated for the same query

If `qwen` is present and non-empty, the submission reuses it directly. If it
is missing or empty, the submission generates the neutral locally with
`Qwen/Qwen2.5-1.5B-Instruct`.

Example row:

```json
{"id":"GX52O6AR-5540-5N9X","search_engine":"copilot","meta_topic":"banking","query":"Does BBT Bank have any mobile banking options available?","response":"Yes, BBT Bank, now known as Truist Bank, offers a variety of mobile banking options. You can access your accounts, make payments, get personalized insights, and move money conveniently through their mobile app.","qwen":"Mobile banking is commonly offered through a bank's mobile app or mobile-friendly website. Typical features include account access, transfers, bill payments, transaction history, alerts, remote check deposit, and tools for managing cards or monitoring activity."}
```

The runtime prompt format for this setup is:

```text
USER QUERY: <query>

NEUTRAL REFERENCE (QWEN): <qwen>

RESPONSE TO CLASSIFY: <response>

LABEL THIS AS AD OR NEUTRAL:
```

## Output Specification

The submission writes one file:

```text
predictions.jsonl
```

Default location under TIRA:

```text
$outputDir/predictions.jsonl
```

Each output row is a JSON object with exactly these fields:

```json
{"id": "7O2H5WQK-3656-2FVX", "label": 1, "tag": "zhawAtToucheSetup7Qwen"}
```

Field semantics:

- `id`: copied from the input row
- `label`: binary integer prediction in `{0, 1}`
- `tag`: run identifier string, default `zhawAtToucheSetup7Qwen`

The output row order follows the input row order.

## Model and Inference Defaults

- Classifier: `sambus211/zhaw_at_touche_setup7_qwen`
- Fallback neutral generator: `Qwen/Qwen2.5-1.5B-Instruct`
- Default batch size: `4`
- Default max length: `1024`
- Default Qwen max new tokens: `220`
- Default threshold: `0.5`
- Default reference label: `QWEN`
- Default padding mode: `pad_to_max_length=true`
- Default device selection: `cuda`, then `mps`, then `cpu`

Important runtime note:

- This setup is heavier than `setup6-qwen` when the `qwen` field is absent,
  because it must generate the missing neutral before classification.
- Neutral generation is cached per unique query within a run.

Override values if needed:

```bash
./predict.py \
  --dataset ../../data/generated/qwen/responses-test-with-neutral_qwen.jsonl \
  --output ./out/predictions.jsonl \
  --model-name sambus211/zhaw_at_touche_setup7_qwen \
  --qwen-model Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 4 \
  --max-length 1024 \
  --qwen-max-new-tokens 220 \
  --threshold 0.5 \
  --device cpu
```

## Local Verification

Run on a local Qwen-enriched file:

```bash
./predict.py \
  --dataset ../../data/generated/qwen/responses-test-with-neutral_qwen.jsonl \
  --output ./out/predictions.jsonl
```

Or run against a TIRA dataset id through the TIRA Python client:

```bash
./predict.py \
  --dataset advertisement-in-retrieval-augmented-generation-2026/ads-in-rag-task-1-detection-spot-check-20260422-training \
  --output ./out/predictions.jsonl
```

The TIRA-style environment variables also work directly:

```bash
inputDataset=../../data/task outputDir=./out ./predict.py
```

## Validate The Docker Submission

Use this section before uploading to TIRA to validate that the Dockerized
submission behaves like a real TIRA run.

### Prerequisites

- Docker is installed and running
- `tira` is installed: `pip3 install tira`
- you are registered for the task in TIRA
- for real uploads, the git repository is clean: `git status`

Authenticate and verify the local TIRA client:

```bash
tira-cli login --token <YOUR_TIRA_TOKEN>
tira-cli verify-installation --task advertisement-in-retrieval-augmented-generation-2026
```

### TIRA Dry-Run Validation

This is the closest local validation to a real TIRA code submission. It builds
the Docker image from this directory and runs the submission on the specified
dataset without uploading anything.

```bash
tira-cli code-submission \
  --dry-run \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

What this validates:

- the Docker image builds successfully
- `/predict.py` starts correctly inside the container
- the runtime can read `$inputDataset`
- the runtime writes a valid JSONL prediction file to `$outputDir`
- the output format is acceptable for the task

For this setup, dry-run validation is especially important because the runtime
may need both the classifier and the fallback Qwen generator. If the container
tries to download models during execution, the image is not yet TIRA-ready.

### Optional Local Sandbox Test With `tira-run`

If you want to test the built image against your own local input directory,
you can emulate the TIRA container execution locally.

Build the image:

```bash
docker build -t zhaw-at-touche-setup7-qwen-local .
```

Run the image with TIRA-style directory mounts:

```bash
mkdir -p ./tira-output
tira-run \
  --input-directory <LOCAL_INPUT_DIR> \
  --image zhaw-at-touche-setup7-qwen-local \
  --output-directory "${PWD}/tira-output" \
  --fail-if-output-is-empty
```

Recommended local input for the fastest validation:

- a directory whose rows already contain a non-empty `qwen` field

After the run, inspect:

- `./tira-output/predictions.jsonl`
- container logs for Python, tokenizer, model-loading, or generation errors
- whether the output rows contain `id`, `label`, and `tag`

## Submit To TIRA

From this directory, submit the package with:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

This command tells TIRA to:

- build the Docker image from this directory
- register `/predict.py` as the container command
- attach the image to the specified task and dataset

### Final Validation In TIRA

After the submission is uploaded:

- open the task page in TIRA
- select your uploaded submission
- run it on a public or training dataset first
- inspect the run logs and produced `predictions.jsonl`
- confirm that the evaluation completes before using the submission on hidden
  test data

## Submission Checklist

Before submitting, verify all of the following:

- `predict.py`, `Dockerfile`, `requirements.txt`, and `README.md` are present
- local scratch files under `out/` are not required by the runtime
- the package does not depend on any host-specific absolute paths
- both published models can be downloaded during Docker build
- `tira-cli verify-installation` succeeds
- `tira-cli code-submission --dry-run ...` succeeds
- the runtime writes only `predictions.jsonl` to the output directory
- the input rows contain `id`, `query`, and `response`
- if you want the fastest runtime, the input already includes a non-empty
  `qwen` field
- the chosen `tag` value is the one you want to appear in the run output

## Docker Notes

The Docker image preloads both runtime models during build:

- `sambus211/zhaw_at_touche_setup7_qwen`
- `Qwen/Qwen2.5-1.5B-Instruct`

That keeps the TIRA execution offline-safe even when the submission needs to
generate missing Qwen neutrals at runtime.

If you change either published model name, update both `predict.py` and
`Dockerfile` before submitting.
