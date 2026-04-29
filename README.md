# setup7-qwen Code Submission

This directory contains a TIRA-style code submission for `setup7-qwen`.

The runner loads the Longformer classifier from
`sambus211/zhaw_at_touche_setup7_qwen`, generates a local Qwen neutral
reference when the input does not already contain a non-empty `qwen` field,
and then scores `query + neutral + response`.

Each output row has the required format:

```json
{"id": "7O2H5WQK-3656-2FVX", "label": 1, "tag": "zhawAtToucheSetup7Qwen"}
```

## Expected Input

The input rows are expected to contain at least `id`, `query`, and `response`.
If a non-empty `qwen` field is present, the runner reuses it by default.
Otherwise it generates the neutral locally with `Qwen/Qwen2.5-1.5B-Instruct`.

Example:

```json
{"id":"GX52O6AR-5540-5N9X","search_engine":"copilot","meta_topic":"banking","query":"Does BBT Bank have any mobile banking options available?","response":"Yes, BBT Bank, now known as Truist Bank, offers a variety of mobile banking options. You can access your accounts, make payments, get personalized insights, and move money conveniently through their mobile app."}
```

## Local Run

Run on a local input directory or JSONL file:

```bash
./predict.py \
  --dataset ../../data/task \
  --output ./out/predictions.jsonl
```

Or run directly against a TIRA dataset id via the TIRA Python client:

```bash
./predict.py \
  --dataset advertisement-in-retrieval-augmented-generation-2026/ads-in-rag-task-1-detection-spot-check-20260422-training \
  --output ./out/predictions.jsonl
```

The code submission entrypoint also supports the standard TIRA runtime
environment variables directly:

```bash
inputDataset=../../data/task outputDir=./out ./predict.py
```

## Runtime Notes

- Runtime prompt format is `query + qwen neutral + response`.
- The default classifier is `sambus211/zhaw_at_touche_setup7_qwen`.
- The default local neutral generator is `Qwen/Qwen2.5-1.5B-Instruct`.
- Classification defaults match the repo setup: `max_length=1024`,
  `pad_to_max_length=true`.
- Neutral generation is cached per unique query within a run.

## TIRA Submission

Submit this directory as a code submission:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

## Notes

- This submission is slower and heavier than `setup6-qwen` because it must
  generate the Qwen neutral before classification when the neutral is absent.
- The Docker image preloads both the classifier and the local Qwen generator
  during build so execution remains offline-safe inside TIRA.
