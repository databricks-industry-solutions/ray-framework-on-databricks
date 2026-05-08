# Video Batch Inference on Ray + vLLM

End-to-end accelerator for distributed Vision Language Model (VLM) batch inference over video, using Ray Core to orchestrate CPU + GPU stages and vLLM for efficient large-model serving on Databricks.

The worked example processes the public [Kaggle CCTV anomaly dataset](https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance/data), runs `Qwen/Qwen2.5-VL-32B-Instruct` over the shoplifting subset, and structures the prose output into a queryable Delta table via Databricks AI Functions.

## Run order

| # | Notebook | Cluster | Time (small subset) |
|---|---|---|---|
| 00 | `00 - Download Video Data.py` | CPU classic (`Standard_E96ads_v5`, ~33 DBU/h) | ~25 min |
| 01 | `01 - Download and Register Models.py` | CPU classic (`Standard_E96ads_v5`, ~33 DBU/h) | ~15 min |
| 02 | `02 - Batch Video Querying on Ray + VLLM.py` | GPU classic (`Standard_NC96ads_A100_v4`, ~44 DBU/h) | ~20 min on 2 GB shoplifting subset |
| 03 | `03 - Structured Entity Extraction.py` | Any classic / SQL warehouse | ~5 min |

Run them in order. Output of 00 → input of 02. Output of 01 → loaded by 02. Output of 02 → input of 03.

## Prerequisites

- **Unity Catalog** enabled in the workspace.
- **GPU compute quota** for `Standard_NC96ads_A100_v4` (Azure) or equivalent (AWS `p4d.24xlarge`).
- **Kaggle API token** in either:
  - Databricks secret scope `kaggle` with keys `username` + `key` (recommended), or
  - `~/.kaggle/kaggle.json` on the cluster driver
- **`databricks-meta-llama-3-3-70b-instruct`** foundation model API endpoint enabled (used by `03`).

## Spark configs (notebook 02 only)

Set on the GPU cluster before starting:

```
spark.databricks.pyspark.dataFrameChunk.enabled true
spark.task.resource.gpu.amount 0
```

## Widgets

All four notebooks share a common widget shape so a single set of parameter values flows through the whole pipeline.

| Widget | Default | Used in | Notes |
|---|---|---|---|
| `CATALOG` | `main` | 00, 01, 02, 03 | UC catalog. |
| `SCHEMA` | `default` | 00, 01, 02, 03 | UC schema. |
| `VOLUME` | `transcribe-video` | 00 | Where raw video bytes land. |
| `MODEL_NAME` | `qwen2_5_vl-32b` | 01, 02 | UC model name suffix. |
| `MODEL_ALIAS` | `production` | 02 | MLflow alias. **Lowercase** — aliases are case-sensitive. |
| `INPUT_TABLE` | `videos_file_reference` (in 02), `video_inferences` (in 03) | 02, 03 | Source Delta table name. |
| `OUTPUT_TABLE` | `video_inferences` (in 02), `video_events` (in 03) | 02, 03 | Destination Delta table name. |
| `CATEGORY_FILTER` | `shoplifting` | 02 | Path-derived category to restrict the run to. |
| `MAX_DURATION_SECONDS` | `1200` | 02 | Skip pathologically long videos. |
| `LLM_ENDPOINT` | `databricks-meta-llama-3-3-70b-instruct` | 03 | Foundation model API endpoint name. |

## Adapting the pattern to a different use case

Change three things only — the architecture stays the same:

1. **The prompt** — `QUERY` constant at the top of `02 - Batch Video Querying on Ray + VLLM.py`.
2. **The JSON schema** — body of the `responseFormat` argument in `03 - Structured Entity Extraction.py`, plus the matching `schema_of_json` shape.
3. **The category filter / source data** — point `CATEGORY_FILTER` and `INPUT_TABLE` at your own reference table.

The Ray DAG, the vLLM tuning, the model registration, and the AI Functions structuring step are all use-case agnostic.

## Architecture

```
[Source video] → UC Volume → Delta reference (00) → MLflow @production (01)
                                                  ↓
                                   ┌──────── Ray on Databricks ────────┐
                                   │ ConvertToPrompt (CPU) ────────┐  │
                                   │   ffmpeg · 16 frames @ 1 fps   │  │
                                   │   concurrency=(10, 24)         │  │
                                   │                ↓               │  │
                                   │ QwenVideoProcessing (GPU, vLLM)│  │
                                   │   tensor_parallel_size=2       │  │
                                   │   gpu_memory_utilization=0.95  │  │
                                   │   kv_cache_dtype=fp8           │  │
                                   │   concurrency=(2, 12)          │  │
                                   └────────────────────────────────┘  │
                                                  ↓
                              Delta raw inferences (02)
                                                  ↓
                              AI Functions · ai_query · responseFormat (03)
                                                  ↓
                              Delta structured events
```

## Troubleshooting

- **`Alias 'Production' not found`** — alias casing mismatch. Use `production` (lowercase) consistently.
- **Kaggle download returns HTML / 401** — credentials missing. See Prerequisites.
- **`OOM during vLLM init`** — drop `gpu_memory_utilization` from 0.95 to 0.85 in `02`, lower `max_num_seqs`, or reduce `max_pixels`.
- **`No videos matched filters`** — check that `CATEGORY_FILTER` matches one of the subdirectory names under your raw video path (e.g., `shoplifting`, `assault`, `arson`).
- **Long-tail videos crash actors** — wider issue if the `MAX_DURATION_SECONDS` filter is too permissive; tighten or pre-segment.
- **Model registration step too slow** — first run is slow because the 60 GB Qwen weights must download from HF. Subsequent runs use the local cache.

## Sibling examples

This is example `3_Video_Batch_Inference_Ray_Core` in the broader `Batch_AI_Inference/` directory. The siblings demonstrate the same Ray + vLLM batch pattern across other modalities:

- `1_SLM_Batch_Inference_Ray_Core/` — image-to-text with a small language model.
- `2_Automatic_Speech_Recognition_Ray_Core/` — audio transcription with Whisper.

The Ray DAG, governance posture, and output shape are symmetric across all three.
