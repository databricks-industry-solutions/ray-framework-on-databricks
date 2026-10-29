# Video Batch Inference on Serverless GPU (Ray + vLLM)

Serverless port of [`3_Video_Batch_Inference_Ray_Core`](../3_Video_Batch_Inference_Ray_Core). It runs a vision-language model (Qwen2.5-VL) over a folder of CCTV videos with Ray Data and vLLM, turns the model's descriptions into a structured Delta table with AI Functions, and produces a face-redacted frame for presentations.

No classic GPU cluster, init scripts, or `apt-get` are needed. Only notebook 02 uses a GPU (one H100 on serverless GPU compute).

## Notebooks

| Notebook | Compute | What it does | Output | Typical time |
| --- | --- | --- | --- | --- |
| `00 - Download Video Data` | Serverless CPU | Downloads the Kaggle CCTV dataset (one category by default) into a UC Volume and records each video's duration with PyAV | `videos_file_reference` table | ~10 min per category |
| `01 - Download and Register Models` | Serverless CPU | Downloads the Qwen2.5-VL weights and registers them in Unity Catalog with the `production` alias | UC model `qwen2_5_vl-7b` | ~30 min (7B or 32B) |
| `02 - Batch Video Inference on Ray + vLLM` | **Serverless GPU, H100**, base environment **Databricks AI** (v6) | Samples frames at ~1 fps (at most 64), runs them through vLLM in batches | `video_inferences` table | 7B: ~4 min staging + ~3 min for 49 videos; 32B: ~16 min staging + ~3.5 min |
| `03 - Structured Entity Extraction` | Serverless CPU or SQL warehouse | `ai_query` with a strict JSON schema turns each description into incident fields (assessment, confidence, location, time range, suspects) | `video_events` table | a few minutes |
| `04 - Redacted Frame Extraction` | Serverless CPU | Extracts one frame, pixelates detected faces (OpenCV YuNet + Haar), saves a PNG | PNG in the Volume | < 1 min |

Run them in order. Every notebook has `CATALOG` / `SCHEMA` / `VOLUME` widgets (default `main.vlm_video`, volume `transcribe-video`); use the same values in all five. Notebook 00 creates the schema and volume if they don't exist.

When comparing models or categories, give each run its own `OUTPUT_TABLE` in 02 and 03 (for example `video_inferences_32b` / `video_events_32b`) so runs don't overwrite each other.

## Prerequisites

* **Kaggle API token** in a secret scope `kaggle` with keys `username` and `key` (or `~/.kaggle/kaggle.json`).
* **Serverless GPU** with H100 available in the workspace.
* **Foundation Model API** endpoint `databricks-meta-llama-3-3-70b-instruct` (set `LLM_ENDPOINT` in 03 to use another).
* Permission to create a schema, volume, tables, and a registered model in the target catalog.

## Setting up notebook 02 compute

1. **Connect** -> **Serverless GPU**.
2. In the **Environment** panel, set **Accelerator** to **H100** and the base environment to **Databricks AI** (v6), then apply.
3. Run the install cell. It installs only `av`; torch, vLLM, Ray, and transformers come with the environment.

Set `MAX_VIDEOS` to 3 for a quick smoke test.

## Results (1xH100, Databricks AI v6)

Same 49 shoplifting videos (under 1200 s), same prompt, notebook 03 for the assessments.

| | Qwen2.5-VL-7B (bf16) | Qwen2.5-VL-32B (fp8) |
| --- | --- | --- |
| Videos processed | 49 / 49, 0 decode failures | 49 / 49, 0 decode failures |
| Inference time (49 videos) | ~2.7 min | ~3.5 min |
| Weight staging per new GPU session | ~4 min | ~16 min |
| Average output tokens | 384 | 648 |
| Cut off at `max_tokens` (1024) | 3 | 2 |
| CONFIRMED / LIKELY / POSSIBLE / NORMAL / INSUFFICIENT_EVIDENCE | 2 / 12 / 10 / 22 / 3 | 35 / 10 / 2 / 2 / 0 |
| **Flagged (CONFIRMED or LIKELY)** | **14 / 49 (29%)** | **45 / 49 (92%)** |

**False positives (32B).** On 60 clips from the dataset's `normal` folder, 32B flagged 6 (10%) as CONFIRMED or LIKELY (4 CONFIRMED, 2 LIKELY, 3 POSSIBLE, 2 INSUFFICIENT_EVIDENCE, 49 NORMAL). About two thirds of those clips are offices, streets, or hallways; on the ~24 that look like shops, about 5 were flagged (~20%). The setting comes from the model's `location` field, so that split is approximate. Two of the six flagged clips were watched by eye and both are genuine false positives: a parcel delivery being signed for in an office (reported as "repeated concealment" and "a confrontation with staff") and a woman waiting with a drink in a restaurant (reported as "concealed an item in her bag"). Both were CONFIRMED with confidence 1.0.

**Spot check.** Four clips that 7B called NORMAL and 32B called CONFIRMED were watched by eye. All four were real thefts, so the verdicts held up. The details did not: the item was right in about one of four and the time range in about one of four. In one clip the person picks up a tablet as cover and takes a phone; 32B reported the tablet.

**Takeaway.** 32B is much better at spotting theft than 7B at little extra cost per video, but treat its output as triage that points reviewers at footage, not as a record of what was taken or when. In a real store theft is rare, so most flags will be false alarms. Do not filter on `confidence`: invented incidents score as high as real ones.

These numbers come from a public dataset that mixes shops, offices, streets, and restaurants, filmed at 320 px and sampled at ~1 fps, with a generic shoplifting prompt. They are a baseline rather than a ceiling: on a customer's own footage, with site context in the prompt (store layout, normal staff behaviour) and a small labelled evaluation set, the same pipeline can be tuned and measured the same way.

## Using the 32B model

1. In 01, set `HF_MODEL_ID` = `Qwen/Qwen2.5-VL-32B-Instruct`, `MODEL_NAME` = `qwen2_5_vl-32b`, and `STAGING_DIR` to a Volume path (for example `/Volumes/main/vlm_video/transcribe-video/hf_staging`). The weights are ~68 GB; logging makes a second copy, and both don't fit on local disk. Measured: ~5 min download, ~21 min to log and register. If the runtime detaches during registration, re-run from the widgets cell.
2. In 02, set `MODEL_NAME` = `qwen2_5_vl-32b` and one of:
   * `QUANTIZATION` = `fp8` on 1xH100 (weights quantized at load time). **Validated:** engine start ~70 s (first time, including compilation), 46 GiB left for the KV cache.
   * `TENSOR_PARALLEL` = `2` on a multi-GPU node (bf16, sharded over two GPUs per vLLM engine). Not yet tested.

For a live demo, 7B is the practical choice; a cold 32B start spends ~16 min copying weights to the GPU before inference begins.

## Measuring false positives

Every clip in `shoplifting/` is labelled as theft, so that folder alone cannot show whether the model over-flags. To test on footage without theft:

1. In 00, set `CATEGORY_FILTER` = `normal`, `FILE_PREFIX` = `Normal_Videos`, `MAX_FILES` = `60`. The `normal` folder has 950 files with gaps in the numbering; `MAX_FILES` downloads a sample. Existing categories stay in `videos_file_reference`.
2. In 02, set `CATEGORY_FILTER` = `normal` and a new `OUTPUT_TABLE` (for example `video_inferences_32b_neg`).
3. In 03, point `INPUT_TABLE` / `OUTPUT_TABLE` at the matching tables (for example `video_events_32b_neg`). The share of rows assessed CONFIRMED or LIKELY is the false-positive rate.

## Differences from the classic version

* **Correct timestamps.** Each prompt gives vLLM the real sampling rate (`video_metadata`), and the prompt states the video length and frame interval. The classic notebook always passed `fps=1`.
* **Real batching.** Each `generate()` call processes a batch of videos.
* **No system ffmpeg.** PyAV replaces `ffmpeg` / `ffprobe`.
* **Model registration without loading it.** 01 downloads the weight files and registers them as a pyfunc artifact. The classic version loaded the model into memory on a 672 GB driver.
* **Structured output fix.** 03 lists every field in `required` and sets `strict` on `json_schema`. Before, `location`, `time_range`, and `confidence` were empty on every row. (The classic notebook now has the same fix.)
* **No MediaPipe.** MediaPipe pins `numpy<2` and an old protobuf, which crashes the serverless kernel. 04 uses OpenCV YuNet and Haar cascades on a 3x upscaled frame instead.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `undefined symbol` from flash-attn, or vLLM/torch version mismatches | Something `%pip install`ed vLLM, torch, or Ray over the AI environment. **New Session does not undo it.** Uninstall those packages, or reset the environment in the Environment panel. |
| `FileNotFoundError: 'ninja'` or a FlashInfer build error at engine start | The environment's `bin/` is not on `PATH`. The pipeline cell passes `PATH` and `VLLM_USE_FLASHINFER_SAMPLER=0` to the Ray workers; keep those lines. |
| Ray fails at init with a `/local_disk0/tmp` error | `/local_disk0` is not writable on serverless. The pipeline cell sets `RAY_TMPDIR=/tmp/ray`. |
| `No videos matched ...` in 02 | `CATALOG` / `SCHEMA` differ from the values used in 00, or `CATEGORY_FILTER` doesn't match the downloaded category. |
| `location` / `confidence` empty in `video_events` | An older copy of 03 without the schema fix. Re-run the structuring cell. |
| A face is visible in the redacted frame | Detection is automatic and can miss small faces in low-resolution footage. Check the preview (its axes are in frame pixels), add the missed region to `MANUAL_BOXES` as `x,y,w,h` (separate several with `;`), and re-run from the frame extraction cell. |

## Known limitations

* A few long videos still run up to `max_tokens` (1024) and are cut off (`finish_reason = length`). With 32B the cut-offs are long conclusions, not loops.
* `time_range` formats vary (`60 - 75 seconds`, `0:00 - 0:23`), because they come from the model's prose. 32B follows the seconds-from-start instruction more often than 7B, but its times can still be off by tens of seconds.
* Frames are sampled at ~1 fps from low-resolution (320 px) footage, so brief actions and small figures can be missed or described vaguely.
* The `normal` labels come from the dataset. Two of the flagged normal clips were checked by eye and were correctly labelled; the rest have not been checked.
* `confidence` from notebook 03 is not calibrated. It reflects how certain the model's prose sounds, not whether the incident happened.
* The prompt names shoplifting and lists examples (hiding items in bags, bypassing checkout). On footage with nothing to report, the model sometimes recasts ordinary actions as those examples. A neutral first pass ("describe what happens") followed by a separate theft check is a likely fix, but has not been tested.
