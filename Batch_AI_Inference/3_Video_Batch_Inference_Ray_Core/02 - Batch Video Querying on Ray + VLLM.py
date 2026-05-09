# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Batch Video Inference on Ray + vLLM
# MAGIC
# MAGIC Distributes a two-stage Ray pipeline across the cluster:
# MAGIC 1. `ConvertToPrompt` (CPU) — extracts frames via ffmpeg, builds the Qwen-VL prompt.
# MAGIC 2. `QwenVideoProcessing` (GPU) — loads the registered Qwen2.5-VL-32B weights into vLLM and runs inference.
# MAGIC
# MAGIC Output is written to a Delta table (default `video_inferences`) keyed by source video path.
# MAGIC
# MAGIC **Cluster:** GPU compute (e.g., `Standard_NC96ads_A100_v4` — 4×A100, 880 GB memory, 96 cores, 44 DBU/h).
# MAGIC - **Runtime:** 16.1.x-scala2.12, Unity Catalog enabled.
# MAGIC - **Spark configs:**
# MAGIC   - `spark.databricks.pyspark.dataFrameChunk.enabled true`
# MAGIC   - `spark.task.resource.gpu.amount 0`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install required packages

# COMMAND ----------

# MAGIC %pip install -qU databricks-sdk vllm==0.8.2 ffmpeg-python git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Imports

# COMMAND ----------

import os
import warnings

import pyspark.sql.functions as F
import ray
from mlflow.utils.databricks_utils import get_databricks_env_vars
from util import stage_registered_model, flatten_folder
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
import ffmpeg

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — MLflow auth + widgets

# COMMAND ----------

# Propagate Databricks auth into env vars so MLflow on Ray workers can reach UC.
mlflow_dbrx_creds = get_databricks_env_vars("databricks")
os.environ["DATABRICKS_HOST"] = mlflow_dbrx_creds["DATABRICKS_HOST"]
os.environ["DATABRICKS_TOKEN"] = mlflow_dbrx_creds["DATABRICKS_TOKEN"]

dbutils.widgets.text("CATALOG", "main", label="CATALOG")
dbutils.widgets.text("SCHEMA", "default", label="SCHEMA")
dbutils.widgets.text("MODEL_NAME", "qwen2_5_vl-32b", label="MODEL_NAME")
# Lowercase 'production' to match notebook 01 — MLflow aliases are case-sensitive.
dbutils.widgets.text("MODEL_ALIAS", "production", label="MODEL_ALIAS")
dbutils.widgets.text("INPUT_TABLE", "videos_file_reference", label="INPUT_TABLE")
dbutils.widgets.text("OUTPUT_TABLE", "video_inferences", label="OUTPUT_TABLE")
dbutils.widgets.text("CATEGORY_FILTER", "shoplifting", label="CATEGORY_FILTER")
dbutils.widgets.text("MAX_DURATION_SECONDS", "1200", label="MAX_DURATION_SECONDS")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")
MODEL_ALIAS = dbutils.widgets.get("MODEL_ALIAS")
INPUT_TABLE = dbutils.widgets.get("INPUT_TABLE")
OUTPUT_TABLE = dbutils.widgets.get("OUTPUT_TABLE")
CATEGORY_FILTER = dbutils.widgets.get("CATEGORY_FILTER")
MAX_DURATION_SECONDS = int(dbutils.widgets.get("MAX_DURATION_SECONDS"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Start Ray cluster
# MAGIC
# MAGIC Single-node Ray on the driver — sufficient for a 4×A100 cluster. For multi-node scaling, swap `ray.init(...)` for `setup_ray_cluster(...)`. See https://docs.databricks.com/en/machine-learning/ray/scale-ray.html

# COMMAND ----------

# Dashboard disabled here to keep the pip install minimal — the runtime's Ray
# version is what RayPatches.py is built against, so we don't pip-upgrade Ray.
# To enable the Ray dashboard, install `ray[default]` pinned to the runtime's
# version (check runtime release notes for the exact pin).
context = ray.init(include_dashboard=False)
print(f"Ray initialised: {context}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Define inference classes
# MAGIC
# MAGIC Two Ray actor classes:
# MAGIC - `ConvertToPrompt` — CPU stage. Probes the video for readability, samples `num_frames` frames, builds the Qwen-VL prompt.
# MAGIC - `QwenVideoProcessing` — GPU stage. Stages the registered model weights into local disk, instantiates vLLM, runs inference per row.
# MAGIC
# MAGIC Key vLLM tuning knobs (see Step 7):
# MAGIC - `tensor_parallel_size` — shards the 32B model across N GPUs.
# MAGIC - `gpu_memory_utilization` — how aggressively to pack GPU VRAM (PagedAttention prevents fragmentation, so 0.95 is safe in practice).
# MAGIC - `kv_cache_dtype="fp8"` — halves KV cache size, lets longer contexts fit.

# COMMAND ----------

class ConvertToPrompt:
    """CPU-stage Ray actor: probes the video, extracts N frames, builds the Qwen-VL prompt."""

    def __init__(self, query, num_frames=16):
        self.num_frames = num_frames
        self.query = query

    def transform(self, video_filename):
        """Returns a (frames, ok, error) tuple. On ffmpeg failure, returns (None, False, msg).

        Frame-count strategy: extract roughly one frame per second of source-video frame
        rate (i.e. for typical 30-fps CCTV that's ~30 frames per video, evenly sampled
        across the duration). This was the original recipe's behaviour and tested at
        ~60% recall on the public Kaggle CCTV anomaly subset. Reducing to a fixed
        16 frames noticeably cut recall on long videos. self.num_frames is the
        fallback for videos whose ffprobe metadata is missing the rate field.
        """
        try:
            probe = ffmpeg.probe(video_filename)
        except ffmpeg.Error as e:
            return None, False, e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)

        streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        adaptive_n = self.num_frames
        if streams:
            rate_str = streams[0].get("r_frame_rate", "")
            try:
                num, denom = rate_str.split("/")
                adaptive_n = max(self.num_frames, int(float(num) / float(denom)))
            except (ValueError, ZeroDivisionError):
                pass  # fall back to self.num_frames

        try:
            video_data = VideoAsset(name=video_filename, num_frames=adaptive_n)
            return video_data.np_ndarrays, True, ""
        except Exception as e:
            return None, False, f"VideoAsset failed: {e}"

    def __call__(self, row):
        frames, ok, err = self.transform(row["file_path"])
        row["frames"] = frames
        row["ffmpeg_ok"] = ok
        row["ffmpeg_error"] = err
        row["prompt"] = self.create_prompt(self.query)
        return row

    def create_prompt(self, question):
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class QwenVideoProcessing:
    """GPU-stage Ray actor: stages the UC-registered model into vLLM and runs inference."""

    def __init__(self, catalog: str, schema: str, model_name: str, model_kwargs: dict, model_alias: str = "production"):
        print("Loading model from UC registry...")

        import mlflow
        mlflow.set_registry_uri("databricks-uc")

        model_weights_path = stage_registered_model(
            catalog=catalog,
            schema=schema,
            model_name=model_name,
            alias=model_alias,
            local_base_path="/local_disk0/models/",
            overwrite=False,
        )
        flatten_folder(model_weights_path)
        model_weights_path = str(model_weights_path)

        # Each actor uses tensor_parallel_size GPUs. On a 4×A100 cluster with TP=2,
        # Ray can sustain 2 actors live in parallel. gpu_memory_utilization=0.95 packs
        # each shard aggressively — safe because PagedAttention prevents fragmentation.
        self.video_inference_pipeline = LLM(
            model=model_weights_path,
            max_model_len=model_kwargs["max_model_len"],
            max_num_seqs=model_kwargs["max_num_seqs"],
            mm_processor_kwargs={
                "min_pixels": model_kwargs["min_pixels"],
                "max_pixels": model_kwargs["max_pixels"],
                "fps": model_kwargs["fps"],
            },
            disable_mm_preprocessor_cache=True,
            tensor_parallel_size=model_kwargs["tensor_parallel_size"],
            kv_cache_dtype="fp8",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1, "video": 1},
            gpu_memory_utilization=0.95,
        )
        print("Model loaded.")

    def __call__(self, row):
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=1,
            repetition_penalty=1.05,
            max_tokens=2500,
            stop_token_ids=[],
        )

        prompts = [{
            "prompt": row["prompt"],
            "multi_modal_data": {"video": row["frames"]},
        }]

        outputs = self.video_inference_pipeline.generate(prompts, sampling_params)
        row["generated_text"] = outputs[0].outputs[0].text
        return row

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Load input reference table

# COMMAND ----------

input_table_fqn = f"{CATALOG}.{SCHEMA}.{INPUT_TABLE}"

video_files_reference_df = (
    spark.table(input_table_fqn)
    .withColumn("category", F.element_at(F.split("path", "/"), -2))
    .filter((F.col("category") == CATEGORY_FILTER) & (F.col("duration_seconds") <= MAX_DURATION_SECONDS))
)

display(video_files_reference_df)

# COMMAND ----------

# DBTITLE 1,Estimate input size
total_size = video_files_reference_df.agg(F.sum("size").alias("total_size")).collect()[0]["total_size"]
total_count = video_files_reference_df.count()
if total_size is None:
    print(f"No videos matched filters (category={CATEGORY_FILTER}, duration<={MAX_DURATION_SECONDS}s).")
else:
    print(f"{total_count} videos · {total_size / 1e9:.2f} GB matched.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Build Ray batch inference pipeline
# MAGIC
# MAGIC Two `.map` stages with different resource specs:
# MAGIC - **CPU stage** — `concurrency=(10, 24)`, `num_cpus=1`. Wide concurrency hides ffmpeg latency.
# MAGIC - **GPU stage** — `concurrency=(2, 12)`, `num_gpus=2`. Each actor uses `tensor_parallel_size=2`, so on a 4×A100 cluster ~2 actors run in parallel.
# MAGIC
# MAGIC A `.filter` between the two stages drops rows where ffmpeg failed (e.g., corrupt videos), so the GPU stage never sees `None` frames.

# COMMAND ----------

QUERY = """
Analyze the surveillance video and determine if any shoplifting activity occurs. Focus on suspicious behaviors such as: hiding merchandise in clothing or bags, intentionally bypassing the checkout, switching price tags, or attempting to distract staff.  If shoplifting is detected, describe what happened, when it occurred (with timestamps), and which person was involved. Be specific and detailed in your explanation.
"""

NUM_FRAMES = 16

MODEL_KWARGS = {
    "max_model_len": 32000,
    "max_num_seqs": 5,
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
    "fps": 1,
    "tensor_parallel_size": 2,
}

# Production target — direct Spark→Ray dataset:
#     ds = ray.data.from_spark(video_files_reference_df)
#
# Current workaround: vLLM 0.8.2 transitively pulls in a Ray version that
# conflicts with the runtime's dbruntime/RayPatches.py (BlockMetadata API
# drift), breaking ray.data.from_spark. For reference-table-sized inputs,
# converting via Pandas is a cheap bypass. For very large inputs, pin Ray to
# the runtime's exact version and use from_spark.
rows = video_files_reference_df.toPandas().to_dict("records")
ds = ray.data.from_items(rows)

ds = (
    ds.repartition(200)
    .map(
        ConvertToPrompt,
        fn_constructor_kwargs={"num_frames": NUM_FRAMES, "query": QUERY},
        concurrency=(10, 24),
        num_cpus=1,
    )
    .filter(lambda row: row.get("ffmpeg_ok", False))
    .map(
        QwenVideoProcessing,
        fn_constructor_kwargs={
            "catalog": CATALOG,
            "schema": SCHEMA,
            "model_name": MODEL_NAME,
            "model_kwargs": MODEL_KWARGS,
            "model_alias": MODEL_ALIAS,
        },
        concurrency=(2, 12),
        num_gpus=2,
    )
    .drop_columns(["frames"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Configure ray-uc-volumes-fuse temp dir + write Delta

# COMMAND ----------

# ray-uc-volumes-fuse needs a UC Volume path to stage parquet shards before the
# Delta write. We isolate it in a dedicated volume so it never collides with
# raw video bytes or output tables.
TEMP_VOLUME = "ray_fuse_temp"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{TEMP_VOLUME}")

tmp_dir_fs = f"/Volumes/{CATALOG}/{SCHEMA}/{TEMP_VOLUME}/tmp_inference"
dbutils.fs.mkdirs(tmp_dir_fs)
os.environ["RAY_UC_VOLUMES_FUSE_TEMP_DIR"] = tmp_dir_fs

# COMMAND ----------

output_table_fqn = f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}"

ds.write_databricks_table(
    output_table_fqn,
    mode="overwrite",
    mergeSchema=True,
)

print(f"Wrote inferences to {output_table_fqn}.")
