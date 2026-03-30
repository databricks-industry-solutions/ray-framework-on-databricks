<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

# Batch AI Inference on Databricks with Ray

## Business Problem
Organizations need to process large volumes of unstructured data (text, images, speech, and video) efficiently at scale. Traditional single-node inference approaches become bottlenecks when dealing with massive datasets. This solution demonstrates how to leverage Ray's distributed computing capabilities on Databricks to perform efficient batch inference across different AI modalities.

By implementing these patterns, organizations can significantly reduce inference time and process large batches of multimedia data efficiently while maintaining high throughput and resource utilization.

## Examples

### Ray on Classic Compute (Examples 1–3)

These examples use Ray Core on Databricks Classic Compute clusters with GPU workers:

- **1_SLM_Batch_Inference_Ray_Core** — Text and image-to-text processing using Vision models
- **2_Automatic_Speech_Recognition_Ray_Core** — Automatic Speech Recognition using Whisper-v3
- **3_Video_Batch_Inference_Ray_Core** — Text and video-to-text processing using Qwen2.5 VL

### Ray on AI Runtime / Serverless GPU (Examples 4–5)

These examples use Ray Data with the Distributed Serverless GPU API to automatically provision and manage multi-node A10 GPUs for distributed LLM inference:

- **4_vLLM_Batch_Inference_AI_Runtime** — Distributed LLM inference with Ray Data and vLLM on Serverless GPU
- **5_SGLang_Batch_Inference_AI_Runtime** — Distributed LLM inference with Ray Data and SGLang on Serverless GPU

## Authors
- <amine.elhelou@databricks.com>
- <samantha.wise@databricks.com>
- <puneet.jain@databricks.com>
- <tj@databricks.com>