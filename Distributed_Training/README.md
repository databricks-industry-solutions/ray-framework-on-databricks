<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

# Distributed Training on Databricks with Ray

## Why Ray for Distributed Training?

As models grow in complexity and datasets grow in size, single-node training quickly becomes a bottleneck. Ray addresses this by providing a unified framework for distributed computing that integrates natively with popular ML libraries:

- **Seamless scaling** — Ray Train provides drop-in distributed wrappers for PyTorch, XGBoost, and other frameworks. You write standard training code and Ray handles sharding data across workers, synchronizing gradients, and managing checkpoints.
- **Hyperparameter optimization at scale** — Ray Tune orchestrates parallel HPO trials across a GPU cluster, with intelligent search algorithms (e.g., Optuna) and early stopping. Combined with Ray Train, you can run distributed DDP training *inside* each HPO trial.
- **Flexible data ingestion** — Ray Data reads directly from Unity Catalog tables, Delta Lake, or Parquet files, and streams data to training workers without requiring full materialization in memory.
- **Databricks integration** — Running Ray on Databricks gives you managed infrastructure, MLflow experiment tracking, Unity Catalog model registry, and governance — all in one platform.

## Examples

### PyTorch

| Notebook | Compute | Description |
| --- | --- | --- |
| `sgc-ray-resnet18` | AI Runtime / Serverless GPU | Distributed PyTorch ResNet18 training on FashionMNIST using Ray Train and Ray Data on multi-node A10 GPUs with MLflow logging and UC model registration |

### XGBoost

Start with `00-create-dataset` to generate the synthetic dataset used by the training notebooks.

| Notebook | Compute | Description |
| --- | --- | --- |
| `00-create-dataset` | Classic Compute | Generates a synthetic classification dataset (configurable rows, columns, labels) and writes to Delta and Parquet in Unity Catalog |
| `01a-train-with-GPUs-In-Core` | Classic Compute | In-core DDP training — loads the full dataset into GPU VRAM across multiple workers. Requires ~1.5x dataset size in VRAM and ~6-7x in RAM |
| `01b-train-with-GPUs-Out-of-Core` | Classic Compute | Out-of-core DDP training — streams data in batches via XGBoost's `DataIter`, requiring only ~0.5x dataset size in VRAM and ~2x in RAM |
| `01c-train-with-serverless-GPUs-Out-of-Core` | AI Runtime / Serverless GPU | Out-of-core DDP training on Serverless GPU using the Distributed Serverless GPU API — same streaming approach as 01b but with on-demand A10 provisioning |

> **Note:** Classic Compute examples use `ray.util.spark.setup_ray_cluster` to launch Ray on top of a Spark cluster. AI Runtime examples use the `@ray_launch` decorator from the Distributed Serverless GPU API to provision remote GPUs.

## Authors
- <jon.cheung@databricks.com>
- <puneet.jain@databricks.com>