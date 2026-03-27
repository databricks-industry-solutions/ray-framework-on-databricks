# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC ## Reading Unity Catalog Tables with `ray.data.read_unity_catalog` on Serverless GPU
# MAGIC
# MAGIC This notebook demonstrates how to load Unity Catalog tables into Ray Datasets on **Databricks Serverless GPU** compute using `ray.data.read_unity_catalog`. This method leverages [Unity Catalog credential vending](https://docs.databricks.com/en/external-access/credential-vending.html) to issue short-lived, least-privilege cloud storage tokens — allowing Ray workers to read table data directly from S3/ADLS/GCS without routing through Spark or a SQL Warehouse.
# MAGIC
# MAGIC ### How it works
# MAGIC 1. The driver contacts the Unity Catalog REST API to obtain vended credentials for the target table's underlying cloud storage.
# MAGIC 2. Ray workers read Parquet/Delta files directly from S3/ADLS/GCS using those credentials.
# MAGIC 3. No SQL Warehouse or Spark cluster is involved in the data path.
# MAGIC
# MAGIC ### Lineage
# MAGIC Because `read_unity_catalog` bypasses Spark and SQL Warehouse, **Unity Catalog does not automatically capture lineage** for these reads. See the [read_from_unity_catalog](https://github.com/jacheung/ray-on-databricks-rct/blob/main/ray-data-read-examples/classic-compute/read_from_unity_catalog.ipynb) reference notebook in `classic-compute/` for an optional lineage registration pattern using the External Metadata APIs.
# MAGIC
# MAGIC ### Requirements
# MAGIC * **`EXTERNAL USE SCHEMA`** must be granted on the target schema. Only the **catalog owner** can grant this privilege — it is *not* included in `ALL PRIVILEGES`.
# MAGIC * `SELECT` on the target table, `USE CATALOG`, and `USE SCHEMA`.
# MAGIC * The Unity Catalog metastore must have **external access enabled**.
# MAGIC * `deltalake>=1.5.0` and `unitycatalog` Python packages.
# MAGIC * Serverless GPU compute (single-node or multi-node).
# MAGIC
# MAGIC > **Note on `deltalake` 1.5.0+:** The latest delta-rs release adds native support for [deletion vectors](https://docs.delta.io/latest/delta-deletion-vectors.html), making the end-to-end OSS read path — UC REST API → credential vending → delta-rs — fully seamless.
# MAGIC
# MAGIC ### Key difference from Classic Compute
# MAGIC On serverless GPU, there is no Spark cluster to host Ray workers. Instead of `setup_ray_cluster()`, we use:
# MAGIC - **Single-node**: `ray.init()` to start Ray locally.
# MAGIC - **Multi-node**: `@ray_launch` from `serverless_gpu.ray` to provision a multi-node Ray cluster on demand.
# MAGIC
# MAGIC ### Notebook overview
# MAGIC | Cell | Purpose |
# MAGIC |---|---|
# MAGIC | **2 – Install packages** | Pins `deltalake`, `unitycatalog`, `ray[all]`, and `databricks-sdk`. |
# MAGIC | **3 – Configuration** | Sets `uc_catalog`, `uc_schema`, `uc_table`, and the fully qualified `uc_table_path`. |
# MAGIC | **4 – Verify privileges** | Checks for `EXTERNAL USE SCHEMA`; attempts to grant it or raises an actionable error. |
# MAGIC | **5–6 – Single-node** | Initialises Ray locally with `ray.init()` and reads the table via credential vending. |
# MAGIC | **7–9 – Multi-node** | Defines a `TaskRunner` actor and launches it on a multi-node cluster via `@ray_launch`. |

# COMMAND ----------

# DBTITLE 1,Install packages
# MAGIC %pip install deltalake==1.5.0 unitycatalog==0.0.1a0 ray[all]==2.54.0 databricks-sdk>=0.102.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration: Unity Catalog table coordinates
uc_catalog = "main"
uc_schema  = "ray_gtm_examples"
uc_table   = "synthetic_data_20000000_rows_100_columns_5_labels_4_groups"

uc_table_path = f"{uc_catalog}.{uc_schema}.{uc_table}"
print(f"Target table: {uc_table_path}")

# COMMAND ----------

# DBTITLE 1,Verify EXTERNAL USE SCHEMA privileges
current_user = spark.sql("SELECT current_user()").collect()[0][0]

print(f"Checking EXTERNAL USE SCHEMA on `{uc_catalog}`.`{uc_schema}` for {current_user} ...")

# Check existing grants on the schema
grants_df = spark.sql(f"SHOW GRANTS ON SCHEMA `{uc_catalog}`.`{uc_schema}`")
has_privilege = (
    grants_df.filter(
        (grants_df.ActionType == "EXTERNAL USE SCHEMA")
        & (grants_df.Principal == current_user)
    ).count()
    > 0
)

if has_privilege:
    print("EXTERNAL USE SCHEMA is already granted. Ready to read with ray.data.read_unity_catalog.")
else:
    print("EXTERNAL USE SCHEMA not found. Attempting to grant ...")
    try:
        spark.sql(
            f"GRANT EXTERNAL USE SCHEMA ON SCHEMA `{uc_catalog}`.`{uc_schema}` TO `{current_user}`"
        )
        print("Successfully granted EXTERNAL USE SCHEMA.")
    except Exception as e:
        raise PermissionError(
            f"Failed to grant EXTERNAL USE SCHEMA on `{uc_catalog}`.`{uc_schema}`.\n"
            f"Only the catalog owner can grant this privilege.\n"
            f"Please ask the owner of the `{uc_catalog}` catalog to run:\n\n"
            f"  GRANT EXTERNAL USE SCHEMA ON SCHEMA `{uc_catalog}`.`{uc_schema}` TO `{current_user}`;\n\n"
            f"Original error: {e}"
        )

# COMMAND ----------

# DBTITLE 1,Single-node Ray
# MAGIC %md
# MAGIC ## Single-Node Ray
# MAGIC
# MAGIC On serverless GPU, `ray.init()` starts a local Ray instance on the current node. `read_unity_catalog` reads directly from cloud storage via credential vending — no SQL Warehouse needed.

# COMMAND ----------

# DBTITLE 1,Initialize Ray and read from Unity Catalog
import ray
import os

# Start Ray locally on the serverless node
ray.init(ignore_reinit_error=True)

# Retrieve workspace details for credential vending
workspace_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# On serverless GPU, clusterUsageTags.region may not be set — fall back to us-west-2
try:
    region = spark.conf.get("spark.databricks.clusterUsageTags.region")
except:
    region = "us-west-2"

# Read directly from cloud storage via credential vending
ds = ray.data.read_unity_catalog(
    table=uc_table_path,
    url=workspace_url,
    token=token,
    region=region,
)

print(ds.schema())
print(f"Count: {ds.count()}")

# COMMAND ----------

# DBTITLE 1,Multi-node Ray
# MAGIC %md
# MAGIC ## Multi-Node Ray
# MAGIC
# MAGIC On serverless GPU, the `@ray_launch` decorator from `serverless_gpu.ray` provisions a multi-node Ray cluster on demand. The two-step pattern:
# MAGIC
# MAGIC 1. **`TaskRunner`** (`@ray.remote(num_cpus=1)`): A Ray actor that encapsulates the actual work. Setting `num_cpus=1` ensures it is scheduled on a worker node, not the head.
# MAGIC 2. **Launch function** (`@ray_launch(gpus=N, gpu_type, remote=True)`): A thin wrapper that creates the TaskRunner, calls its `.run.remote()`, and returns the result. The `remote=True` flag tells Databricks to provision additional serverless GPU nodes.
# MAGIC 3. **`.distributed()`**: Triggers cluster provisioning and executes the function across the multi-node cluster.
# MAGIC
# MAGIC See [01c-train-with-serverless-GPUs-Out-of-Core](https://github.com/jacheung/ray-on-databricks-rct/blob/main/distributed-training/XGBoost/01c-train-with-serverless-GPUs-Out-of-Core.ipynb) for the full training example this pattern is drawn from.

# COMMAND ----------

# DBTITLE 1,Define TaskRunner actor
import ray
import os

# Capture credentials on the driver so they can be forwarded to remote workers
_workspace_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
_db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
try:
    _region = spark.conf.get("spark.databricks.clusterUsageTags.region")
except:
    _region = "us-west-2"

@ray.remote(num_cpus=1)  # schedule on a worker, not the head node
class TaskRunner:
    def run(self):
        ds = ray.data.read_unity_catalog(
            table=uc_table_path,
            url=_workspace_url,
            token=_db_token,
            region=_region,
        )

        # Input your Ray code for unstructured data processing (e.g., image transforms) or distributed model training.

        print(ds.schema())
        print(f"Count: {ds.count()}")
        return ds

# COMMAND ----------

# DBTITLE 1,Launch multi-node Ray cluster and read
from serverless_gpu.ray import ray_launch

@ray_launch(gpus=4, gpu_type='A10', remote=True)
def read_uc_multinode():
    runner = TaskRunner.remote()
    return ray.get(runner.run.remote())

ds = read_uc_multinode.distributed()