# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC ## Reading Unity Catalog Tables with `ray.data.read_databricks_tables` on Serverless GPU
# MAGIC
# MAGIC This notebook demonstrates how to load Unity Catalog tables into Ray Datasets on **Databricks Serverless GPU** compute. It uses `ray.data.read_databricks_tables`, which reads through the [Databricks Statement Execution API](https://docs.databricks.com/api/workspace/statementexecution) via a SQL Warehouse.
# MAGIC
# MAGIC ### How it works
# MAGIC 1. The driver submits a `SELECT * FROM <table>` (or a custom `query=`) to the specified SQL Warehouse via the Statement Execution API.
# MAGIC 2. The warehouse executes the query and stages the result set.
# MAGIC 3. Ray workers fetch result chunks in parallel over HTTP.
# MAGIC
# MAGIC Because the data path goes through the warehouse, **Unity Catalog automatically captures lineage** — every read appears in the table's Lineage tab.
# MAGIC
# MAGIC ### Key difference from Classic Compute
# MAGIC On serverless GPU, there is no Spark cluster to host Ray workers. Instead of `setup_ray_cluster()`, we use:
# MAGIC - **Single-node**: `ray.init()` to start Ray locally.
# MAGIC - **Multi-node**: `@ray_launch` from `serverless_gpu.ray` to provision a multi-node Ray cluster on demand.
# MAGIC
# MAGIC ### Requirements
# MAGIC * A running **SQL Warehouse** (serverless, pro, or classic).
# MAGIC * `SELECT` on the target table, `USE CATALOG`, and `USE SCHEMA`.
# MAGIC * Serverless GPU compute (single-node or multi-node).
# MAGIC
# MAGIC ### Notebook overview
# MAGIC | Cell | Purpose |
# MAGIC |---|---|
# MAGIC | **2 – Install packages** | Pins `ray[all]` and `databricks-sdk` to tested versions. |
# MAGIC | **3 – Configuration** | Sets `uc_catalog`, `uc_schema`, `uc_table`, and the fully qualified `uc_table_path`. |
# MAGIC | **4 – SQL Warehouse** | Finds or creates a serverless SQL Warehouse via the Databricks SDK. |
# MAGIC | **5–6 – Single-node** | Initialises Ray locally with `ray.init()` and reads the table. |
# MAGIC | **7–9 – Multi-node** | Defines a `TaskRunner` actor and launches it on a multi-node cluster via `@ray_launch`. |

# COMMAND ----------

# DBTITLE 1,Install packages
# MAGIC %pip install ray[all]==2.54.0 databricks-sdk>=0.49.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration: Unity Catalog table coordinates
uc_catalog = "main"
uc_schema  = "ray_gtm_examples"
uc_table   = "synthetic_data_20000000_rows_100_columns_5_labels_4_groups"

uc_table_path = f"{uc_catalog}.{uc_schema}.{uc_table}"
print(f"Target table: {uc_table_path}")

# COMMAND ----------

# DBTITLE 1,Provision or reuse a SQL Warehouse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import State

WAREHOUSE_NAME = "ray-read-warehouse"
AUTO_STOP_MINS = 10
CLUSTER_SIZE = "Small"

w = WorkspaceClient()

# Look for an existing warehouse by name that is running or can be started
warehouse_id = None
for wh in w.warehouses.list():
    if wh.name == WAREHOUSE_NAME:
        warehouse_id = wh.id
        if wh.state in (State.STOPPED, State.STOPPING):
            print(f"Starting existing warehouse '{WAREHOUSE_NAME}' ({wh.id}) ...")
            w.warehouses.start(wh.id).result()
        elif wh.state in (State.STARTING,):
            print(f"Warehouse '{WAREHOUSE_NAME}' is starting, waiting ...")
            w.warehouses.get_and_wait(wh.id)
        print(f"Using warehouse '{WAREHOUSE_NAME}' ({wh.id})")
        break

if warehouse_id is None:
    print(f"Creating serverless warehouse '{WAREHOUSE_NAME}' ...")
    created = w.warehouses.create(
        name=WAREHOUSE_NAME,
        cluster_size=CLUSTER_SIZE,
        auto_stop_mins=AUTO_STOP_MINS,
        enable_serverless_compute=True,
        min_num_clusters=1,
        max_num_clusters=1,
    ).result()
    warehouse_id = created.id
    print(f"Created warehouse '{WAREHOUSE_NAME}' ({warehouse_id})")

print(f"\nWarehouse ID: {warehouse_id}")

# COMMAND ----------

# DBTITLE 1,Single-node Ray
# MAGIC %md
# MAGIC ## Single-Node Ray
# MAGIC
# MAGIC On serverless GPU, `ray.init()` starts a local Ray instance on the current node. A single node can be sufficient for tasks that don't require much GPU memory. This means that with Ray we can utilize fractional GPUs to parallelize model training or feature engineering that require GPUs (e.g., image transformations). 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Initialize Ray and read from Unity Catalog
import ray
import os

# Set env vars so Ray workers can authenticate to DBSQL
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Start Ray locally on the serverless node
ray.init(ignore_reinit_error=True)

# Read from UC via the SQL warehouse
ds = ray.data.read_databricks_tables(
    warehouse_id=warehouse_id,
    catalog=uc_catalog,
    schema=uc_schema,
    query=f'SELECT * FROM {uc_table}',
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
_db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
_db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

@ray.remote(num_cpus=1)  # schedule on a worker, not the head node
class TaskRunner:
    def run(self):
        # Set credentials on the worker so read_databricks_tables can authenticate
        os.environ['DATABRICKS_HOST'] = _db_host
        os.environ['DATABRICKS_TOKEN'] = _db_token

        ds = ray.data.read_databricks_tables(
            warehouse_id=warehouse_id,
            catalog=uc_catalog,
            schema=uc_schema,
            query=f'SELECT * FROM {uc_table}',
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