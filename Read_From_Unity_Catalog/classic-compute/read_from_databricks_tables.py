# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC ## Reading Unity Catalog Tables with `ray.data.read_databricks_tables`
# MAGIC
# MAGIC `ray.data.read_databricks_tables` reads UC tables through the [Databricks Statement Execution API](https://docs.databricks.com/api/workspace/statementexecution). Queries execute on a **SQL Warehouse**, not on the local Spark cluster.
# MAGIC
# MAGIC ### How it works
# MAGIC 1. The driver submits a `SELECT * FROM <table>` (or a custom `query=`) to the specified SQL Warehouse via the Statement Execution API.
# MAGIC 2. The warehouse executes the query and stages the result set.
# MAGIC 3. Ray workers fetch result chunks in parallel over HTTP.
# MAGIC
# MAGIC Because the data path goes through the warehouse, **Unity Catalog automatically captures lineage** — every read appears in the table's Lineage tab, attributed to the warehouse and the token used.
# MAGIC
# MAGIC ### Requirements
# MAGIC * A running **SQL Warehouse** (serverless, pro, or classic).
# MAGIC * `SELECT` on the target table, `USE CATALOG`, and `USE SCHEMA`.
# MAGIC * `DATABRICKS_TOKEN` and `DATABRICKS_HOST` environment variables (set in cell 5 before `setup_ray_cluster` so Ray workers inherit them).
# MAGIC
# MAGIC ### Notebook overview
# MAGIC | Cell | Purpose |
# MAGIC |---|---|
# MAGIC | **2 – Install packages** | Pins `ray[all]` and `databricks-sdk` to tested versions. |
# MAGIC | **3 – Configuration** | Sets `uc_catalog`, `uc_schema`, `uc_table`, and the fully qualified `uc_table_path`. |
# MAGIC | **4 – SQL Warehouse** | Finds or creates a serverless SQL Warehouse via the Databricks SDK. |
# MAGIC | **5 – Ray cluster setup** | Sets credentials as env vars, then initialises a Ray-on-Spark cluster using all 4 worker nodes. |
# MAGIC | **6 – Read table** | Calls `ray.data.read_databricks_tables` directly with explicit `catalog`, `schema`, and `warehouse_id`. |

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

# DBTITLE 1,Ray cluster setup
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import os

# Cleanly restart Ray if it was already running
try:
    shutdown_ray_cluster()
except:
    pass
try:
    ray.shutdown()
except:
    pass

# Dynamically grab the current user for log paths
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
ray_logs_path = f"/dbfs/Users/{user}/ray_collected_logs/"

# Set env vars so Ray workers can authenticate to DBSQL
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


setup_ray_cluster(
    min_worker_nodes=4,
    max_worker_nodes=4,
    num_cpus_worker_node=8,
    num_gpus_worker_node=0,
    collect_log_to_path=ray_logs_path,
)

# COMMAND ----------

ds = ray.data.read_databricks_tables(
  warehouse_id= warehouse_id,
  catalog=uc_catalog,
  schema=uc_schema,
  query=f'SELECT * FROM {uc_table}',
)
print('read directly from UC')