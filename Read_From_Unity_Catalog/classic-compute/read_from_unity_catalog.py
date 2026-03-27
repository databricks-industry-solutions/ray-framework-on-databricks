# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC ## Reading Unity Catalog Tables with `ray.data.read_unity_catalog`
# MAGIC
# MAGIC `ray.data.read_unity_catalog` leverages [Unity Catalog credential vending](https://docs.databricks.com/en/external-access/credential-vending.html) to issue short-lived, least-privilege cloud storage tokens — allowing Ray workers to read table data directly from S3/ADLS/GCS without routing through Spark.
# MAGIC
# MAGIC ### Requirements
# MAGIC * **`EXTERNAL USE SCHEMA`** must be granted on the target schema. Only the **catalog owner** can grant this privilege — it is *not* included in `ALL PRIVILEGES` and schema owners do not have it by default.
# MAGIC * The requesting principal also needs `SELECT` on the table, `USE CATALOG`, and `USE SCHEMA`.
# MAGIC * The Unity Catalog metastore must have **external access enabled** (metastore-level admin setting).
# MAGIC
# MAGIC > **Note on `deltalake` 1.5.0+:** The latest delta-rs release adds native support for [deletion vectors](https://docs.delta.io/latest/delta-deletion-vectors.html), which means `read_unity_catalog` can now read Delta tables with DV-enabled operations (e.g., `DELETE`, `MERGE`, `UPDATE`) without requiring a Spark compatibility layer. This makes the end-to-end OSS read path — Unity Catalog REST API → credential vending → delta-rs — fully seamless.
# MAGIC
# MAGIC ### Lineage
# MAGIC
# MAGIC Because `read_unity_catalog` reads directly from cloud storage via credential vending (bypassing Spark), **Unity Catalog does not automatically capture lineage** for these reads. The `read_uc_table()` function in cell 7 provides an optional `log_lineage=True` flag that works around this using the [External Metadata / External Lineage APIs](https://docs.databricks.com/en/data-governance/unity-catalog/external-lineage.html). See the dedicated markdown cell (cell 6) for details.
# MAGIC
# MAGIC ### Notebook Overview
# MAGIC | Cell | Purpose |
# MAGIC |---|---|
# MAGIC | **2 – Install packages** | Pins `deltalake`, `unitycatalog`, `ray[all]`, and `databricks-sdk` to tested versions. |
# MAGIC | **3 – Configuration** | Sets `uc_catalog`, `uc_schema`, and `uc_table` as user-editable variables. |
# MAGIC | **4 – Verify privileges** | Checks for `EXTERNAL USE SCHEMA`; attempts to grant it or raises an actionable error. |
# MAGIC | **5 – Ray cluster setup** | Initializes a Ray on Spark cluster with environment credentials for MLflow. |
# MAGIC | **6 – Lineage note** | Explains the lineage gap and how external metadata registration works. |
# MAGIC | **7 – `read_uc_table()`** | Parameterized function wrapping `ray.data.read_unity_catalog` with optional lineage registration. |
# MAGIC | **8 – Usage** | Example call: `read_uc_table(uc_table_path, log_lineage=True)`. |

# COMMAND ----------

# DBTITLE 1,Install delta-oss and unitycatalog packages
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

# Set env vars so MLflow works on Ray head + worker nodes
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Launch Ray — 4 worker nodes × 8 CPUs = 32 CPUs total
setup_ray_cluster(
    min_worker_nodes=4,
    max_worker_nodes=4,
    num_cpus_worker_node=8,
    num_gpus_worker_node=0,   
    collect_log_to_path=ray_logs_path,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lineage
# MAGIC
# MAGIC Because `read_unity_catalog` reads directly from cloud storage via credential vending (bypassing Spark), **Unity Catalog does not automatically capture lineage** for these reads. UC lineage requires queries to flow through the Spark DataFrame API or Databricks SQL interfaces.
# MAGIC
# MAGIC To work around this, the below cell provides an optional `log_lineage=True` flag that registers each read in the UC lineage graph using the [External Metadata / External Lineage APIs](https://docs.databricks.com/en/data-governance/unity-catalog/external-lineage.html) (Public Preview). When enabled, it:
# MAGIC 1. Creates an **external metadata object** representing the Ray reader (with notebook URL, user, timestamp, and table properties).
# MAGIC 2. Creates a **lineage relationship** from the UC table (upstream) to the external metadata object (downstream).
# MAGIC
# MAGIC This makes the read visible in the table's **Lineage** tab in Catalog Explorer. Requires `databricks-sdk>=0.102.0` and the `CREATE EXTERNAL METADATA` privilege on the metastore.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Read Unity Catalog table with ray.data
import ray
from datetime import datetime, timezone
from typing import Optional


def read_uc_table(
    table: str,
    *,
    log_lineage: bool = False,
    limit: Optional[int] = None,
) -> ray.data.Dataset:
    """Read a Unity Catalog table with Ray using vended credentials and optionally register external lineage.

    Args:
        table: Fully qualified UC table name (catalog.schema.table).
        log_lineage: If True, register an external metadata object and lineage
            relationship in Unity Catalog so the read appears in the lineage graph.
            Requires ``databricks-sdk>=0.102.0``. Default False.
        limit: Optional row limit passed to ``read_unity_catalog``.

    Returns:
        A ``ray.data.Dataset`` backed by the UC table.
    """
    # Retrieve necessary Databricks Workspace details and access tokens to read Unity Catalog dataset
    workspace_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
    region = spark.conf.get("spark.databricks.clusterUsageTags.region")
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    
    ds = ray.data.read_unity_catalog(
        table=table,
        url=workspace_url,
        token=token,
        region=region,
    )

    if log_lineage:
        _register_lineage(table, workspace_url, token)

    return ds


def _register_lineage(table: str, workspace_url: str, token: str) -> None:
    """Create an external metadata object + lineage relationship for a Ray read."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import (
        ExternalMetadata,
        SystemType,
        CreateRequestExternalLineage,
        ExternalLineageObject,
        ExternalLineageTable,
        ExternalLineageExternalMetadata,
    )

    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = ctx.notebookPath().get()
    notebook_id = ctx.tags().get("notebookId").get()
    current_user = ctx.userName().get()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    table_short = table.split(".")[-1]
    em_name = f"ray-read-{table_short}-{ts}"

    w = WorkspaceClient()

    em = w.external_metadata.create_external_metadata(
        ExternalMetadata(
            name=em_name,
            system_type=SystemType.DATABRICKS,
            entity_type="notebook",
            description=(
                f"Ray read_unity_catalog of '{table}' "
                f"from notebook '{notebook_path}' "
                f"by {current_user} at {ts}"
            ),
            url=f"{workspace_url}/#notebook/{notebook_id}",
            properties={
                "reader": "ray.data.read_unity_catalog",
                "table": table,
                "notebook_path": notebook_path,
                "user": current_user,
                "timestamp": ts,
            },
        )
    )

    rel = w.external_lineage.create_external_lineage_relationship(
        CreateRequestExternalLineage(
            source=ExternalLineageObject(
                table=ExternalLineageTable(name=table)
            ),
            target=ExternalLineageObject(
                external_metadata=ExternalLineageExternalMetadata(name=em_name)
            ),
        )
    )

    print(f"Lineage registered: {table} \u2192 {em_name} (id: {rel.id})")
    print(
        f"View: {workspace_url}/explore/data/"
        f"{table.replace('.', '/')}?tab=lineage"
    )



# COMMAND ----------

# Example usage to read UC table as a Ray Dataset and log lineage through External Metadata
ds = read_uc_table(uc_table_path, log_lineage=True)
print(ds.schema())
print(f"Count: {ds.count()}")