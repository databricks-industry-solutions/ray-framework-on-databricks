# Reading Databricks Tables into Ray Data

Ray workloads on Databricks — distributed training, hyperparameter tuning, batch inference — need data. That data lives in Databricks as tables managed by Unity Catalog. The challenge is bridging the two: getting governed, production data out of your tables and into Ray's distributed dataset format without bottlenecks, permission gaps, or loss of lineage.

Ray Data provides two native connectors for reading Databricks tables. Both read from Unity Catalog — they differ in **how** the data travels from storage to your Ray workers. This folder provides notebooks for each approach across two compute environments.

## Two compute environments

Ray on Databricks runs in two modes, each with its own cluster lifecycle:

### Classic Compute (`classic-compute/`)

Uses **Ray on Spark** — Ray workers are launched on Spark executor nodes via `setup_ray_cluster()`. Best when you already have a Classic Compute cluster or need tight Spark interop.

### AI Runtime / Serverless GPU (`ai-runtime/`)

Uses **Databricks Serverless GPU** — no Spark cluster to manage. Single-node workloads use `ray.init()` directly; multi-node workloads use the `@ray_launch` decorator from `serverless_gpu.ray` to provision GPU nodes on demand.

## Two data paths

### Via SQL Warehouse (`ray.data.read_databricks_tables`)

Submits a SQL query to a **Databricks SQL Warehouse** through the Statement Execution API. The warehouse executes the query and Ray workers fetch result chunks over HTTP.

- Supports arbitrary SQL via the `query=` parameter (filters, joins, aggregations).
- Unity Catalog captures lineage automatically — every read appears in the table's Lineage tab.
- Requires a running SQL Warehouse.
- Performance is bounded by warehouse size and Statement Execution API result limits.

### Via direct cloud storage (`ray.data.read_unity_catalog`)

Uses **Unity Catalog credential vending** to issue short-lived cloud storage tokens. Ray workers read the underlying Delta Parquet files directly from S3/ADLS/GCS — no warehouse, no Spark.

- Performance scales with the Ray cluster — reads go straight to object storage.
- Requires the `EXTERNAL USE SCHEMA` privilege (only the catalog owner can grant it).
- Reads the full table; does not support custom SQL.
- Lineage is **not** captured automatically. See the classic-compute reference notebook for an optional registration pattern using the External Metadata API.

## Choosing between them

| | Via SQL Warehouse | Via direct cloud storage |
|---|---|---|
| **API** | `ray.data.read_databricks_tables` | `ray.data.read_unity_catalog` |
| **Data path** | Statement Execution API → HTTP chunks | Credential vending → S3/ADLS/GCS |
| **Requires** | SQL Warehouse (running) | `EXTERNAL USE SCHEMA` grant |
| **Custom SQL** | Yes (`query=` parameter) | No (full table only) |
| **Lineage** | Automatic | Manual (External Metadata API) |
| **Performance ceiling** | Warehouse size + API limits | Ray cluster size |
| **Best for** | Filtered/transformed reads, smaller result sets, lineage-sensitive workflows | Full table reads, large-scale training, maximum throughput |

## Folder contents

```
ray-data-read-examples/
├── README.md
├── classic-compute/
│   ├── read_from_databricks_tables   # SQL Warehouse path on Classic Compute (Ray on Spark)
│   └── read_from_unity_catalog       # Credential vending path on Classic Compute (Ray on Spark)
└── ai-runtime/
    ├── read_from_databricks_tables_serverless_gpu   # SQL Warehouse path on Serverless GPU
    └── read_from_unity_catalog_serverless_gpu       # Credential vending path on Serverless GPU
```

| Folder | Notebook | Data path | Compute |
|---|---|---|---|
| `classic-compute/` | `read_from_databricks_tables` | SQL Warehouse | Ray on Spark |
| `classic-compute/` | `read_from_unity_catalog` | Credential vending | Ray on Spark |
| `ai-runtime/` | `read_from_databricks_tables_serverless_gpu` | SQL Warehouse | Serverless GPU |
| `ai-runtime/` | `read_from_unity_catalog_serverless_gpu` | Credential vending | Serverless GPU |

## Prerequisites

**Classic Compute:**
- Databricks ML Runtime with `ray[all]==2.54.0`.
- `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables set before `setup_ray_cluster()` so Ray workers inherit credentials.

**AI Runtime (Serverless GPU):**
- Databricks Serverless GPU compute.
- `ray[all]==2.54.0` installed via `%pip`.
- For multi-node: `serverless_gpu` package (pre-installed or via wheel).

See each notebook's introduction cell for method-specific requirements.
