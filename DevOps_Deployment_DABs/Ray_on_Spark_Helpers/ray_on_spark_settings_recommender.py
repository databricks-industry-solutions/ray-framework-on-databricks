# Databricks notebook source
# MAGIC %md
# MAGIC # Ray on Spark - Cluster Setup Recommender
# MAGIC
# MAGIC ### Overview
# MAGIC * This script will provide recommendations for configurations for the `ray.util.spark.setup_ray_cluster()` command to launch a "Ray on Spark" cluster. 
# MAGIC * Attach it to any Classic All-Purpose Cluster during development; you can then take the recommendations after testing to an automated Job Cluster. 
# MAGIC * *As of August 2025, Ray on Spark will not work on Serverless clusters*
# MAGIC * The setup script will also confirm baseline cluster settings (such as runtime version, security mode).
# MAGIC
# MAGIC ### Steps
# MAGIC * Attach this script to a Classic All-Purpose cluster
# MAGIC * Click "Run all"
# MAGIC * Copy/Paste the [Ray on Spark setup command](https://docs.databricks.com/aws/en/machine-learning/ray/ray-create) at the end into a different notebook, then run (on the same cluster).
# MAGIC * Continue to modify this baseline setup as your workload evolves

# COMMAND ----------

# MAGIC %pip install -qqq --upgrade databricks-sdk bs4 lxml
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from pprint import pprint
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_workspace_client() -> WorkspaceClient: 
    """
    Returns an authenticated WorkspaceClient using the current Databricks notebook context.
    """
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    DATABRICKS_TOKEN = ctx.apiToken().getOrElse(None)
    DATABRICKS_URL = ctx.apiUrl().getOrElse(None)
    return WorkspaceClient(host=DATABRICKS_URL, token=DATABRICKS_TOKEN)
  
def get_cluster_id() -> str:
    """
    Returns the cluster ID of the current Databricks notebook context.
    """
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().clusterId().get()

def get_gpu_data(providers=['aws', 'gcp', 'azure']):
    """
    Scrapes GPU instance data for specified cloud providers and combines it
    into a single pandas DataFrame.

    Args:
        providers (list, optional): A list of providers to scrape.
                                    Defaults to ['aws', 'gcp', 'azure'].

    Returns:
        A pandas DataFrame containing all instance types, or an empty DataFrame if failed.
    """
    list_of_dataframes = []

    for provider in providers:
        try:
            # Step 1: Determine the correct URL based on the provider.
            if provider == 'azure':
                url = "https://learn.microsoft.com/en-us/azure/databricks/compute/gpu"
            elif provider in ['aws', 'gcp']:
                url = f"https://docs.databricks.com/{provider}/en/compute/gpu.html"
            else:
                print(f"Warning: Provider '{provider}' is not recognized. Skipping.")
                continue

            response = requests.get(url)
            response.raise_for_status()
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # Step 2: Apply the correct scraping strategy for the provider.
            # --- STRATEGY FOR AWS & GCP (Tab-based layout) ---
            if provider in ['aws', 'gcp']:
                tabs = soup.select("li[role='tab']")
                panels = soup.select("div[role='tabpanel']")

                if not tabs or len(tabs) != len(panels):
                    print(f"Warning: Page structure mismatch for {provider.upper()}. Skipping.")
                    continue

                for i, panel in enumerate(panels):
                    table_tag = panel.find('table')
                    if table_tag:
                        df = pd.read_html(str(table_tag))[0]
                        df['Provider'] = provider.upper()
                        # df['Series'] = tabs[i].get_text(strip=True)
                        list_of_dataframes.append(df)

            # --- STRATEGY FOR AZURE (Heading-based layout) ---
            elif provider == 'azure':
                headings = soup.find_all('h4')
                for heading in headings:
                    table_tag = heading.find_next('table')
                    if table_tag:
                        df = pd.read_html(str(table_tag))[0]
                        df['Provider'] = provider.upper()
                        # df['Series'] = heading.get_text(strip=True)
                        list_of_dataframes.append(df)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the request for {provider.upper()}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {provider.upper()}: {e}")

    # Step 3: Combine all collected data.
    if not list_of_dataframes:
        print("No data was collected.")
        return pd.DataFrame()

    combined_df = pd.concat(list_of_dataframes, ignore_index=True).set_index('Instance Name')
    return combined_df

def get_cloud_provider(details):
    if details.aws_attributes:
        return "aws"
    elif details.azure_attributes:
        return "azure"
    elif details.gcp_attributes:
        return "gcp"
    return "Unknown"

w = get_workspace_client()

current_cluster = w.clusters.get(get_cluster_id())
# pprint(current_cluster)

# COMMAND ----------



# COMMAND ----------

def check_security_mode(current_cluster):
    """
    Checks if the current cluster's data_security_mode is allowed for running Ray on Spark.
    Raises a ValueError if the mode is not permitted.
    """
    allowed_modes = ["USER_ISOLATION", "SINGLE_USER", "DEDICATED", "NONE"]
    pattern = re.compile(rf"({'|'.join(allowed_modes)})")

    data_security_mode_str = str(current_cluster.data_security_mode)

    if not pattern.search(data_security_mode_str):
        raise ValueError(
            f"data_security_mode '{data_security_mode_str}' is not allowed for Ray on Spark clusters. Allowed: {allowed_modes}. See docs: https://docs.databricks.com/aws/en/machine-learning/ray/#limitations"
        )
    else:
        print(f"Clusters with data_security_mode '{data_security_mode_str}' can be used to create Ray on Spark clusters.")

check_security_mode(current_cluster=current_cluster)

# COMMAND ----------

def check_runtime_version(current_cluster):
    """
    Checks if the current cluster's spark_version meets the minimum required version for running Ray on Spark.
    Raises a ValueError if the version is not permitted. Also recommends using the ML Runtime for best compatibility.
    """
    allowed_starting_version = ["12.2"]
    # Extract numeric parts from spark_version (e.g., "12.2.x-cpu-ml-scala2.12" -> "12.2")
    version_match = re.match(r"(\d+\.\d+)", str(current_cluster.spark_version))
    if not version_match:
        raise ValueError(f"Could not parse spark_version: {current_cluster.spark_version}")
    spark_version_num = version_match.group(1)

    # Compare as floats
    if float(spark_version_num) < float(allowed_starting_version[0]):
        raise ValueError(
            f"Cluster spark_version '{current_cluster.spark_version}' is less than the required version {allowed_starting_version[0]}"
        )
    else:
        print(f"Cluster Runtime '{current_cluster.spark_version}' can be used to run Ray on Spark. Consider upgrading to latest stable LTS ML Runtime for the best performance. ")
    
    if not getattr(current_cluster, "use_ml_runtime", False):
        print("Your cluster is not using ML Runtime. Recommend upgrading to most recent LTS Machine Learning Runtime where Ray and other dependencies are pre-installed")
    
check_runtime_version(current_cluster=current_cluster)

# COMMAND ----------

def check_single_node(current_cluster):
  """
  Checks if the current cluster is a single-node cluster.
  Raises a ValueError if the cluster is single-node, as Ray on Spark setup is intended for multi-node clusters.
  """
  if getattr(current_cluster, "is_single_node", False):
    raise ValueError("This script is intended to determine setup for a multi-node cluster to use Ray on Spark. This is a single-node cluster. To use ray, just run ray.init(). See Ray docs for more info.")

check_single_node(current_cluster=current_cluster)

# COMMAND ----------

def check_gpu(current_cluster):
    """
    Checks if the current cluster is using GPU-enabled instance types based on the cloud provider and node type.
    Returns True if GPUs are detected, otherwise False.
    """
    gpus = {
        "gpu_instance_types": {
            "aws": ["p5", "p4", "g6e", "g6", "g5", "g4dn", "p3"],
            "azure": [
                "Standard_NC40ads_H100_v5",
                "Standard_NC80adis_H100_v5",
                "Standard_NC24ads_A100_v4",
                "Standard_NC48ads_A100_v4",
                "Standard_NC96ads_A100_v4",
                "Standard_ND96asr_v4",
                "Standard_NV36ads_A10_v5",
                "Standard_NV36adms_A10_v5",
                "Standard_NV72ads_A10_v5",
                "Standard_NC4as_T4_v3",
                "Standard_NC8as_T4_v3",
                "Standard_NC16as_T4_v3",
                "Standard_NC64as_T4_v3",
                "Standard_NC6s_v3",
                "Standard_NC12s_v3",
                "Standard_NC24s_v3",
                "Standard_NC24rs_v3",
            ],
            "gcp": [
                "a2-ultragpu-8g",
                "a2-highgpu-1g",
                "a2-highgpu-2g",
                "a2-highgpu-4g",
                "a2-megagpu-16g",
                "g2-standard-8",
            ],
        }
    }

    if hasattr(current_cluster, "aws_attributes"):
        current_cloud = "aws"
    elif hasattr(current_cluster, "azure_attributes"):
        current_cloud = "azure"
    elif hasattr(current_cluster, "gcp_attributes"):
        current_cloud = "gcp"
    else:
        current_cloud = None

    has_gpu = False
    if current_cloud and hasattr(current_cluster, "node_type_id"):
        node_type = str(current_cluster.node_type_id).lower()
        gpu_types = [t.lower() for t in gpus["gpu_instance_types"].get(current_cloud, [])]
        for gpu_type in gpu_types:
            if gpu_type in node_type:
                has_gpu = True
                # print("Using instances with GPUs, additional setup required.")
                break

    # else:
      # print("CPU-only instances.")
    return has_gpu

check_gpu(current_cluster=current_cluster)

# COMMAND ----------



# COMMAND ----------

# TODO: Change to function input
spark_share = 0.0

setup_cmd = """
>>> Use setup command >>>
setup_ray_cluster(
"""

print("Observations for setup script recommendation:")
# STEP 1: Determine min and max worker nodes
## Autoscaling = FALSE
if current_cluster.autoscale:
  print(" - Autoscaling cluster")
  min_workers = worker_nodes = current_cluster.autoscale.min_workers
  max_workers = current_cluster.autoscale.max_workers

  setup_cmd += f"""  min_worker_nodes={min_workers},
  max_worker_nodes={max_workers},
  """
## Autoscaling = TRUE
else:
  print(" - Non-Autoscaling cluster")
  worker_nodes = current_cluster.num_workers

  setup_cmd += f"""  min_worker_nodes={worker_nodes},
  max_worker_nodes={worker_nodes},
  """


# STEP 2: Determine if Driver and Worker nodes match (homogenous cluster)
worker_driver_match = current_cluster.driver_node_type_id == current_cluster.node_type_id
## Worker Driver Match = FALSE
if not worker_driver_match:
  print(" - Heterogenous cluster, Driver and Workers are different instance types")
  driver_cores = int(current_cluster.cluster_cores - spark.sparkContext.defaultParallelism)
  worker_cores = int(spark.sparkContext.defaultParallelism / worker_nodes)

  setup_cmd += f"""num_cpus_worker_node={worker_cores},
  num_cpus_head_node={driver_cores},
  """
## Homogenous cluster
else:
  print(" - Homogenous cluster, Driver and Workers are same instance type:")
  worker_nodes = current_cluster.num_workers
  cores_per_node = int(current_cluster.cluster_cores/(worker_nodes+1))
  
  setup_cmd += f"""num_cpus_worker_node={cores_per_node}
  num_cpus_head_node={cores_per_node},
  """


# STEP 3: Determine if GPUs onboard
# TODO: Update and test this
try:
  provider = get_cloud_provider(current_cluster)
  gpus_list = get_gpu_data([provider])
  gpus_per_node = gpus_list.at[current_cluster.node_type_id, 'Number of GPUs']
  driver_gpus_per_node = gpus_list.at[current_cluster.driver_node_type_id, 'Number of GPUs']
  print(" - Detected GPU cluster; ensure you set spark.config('spark.task.resource.gpu.amount', 0) in the notebook or via Spark configs")

  setup_cmd += f"""num_gpus_worker_node={gpus_per_node},
  num_gpus_head_node={driver_gpus_per_node},
  """
except:
  print(" - no GPUs detected... skipping set up for GPUs.")

setup_cmd += """head_node_options={
      'dashboard_port': 9999,
      'include_dashboard':True,
    }
)
"""

# STEP 4: Determine if Spark Share is enabled
if spark_share > 0.0:
  print(" - Determine how many resources to give to Spark, then decrease the values of num_cpus_worker_node. Disabling Ray's usage of head node because Spark requires the driver node to orchestrate.")

  params_to_remove = ['num_gpus_head_node', 'num_cpus_head_node']
  # Split, filter, and join the lines
  lines = setup_cmd.splitlines()
  filtered_lines = [line for line in lines if not any(line.strip().startswith(param) for param in params_to_remove)]
  setup_cmd = '\n'.join(filtered_lines)

print(setup_cmd)


# COMMAND ----------

# MAGIC %md
# MAGIC ## End 
# MAGIC Copy the `setup_ray_cluster()` command printed after running the previous cell.