# Databricks notebook source
# MAGIC %md 
# MAGIC ## Distributed XGBoost with GPUs on Ray 
# MAGIC
# MAGIC Ray provides a version of XGBoost to perform distributed data parallelism (DDP). With drop-in replacements of `xgboost` native classes, XGboost Ray allows you to leverage multi-node clusters to distribute your training. 
# MAGIC
# MAGIC This demo uses a dataset created from `00-create-dataset` with 100M rows x 100 features columns x 1 target column (5 classes) for multi-class classification. This dataset is ~40GiB. 
# MAGIC
# MAGIC `01a-train-with-GPUs-In-Core` demonstrates in-core distributed training. This means you need 1.5x the dataset size in VRAM and 6-7x the dataset size in RAM. (e.g., my 40GB dataset would require about 60GB VRAM and ~280GB RAM). Since my cluster below consists of worker nodes w/ 64GB RAM and 24GB VRAM (A10) per worker, I'll need at least 5 workers for DDP. 
# MAGIC
# MAGIC `01b-train-with-GPUs-Out-of-Core` (**this notebook**) demonstrates out-of-core distributed training. In my tests, I only needed approximately 0.5x the dataset size in VRAM and 2x the dataset size in RAM. (e.g., my 40GB dataset would require about 20GB VRAM and ~80GB RAM). Since my cluster consists of worker nodes with 64GB RAM and 24GB VRAM (A10) per worker, I'll need at least 2 workers for DDP. 
# MAGIC
# MAGIC
# MAGIC #### Compute specifications to run this notebook
# MAGIC ```json
# MAGIC {
# MAGIC     "num_workers": 8,
# MAGIC     "cluster_name": "Multi-node MLR w/ GPUs",
# MAGIC     "spark_version": "17.3.x-gpu-ml-scala2.13",
# MAGIC     "spark_conf": {
# MAGIC         "spark.task.resource.gpu.amount": "0",
# MAGIC         "spark.executor.memory": "1g"
# MAGIC     },
# MAGIC     "aws_attributes": {
# MAGIC         "first_on_demand": 1,
# MAGIC         "availability": "SPOT_WITH_FALLBACK",
# MAGIC         "zone_id": "auto",
# MAGIC         "spot_bid_price_percent": 100,
# MAGIC         "ebs_volume_count": 0
# MAGIC     },
# MAGIC     "node_type_id": "g5.4xlarge",
# MAGIC     "driver_node_type_id": "g5.4xlarge",
# MAGIC     "autotermination_minutes": 60,
# MAGIC     "enable_elastic_disk": false,
# MAGIC     "single_user_name": "jon.cheung@databricks.com",
# MAGIC     "enable_local_disk_encryption": false,
# MAGIC     "data_security_mode": "SINGLE_USER",
# MAGIC     "runtime_engine": "STANDARD",
# MAGIC     "assigned_principal": "user:jon.cheung@databricks.com",
# MAGIC }
# MAGIC ```

# COMMAND ----------

# When running Ray on Spark with GPUs, we need to ensure that Spark isn't occupying the GPUs, so Ray can use it. 
if spark.conf.get("spark.task.resource.gpu.amount") != "0":
  print('Setting Spark GPU Usage to 0 so Ray can use.')
  spark.conf.set("spark.task.resource.gpu.amount", "0")

# COMMAND ----------

# MAGIC %pip install -qU ray[all] rmm-cu12 xgboost
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Define job inputs
dbutils.widgets.text("catalog_name", "main", "Unity Catalog Name")
dbutils.widgets.text("schema_name", "ray_gtm_examples", "Unity Catalog Schema Name")
dbutils.widgets.text("num_training_rows", "100_000_000", "rows of data to generate")
dbutils.widgets.text("num_training_columns", "100", "number of feature columns")
dbutils.widgets.text("num_labels", "5", "number of labels in the target column")
dbutils.widgets.text("warehouse_id", "8baced1ff014912d", "ID of warehouse to use for reading in Databricks Table as Ray Data")


# Get parameter values (will override widget defaults if run by job)
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
num_training_rows = int(dbutils.widgets.get("num_training_rows"))
num_training_columns = int(dbutils.widgets.get("num_training_columns"))
num_labels = int(dbutils.widgets.get("num_labels"))
sql_warehouse_id = dbutils.widgets.get("warehouse_id")

# COMMAND ----------

# This is for writing Ray Tune results to MLflow
mlflow_experiment_name = f"/Users/jon.cheung@databricks.com/ray_xgboost"

# If running in a multi-node cluster, this is where you
# should configure the run's persistent storage that is accessible
# across all worker nodes.
ray_xgboost_path = '/dbfs/Users/jon.cheung@databricks.com/ray_xgboost/' 

# This is for stashing the cluster logs
ray_logs_path = "/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Ray Cluster
# MAGIC
# MAGIC The below function highlights how to create a compute using `ray.util.spark.setup_ray_cluster`. In short, this code snippet launches a Ray cluster on top of the existing Spark cluster that's created from Databricks Classic Compute. 
# MAGIC
# MAGIC See this [link](https://docs.databricks.com/aws/en/machine-learning/ray/ray-create#fixed-size-ray-cluster) for the official Databricks documentation on setting up a Ray cluster.
# MAGIC
# MAGIC Furthermore, see this [blog](https://community.databricks.com/t5/technical-blog/ray-on-spark-a-practical-architecture-and-setup-guide/ba-p/127511) for an in-depth guide for the Ray on Spark cluster and how to set up the parameters. 

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import os

restart = True
if restart is True:
  try:
    shutdown_ray_cluster()
  except:
    pass
  try:
    ray.shutdown()
  except:
    pass

# Set the parameters here so mlflow works properly at the Ray head + worker nodes
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


# The below configuration mirrors my Spark worker cluster set up. Change this to match your cluster configuration. 
setup_ray_cluster(
  min_worker_nodes=8,
  max_worker_nodes=8,
  num_cpus_worker_node=16,
  num_gpus_worker_node=1,
  # num_cpus_head_node=8, # OPTIONAL set this to utilize compute on head node
  # num_gpus_head_node=1, # OPTIONAL set this to utilize compute on head node
  collect_log_to_path=ray_logs_path
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Define the low-level Trainable
# MAGIC We need to define the low-level trainable (i.e. the function that will perform the model training). To leverage out-of-core training we need to do two things: 
# MAGIC 1. Construct a data iterator.  
# MAGIC The Data Iterator from the XGBoost python package is the core component for performing out-of-core training. The goal of this is to pipe in data in batches so that the CPU/GPU (GPU in the example below) can construct the DMatrix/QuantileDMatrix with a dataset that far exceeds the size of the original dataset. 
# MAGIC 2. Write low-level trainable. 
# MAGIC Here, we pass in the data iterator to the DMatrix or QuantileDMatrix construction and train an XGBoost model as usual. 

# COMMAND ----------

import cupy as cp
import xgboost
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

# It's important to use RMM for GPU-based external memory to improve performance.
# If XGBoost is not built with RMM support, a warning will be raised.
# We use the pool memory resource here for simplicity, you can also try the
# `ArenaMemoryResource` for improved memory fragmentation handling.
mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource())
rmm.mr.set_current_device_resource(mr)
# Set the allocator for cupy as well.
cp.cuda.set_allocator(rmm_cupy_allocator)

class RayDataIter(xgboost.core.DataIter):
    def __init__(self, ray_iterator: ray.data.DataIterator, label_col: str):
        super().__init__()
        self.label_col = label_col
        self.iterator = ray_iterator
        self._generator = None

    def reset(self):
        self._generator = iter(self.iterator)

    def next(self, input_data):
        try:
            batch = next(self._generator)
        except StopIteration:
            return False
        
        y = cp.asarray(batch[self.label_col])
        X = cp.column_stack([cp.asarray(batch[col]) for col in batch if col != self.label_col])

        input_data(data=X, label=y)
        return True

# COMMAND ----------

import xgboost
import ray.train
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback


def train_fn_per_worker(params: dict):
    """
    Trains an XGBoost model on a shard of the distributed dataset assigned to this worker.

    This should look very similar to a vanilla XGboost training.

    This function is designed to be executed by individual Ray Train workers.
    It retrieves the training and validation data shards, converts them to DMatrix format,
    and performs a portion of the distributed XGBoost training. Ray Train handles
    the inter-worker communication.

    Args:
        params (dict): A dictionary of XGBoost training parameters, including
                       'num_estimators', 'eval_metric', and potentially other
                       XGBoost-specific parameters.
    """

    # Get dataset shards for this worker
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    iterator = train_shard.iter_batches(batch_format="numpy",
                                         batch_size=params['batch_size'],
                                          prefetch_batches=params['prefetch_batches'])
    streaming_iter = RayDataIter(iterator, label_col=label)

    val_iterator = val_shard.iter_batches(batch_format="numpy",
                                           batch_size=params['batch_size'],
                                            prefetch_batches=params['prefetch_batches'])
    streaming_val_iter = RayDataIter(val_iterator, label_col=label)
    with xgboost.config_context(use_rmm=True):
        # External Quantile DMatrix (streams data, minimal memory usage, GPU-optimized)
        # qdm_train = xgboost.DMatrix(streaming_iter)
        # qdm_val = xgboost.DMatrix(streaming_val_iter)
        qdm_train = xgboost.ExtMemQuantileDMatrix(streaming_iter,
                                                  max_quantile_batches=params['max_quantile_batches'])
        qdm_val = xgboost.ExtMemQuantileDMatrix(streaming_val_iter, ref=qdm_train, max_quantile_batches=params['max_quantile_batches'])

        # Do distributed data-parallel training.
        # Ray Train sets up the necessary coordinator processes and
        # environment variables for workers to communicate with each other.
        evals_results = {}
        bst = xgboost.train(
            params,
            dtrain=qdm_train,
            evals=[(qdm_val, "validation")],
            num_boost_round=params['num_estimators'],
            evals_result=evals_results,
            # early_stopping_rounds=params['early_stopping_rounds'],
            callbacks=[RayTrainReportCallback(metrics={params['eval_metric']: f"validation-{params['eval_metric']}"},
                                            frequency=1)],
        )


# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Define the DDP Driver Function
# MAGIC
# MAGIC In most cases where we're leveraging GPUs to train a model, we likely have a dataset that far exceeds the capability of what a single worker node can accomplish. Distributed data parallel (DDP) shards the work over x number of workers, making training both feasible and more manageable. 
# MAGIC
# MAGIC The below driver function wraps the low-level trainable above with Ray Train's DDP version of XGBoost

# COMMAND ----------

def train_driver_fn(config: dict, train_dataset, val_dataset):
    """
    Drives the distributed XGBoost training process using Ray Train.

    This function sets up the XGBoostTrainer, configures scaling (number of workers, GPU usage,
    and resources per worker), and initiates the distributed training by calling `trainer.fit()`.
    It also propagates metrics back to Ray Tune if integrated.

    Args:
        config (dict): A dictionary containing run-level hyperparameters such as
                       'num_workers', 'use_gpu', and a nested 'params' dictionary
                       for XGBoost training parameters.
        train_dataset: The Ray Dataset for training.
        val_dataset: The Ray Dataset for validation.

    Returns:
        None: The function reports metrics to Ray Tune but does not explicitly return a value.
              The trained model artifact is typically handled by Ray Train's checkpointing
              or by the `train_fn_per_worker` if saved directly.
    """
    # Unpack run-level hyperparameters.
    num_workers = config["num_workers"]
    use_gpu = config["use_gpu"]
    params = config['params']

    # Initialize the XGBoostTrainer, which orchestrates the distributed training using Ray.
    trainer = XGBoostTrainer(
      train_loop_per_worker=train_fn_per_worker, # The function to be executed on each worker
      train_loop_config=params,
      # By default Ray uses 1 GPU and 1 CPU per worker if resources_per_worker is not specified.
      # XGBoost is multi-threaded, so multiple CPUs can be assigned per worker, but not GPUs.
      scaling_config=ray.train.ScalingConfig(num_workers=num_workers, 
                                             use_gpu=use_gpu),
      datasets={"train": train_dataset, "val": val_dataset},  # Ray Datasets to be used by the trainer + workers
      run_config=ray.train.RunConfig(storage_path=ray_xgboost_path,                                  
                                    #  name=f"train-trial_id={ray.tune.get_context().get_trial_id()}")
      )
    )
                                    
    result = trainer.fit()
    
    # Propagate metrics back up for Ray Tune. 
    # Ensure 'mlogloss' is the correct metric key based on your eval_metric and results.
    ray.tune.report({params['eval_metric']: result.metrics['mlogloss']})

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3. Ray Tune with Ray Train and MLflow
# MAGIC
# MAGIC XGboost, like most boosting models, are prime for overfitting to the training dataset. Hyperparameter search (HPO) ensures we maximize the performance of the model so it generalizes well beyond the training data. However, orchestrating HPO beyond a single node, or in this case, DDP is challenging. Fortunately, Ray Train slots in nicely with Ray Tune and allows us to simply wrap our DDP driver function in a Ray Tune object. 
# MAGIC
# MAGIC In the Ray Tune object we simply define the parameter space, the integrations with MLFlow, and the run configurations like the number of samples and number of concurrent runs. Furthermore, since we need to shard the dataset for Ray Train, we'll use Ray Data. Ray Data is highly compatible with Databricks tables and can be read in using a SQL warehouse. Note that if your HPO run is going to take long we recommend running `ray_data.materialize()` to materialize the dataset into the object store. 
# MAGIC
# MAGIC https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html#hyperparameter-tuning-with-ray-tune

# COMMAND ----------

import ray
import ray.train
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback
import os


try: 
  ## Option 1 (PREFERRED): Build a Ray Dataset using a Databricks SQL Warehouse
  # Insert your SQL warehouse ID here. I've queried my 100M row dataset using a Small t-shirt sized cluster.

  # Ensure you've set the DATABRICKS_TOKEN so you can query using the warehouse compute
  ds = ray.data.read_databricks_tables(
    warehouse_id= sql_warehouse_id,
    catalog=catalog,
    schema=schema,
    query=f'SELECT * FROM {table}',
  )
  print('read directly from UC')
except: 
  ## Option 2: Build a Ray Dataset using a Parquet files
  # If you have too many Ray nodes, you may not be able to create a Ray dataset using the warehouse method above because of rate limits. One back up solution is to create parquet files from the delta table and build a ray dataset from that. This is not the recommended route because, in essence, you are duplicating data.
  parquet_path = f'/Volumes/{catalog}/{schema}/synthetic_data/{table}'
  ds = ray.data.read_parquet(parquet_path)
  print('read directly from parquet')

train_dataset, val_dataset = ds.train_test_split(test_size=0.25)

# NOTE: If you're expecting the job to run longer than 1 hour, we recommend you materialize the dataset.
# This dumps the datasets into the object store and if you don't have enough memory there, it'll spill over into disk. 
# Fortunately Databricks Clusters are configured to have auto-scaling disk so in theory you can materialize a very large dataset.
# train_dataset.materialize()
# val_dataset.materialize()

# COMMAND ----------

from ray import tune
from ray.tune.tuner import Tuner
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback


# Define resources per HPO trial and calculate max concurrent HPO trials
num_gpu_workers_per_trial = 2
num_hpo_trials = 8
resources = ray.cluster_resources()
total_cluster_gpus = resources.get("GPU") 
max_concurrent_trials = int(total_cluster_gpus // num_gpu_workers_per_trial)


# Define the hyperparameter search space.
# XGB sample hyperparameter configs
param_space = {
    "num_workers": num_gpu_workers_per_trial,
    "use_gpu": True,
    "params":{"objective": "multi:softmax",
              'eval_metric': 'mlogloss', 
              "device": "cuda",
              "num_class": num_labels,
              "tree_method": "hist",
              'sampling_method': 'gradient_based', # gradient memory pressure (default uniform) HIGHLY recommended if using GPU (i.e. device is cuda) + tree_method hist
              'subsample': 0.1, # gradient memory pressure (default 1) but with gradient_based' we can use 0.1
              "learning_rate": tune.uniform(0.01, 0.3),
              "num_estimators": tune.randint(25, 50),
              "max_bin": 256, # histogram features (default 256)
              "max_depth": tune.randint(4, 6), # model complexity (default 6),
              "gamma": 0, # model complexity (default 0) 
              'batch_size': 256*4096, # data complexity
              'prefetch_batches': 4, # data complexity
              'max_quantile_batches': None # data complexity
              } 
}

# Set up search algorithm. Here we use Optuna and use the default the Bayesian sampler (i.e. TPES)
optuna = OptunaSearch(metric=param_space['params']['eval_metric'], 
                      mode="min")

# Set up Tuner job and run.
tuner = tune.Tuner(
    tune.with_parameters(train_driver_fn,
                         train_dataset = train_dataset,
                         val_dataset = val_dataset),
    run_config=tune.RunConfig(name='mlflow',
                              callbacks=[MLflowLoggerCallback(
                                  experiment_name=mlflow_experiment_name,
                                  save_artifact=True,
                                  log_params_on_trial_end=True)]
                              ),
    tune_config=tune.TuneConfig(num_samples=num_hpo_trials,
                                max_concurrent_trials=max_concurrent_trials,
                                search_alg=optuna,
                                ),
    param_space=param_space,
    )

results = tuner.fit()

best_params = results.get_best_result(metric=param_space['params']['eval_metric'], 
                        mode="min").config

print(best_params)