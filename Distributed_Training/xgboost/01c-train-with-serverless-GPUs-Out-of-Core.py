# Databricks notebook source
# MAGIC %md
# MAGIC ## Distributed XGBoost with GPUs using Ray on Serverless GPU Compute
# MAGIC
# MAGIC Ray provides a version of XGBoost to perform distributed data parallelism. With drop-in replacements of `xgboost` native classes, XGBoost Ray allows you to leverage multi-node clusters to distribute your training across serverless GPU infrastructure.
# MAGIC
# MAGIC This demo uses a dataset created from `00-create-dataset` with **30M rows × 100 feature columns × 1 target column (2 classes)** for binary classification. This dataset is ~12GB compressed and provides an excellent foundation for distributed training experiments.
# MAGIC
# MAGIC ### Serverless GPU Compute Benefits
# MAGIC
# MAGIC - **On-Demand Scaling**: Automatically provision and scale GPU resources based on workload demands
# MAGIC - **Cost Optimization**: Pay only for compute time used, with automatic resource cleanup
# MAGIC - **No Infrastructure Management**: Focus on ML training without managing underlying hardware
# MAGIC - **High Availability**: Built-in fault tolerance and automatic failover capabilities 
# MAGIC
# MAGIC
# MAGIC #### FAQs
# MAGIC
# MAGIC **When do I switch to a distributed version of XGBoost?**
# MAGIC - Large XGBoost datasets should use distributed data parallelism (DDP). We're using 30M rows here for demonstration purposes.
# MAGIC - Consider single-node and multi-threading across all CPUs, then DDP across multiple nodes with CPUs, then DDP leveraging multiple GPUs.
# MAGIC
# MAGIC **If I'm using GPUs, how much memory (VRAM) do I need for my dataset?**
# MAGIC - 30M rows × 100 columns × 4 bytes (float32) = ~12GB
# MAGIC - We'll need a total of 2-3x the data footprint in VRAM across our GPUs (we'll go with 2x so ~24GB) to train our model
# MAGIC - This total memory accounts for the xgboost DMatrix (can sometimes be bigger than the original dataset), boosting rounds, model size, gradients, and intermediate computations
# MAGIC - **A10G GPUs** (24GB VRAM each) are perfect for this workload - we'll use 1-2 GPUs per model
# MAGIC
# MAGIC ### Serverless GPU Compute Specifications
# MAGIC **Databricks Serverless GPU Compute:**
# MAGIC - **GPU Types**: NVIDIA A10G (24GB VRAM), H100 (80GB VRAM)
# MAGIC - **Auto-scaling**: Automatically scales based on workload demands
# MAGIC - **Billing**: Pay-per-second billing with automatic resource cleanup
# MAGIC - **Availability**: Multi-region support with high availability
# MAGIC - **Integration**: Seamless integration with Unity Catalog and MLflow
# MAGIC
# MAGIC **Recommended Configuration:**
# MAGIC - **Workers**: 2-4 Ray workers for optimal performance
# MAGIC - **GPU Allocation**: 1 A10G GPU per worker (24GB VRAM)
# MAGIC - **Memory**: 32GB RAM per worker for data preprocessing
# MAGIC - **Storage**: Unity Catalog integration for data access
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ```

# COMMAND ----------

# Install required packages for distributed XGBoost training
%pip install -qU ray[all]=2.49.1 xgboost optuna "mlflow<3.0,>=2.17"
%pip install '/Workspace/Users/jon.cheung@databricks.com/ray-on-databricks-rct/distributed-training/XGBoost/databricks.serverless_gpu-0.5.3-py3-none-any.whl'

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure Dataset Parameters
# MAGIC
# MAGIC Set up the parameters for accessing our synthetic dataset and configure the training environment.
# MAGIC

# COMMAND ----------

# Define job inputs
dbutils.widgets.text("catalog_name", "main", "Unity Catalog Name")
dbutils.widgets.text("schema_name", "ray_gtm_examples", "Unity Catalog Schema Name")
dbutils.widgets.text("num_training_rows", "30000000", "rows of data to generate")
dbutils.widgets.text("num_training_columns", "100", "number of feature columns")
dbutils.widgets.text("num_labels", "2", "number of labels in the target column")
dbutils.widgets.text("warehouse_id", "8baced1ff014912d", "ID of warehouse to use")


# Get parameter values (will override widget defaults if run by job)
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
num_training_rows = int(dbutils.widgets.get("num_training_rows"))
num_training_columns = int(dbutils.widgets.get("num_training_columns"))
num_labels = int(dbutils.widgets.get("num_labels"))
warehouse_id = dbutils.widgets.get("warehouse_id")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Set Up Storage and Environment
# MAGIC
# MAGIC Configure the storage paths for Ray data and set up the Databricks environment for distributed training.
# MAGIC

# COMMAND ----------

import os 

table = f"synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels"
label="target"

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.errors import ResourceAlreadyExists

w = WorkspaceClient()

# If running in a multi-node cluster, this is where you
# should configure the run's persistent storage that is accessible
# across all worker nodes.
ray_xgboost_path = f'/Volumes/{catalog_name}/{schema_name}/ray_data_tmp_dir' 

if not os.path.exists(ray_xgboost_path):
    created_volume = w.volumes.create(catalog_name=catalog_name,
                                        schema_name=schema_name,
                                        name='ray_data_tmp_dir',
                                        volume_type=catalog.VolumeType.MANAGED
                                        )
    print(f"Volume 'synthetic_data' at {ray_xgboost_path} created successfully")
else:
    print(f"Volume {ray_xgboost_path} already exists. Skipping volumes creation.")




# # Set the parameters here so mlflow works properly at the Ray head + worker nodes
os.environ['DATABRICKS_HOST'] =  dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Load Dataset with Ray
# MAGIC
# MAGIC Load the synthetic dataset using Ray Datasets, with fallback to Parquet files if needed.
# MAGIC

# COMMAND ----------

import ray

def read_ray_dataset(catalog, schema, table):
  try: 
    ## Option 1 (PREFERRED): Build a Ray Dataset using a Databricks SQL Warehouse
    # Insert your SQL warehouse ID here. I've queried my 100M row dataset using a Small t-shirt sized cluster.
    # Ensure you've set the DATABRICKS_TOKEN so you can query using the warehouse compute
    ds = ray.data.read_databricks_tables(
      warehouse_id=warehouse_id,
      catalog=catalog,
      schema=schema,
      query=f'SELECT * FROM {table}',
    )
    print('read directly from UC')
  except: 
    ## Option 2: Fallback option to build a Ray Dataset using Parquet files
    # If you have too many Ray nodes, you may not be able to create a Ray dataset using the warehouse method above because of rate limits. One back up solution is to create parquet files from the delta table and build a ray dataset from that. This is not a recommended route because you are duplicating data.
    parquet_path = f'/Volumes/{catalog}/{schema}/synthetic_data/{table}'
    ds = ray.data.read_parquet(parquet_path)
    print('read directly from parquet')

  train_dataset, val_dataset = ds.train_test_split(test_size=0.25)
  return train_dataset, val_dataset




# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Define Training Function
# MAGIC
# MAGIC Create the per-worker training function that will be executed on each Ray worker node.
# MAGIC

# COMMAND ----------

import xgboost
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

    # Convert shards to pandas DataFrames
    train_df = train_shard.materialize().to_pandas()
    val_df = val_shard.materialize().to_pandas()

    train_X = train_df.drop(label, axis=1)
    train_y = train_df[label]
    val_X = val_df.drop(label, axis=1)
    val_y = val_df[label]
    
    dtrain = xgboost.DMatrix(train_X, label=train_y)
    deval = xgboost.DMatrix(val_X, label=val_y)

    # Do distributed data-parallel training.
    # Ray Train sets up the necessary coordinator processes and
    # environment variables for workers to communicate with each other.
    evals_results = {}
    bst = xgboost.train(
        params,
        dtrain=dtrain,
        evals=[(deval, "validation")],
        num_boost_round=params['num_estimators'],
        evals_result=evals_results,
        callbacks=[RayTrainReportCallback(metrics={params['eval_metric']: f"validation-{params['eval_metric']}"},
                                          frequency=1)],
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Define Training Driver
# MAGIC
# MAGIC Create the training driver function that orchestrates the distributed XGBoost training process.
# MAGIC

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
                                             use_gpu=use_gpu,
                                             resources_per_worker={"CPU": 12, "GPU": 1}),
      datasets={"train": train_dataset, "val": val_dataset},  # Ray Datasets to be used by the trainer + workers
      run_config=ray.train.RunConfig(storage_path=ray_xgboost_path,                                  
                                    #  name=f"train-trial_id={ray.tune.get_context().get_trial_id()}"
                                    )
    )
    
                                    
    result = trainer.fit()
    
    # Propagate metrics back up for Ray Tune. 
    # Ensure 'mlogloss' is the correct metric key based on your eval_metric and results.
    ray.tune.report({params['eval_metric']: result.metrics['mlogloss']},checkpoint=result.checkpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Hyperparameter Optimization with Ray Tune
# MAGIC
# MAGIC Configure Ray Tune for automated hyperparameter optimization with MLflow integration.
# MAGIC
# MAGIC **Key Features:**
# MAGIC - **Optuna Search**: Bayesian optimization for efficient hyperparameter search
# MAGIC - **MLflow Integration**: Automatic experiment tracking and model logging
# MAGIC - **Distributed Training**: Parallel hyperparameter trials across multiple workers
# MAGIC - **Resource Management**: Automatic scaling and resource allocation
# MAGIC
# MAGIC **Documentation**: [Ray Tune Hyperparameter Optimization](https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html#hyperparameter-tuning-with-ray-tune)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Configurable Hyperparameter Optimization (HPO) Parameters
# MAGIC
# MAGIC The following parameters can be configured for hyperparameter optimization (HPO) with Ray Tune and Optuna. Some are fixed for this demo, while others are tunable as part of the search space.
# MAGIC
# MAGIC | Parameter              | Description                                                      | Tunable (HPO) | Example/Default Value or Range |
# MAGIC |------------------------|------------------------------------------------------------------|:-------------:|-------------------------------|
# MAGIC | `num_workers`          | Number of Ray workers (each with 1 GPU)                          | No            | 2                             |
# MAGIC | `use_gpu`              | Whether to use GPU for training                                  | No            | True                          |
# MAGIC | `objective`            | XGBoost objective function                                       | No            | `multi:softmax`               |
# MAGIC | `eval_metric`          | Evaluation metric for optimization                               | No            | `mlogloss`                    |
# MAGIC | `tree_method`          | XGBoost tree construction method                                 | No            | `hist`                        |
# MAGIC | `device`               | Device type for XGBoost                                          | No            | `cuda`                        |
# MAGIC | `num_class`            | Number of classes (labels) in the target column                  | No            | 2                             |
# MAGIC | `learning_rate`        | Learning rate for boosting                                       | Yes           | 0.01–0.3 (uniform)            |
# MAGIC | `num_estimators`       | Number of boosting rounds (trees)                                | Yes           | 200–300 (integer)               |
# MAGIC | `max_concurrent_trials`| Maximum concurrent HPO trials                                    | No            | 4                             |
# MAGIC | `num_hpo_trials`       | Total number of HPO trials to run                                | No            | 8                             |
# MAGIC
# MAGIC > **Note:**
# MAGIC > - `learning_rate` and `num_estimators` are tunable by Optuna during HPO.
# MAGIC > - You can adjust the search space and fixed values in the code (see the next cell for details).

# COMMAND ----------

import mlflow
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self):
      
      # Load dataset as distributed Ray Dataset
      train_dataset, val_dataset = read_ray_dataset(catalog_name, schema_name, table)
      
      # Define resources per HPO trial and calculate max concurrent HPO trials
      num_workers = 2
      num_hpo_trials = 8
      resources = ray.cluster_resources()
      total_cluster_gpus = resources.get("GPU") 
      # max_concurrent_trials = int(total_cluster_gpus // num_gpu_workers_per_trial)
      max_concurrent_trials = 4


      # Define the hyperparameter search space.
      # XGB sample hyperparameter configs
      param_space = {
          "num_workers": num_workers,
          "use_gpu": True,
          "params":{"objective": "multi:softmax",
                    'eval_metric': 'mlogloss', 
                    "tree_method": "hist",
                    "device": "cuda",
                    "num_class": num_labels,
                    "learning_rate": tune.uniform(0.01, 0.3),
                    "num_estimators": tune.randint(20, 30)}
      }

      # # Set up search algorithm. Here we use Optuna and use the default the Bayesian sampler (i.e. TPES)
      optuna = OptunaSearch(metric=param_space['params']['eval_metric'], 
                            mode="min")

      with mlflow.start_run() as run:
      # Set up Tuner job and run.
        tuner = tune.Tuner(
          tune.with_parameters(train_driver_fn,
                              train_dataset = train_dataset,
                              val_dataset = val_dataset),
          run_config=tune.RunConfig(name='test_run',
                                      storage_path = '/Volumes/main/ray_gtm_examples/ray_data_tmp_dir',
                                      callbacks=[MLflowLoggerCallback(
                                      save_artifact=True,
                                      tags={"mlflow.parentRunId": run.info.run_id},
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

        return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Execute Distributed Training
# MAGIC
# MAGIC Launch the distributed XGBoost training with hyperparameter optimization using Serverless GPU compute.
# MAGIC

# COMMAND ----------

from serverless_gpu.ray import ray_launch 

@ray_launch(gpus=8, gpu_type='A10', remote=True)
def my_ray_function():
    runner = TaskRunner.remote()
    return ray.get(runner.run.remote())

results = my_ray_function.distributed()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Model Registration and Inference
# MAGIC
# MAGIC Register the best model with MLflow and demonstrate inference capabilities.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log and Register model to MLflow for inference

# COMMAND ----------

notebook_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-1]
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[2]

# COMMAND ----------

import mlflow

def get_run_id_by_tag(experiment_name, tag_key, tag_value):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.{tag_key} = '{tag_value}'"
        )
        if runs:
            return runs[0].info.run_id
    return None

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(f"/Users/{username}/{notebook_name}")
latest_run = None
if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string='attributes.run_name LIKE "jobTaskRun%"',
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if runs:
        latest_run = runs[0]
run_id = latest_run.info.run_id

# COMMAND ----------

results = results[0] if type(results) == list else results
best_params = results.get_best_result(metric="mlogloss", 
                    mode="min")
booster = RayTrainReportCallback.get_model(best_params.checkpoint)


# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Sample Data
sample_data = spark.read.table(f"{catalog_name}.{schema_name}.{table}").limit(5).toPandas()

with mlflow.start_run(run_id=run_id) as run:
    logged_model = mlflow.xgboost.log_model(
        booster, 
        "model",
        input_example=sample_data[[col for col in sample_data.columns if col != 'target']])

# COMMAND ----------

# Load the registered model and make predictions
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

# Make predictions on sample data
predictions = loaded_model.predict(sample_data[[col for col in sample_data.columns if col != 'target']])
print(f"Predictions: {predictions}")
print(f"Prediction shape: {predictions.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps
# MAGIC
# MAGIC ### What We Accomplished
# MAGIC
# MAGIC This notebook demonstrated a complete end-to-end distributed machine learning pipeline:
# MAGIC
# MAGIC 1. **Dataset Loading**: Successfully loaded 30M synthetic records using Ray Datasets
# MAGIC 2. **Distributed Training**: Trained XGBoost models across multiple GPU workers
# MAGIC 3. **Hyperparameter Optimization**: Used Ray Tune with Optuna for automated HPO
# MAGIC 4. **Model Registration**: Registered the best model with MLflow for production use
# MAGIC 5. **Inference**: Demonstrated model loading and prediction capabilities
# MAGIC
# MAGIC ### Key Technologies Used
# MAGIC
# MAGIC - **Ray**: Distributed computing framework for scalable ML
# MAGIC - **XGBoost**: Gradient boosting with GPU acceleration
# MAGIC - **Serverless GPU**: On-demand GPU compute without infrastructure management
# MAGIC - **MLflow**: Model lifecycle management and experiment tracking
# MAGIC - **Unity Catalog**: Data governance and access control
# MAGIC
# MAGIC ### Performance Benefits
# MAGIC
# MAGIC - **Scalability**: Handle datasets from 30M to 1B+ rows
# MAGIC - **Cost Efficiency**: Pay-per-second billing with automatic resource cleanup
# MAGIC - **Speed**: Multi-GPU training reduces training time significantly
# MAGIC - **Reliability**: Built-in fault tolerance and checkpointing
# MAGIC
# MAGIC
# MAGIC ### Additional Resources
# MAGIC
# MAGIC - [Ray Documentation](https://docs.ray.io/)
# MAGIC - [XGBoost Documentation](https://xgboost.readthedocs.io/)
# MAGIC - [MLflow Documentation](https://mlflow.org/docs/)
# MAGIC - [Databricks Serverless GPU](https://docs.databricks.com/en/compute/serverless/gpu.html)
# MAGIC

# COMMAND ----------

