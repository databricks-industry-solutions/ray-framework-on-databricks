# Databricks notebook source
# MAGIC %md
# MAGIC # HPO for traditional ML using Optuna and Ray Tune
# MAGIC
# MAGIC In this example we'll cover the new recommended ways for performing distributed hyperparameter optimization on databricks, while also leveraging MLflow for tracking experiments and the feature engineering client to ensure lineage between models and feature tables.
# MAGIC
# MAGIC Based on dataset size, training time/model complexity, hardware availability and auto-scaling needs, both single and multinode approaches can be considered:
# MAGIC
# MAGIC 1. For single node use [Optuna](https://github.com/optuna/optuna) which naitvely supports mlflow callbacks. This approach is recommended if:
# MAGIC     1. You only have a single/big machine available
# MAGIC     2. Single training run takes less than 2 seconds (based on dataset size and model architecture)
# MAGIC 2. For multi-node use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) by leveraging [ray on spark](https://docs.databricks.com/en/machine-learning/ray/index.html)
# MAGIC
# MAGIC
# MAGIC **VERY IMPORTANT:** If using GPUs set the `spark.task.resource.gpu.amount 0` spark config on your (multinode) cluster.
# MAGIC
# MAGIC Tested on:
# MAGIC ```
# MAGIC Databricks Machine Learning Runtime 15.4LTS
# MAGIC databricks-feature-engineering-client==0.8.0
# MAGIC mlflow=2.19.0
# MAGIC ray==2.40.0
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -qU databricks-feature-engineering optuna optuna-integration mlflow ray[default]
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

catalog = "amine_elhelou" # Change This
schema = "ray_gtm_examples"
table = "adult_synthetic_raw"
feature_table_name = "features_synthetic"
labels_table_name = "labels_synthetic"
label="income"
primary_key = "id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Create feature table

# COMMAND ----------

# MAGIC %md
# MAGIC For this example we'll be using an augmented version of the [UCI's Adult Census](https://archive.ics.uci.edu/dataset/2/adult) classification dataset

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Drop feature tables if they already exist [OPTIONAL]
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{feature_table_name}")
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{labels_table_name}")

# COMMAND ----------

# DBTITLE 1,Read synthetic dataset and write as feature and labels tables
# Read dataset and remove duplicate keys
adult_synthetic_df = spark.read.table(f"{catalog}.{schema}.{table}").dropDuplicates([primary_key])

# Do this ONCE
try:
  print(f"Creating features and labels tables...")
  fe.create_table(
    name=f"{catalog}.{schema}.{feature_table_name}",
    primary_keys=[primary_key],
    df=adult_synthetic_df.drop(label),
    description="Adult Census feature"
  )

  # Extract ground-truth labels and primary key into separate dataframe
  training_ids_df = adult_synthetic_df.select(primary_key, label)
  training_ids_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{labels_table_name}")
  print(f"... OK!")

except Exception as e:
  print(f"Table {feature_table_name} and {labels_table_name} already exists")

# COMMAND ----------

print(f"Created following catalog/schema/tables and variables:\n catalog = {catalog}\n schema = {schema}\n labels_table_name = {labels_table_name}\n feature_table_name = {feature_table_name}\n label = {label}\n primary_key = {primary_key}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load training dataset
# MAGIC
# MAGIC First create feature lookups and training dataset specs

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup


feature_lookups = [
  FeatureLookup(
    table_name=f"{catalog}.{schema}.{feature_table_name}",
    lookup_key=primary_key
  )
]

# COMMAND ----------

# Read labels and ids
training_ids_df = spark.table(f"{catalog}.{schema}.{labels_table_name}")

# Create training set
training_set = fe.create_training_set(
    df=training_ids_df,
    feature_lookups=feature_lookups,
    label=label,
    exclude_columns=[primary_key]
  )

# Get raw features and label
training_pdf = training_set.load_df().toPandas()
X, Y = (training_pdf.drop(label, axis=1), training_pdf[label])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Universal preprocessing pipeline

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def initialize_preprocessing_pipeline(X_train_in:pd.DataFrame):
  """
  Helper function to create pre-processing pipeline
  """
  # Universal preprocessing pipeline
  categorical_cols = [col for col in X_train_in if X_train_in[col].dtype == "object"]
  numerical_cols = [col for col in X_train_in if X_train_in[col].dtype != "object"]
  cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='most_frequent')),("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])
  num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='median')), ("scaler", StandardScaler())])

  preprocessor = ColumnTransformer(
    transformers=[
      ("cat", cat_pipeline, categorical_cols),
      ("num", num_pipeline, numerical_cols)
    ],
    remainder="passthrough",
    sparse_threshold=0
  )
  
  return preprocessor

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define high-level experiments parameters
# MAGIC
# MAGIC Recommended Cluster Size for this tutorial:
# MAGIC * Driver node with 4 cores & min 30GB of RAM
# MAGIC * 2 worker nodes with 4 cores each & min 30GB of RAM

# COMMAND ----------

num_cpu_cores_per_worker = 4 # total cpu to use in each worker node
max_worker_nodes = 2
n_trials = 40 # Number of trials (arbitrary for demo purposes but has to be at least > 30 to see benefits of multinode)
rng_seed = 2025 # Random Number Generation Seed for random states

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Scaling HPO using Optuna on a single node
# MAGIC Optuna is an advanced hyperparameter optimization framework designed specifically for machine learning tasks. Here are the key ways Optuna conducts hyperparameter optimization:
# MAGIC
# MAGIC 1. Define-by-run API: Optuna uses an imperative, define-by-run style API that allows users to dynamically construct the search space for hyperparameters.
# MAGIC 2. Objective Function: Users define an objective function that takes a "trial" object as input and returns a score to be optimized. The objective function typically constructs and evaluates a model using hyperparameters suggested by the trial object.
# MAGIC 3. Sampling Algorithms:
# MAGIC     1. **Tree-structured Parzen Estimator (Default)** - Bayesian optimization to efficiently search the hyperparameter space.
# MAGIC     2. Random Sampling - Randomly samples hyperparameter values from the search space.
# MAGIC     3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - An evolutionary algorithm for difficult non-linear non-convex optimization problems.
# MAGIC     4. Grid Search - Exhaustively searches through a manually specified subset of the hyperparameter space.
# MAGIC     5. Quasi-Monte Carlo sampling - Uses low-discrepancy sequences to sample the search space more uniformly than pure random sampling.
# MAGIC     6. NSGA-II (Non-dominated Sorting Genetic Algorithm II) - A multi-objective optimization algorithm.
# MAGIC     7. Gaussian Process-based sampling (i.e. Kriging) - Uses Gaussian processes for Bayesian optimization.
# MAGIC     8. Optuna also allows implementing custom samplers by inheriting from the `BaseSampler` class.
# MAGIC 4. Pruning: Optuna implements pruning algorithms to early-stop unpromising trials
# MAGIC 5. Study Object: Users create a "study" object that manages the optimization process. The study.optimize() method is called to start the optimization, specifying the objective function and number of trials.
# MAGIC 6. Parallel Execution: Optuna scales to parallel execution across single or multiple nodes.
# MAGIC
# MAGIC
# MAGIC We'll leverage Optuna's native MLflow integration: the `MLflowCallback` which helps automatically logging the hyperparameters and metrics.
# MAGIC
# MAGIC Then we'll run an Optuna hyperparameter optimization study by passing the `MLflowCallback` object to the optimize function.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Define objective/loss function and search space to optimize
# MAGIC The search space here is defined by calling functions such as `suggest_categorical`, `suggest_float`, `suggest_int` for the Trial object that is passed to the objective function. Optuna allows to define the search space dynamically.
# MAGIC
# MAGIC Refer to the documentation for:
# MAGIC * [optuna.samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) for the choice of samplers
# MAGIC * [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) for a full list of functions supported to define a hyperparameter search space.

# COMMAND ----------

# DBTITLE 1,Choose sampler type
import optuna


optuna_sampler = optuna.samplers.TPESampler(
  n_startup_trials=num_cpu_cores_per_worker*max_worker_nodes,
  n_ei_candidates=num_cpu_cores_per_worker*max_worker_nodes,
  seed=rng_seed
)

# COMMAND ----------

import mlflow
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class ObjectiveOptuna(object):
  """
  a callable class for implementing the objective function. It takes the training dataset by a constructor's argument
  instead of loading it in each trial execution. This will speed up the execution of each trial
  """
  def __init__(self, X_train_in:pd.DataFrame, Y_train_in:pd.Series):
    """
    X_train_in: features
    Y_train_in: label
    """

    # Create pre-processing pipeline
    self.preprocessor = initialize_preprocessing_pipeline(X_train_in)

    # Split into training and validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_in, Y_train_in, test_size=0.1, random_state=rng_seed)

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_val = X_val
    self.Y_val = Y_val
    
  def __call__(self, trial):
    """
    Wrapper call containing data processing pipeline, training and hyperparameter tuning code.
    The function returns the weighted F1 accuracy metric to maximize in this case.
    """

    # Define list of classifiers to test
    classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "RandomForest", "LightGBM"]) #, "XGBoost"])
    
    if classifier_name == "LogisticRegression":
      # Optimize tolerance and C hyperparameters
      lr_C = trial.suggest_float("C", 1e-2, 1, log=True)
      lr_tol = trial.suggest_float('tol' , 1e-6 , 1e-3, step=1e-6)
      classifier_obj = LogisticRegression(C=lr_C, tol=lr_tol, random_state=rng_seed)

    elif classifier_name == "RandomForest":
      # Optimize number of trees, tree depth, min_sample split and leaf hyperparameters
      n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
      max_depth = trial.suggest_int("max_depth", 3, 10)
      min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
      min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
      classifier_obj = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=rng_seed)

    elif classifier_name == "LightGBM":
      # Optimize number of trees, tree depth, learning rate and maximum number of bins hyperparameters
      n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
      max_depth = trial.suggest_int("max_depth", 3, 10)
      learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.9)
      max_bin = trial.suggest_int("max_bin", 2, 256)
      num_leaves = trial.suggest_int("num_leaves", 2, 256),
      classifier_obj = LGBMClassifier(force_row_wise=True, verbose=-1,
                                      n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, max_bin=max_bin, num_leaves=num_leaves, random_state=rng_seed)
    
    # Assemble the pipeline
    this_model = Pipeline(steps=[("preprocessor", self.preprocessor), ("classifier", classifier_obj)])

    # Fit the model
    mlflow.sklearn.autolog(disable=True) # Disable mlflow autologging to avoid logging artifacts for every run
    this_model.fit(self.X_train, self.Y_train)

    # Predict on validation set
    y_val_pred = this_model.predict(self.X_val)

    # Calculate and return F1-Score
    f1_score_binary= f1_score(self.Y_val, y_val_pred, average="binary", pos_label='>50K')

    return f1_score_binary

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Test on driver node

# COMMAND ----------

# DBTITLE 1,Quick test/debug
objective_fn = ObjectiveOptuna(X, Y)
study_debug = optuna.create_study(direction="maximize", study_name="test_debug", sampler=optuna_sampler)
study_debug.optimize(objective_fn, n_trials=2)
print(f"Elapsed time test: {study_debug.trials[-1].datetime_complete - study_debug.trials[0].datetime_start}")

# COMMAND ----------

# DBTITLE 1,Get best run's params
print("Best trial:")
best_trial = study_debug.best_trial
print(f"  F1_score: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Add `MLflowCallback`, wrap training function and execute as part of parent/child mlflow run

# COMMAND ----------

from optuna.integration.mlflow import MLflowCallback


experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflc = MLflowCallback(
    tracking_uri="databricks",
    metric_name="f1_score_val",
    create_experiment=False,
    mlflow_kwargs={
        "experiment_id": experiment_id,
        "nested":True
    }
)

# COMMAND ----------

import warnings
import pandas as pd
from mlflow.types.utils import _infer_schema
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc


def optuna_hpo_fn(n_trials: int, features: pd.DataFrame, labels: pd.Series, model_name: str, experiment_id: str, include_mlflc: bool) -> optuna.study.study.Study:
    # Start mlflow run
    with mlflow.start_run(run_name="single_node_hpo", experiment_id=experiment_id) as parent_run:

        # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=rng_seed)

        # Kick distributed HPO as nested runs
        objective_fn = ObjectiveOptuna(X_train, Y_train)
        optuna_study = optuna.create_study(direction="maximize", study_name=f"single_node_hpo_study", sampler=optuna_sampler)
        if include_mlflc:
            optuna_study.optimize(objective_fn, n_trials=n_trials, callbacks=[mlflc])
        else:
            optuna_study.optimize(objective_fn, n_trials=n_trials)

        # Extract best trial info
        best_model_params = optuna_study.best_params
        best_model_params["random_state"] = rng_seed
        classifier_type = best_model_params.pop('classifier')

        # Reproduce best classifier
        if classifier_type  == "LogisticRegression":
            best_model = LogisticRegression(**best_model_params)
        elif classifier_type == "RandomForestClassifier":
            best_model = RandomForestClassifier(**best_model_params)
        elif classifier_type == "LightGBM":
            best_model = LGBMClassifier(force_row_wise=True, verbose=-1, **best_model_params)

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, log_models=False, silent=True)
        
        # Fit best model and log using FE client in parent run
        model_pipeline = Pipeline(steps=[("preprocessor", objective_fn.preprocessor), ("classifier", best_model)])
        model_pipeline.fit(X_train, Y_train)
        
        fe.log_model(
            model=model_pipeline,
            artifact_path="model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name
        )

        # Evaluate model and log into experiment
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model_pipeline)
        
        # Log metrics for the training set
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(label):Y_train}),
            targets=label,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "training_" , "pos_label": ">50K" }
        )

        # Log metrics for the test set
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(label):Y_test}),
            targets=label,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "test_" , "pos_label": ">50K" }
        )

        mlflow.end_run()
        
        return optuna_study

# COMMAND ----------

# MAGIC %md
# MAGIC #### Execute
# MAGIC

# COMMAND ----------

# Disable mlflow autologging to minimize overhead
mlflow.autolog(disable=True) # Disable mlflow autologging

# Setting the logging level DEBUG to avoid too verbose logs
optuna.logging.set_verbosity(optuna.logging.DEBUG)
optuna.logging.disable_propagation()

# Invoke training function on driver node
single_node_study = optuna_hpo_fn(
  n_trials=n_trials,
  features=X,
  labels=Y,
  model_name=f"{catalog}.{schema}.hpo_model_optuna_single_node",
  experiment_id=experiment_id,
  include_mlflc=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **PS : Enabling mlflow logging adds substantial overhead so it can be removed/disabled**

# COMMAND ----------

# DBTITLE 1,Elapsed time excluding best model fitting, logging and evaluation
print(f"Elapsed time for HPO on driver/single node for {n_trials} experiments: {single_node_study.trials[-1].datetime_complete - single_node_study.trials[0].datetime_start}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Scaling HPO with Ray Tune
# MAGIC
# MAGIC Ray Tune allows hyperparameter optimization through several key capabilities:
# MAGIC 1. **Search space definition**: defining a search space for hyperparameters using Ray Tune's API, specifying ranges and distributions for parameters to explore.
# MAGIC 2. **Trainable functions**: wrapping the model training code in a "trainable" function that Ray Tune can call with different hyperparameter configurations.
# MAGIC 3. **Search algorithms**: Ray Tune provides search algorithms like grid, random, and Bayesian. Ray Tune also integrates well with other frameworks like Optuna and Hyperopt. 
# MAGIC 4. **Parallel execution**: running multiple trials (hyperparameter configurations) in parallel across a cluster of machines.
# MAGIC 5. **Early stopping**: Ray Tune supports early stopping of poorly performing trials to save compute resources
# MAGIC
# MAGIC To get started with Ray, we'll set up a Ray cluster and then Ray Tune workflow, encapsulating all the above.

# COMMAND ----------

# DBTITLE 1,Initialize Ray on spark cluster
import os
import ray
from mlflow.utils.databricks_utils import get_databricks_env_vars
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES


# Cluster cleanup
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

# Set configs based on your cluster size
num_cpu_cores_per_worker = 3 # total cpu to use in each worker node (total_cores - 1 to leave one core for spark)
num_cpus_head_node = 3 # Cores to use in driver node (total_cores - 1)
max_worker_nodes = 2

# Set databricks credentials as env vars
mlflow_dbrx_creds = get_databricks_env_vars("databricks")
os.environ["DATABRICKS_HOST"] = mlflow_dbrx_creds['DATABRICKS_HOST']
os.environ["DATABRICKS_TOKEN"] = mlflow_dbrx_creds['DATABRICKS_TOKEN']

ray_conf = setup_ray_cluster(
  min_worker_nodes=max_worker_nodes,
  max_worker_nodes=max_worker_nodes,
  num_cpus_head_node= num_cpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_head_node=0,
  num_gpus_worker_node=0
)
os.environ['RAY_ADDRESS'] = ray_conf[0]

# COMMAND ----------

# MAGIC %md
# MAGIC **Ray Tune workflow:**
# MAGIC 1. Define search space by wrapping training code in a trainable function
# MAGIC 2. Configure the tuning process (algorithm, resources, etc.): we'll use `Optuna` as the search algorithm for an apples-to-apples comparison but different schedulers can be used as well.
# MAGIC 3. Launch the tuning job using the `Tuner` class
# MAGIC 4. Analyze results to find the best hyperparameters and fit the best model usinf the `fe()` client

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define objective/loss function and search space
# MAGIC
# MAGIC * pick and configure `Optuna` from [ray.tune.search](https://docs.ray.io/en/latest/tune/api/suggestion.html)
# MAGIC * use the [ConcurrencyLimiter](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.ConcurrencyLimiter.html#ray.tune.search.ConcurrencyLimiter) to limit number of concurrent trials

# COMMAND ----------

import pandas as pd
import ray
from lightgbm import LGBMClassifier
from ray import train
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from typing import Dict, Optional, Any


def objective_ray(config: dict, X_train_in:pd.DataFrame, Y_train_in:pd.Series):
    """
    Wrapper training/objective function for ray tune
    """

    if config['classifier']== "LogisticRegression":
      # Optimize tolerance and C hyperparameters
        lr_C = config["C"]
        lr_tol = config['tol']
        classifier_obj = LogisticRegression(C=lr_C, tol=lr_tol, random_state=rng_seed)

    elif config['classifier']== "RandomForest":
      # Optimize number of trees, tree depth, min_sample split and leaf hyperparameters
        n_estimators = config["n_estimators"]
        max_depth = config["max_depth"]
        min_samples_split = config["min_samples_split"]
        min_samples_leaf = config["min_samples_leaf"]
        classifier_obj = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf, random_state=rng_seed)

    else:
        # Optimize number of trees, tree depth, learning rate and maximum number of bins hyperparameters
        n_estimators = config["n_estimators"]
        max_depth = config["max_depth"]
        learning_rate = config["learning_rate"]
        max_bin = config["max_bin"]
        num_leaves = config["num_leaves"],
        classifier_obj = LGBMClassifier(force_row_wise=True, verbose=-1,
                                        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, max_bin=max_bin, num_leaves=num_leaves, random_state=rng_seed)


    # Initialize pipeline
    preprocessor = initialize_preprocessing_pipeline(X_train_in)
    this_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier_obj)])

    # Split into training and validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_in, Y_train_in, test_size=0.1, random_state=rng_seed)

    # Fit the model
    this_model.fit(X_train, Y_train)

    # Predict on validation set
    y_val_pred = this_model.predict(X_val)

    # Calculate and return F1-Score
    f1_score_binary= f1_score(Y_val, y_val_pred, average="binary", pos_label='>50K')

    # Log for ray report/verbose
    train.report({"f1_score_val": f1_score_binary}) # [OPTIONAL] to view in ray logs


# The search space sampling can be defined by 
# 1) a "define by run function" (see below) which uses Optuna's native sampling OR
# 2) a dictionary with a different sampling function (i.e. Ray Tune's sampler)
def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
    """
    Define-by-run function to create the search space.
    For more information, see https://optuna.readthedocs.io/en/stable\
    /tutorial/10_key_features/002_configurations.html
    """

    classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "RandomForest", "LightGBM"])

    # Define-by-run allows for conditional search spaces.
    if classifier_name == "LogisticRegression":
        trial.suggest_float("C", 1e-2, 1, log=True)
        trial.suggest_float('tol' , 1e-6 , 1e-3, step=1e-6)
    elif classifier_name == "RandomForest":
        trial.suggest_int("n_estimators", 10, 200, log=True)
        trial.suggest_int("max_depth", 3, 10)
        trial.suggest_int("min_samples_split", 2, 10)
        trial.suggest_int("min_samples_leaf", 1, 10)
    else:
        trial.suggest_int("n_estimators", 10, 200, log=True)
        trial.suggest_int("max_depth", 3, 10)
        trial.suggest_float("learning_rate", 1e-2, 0.9)
        trial.suggest_int("max_bin", 2, 256)
        trial.suggest_int("num_leaves", 2, 256)


# Define Optuna search algo
searcher = OptunaSearch(space=define_by_run_func, metric="f1_score_val", mode="max")
# Note: A concurrency limiter, limits the number of parallel trials. This is important for Bayesian search (inherently sequential) as too many parallel trials reduces the benefits of priors to inform the next search round.
algo = ConcurrencyLimiter(searcher, max_concurrent=num_cpu_cores_per_worker*max_worker_nodes+num_cpus_head_node)

# COMMAND ----------

import mlflow


# Grab experiment and model name
experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
model_name=f"{catalog}.{schema}.hpo_model_ray_tune_optuna"

# Hold-out Test/Train set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rng_seed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute

# COMMAND ----------

import warnings
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback


mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name ='ray_tune_native_mlflow_callback', experiment_id=experiment_id) as parent_run:
    
    # Tune with callback
    tuner = tune.Tuner(
        ray.tune.with_parameters(
            objective_ray,
            X_train_in = X_train, Y_train_in = Y_train),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=n_trials,
            reuse_actors = True # Highly recommended for short training jobs (NOT RECOMMENDED FOR GPU AND LONG TRAINING JOBS)
            ),
        run_config=train.RunConfig(
            name="ray-tune-optuna",
            callbacks=[
                MLflowLoggerCallback(
                    experiment_name=experiment_name,
                    save_artifact=False,
                    tags={"mlflow.parentRunId": parent_run.info.run_id})]
        )
    )

    multinode_results = tuner.fit()

    # Extract best trial info
    best_model_params = multinode_results.get_best_result(metric="f1_score_val", mode="max", scope='last').config
    best_model_params["random_state"] = rng_seed
    classifier_type = best_model_params.pop('classifier')
    
    # Reproduce best classifier
    if classifier_type  == "LogisticRegression":
        best_model = LogisticRegression(**best_model_params)
    elif classifier_type == "RandomForestClassifier":
        best_model = RandomForestClassifier(**best_model_params)
    elif classifier_type == "LightGBM":
        best_model = LGBMClassifier(force_row_wise=True, verbose=-1, **best_model_params)

    
    # Fit best model and log using FE client in parent run
    preprocessor = initialize_preprocessing_pipeline(X_train)
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_model)])
    model_pipeline.fit(X_train, Y_train)
    
    fe.log_model(
        model=model_pipeline,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=model_name
    )

    # Evaluate model and log into experiment
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model_pipeline)
    
    # Log metrics for the training set
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(label):Y_train}),
        targets=label,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": ">50K" }
    )

    # Log metrics for the test set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(label):Y_test}),
        targets=label,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": ">50K" }
    )

    mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Elapsed time excluding best model fitting, logging and evaluation
rt_trials_pdf = multinode_results.get_dataframe()
print(f"Elapsed time for multinode HPO with ray tune for {n_trials} experiments:: {(rt_trials_pdf['timestamp'].iloc[-1] - rt_trials_pdf['timestamp'].iloc[0] + rt_trials_pdf['time_total_s'].iloc[-1])/60} min")
