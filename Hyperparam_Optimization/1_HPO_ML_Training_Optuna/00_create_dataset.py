# Databricks notebook source
# MAGIC %md
# MAGIC TO-DO:
# MAGIC Fit a simple regression model on real data and batch infer on synthetic dataset instead of manually defining the output/`income` column

# COMMAND ----------

# MAGIC %pip install dbldatagen -qU
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "amine_elhelou" # Change This
schema = "ray_gtm_examples"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T


table_schema = """`age` DOUBLE,
`workclass` STRING,
`fnlwgt` DOUBLE,
`education` STRING,
`education_num` DOUBLE,
`marital_status` STRING,
`occupation` STRING,
`relationship` STRING,
`race` STRING,
`sex` STRING,
`capital_gain` DOUBLE,
`capital_loss` DOUBLE,
`hours_per_week` DOUBLE,
`native_country` STRING,
`income` STRING"""

primary_key = "id"

# Read data and add a unique id column to use as primary key
raw_df = (
  spark.read.csv("/databricks-datasets/adult/adult.data", schema=table_schema)
  .withColumn(primary_key, F.expr("uuid()"))
  .drop("education")
)

# Trim all strings and replace "?" with None/Null value
clean_df = raw_df
for field in raw_df.schema.fields:
    if isinstance(field.dataType, T.StringType):
        clean_df = clean_df.withColumn(field.name, F.trim(F.col(field.name)))

clean_df = clean_df.na.replace("?", None)

# COMMAND ----------

import dbldatagen as dg


analyzer = dg.DataAnalyzer(sparkSession=spark, df=clean_df)
code =  dg.DataAnalyzer.scriptDataGeneratorFromSchema(clean_df.schema)

# COMMAND ----------

clean_df.summary().display()

# COMMAND ----------

unique_occupations = clean_df.select("occupation").dropna().distinct().collect()
occupation_list = [row.occupation for row in unique_occupations]

unique_relations = clean_df.select("relationship").dropna().distinct().collect()
relation_list = [row.relationship for row in unique_relations]

unique_races = clean_df.select("race").distinct().dropna().collect()
race_list = [row.race for row in unique_races]

unique_genders = clean_df.select("sex").distinct().dropna().collect()
gender_list = [row.sex for row in unique_genders]

unique_countries = clean_df.select("native_country").dropna().distinct().collect()
country_list = [row.native_country for row in unique_countries]

# COMMAND ----------

n_rows = 123000
generation_spec = (
    dg.DataGenerator(sparkSession=spark, 
                     name='synthetic_data', 
                     rows=n_rows,
                     random=True,
                     )
    .withColumn('age', 'double', minValue=17, maxValue=95, step=1)
    .withColumn('workclass', 'string', values=['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Without-pay', 'Never-worked'])
    .withColumn('fnlwgt', 'double', minValue=12285.0, maxValue=1484705.0, step=2)
    .withColumn('education_num', 'double', minValue=1, maxValue=16, step=1)
    .withColumn('marital_status', 'string', values=['Married-spouse-absent','Married-civ-spouse','Divorced','Separated','Never-married', 'Married-AF-spouse'])
    .withColumn('occupation', 'string', values=occupation_list)
    .withColumn('relationship', 'string', values=relation_list)
    .withColumn('race', 'string', values=race_list)
    .withColumn('sex', 'string', values=gender_list)
    .withColumn('capital_gain', 'double', minValue=0.0, maxValue=99999.0, step=100)
    .withColumn('capital_loss', 'double', minValue=0.0, maxValue=5000, step=100)
    .withColumn('hours_per_week', 'double', minValue=1.0, maxValue=100.0, step=1)
    .withColumn('native_country', 'string', values=country_list)
    )

# COMMAND ----------

df_synthetic_data = generation_spec.build()

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import pandas as pd


# Convert Spark DataFrame to Pandas DataFrame
clean_pdf = clean_df.toPandas()

# Define features and target
X = clean_pdf.drop(columns=[primary_key, 'income'])
y = clean_pdf['income']

# Split and train on only 60% of dataset
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=2025)

# Preprocessing pipeline
categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]
numerical_cols = [col for col in X_train if X_train[col].dtype != "object"]
cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='most_frequent')),("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)])

# Create logistic regression pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

# Train the model
clf.fit(X_train, y_train)

# COMMAND ----------

# Predict on synthetic dataset
X_synth_pdf = df_synthetic_data.toPandas()
X_synth_pdf["income"] = clf.predict(X_synth_pdf)

# Convert Pandas DataFrame to Spark DataFrame
df_synthetic_with_label = spark.createDataFrame(X_synth_pdf)

# COMMAND ----------

# Add random id column
df_synthetic_data_with_id =  df_synthetic_with_label \
  .withColumn("id", F.expr("uuid()"))

# COMMAND ----------

# Join
df_full = clean_df.union(df_synthetic_data_with_id)

# COMMAND ----------

df_full.count()

# COMMAND ----------

df_full.tail(1)

# COMMAND ----------

df_full.write.mode("overwrite").option("mergeSchema", True).saveAsTable(f"{catalog}.{schema}.adult_synthetic_raw")

# COMMAND ----------

# df_full.write.format("delta").save(f"/Volumes/{catalog}/{schema}/synthetic-dataset/adult_synthetic")
