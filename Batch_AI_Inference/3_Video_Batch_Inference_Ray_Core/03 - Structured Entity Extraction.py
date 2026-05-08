# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Structured Entity Extraction
# MAGIC
# MAGIC Reads the raw VLM output from notebook `02` (default table `video_inferences`) and structures it into a queryable Delta table using Databricks AI Functions.
# MAGIC
# MAGIC The structuring step calls `databricks-meta-llama-3-3-70b-instruct` via `ai_query` with a `responseFormat` JSON schema. Server-side schema enforcement guarantees the column always parses — no `try/except json.loads` needed downstream.
# MAGIC
# MAGIC **Cluster:** any classic compute (or attach to a SQL warehouse and run as Databricks SQL). Inference is delegated to the foundation model API endpoint, so this notebook does no heavy lifting locally.
# MAGIC
# MAGIC **Prerequisite:** `databricks-meta-llama-3-3-70b-instruct` foundation model API endpoint enabled in the workspace.

# COMMAND ----------

# DBTITLE 1,Define catalog, schema, table widgets
dbutils.widgets.text("CATALOG", "main", label="CATALOG")
dbutils.widgets.text("SCHEMA", "default", label="SCHEMA")
dbutils.widgets.text("INPUT_TABLE", "video_inferences", label="INPUT_TABLE")
dbutils.widgets.text("OUTPUT_TABLE", "video_events", label="OUTPUT_TABLE")
dbutils.widgets.text("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct", label="LLM_ENDPOINT")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
INPUT_TABLE = dbutils.widgets.get("INPUT_TABLE")
OUTPUT_TABLE = dbutils.widgets.get("OUTPUT_TABLE")
LLM_ENDPOINT = dbutils.widgets.get("LLM_ENDPOINT")

input_fqn = f"{CATALOG}.{SCHEMA}.{INPUT_TABLE}"
output_fqn = f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run AI Functions over the raw VLM outputs
# MAGIC
# MAGIC The JSON schema is defined inline below. To adapt this notebook for a different use case, change two things only:
# MAGIC 1. The instruction string passed to `ai_query` (currently asks for shoplifting incident metadata).
# MAGIC 2. The JSON schema body (currently `shoplifting_incident_metadata`).
# MAGIC
# MAGIC Everything else — the table names, the LLM endpoint, the `from_json` parsing — is generic.

# COMMAND ----------

# DBTITLE 1,Structure prose into Delta rows
sql = f"""
CREATE OR REPLACE TABLE {output_fqn} AS
SELECT
  *,
  parsed_result.timestamp AS timestamp,
  parsed_result.time_range AS time_range,
  parsed_result.location AS location,
  parsed_result.camera_view AS camera_view,
  parsed_result.suspects AS suspects,
  parsed_result.incident_confirmed AS incident_confirmed,
  parsed_result.witnesses_or_staff_present AS witnesses_or_staff_present,
  parsed_result.response_observed AS response_observed,
  parsed_result.notable_behaviors AS notable_behaviors,
  parsed_result.video_quality_notes AS video_quality_notes,
  parsed_result.contextual_notes AS contextual_notes
FROM (
  SELECT *,
    from_json(
      ai_query(
        '{LLM_ENDPOINT}',
        CONCAT(
          "Using the following description of some CCTV in a store, please extract the relevant information related to the incident as metadata: ",
          generated_text
        ),
        responseFormat => '{{
          "type": "json_schema",
          "json_schema": {{
            "name": "shoplifting_incident_metadata",
            "schema": {{
              "type": "object",
              "properties": {{
                "timestamp": {{ "type": "string", "format": "date-time" }},
                "time_range": {{ "type": "string", "description": "Time range in the video when the incident occurs (e.g., 03:12 - 03:48)" }},
                "location": {{ "type": "string", "description": "General store area (e.g., cosmetics aisle, electronics section)" }},
                "camera_view": {{ "type": "string", "description": "Camera perspective or label if visible (e.g., CAM3, front entrance)" }},
                "suspects": {{
                  "type": "array",
                  "items": {{
                    "type": "object",
                    "properties": {{
                      "person_id": {{ "type": "string", "description": "Identifier or label for tracking this individual (e.g., Person 1)" }},
                      "appearance": {{ "type": "string", "description": "Clothing and distinguishing features" }},
                      "actions": {{
                        "type": "array",
                        "items": {{ "type": "string" }},
                        "description": "Sequence of actions that indicate suspicious or shoplifting behavior"
                      }},
                      "item_taken": {{ "type": "string", "description": "Description of the item suspected to be stolen (if identifiable)" }},
                      "concealment_method": {{ "type": "string", "description": "How the item was hidden (e.g., in bag, under jacket)" }},
                      "exit_behavior": {{ "type": "string", "description": "How the suspect exited (e.g., ran out, walked past checkout)" }}
                    }}
                  }}
                }},
                "incident_confirmed": {{ "type": "boolean", "description": "True if the video clearly shows shoplifting behavior" }},
                "witnesses_or_staff_present": {{ "type": "boolean", "description": "True if other people (e.g., staff or customers) witnessed the event" }},
                "response_observed": {{ "type": "string", "description": "Any staff/security reaction captured in the footage" }},
                "notable_behaviors": {{
                  "type": "array",
                  "items": {{ "type": "string" }},
                  "description": "Additional suspicious behaviors observed (e.g., pacing, watching staff)"
                }},
                "video_quality_notes": {{ "type": "string", "description": "Notes on visibility, resolution, obstructions, or clarity" }},
                "contextual_notes": {{ "type": "string", "description": "Other useful context (e.g., group coordination, distraction tactics, time of day)" }}
              }},
              "strict": true
            }}
          }}
        }}'
      ),
      schema_of_json('{{
        "timestamp": "string",
        "time_range": "string",
        "location": "string",
        "camera_view": "string",
        "suspects": "array<struct<person_id:string,appearance:string,actions:array<string>,item_taken:string,concealment_method:string,exit_behavior:string>>",
        "incident_confirmed": "boolean",
        "witnesses_or_staff_present": "boolean",
        "response_observed": "string",
        "notable_behaviors": "array<string>",
        "video_quality_notes": "string",
        "contextual_notes": "string"
      }}')
    ) AS parsed_result
  FROM {input_fqn}
)
"""

spark.sql(sql)
print(f"Wrote structured events to {output_fqn}.")

# COMMAND ----------

# DBTITLE 1,Preview structured output
display(spark.table(output_fqn).limit(20))
