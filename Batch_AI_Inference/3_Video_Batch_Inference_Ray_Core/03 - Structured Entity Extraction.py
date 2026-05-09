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

# DBTITLE 1,Structure prose into Delta rows (multi-class assessment)
# Two improvements over the original schema:
#   1. assessment is a 5-level enum, not a binary boolean. Recovers VLM prose
#      that says "suspected" / "appears to be" / "likely" — these were being
#      classified as incident_confirmed=false in the binary version.
#   2. schema_of_json is replaced with explicit struct DDL so column types are
#      correct (BOOLEAN / DOUBLE / ARRAY<STRUCT<...>>) instead of all STRING.
RESULT_STRUCT = (
    "STRUCT<"
    "timestamp:STRING, "
    "time_range:STRING, "
    "location:STRING, "
    "camera_view:STRING, "
    "suspects:ARRAY<STRUCT<"
    "person_id:STRING, appearance:STRING, actions:ARRAY<STRING>, "
    "item_taken:STRING, concealment_method:STRING, exit_behavior:STRING"
    ">>, "
    "assessment:STRING, "
    "confidence:DOUBLE, "
    "witnesses_or_staff_present:BOOLEAN, "
    "response_observed:STRING, "
    "notable_behaviors:ARRAY<STRING>, "
    "video_quality_notes:STRING, "
    "contextual_notes:STRING"
    ">"
)

INSTRUCTION = (
    "You are a loss-prevention analyst classifying a CCTV-footage analysis. "
    "Read the description below and extract structured incident metadata. "
    "Use this scale for the assessment field: "
    "CONFIRMED = analyst explicitly states shoplifting occurred with clear evidence "
    "(e.g. concealment shown, exit without payment). "
    "LIKELY = analyst describes suspicious behaviour consistent with shoplifting but "
    "stops short of formal confirmation (uses words like suspected, appears to be, "
    "likely). "
    "POSSIBLE = a behaviour that could be suspicious but is also consistent with "
    "normal customer behaviour. "
    "NORMAL = analyst describes normal customer behaviour with no theft indicators. "
    "INSUFFICIENT_EVIDENCE = description too sparse, ambiguous, or video quality too "
    "poor to make any assessment. "
    "Confidence is 0.0-1.0 reflecting the analyst certainty in their description. "
    "Description: "
)

sql = f"""
CREATE OR REPLACE TABLE {output_fqn} AS
SELECT
  *,
  parsed_result.timestamp AS timestamp,
  parsed_result.time_range AS time_range,
  parsed_result.location AS location,
  parsed_result.camera_view AS camera_view,
  parsed_result.suspects AS suspects,
  parsed_result.assessment AS assessment,
  parsed_result.confidence AS confidence,
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
        CONCAT('{INSTRUCTION}', generated_text),
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
                "assessment": {{
                  "type": "string",
                  "enum": ["CONFIRMED", "LIKELY", "POSSIBLE", "NORMAL", "INSUFFICIENT_EVIDENCE"],
                  "description": "Multi-class incident assessment level"
                }},
                "confidence": {{ "type": "number", "minimum": 0, "maximum": 1, "description": "Analyst certainty (0.0 to 1.0)" }},
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
      '{RESULT_STRUCT}'
    ) AS parsed_result
  FROM {input_fqn}
)
"""

spark.sql(sql)
print(f"Wrote structured events to {output_fqn}.")

# COMMAND ----------

# DBTITLE 1,Preview structured output
display(spark.table(output_fqn).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Slide 9 — Trimmed schema reference (for talk screenshot)
# MAGIC
# MAGIC The full JSON schema above has 12 fields. For the DAIS 2026 talk slide 9 we want a
# MAGIC version that fits cleanly on a Card-Left layout at DM Mono ~14pt — about 18 lines of
# MAGIC SQL. The block below is the abbreviated visual reference. It is **not runnable** (the
# MAGIC `…` placeholders aren't valid JSON); it exists purely as a screenshot source.
# MAGIC
# MAGIC The production version that actually runs is the cell above. Use this one only for
# MAGIC the slide capture — dark IDE theme, crop tight to the SQL block.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT
# MAGIC   parsed.timestamp,
# MAGIC   parsed.location,
# MAGIC   parsed.incident_confirmed,
# MAGIC   parsed.suspects
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     from_json(
# MAGIC       ai_query(
# MAGIC         'databricks-meta-llama-3-3-70b-instruct',
# MAGIC         CONCAT('Extract incident metadata from: ', generated_text),
# MAGIC         responseFormat => '{
# MAGIC           "type": "json_schema",
# MAGIC           "json_schema": {
# MAGIC             "name": "shoplifting_incident_metadata",
# MAGIC             "schema": {
# MAGIC               "type": "object",
# MAGIC               "properties": {
# MAGIC                 "timestamp":          { "type": "string" },
# MAGIC                 "location":           { "type": "string" },
# MAGIC                 "incident_confirmed": { "type": "boolean" },
# MAGIC                 "suspects":           { "type": "array", "items": { ... } }
# MAGIC                 // ... 8 more fields in the full schema ...
# MAGIC               }
# MAGIC             }
# MAGIC           }
# MAGIC         }'
# MAGIC       ),
# MAGIC       'STRUCT<...>'
# MAGIC     ) AS parsed
# MAGIC   FROM samantha_wise.dais_2026.video_inferences_summit
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Slide 9 — Live demo query (run this on stage)
# MAGIC
# MAGIC Returns 5 rows of structured incident metadata from the events table for the
# MAGIC slide-9 live moment. Targets the same schema as the production cell above.
# MAGIC Run from the Databricks SQL editor (warehouse `Shared Endpoint`) for the demo.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   element_at(split(path, '/'), -1)  AS filename,
# MAGIC   assessment,
# MAGIC   confidence,
# MAGIC   location,
# MAGIC   time_range,
# MAGIC   suspects[0].person_id            AS person,
# MAGIC   suspects[0].item_taken           AS item_taken,
# MAGIC   suspects[0].concealment_method   AS concealment
# MAGIC FROM samantha_wise.dais_2026.video_events_summit
# MAGIC WHERE assessment IN ('CONFIRMED', 'LIKELY')
# MAGIC   AND location IS NOT NULL AND location != ''
# MAGIC   AND size(suspects) > 0
# MAGIC ORDER BY assessment, confidence DESC
# MAGIC LIMIT 5;
