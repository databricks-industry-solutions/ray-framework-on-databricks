<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

# Deploying Ray Apps with Databricks Asset Bundles (DABs)

## Business Problem
Developers building production data processing pipelines with Ray need standardized CICD/DevOps interfaces to deploy them to one or more Databricks workspace or clusters. This is exactly what Databricks Asset Bundles (DABs) provides: an infrastructure-as-code (IaC) approach to managing your Databricks projects, including for Ray workloads.

<img src=https://docs.databricks.com/aws/en/assets/images/bundles-cicd-53be5f4860e8ebcedc2702f870290cda.png width="600px">

## Reference Architecture

The examples in this repository demonstrate the following solutions:
1. **Submit Ray code to Global Ray Cluster from CLI/IDE**
    * This is a drop-in replacement to [Ray Jobs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html), as Databricks clusters do not allow you to address the head node IP directly for security reasons.

2. **Define a multi-step DABs with Ray and non-Ray tasks**


For further documentation and customization opportunities with DABs, please see: 
* Docs: [What are Databricks Asset Bundles?](https://docs.databricks.com/aws/en/dev-tools/bundles/)
* Examples: [databricks/bundle-examples](https://github.com/databricks/bundle-examples/tree/main/knowledge_base)

## Authors
<tj@databricks.com>