<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

# Ray Dashboard and Metrics Integration on Databricks

## Business Problem
When running distributed applications on Ray, organizations need visibility into cluster performance, resource utilization, and application metrics. This solution demonstrates how to:

- Set up and configure the Ray Dashboard on Databricks
- Monitor Ray cluster metrics and performance
- Integrate with monitoring tools for real-time insights
- Track and visualize distributed application metrics

This visibility is crucial for optimizing resource usage, troubleshooting performance issues, and ensuring efficient operation of Ray-based applications.

There are 2 approaches detailed in this folder:
- Recommended - `setup_monitoring.sh` shell script with accompanying `prometheus-metrics-with-ray-dashboard` notebook showing how to run the script on a cluster (either All Purpose or Job)where a Ray cluster is already running. **As of September 2025, this is the recommended approach.**
- Deprecated - `init_ray_prometheus_grafana.sh` utilized (legacy) cluster init scripts, this approach has limited support for recent Ray on Spark features and Unity Catalog. 


## Authors
- <tj@databricks.com>
