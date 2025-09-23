#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
cd "/"

# Version Identifiers and Ports
PROMETHEUS_VERSION="3.0.1"
GRAFANA_VERSION="9.2.4"
RAY_DASHBOARD_PORT="9999"

# PART 0: Helper functions

determine_cloud() {
  # Determine cloud provider for current cluster
  local cloud_conf
  cloud_conf=$(grep -oP '(?<=databricks.instance.metadata.cloudProvider = ")[^"]*' /databricks/common/conf/deploy.conf) || {
    echo "[ERROR] Could not determine cloud provider."
    exit 1
  }
  echo "${cloud_conf}"
}

determine_org_id() {
  # Determine organization id for current workspace
  local org_id
  org_id=$(grep -oP '/organization\K[^?]+' /databricks/hive/conf/hive-site.xml) || {
    echo "[ERROR] Could not determine Org ID."
    exit 1
  }
  echo "${org_id}"
}

get_data_plane_url() {
  # Determine control plane URL from cloud provider
  local cloud="$1"
  local org_id="$2"
  if [[ "$cloud" == "Azure" ]]; then
    # Azure workspaces have CP URL in conf file
    cpurl_base=$(grep 'databricks.manager.defaultControlPlaneClientUrl' /databricks/common/conf/deploy.conf | awk -F'=' '{print $2}' | tr -d ' ') || {
      echo "[ERROR] Could not determine control plane URL for Azure workspace."
      exit 1
    }
    # Add dataplane component to URL before org ID
    dpurl=$(echo $cpurl_base | sed 's/\([0-9]\)/dp-\1/')
    echo "${dpurl}"
  elif [[ "$cloud" == "AWS" ]]; then
    # AWS workspaces have standardized URLs
    dpurl="dbc-dp-${org_id}.cloud.databricks.com"
    echo "${dpurl}"
  else
    echo "[ERROR] Cannot determine data plane URL from cloud input."
    exit 1
  fi
}

# PART 1: Initialize environment variables
BASE_DIR="/local_disk0/tmp"

CLOUD=$(determine_cloud)
echo "[INFO] Using Cloud=$CLOUD"

ORG_ID=$(determine_org_id)
echo "[INFO] Using Org ID=$ORG_ID"

# Env variable must be set in notebook
CLUSTER_ID=$CLUSTER_ID
echo "[INFO] Using Cluster Id=$CLUSTER_ID"

DPURL=$(get_data_plane_url $CLOUD $ORG_ID)
echo "[INFO] Using Data Plane URL=$DPURL"

# Determine Ray Dashboard port access to Grafana UI. Grafana uses port 3000 by default
IFRAME_HOST=https://"${DPURL//\"/}"/driver-proxy/o/"${ORG_ID//\"/}"/"${CLUSTER_ID//\"/}"/3000
echo "[INFO] Using ray_grafana_iframe_host=$IFRAME_HOST"
# Env variable must be set for Ray dashboard init
export RAY_GRAFANA_IFRAME_HOST=$IFRAME_HOST

PROM_HOST=https://"${DPURL//\"/}"/driver-proxy/o/"${ORG_ID//\"/}"/"${CLUSTER_ID//\"/}"/9090

# Replace Grafana ini file
cat <<EOL > ${BASE_DIR}/ray/session_latest/metrics/grafana/grafana.ini
[server]
domain = ${DPURL//\"/}
root_url = /driver-proxy/o/${ORG_ID//\"/}/${CLUSTER_ID//\"/}/3000/
serve_from_sub_path = false

[security]
allow_embedding = true

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer

[paths]
provisioning = $BASE_DIR/ray/session_latest/metrics/grafana/provisioning
EOL

# PART 2: Get and start Prometheus and Grafana

# Get Prometheus and Grafana
sudo wget -q https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz -P ${BASE_DIR}/
tar xfz ${BASE_DIR}/prometheus-*.tar.gz -C ${BASE_DIR}

sudo wget -q https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz -P ${BASE_DIR}/
tar xfz ${BASE_DIR}/grafana-*.tar.gz -C ${BASE_DIR}

# Start Prometheus with options
nohup ${BASE_DIR}/prometheus*amd64/prometheus --config.file=${BASE_DIR}/ray/session_latest/metrics/prometheus/prometheus.yml > ${BASE_DIR}/prometheus.log 2>&1 &

# Wait until Prometheus is initialized
sleep 30

# Start Grafana with options
nohup ${BASE_DIR}/grafana-${GRAFANA_VERSION}/bin/grafana-server --config=${BASE_DIR}/ray/session_latest/metrics/grafana/grafana.ini --homepath=${BASE_DIR}/grafana-${GRAFANA_VERSION} web > ${BASE_DIR}/grafana.log 2>&1 &


echo "[INFO] Setup completed! Check logs in ${BASE_DIR}/ for details."
echo "[INFO] Grafana is running on port 3000. Access via: ${IFRAME_HOST}"
