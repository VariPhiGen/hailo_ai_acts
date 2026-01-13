#!/usr/bin/env bash
set -euo pipefail

# Path to target .env (default: project root .env)
ENV_FILE="${1:-./.env}"

# Key/values to enforce (from env_file_template.txt lines 3-38)
declare -A KV=(
  [BROKER_PRIMARY]="43.205.29.76:9092"
  [BROKER_FAILOVER_TIMEOUT]="30"
  [RESET_THRESHOLD]="3000"
  [SEND_ANALYTICS_PIPELINE]="aianalytics"
  [SEND_EVENTS_PIPELINE]="arresto-event-consume"
  [LOG_TOPIC]="error_log"
  [AWS_PRIMARY_KEY]="minio"
  [AWS_PRIMARY_SECRET]="minioAspl"
  [AWS_PRIMARY_REGION]="ap-south-1"
  [AWS_PRIMARY_BUCKET_NAME]="ai-labelled"
  [AWS_PRIMARY_ORG_IMG_FN]="violationoriginalimages/"
  [AWS_PRIMARY_VIDEO_FN]="videoclips/"
  [AWS_PRIMARY_CGI_FN]="cgisnapshots/"
  [AWS_PRIMARY_ENDPOINT]="http://5200001-minio.arresto.io"
  [API_AWS_PRIMARY_KEY]="minio"
  [API_AWS_PRIMARY_SECRET]="minioAspl"
  [API_AWS_PRIMARY_REGION]="ap-south-1"
  [API_AWS_PRIMARY_BUCKET_NAME]="ai-labelled"
  [API_AWS_PRIMARY_ORG_IMG_FN]="violationoriginalimages/"
  [API_AWS_PRIMARY_VIDEO_FN]="videoclips/"
  [API_AWS_PRIMARY_CGI_FN]="cgisnapshots/"
  [API_AWS_PRIMARY_ENDPOINT]="http://5200001-minio.arresto.io"
  [API_POST_URL]="https://5200001-api.arresto.io/api/client/1825/mfb/forms_data"
  [API_REQUEST_TIMEOUT]="30"
  [API_MAX_RETRIES]="3"
  [S3_FAILOVER_TIMEOUT]="30"
  [S3_UPLOAD_RETRIES]="3"
  [WS_SERVER_URL]="ws://54.210.59.224:8000/ws"
  [LOG_LEVEL]="INFO"
)

touch "$ENV_FILE"

for key in "${!KV[@]}"; do
  value="${KV[$key]}"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    # Replace existing line
    sed -i.bak -E "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    # Append if missing
    echo "${key}=${value}" >> "$ENV_FILE"
  fi
done

echo "Updated $ENV_FILE with template values (existing keys replaced, others appended)."