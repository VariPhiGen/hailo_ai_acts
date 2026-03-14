#!/usr/bin/env bash
# Must run with bash (not sh) — uses string arrays

set -eo pipefail

# Path to target .env (default: project root .env)
ENV_FILE="${1:-./.env}"

# ---------------------------------------------------------------------------
# Template key=value pairs.
# Rule:
#   - Key missing from .env                      → APPEND template value
#   - Key exists, non-empty, no PLACEHOLDER      → PRESERVE (do nothing)
#   - Key exists, empty OR contains PLACEHOLDER  → OVERWRITE with template
# ---------------------------------------------------------------------------
TEMPLATE_LINES=(
  "SENSOR_ID=SENSOR_ID_PLACEHOLDER"
  "BROKER_PRIMARY=43.205.29.76:9092"
  "BROKER_FAILOVER_TIMEOUT=30"
  "RESET_THRESHOLD=3000"
  "SEND_ANALYTICS_PIPELINE=aianalytics"
  "SEND_EVENTS_PIPELINE=arresto-event-consume"
  "LOG_TOPIC=error_log"
  "AWS_PRIMARY_KEY=minio"
  "AWS_PRIMARY_SECRET=minioAspl"
  "AWS_PRIMARY_REGION=ap-south-1"
  "AWS_PRIMARY_BUCKET_NAME=ai-labelled"
  "AWS_PRIMARY_ORG_IMG_FN=violationoriginalimages/"
  "AWS_PRIMARY_VIDEO_FN=videoclips/"
  "AWS_PRIMARY_CGI_FN=cgisnapshots/"
  "AWS_PRIMARY_ENDPOINT=http://5200001-minio.arresto.io"
  "API_AWS_PRIMARY_KEY=minio"
  "API_AWS_PRIMARY_SECRET=minioAspl"
  "API_AWS_PRIMARY_REGION=ap-south-1"
  "API_AWS_PRIMARY_BUCKET_NAME=ai-labelled"
  "API_AWS_PRIMARY_ORG_IMG_FN=violationoriginalimages/"
  "API_AWS_PRIMARY_VIDEO_FN=videoclips/"
  "API_AWS_PRIMARY_CGI_FN=cgisnapshots/"
  "API_AWS_PRIMARY_ENDPOINT=http://5200001-minio.arresto.io"
  "API_POST_URL=https://5200001-api.arresto.io/api/client/1825/mfb/forms_data"
  "API_REQUEST_TIMEOUT=30"
  "API_MAX_RETRIES=3"
  "S3_FAILOVER_TIMEOUT=30"
  "S3_UPLOAD_RETRIES=3"
  "WS_SERVER_URL=ws://54.210.59.224:8000/ws"
  "LOG_LEVEL=INFO"
  "YOLOE_API_URL=http://yoloe.vgiskill.com/predict_prompt"
  "YOLOE_CONF=0.1"
)

touch "$ENV_FILE"

for entry in "${TEMPLATE_LINES[@]}"; do
  key="${entry%%=*}"
  default_value="${entry#*=}"

  if grep -qE "^${key}=" "$ENV_FILE"; then
    # Key exists — extract current value
    current_value="$(grep -E "^${key}=" "$ENV_FILE" | head -1 | cut -d'=' -f2-)"

    if [[ -n "$current_value" ]]; then
      # Non-empty value exists → always preserve it, regardless of what default says
      echo "  ✓  PRESERVED  ${key}=${current_value}"
    else
      # Key exists but is empty → fill with default
      sed -i.bak -E "s|^${key}=.*|${key}=${default_value}|" "$ENV_FILE"
      echo "  ↻  FILLED     ${key}  (was empty → ${default_value})"
    fi
  else
    # Key not found → append default value
    echo "${key}=${default_value}" >> "$ENV_FILE"
    echo "  +  ADDED      ${key}=${default_value}"
  fi
done

# Clean up sed backup files
rm -f "${ENV_FILE}.bak"

echo ""
echo "Done updating $ENV_FILE"
echo "  ✓  PRESERVED = kept your existing real value"
echo "  ↻  OVERWRITE  = was empty/placeholder, set to template default"
echo "  +  ADDED      = key was missing, appended"