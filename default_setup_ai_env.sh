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
  "BROKER_PRIMARY=PLACEHOLDER_BROKER_PRIMARY"
  "BROKER_FAILOVER_TIMEOUT=30"
  "RESET_THRESHOLD=3000"
  "SEND_ANALYTICS_PIPELINE=PLACEHOLDER_SEND_ANALYTICS_PIPELINE"
  "SEND_EVENTS_PIPELINE=PLACEHOLDER_SEND_EVENTS_PIPELINE"
  "LOG_TOPIC=error_log"
  "AWS_PRIMARY_KEY=PLACEHOLDER_AWS_PRIMARY_KEY"
  "AWS_PRIMARY_SECRET=PLACEHOLDER_AWS_PRIMARY_SECRET"
  "AWS_PRIMARY_REGION=ap-south-1"
  "AWS_PRIMARY_BUCKET_NAME=PLACEHOLDER_AWS_PRIMARY_BUCKET_NAME"
  "AWS_PRIMARY_ORG_IMG_FN=violationoriginalimages/"
  "AWS_PRIMARY_VIDEO_FN=PLACEHOLDER_AWS_PRIMARY_VIDEO_FN"
  "AWS_PRIMARY_CGI_FN=PLACEHOLDER_AWS_PRIMARY_CGI_FN"
  "AWS_PRIMARY_ENDPOINT=PLACEHOLDER_AWS_PRIMARY_ENDPOINT"
  "API_AWS_PRIMARY_KEY=minio"
  "API_AWS_PRIMARY_SECRET=PLACEHOLDER_API_AWS_PRIMARY_SECRET"
  "API_AWS_PRIMARY_REGION=ap-south-1"
  "API_AWS_PRIMARY_BUCKET_NAME=PLACEHOLDER_API_AWS_PRIMARY_BUCKET_NAME"
  "API_AWS_PRIMARY_ORG_IMG_FN=violationoriginalimages/"
  "API_AWS_PRIMARY_VIDEO_FN=PLACEHOLDER_API_AWS_PRIMARY_VIDEO_FN"
  "API_AWS_PRIMARY_CGI_FN=PLACEHOLDER_API_AWS_PRIMARY_CGI_FN"
  "API_AWS_PRIMARY_ENDPOINT=http://5200001-minio.arresto.io"
  "API_POST_URL=PLACEHOLDER_API_POST_URL"
  "API_REQUEST_TIMEOUT=30"
  "API_MAX_RETRIES=3"
  "S3_FAILOVER_TIMEOUT=30"
  "S3_UPLOAD_RETRIES=3"
  "WS_SERVER_URL=PLACEHOLDER_WS_SERVER_URL"
  "LOG_LEVEL=INFO"
  "YOLOE_API_URL=http://yoloe.vgiskill.com/predict_prompt"
  "YOLOE_CONF=0.1"
)

touch "$ENV_FILE"

for entry in "${TEMPLATE_LINES[@]}"; do
  key="${entry%%=*}"
  template_value="${entry#*=}"

  if grep -qE "^${key}=" "$ENV_FILE"; then
    # Key exists — extract current value (everything after first '=')
    current_value="$(grep -E "^${key}=" "$ENV_FILE" | head -1 | cut -d'=' -f2-)"

    if [[ -z "$current_value" || "$current_value" == *"PLACEHOLDER"* ]]; then
      # Empty or placeholder → overwrite
      sed -i.bak -E "s|^${key}=.*|${key}=${template_value}|" "$ENV_FILE"
      echo "  ↻  OVERWRITE  ${key}  (was empty/placeholder → ${template_value})"
    else
      # Real value — preserve
      echo "  ✓  PRESERVED  ${key}=${current_value}"
    fi
  else
    # Missing — append
    echo "${key}=${template_value}" >> "$ENV_FILE"
    echo "  +  ADDED      ${key}=${template_value}"
  fi
done

# Clean up sed backup files
rm -f "${ENV_FILE}.bak"

echo ""
echo "Done updating $ENV_FILE"
echo "  ✓  PRESERVED = kept your existing real value"
echo "  ↻  OVERWRITE  = was empty/placeholder, set to template default"
echo "  +  ADDED      = key was missing, appended"