#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-hb-gcp-test-project}"
REGION="us-central1"
SERVICE_NAME="healthcare-companion-agent"

# Validate Terraform Infrastructure as Code configuration
terraform init -backend=false
terraform validate

# Deploy ADK agent service to Cloud Run with Secret Manager binding
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --ingress internal \
  --set-secrets="GEMINI_API_KEY=gemini-api-key:latest"