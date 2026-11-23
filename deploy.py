"""Programmatic Infrastructure-as-Code deployment script for Cloud Run."""
import subprocess

PROJECT_ID = "hb-gcp-test-project"
REGION = "us-central1"
SERVICE_NAME = "healthcare-companion-agent"


def deploy_to_cloud_run() -> None:
    """Provisions and deploys the ADK Healthcare Companion to Cloud Run."""
    cmd = [
        "gcloud",
        "run",
        "deploy",
        SERVICE_NAME,
        "--source",
        ".",
        "--project",
        PROJECT_ID,
        "--region",
        REGION,
        "--ingress",
        "internal",
        "--set-secrets",
        "GEMINI_API_KEY=gemini-api-key:latest",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    deploy_to_cloud_run()