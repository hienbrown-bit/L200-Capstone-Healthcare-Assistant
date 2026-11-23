"""Programmatic Infrastructure-as-Code deployment using ADK CLI and Terraform."""
import subprocess

PROJECT_ID = "hb-gcp-test-project"
REGION = "us-central1"
SERVICE_NAME = "healthcare-companion-agent"


def deploy_infrastructure() -> None:
    """Provisions cloud resources via Terraform IaC and deploys via ADK CLI."""
    subprocess.run(["terraform", "-chdir=terraform", "init"], check=True)
    subprocess.run(["terraform", "-chdir=terraform", "apply", "-auto-approve"], check=True)
    subprocess.run(
        [
            "adk",
            "deploy",
            "cloud_run",
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--service_name={SERVICE_NAME}",
            ".",
        ],
        check=True,
    )


if __name__ == "__main__":
    deploy_infrastructure()