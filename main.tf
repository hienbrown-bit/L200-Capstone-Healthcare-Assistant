terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = "hb-gcp-test-project"
}

variable "region" {
  description = "Google Cloud deployment region"
  type        = string
  default     = "us-central1"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Pillar 5: Secure Secret Management via Google Secret Manager
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"

  replication {
    auto {}
  }
}

# Pillar 5: Infrastructure as Code for Cloud Run v2 Deployment
resource "google_cloud_run_v2_service" "healthcare_companion" {
  name     = "healthcare-companion-agent"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  template {
    containers {
      image = "gcr.io/${var.project_id}/healthcare-companion:latest"

      ports {
        container_port = 5000
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}