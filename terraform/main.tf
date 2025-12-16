terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_service" "openbook_api" {
  name     = "openbook-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/openbook-api"
        resources {
          limits = {
            memory = "1Gi"
            cpu    = "1"
          }
        }
        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.openbook_api.name
  location = google_cloud_run_service.openbook_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_pubsub_topic" "retrain_trigger" {
  name = "model-retrain-trigger"
}

resource "google_storage_bucket" "data_lake" {
  name     = "${var.project_id}-data-lake"
  location = var.region
}

output "api_url" {
  value = google_cloud_run_service.openbook_api.status[0].url
}
