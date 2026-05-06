# ── App Service Plan ─────────────────────────────────────────────────
# S1 Standard tier required for: autoscaling + deployment slots.
# Cost: ~$55/month. Switch back to B1 (~$13) after the demo.
# B1 = "B1", S1 = "S1"
resource "azurerm_service_plan" "main" {
  name                = "autodrive-plan"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = var.location
  os_type             = "Linux"
  sku_name            = var.app_service_sku  # set "S1" for demo, "B1" normally
  tags                = { project = "autodrive" }
}

# ── Chatbot Web App ──────────────────────────────────────────────────
resource "azurerm_linux_web_app" "chatbot" {
  name                = "autodrive-chatbot"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = var.location
  service_plan_id     = azurerm_service_plan.main.id
  https_only          = true
  tags                = { project = "autodrive" }

  # User-assigned Managed Identity — used to read Key Vault secrets
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.chatbot.id]
  }

  site_config {
    always_on        = var.app_service_sku != "B1"  # B1 doesn't support always_on
    health_check_path = "/health"                    # Azure auto-restarts unhealthy instances
    http2_enabled    = true

    application_stack {
      docker_image_name   = var.chatbot_image
      docker_registry_url = "https://ghcr.io"
    }
  }

  app_settings = {
    # ── LLM ─────────────────────────────────────────────────────────
    GROQ_API_KEY  = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault.main.vault_uri}secrets/groq-api-key/)"
    GROQ_MODEL    = "llama-3.3-70b-versatile"

    # ── Azure OpenAI (embeddings fallback) ──────────────────────────
    AZURE_OPENAI_ENDPOINT             = data.azurerm_cognitive_account.openai.endpoint
    AZURE_OPENAI_KEY                  = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault.main.vault_uri}secrets/azure-openai-key/)"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"

    # ── App Insights telemetry ───────────────────────────────────────
    APPLICATIONINSIGHTS_CONNECTION_STRING = azurerm_application_insights.main.connection_string
    ApplicationInsightsAgent_EXTENSION_VERSION = "~3"

    # ── Runtime ─────────────────────────────────────────────────────
    WEBSITES_PORT   = "8002"
    PYTHONPATH      = "/app"
    ALLOWED_ORIGINS = "*"

    # Force container pull when image tag changes
    DOCKER_ENABLE_CI = "true"
  }

  logs {
    http_logs {
      file_system {
        retention_in_days = 7
        retention_in_mb   = 35
      }
    }
    application_logs {
      file_system_level = "Information"
    }
  }

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# ── Deployment Slot: staging ─────────────────────────────────────────
# CI/CD deploys here first, runs health check, then swaps to production.
# Requires Standard tier (S1+).
resource "azurerm_linux_web_app_slot" "staging" {
  name           = "staging"
  app_service_id = azurerm_linux_web_app.chatbot.id
  tags           = { project = "autodrive" }

  # Reuse the same user-assigned identity as production
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.chatbot.id]
  }

  site_config {
    health_check_path = "/health"
    http2_enabled     = true

    application_stack {
      docker_image_name   = var.chatbot_image
      docker_registry_url = "https://ghcr.io"
    }
  }

  app_settings = {
    GROQ_API_KEY    = "@Microsoft.KeyVault(SecretUri=${azurerm_key_vault.main.vault_uri}secrets/groq-api-key/)"
    GROQ_MODEL      = "llama-3.3-70b-versatile"
    WEBSITES_PORT   = "8002"
    PYTHONPATH      = "/app"
    SLOT_NAME       = "staging"
    APPLICATIONINSIGHTS_CONNECTION_STRING = azurerm_application_insights.main.connection_string
    ApplicationInsightsAgent_EXTENSION_VERSION = "~3"
  }
}

# ── Autoscale rules ──────────────────────────────────────────────────
# Scale out (add instances) when CPU > 70% for 5 minutes.
# Scale in  (remove instances) when CPU < 30% for 10 minutes.
# Requires Standard tier (S1+). Ignored silently on B1.
resource "azurerm_monitor_autoscale_setting" "chatbot" {
  name                = "autodrive-autoscale"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = var.location
  target_resource_id  = azurerm_service_plan.main.id
  tags                = { project = "autodrive" }

  profile {
    name = "default"

    capacity {
      default = 1
      minimum = 1
      maximum = 3  # keep max low to control student credit spend
    }

    # Scale OUT: CPU > 70% sustained for 5 minutes → add 1 instance
    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.main.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 70
      }
      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }
    }

    # Scale IN: CPU < 30% sustained for 10 minutes → remove 1 instance
    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.main.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT10M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 30
      }
      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT10M"
      }
    }

    # Scale OUT: HTTP queue > 10 requests → add instance immediately
    rule {
      metric_trigger {
        metric_name        = "HttpQueueLength"
        metric_resource_id = azurerm_service_plan.main.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 10
      }
      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT3M"
      }
    }
  }

  notification {
    email {
      send_to_subscription_administrator = false
      custom_emails                       = [var.alert_email]
    }
  }
}
