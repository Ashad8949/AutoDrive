# ── Log Analytics Workspace ──────────────────────────────────────────
# Free tier: 5 GB/day ingestion. Stores logs from App Service + App Insights.
resource "azurerm_log_analytics_workspace" "main" {
  name                = "autodrive-logs"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30  # minimum — keeps costs at zero on free tier
  tags                = { project = "autodrive" }
}

# ── Application Insights ─────────────────────────────────────────────
# Tracks requests, failures, response times, custom metrics.
# Free tier: 5 GB/month — more than enough for a demo project.
resource "azurerm_application_insights" "main" {
  name                = "autodrive-insights"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = var.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = { project = "autodrive" }
}

# ── Alert: High error rate ───────────────────────────────────────────
# Fires when server-side exceptions exceed 10 in 5 minutes.
resource "azurerm_monitor_metric_alert" "high_errors" {
  name                = "autodrive-high-errors"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [azurerm_application_insights.main.id]
  severity            = 2  # Warning
  frequency           = "PT5M"
  window_size         = "PT15M"
  description         = "Server exception rate is elevated"

  criteria {
    metric_namespace = "microsoft.insights/components"
    metric_name      = "exceptions/server"
    aggregation      = "Count"
    operator         = "GreaterThan"
    threshold        = 10
  }

  action {
    action_group_id = azurerm_monitor_action_group.email.id
  }

  tags = { project = "autodrive" }
}

# ── Alert: Slow response time ────────────────────────────────────────
resource "azurerm_monitor_metric_alert" "slow_response" {
  name                = "autodrive-slow-response"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [azurerm_application_insights.main.id]
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"
  description         = "Average server response time exceeds 3 seconds"

  criteria {
    metric_namespace = "microsoft.insights/components"
    metric_name      = "requests/duration"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 3000  # milliseconds
  }

  action {
    action_group_id = azurerm_monitor_action_group.email.id
  }

  tags = { project = "autodrive" }
}

# ── Alert: App Service CPU > 80% ────────────────────────────────────
resource "azurerm_monitor_metric_alert" "high_cpu" {
  name                = "autodrive-high-cpu"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [azurerm_service_plan.main.id]
  severity            = 1  # Error
  frequency           = "PT1M"
  window_size         = "PT5M"
  description         = "App Service CPU exceeds 80%"

  criteria {
    metric_namespace = "Microsoft.Web/serverfarms"
    metric_name      = "CpuPercentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.email.id
  }

  tags = { project = "autodrive" }
}

# ── Action Group (email notifications) ──────────────────────────────
resource "azurerm_monitor_action_group" "email" {
  name                = "autodrive-alerts"
  resource_group_name = data.azurerm_resource_group.main.name
  short_name          = "autodrive"

  email_receiver {
    name                    = "admin"
    email_address           = var.alert_email
    use_common_alert_schema = true
  }

  tags = { project = "autodrive" }
}
