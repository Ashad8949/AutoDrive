output "chatbot_url" {
  description = "Production URL"
  value       = "https://${azurerm_linux_web_app.chatbot.default_hostname}"
}

output "staging_url" {
  description = "Staging slot URL (for pre-deployment testing)"
  value       = "https://${azurerm_linux_web_app_slot.staging.default_hostname}"
}

output "app_insights_connection_string" {
  description = "Application Insights connection string — add to .env for local tracing"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
}

output "openai_endpoint" {
  description = "Azure OpenAI endpoint"
  value       = data.azurerm_cognitive_account.openai.endpoint
}
