variable "resource_group_name" {
  description = "Name for the new Azure resource group"
  type        = string
  default     = "autodrive-rg"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "Southeast Asia"
}

# ── Existing Azure OpenAI resource (already created in portal) ──────
variable "openai_resource_name" {
  description = "Name of the Azure OpenAI resource you already created in the portal"
  type        = string
}

variable "openai_resource_group" {
  description = "Resource group that contains the existing Azure OpenAI resource"
  type        = string
}

# ── Container image ─────────────────────────────────────────────────
variable "chatbot_image" {
  description = "Full image path pushed by GitHub Actions (ghcr.io/<org>/autodrive-chatbot:latest)"
  type        = string
  default     = "ghcr.io/your-github-username/autodrive-chatbot:latest"
}

# ── Secrets (passed via TF_VAR_ env vars — never commit these) ──────
variable "azure_openai_key" {
  description = "Azure OpenAI API key"
  type        = string
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq API key — stored in Key Vault, never in app settings directly"
  type        = string
  sensitive   = true
}

variable "alert_email" {
  description = "Email address for Azure Monitor alert notifications"
  type        = string
  default     = "ashad8949@gmail.com"
}

variable "app_service_sku" {
  description = "App Service Plan SKU. Use S1 for autoscale+slots demo, B1 for normal use"
  type        = string
  default     = "S1"
}
