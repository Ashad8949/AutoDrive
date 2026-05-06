# ── User-assigned Managed Identity ──────────────────────────────────
# Created before the web app, so its principal_id is known at plan time.
# The App Service uses this identity to read Key Vault secrets.
resource "azurerm_user_assigned_identity" "chatbot" {
  name                = "autodrive-identity"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = var.location
  tags                = { project = "autodrive" }
}

# ── Key Vault ────────────────────────────────────────────────────────
data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                       = "autodrive-kv-${substr(data.azurerm_client_config.current.subscription_id, 0, 8)}"
  resource_group_name        = data.azurerm_resource_group.main.name
  location                   = var.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  tags                       = { project = "autodrive" }
}

# ── Give Terraform permission to write secrets ───────────────────────
resource "azurerm_key_vault_access_policy" "terraform" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = ["Get", "Set", "List", "Delete", "Purge"]
}

# ── Give the managed identity read access to secrets ─────────────────
resource "azurerm_key_vault_access_policy" "app_service" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.chatbot.principal_id  # known at plan time

  secret_permissions = ["Get", "List"]
}

# ── Secrets ──────────────────────────────────────────────────────────
resource "azurerm_key_vault_secret" "groq_api_key" {
  name         = "groq-api-key"
  value        = var.groq_api_key
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "azure_openai_key" {
  name         = "azure-openai-key"
  value        = var.azure_openai_key
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault_access_policy.terraform]
}
