# ── Reference the Azure OpenAI resource you already created ─────────
# This is a DATA SOURCE — it reads the existing resource without recreating it.
# No charges. No drift. Terraform just reads the endpoint for outputs.
data "azurerm_cognitive_account" "openai" {
  name                = var.openai_resource_name
  resource_group_name = var.openai_resource_group
}
