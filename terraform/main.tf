terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
  }
}

provider "azurerm" {
  features {}
}

# ── Use existing resource group (already created in portal) ─────────
data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}
