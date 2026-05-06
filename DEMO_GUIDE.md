# AutoDrive AI — Professor Demo Guide
## Open these 5 browser tabs BEFORE your presentation starts

| Tab | URL | Purpose |
|---|---|---|
| 1 | https://autodrive-chatbot.azurewebsites.net | Live chatbot |
| 2 | https://github.com/Ashad8949/AutoDrive/actions | CI/CD pipeline |
| 3 | portal.azure.com → autodrive-chatbot | App Service |
| 4 | portal.azure.com → autodrive-insights → Live Metrics | Real-time monitoring |
| 5 | https://github.com/Ashad8949/AutoDrive/tree/master/terraform | IaC code |

---

## Demo Flow (5 minutes, high impact)

### 1. Live Chatbot (60 seconds)
- Open Tab 1 → ask: **"SUVs under ₹20 lakh"**
- Point out: ₹ prices, Indian cars, real-time token streaming
- Ask: **"Book a test drive for Hyundai Creta"** → show booking widget popup
- Say: *"This is answering from Mahesh's live PostgreSQL database"*

### 2. Application Insights Live (45 seconds)
- Open Tab 4 (Live Metrics)
- Ask another question on the chatbot
- Point at the spike in the live graph
- Say: *"Every request is tracked in real-time — latency, errors, request rate"*

### 3. CI/CD Pipeline (60 seconds)
- Open Tab 2 → show the latest green run
- Click into it → show 4 stages: Test → Build → Staging → Production
- Say: *"Every push to master automatically runs tests, builds Docker, deploys to staging, runs health checks, then swaps to production"*

### 4. Autoscale Email (30 seconds)
- Show the email Azure Monitor sent when CPU exceeded 70%
- Say: *"This is real proof the system auto-scaled in production. Azure automatically added an instance when load increased"*

### 5. Azure Portal (60 seconds)
- Open Tab 3 → Deployment Slots → show staging + production slots
- Click Scale Out → show the 3 autoscale rules
- Open Key Vault → show secrets stored (values hidden)
- Say: *"All of this was created by 3 Terraform files — Infrastructure as Code"*

### 6. Show Terraform Code (30 seconds)
- Open Tab 5 → open container_apps.tf
- Point at autoscale rules defined as code
- Say: *"Nothing was clicked in the portal. Everything is version-controlled"*

---

## If Professor Asks to Trigger a Deployment Live

```bash
# Make a visible change, push, watch the 4-stage pipeline
git commit --allow-empty -m "demo: show live CI/CD to professor"
git push origin master
# Then open GitHub Actions → watch it run live
```

## If Professor Asks to Force Refresh Inventory

```bash
curl -X POST https://autodrive-chatbot.azurewebsites.net/inventory/refresh
# Returns: {"status":"refreshed","cars":12}
# Then ask bot about a new car Mahesh just added
```

## Key Numbers to Remember
- **Autoscale:** Min 1, Max 3 instances. Triggers: CPU >70% or HTTP queue >10
- **Health check:** Azure polls /health every 60s, restarts if down 2 min
- **Cache TTL:** 24 hours (inventory refreshed daily automatically)
- **CI/CD time:** ~8-10 minutes end to end (tests + build + staging + swap)
- **First token latency:** <1 second (Groq LLaMA 3.3-70B)
- **Cost:** ~$55/month on S1 (demo), ~$13/month on B1 (normal)
- **LLM cost:** $0 (Groq free tier)
- **Azure resources:** 13 (all created by Terraform)
