# RideWise – Rider Churn Prediction for Ride‑Hailing Platforms

RideWise is an end‑to‑end churn prediction solution for the ride‑hailing / ride‑sharing industry.  
It combines a **FastAPI** machine‑learning backend with a **Next.js** analytics dashboard to help product and CRM teams:

- Forecast which riders are most likely to churn
- Understand *why* using SHAP explainability
- Design targeted retention strategies based on risk level

---

## Key Features

- **Production‑ready REST API**
  - FastAPI backend serving predictions, SHAP explanations, feature importance, and sample riders
  - Health check endpoint for monitoring
- **Modern Analytics Dashboard**
  - Next.js 14 + Tailwind UI with a ridesharing‑themed background
  - Real‑time predictions, charts, and explainability panels
  - Sample rider profiles for demos and experimentation
- **Model Explainability**
  - SHAP values per feature for each prediction
  - Global feature importance ranking
- **Ride‑Hailing Domain Focus**
  - Features like recency, ride frequency, monetary value, surge exposure, loyalty tier, churn scores, active days, and ratings
  - Risk‑aware recommendations for retention playbooks
- **Cloud‑friendly**
  - First‑class Railway deployment for both API and dashboard
  - Docker‑based builds, suitable for most PaaS providers

---

