# Add Frontend Service to Railway – Step-by-Step Guide

Follow these steps to deploy the dashboard UI as a separate service alongside your API.

---

## Prerequisites

- Your **API** is already deployed at a URL like `https://ride-wise-app-production.up.railway.app` (or similar)
- You have access to the Railway project

---

## Step 1: Add a New Service

1. Open your Railway project: https://railway.app/dashboard
2. Click **"+ New"** (or **"Add Service"**)
3. Choose **"GitHub Repo"**
4. Select your repository: **Ride-Wise-App** (or your repo name)
5. Click **"Deploy Now"** (or **"Add Service"**)

---

## Step 2: Configure the Frontend Service

1. Open the new service.
2. Go to **Settings** (gear icon).
3. Under **Source**:
   - **Root Directory**: Set to `frontend`
   - **Watch Paths**: Leave default or set to `frontend/**`

4. Under **Build**:
   - Railway should auto-detect the Dockerfile
   - If not, ensure **Builder** is `DOCKERFILE` and **Dockerfile Path** is `Dockerfile`

5. Under **Environment Variables** (Variables tab):
   - Click **"+ New Variable"**
   - **Name**: `NEXT_PUBLIC_API_URL` (or `API_URL`)
   - **Value**: `https://YOUR-API-URL.up.railway.app`  
     (Replace with your actual API URL. No trailing slash.)

   Example:
   ```
   NEXT_PUBLIC_API_URL=https://ride-wise-app-production.up.railway.app
   ```
   
   **Note:** The app fetches this at runtime via `/api/config`, so it works even if the variable wasn't set at build time. Add it, then redeploy.

---

## Step 3: Generate a Domain

1. Go to the **Settings** tab of your frontend service.
2. Go to **Networking** → **Public Networking**.
3. Click **"Generate Domain"**.
4. Railway will create a URL like `https://your-frontend-name.up.railway.app`.

---

## Step 4: Deploy

1. Push the changes above or trigger a **Redeploy** on the service.
2. Wait for the build to complete.
3. Open the generated domain URL.

---

## Quick Reference

| Setting | Value |
|--------|--------|
| **Root Directory** | `frontend` |
| **Builder** | DOCKERFILE |
| **NEXT_PUBLIC_API_URL** | `https://your-api-url.up.railway.app` |

---

## Troubleshooting

**"API Offline"** or **"Failed to fetch"** when clicking Predict:
- Add `NEXT_PUBLIC_API_URL` or `API_URL` in the frontend service's **Variables** tab.
- Value must be your full API URL, e.g. `https://ride-wise-app-production.up.railway.app` (no trailing slash).
- **Redeploy** so the new variable is applied.

**CORS errors**:
- The API should already allow all origins (`allow_origins=["*"]`). If issues persist, check the API logs.

**Build fails / "context canceled"**:
- Ensure **Root Directory** is `frontend` in the service Settings.
- **Redeploy** – "context canceled" is often a transient timeout; try again.
- Verify no other `railway.toml` is overriding (the API has one at project root; the frontend uses `frontend/railway.toml`).
