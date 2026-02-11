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

5. Under **Environment Variables**:
   - Click **"+ New Variable"**
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: `https://YOUR-API-URL.up.railway.app`  
     (Replace with your actual API URL. Do not add a trailing slash.)

   Example for API at `ride-wise-app-production.up.railway.app`:
   ```
   NEXT_PUBLIC_API_URL=https://ride-wise-app-production.up.railway.app
   ```
   
   **Tip:** If your API is in the same project, you can use a reference variable:
   `${{YOUR_API_SERVICE_NAME.RAILWAY_PUBLIC_DOMAIN}}` and prefix with `https://`

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

**"API Offline"** in the dashboard:
- Add `NEXT_PUBLIC_API_URL` and ensure it matches your API URL.
- Redeploy after adding the variable (build-time env vars require a rebuild).

**CORS errors**:
- The API should already allow all origins (`allow_origins=["*"]`). If issues persist, check the API logs.

**Build fails / "context canceled"**:
- Ensure **Root Directory** is `frontend` in the service Settings.
- **Redeploy** – "context canceled" is often a transient timeout; try again.
- Verify no other `railway.toml` is overriding (the API has one at project root; the frontend uses `frontend/railway.toml`).
