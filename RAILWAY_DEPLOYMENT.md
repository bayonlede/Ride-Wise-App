# Railway Deployment Guide

## Fix for "Error creating build plan with Railpack"

Railway now uses **Dockerfile** instead of Railpack. Follow these steps:

---

## Deploy API (Backend)

1. **New Project** → Deploy from GitHub → Select your repo
2. **Root Directory**: Leave **empty** (project root) - required for model & data access
3. **Build**: Railway will use `Dockerfile.api` automatically
4. **Environment Variables**: Add `PORT` (Railway sets this automatically)
5. **Deploy** → Copy the generated URL (e.g. `https://your-api.up.railway.app`)

---

## Deploy Frontend (Dashboard UI)

1. **New Service** in the same project → Deploy from same GitHub repo
2. **Root Directory**: Set to `frontend`
3. **Environment Variables**:
   - `NEXT_PUBLIC_API_URL` = `https://your-api.up.railway.app` (from step above)
   - `PORT` - Railway sets automatically
4. **Deploy**

---

## Summary

| Service | Root Directory | Config |
|---------|----------------|--------|
| API | *(empty)* | Uses `Dockerfile.api` |
| Frontend | `frontend` | Uses `Dockerfile` |

**Important**: API must use project root (empty) so the build includes `model/` and `data/` folders.
