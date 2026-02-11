# RideWise Frontend Dashboard

A modern, professional churn prediction dashboard built with Next.js 14, TypeScript, and Tailwind CSS. Fully integrated with the RideWise FastAPI backend for real-time predictions.

## Features

- **Real-time Churn Prediction** - Enter rider metrics and get instant ML predictions
- **SHAP Explanations** - Understand which factors drive churn risk
- **Interactive Visualizations** - Feature importance charts and risk gauges
- **Sample Profiles** - Pre-loaded rider profiles for quick testing
- **Actionable Recommendations** - Retention strategies based on risk level
- **Responsive Design** - Works seamlessly on desktop, tablet, and mobile

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React

## Getting Started

### Prerequisites

- Node.js 18+ installed
- RideWise API running (see `../dashboard` folder)

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp env.example .env.local

# Update API_URL in .env.local to point to your API
```

### Development

```bash
# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm run start
```

## Deployment on Railway

### Option 1: Using Nixpacks (Recommended)

1. Connect your GitHub repository to Railway
2. Set the root directory to `frontend`
3. Add environment variable: `API_URL=<your-api-url>`
4. Railway will auto-detect Next.js and deploy

### Option 2: Using Docker

1. Railway will auto-detect the Dockerfile
2. Set environment variables in Railway dashboard
3. Deploy

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_URL` | Backend API URL | Yes |
| `PORT` | Server port (Railway sets automatically) | No |
| `NEXT_PUBLIC_APP_URL` | Public URL for SEO | No |

## API Integration

The dashboard connects to these API endpoints:

- `GET /health` - Health check
- `POST /predict` - Get churn prediction
- `POST /explain` - Get SHAP explanations
- `GET /feature-importance` - Global feature importance
- `GET /sample-riders` - Sample rider profiles

## Project Structure

```
frontend/
├── src/
│   └── app/
│       ├── globals.css    # Global styles & Tailwind
│       ├── layout.tsx     # Root layout with metadata
│       └── page.tsx       # Main dashboard page
├── public/                # Static assets
├── railway.toml           # Railway configuration
├── Dockerfile             # Docker configuration
├── tailwind.config.ts     # Tailwind configuration
└── next.config.js         # Next.js configuration
```

## License

MIT
