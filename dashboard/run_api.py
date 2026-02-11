"""
RideWise API Runner
Starts the FastAPI server for the churn prediction API
"""

import uvicorn
import sys
import os

# Add the dashboard directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 8000))
    is_production = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("PRODUCTION")
    
    print("=" * 60)
    print("RideWise Churn Prediction API")
    print("=" * 60)
    print(f"\nStarting API server on port {port}...")
    if not is_production:
        print(f"API Documentation: http://localhost:{port}/docs")
        print(f"Health Check: http://localhost:{port}/health")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=not is_production,
        log_level="info"
    )
