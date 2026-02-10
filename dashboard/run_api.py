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
    print("=" * 60)
    print("RideWise Churn Prediction API")
    print("=" * 60)
    print("\nStarting API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
