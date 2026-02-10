"""
RideWise Dashboard Runner
Starts the Streamlit dashboard
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    print("=" * 60)
    print("RideWise Churn Prediction Dashboard")
    print("=" * 60)
    print("\nStarting Streamlit dashboard...")
    print("Dashboard URL: http://localhost:8501")
    print("\n[!] Make sure the API is running first!")
    print("    Run: python run_api.py")
    print("\n" + "=" * 60)
    
    # Get the path to the streamlit app
    app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "app",
        "streamlit_app.py"
    )
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port", "8501",
        "--server.headless", "true"
    ])
