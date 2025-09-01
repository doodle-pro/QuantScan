#!/usr/bin/env python3
"""
Q-MediScan Backend Runner
Development server startup script
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def main():
    """Run the Q-MediScan backend server"""
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("[DNA] Starting Q-MediScan Backend Server")
    print("=" * 50)
    print(f"[GLOBE] Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"[REFRESH] Reload: {reload}")
    print(f"📝 Log Level: {log_level}")
    print("=" * 50)
    print("[ROCKET] Server starting...")
    print()
    print("[CHART] API Documentation: http://localhost:8000/docs")
    print("[MICROSCOPE] Quantum Analysis: http://localhost:8000/predict")
    print("[BULB] Health Check: http://localhost:8000/")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()