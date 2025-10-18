"""Quick script to run the FastAPI backend."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the FastAPI backend."""
    print("🎓 Starting AggieConnect API Server...")
    
    # Check if we're in the right directory
    if not Path("src/api/main.py").exists():
        print("❌ Error: Please run this script from the AggieConnect root directory")
        sys.exit(1)
    
    # Run FastAPI with uvicorn
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 API server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running API server: {e}")

if __name__ == "__main__":
    main()
