"""Quick script to run the Streamlit web interface."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit web interface."""
    print("🎓 Starting AggieConnect Web Interface...")
    
    # Check if we're in the right directory
    if not Path("src/web/app.py").exists():
        print("❌ Error: Please run this script from the AggieConnect root directory")
        sys.exit(1)
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/web/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running web interface: {e}")

if __name__ == "__main__":
    main()
