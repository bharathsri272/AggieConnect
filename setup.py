"""Setup script for AggieConnect."""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def setup_environment():
    """Set up the development environment."""
    print("🎓 Setting up AggieConnect Development Environment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "models/embeddings",
        "models/checkpoints",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Copy environment file
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from .env.example")
        print("⚠️  Please update .env with your API keys and configuration")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update .env file with your OpenAI API key")
    print("2. Run: python src/data/processor.py (to process sample data)")
    print("3. Run: python src/models/embedding_model.py (to train embeddings)")
    print("4. Run: python src/rag/vector_store.py (to build vector store)")
    print("5. Run: streamlit run src/web/app.py (to start web interface)")
    print("6. Or run: python demo.py (for command-line demo)")
    
    return True


if __name__ == "__main__":
    setup_environment()
