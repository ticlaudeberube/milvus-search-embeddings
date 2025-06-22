import sys
import os
import platform
import subprocess

def check_package(package_name):
    """Check if a package is installed and get its version"""
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "unknown")
        return f"{package_name} version: {version}"
    except ImportError:
        return f"{package_name} not found"

def check_command(command):
    """Check if a command is available and get its version"""
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "Command not found"

# System information
print("=== SYSTEM INFORMATION ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# Check required packages
print("\n=== REQUIRED PACKAGES ===")
required_packages = [
    "streamlit", 
    "langchain", 
    "langchain_core",
    "langchain_community",
    "langchain_milvus", 
    "langchain_huggingface",
    "langchain_ollama"
]

for package in required_packages:
    print(check_package(package))

# Check Milvus connection
print("\n=== MILVUS CONNECTION ===")
print(f"MILVUS_HOST: {os.environ.get('MILVUS_HOST', 'Not set')}")
print(f"MILVUS_PORT: {os.environ.get('MILVUS_PORT', 'Not set')}")

# Check Ollama
print("\n=== OLLAMA STATUS ===")
print(f"OLLAMA_BASE_URL: {os.environ.get('OLLAMA_BASE_URL', 'Not set')}")
print("Ollama status: " + check_command(["curl", "-s", "http://localhost:11434/api/version"]))

# Environment variables
print("\n=== ENVIRONMENT VARIABLES ===")
for key, value in sorted(os.environ.items()):
    if any(prefix in key for prefix in ["CONDA", "PYTHON", "MILVUS", "OLLAMA", "MODEL"]):
        print(f"{key}: {value}")

if __name__ == "__main__":
    print("\nVerification complete. Check for any errors above.")