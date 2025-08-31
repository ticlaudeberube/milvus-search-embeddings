#!/usr/bin/env python3
"""Diagnose environment setup requirements for Milvus Search Embeddings project."""

import sys
import os
import subprocess

def check_python_version():
    """Check Python version requirement."""
    version = sys.version_info
    required = (3, 12)
    
    print("Python Version: {}.{}.{}".format(version.major, version.minor, version.micro))
    if version >= required:
        print("[PASS] Python version meets requirement (>=3.12)")
        return True
    else:
        print("[FAIL] Python version too old. Required: >={}.{}".format(required[0], required[1]))
        return False

def check_virtual_env():
    """Check if virtual environment is active."""
    venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if venv_active:
        print("[PASS] Virtual environment is active")
        print("   Path: {}".format(sys.prefix))
        return True
    else:
        print("[FAIL] Virtual environment not active")
        print("   Run: source venv/bin/activate")
        return False

def check_core_package():
    """Check if core package is installed."""
    # Check if core directory exists
    if not os.path.exists('core'):
        print("[FAIL] Core directory not found")
        return False
    
    # Check if __init__.py exists
    if not os.path.exists('core/__init__.py'):
        print("[FAIL] core/__init__.py not found")
        return False
    
    # Check Python path
    print("   Current directory: {}".format(os.getcwd()))
    print("   Python path: {}".format(sys.path[:3]))
    
    # Try importing pymilvus first
    try:
        import pymilvus
        print("   pymilvus available")
    except ImportError:
        print("[FAIL] pymilvus not installed")
        print("   Run: pip install pymilvus")
        return False
    
    # Try importing core package
    try:
        from core import MilvusUtils
        print("[PASS] Core package installed and importable")
        return True
    except ImportError as e:
        print("[FAIL] Core package import failed: {}".format(e))
        print("   Run: pip install -e .")
        return False

def check_env_file():
    """Check if .env file exists."""
    env_file = ".env"
    env_example = ".env.example"
    
    if os.path.exists(env_file):
        print("[PASS] .env file exists")
        return True
    elif os.path.exists(env_example):
        print("[WARN] .env file missing, but .env.example found")
        print("   Run: cp .env.example .env")
        return False
    else:
        print("[FAIL] No .env or .env.example file found")
        return False

def check_milvus_connection():
    """Check Milvus connection."""
    try:
        from core import get_client
        client = get_client()
        print("[PASS] Milvus connection successful")
        return True
    except Exception as e:
        print("[FAIL] Milvus connection failed: {}".format(e))
        print("   Start Milvus: docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest")
        return False

def check_ollama():
    """Check Ollama availability."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("[PASS] Ollama is available")
            return True
        else:
            print("[FAIL] Ollama command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[FAIL] Ollama not installed or not in PATH")
        print("   Install: https://ollama.ai/")
        return False

def main():
    """Run all diagnostic checks."""
    print("Diagnosing Milvus Search Embeddings Environment\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Core Package", check_core_package),
        ("Environment File", check_env_file),
        ("Milvus Connection", check_milvus_connection),
        ("Ollama", check_ollama),
    ]
    
    results = []
    for name, check_func in checks:
        print("\nChecking {}:".format(name))
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print("[ERROR] Error during {} check: {}".format(name, e))
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print("{} {}".format(status, name))
    
    print("\nScore: {}/{} checks passed".format(passed, total))
    
    if passed == total:
        print("Environment is ready!")
    else:
        print("Some issues need attention")

if __name__ == "__main__":
    main()