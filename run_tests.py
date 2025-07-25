#!/usr/bin/env python3
"""
Test runner script for the utils folder tests
"""
import subprocess
import sys
import os

def run_tests():
    """Run pytest with coverage for utils tests"""
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("Running unit tests for utils folder...")
    
    # Run unit tests (excluding integration tests)
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_utils.py",
        "-v",
        "--tb=short",
        "-m", "not integration",
        "--cov=utils",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Unit tests passed!")
        
        # Optionally run integration tests if Milvus is available
        print("\nTo run integration tests (requires running Milvus):")
        print("pytest tests/test_integration.py -m integration -v")
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with return code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest pytest-cov")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())