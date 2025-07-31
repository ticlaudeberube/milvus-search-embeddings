#!/usr/bin/env python3
"""Test all document loaders to ensure they work correctly."""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from core import MilvusUtils


class LoaderTester:
    """Test runner for document loaders."""
    
    def __init__(self):
        self.results: Dict[str, Tuple[bool, str]] = {}
        self.client = None
        
    def setup(self) -> bool:
        """Setup test environment."""
        try:
            # Test Milvus connection
            self.client = MilvusUtils.get_client()
            print("[OK] Milvus client connection successful")
            return True
        except Exception as e:
            print(f"[FAIL] Failed to connect to Milvus: {e}")
            return False
    
    def test_download_docs(self) -> Tuple[bool, str]:
        """Test download_milvus_docs.py"""
        try:
            # Run as subprocess to avoid import issues
            result = subprocess.run(
                [sys.executable, "document-loaders/download_milvus_docs.py"],
                capture_output=True, text=True, timeout=300
            )
            
            # Check if docs were downloaded
            docs_path = Path("document-loaders/milvus_docs")
            if docs_path.exists():
                return True, "Milvus docs downloaded successfully"
            else:
                return False, f"Download failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_milvus_docs_ollama(self) -> Tuple[bool, str]:
        """Test load_milvus_docs_ollama.py"""
        try:
            # Check if docs exist first
            docs_path = Path("document-loaders/milvus_docs/en")
            if not docs_path.exists():
                return False, "Milvus docs not found. Run download first."
            
            # Run as subprocess
            result = subprocess.run(
                [sys.executable, "document-loaders/load_milvus_docs_ollama.py"],
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                return True, "Milvus docs loaded with Ollama embeddings"
            else:
                return False, f"Failed: {result.stderr}"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_milvus_docs_hf(self) -> Tuple[bool, str]:
        """Test load_milvus_docs_hf.py"""
        try:
            # Check if docs exist first
            docs_path = Path("document-loaders/milvus_docs/en")
            if not docs_path.exists():
                return False, "Milvus docs not found. Run download first."
            
            # Run as subprocess
            result = subprocess.run(
                [sys.executable, "document-loaders/load_milvus_docs_hf.py"],
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                return True, "Milvus docs loaded with HuggingFace embeddings"
            else:
                return False, f"Failed: {result.stderr}"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
            
    def test_various_docs_ollama(self) -> Tuple[bool, str]:
        """Test load-various-docs-scatterplot.py (load only)"""
        try:
            # Create a test script that only runs the load function
            test_script = '''
import asyncio
import sys
sys.path.insert(0, ".")
from load_various_docs_scatterplot import load
asyncio.run(load())
print("Load completed successfully")
'''
            
            # Write temporary test script
            with open("document-loaders/test_load_only.py", "w") as f:
                f.write(test_script)
            
            # Run the test script
            result = subprocess.run(
                [sys.executable, "test_load_only.py"],
                capture_output=True, text=True, timeout=600, cwd="document-loaders"
            )
            
            # Clean up
            Path("document-loaders/test_load_only.py").unlink(missing_ok=True)
            
            if "Load completed successfully" in result.stdout:
                return True, "Various docs loaded with Ollama embeddings"
            else:
                return False, f"Failed: {result.stderr or 'Load not completed'}"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
        
    def run_all_tests(self) -> None:
        """Run all loader tests."""
        print("Starting Document Loader Tests\n")
        
        if not self.setup():
            print("Setup failed. Cannot proceed with tests.")
            return
        
        tests = [
            ("Download Milvus Docs", self.test_download_docs),
            ("Milvus Docs (Ollama)", self.test_milvus_docs_ollama),
            ("Milvus Docs (HuggingFace)", self.test_milvus_docs_hf),
            ("Various Docs (Ollama)", self.test_various_docs_ollama),
            ("Various Docs (HuggingFace)", self.test_various_docs_hf),
        ]
        
        for test_name, test_func in tests:
            print(f"Testing: {test_name}")
            try:
                success, message = test_func()
                status = "[OK]" if success else "[FAIL]"
                print(f"  {status} {message}")
                self.results[test_name] = (success, message)
            except Exception as e:
                print(f"  [FAIL] Unexpected error: {str(e)}")
                self.results[test_name] = (False, f"Unexpected error: {str(e)}")
            print()
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary."""
        print("=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for success, _ in self.results.values() if success)
        total = len(self.results)
        
        for test_name, (success, message) in self.results.items():
            status = "PASS" if success else "FAIL"
            print(f"{test_name:<30} {status}")
            if not success:
                print(f"    -> {message}")
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("All tests passed!")
        else:
            print("Some tests failed. Check the details above.")


def main():
    """Main test runner."""
    tester = LoaderTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()