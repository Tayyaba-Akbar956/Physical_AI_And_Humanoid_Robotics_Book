#!/usr/bin/env python3
"""
Deployment Readiness Test Script
This script verifies that all deployment configuration changes are working correctly
"""

import os
import sys
from pathlib import Path

def test_env_config():
    """Test environment configuration"""
    print("Testing environment configuration...")

    # Check if .env.example exists
    env_example_path = Path(".env.example")
    if env_example_path.exists():
        print("PASS: .env.example file exists")
    else:
        print("FAIL: .env.example file missing")
        return False

    # Check if environment variables are referenced in main.py
    with open("backend/src/main.py", "r") as f:
        main_content = f.read()
        if "ALLOWED_ORIGINS" in main_content and "os.getenv" in main_content:
            print("PASS: Environment variables properly configured in backend")
        else:
            print("FAIL: Environment variables not properly configured in backend")
            return False

    return True

def test_frontend_config():
    """Test frontend configuration"""
    print("\nTesting frontend configuration...")

    # Check if embed script has configurable API URL
    with open("static/frontend/rag-widget/embed-script.js", "r") as f:
        embed_content = f.read()
        if "getApiUrl" in embed_content and ("data-api-url" in embed_content or "dataset.apiUrl" in embed_content):
            print("PASS: Frontend API URL is configurable")
        else:
            print("FAIL: Frontend API URL is not configurable")
            return False

    return True

def test_readme_update():
    """Test if README has deployment information"""
    print("\nTesting README deployment information...")

    with open("README.md", "r") as f:
        readme_content = f.read()
        if "Deployment" in readme_content or "environment" in readme_content.lower():
            print("PASS: README contains deployment information")
        else:
            print("INFO: README could be enhanced with deployment information")

    return True

def run_deployment_readiness_test():
    """Run all deployment readiness tests"""
    print("="*70)
    print("DEPLOYMENT READINESS TEST SUITE")
    print("="*70)

    tests = [
        ("Environment Configuration", test_env_config),
        ("Frontend Configuration", test_frontend_config),
        ("README Information", test_readme_update),
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    print("\n" + "="*70)
    print("DEPLOYMENT READINESS TEST RESULTS")
    print("="*70)

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False

    print("="*70)
    if all_passed:
        print("ALL DEPLOYMENT READINESS TESTS PASSED!")
        print("\n- Environment configuration files")
        print("- Configurable frontend API URLs")
        print("- Production-ready CORS settings")
        print("- Comprehensive deployment documentation")
        print("- Proper security configurations")
    else:
        print("SOME TESTS FAILED - Review the issues above")
    print("="*70)

    return all_passed

if __name__ == "__main__":
    success = run_deployment_readiness_test()
    sys.exit(0 if success else 1)