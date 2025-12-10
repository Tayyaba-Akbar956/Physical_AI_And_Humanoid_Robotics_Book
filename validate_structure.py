#!/usr/bin/env python3
"""
Code Structure Validation for RAG Chatbot Implementation
This script validates that all required files and structures exist
"""

import os
import sys
from pathlib import Path

def check_directory_structure():
    """Check that all required directories exist"""
    print("[DIR] Checking directory structure...")

    expected_directories = [
        "backend/src",
        "backend/src/api",
        "backend/src/services",
        "backend/src/models",
        "backend/src/db",
        "backend/src/middleware",
        "frontend",
        "frontend/rag-widget",
        "specs",
        "specs/1-rag-chatbot",
        "history",
        "history/prompts"
    ]

    all_exist = True
    for directory in expected_directories:
        full_path = os.path.join(os.getcwd(), "..", directory)
        if os.path.isdir(full_path):
            print(f"  [OK] {directory}/")
        else:
            print(f"  [MISSING] {directory}/ - MISSING")
            all_exist = False

    return all_exist


def check_backend_files():
    """Check that all required backend files exist"""
    print("\n[FILE] Checking backend files...")

    expected_files = [
        "backend/src/main.py",
        "backend/src/api/chat_endpoints.py",
        "backend/src/api/text_selection_endpoints.py",
        "backend/src/api/conversation_endpoints.py",
        "backend/src/api/module_context_endpoints.py",
        "backend/src/api/system_endpoints.py",
        "backend/src/services/rag_agent.py",
        "backend/src/services/session_manager.py",
        "backend/src/services/semantic_search.py",
        "backend/src/services/conversational_context_manager.py",
        "backend/src/models/chat_session.py",
        "backend/src/models/message.py",
        "backend/src/models/student.py",
        "backend/src/models/textbook_content.py",
        "backend/src/models/selected_text.py",
        "backend/src/middleware/error_handling.py",
        "backend/src/middleware/logging.py",
        "backend/requirements.txt",
        "backend/test_core_implementation.py"
    ]

    all_exist = True
    for file in expected_files:
        full_path = os.path.join(os.getcwd(), "..", file)
        if os.path.isfile(full_path):
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file} - MISSING")
            all_exist = False

    return all_exist


def check_frontend_files():
    """Check that all required frontend files exist"""
    print("\n[FILE] Checking frontend files...")

    expected_files = [
        "frontend/rag-widget/chat-widget.js",
        "frontend/rag-widget/text-selector.js",
        "frontend/rag-widget/styles.css",
        "frontend/rag-widget/embed-script.js"
    ]

    all_exist = True
    for file in expected_files:
        full_path = os.path.join(os.getcwd(), "..", file)
        if os.path.isfile(full_path):
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file} - MISSING")
            all_exist = False

    return all_exist


def check_spec_files():
    """Check that all required spec files exist"""
    print("\n[SPEC] Checking specification files...")

    expected_files = [
        "specs/1-rag-chatbot/spec.md",
        "specs/1-rag-chatbot/plan.md",
        "specs/1-rag-chatbot/tasks.md",
        "specs/1-rag-chatbot/data-model.md",
        "specs/1-rag-chatbot/research.md",
        "specs/1-rag-chatbot/quickstart.md",
        "specs/1-rag-chatbot/contracts/api-contracts.md"
    ]

    all_exist = True
    for file in expected_files:
        full_path = os.path.join(os.getcwd(), "..", file)
        if os.path.isfile(full_path):
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file} - MISSING")
            all_exist = False

    return all_exist


def check_completion_status():
    """Check that the tasks have been marked as completed"""
    print("\n[TASK] Checking completion status...")

    tasks_file = os.path.join(os.getcwd(), "..", "specs/1-rag-chatbot/tasks.md")

    if not os.path.isfile(tasks_file):
        print("  [MISSING] specs/1-rag-chatbot/tasks.md - MISSING")
        return False

    with open(tasks_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count completed vs incomplete tasks
    completed_tasks = content.count('[X]')
    incomplete_tasks = content.count('[ ]')

    print(f"  [INFO] Completed tasks: {completed_tasks}")
    print(f"  [INFO] Incomplete tasks: {incomplete_tasks}")

    if incomplete_tasks == 0:
        print("  [SUCCESS] All tasks are marked as completed!")
        return True
    else:
        print(f"  [WARNING] {incomplete_tasks} tasks still need to be completed")
        return False


def validate_file_syntax(file_path):
    """Validate Python file syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        compile(source, file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"    [ERROR] Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"    [ERROR] Error reading {file_path}: {e}")
        return False


def check_python_syntax():
    """Check Python file syntax"""
    print("\n[SYNTAX] Checking Python file syntax...")

    python_files = [
        "backend/src/main.py",
        "backend/src/services/rag_agent.py",
        "backend/src/services/session_manager.py",
        "backend/src/services/semantic_search.py",
        "backend/src/services/conversational_context_manager.py",
        "backend/src/api/chat_endpoints.py",
        "backend/src/api/conversation_endpoints.py",
        "backend/src/api/module_context_endpoints.py",
        "backend/src/api/system_endpoints.py",
        "backend/src/middleware/error_handling.py",
        "backend/src/middleware/logging.py"
    ]

    all_valid = True
    for file in python_files:
        full_path = os.path.join(os.getcwd(), "..", file)
        if os.path.isfile(full_path):
            if validate_file_syntax(full_path):
                print(f"  [OK] {file}")
            else:
                all_valid = False
        else:
            print(f"  [MISSING] {file} - MISSING")
            all_valid = False

    return all_valid


def run_validation():
    """Run all validation checks"""
    print("="*70)
    print("RAG CHATBOT IMPLEMENTATION STRUCTURE VALIDATION")
    print("="*70)

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Backend Files", check_backend_files),
        ("Frontend Files", check_frontend_files),
        ("Specification Files", check_spec_files),
        ("Completion Status", check_completion_status),
        ("Python Syntax", check_python_syntax),
    ]

    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))

    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print("="*70)
    if all_passed:
        print("ALL VALIDATIONS PASSED - Implementation structure is complete!")
        print("\nTo run the application:")
        print("   1. Install requirements: pip install -r backend/requirements.txt")
        print("   2. Create .env file with required environment variables:")
        print("      - GEMINI_API_KEY")
        print("      - QDRANT_URL, QDRANT_API_KEY")
        print("      - NEON_DB_URL")
        print("   3. Run the application: cd backend && uvicorn src.main:app --reload --port 8000")
        print("\nSee README.md for complete setup and usage instructions")
    else:
        print("SOME VALIDATIONS FAILED - Implementation structure has issues")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    # Change to the project root directory
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    success = run_validation()
    sys.exit(0 if success else 1)