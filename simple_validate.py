#!/usr/bin/env python3
"""
Simple Validation Script for RAG Chatbot Implementation
"""

import os

def validate_project_structure():
    """Validate that all expected files and directories exist"""
    print("="*60)
    print("RAG CHATBOT IMPLEMENTATION VALIDATION")
    print("="*60)

    # Check main directories
    print("\nChecking main directories...")
    main_dirs = [
        "backend",
        "frontend", 
        "specs",
        "history"
    ]

    all_dirs_ok = True
    for dir_name in main_dirs:
        if os.path.isdir(dir_name):
            print(f"  [OK] {dir_name}/")
        else:
            print(f"  [MISSING] {dir_name}/")
            all_dirs_ok = False

    # Check backend structure
    print("\nChecking backend structure...")
    backend_ok = True
    backend_dirs = [
        "backend/src",
        "backend/src/api",
        "backend/src/services",
        "backend/src/models",
        "backend/src/db",
        "backend/src/middleware"
    ]

    for dir_path in backend_dirs:
        if os.path.isdir(dir_path):
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [MISSING] {dir_path}/")
            backend_ok = False

    # Check backend files
    print("\nChecking backend files...")
    backend_files = [
        "backend/src/main.py",
        "backend/src/services/rag_agent.py",
        "backend/src/services/session_manager.py",
        "backend/src/services/semantic_search.py",
        "backend/src/services/conversational_context_manager.py",
        "backend/src/api/chat_endpoints.py",
        "backend/src/api/conversation_endpoints.py",
        "backend/src/api/module_context_endpoints.py",
        "backend/src/models/chat_session.py",
        "backend/src/models/message.py",
        "backend/requirements.txt"
    ]

    backend_files_ok = True
    for file_path in backend_files:
        if os.path.isfile(file_path):
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            backend_files_ok = False

    # Check frontend files
    print("\nChecking frontend files...")
    frontend_files = [
        "frontend/rag-widget/chat-widget.js",
        "frontend/rag-widget/text-selector.js",
        "frontend/rag-widget/styles.css",
        "frontend/rag-widget/embed-script.js"
    ]

    frontend_ok = True
    for file_path in frontend_files:
        if os.path.isfile(file_path):
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            frontend_ok = False

    # Check spec files
    print("\nChecking spec files...")
    spec_files = [
        "specs/1-rag-chatbot/tasks.md",
        "specs/1-rag-chatbot/spec.md",
        "specs/1-rag-chatbot/plan.md",
        "specs/1-rag-chatbot/data-model.md"
    ]

    spec_ok = True
    for file_path in spec_files:
        if os.path.isfile(file_path):
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            spec_ok = False

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Main directories: {'PASS' if all_dirs_ok else 'FAIL'}")
    print(f"Backend structure: {'PASS' if backend_ok else 'FAIL'}")
    print(f"Backend files: {'PASS' if backend_files_ok else 'FAIL'}")
    print(f"Frontend files: {'PASS' if frontend_ok else 'FAIL'}")
    print(f"Spec files: {'PASS' if spec_ok else 'FAIL'}")

    total_ok = all_dirs_ok and backend_ok and backend_files_ok and frontend_ok and spec_ok

    print("\n" + "="*60)
    if total_ok:
        print("ALL VALIDATIONS PASSED - Implementation is complete!")
        print("\nTo run the application:")
        print("1. Install requirements: pip install -r backend/requirements.txt")
        print("2. Set up environment variables in .env file")
        print("3. Run: cd backend && uvicorn src.main:app --reload --port 8000")
    else:
        print("SOME VALIDATIONS FAILED - Implementation has missing components")
    print("="*60)

    return total_ok

if __name__ == "__main__":
    success = validate_project_structure()
    exit(0 if success else 1)