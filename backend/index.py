# Vercel entry point for the backend API with minimal initialization
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path to handle relative imports
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    # Import the main app from src.main
    from src.main import app

    # Add Mangum handler for Vercel serverless compatibility
    from mangum import Mangum

    # Disable lifespan handling for Vercel serverless functions to avoid initialization issues
    handler = Mangum(app, lifespan="off")

    print("Backend initialized successfully")

except Exception as e:
    print(f"Error initializing backend: {e}")

    # Create a minimal app for health checks if main app fails to initialize
    from fastapi import FastAPI
    app = FastAPI(title="Physical AI RAG Backend - Fallback")

    @app.get("/")
    @app.get("/health")
    async def health():
        return {
            "status": "error",
            "message": f"Backend failed to initialize: {str(e)}",
            "required_env_vars": [
                "GEMINI_API_KEY",
                "QDRANT_URL",
                "QDRANT_API_KEY",
                "NEON_DB_URL"
            ]
        }

    from mangum import Mangum
    handler = Mangum(app, lifespan="off")

# For Vercel serverless functions
def handler_func(event, context):
    return handler(event, context)