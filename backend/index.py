# Vercel entry point for the backend API
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path to handle relative imports
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import the main app from src.main
from src.main import app

# Add Mangum handler for Vercel serverless compatibility
from mangum import Mangum

# Disable lifespan handling for Vercel serverless functions
handler = Mangum(app, lifespan="off")

# For Vercel serverless functions
def handler_func(event, context):
    return handler(event, context)