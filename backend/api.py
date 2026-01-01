# API entry point for Vercel deployment
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path to handle relative imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Change to the backend directory to handle relative imports properly
original_cwd = os.getcwd()
os.chdir(str(backend_path))

# Import the main app after adjusting the path
from src.main import app

# Restore the original working directory
os.chdir(original_cwd)

# Import Mangum for Vercel serverless deployment
from mangum import Mangum

# This file serves as the entry point for Vercel's Python runtime
# Vercel will look for a top-level app or handler object to handle requests
handler = Mangum(app)