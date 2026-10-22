"""
PolicyLens Vercel Serverless Entrypoint
Exposes the FastAPI 'app' instance for Vercel Python runtime.
"""
import sys
import os
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import FastAPI application
from server import app
