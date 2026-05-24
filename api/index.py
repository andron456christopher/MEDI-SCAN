from app import app
import sys
import os

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Vercel expects the app to be exported as 'app'
export = app
