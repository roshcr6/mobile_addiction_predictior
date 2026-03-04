"""
Shared pytest fixtures and configuration.
"""
import sys
from pathlib import Path

# Make sure the backend root (where 'app' lives) is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
