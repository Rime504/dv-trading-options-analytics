"""
Root Streamlit entry point for Streamlit Cloud deployment.
Set Main file path to: streamlit_app.py
"""
import sys
import os

# Add src/ to Python path so all local modules resolve correctly
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now execute dashboard as if running from src/
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dashboard",
    os.path.join(src_path, "dashboard.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
