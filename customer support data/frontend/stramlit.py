"""
Compatibility file for requested name `stramlit.py`.
Use: streamlit run frontend/stramlit.py
"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("streamlit.py")), run_name="__main__")
