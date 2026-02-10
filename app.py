"""
Agentic RAG Application
=======================
Run this file to start the Streamlit app.

    python app.py

Or directly:
    streamlit run UI/streamlit_app.py
"""
import subprocess
import sys


if __name__ == "__main__":
    print("=" * 60)
    print("  Starting Agentic RAG System")
    print("  → http://localhost:8501")
    print("=" * 60)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "UI/streamlit_app.py",
         "--server.port", "8501", "--server.runOnSave", "true"],
    )
