"""
Entrypoint for LegalCheckAgent on Railway / Cloud hosting.
Navigates into legal-check-app and launches Streamlit.
"""
import os
import sys

port = os.environ.get("PORT", "8501")
print(f"[LegalCheckAgent] Starting Streamlit dashboard on port {port}...")
sys.stdout.flush()

app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal-check-app")
if os.path.exists(app_dir):
    os.chdir(app_dir)

cmd = [
    "streamlit",
    "run",
    "app.py",
    "--server.port",
    str(port),
    "--server.address",
    "0.0.0.0",
    "--server.headless",
    "true",
    "--browser.gatherUsageStats",
    "false",
]
os.execvp("streamlit", cmd)
