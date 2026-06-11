"""
config.py - File paths, constants and configuration for the FYP Dashboard.
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from your local .env file
load_dotenv()

# Data directory and file paths
DATA_DIR      = Path("./data")
BASELINE_PATH = DATA_DIR / "baseline_results.csv"
CLUSTERS_PATH = DATA_DIR / "clusters.csv"
SESSIONS_PATH = DATA_DIR / "sessions_raw.jsonl"
REPORT_PATH   = DATA_DIR / "daily_report.md"
GEOIP_DB_PATH = DATA_DIR / "GeoLite2-Country.mmdb"
USERS_PATH    = DATA_DIR / "users.json"

# Safely pull credentials from the environment (with secure fallbacks)
admin_user = os.getenv("ADMIN_USER", "admin")
admin_pass = os.getenv("ADMIN_PASS", "default_secure_password")
super_pass = os.getenv("SUPERVISOR_PASS", "default_secure_password")

# Default accounts, only used if users.json doesn't exist yet
DEFAULT_USERS = {
    admin_user: {"hash": hashlib.sha256(admin_pass.encode()).hexdigest(), "role": "admin"},
    "nashnush": {"hash": hashlib.sha256(super_pass.encode()).hexdigest(), "role": "viewer"},
    "bryant":   {"hash": hashlib.sha256(super_pass.encode()).hexdigest(), "role": "viewer"},
}