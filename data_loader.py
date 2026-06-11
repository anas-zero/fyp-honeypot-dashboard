"""
data_loader.py - Cached data loading functions for the FYP Dashboard.

Reads CSV, JSONL and Markdown files from the data directory.
All functions use Streamlit caching for performance.
"""

import json
from typing import Optional

import pandas as pd
import streamlit as st

from config import BASELINE_PATH, CLUSTERS_PATH, SESSIONS_PATH, REPORT_PATH


@st.cache_data
def load_baseline() -> Optional[pd.DataFrame]:
    """Load the merged baseline results CSV (sessions + anomaly flags)."""
    if not BASELINE_PATH.exists():
        return None
    return pd.read_csv(BASELINE_PATH)


@st.cache_data
def load_clusters() -> Optional[pd.DataFrame]:
    """Load the cluster assignments CSV."""
    if not CLUSTERS_PATH.exists():
        return None
    return pd.read_csv(CLUSTERS_PATH)


@st.cache_data
def load_sessions() -> Optional[list]:
    """Load raw session data from JSONL (one JSON object per line).
    Returns a list of dicts, each representing a full session record
    including commands, timestamps and category labels."""
    if not SESSIONS_PATH.exists():
        return None
    records = []
    with open(SESSIONS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@st.cache_data
def load_report() -> Optional[str]:
    """Load the daily threat report (Markdown format)."""
    if not REPORT_PATH.exists():
        return None
    return REPORT_PATH.read_text(encoding="utf-8")


def missing_file_error(name: str):
    """Display a user-friendly error when a required data file is missing."""
    st.error(f"File not found: `{name}`. Please place it in `./data/`.")