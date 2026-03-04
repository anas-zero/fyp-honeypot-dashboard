"""
geoip_lookup.py
───────────────
Resolves IP addresses to country names + approximate coordinates
using the offline MaxMind GeoLite2-Country database.

Usage:
    from geoip_lookup import enrich_with_geo
    df = enrich_with_geo(df, ip_column="src_ip", db_path="data/GeoLite2-Country.mmdb")

Requires:
    pip install geoip2
    Download GeoLite2-Country.mmdb from https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Country centroid coordinates (ISO alpha-2 → lat, lon)
# Covers the most commonly seen attacker source countries.
# Falls back to (0, 0) for unmapped codes.
# ──────────────────────────────────────────────
COUNTRY_CENTROIDS = {
    "US": (39.83, -98.58),
    "CN": (35.86, 104.20),
    "RU": (61.52, 105.32),
    "DE": (51.17, 10.45),
    "BR": (-14.24, -51.93),
    "IN": (20.59, 78.96),
    "KR": (35.91, 127.77),
    "JP": (36.20, 138.25),
    "VN": (14.06, 108.28),
    "NL": (52.13, 5.29),
    "FR": (46.23, 2.21),
    "GB": (55.38, -3.44),
    "UA": (48.38, 31.17),
    "TW": (23.70, 120.96),
    "ID": (-0.79, 113.92),
    "TH": (15.87, 100.99),
    "IR": (32.43, 53.69),
    "PK": (30.38, 69.35),
    "PH": (12.88, 121.77),
    "SG": (1.35, 103.82),
    "HK": (22.40, 114.11),
    "AR": (-38.42, -63.62),
    "MX": (23.63, -102.55),
    "CO": (4.57, -74.30),
    "EG": (26.82, 30.80),
    "ZA": (-30.56, 22.94),
    "NG": (9.08, 8.68),
    "PL": (51.92, 19.15),
    "RO": (45.94, 24.97),
    "IT": (41.87, 12.57),
    "ES": (40.46, -3.75),
    "CA": (56.13, -106.35),
    "AU": (-25.27, 133.78),
    "BD": (23.68, 90.36),
    "TR": (38.96, 35.24),
    "SE": (60.13, 18.64),
    "CZ": (49.82, 15.47),
    "BG": (42.73, 25.49),
    "MY": (4.21, 101.98),
    "CL": (-35.68, -71.54),
    "LT": (55.17, 23.88),
    "PA": (8.54, -80.78),
    "VE": (6.42, -66.59),
}


def enrich_with_geo(
    df: pd.DataFrame,
    ip_column: str = "src_ip",
    db_path: str = "data/GeoLite2-Country.mmdb",
) -> pd.DataFrame:
    """
    Add 'country', 'country_code', 'lat', 'lon' columns to a dataframe
    by resolving the IP column against the GeoLite2-Country database.

    Returns the dataframe with new columns. Rows where lookup fails
    get country='Unknown', code='--', and lat/lon of (0, 0).
    """
    db = Path(db_path)
    if not db.exists():
        logger.warning("GeoLite2 database not found at %s — returning empty geo columns", db_path)
        df["country"] = "Unknown"
        df["country_code"] = "--"
        df["lat"] = 0.0
        df["lon"] = 0.0
        return df

    try:
        import geoip2.database
    except ImportError:
        logger.error("geoip2 not installed — run: pip install geoip2")
        df["country"] = "Unknown"
        df["country_code"] = "--"
        df["lat"] = 0.0
        df["lon"] = 0.0
        return df

    # Build a cache so each unique IP is only looked up once
    unique_ips = df[ip_column].dropna().unique()
    ip_cache = {}

    with geoip2.database.Reader(str(db)) as reader:
        for ip in unique_ips:
            try:
                resp = reader.country(str(ip))
                code = resp.country.iso_code or "--"
                name = resp.country.name or "Unknown"
                lat, lon = COUNTRY_CENTROIDS.get(code, (0.0, 0.0))
                ip_cache[ip] = (name, code, lat, lon)
            except Exception:
                ip_cache[ip] = ("Unknown", "--", 0.0, 0.0)

    df["country"] = df[ip_column].map(lambda ip: ip_cache.get(ip, ("Unknown",))[0])
    df["country_code"] = df[ip_column].map(lambda ip: ip_cache.get(ip, ("", "--"))[1])
    df["lat"] = df[ip_column].map(lambda ip: ip_cache.get(ip, ("", "", 0.0))[2])
    df["lon"] = df[ip_column].map(lambda ip: ip_cache.get(ip, ("", "", "", 0.0))[3])

    return df
