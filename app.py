"""
FYP Security Dashboard — Streamlit app
Reads data from ./data/ and renders a four-tab analytical dashboard.

Tabs:
  1. Overview   — KPI cards, sessions-over-time chart, daily report
  2. Anomalies  — filterable table, command chain, "why flagged?" panel
  3. Clusters   — cluster-size bar chart, drill-down per cluster
  4. Explain    — narrative bullets + template-based Q&A
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import streamlit as st

from geoip_lookup import enrich_with_geo

# ──────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────
DATA_DIR      = Path("./data")
BASELINE_PATH = DATA_DIR / "baseline_results.csv"
CLUSTERS_PATH = DATA_DIR / "clusters.csv"
SESSIONS_PATH = DATA_DIR / "sessions_raw.jsonl"
REPORT_PATH   = DATA_DIR / "daily_report.md"
GEOIP_DB_PATH = DATA_DIR / "GeoLite2-Country.mmdb"

# ──────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(page_title="SSH Honeypot Dashboard", layout="wide", page_icon="🛡️")

# ──────────────────────────────────────────────
# Data loaders  (cached; show friendly errors if missing)
# ──────────────────────────────────────────────

@st.cache_data
def load_baseline() -> Optional[pd.DataFrame]:
    if not BASELINE_PATH.exists():
        return None
    return pd.read_csv(BASELINE_PATH)


@st.cache_data
def load_clusters() -> Optional[pd.DataFrame]:
    if not CLUSTERS_PATH.exists():
        return None
    return pd.read_csv(CLUSTERS_PATH)


@st.cache_data
def load_sessions() -> Optional[list]:
    """Return a list of dicts, one per line of the JSONL file."""
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
    if not REPORT_PATH.exists():
        return None
    return REPORT_PATH.read_text(encoding="utf-8")


def missing_file_error(name: str):
    """Show a friendly error banner when a data file is absent."""
    st.error(f"⚠️  File not found: `{name}` — please place it in `./data/`.")


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def epoch_to_utc(ts: float) -> datetime:
    """Convert a Unix timestamp (float) to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def why_flagged(row: pd.Series) -> list:
    """
    Generate human-readable bullet points explaining why a session
    looks suspicious, based on its feature values.
    """
    bullets = []

    duration = row.get("duration_seconds", 0)
    if duration < 3:
        bullets.append(
            f"**Very short session** ({duration:.2f} s) — typical of automated probes "
            "that connect, attempt auth, and disconnect immediately."
        )

    if row.get("had_commands", 1) == 0 and duration >= 3:
        bullets.append(
            "**No commands executed** despite a session lasting several seconds "
            "(had_commands = 0) — common pattern in banner-grabbing scanners."
        )

    auth_fail = int(row.get("auth_fail_count", 0))
    if auth_fail > 0:
        bullets.append(
            f"**Authentication failure(s):** {auth_fail} failed attempt(s) recorded "
            "— consistent with brute-force or credential-stuffing behaviour."
        )

    if int(row.get("download_count", 0)) > 0:
        bullets.append(
            f"**File download detected** ({int(row['download_count'])} event(s)) "
            "— attacker may have retrieved a payload or exfiltrated data."
        )

    if int(row.get("priv_esc_count", 0)) > 0:
        bullets.append(
            f"**Privilege-escalation commands** present "
            f"({int(row['priv_esc_count'])} event(s)) — attempts to gain root/sudo access."
        )

    if int(row.get("sensitive_path_count", 0)) > 0:
        bullets.append(
            f"**Sensitive path access** detected ({int(row['sensitive_path_count'])} event(s)) "
            "— e.g. /etc/passwd, /etc/shadow, SSH keys."
        )

    cmds_per_sec = row.get("cmds_per_sec_x", 0)
    if cmds_per_sec > 1.0:
        bullets.append(
            f"**High command rate** ({cmds_per_sec:.2f} cmds/s) — "
            "scripted, non-interactive execution; commands run programmatically."
        )

    novelty = row.get("novelty_rate", 0)
    if novelty > 0.5:
        bullets.append(
            f"**High novelty rate** ({novelty:.2f}) — the majority of commands "
            "in this session were unseen in the baseline, suggesting novel tooling."
        )

    cat_switch = row.get("category_switch_rate", 0)
    if cat_switch > 0.3:
        bullets.append(
            f"**Frequent category switching** ({cat_switch:.2f}) — "
            "rapid alternation between recon, download, and escalation commands "
            "is a multi-stage attack signature."
        )

    recon = int(row.get("recon_count", 0))
    if recon >= 3:
        bullets.append(
            f"**Heavy reconnaissance** ({recon} commands) — "
            "extensive system enumeration (uname, cpuinfo, uptime, last, etc.)."
        )

    if not bullets:
        bullets.append(
            "Flagged as a **statistical outlier** by Isolation Forest — "
            "the combination of features falls outside the normal session distribution, "
            "even if no single feature is individually alarming."
        )

    return bullets


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
baseline_df  = load_baseline()
clusters_df  = load_clusters()
sessions_raw = load_sessions()
report_md    = load_report()

# Fast lookup: session_id → raw record
sessions_by_id = {}
if sessions_raw:
    sessions_by_id = {s["session_id"]: s for s in sessions_raw}

# Merged dataframe (baseline + cluster labels)
merged_df = None
if baseline_df is not None and clusters_df is not None:
    merged_df = baseline_df.merge(clusters_df, on="session_id", how="left")

# GeoIP enrichment (cached — runs once per data reload)
geo_df = None
if baseline_df is not None:
    geo_df = enrich_with_geo(
        baseline_df.copy(),
        ip_column="src_ip",
        db_path=str(GEOIP_DB_PATH),
    )

# ──────────────────────────────────────────────
# Page header
# ──────────────────────────────────────────────
st.title("🛡️ SSH Honeypot — Security Dashboard")

tab_overview, tab_anomalies, tab_clusters, tab_map, tab_explain = st.tabs(
    ["📊 Overview", "🚨 Anomalies", "🔵 Clusters", "🌍 Map", "💡 Explain"]
)

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab_overview:
    st.header("Overview")

    # ── KPI cards ──────────────────────────────
    if baseline_df is None:
        missing_file_error("baseline_results.csv")
    else:
        total_sessions    = len(baseline_df)
        unique_ips        = baseline_df["src_ip"].nunique()
        rule_flagged      = int(baseline_df["baseline_rule_flag"].sum())
        iforest_anomalies = int(baseline_df["anomaly_flag_iforest"].sum())
        lof_anomalies     = int(baseline_df["anomaly_flag_lof"].sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sessions",    total_sessions)
        c2.metric("Unique Source IPs", unique_ips)
        c3.metric("Rule-Flagged",      rule_flagged)
        c4.metric("IForest Anomalies", iforest_anomalies)
        c5.metric("LOF Anomalies",     lof_anomalies)

    st.divider()

    # ── Sessions over time (hourly) ─────────────
    st.subheader("Sessions over time (hourly, UTC)")

    if sessions_raw is None:
        missing_file_error("sessions_raw.jsonl")
    else:
        # Convert epoch timestamps → UTC datetimes, then floor to the hour
        utc_times = pd.Series(
            [epoch_to_utc(s["start_ts"]) for s in sessions_raw],
            name="start_utc",
        )
        hourly = (
            utc_times
            .dt.floor("h")
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(
            [t.strftime("%Y-%m-%d %H:%M") for t in hourly.index],
            hourly.values,
            color="#4C72B0",
            width=0.6,
        )
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Session count")
        ax.set_title("SSH session volume per hour")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Anomaly score distribution ──────────────
    st.subheader("Anomaly score distribution (Isolation Forest)")

    if baseline_df is None:
        missing_file_error("baseline_results.csv")
    else:
        scores = baseline_df["anomaly_score_iforest"].dropna()

        fig_dist, ax_dist = plt.subplots(figsize=(12, 3))
        ax_dist.hist(
            scores,
            bins=40,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )

        # Add a vertical line at the anomaly threshold
        threshold = scores[baseline_df["anomaly_flag_iforest"] == 1].min() if (baseline_df["anomaly_flag_iforest"] == 1).any() else None
        if threshold is not None:
            ax_dist.axvline(
                x=threshold,
                color="#D64545",
                linestyle="--",
                linewidth=1.5,
                label=f"Anomaly threshold ({threshold:.3f})",
            )
            ax_dist.legend(fontsize=9)

        ax_dist.set_xlabel("Anomaly score (higher = more anomalous)")
        ax_dist.set_ylabel("Session count")
        ax_dist.set_title("Distribution of Isolation Forest anomaly scores")
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        # Quick context line
        flagged = int(baseline_df["anomaly_flag_iforest"].sum())
        median_score = scores.median()
        st.caption(
            f"{flagged} sessions flagged as anomalies out of {len(baseline_df)} total. "
            f"Median score: {median_score:.4f}."
        )

    st.divider()

    # ── Daily markdown report ───────────────────
    st.subheader("Daily threat report")

    if report_md is None:
        missing_file_error("daily_report.md")
    else:
        st.markdown(report_md)


# ══════════════════════════════════════════════
# TAB 2 — ANOMALIES
# ══════════════════════════════════════════════
with tab_anomalies:
    st.header("Anomaly Explorer")

    if baseline_df is None:
        missing_file_error("baseline_results.csv")
    else:
        # ── Filter controls ─────────────────────
        col_f1, col_f2, col_f3 = st.columns([1, 1, 2])

        with col_f1:
            only_anomalies = st.checkbox("Anomalies only (IForest flag = 1)", value=True)
        with col_f2:
            only_downloads = st.checkbox("Downloads only (download_count ≥ 1)")
        with col_f3:
            max_cmd = int(baseline_df["command_count"].max())
            min_cmds = st.slider("Minimum command_count", 0, max(max_cmd, 1), 0)

        # Apply filters
        anom_df = baseline_df.copy()
        if only_anomalies:
            anom_df = anom_df[anom_df["anomaly_flag_iforest"] == 1]
        if only_downloads:
            anom_df = anom_df[anom_df["download_count"] >= 1]
        anom_df = anom_df[anom_df["command_count"] >= min_cmds]
        anom_df = anom_df.sort_values("anomaly_score_iforest", ascending=False)

        # Columns to display in the table
        display_cols = [
            "session_id", "src_ip", "duration_seconds", "command_count",
            "auth_fail_count", "download_count", "priv_esc_count",
            "sensitive_path_count", "recon_count",
            "anomaly_score_iforest", "lof_score",
            "anomaly_flag_iforest", "anomaly_flag_lof", "baseline_rule_flag",
        ]
        display_cols = [c for c in display_cols if c in anom_df.columns]

        st.caption(f"Showing **{len(anom_df)}** session(s) — click a row to inspect it.")

        # Interactive table with single-row selection
        event = st.dataframe(
            anom_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            key="anomaly_table",
        )

        # ── Row detail panel ────────────────────
        selected_rows = event.selection.rows if (event and event.selection) else []

        if selected_rows:
            idx = selected_rows[0]
            row = anom_df.iloc[idx]
            sid = row["session_id"]

            st.divider()
            st.subheader(f"Session detail — `{sid}`")

            detail_left, detail_right = st.columns(2)

            # Command chain (first 15 commands)
            with detail_left:
                st.markdown("**Command chain (first 15)**")
                raw = sessions_by_id.get(sid)
                if raw and raw.get("commands"):
                    for i, cmd in enumerate(raw["commands"][:15], 1):
                        # Truncate very long commands for readability
                        display_cmd = cmd if len(cmd) <= 120 else cmd[:120] + " …"
                        st.code(f"{i:>2}. {display_cmd}", language=None)
                else:
                    st.info("No commands were captured for this session.")

            # "Why flagged?" explanation panel
            with detail_right:
                st.markdown("**Why flagged?**")
                for bullet in why_flagged(row):
                    st.markdown(f"- {bullet}")


# ══════════════════════════════════════════════
# TAB 3 — CLUSTERS
# ══════════════════════════════════════════════
with tab_clusters:
    st.header("Behaviour Clusters")

    if merged_df is None:
        if baseline_df is None:
            missing_file_error("baseline_results.csv")
        if clusters_df is None:
            missing_file_error("clusters.csv")
    else:
        # ── Cluster size bar chart ───────────────
        cluster_sizes = (
            merged_df["cluster_id"]
            .value_counts()
            .sort_index()
        )

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.bar(
            [f"Cluster {int(k)}" for k in cluster_sizes.index],
            cluster_sizes.values,
            color="#55A868",
            width=0.5,
        )
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Session count")
        ax2.set_title("Session count per behaviour cluster")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.divider()

        # ── Cluster drill-down ───────────────────
        cluster_ids = sorted(
            merged_df["cluster_id"].dropna().unique().astype(int).tolist()
        )
        selected_cluster = st.selectbox("Select a cluster to inspect", cluster_ids)

        cluster_rows = merged_df[merged_df["cluster_id"] == selected_cluster]

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown(
                f"**Cluster {selected_cluster}** — {len(cluster_rows)} session(s)"
            )

            # Representative session: the one closest to the cluster centroid
            numeric_cols = cluster_rows.select_dtypes(include="number").columns
            cluster_mean = cluster_rows[numeric_cols].mean()
            distances = (
                ((cluster_rows[numeric_cols] - cluster_mean) ** 2)
                .sum(axis=1)
                .apply(math.sqrt)
            )
            rep_idx = distances.idxmin()
            rep_row = cluster_rows.loc[rep_idx]
            rep_sid = rep_row["session_id"]

            st.markdown(f"**Representative session:** `{rep_sid}`")

            summary_fields = [
                "duration_seconds", "command_count", "auth_fail_count",
                "download_count", "priv_esc_count", "recon_count",
                "novelty_rate", "anomaly_score_iforest",
            ]
            rep_display = rep_row[
                [f for f in summary_fields if f in rep_row.index]
            ].to_frame("Value")
            rep_display["Value"] = rep_display["Value"].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else x
            )
            st.table(rep_display)

        with col_c2:
            st.markdown("**Average features for this cluster**")
            avg_fields = [
                "duration_seconds", "command_count", "auth_fail_count",
                "download_count", "priv_esc_count", "recon_count",
                "novelty_rate", "category_switch_rate", "cmds_per_sec_x",
                "anomaly_score_iforest",
            ]
            avg_fields = [f for f in avg_fields if f in cluster_rows.columns]
            avg_df = cluster_rows[avg_fields].mean().to_frame("Cluster avg")
            avg_df["Cluster avg"] = avg_df["Cluster avg"].apply(
                lambda x: f"{x:.4f}"
            )
            st.table(avg_df)

        # ── Typical commands from representative session ──
        st.markdown("**Typical commands (from representative session)**")
        raw_rep = sessions_by_id.get(rep_sid)
        if raw_rep and raw_rep.get("commands"):
            for i, cmd in enumerate(raw_rep["commands"][:15], 1):
                display_cmd = cmd if len(cmd) <= 120 else cmd[:120] + " …"
                st.code(f"{i:>2}. {display_cmd}", language=None)
        else:
            st.info("No commands captured for the representative session.")


# ══════════════════════════════════════════════
# TAB 4 — MAP
# ══════════════════════════════════════════════
with tab_map:
    st.header("Attack Source Map")

    if geo_df is None:
        missing_file_error("baseline_results.csv")
    elif not GEOIP_DB_PATH.exists():
        st.warning(
            "⚠️  GeoLite2-Country.mmdb not found in `./data/`.\n\n"
            "**To enable the map:**\n"
            "1. Create a free MaxMind account at https://dev.maxmind.com/geoip/geolite2-free-geolocation-data\n"
            "2. Download **GeoLite2 Country** (`.mmdb` format)\n"
            "3. Place `GeoLite2-Country.mmdb` in your `data/` folder\n"
            "4. Reload this page"
        )
    else:
        # Aggregate sessions per country
        country_stats = (
            geo_df[geo_df["country_code"] != "--"]
            .groupby(["country", "country_code", "lat", "lon"])
            .agg(
                sessions=("session_id", "count"),
                unique_ips=("src_ip", "nunique"),
                avg_duration=("duration_seconds", "mean"),
                anomalies=("anomaly_flag_iforest", "sum"),
            )
            .reset_index()
        )

        if country_stats.empty:
            st.info("No IPs could be resolved to countries. Check the .mmdb file.")
        else:
            # ── KPI row ─────────────────────────
            total_countries = len(country_stats)
            top_country = country_stats.loc[country_stats["sessions"].idxmax()]

            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("Countries seen", total_countries)
            kc2.metric("Top source country", top_country["country"])
            kc3.metric(
                "Sessions from top country",
                int(top_country["sessions"]),
            )

            st.divider()

            # ── Bubble map via pydeck ───────────
            # Scale bubble radius: sqrt so large counts don't dominate
            max_sessions = country_stats["sessions"].max()
            country_stats["radius"] = (
                (country_stats["sessions"] / max(max_sessions, 1)) ** 0.5 * 800000
            ).clip(lower=120000)

            # Colour: bright orange → red based on anomaly ratio
            def anomaly_colour(row):
                ratio = row["anomalies"] / max(row["sessions"], 1)
                r = 255
                g = int(180 * (1 - ratio))
                b = int(50 * (1 - ratio))
                return [r, g, b, 210]

            country_stats["colour"] = country_stats.apply(anomaly_colour, axis=1)

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=country_stats,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color="colour",
                pickable=True,
                auto_highlight=True,
            )

            view = pdk.ViewState(
                latitude=30,
                longitude=20,
                zoom=1.0,
                pitch=0,
            )

            tooltip = {
                "html": (
                    "<b>{country}</b> ({country_code})<br/>"
                    "Sessions: {sessions}<br/>"
                    "Unique IPs: {unique_ips}<br/>"
                    "Anomalies: {anomalies}"
                ),
                "style": {
                    "backgroundColor": "#1a1a2e",
                    "color": "white",
                    "fontSize": "14px",
                    "padding": "10px",
                    "borderRadius": "6px",
                },
            }

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view,
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                ),
                height=500,
            )

            st.divider()

            # ── Country breakdown table ─────────
            st.subheader("Sessions by country")
            display_geo = (
                country_stats[["country", "country_code", "sessions", "unique_ips", "anomalies", "avg_duration"]]
                .sort_values("sessions", ascending=False)
                .reset_index(drop=True)
            )
            display_geo["avg_duration"] = display_geo["avg_duration"].round(2)
            display_geo.columns = ["Country", "Code", "Sessions", "Unique IPs", "Anomalies", "Avg Duration (s)"]
            st.dataframe(display_geo, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 5 — EXPLAIN
# ══════════════════════════════════════════════
with tab_explain:
    st.header("How This Dashboard Works")
    st.caption(
        "This page explains the AI pipeline, models, and features used in this project. "
        "Designed to help assessors and non-technical readers understand the system."
    )

    # ── Section 1: Model Comparison ─────────────
    st.subheader("Model Performance Comparison")
    st.markdown(
        "Three detection methods were evaluated against a manually labelled subset of 75 sessions. "
        "The table below shows precision, recall, and F1-score for each method."
    )

    metrics_data = pd.DataFrame({
        "Method": ["Rule Baseline", "Isolation Forest", "LOF"],
        "Precision": [0.547, 0.500, 0.429],
        "Recall": [0.784, 0.081, 0.081],
        "F1-Score": [0.644, 0.140, 0.136],
        "TP": [29, 3, 3],
        "FP": [24, 3, 4],
        "FN": [8, 34, 34],
        "TN": [14, 35, 34],
    })
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)

    st.markdown(
        "**Why does the rule baseline outperform ML?** "
        "The dataset is dominated by short, automated bot probes — over 99% of sessions "
        "last under 3 seconds with a single command. Isolation Forest and LOF treat this "
        "overwhelming bot traffic as \"normal\" (since it is the majority), making it hard "
        "to flag individual bots as anomalies. The rule baseline uses explicit thresholds "
        "(e.g. high commands-per-second, short duration) that directly match bot patterns, "
        "giving it higher recall. This is a known limitation of unsupervised anomaly detection "
        "when the contamination rate is very high (Rehman et al., 2025)."
    )

    st.divider()

    # ── Section 2: Ablation Study ───────────────
    st.subheader("Ablation Study (Isolation Forest)")
    st.markdown(
        "To understand which features drive anomaly detection, the Isolation Forest was "
        "tested with progressively richer feature sets at a fixed contamination rate of 0.4."
    )

    ablation_data = pd.DataFrame({
        "Feature Set": [
            "A — Timing only",
            "A+B — Timing + intent/auth",
            "A+B+C — Full features",
        ],
        "Precision": [0.400, 0.367, 0.367],
        "Recall": [0.324, 0.297, 0.297],
        "F1-Score": [0.358, 0.328, 0.328],
        "Flagged": [30, 30, 30],
    })
    st.dataframe(ablation_data, use_container_width=True, hide_index=True)

    st.markdown(
        "**Key finding:** Timing-only features (duration, commands-per-second, inter-command gaps) "
        "achieved the best F1-score. Adding intent and auth features did not improve performance. "
        "This confirms that early-stage attacker traffic is best characterised by timing behaviour — "
        "bots act fast and leave quickly, while the rare interactive sessions show longer, variable durations."
    )

    st.divider()

    # ── Section 3: Glossary ─────────────────────
    st.subheader("System Glossary")
    st.markdown("Expand any section below to learn how each component works.")

    with st.expander("🌲 Isolation Forest (main AI model)"):
        st.markdown(
            "Isolation Forest is an **unsupervised** anomaly detection algorithm (Liu et al., 2008). "
            "It works by randomly partitioning data using decision trees — anomalies are data points "
            "that require fewer partitions to isolate, because they are rare and different from the majority. "
            "Unlike density-based methods, it runs efficiently at O(n log n) time complexity.\n\n"
            "**In this project:** Each session is represented as a feature vector (timing, commands, auth). "
            "The model assigns an `anomaly_score_iforest` — higher scores mean the session is more unusual. "
            "Sessions scoring above the threshold are flagged with `anomaly_flag_iforest = 1`."
        )

    with st.expander("📏 Rule Baseline"):
        st.markdown(
            "A simple, explainable heuristic classifier. It flags a session as suspicious if:\n\n"
            "- Duration < 30 seconds **and** command count ≥ 5, **or**\n"
            "- Commands-per-second > 0.5\n\n"
            "This directly targets automated scripts that execute many commands in a short burst. "
            "It serves as a **benchmark** — if the ML models can't beat a simple rule, "
            "the dataset may not have enough diversity for unsupervised learning to add value."
        )

    with st.expander("🔍 Local Outlier Factor (LOF)"):
        st.markdown(
            "LOF is a **density-based** anomaly detection algorithm. It compares the local density "
            "of each data point to the density of its neighbours — points in sparser regions are flagged "
            "as outliers.\n\n"
            "**In this project:** LOF is used as a second baseline comparator alongside the rule baseline. "
            "It uses the same session features as Isolation Forest. The `lof_score` column represents "
            "the outlier factor (higher = more anomalous)."
        )

    with st.expander("🔵 Behaviour Clustering"):
        st.markdown(
            "After anomaly scoring, sessions are grouped into **behaviour clusters** using K-Means "
            "or DBSCAN. Clustering reveals distinct attacker strategies — for example, one cluster "
            "might contain short brute-force probes while another contains longer reconnaissance sessions.\n\n"
            "**In this project:** The Clusters tab shows the size of each cluster, its average features, "
            "and a representative session with its command chain. Clusters with very few sessions "
            "often represent unusual or targeted attack behaviour."
        )

    with st.expander("📊 Features used by the models"):
        st.markdown(
            "Each session is described by ~20 numerical features extracted from raw Cowrie logs:\n\n"
            "**Timing features (Group A):** `duration_seconds`, `cmds_per_sec`, "
            "`mean_inter_command_time`, `std_inter_command_time`, `time_to_first_command`\n\n"
            "**Intent & auth features (Group B):** `auth_fail_count`, `auth_success`, "
            "`recon_count`, `download_count`, `priv_esc_count`, `sensitive_path_count`\n\n"
            "**Behaviour features (Group C):** `novelty_rate` (fraction of unique commands), "
            "`category_switch_rate` (how often the attacker alternates between recon/download/escalation), "
            "`had_commands`, `repeated_command_ratio`\n\n"
            "The ablation study showed Group A (timing) carries the strongest signal for this dataset."
        )

    with st.expander("🌍 GeoIP Map"):
        st.markdown(
            "The Map tab resolves each attacker IP to its **country of origin** using the "
            "MaxMind GeoLite2 offline database. This runs entirely locally — no IP data is sent "
            "to any external service.\n\n"
            "Bubble size represents session volume from that country. "
            "Bubble colour intensity (orange → red) represents the proportion of sessions "
            "flagged as anomalies. Hover over a bubble to see detailed counts."
        )

    with st.expander("🛡️ Safety & ethics"):
        st.markdown(
            "**Port separation:** Cowrie honeypot listens on port 22 (public). Real admin SSH "
            "is on port 2223, restricted to a single IP via AWS Security Group rules.\n\n"
            "**Egress filtering:** UFW blocks all outbound traffic except DNS (53) and HTTPS (443), "
            "preventing attackers from using the honeypot as a launchpad for further attacks.\n\n"
            "**IP anonymisation:** All IPs in the daily threat report are SHA-256 hashed. "
            "The dashboard shows raw IPs for analysis but these are never published or shared.\n\n"
            "**Data handling:** Raw logs are stored only on the EC2 instance and the researcher's "
            "local machine. They will be permanently deleted after project completion."
        )

    st.divider()

    # ── Section 4: Today's Summary (kept from original) ──
    st.subheader("Today's Data Summary")
    if st.button("🔍 Generate summary for today's data", type="primary"):
        if baseline_df is None:
            missing_file_error("baseline_results.csv")
        else:
            total      = len(baseline_df)
            uniq_ips   = baseline_df["src_ip"].nunique()
            rule_cnt   = int(baseline_df["baseline_rule_flag"].sum())
            if_cnt     = int(baseline_df["anomaly_flag_iforest"].sum())
            lof_cnt    = int(baseline_df["anomaly_flag_lof"].sum())
            dl_count   = int((baseline_df["download_count"] > 0).sum())
            priv_count = int((baseline_df["priv_esc_count"] > 0).sum())
            recon_sess = int((baseline_df["recon_count"] > 0).sum())

            top_ips = baseline_df["src_ip"].value_counts().head(3)
            top_ip_str = ", ".join(f"`{ip}` ({n})" for ip, n in top_ips.items())

            max_row   = baseline_df.loc[baseline_df["anomaly_score_iforest"].idxmax()]
            max_sid   = max_row["session_id"]
            max_score = max_row["anomaly_score_iforest"]

            bullets = [
                f"**{total} SSH sessions** from **{uniq_ips} unique IPs** — broad scanning activity.",
                f"**Rule baseline** flagged **{rule_cnt}** sessions ({rule_cnt / total * 100:.1f}%).",
                f"**Isolation Forest** flagged **{if_cnt}** anomalies; **LOF** flagged **{lof_cnt}**.",
                f"**Top source IPs:** {top_ip_str}.",
                f"**Highest anomaly score:** session `{max_sid}` (score: {max_score:.4f}).",
                f"**{dl_count}** download(s), **{priv_count}** priv-esc attempt(s), "
                f"**{recon_sess}** recon session(s).",
            ]

            if merged_df is not None:
                cs = merged_df["cluster_id"].value_counts().sort_index()
                cluster_str = " | ".join(f"Cluster {int(k)}: {v}" for k, v in cs.items())
                bullets.append(f"**Clusters:** {cluster_str}.")

            for b in bullets:
                st.markdown(f"- {b}")