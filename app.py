"""
SSH honeypot threat intelligence dashboard.

Tabs: Overview · Anomalies · Attacker Profiles · Map · Explain
Modules: config, auth, data_loader, models, geoip_lookup

Author: Anas Hussein - BSc CS with Cyber Security, University of Salford 2025/2026
"""

import math
import re as re_mod
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pydeck as pdk
import streamlit as st

from config import GEOIP_DB_PATH
from auth import check_login, is_admin, render_admin_panel
from data_loader import (
    load_baseline, load_clusters, load_sessions, load_report, missing_file_error,
)
from models import (
    epoch_to_utc, why_flagged, classify_cluster, generate_actionable_intel,
)
from geoip_lookup import enrich_with_geo


# --- Page config ---
st.set_page_config(
    page_title="AI Honeypot - Threat Intelligence Dashboard",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Authentication gate ---

if not check_login():
    st.stop()

# --- Load all data ---

baseline_df  = load_baseline()
clusters_df  = load_clusters()
sessions_raw = load_sessions()
report_md    = load_report()

sessions_by_id = {}
if sessions_raw:
    sessions_by_id = {s["session_id"]: s for s in sessions_raw}

merged_df = None
if baseline_df is not None and clusters_df is not None:
    merged_df = baseline_df.merge(clusters_df, on="session_id", how="left")

geo_df = None
if baseline_df is not None:
    geo_df = enrich_with_geo(
        baseline_df.copy(), ip_column="src_ip", db_path=str(GEOIP_DB_PATH),
    )

# --- Page header ---

st.markdown(f"""
<div class="main-header">
    <h1>🛡️ AI-Enhanced SSH Honeypot - Threat Intelligence</h1>
    <p>BSc Computer Science with Cyber Security - Anas Hussein - University of Salford 2025/2026
    &nbsp;|&nbsp; Signed in as <strong>{st.session_state.username}</strong>
    &nbsp;|&nbsp; Data captured 23 Feb to 19 Mar 2026</p></div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Dashboard Controls")
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

    st.divider()
    st.markdown("**Project Info**")
    st.markdown(
        "- **Supervisors:** Dr. Nashnush / Dr. Bryant\n"
        "- **Module:** 50280 Security Project\n"
        "- **Honeypot:** Cowrie on AWS EC2\n"
        "- **AI Model:** Isolation Forest"
    )
    if baseline_df is not None:
        st.divider()
        st.metric("Total Sessions", f"{len(baseline_df):,}")
        st.metric("Unique IPs", baseline_df["src_ip"].nunique())

    # Admin panel (only visible to admin users)
    render_admin_panel()

# --- Tabs ---

tab_overview, tab_anomalies, tab_clusters, tab_map, tab_explain = st.tabs(
    ["Overview ", "Anomalies ", "Attacker Profiles ", "Map ", "Explain "]
)

# --- TAB 1: OVERVIEW ---
with tab_overview:
    if baseline_df is None:
        missing_file_error("baseline_results.csv")
    else:
        total_sessions    = len(baseline_df)
        unique_ips        = baseline_df["src_ip"].nunique()
        rule_flagged      = int(baseline_df["baseline_rule_flag"].sum())
        iforest_anomalies = int(baseline_df["anomaly_flag_iforest"].sum())
        lof_anomalies     = int(baseline_df["anomaly_flag_lof"].sum())
        dl_sessions       = int((baseline_df["download_count"] > 0).sum())
        cmd_sessions_count = int((baseline_df["command_count"] > 0).sum())

        n_countries = "20+"
        if geo_df is not None:
            try:
                n_countries = str(geo_df[geo_df["country_code"] != "--"]["country"].nunique())
            except Exception:
                pass

        # Hero banner
        st.markdown(f"""
        <div class="hero-banner">
            <div class="hero-number">{total_sessions:,}</div>
            <div class="hero-label">Attack sessions intercepted and analysed by AI</div>
            <div style="margin-top: 1rem; position: relative;">
                <span class="hero-stat">🌐 <strong>{unique_ips:,}</strong> unique IPs</span>
                <span class="hero-stat">🗺️ <strong>{n_countries}</strong> countries</span>
                <span class="hero-stat">🚨 <strong>{rule_flagged:,}</strong> rule-flagged</span>
                <span class="hero-stat">🤖 <strong>{iforest_anomalies:,}</strong> AI anomalies</span>
                <span class="hero-stat">⬇️ <strong>{dl_sessions:,}</strong> downloads</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Traffic breakdown + KPIs
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("**⚡ Traffic Breakdown**")
            short_no_cmd = len(baseline_df[(baseline_df["duration_seconds"] < 3) & (baseline_df["command_count"] == 0)])
            short_with_cmd = len(baseline_df[(baseline_df["duration_seconds"] < 3) & (baseline_df["command_count"] > 0)])
            medium_sessions = len(baseline_df[(baseline_df["duration_seconds"] >= 3) & (baseline_df["duration_seconds"] < 60)])
            long_sessions = len(baseline_df[baseline_df["duration_seconds"] >= 60])

            categories = [
                {"name": "Rapid-Fire Bots", "count": short_with_cmd, "color": "#e53935", "desc": "Under 3 seconds, executed commands"},
                {"name": "Banner Grabbers", "count": short_no_cmd, "color": "#78909c", "desc": "Under 3 seconds, no shell interaction"},
                {"name": "Medium Sessions", "count": medium_sessions, "color": "#ff9800", "desc": "3 to 60 seconds, possible targeted scans"},
                {"name": "Extended Sessions", "count": long_sessions, "color": "#1e88e5", "desc": "Over 60 seconds, likely human operators"},
            ]
            for cat in categories:
                pct = cat["count"] / total_sessions * 100
                bar_width = max(pct, 1.5)
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:10px 14px; margin:6px 0; '
                    f'border-top:2px solid {cat["color"]}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<div><strong>{cat["name"]}</strong>'                    f'<br/><span style="color:#a0aec0; font-size:0.85rem;">{cat["desc"]}</span></div>'
                    f'<div style="text-align:right;">'
                    f'<span style="font-size:1.4rem; font-weight:800; color:{cat["color"]};">{cat["count"]:,}</span>'
                    f'<br/><span style="color:#a0aec0; font-size:0.85rem;">{pct:.1f}%</span></div>'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.1); border-radius:4px; height:6px; margin-top:6px;">'
                    f'<div style="background:{cat["color"]}; width:{bar_width}%; height:100%; border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        with col_right:
            st.markdown("**📊 Key Metrics**")
            col_total, col_ips = st.columns(2)
            col_total.metric("Total Sessions", f"{total_sessions:,}")
            col_ips.metric("Unique IPs", f"{unique_ips:,}")
            col_rule, col_ai = st.columns(2)
            col_rule.metric("Rule-Flagged", f"{rule_flagged:,}")
            col_ai.metric("AI Anomalies", f"{iforest_anomalies:,}")

    st.divider()

    # Actionable Intelligence
    st.subheader("🎯 Actionable Threat Intelligence")
    st.caption("Automated recommendations derived from honeypot analysis.")

    if baseline_df is not None:
        actions = generate_actionable_intel(baseline_df, geo_df)
        if actions:
            for action in actions:
                level = action["level"]
                icon = {"critical": "🔴", "warning": "🟡"}.get(level, "🔵")
                css = {"critical": "action-card-critical", "warning": "action-card"}.get(level, "action-card-info")
                st.markdown(
                    f'<div class="{css}"><strong>{icon} {action["title"]}</strong><br/>{action["detail"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No critical actions required based on current data.")

    st.divider()

    # Top 5 Most Dangerous Sessions
    st.subheader("Top 5 Most Dangerous Sessions")
    st.caption("Highest anomaly-scored sessions with shell activity.")

    if baseline_df is not None:
        top5_candidates = baseline_df[baseline_df["command_count"] > 0].copy()
        if len(top5_candidates) == 0:
            top5_candidates = baseline_df.copy()
        top5 = top5_candidates.nlargest(5, "anomaly_score_iforest")

        for rank, (_, t5row) in enumerate(top5.iterrows(), 1):
            sid = t5row["session_id"]
            score = t5row["anomaly_score_iforest"]
            dur = t5row["duration_seconds"]
            ncmds = int(t5row["command_count"])
            src = t5row["src_ip"]

            tags = []
            if int(t5row.get("download_count", 0)) > 0: tags.append("⬇️ Download")
            if int(t5row.get("priv_esc_count", 0)) > 0: tags.append("🔓 Priv-Esc")
            if int(t5row.get("recon_count", 0)) > 0: tags.append("🔍 Recon")
            if dur > 60: tags.append("🕐 Long Session")
            tag_str = " &nbsp;".join(f"`{t}`" for t in tags) if tags else "`🤖 Bot Probe`"

            with st.expander(f"#{rank}  |  Score: {score:.4f}  |  {ncmds} cmds  |  {dur:.1f}s  |  {src}", expanded=(rank == 1)):
                st.markdown(f"**Tags:** {tag_str}")
                col_cmds, col_why = st.columns(2)
                with col_cmds:
                    st.markdown("**Command chain:**")
                    raw = sessions_by_id.get(sid)
                    if raw and raw.get("commands"):
                        for i, cmd in enumerate(raw["commands"][:10], 1):
                            display_cmd = cmd if len(cmd) <= 100 else cmd[:100] + " ..."
                            st.code(f"{i:>2}. {display_cmd}", language=None)
                    else:
                        st.info("No commands captured.")
                with col_why:
                    st.markdown("**Why dangerous:**")
                    for bullet in why_flagged(t5row):
                        st.markdown(f"- {bullet}")

    st.divider()

    # Session Volume Over Time
    st.subheader("Session Volume Over Time")
    st.caption("Interactive chart. Hover for details, drag to zoom.")

    has_time_data = False
    if sessions_raw:
        utc_times = pd.Series([epoch_to_utc(s["start_ts"]) for s in sessions_raw], name="start_utc")
        if len(utc_times) > 0:
            has_time_data = True

    if not has_time_data:
        st.info("No session timestamp data available.")
    else:
        span_days = (utc_times.max() - utc_times.min()).days if len(utc_times) > 1 else 0
        if span_days > 2:
            chart_df = utc_times.dt.floor("D").value_counts().sort_index().reset_index()
            chart_df.columns = ["Date", "Sessions"]
            chart_df = chart_df.set_index("Date")
        else:
            chart_df = utc_times.dt.floor("h").value_counts().sort_index().reset_index()
            chart_df.columns = ["Time", "Sessions"]
            chart_df = chart_df.set_index("Time")

        st.area_chart(chart_df, color="#1e88e5", height=350)
        date_range = f"{utc_times.min().strftime('%d %b %Y')} to {utc_times.max().strftime('%d %b %Y')}"
        st.caption(f"Data range: {date_range} ({len(utc_times):,} sessions)")

        # Attack heatmap
        st.subheader("🗓️ Attack Heatmap")
        st.caption("When do attackers strike? Darker = more sessions.")

        utc_df = pd.DataFrame({"ts": utc_times})
        utc_df["dow"] = utc_df["ts"].dt.dayofweek
        utc_df["hour"] = utc_df["ts"].dt.hour
        heatmap_data = utc_df.groupby(["dow", "hour"]).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.reindex(index=range(7), columns=range(24), fill_value=0)

        fig_heat, ax_heat = plt.subplots(figsize=(14, 3.5))
        cmap = mcolors.LinearSegmentedColormap.from_list("attacks", ["#e8f4f8", "#2196f3", "#0d47a1", "#e53935"])
        im = ax_heat.imshow(heatmap_data.values, aspect="auto", cmap=cmap, interpolation="nearest")
        ax_heat.set_xticks(range(24))
        ax_heat.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
        ax_heat.set_yticks(range(7))
        ax_heat.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=9)
        ax_heat.set_xlabel("Hour (UTC)", fontsize=10)
        ax_heat.set_title("Attack volume by day of week and hour", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02).set_label("Sessions", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    st.divider()

    # Detection Summary
    st.subheader("🔎 What Did the AI Find?")
    if baseline_df is not None:
        total = len(baseline_df)
        rule_n = int(baseline_df["baseline_rule_flag"].sum())
        if_n = int(baseline_df["anomaly_flag_iforest"].sum())
        lof_n = int(baseline_df["anomaly_flag_lof"].sum())
        dl_n = int((baseline_df["download_count"] > 0).sum())
        priv_n = int((baseline_df["priv_esc_count"] > 0).sum())
        recon_n = int((baseline_df["recon_count"] > 0).sum())

        col_rule_det, col_iforest_det, col_lof_det = st.columns(3)
        with col_rule_det:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#fff8e1,#fff3cd); border-radius:12px; '
                f'padding:20px; text-align:center; border:2px solid #ffc107;">'
                f'<div style="font-size:0.9rem; color:#a0aec0;">📏 Rule Baseline</div>'
                f'<div style="font-size:2.5rem; font-weight:800; color:#f57f17;">{rule_n:,}</div>'
                f'<div style="font-size:0.85rem; color:#a0aec0;">flagged ({rule_n/total*100:.1f}%)</div>'
                f'<div style="font-size:0.8rem; color:#999; margin-top:6px;">Best performer, F1: 0.64</div></div>',
                unsafe_allow_html=True)
        with col_iforest_det:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#e3f2fd,#bbdefb); border-radius:12px; '
                f'padding:20px; text-align:center; border:2px solid #2196f3;">'
                f'<div style="font-size:0.9rem; color:#a0aec0;">🌲 Isolation Forest (AI)</div>'
                f'<div style="font-size:2.5rem; font-weight:800; color:#1565c0;">{if_n:,}</div>'
                f'<div style="font-size:0.85rem; color:#a0aec0;">anomalies ({if_n/total*100:.1f}%)</div>'
                f'<div style="font-size:0.8rem; color:#999; margin-top:6px;">Unsupervised ML, F1: 0.14</div></div>',
                unsafe_allow_html=True)
        with col_lof_det:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#f3e5f5,#e1bee7); border-radius:12px; '
                f'padding:20px; text-align:center; border:2px solid #9c27b0;">'
                f'<div style="font-size:0.9rem; color:#a0aec0;">🔍 Local Outlier Factor</div>'
                f'<div style="font-size:2.5rem; font-weight:800; color:#7b1fa2;">{lof_n:,}</div>'
                f'<div style="font-size:0.85rem; color:#a0aec0;">outliers ({lof_n/total*100:.1f}%)</div>'
                f'<div style="font-size:0.8rem; color:#999; margin-top:6px;">Density-based, F1: 0.14</div></div>',
                unsafe_allow_html=True)

        col_dl_act, col_priv_act, col_recon_act = st.columns(3)
        
        with col_dl_act:
            st.markdown(f'<div style="text-align:center; padding:10px;"><span style="font-size:2rem;">⬇️</span><br/><strong style="font-size:1.3rem;">{dl_n}</strong><br/><span style="color:#a0aec0;">Download attempts</span></div>', unsafe_allow_html=True)
        
        with col_priv_act:
            st.markdown(f'<div style="text-align:center; padding:10px;"><span style="font-size:2rem;">🔓</span><br/><strong style="font-size:1.3rem;">{priv_n}</strong><br/><span style="color:#a0aec0;">Privilege escalations</span></div>', unsafe_allow_html=True)
        
        with col_recon_act:
            st.markdown(f'<div style="text-align:center; padding:10px;"><span style="font-size:2rem;">🔍</span><br/><strong style="font-size:1.3rem;">{recon_n}</strong><br/><span style="color:#a0aec0;">Reconnaissance sessions</span></div>', unsafe_allow_html=True)
        
        with st.expander("📊 View detailed anomaly score distribution"):
            scores = baseline_df["anomaly_score_iforest"].dropna()
            fig_dist, ax_dist = plt.subplots(figsize=(12, 3.5))
            ax_dist.hist(scores, bins=50, color="#2c5364", edgecolor="white", linewidth=0.5, alpha=0.85)
            threshold = baseline_df.loc[baseline_df["anomaly_flag_iforest"] == 1, "anomaly_score_iforest"].min() if (baseline_df["anomaly_flag_iforest"] == 1).any() else None
            if threshold is not None:
                ax_dist.axvline(x=threshold, color="#dc3545", linestyle="--", linewidth=2, label=f"Anomaly threshold ({threshold:.3f})")
                ax_dist.legend(fontsize=9)
            ax_dist.set_xlabel("Anomaly score (higher = more anomalous)", fontsize=10)
            ax_dist.set_ylabel("Session count", fontsize=10)
            ax_dist.spines["top"].set_visible(False)
            ax_dist.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)

    st.divider()
    st.subheader("📋 Daily Threat Report")
    if report_md is None:
        missing_file_error("daily_report.md")
    else:
        with st.expander("View full daily report", expanded=False):
            st.markdown(report_md)


# --- TAB 2: ANOMALIES ---

with tab_anomalies:
    st.header("🚨 Anomaly Explorer")
    st.caption("Investigate suspicious sessions flagged by our AI models.")

    if baseline_df is None:
        missing_file_error("baseline_results.csv")
    else:
        col_anom_ai, col_anom_dl, col_anom_priv, col_anom_recon, col_anom_long = st.columns(5)
        col_anom_ai.metric("🤖 AI Anomalies", f"{int(baseline_df['anomaly_flag_iforest'].sum()):,}")
        col_anom_dl.metric("⬇️ Downloads", f"{int((baseline_df['download_count'] > 0).sum()):,}")
        col_anom_priv.metric("🔓 Priv-Esc", f"{int((baseline_df['priv_esc_count'] > 0).sum()):,}")
        col_anom_recon.metric("🔍 Recon", f"{int((baseline_df['recon_count'] > 0).sum()):,}")
        col_anom_long.metric("🕐 Long (>60s)", f"{int((baseline_df['duration_seconds'] > 60).sum()):,}")

        st.divider()
        col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 2])
        with col_f1: only_anomalies = st.checkbox("AI anomalies only", value=True)
        with col_f2: only_downloads = st.checkbox("Downloads only")
        with col_f3: only_recon = st.checkbox("Recon only")
        with col_f4:
            max_cmd = int(baseline_df["command_count"].max())
            min_cmds = st.slider("Min command count", 0, max(max_cmd, 1), 0)

        anom_df = baseline_df.copy()
        if only_anomalies: anom_df = anom_df[anom_df["anomaly_flag_iforest"] == 1]
        if only_downloads: anom_df = anom_df[anom_df["download_count"] >= 1]
        if only_recon: anom_df = anom_df[anom_df["recon_count"] >= 1]
        anom_df = anom_df[anom_df["command_count"] >= min_cmds]
        anom_df = anom_df.sort_values("anomaly_score_iforest", ascending=False)

        display_cols = [c for c in ["session_id", "src_ip", "duration_seconds", "command_count",
            "download_count", "priv_esc_count", "recon_count", "anomaly_score_iforest", "baseline_rule_flag"]
            if c in anom_df.columns]

        st.markdown(f"**{len(anom_df):,}** sessions match your filters.")
        event = st.dataframe(anom_df[display_cols].reset_index(drop=True), use_container_width=True,
                             selection_mode="single-row", on_select="rerun", key="anomaly_table", height=350)
        selected_rows = event.selection.rows if (event and event.selection) else []

        if selected_rows:
            row = anom_df.iloc[selected_rows[0]]
            sid = row["session_id"]
            st.divider()
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:12px; '
                f'padding:16px 20px; color:white; margin-bottom:12px;">'
                f'<div style="font-size:1.1rem; font-weight:700;">🔍 Session: {sid}</div>'
                f'<div style="display:flex; gap:20px; margin-top:8px; flex-wrap:wrap;">'
                f'<span>📊 Score: <strong>{row.get("anomaly_score_iforest",0):.4f}</strong></span>'
                f'<span>⏱️ Duration: <strong>{row.get("duration_seconds",0):.1f}s</strong></span>'
                f'<span>💻 Commands: <strong>{int(row.get("command_count",0))}</strong></span>'
                f'<span>🌐 Source: <strong>{row.get("src_ip","N/A")}</strong></span></div></div>',
                unsafe_allow_html=True)

            detail_left, detail_right = st.columns([1.3, 1])
            with detail_left:
                st.markdown("**💻 Command Chain:**")
                raw = sessions_by_id.get(sid)
                if raw and raw.get("commands"):
                    for i, cmd in enumerate(raw["commands"][:15], 1):
                        st.code(f"{i:>2}. {cmd[:120]}", language="bash")
                else:
                    st.info("No commands captured.")
            with detail_right:
                st.markdown("**⚠️ Why flagged?**")
                for bullet in why_flagged(row):
                    st.markdown(f"- {bullet}")
                st.markdown("---")
                for label, val in [("Auth failures", int(row.get("auth_fail_count",0))),
                                   ("Downloads", int(row.get("download_count",0))),
                                   ("Priv-esc", int(row.get("priv_esc_count",0))),
                                   ("Recon", int(row.get("recon_count",0)))]:
                    color = "#e53935" if val > 0 else "#999"
                    st.markdown(f'<span style="color:{color};">{"🔴" if val > 0 else "⚪"} {label}: <strong>{val}</strong></span>', unsafe_allow_html=True)
        else:
            st.info("👆 Click any row above to see the full session analysis.")



# --- TAB 3: ATTACKER PROFILES ---

with tab_clusters:
    st.header("🎭 Attacker Profiles")
    st.caption("Who is attacking, how and what are they after?")

    if baseline_df is None:
        missing_file_error("baseline_results.csv")
    else:
        # Top 10 Attackers
        st.subheader("🎯 Top 10 Most Active Attackers")
        top_attackers = baseline_df.groupby("src_ip").agg(
            sessions=("session_id", "count"), total_cmds=("command_count", "sum"),
            avg_duration=("duration_seconds", "mean"), downloads=("download_count", "sum"),
            priv_esc=("priv_esc_count", "sum"), recon=("recon_count", "sum"),
        ).sort_values("sessions", ascending=False).head(10).reset_index()

        for rank, (_, a) in enumerate(top_attackers.iterrows(), 1):
            pct = a["sessions"] / len(baseline_df) * 100
            if a["avg_duration"] < 3 and a["total_cmds"] / max(a["sessions"], 1) < 2:
                atype, aicon = "Automated Scanner", "🤖"
            elif a["downloads"] > 0: atype, aicon = "Payload Deployer", "⬇️"
            elif a["recon"] > 5: atype, aicon = "Recon Agent", "🔍"
            elif a["avg_duration"] > 30: atype, aicon = "Interactive Operator", "🕵️"
            else: atype, aicon = "Brute-Force Bot", "🔑"
            tc = "#e53935" if pct > 10 else "#ff9800" if pct > 1 else "#43a047"

            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:12px 16px; margin:5px 0; border-left:4px solid {tc};">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">'
                f'<div><strong>#{rank}</strong> &nbsp; <code>{a["src_ip"]}</code> &nbsp; {aicon} <em>{atype}</em></div>'
                f'<div style="text-align:right;"><span style="font-size:1.3rem; font-weight:800; color:{tc};">{int(a["sessions"]):,}</span>'
                f' <span style="color:#a0aec0;">sessions ({pct:.1f}%)</span></div></div>'
                f'<div style="display:flex; gap:16px; margin-top:6px; font-size:0.85rem; color:#555; flex-wrap:wrap;">'
                f'<span>💻 {int(a["total_cmds"]):,} cmds</span>'
                f'<span>⏱️ {a["avg_duration"]:.1f}s avg</span>'
                f'<span>⬇️ {int(a["downloads"])} downloads</span>'
                f'<span>🔓 {int(a["priv_esc"])} priv-esc</span>'
                f'<span>🔍 {int(a["recon"])} recon</span></div></div>',
                unsafe_allow_html=True)

        st.divider()

        # Credentials
        st.subheader("🔑 Most Targeted Credentials")
        cred_left, cred_right = st.columns(2)
        if report_md:
            user_matches = re_mod.findall(r'`(\S+)` — ([\d,]+)', report_md.split("Usernames")[1].split("###")[0]) if "Usernames" in report_md else []
            pass_section = report_md.split("Passwords")[1].split("##")[0] if "Passwords" in report_md else ""
            pass_matches = re_mod.findall(r'`(.+?)` — ([\d,]+)', pass_section)

            with cred_left:
                st.markdown("**Top Usernames**")
                if user_matches:
                    max_u = int(user_matches[0][1].replace(",", ""))
                    for user, count_str in user_matches[:10]:
                        count = int(count_str.replace(",", ""))
                        st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin:4px 0;"><code style="min-width:100px;">{user}</code><div style="flex:1; background:rgba(255,255,255,0.1); border-radius:3px; height:20px;"><div style="background:#1e88e5; width:{count/max_u*100}%; height:100%; border-radius:3px; display:flex; align-items:center; padding-left:6px;"><span style="color:white; font-size:0.75rem; font-weight:700;">{count:,}</span></div></div></div>', unsafe_allow_html=True)
                else: st.info("No username data available.")
            with cred_right:
                st.markdown("**Top Passwords**")
                if pass_matches:
                    max_p = int(pass_matches[0][1].replace(",", ""))
                    for pw, count_str in pass_matches[:10]:
                        count = int(count_str.replace(",", ""))
                        st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin:4px 0;"><code style="min-width:100px;">{pw[:20]}</code><div style="flex:1; background:rgba(255,255,255,0.1); border-radius:3px; height:20px;"><div style="background:#e53935; width:{count/max_p*100}%; height:100%; border-radius:3px; display:flex; align-items:center; padding-left:6px;"><span style="color:white; font-size:0.75rem; font-weight:700;">{count:,}</span></div></div></div>', unsafe_allow_html=True)
                else: st.info("No password data available.")
        else:
            with cred_left: st.info("Requires daily_report.md")
            with cred_right: st.info("Requires daily_report.md")

        st.divider()

        # Attacker Type Breakdown
        st.subheader("🧬 Attacker Type Breakdown")
        st.caption("AI clustering identifies distinct attack strategies.")
        if merged_df is not None:
            cluster_ids = sorted(merged_df["cluster_id"].dropna().unique().astype(int).tolist())
            cluster_info = {}
            for cid in cluster_ids:
                c_rows = merged_df[merged_df["cluster_id"] == cid]
                cluster_info[cid] = classify_cluster(c_rows)
                cluster_info[cid]["size"] = len(c_rows)

            for cid in cluster_ids:
                info = cluster_info[cid]
                pct = info["size"] / len(merged_df) * 100
                tb = {"High": "#dc3545", "Medium": "#ff9800", "Low": "#43a047"}.get(info["threat"], "#78909c")
                bg = {"High": "rgba(220, 53, 69, 0.1)", "Medium": "rgba(255, 152, 0, 0.1)", "Low": "rgba(67, 160, 71, 0.1)"}.get(info["threat"], "rgba(255, 255, 255, 0.05)")
                st.markdown(
                    f'<div style="background:{bg}; border-radius:10px; padding:14px 18px; margin:6px 0; border-left:5px solid {tb};">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<div><span style="font-size:1.3rem;">{info["icon"]}</span> <strong style="font-size:1.05rem; color:#e2e8f0;">{info["name"]}</strong>'
                    f'<br/><span style="color:#a0aec0; font-size:0.9rem;">{info["desc"]}</span></div>'
                    f'<div style="text-align:right;"><span style="font-size:1.5rem; font-weight:800; color:{tb};">{info["size"]:,}</span>'
                    f'<br/><span style="color:#a0aec0; font-size:0.85rem;">{pct:.1f}% of traffic</span></div></div></div>',
                    unsafe_allow_html=True)

            with st.expander("🔬 Deep dive into a specific attacker type"):
                select_labels = [f"{cluster_info[cid]['icon']} {cluster_info[cid]['name']}" for cid in cluster_ids]
                selected_label = st.selectbox("Select type", select_labels)
                selected_cluster = cluster_ids[select_labels.index(selected_label)]
                cluster_rows = merged_df[merged_df["cluster_id"] == selected_cluster]
                numeric_cols = cluster_rows.select_dtypes(include="number").columns
                cluster_mean = cluster_rows[numeric_cols].mean()
                distances = ((cluster_rows[numeric_cols] - cluster_mean) ** 2).sum(axis=1).apply(math.sqrt)
                rep_sid = cluster_rows.loc[distances.idxmin(), "session_id"]
                st.markdown(f"**Example session:** `{rep_sid}`")
                raw_rep = sessions_by_id.get(rep_sid)
                if raw_rep and raw_rep.get("commands"):
                    for i, cmd in enumerate(raw_rep["commands"][:10], 1):
                        st.code(f"{i:>2}. {cmd[:100]}", language="bash")
                else:
                    st.info("No commands captured for this session type.")



# --- TAB 4: MAP ---

with tab_map:
    st.header("🌍 Attack Source Map")
    if baseline_df is None:
        missing_file_error("baseline_results.csv")
    elif geo_df is None or not GEOIP_DB_PATH.exists():
        st.warning("GeoLite2-Country.mmdb not found in ./data/. Download from MaxMind and place in data/ folder.")
    else:
        country_stats = (
            geo_df[(geo_df["country_code"] != "--") & (geo_df["lat"] != 0.0)]
            .groupby(["country", "country_code", "lat", "lon"])
            .agg(sessions=("session_id", "count"), unique_ips=("src_ip", "nunique"),
                 avg_duration=("duration_seconds", "mean"), anomalies=("anomaly_flag_iforest", "sum"))
            .reset_index()
        )
        if country_stats.empty:
            st.info("No IPs could be resolved to countries.")
        else:
            col_map_count, col_map_top, col_map_sessions = st.columns(3)
            top_c = country_stats.loc[country_stats["sessions"].idxmax()]
            col_map_count.metric("Countries seen", len(country_stats))
            col_map_top.metric("Top source", top_c["country"])
            col_map_sessions.metric("Sessions from top", f"{int(top_c['sessions']):,}")

            st.divider()
            max_s = country_stats["sessions"].max()
            country_stats["radius"] = ((country_stats["sessions"] / max(max_s, 1)) ** 0.5 * 800000).clip(lower=120000)

            def volume_colour(row):
                ratio = (row["sessions"] / max(max_s, 1)) ** 0.4
                if ratio < 0.33:
                    t = ratio / 0.33; return [255, int(235-(235-152)*t), int(59-59*t), 220]
                elif ratio < 0.66:
                    t = (ratio-0.33)/0.33; return [int(255-(255-229)*t), int(152-(152-57)*t), int(53*t), 220]
                else:
                    t = (ratio-0.66)/0.34; return [int(229-(229-183)*t), int(57-(57-28)*t), int(53-(53-28)*t), 220]

            country_stats["colour"] = country_stats.apply(volume_colour, axis=1)
            st.pydeck_chart(pdk.Deck(
                layers=[pdk.Layer("ScatterplotLayer", data=country_stats,
                    get_position=["lon", "lat"], get_radius="radius", get_fill_color="colour",
                    pickable=True, auto_highlight=True)],
                initial_view_state=pdk.ViewState(latitude=30, longitude=20, zoom=1.0, pitch=0),
                tooltip={"html": "<b>{country}</b> ({country_code})<br/>Sessions: {sessions}<br/>Unique IPs: {unique_ips}",
                        "style": {"backgroundColor": "#1a1a2e", "color": "white", "fontSize": "24px", "padding": "10px", "borderRadius": "6px"}},
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            ), height=500)

            st.divider()
            st.subheader("Sessions by country")
            display_geo = country_stats[["country", "country_code", "sessions", "unique_ips", "avg_duration"]].sort_values("sessions", ascending=False).reset_index(drop=True)
            display_geo["avg_duration"] = display_geo["avg_duration"].round(2)
            display_geo.columns = ["Country", "Code", "Sessions", "Unique IPs", "Avg Duration (s)"]
            st.dataframe(display_geo, use_container_width=True, hide_index=True)



# --- TAB 5: EXPLAIN ---

with tab_explain:
    st.header("💡 How This System Works")
    st.caption("Technical explanation for assessors and non-technical readers.")

    st.subheader("Model Performance Comparison")
    st.markdown("Three detection methods evaluated against 75 manually labelled sessions.")
    st.dataframe(pd.DataFrame({
        "Method": ["Rule Baseline", "Isolation Forest", "LOF"],
        "Precision": [0.547, 0.500, 0.429], "Recall": [0.784, 0.081, 0.081],
        "F1-Score": [0.644, 0.140, 0.136],
        "TP": [29, 3, 3], "FP": [24, 3, 4], "FN": [8, 34, 34], "TN": [14, 35, 34],
    }), use_container_width=True, hide_index=True)

    st.markdown(
        "**Why does the rule baseline outperform ML?** "
        "The dataset is dominated by short, automated bot probes. Isolation Forest treats this "
        "overwhelming bot traffic as \"normal\" since it is the majority, making it hard to flag bots. "
        "The rule baseline uses explicit thresholds that directly match bot patterns, "
        "giving it higher recall. This is a known limitation of unsupervised anomaly detection "
        "when the contamination rate is very high (Rehman et al., 2025)."
    )
    st.divider()

    st.subheader("Ablation Study (Isolation Forest)")
    st.dataframe(pd.DataFrame({
        "Feature Set": ["A: Timing only", "A+B: Timing + intent/auth", "A+B+C: Full features"],
        "Precision": [0.400, 0.367, 0.367], "Recall": [0.324, 0.297, 0.297],
        "F1-Score": [0.358, 0.328, 0.328], "Flagged": [30, 30, 30],
    }), use_container_width=True, hide_index=True)
    st.markdown("**Key finding:** Timing-only features achieved the best F1-score.")
    st.divider()

    st.subheader("System Glossary")
    with st.expander("🌲 Isolation Forest"):
        st.markdown("Unsupervised anomaly detection (Liu et al., 2008). Randomly partitions data with trees. Anomalies need fewer partitions. 200 estimators, contamination=0.1.")
    with st.expander("📏 Rule Baseline"):
        st.markdown("Heuristic classifier. Flags sessions where: duration < 30s AND commands >= 5, OR commands-per-second > 0.5.")
    with st.expander("🔍 Local Outlier Factor"):
        st.markdown("Density-based. Compares local density of each point to its neighbours. Sparser regions = outliers.")
    with st.expander("🔵 Behaviour Clustering"):
        st.markdown("Sessions grouped by K-Means into distinct attacker profiles based on 20 behavioural features.")
    with st.expander("📊 Feature Groups"):
        st.markdown("**A (Timing):** duration, cmds/sec, inter-command gaps, time to first command\n\n**B (Intent):** auth failures, recon, downloads, priv-esc, sensitive paths\n\n**C (Behaviour):** novelty rate, category switching, repeated commands")
    with st.expander("🌍 GeoIP"):
        st.markdown("MaxMind GeoLite2 offline database. No IP data sent externally. Bubble size = sessions, colour = volume.")
    with st.expander("🛡️ Safety & Ethics"):
        st.markdown("**Port separation:** Cowrie on port 22, admin SSH on port 2222 (restricted).\n\n**Egress filtering:** UFW blocks all outbound except DNS and HTTPS.\n\n**IP anonymisation:** SHA-256 hashed in reports. Raw IPs behind authentication only.\n\n**Ethics:** Type 2 classification approved by University of Salford.")