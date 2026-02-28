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
import streamlit as st

# ──────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────
DATA_DIR      = Path("./data")
BASELINE_PATH = DATA_DIR / "baseline_results.csv"
CLUSTERS_PATH = DATA_DIR / "clusters.csv"
SESSIONS_PATH = DATA_DIR / "sessions_raw.jsonl"
REPORT_PATH   = DATA_DIR / "daily_report.md"

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

# ──────────────────────────────────────────────
# Page header
# ──────────────────────────────────────────────
st.title("🛡️ SSH Honeypot — Security Dashboard")

tab_overview, tab_anomalies, tab_clusters, tab_explain = st.tabs(
    ["📊 Overview", "🚨 Anomalies", "🔵 Clusters", "💡 Explain"]
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
# TAB 4 — EXPLAIN
# ══════════════════════════════════════════════
with tab_explain:
    st.header("Explain Dashboard")

    # ── Narrative explanation button ─────────────
    if st.button("🔍 Explain today's dashboard", type="primary"):
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

            # Top 3 source IPs by session volume
            top_ips = baseline_df["src_ip"].value_counts().head(3)
            top_ip_str = ", ".join(f"`{ip}` ({n})" for ip, n in top_ips.items())

            # Highest-scoring anomaly
            max_row   = baseline_df.loc[baseline_df["anomaly_score_iforest"].idxmax()]
            max_sid   = max_row["session_id"]
            max_score = max_row["anomaly_score_iforest"]

            bullets = [
                f"**{total} SSH sessions** were observed today originating from "
                f"**{uniq_ips} unique source IPs** — indicating broad scanning activity.",

                f"**Signature-based rules** flagged **{rule_cnt}** sessions "
                f"({rule_cnt / total * 100:.1f}% of total), catching known bad patterns "
                f"such as failed auth and command-category abuse.",

                f"**Isolation Forest** identified **{if_cnt}** statistical anomalies "
                f"and **LOF** identified **{lof_cnt}** — sessions whose feature combinations "
                "fall well outside the normal distribution.",

                f"**Top source IPs** by session count: {top_ip_str}. "
                "Repeated connections from the same IP suggest persistent scanning or targeting.",

                f"The **highest-scoring anomaly** is session `{max_sid}` "
                f"(IForest score: **{max_score:.4f}**) — inspect it in the Anomalies tab.",

                f"**{dl_count}** session(s) contained file-download commands, "
                f"**{priv_count}** attempted privilege escalation, and "
                f"**{recon_sess}** performed system reconnaissance.",
            ]

            # Add cluster summary if available
            if merged_df is not None:
                cs = merged_df["cluster_id"].value_counts().sort_index()
                cluster_str = " | ".join(
                    f"Cluster {int(k)}: {v}" for k, v in cs.items()
                )
                bullets.append(
                    f"**Behaviour clustering** grouped sessions as: {cluster_str}. "
                    "Clusters with few sessions often represent distinct attacker toolkits."
                )

            bullets.append(
                "Use the **Anomalies tab** for per-session drill-down with command chains, "
                "and the **Clusters tab** to explore attacker behaviour groups."
            )

            for b in bullets:
                st.markdown(f"- {b}")

    st.divider()

    # ── Template-based Q&A ────────────────────
    st.subheader("Ask a question")
    st.caption(
        "Try: *How many anomalies were detected?*, *Which IPs were most active?*, "
        "*Were any files downloaded?*, *How are sessions clustered?*"
    )

    question = st.text_input(
        "Type your question about today's data…",
        placeholder="e.g. How many anomalies were detected?",
    )

    def answer_qa(q: str) -> str:
        """
        Return a template-based answer derived from the loaded dataframes.
        No external API is used — all answers come from the local data.
        """
        if baseline_df is None:
            return "Data not available — please ensure `baseline_results.csv` is in `./data/`."

        q = q.lower().strip()

        total      = len(baseline_df)
        uniq_ips   = baseline_df["src_ip"].nunique()
        rule_cnt   = int(baseline_df["baseline_rule_flag"].sum())
        if_cnt     = int(baseline_df["anomaly_flag_iforest"].sum())
        lof_cnt    = int(baseline_df["anomaly_flag_lof"].sum())

        # Match by keywords — order from most specific to most general
        if any(w in q for w in ["how many session", "total session", "number of session", "session count"]):
            return f"**{total}** sessions were recorded today."

        if any(w in q for w in ["unique ip", "source ip", "how many ip", "distinct ip"]):
            return f"**{uniq_ips}** unique source IPs were observed."

        if any(w in q for w in ["anomal"]):
            return (
                f"**{if_cnt}** sessions were flagged as anomalies by Isolation Forest "
                f"and **{lof_cnt}** by LOF. "
                f"Additionally, **{rule_cnt}** sessions were caught by signature-based rules."
            )

        if any(w in q for w in ["download", "payload", "malware", "file"]):
            dl = int((baseline_df["download_count"] > 0).sum())
            return f"**{dl}** session(s) contained file-download commands."

        if any(w in q for w in ["priv", "escalat", "root", "sudo"]):
            priv = int((baseline_df["priv_esc_count"] > 0).sum())
            return f"**{priv}** session(s) attempted privilege escalation."

        if any(w in q for w in ["recon", "enumerat", "scan", "uname", "cpuinfo"]):
            rec = int((baseline_df["recon_count"] > 0).sum())
            return f"**{rec}** session(s) performed reconnaissance commands."

        if any(w in q for w in ["cluster", "group", "behaviour", "behavior"]):
            if merged_df is not None:
                cs = merged_df["cluster_id"].value_counts().sort_index()
                parts = " | ".join(f"Cluster {int(k)}: {v} sessions" for k, v in cs.items())
                return (
                    f"Sessions are grouped into **{len(cs)} behaviour clusters**: {parts}."
                )
            return "Cluster data is not available."

        if any(w in q for w in ["rule", "signature", "flag"]):
            return (
                f"**{rule_cnt}** sessions were flagged by baseline signature rules "
                f"({rule_cnt / total * 100:.1f}% of all sessions)."
            )

        if any(w in q for w in ["top ip", "most active", "busiest", "frequent"]):
            top = baseline_df["src_ip"].value_counts().head(5)
            parts = " | ".join(f"`{ip}` ({n})" for ip, n in top.items())
            return f"Top source IPs: {parts}."

        if any(w in q for w in ["command", "cmd"]):
            avg_cmd = baseline_df["command_count"].mean()
            max_cmd = int(baseline_df["command_count"].max())
            return (
                f"Average commands per session: **{avg_cmd:.1f}**; "
                f"maximum in a single session: **{max_cmd}**."
            )

        if any(w in q for w in ["duration", "long", "short", "time"]):
            avg_dur = baseline_df["duration_seconds"].mean()
            max_dur = baseline_df["duration_seconds"].max()
            return (
                f"Average session duration: **{avg_dur:.2f} s**; "
                f"longest session: **{max_dur:.2f} s**."
            )

        if any(w in q for w in ["auth", "password", "credential", "brute"]):
            fail_sess = int((baseline_df["auth_fail_count"] > 0).sum())
            avg_fail  = baseline_df["auth_fail_count"].mean()
            return (
                f"**{fail_sess}** session(s) had at least one authentication failure. "
                f"Average failures per session: **{avg_fail:.2f}**."
            )

        if any(w in q for w in ["sensitive", "path", "etc", "shadow", "passwd"]):
            sens = int((baseline_df["sensitive_path_count"] > 0).sum())
            return f"**{sens}** session(s) accessed or attempted to access sensitive file paths."

        return (
            "I don't have a template answer for that yet. "
            "Try asking about: **sessions**, **IPs**, **anomalies**, **downloads**, "
            "**privilege escalation**, **recon**, **clusters**, **rules**, "
            "**commands**, **duration**, or **authentication failures**."
        )

    if question:
        st.info(answer_qa(question))
