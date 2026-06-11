"""
models.py - Detection logic, cluster classification and threat intelligence.
"""

from datetime import datetime, timezone

import pandas as pd


def epoch_to_utc(ts: float) -> datetime:
    """Convert a Unix epoch timestamp to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def why_flagged(row: pd.Series) -> list:
    """Generate human-readable bullet points explaining why a session
    was flagged as suspicious, based on its feature values.

    Args:
        row: A single row from the baseline DataFrame.

    Returns:
        List of markdown-formatted explanation strings.
    """
    bullets = []

    duration = row.get("duration_seconds", 0)
    if duration < 3:
        bullets.append(
            f"**Very short session** ({duration:.2f}s) - typical of automated probes "
            "that connect, attempt auth and disconnect immediately."
        )

    if row.get("had_commands", 1) == 0 and duration >= 3:
        bullets.append(
            "**No commands executed** despite lasting several seconds "
            "- common pattern in banner-grabbing scanners."
        )

    auth_fail = int(row.get("auth_fail_count", 0))
    if auth_fail > 0:
        bullets.append(
            f"**{auth_fail} authentication failure(s)** - consistent with "
            "brute-force or credential-stuffing behaviour."
        )

    if int(row.get("download_count", 0)) > 0:
        bullets.append(
            f"**File download detected** ({int(row['download_count'])} event(s)) "
            "- attacker may have retrieved a payload or exfiltrated data."
        )

    if int(row.get("priv_esc_count", 0)) > 0:
        bullets.append(
            f"**Privilege-escalation commands** ({int(row['priv_esc_count'])} event(s)) "
            "- attempts to gain root/sudo access."
        )

    if int(row.get("sensitive_path_count", 0)) > 0:
        bullets.append(
            f"**Sensitive path access** ({int(row['sensitive_path_count'])} event(s)) "
            "- e.g. /etc/passwd, /etc/shadow, SSH keys."
        )

    cmds_per_sec = row.get("cmds_per_sec_x", 0)
    if cmds_per_sec > 1.0:
        bullets.append(
            f"**High command rate** ({cmds_per_sec:.2f} cmds/s) - "
            "scripted, non-interactive execution."
        )

    novelty = row.get("novelty_rate", 0)
    if novelty > 0.5:
        bullets.append(
            f"**High novelty rate** ({novelty:.2f}) - majority of commands "
            "were unseen in the baseline, suggesting novel tooling."
        )

    cat_switch = row.get("category_switch_rate", 0)
    if cat_switch > 0.3:
        bullets.append(
            f"**Frequent category switching** ({cat_switch:.2f}) - "
            "rapid alternation between recon, download and escalation."
        )

    recon = int(row.get("recon_count", 0))
    if recon >= 3:
        bullets.append(
            f"**Heavy reconnaissance** ({recon} commands) - "
            "extensive system enumeration."
        )

    if not bullets:
        bullets.append(
            "Flagged as a **statistical outlier** by Isolation Forest - "
            "the combination of features falls outside the normal distribution."
        )

    return bullets


def classify_cluster(cluster_rows: pd.DataFrame) -> dict:
    """Analyse average features of a cluster and assign a human-readable
    name, icon, description and threat level.

    The classification uses a priority-ordered rule chain. Each cluster
    is matched against behavioural patterns derived from domain knowledge
    of SSH attack types.

    Args:
        cluster_rows: DataFrame containing all sessions in one cluster.

    Returns:
        Dict with keys: name, icon, desc, threat.
    """
    avg = cluster_rows.mean(numeric_only=True)

    dur = avg.get("duration_seconds", 0)
    cmds = avg.get("command_count", 0)
    cps = avg.get("cmds_per_sec_x", avg.get("cmds_per_sec", 0))
    dl = avg.get("download_count", 0)
    priv = avg.get("priv_esc_count", 0)
    recon = avg.get("recon_count", 0)
    auth_fail = avg.get("auth_fail_count", 0)
    had_cmds = avg.get("had_commands", 0)
    novelty = avg.get("novelty_rate", 0)
    cat_switch = avg.get("category_switch_rate", 0)
    sensitive = avg.get("sensitive_path_count", 0)

    # HIGH THREAT
    if dl > 0.1:
        return {"name": "Payload Downloaders", "icon": "⬇️",
                "desc": "Sessions retrieving external payloads via wget/curl/tftp - botnet recruitment or malware staging.",
                "threat": "High"}
    if priv > 0.1:
        return {"name": "Privilege Escalators", "icon": "🔓",
                "desc": "Attackers attempting sudo/su/chmod - trying to gain root access to the system.",
                "threat": "High"}
    if sensitive > 0.1:
        return {"name": "Data Harvesters", "icon": "📂",
                "desc": "Sessions accessing /etc/passwd, SSH keys and other sensitive files - credential theft or data exfiltration.",
                "threat": "High"}
    if recon > 2 and dur > 20:
        return {"name": "Interactive Intruders", "icon": "🕵️",
                "desc": "Long exploratory sessions with heavy system enumeration - likely human operators, not bots.",
                "threat": "High"}

    # MEDIUM THREAT
    if cat_switch > 0.2 and cmds > 2:
        return {"name": "Multi-Stage Attackers", "icon": "🔗",
                "desc": "Switching between recon, download and escalation - coordinated automated attack chains.",
                "threat": "Medium"}
    if recon > 0.5:
        return {"name": "Reconnaissance Scanners", "icon": "🔍",
                "desc": "Sessions running system enumeration commands (uname, whoami, /proc) - gathering intelligence.",
                "threat": "Medium"}
    if dur > 10 and cmds > 1:
        return {"name": "Slow & Methodical", "icon": "🐌",
                "desc": "Longer sessions with deliberate command execution - may be human operators or advanced bots.",
                "threat": "Medium"}

    # LOW THREAT
    if dur < 3 and cmds >= 1:
        return {"name": "Automated Bot Swarm", "icon": "🤖",
                "desc": "Rapid automated sessions executing scripted commands in under 3 seconds - the bulk of internet-wide scanning.",
                "threat": "Low"}
    if auth_fail > 1 and had_cmds < 0.5:
        return {"name": "Credential Stuffers", "icon": "🔑",
                "desc": "Multiple failed login attempts with no shell interaction - brute-force password attacks.",
                "threat": "Low"}
    if dur < 5 and cmds < 1:
        return {"name": "Banner Grabbers", "icon": "👀",
                "desc": "Ultra-short connections with no commands - scanning for SSH version strings and service banners.",
                "threat": "Low"}

    # FALLBACKS
    if dur < 5:
        return {"name": "Quick-Hit Probes", "icon": "💨",
                "desc": "Brief automated connections - testing if the service is alive before moving on.",
                "threat": "Low"}
    if cmds > 0:
        return {"name": "Exploratory Sessions", "icon": "🧭",
                "desc": "Sessions with some command activity that don't fit standard attack patterns - potentially novel techniques.",
                "threat": "Medium"}

    return {"name": "Silent Connections", "icon": "🔇",
            "desc": "Sessions with minimal activity - likely connection tests or failed automation scripts.",
            "threat": "Low"}


def generate_actionable_intel(df: pd.DataFrame, geo_df: pd.DataFrame = None) -> list:
    """Produce actionable threat intelligence recommendations
    based on the current dataset.

    This function analyses the honeypot data and generates colour-coded
    defensive recommendations (critical/warning/info). It is the primary
    'so what?' output of the threat intelligence system.

    Args:
        df: The baseline results DataFrame.
        geo_df: Optional GeoIP-enriched DataFrame for geographic analysis.

    Returns:
        List of dicts with keys: level, title, detail.
    """
    actions = []
    total = len(df)

    # 1. Top offending IPs
    top_ips = df["src_ip"].value_counts().head(5)
    top_ip_pct = top_ips.iloc[0] / total * 100 if len(top_ips) > 0 else 0

    if top_ip_pct > 20:
        actions.append({
            "level": "critical",
            "title": "Block dominant attacker IP",
            "detail": (
                f"A single IP accounts for **{top_ips.iloc[0]:,}** sessions "
                f"(**{top_ip_pct:.1f}%** of all traffic). "
                f"Recommended action: add `{top_ips.index[0]}` to firewall deny list. "
                f"This volume indicates persistent automated scanning."
            ),
        })

    # 2. Credential attacks
    if "auth_fail_count" in df.columns:
        brute_force = df[df["auth_fail_count"] > 3]
        if len(brute_force) > 0:
            actions.append({
                "level": "warning",
                "title": "Brute-force credential attacks detected",
                "detail": (
                    f"**{len(brute_force)}** sessions had more than 3 failed auth attempts. "
                    "Recommended action: enforce key-based SSH authentication, disable password login, "
                    "and deploy fail2ban with aggressive thresholds."
                ),
            })

    # 3. Download activity
    downloads = df[df["download_count"] > 0]
    if len(downloads) > 0:
        actions.append({
            "level": "critical",
            "title": f"{len(downloads)} sessions with file download activity",
            "detail": (
                "Attackers attempted to download payloads (wget, curl, tftp). "
                "Recommended action: inspect captured files for malware signatures, "
                "block C2 domains at DNS level and alert SOC for payload analysis."
            ),
        })

    # 4. Privilege escalation
    priv_esc = df[df["priv_esc_count"] > 0]
    if len(priv_esc) > 0:
        actions.append({
            "level": "warning",
            "title": f"{len(priv_esc)} sessions with privilege escalation attempts",
            "detail": (
                "Commands like `sudo`, `su`, `chmod 777` were detected. "
                "Recommended action: audit sudoers configuration, enforce least-privilege access, "
                "and monitor for unauthorised privilege changes."
            ),
        })

    # 5. Geographic concentration
    if geo_df is not None:
        country_counts = geo_df.groupby("country")["session_id"].count().sort_values(ascending=False)
        if len(country_counts) > 0:
            top_country = country_counts.index[0]
            top_count = country_counts.iloc[0]
            top_pct = top_count / total * 100
            if top_pct > 30:
                actions.append({
                    "level": "info",
                    "title": f"Geographic concentration: {top_pct:.0f}% from {top_country}",
                    "detail": (
                        f"**{top_count:,}** sessions originated from **{top_country}**. "
                        "If this region is outside your user base, consider geo-blocking or "
                        "rate-limiting at the network perimeter."
                    ),
                })

    # 6. Bot ratio
    short_sessions = df[df["duration_seconds"] < 3]
    bot_pct = len(short_sessions) / total * 100 if total > 0 else 0
    if bot_pct > 90:
        actions.append({
            "level": "info",
            "title": f"Automated traffic dominance ({bot_pct:.1f}% bot-like)",
            "detail": (
                f"**{len(short_sessions):,}** out of {total:,} sessions lasted under 3 seconds, "
                "indicating automated scanning tools. "
                "Recommended action: implement connection rate-limiting and "
                "tarpit slow-response techniques to waste bot resources."
            ),
        })

    # 7. Superhuman speed
    if "cmds_per_sec_x" in df.columns:
        fast = df[df["cmds_per_sec_x"] > 2.0]
        if len(fast) > 10:
            actions.append({
                "level": "warning",
                "title": f"{len(fast)} sessions with superhuman command speed",
                "detail": (
                    "Commands executed at >2 per second, impossible for a human operator. "
                    "These are scripted attack chains. "
                    "Recommended action: deploy keystroke timing analysis for real-time bot detection."
                ),
            })

    return actions