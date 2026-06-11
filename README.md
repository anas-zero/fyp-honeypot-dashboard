# AI-Enhanced SSH Honeypot for Threat Intelligence

A live SSH honeypot deployed on AWS EC2, paired with a Python pipeline benchmarking three detection approaches against manually-labelled ground truth. Captured 65,172 real attack sessions from 850 unique IPs across 40 countries over six weeks of continuous deployment.

BSc Computer Science with Cyber Security, Final Year Project, University of Salford, 2025/26.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Finding: Contamination Inversion](#key-finding-contamination-inversion)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dashboard Features](#dashboard-features)
- [Attacker Clusters](#attacker-clusters)
- [Results](#results)
- [Detection Performance](#detection-performance)
- [Setup Instructions](#setup-instructions)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project explores whether unsupervised machine learning can outperform rule-based detection on real-world SSH honeypot data. A Cowrie honeypot was deployed on AWS EC2 behind two firewall layers and ran continuously for six weeks. It accepted connections on the public-facing port 22 while admin SSH was kept on a restricted port 2222.

A Python pipeline runs daily via cron at 09:00 UTC. It parses Cowrie's JSON logs, engineers 20 behavioural features per session and runs three detection models in parallel: a rule-based heuristic baseline, Isolation Forest and Local Outlier Factor. Outputs are merged on session ID and consumed by an interactive Streamlit dashboard that turns raw findings into defensive recommendations.

This repository contains the dashboard application code shown for reference. The captured attack data and authentication credentials are kept private to protect the IPs of compromised devices observed during the deployment.

### Aims

- Capture live SSH attack traffic at scale on a public-facing honeypot
- Engineer behavioural features sufficient to distinguish bot from non-bot sessions
- Benchmark unsupervised machine learning against rule-based detection
- Present findings through an interactive dashboard accessible to both technical and non-technical audiences
- Generate concrete defensive recommendations grounded in observed attacker behaviour

---

## Key Finding: Contamination Inversion

The headline result is a counter-intuitive performance pattern that emerges when malicious traffic dominates a dataset.

Unsupervised anomaly detection algorithms such as Isolation Forest and Local Outlier Factor assume that anomalies are statistically rare. They learn what is "normal" from the majority of the data and flag everything that deviates from that majority.

In this dataset, 95.8% of all sessions were automated bot traffic (sessions under three seconds, executing scripted commands). When the ML models trained on this data, they learned bot behaviour as the norm and began flagging the rare non-bot sessions (recon scanners, exploratory traffic) as anomalies, while letting the obvious bot attacks pass as normal.

A 12-line rule-based baseline outperformed two well-known ML algorithms by more than four times in F1 score. The rule baseline achieved F1 of 0.64. Isolation Forest and Local Outlier Factor both achieved F1 of 0.14.

This finding builds on Rehman et al. (2025), who documented the mechanism on synthetic data. This project demonstrates the same effect on real, internet-facing honeypot traffic and proposes two-stage detection (rule-based filtering followed by ML on the residual) as a practical solution for adversarial environments where attack traffic dominates.

---

## Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│              │      │              │      │              │      │              │
│  Attackers   │────▶│   Cowrie     │────▶│ AI Pipeline  │────▶│  Dashboard   │
│  (port 22)   │      │  (Docker)    │      │ (3 detectors)│      │ (Streamlit)  │
│              │      │              │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                            AWS EC2                              Local machine
                       Ubuntu 22.04 +                          (CSVs synced via SCP)
                       UFW + Security
                          Groups
```

Cowrie SSH honeypot runs in a Docker container on AWS EC2 (Ubuntu 22.04, t3.micro). UFW blocks all outbound traffic except DNS, HTTP and HTTPS. AWS Security Groups restrict admin SSH on port 2222 to a single source IP. Public Cowrie on port 22 accepts all incoming connections.

A bash script (`run_pipeline.sh`) executes daily via cron. It concatenates rotated logs, parses sessions, engineers features and runs all three detectors in parallel. Results are merged by session ID and clustered using K-Means.

A locally-run Streamlit dashboard reads the processed CSVs and renders five tabs covering overview metrics, anomaly inspection, attacker profiling, geographic mapping and methodology explanation.

---

## Tech Stack

### Languages and Runtimes
- Python 3.10+ for the pipeline, ML and dashboard
- Bash for automation and scheduling
- PowerShell for local data sync from AWS

### Honeypot
- Cowrie, a medium-interaction SSH and Telnet honeypot
- Docker for container isolation

### Machine Learning and Data Processing
- scikit-learn (Isolation Forest, Local Outlier Factor, K-Means)
- pandas for feature engineering and merging
- NumPy for numerical operations

### Visualisation
- Streamlit for the dashboard framework
- Plotly for interactive charts and maps
- MaxMind GeoLite2 for offline GeoIP lookup

### Infrastructure and Operations
- AWS EC2 (Ubuntu 22.04, t3.micro)
- AWS Elastic IP for a static public address
- UFW host-level firewall
- AWS Security Groups for network-level firewall
- Cron for scheduled pipeline execution
- Git and GitHub for version control

---

## Dashboard Features

The Streamlit dashboard has five tabs, each designed to answer a different question.

### Overview
The landing page. Shows top-line statistics (sessions captured, unique IPs, countries observed, percentage of bot traffic), a traffic breakdown by behavioural category, a threat level bar showing the proportion of flagged versus normal sessions, the top five most dangerous sessions with expandable command chains and the threat intelligence panel with colour-coded recommendations.

### Anomalies
Every session ranked by anomaly score with filters for which model flagged it (Rule, Isolation Forest, LOF). Each entry expands to show the full command chain, attacker metadata and reasoning. Allows direct visual comparison of model outputs and demonstrates contamination inversion.

### Attacker Profiles
Top 10 most active attackers with auto-classified behaviour types. Most-targeted credentials (usernames and passwords). Attacker type breakdown showing the four K-Means clusters with their percentages and characteristics.

### Map
A GeoIP bubble map of attack origins powered by the offline MaxMind GeoLite2 database. Bubbles are sized by session count and coloured on a yellow-to-dark-red scale by volume. Lookups happen locally with no external API calls.

### Explain
Methodology and reasoning made accessible to non-technical readers. Contains a model comparison table (Rule vs Isolation Forest vs LOF), the ablation study results and seven expandable glossary sections explaining every component of the AI pipeline.

---

## Attacker Clusters

K-Means clustering with k=4 was applied to the full dataset using a subset of behavioural features. The cluster names below are assigned dynamically by the dashboard based on each cluster's average feature values.

| Cluster | Sessions | Share | Description |
|---------|---------:|------:|-------------|
| Automated Bot Swarm | 64,949 | 99.70% | Rapid automated sessions executing scripted commands in under 3 seconds. The bulk of internet-wide scanning activity. |
| Reconnaissance Scanners | 101 | 0.16% | Sessions running system enumeration commands (`uname`, `whoami`, `/proc`). Slower and more deliberate than the bot swarm. |
| Exploratory Sessions | 93 | 0.14% | Sessions with command activity that does not fit standard attack patterns. Potentially novel techniques or human-led activity. |
| Payload Downloaders | 29 | 0.04% | Sessions retrieving external payloads via `wget`, `curl` or `tftp`. Highest-severity attacks despite being the smallest cluster. |

The dominance of the Automated Bot Swarm cluster (99.7%) is the structural cause of contamination inversion. From the perspective of an unsupervised algorithm, this cluster IS the dataset.

---

## Results

| Metric | Value |
|--------|------:|
| Sessions captured | 65,172 |
| Unique source IPs | 850 |
| Countries observed | 40 |
| Days deployed | 42 (6 weeks) |
| Pipeline uptime | 99.9% |
| Bot traffic share | 95.8% |
| Top 3 countries share | 94.6% of all traffic |

### Top Source Countries

| Rank | Country | Sessions | Share |
|-----:|---------|---------:|------:|
| 1 | Indonesia | 39,657 | 60.8% |
| 2 | Brazil | 20,342 | 31.2% |
| 3 | United States | 1,657 | 2.5% |

Indonesia and Brazil together account for 92% of all traffic, despite originating from just six unique IPs combined. The United States contribution is more distributed, with 300 unique IPs producing relatively low session counts each.

### Most Targeted Credentials

The most-attempted username was `root` (62,379 attempts), followed by `admin` (420), `postgres` (44) and `user` (40). The most-attempted passwords were `123456`, `password`, `admin`, `root` and `12345`. This pattern is consistent with default credential stuffing against IoT devices and network appliances.

---

## Detection Performance

All three models were evaluated against 75 manually-labelled ground-truth sessions, deliberately balanced at 37 bot and 38 non-bot. Labels were applied using a 7-point behavioural classification system applied blind to each session.

| Model | Type | Precision | Recall | F1-Score | Verdict |
|-------|------|----------:|-------:|---------:|---------|
| Rule Baseline | Heuristic | 0.55 | 0.78 | 0.64 | Best performer |
| Isolation Forest | Unsupervised ML | 0.50 | 0.08 | 0.14 | Affected by contamination inversion |
| Local Outlier Factor | Density-based ML | 0.43 | 0.08 | 0.14 | Affected by contamination inversion |

### Rule Baseline Logic

```python
# Rule A: short session with many commands
flag_a = (duration_seconds < 30) & (command_count >= 5)

# Rule B: high command rate
flag_b = (cmds_per_sec > 0.5)

# Final flag: A or B
baseline_rule_flag = flag_a | flag_b
```

Twelve lines of code total. Targets known bot signatures regardless of class distribution. The 30-second threshold is deliberately wider than the observed 3-second bot mean, giving the rule headroom to catch slightly slower variants without missing the dominant bot signature.

### Ablation Study

A feature-group ablation isolated which engineered features contributed most to detection performance. Timing-only features (duration and command rate) achieved the best Isolation Forest F1 of 0.358. Adding behavioural and identity features (command type counts, credentials used, download flags) did not improve performance, suggesting that bot-versus-non-bot separability is encoded primarily in timing patterns rather than command content.

---

## Setup Instructions

### Prerequisites

- Python 3.10 or newer
- pip

### Code Reference

The dashboard code is shown here for code-quality review and as a portfolio reference. To protect the privacy of the IPs of compromised devices observed during deployment, the captured dataset and authentication credentials are not included in this public repository.

To run the dashboard against your own honeypot data, you would need:

- A `data/` folder containing parsed sessions, baseline results, anomaly scores and K-Means cluster assignments (CSV format)
- A hashed credentials file (`data/users.json`) with your own dashboard logins
- The MaxMind GeoLite2 country database (free, available from MaxMind)

The pipeline that produces these files (parsing Cowrie JSON logs, engineering features, running the three detectors, clustering with K-Means) is documented inline within the codebase. The Cowrie honeypot itself is open source and available at https://github.com/cowrie/cowrie.

---

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── app.py                             # Streamlit dashboard entry point
├── auth.py                            # Login system (SHA-256 hashed credentials)
├── config.py                          # Dashboard configuration
├── data_loader.py                     # CSV loading and caching
├── geoip_lookup.py                    # MaxMind GeoLite2 wrapper
├── models.py                          # Cluster classification logic
├── .gitignore                         # Excludes data folder, SSH keys, Python cache
└── LICENSE                            # MIT License
```

Note: the captured dataset and hashed credentials referenced in the code are not included in this public repository for security and privacy reasons. The pipeline scripts that ran on AWS (`parse_cowrie.py`, `detect_anomalies.py`, etc.) are also kept private to avoid leaking server-specific configuration.

---

## Limitations

This project is honest about what it does and does not establish.

All 75 labelled sessions were classified by one researcher using a fixed checklist applied blind. A second rater and Cohen's kappa would strengthen the methodology and is proposed as future work.

The labelling distinguishes bot from non-bot, not malicious from benign. Everything that hits a honeypot is non-legitimate by definition, so the meaningful distinction is operational. Extending to a malicious-versus-benign classification would require a different ground truth scheme.

The honeypot was deployed in eu-north-1 (Stockholm) only. A multi-region deployment would help establish whether the geographic distribution generalises.

A six-week capture window was the maximum within project constraints. A longer capture would reveal seasonal patterns and attacker campaign cycles.

---

## Acknowledgements

Supervised by Dr Nashnush (primary supervisor) and Dr Bryant (secondary supervisor) at the University of Salford. Their guidance throughout the project was deeply appreciated.

This project builds on the contamination inversion mechanism documented by Rehman et al. (2025) on synthetic data. The contribution of this work is demonstrating the same effect on real, internet-facing honeypot traffic and proposing two-stage detection as a practical mitigation.

---

## License

Released under the MIT License. See `LICENSE` for details.

The MaxMind GeoLite2 database is included under its own licence. See https://dev.maxmind.com/geoip/geolite2-free-geolocation-data for terms.

---

## Contact

Anas Hussein
BSc (Hons) Computer Science with Cyber Security
University of Salford Manchester, 2025/26

Connect on LinkedIn: [linkedin.com/in/anas-y-hussein](https://www.linkedin.com/in/anas-y-hussein/)
