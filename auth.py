"""
auth.py - Authentication, user management and admin panel.

Handles login/logout, password hashing and persistent user storage.
Admin users can add, remove and manage accounts through the dashboard.
"""

import json
import hashlib
from pathlib import Path

import streamlit as st

from config import USERS_PATH, DEFAULT_USERS


# --- User storage ---

def load_users() -> dict:
    """Load users from JSON file, or create from defaults if missing."""
    if USERS_PATH.exists():
        with open(USERS_PATH, "r") as f:
            return json.load(f)
    else:
        save_users(DEFAULT_USERS)
        return DEFAULT_USERS.copy()


def save_users(users: dict):
    """Persist user accounts to JSON file."""
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    """Hash a password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()



# --- Login screen ---

def check_login() -> bool:
    """Display login form and manage authentication state.
    Returns True if the user is authenticated."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""

    if st.session_state.authenticated:
        return True

    st.markdown("""
    <div style="text-align: center; padding: 3rem 0 1rem 0;">
        <h1>🛡️ AI Honeypot Dashboard</h1>
        <p style="color: #6c757d; font-size: 1.1rem;">
            Threat Intelligence - University of Salford
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        with st.form("login_form"):
            st.markdown("#### Sign In")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                users = load_users()
                pw_hash = hash_password(password)
                uname = username.lower().strip()

                if uname in users and users[uname]["hash"] == pw_hash:
                    st.session_state.authenticated = True
                    st.session_state.username = uname
                    st.session_state.role = users[uname].get("role", "viewer")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    return False


def is_admin() -> bool:
    """Check if the current user has admin privileges."""
    return st.session_state.get("role", "") == "admin"



# --- Admin panel (sidebar) ---

def render_admin_panel():
    """Render admin controls in the sidebar for user management.
    Only visible to admin users."""
    if not is_admin():
        return

    st.sidebar.divider()
    st.sidebar.markdown("### 🔧 Admin Panel")

    users = load_users()

    # ---Show current users ---
    st.sidebar.markdown("**Current accounts:**")
    for uname, info in sorted(users.items()):
        role_badge = "🔑" if info.get("role") == "admin" else "👤"
        st.sidebar.markdown(f"{role_badge} `{uname}` ({info.get('role', 'viewer')})")

    st.sidebar.markdown("---")

    # --- Add new user ---
    st.sidebar.markdown("**Add new account**")
    with st.sidebar.form("add_user_form", clear_on_submit=True):
        new_user = st.text_input("Username", key="new_username", placeholder="e.g. analyst1")
        new_pass = st.text_input("Password", key="new_password", type="password")
        new_role = st.selectbox("Role", ["viewer", "admin"], key="new_role")
        add_submitted = st.form_submit_button("Add Account", use_container_width=True)

        if add_submitted:
            new_user = new_user.lower().strip()
            if not new_user or not new_pass:
                st.error("Username and password are required.")
            elif len(new_pass) < 4:
                st.error("Password must be at least 4 characters.")
            elif new_user in users:
                st.error(f"User '{new_user}' already exists.")
            else:
                users[new_user] = {
                    "hash": hash_password(new_pass),
                    "role": new_role,
                }
                save_users(users)
                st.success(f"Account '{new_user}' created.")
                st.rerun()

    # --- Remove user ---
    st.sidebar.markdown("**Remove account**")
    removable = [u for u in users if u != st.session_state.username]
    if removable:
        with st.sidebar.form("remove_user_form"):
            remove_user = st.selectbox("Select user to remove", removable, key="remove_select")
            remove_submitted = st.form_submit_button("Remove Account", use_container_width=True)

            if remove_submitted and remove_user:
                del users[remove_user]
                save_users(users)
                st.success(f"Account '{remove_user}' removed.")
                st.rerun()
    else:
        st.sidebar.caption("No removable accounts (you cannot remove yourself).")

    # --- Reset password ---
    st.sidebar.markdown("**Reset password**")
    resetable = list(users.keys())
    if resetable:
        with st.sidebar.form("reset_pass_form", clear_on_submit=True):
            reset_user = st.selectbox("Select user", resetable, key="reset_select")
            reset_pass = st.text_input("New password", type="password", key="reset_pass")
            reset_submitted = st.form_submit_button("Reset Password", use_container_width=True)

            if reset_submitted:
                if not reset_pass or len(reset_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    users[reset_user]["hash"] = hash_password(reset_pass)
                    save_users(users)
                    st.success(f"Password for '{reset_user}' updated.")