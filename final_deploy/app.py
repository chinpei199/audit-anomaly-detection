import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import hashlib
import joblib
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from datetime import datetime
import base64

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audit Anomaly Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

USERS_FILE = "users.json"
AUDIT_ACTIONS_FILE = "audit_actions.json"
EVIDENCE_DIR = "evidence"

# ─── Utility ─────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_audit_actions():
    if os.path.exists(AUDIT_ACTIONS_FILE):
        with open(AUDIT_ACTIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_audit_actions(actions: dict):
    with open(AUDIT_ACTIONS_FILE, "w") as f:
        json.dump(actions, f, indent=2, default=str)

def save_evidence(anomaly_idx, uploaded_file):
    """Save uploaded file to evidence/<anomaly_idx>_<filename> and return the saved path."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    safe_name = f"anomaly_{anomaly_idx}_{uploaded_file.name}"
    filepath  = os.path.join(EVIDENCE_DIR, safe_name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

def authenticate(email: str, password: str):
    users = load_users()
    hashed = hash_password(password)
    for uid, udata in users.items():
        stored = udata.get("password", "")
        # Accept plain-text OR pre-hashed passwords stored in users.json
        if udata.get("email") == email and (stored == password or stored == hashed):
            return uid, udata
    return None, None

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (f'<a href="data:file/csv;base64,{b64}" download="{filename}" '
            f'style="color:#2e7d32;font-weight:600;text-decoration:none;'
            f'border:1.5px solid #2e7d32;padding:8px 18px;border-radius:6px;">'
            f'Download CSV</a>')

def safe_kde(values, x_range):
    """Compute KDE safely; returns None if data is degenerate."""
    arr = np.asarray(values, dtype=float)
    # Need enough unique values and non-zero variance
    if len(arr) < 3:
        return None
    if np.std(arr) == 0:
        return None
    unique = np.unique(arr)
    if len(unique) < 2:
        return None
    try:
        kde = stats.gaussian_kde(arr, bw_method='silverman')
        return kde(x_range)
    except Exception:
        # Fallback: try Scott's rule with added jitter
        try:
            jittered = arr + np.random.default_rng(0).normal(0, np.std(arr) * 1e-6, size=len(arr))
            kde = stats.gaussian_kde(jittered, bw_method='scott')
            return kde(x_range)
        except Exception:
            return None

# ─── CSS ─────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #f0f4f0; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Hide sidebar collapse/expand arrow button permanently */
    [data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #e8f5e9 !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        padding: 6px 0 !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

    /* Sign Out button — white bg, dark green text */
    [data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        color: #1b5e20 !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div {
        color: #1b5e20 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #e8f5e9 !important;
    }



    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border: 1px solid #e8f5e9;
        margin-bottom: 12px;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-left: 4px solid #43a047;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card.red    { border-left-color: #e53935; }
    .metric-card.orange { border-left-color: #fb8c00; }
    .metric-card.blue   { border-left-color: #1e88e5; }
    .metric-card.teal   { border-left-color: #00897b; }
    .metric-card.purple { border-left-color: #8e24aa; }
    .metric-card-value {
        font-size: 1.9rem; font-weight: 700; color: #1b5e20; line-height: 1.1;
    }
    .metric-card-label {
        font-size: 0.72rem; color: #666; text-transform: uppercase;
        letter-spacing: 0.06em; margin-top: 4px; font-weight: 500;
    }

    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #1b5e20;
        margin-bottom: 12px; padding-bottom: 6px;
        border-bottom: 2px solid #c8e6c9;
    }
    .badge-open {
        background: #fff3e0; color: #e65100;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; border: 1px solid #ffcc80;
    }
    .badge-closed {
        background: #e8f5e9; color: #2e7d32;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; border: 1px solid #a5d6a7;
    }
    .page-title { font-size: 1.5rem; font-weight: 700; color: #1b5e20; margin-bottom: 4px; }
    .page-subtitle { font-size: 0.85rem; color: #666; margin-bottom: 20px; }
    .green-divider { border: none; border-top: 2px solid #c8e6c9; margin: 20px 0; }
    .info-box {
        background: #e8f5e9; border-left: 4px solid #43a047;
        border-radius: 6px; padding: 12px 16px;
        font-size: 0.88rem; color: #2e7d32; margin-bottom: 16px;
    }

    /* Main area buttons */
    .stButton > button {
        background: #2e7d32; color: white; border: none;
        border-radius: 7px; font-family: 'DM Sans', sans-serif;
        font-weight: 600; font-size: 0.88rem; padding: 8px 20px;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #1b5e20; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-weight: 600; }

    </style>
    """, unsafe_allow_html=True)



# ─── Login ────────────────────────────────────────────────────────────────────


def login_page():
    
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 50%, #1b5e20 100%) !important;
    }
    .block-container { padding-top: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
        st.markdown("""
       
        <div style='color:#fff;font-size:1.65rem;font-weight:700;text-align:center;margin-bottom:4px;'>
            Member Login
        </div>
        <div style='color:rgba(255,255,255,0.78);font-size:0.82rem;text-align:center;
                    margin-bottom:28px;letter-spacing:0.04em;'>
            Audit Anomaly Detection System
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            email    = st.text_input("", placeholder="email",    label_visibility="collapsed")
            password = st.text_input("", placeholder="password", type="password", label_visibility="collapsed")
            submitted = st.form_submit_button("SIGN IN", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                uid, udata = authenticate(email, password)
                if uid:
                    st.session_state["logged_in"]  = True
                    st.session_state["user_id"]    = uid
                    st.session_state["user_data"]  = udata
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:16px 0 8px 0;'>
            <div style='font-size:1.15rem;font-weight:700;color:#fff;'>Nation Bank Bhd</div>
            <div style='font-size:0.72rem;color:rgba(255,255,255,0.6);
                        letter-spacing:0.08em;text-transform:uppercase;'>
                Audit Anomaly Detection
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        udata = st.session_state.get("user_data", {})
        st.markdown(f"""
        <div style='padding:8px 0 14px 0;'>
            <div style='font-size:0.78rem;color:rgba(255,255,255,0.55);
                        text-transform:uppercase;letter-spacing:0.07em;'>Signed in as</div>
            <div style='font-size:0.92rem;font-weight:600;color:#fff;margin-top:3px;'>
                {udata.get('name','User')}
            </div>
            <div style='font-size:0.75rem;color:rgba(255,255,255,0.65);'>
                {udata.get('email','')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio("Navigation", ["Prediction", "Analysis", "User Profile"],
                        label_visibility="collapsed")

        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    return page

# ─── Load Models ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        model    = joblib.load("isolation_forest_model.pkl")
        features = joblib.load("features.pkl")
        scaler   = joblib.load("scaler.pkl")
        return model, features, scaler
    except Exception:
        return None, None, None

# ─── Prediction Page ─────────────────────────────────────────────────────────

def prediction_page():
    st.markdown("<div class='page-title'>Anomaly Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Upload your audit dataset (CSV) to detect anomalies.</div>",
                unsafe_allow_html=True)

    model, features, scaler = load_models()
    if model is None:
        st.warning("Model files not found. Place isolation_forest_model.pkl, features.pkl, "
                   "and scaler.pkl in the app directory.")
        return

    # ── File uploader — detect NEW upload and clear old state ──────────────
    uploaded = st.file_uploader("Upload Audit Dataset (CSV)", type=["csv"], key="pred_upload")

    if uploaded is not None:
        # If a brand-new file is uploaded (different name or first upload), reset everything
        prev_name = st.session_state.get("_uploaded_filename", None)
        if prev_name != uploaded.name:
            # Clear old results and audit actions for a fresh start
            for key in ["pred_df_raw", "pred_df_result"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear the persistent audit_actions.json and in-memory copy
            save_audit_actions({})
            st.session_state["audit_actions"] = {}
            st.session_state["_uploaded_filename"] = uploaded.name

        if "pred_df_raw" not in st.session_state:
            df = pd.read_csv(uploaded)
            st.session_state["pred_df_raw"] = df

        df = st.session_state["pred_df_raw"].copy()

        st.markdown("<div class='section-header'>Dataset Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")

        if st.button("Run Anomaly Detection", type="primary"):
            with st.spinner("Running detection..."):
                try:
                    if "Amount" in df.columns:
                        df["log_Amount"] = np.log1p(df["Amount"])

                    available_features = [f for f in features if f in df.columns]
                    if not available_features:
                        st.error("None of the required model features found in the dataset.")
                        return

                    X        = df[available_features].copy()
                    X_scaled = scaler.transform(X)
                    preds    = model.predict(X_scaled)
                    scores   = model.decision_function(X_scaled)

                    df["_prediction"]    = preds
                    df["_anomaly_score"] = scores
                    df["_is_anomaly"]    = (preds == -1)
                    df["_anomaly_label"] = df["_is_anomaly"].map({True: "Anomaly", False: "Normal"})

                    st.session_state["pred_df_result"] = df
                    # Start fresh audit actions for this new detection run
                    st.session_state["audit_actions"] = {}
                    save_audit_actions({})
                    st.success("Detection complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during detection: {e}")

    # ── Results ──────────────────────────────────────────────────────────────
    if "pred_df_result" in st.session_state:
        df            = st.session_state["pred_df_result"]
        audit_actions = st.session_state.get("audit_actions", {})

        anomalies = df[df["_is_anomaly"] == True].copy()
        normal    = df[df["_is_anomaly"] == False].copy()

        total        = len(df)
        n_normal     = len(normal)
        n_anomaly    = len(anomalies)
        anomaly_rate = (n_anomaly / total * 100) if total > 0 else 0

        # Only count closed if user explicitly set status = "closed"
        n_closed  = sum(1 for i in anomalies.index
                        if audit_actions.get(str(i), {}).get("status") == "closed")
        n_open    = n_anomaly - n_closed
        n_further = sum(1 for i in anomalies.index
                        if audit_actions.get(str(i), {}).get("action") == "required_further_action")

        st.markdown("<hr class='green-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Summary</div>", unsafe_allow_html=True)

        def metric_card(col, value, label, cls=""):
            col.markdown(f"""
            <div class='metric-card {cls}'>
                <div class='metric-card-value'>{value}</div>
                <div class='metric-card-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        metric_card(c1, f"{total:,}",          "Total Transactions")
        metric_card(c2, f"{n_normal:,}",        "Normal",            "teal")
        metric_card(c3, f"{n_anomaly:,}",       "Anomalies",         "red")
        metric_card(c4, f"{anomaly_rate:.1f}%", "Anomaly Rate",      "orange")
        metric_card(c5, f"{n_open}",            "Open Cases",        "orange")
        metric_card(c6, f"{n_closed}",          "Closed Cases",      "blue")
        metric_card(c7, f"{n_further}",         "Further Action",    "purple")

        # ── Pie chart ────────────────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        fig_pie = px.pie(
            values=[n_normal, n_anomaly],
            names=["Normal", "Anomaly"],
            color_discrete_sequence=["#43a047", "#e53935"],   # FIX: was color_sequence
            hole=0.45,
            title="Transaction Distribution"
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="DM Sans", title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=20, l=0, r=0), height=280
        )
        fig_pie.update_traces(textinfo="percent+label")
        col_pie, _ = st.columns([1, 1.5])
        with col_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Anomaly list ──────────────────────────────────────────────────────
        display_cols      = [c for c in anomalies.columns if not c.startswith("_")]
        anomalies_display = anomalies[display_cols].copy() if display_cols else anomalies.copy()

        st.markdown("<hr class='green-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Anomaly List</div>", unsafe_allow_html=True)

        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.markdown(get_download_link(anomalies_display, "anomalies.csv"),
                        unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if n_anomaly == 0:
            st.success("No anomalies detected.")
        else:
            for i, (idx, row) in enumerate(anomalies.iterrows()):
                action_data = audit_actions.get(str(idx), {})
                status      = action_data.get("status", "open")
                badge       = (f"<span class='badge-open'>Open</span>"
                               if status == "open"
                               else f"<span class='badge-closed'>Closed</span>")

                disp_cols = [c for c in df.columns if not c.startswith("_")][:6]
                row_info  = " &nbsp;|&nbsp; ".join(
                    [f"<b>{c}:</b> {row.get(c,'')}" for c in disp_cols])

                status_symbol = "[Open]" if status == "open" else "[Closed]"
                with st.expander(
                    f"Anomaly #{i+1}  {status_symbol}  —  Score: {row['_anomaly_score']:.4f}",
                    expanded=False
                ):
                    st.markdown(
                        f"<div style='font-size:0.82rem;color:#444;margin-bottom:10px;'>"
                        f"{row_info}</div>",
                        unsafe_allow_html=True
                    )
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        audit_work = st.text_area(
                            "Audit Work Performed",
                            value=action_data.get("audit_work", ""),
                            key=f"aw_{idx}", height=90
                        )
                        doc      = st.file_uploader("Supporting Document",
                                                    key=f"doc_{idx}")
                        doc_name = action_data.get("doc_name", "")
                        doc_path = action_data.get("doc_path", "")
                        if doc_name:
                            st.caption(f"Previously uploaded: {doc_name}")
                            if doc_path and os.path.exists(doc_path):
                                st.caption(f"Saved at: {doc_path}")

                    with col_b:
                        action_choice = st.selectbox(
                            "Action",
                            ["no_further_action", "required_further_action"],
                            index=0 if action_data.get("action", "no_further_action")
                                       == "no_further_action" else 1,
                            key=f"ac_{idx}",
                            format_func=lambda x: (
                                "No Further Action" if x == "no_further_action"
                                else "Required Further Action"
                            )
                        )
                        new_status = st.selectbox(
                            "Status", ["open", "closed"],
                            index=0 if status == "open" else 1,
                            key=f"st_{idx}"
                        )

                    if st.button("Submit", key=f"sub_{idx}"):
                        saved_path = doc_path  # keep existing path by default
                        saved_name = doc_name
                        if doc is not None:
                            saved_path = save_evidence(idx, doc)
                            saved_name = doc.name
                        audit_actions[str(idx)] = {
                            "audit_work": audit_work,
                            "action":     action_choice,
                            "status":     new_status,
                            "doc_name":   saved_name,
                            "doc_path":   saved_path,
                            "updated_at": str(datetime.now()),
                            "updated_by": st.session_state.get("user_data", {}).get("name", "")
                        }
                        save_audit_actions(audit_actions)
                        st.session_state["audit_actions"] = audit_actions
                        st.success("Submitted successfully.")
                        st.rerun()

# ─── Analysis Page ────────────────────────────────────────────────────────────

def analysis_page():
    st.markdown("<div class='page-title'>Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Explore and visualise your audit dataset.</div>",
                unsafe_allow_html=True)

    if "pred_df_result" not in st.session_state:
        st.markdown("<div class='info-box'>Please run anomaly detection on the Prediction page first.</div>",
                    unsafe_allow_html=True)
        return

    df           = st.session_state["pred_df_result"]
    num_cols     = df.select_dtypes(include=np.number).columns.tolist()
    non_internal = [c for c in num_cols if not c.startswith("_")]

    anomalies = df[df["_is_anomaly"] == True]
    normal    = df[df["_is_anomaly"] == False]

    total         = len(df)
    anomaly_rate  = (len(anomalies) / total * 100) if total > 0 else 0
    total_amount  = df["Amount"].sum()        if "Amount" in df.columns       else None
    anomaly_amount= anomalies["Amount"].sum() if "Amount" in anomalies.columns else None

    # ── Overview ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    def ov_metric(col, val, label, cls=""):
        col.markdown(
            f"<div class='metric-card {cls}'>"
            f"<div class='metric-card-value'>{val}</div>"
            f"<div class='metric-card-label'>{label}</div></div>",
            unsafe_allow_html=True
        )
    ov_metric(oc1, f"{total:,}",               "Total Rows")
    ov_metric(oc2, f"{len(df.columns)}",        "Total Columns")
    ov_metric(oc3, f"{anomaly_rate:.1f}%",      "Anomaly Rate",    "red")
    ov_metric(oc4, f"{total_amount:,.0f}"   if total_amount   is not None else "N/A", "Total Amount",   "teal")
    ov_metric(oc5, f"{anomaly_amount:,.0f}" if anomaly_amount is not None else "N/A", "Anomaly Amount", "orange")

    st.markdown("<hr class='green-divider'>", unsafe_allow_html=True)

    # ── Scatter Plot ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Scatter Plot</div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        x_axis = st.selectbox("X-Axis", non_internal, key="sc_x")
    with sc2:
        y_opts = [c for c in non_internal if c != x_axis] or non_internal
        y_axis = st.selectbox("Y-Axis", y_opts, key="sc_y")

    fig_scatter = px.scatter(
        df, x=x_axis, y=y_axis,
        color="_anomaly_label",
        color_discrete_map={"Normal": "#43a047", "Anomaly": "#e53935"},
        opacity=0.65, title=f"{x_axis} vs {y_axis}"
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
        font_family="DM Sans", legend_title_text="",
        margin=dict(t=40, b=20, l=0, r=0), height=380
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<hr class='green-divider'>", unsafe_allow_html=True)

    # ── KDE Distribution ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Feature Distribution (KDE)</div>",
                unsafe_allow_html=True)
    kde_feat = st.selectbox("Select Feature", non_internal, key="kde_feat")

    if kde_feat:
        normal_vals  = normal[kde_feat].dropna().values.astype(float)
        anomaly_vals = anomalies[kde_feat].dropna().values.astype(float)

        # Build a shared x-range that covers both groups
        all_vals = np.concatenate([normal_vals, anomaly_vals])
        if len(all_vals) < 2 or np.std(all_vals) == 0:
            st.info("Not enough variation in this feature to plot a KDE.")
        else:
            x_min   = np.percentile(all_vals, 1)
            x_max   = np.percentile(all_vals, 99)
            x_range = np.linspace(x_min, x_max, 400)

            fig_kde = go.Figure()

            # Normal KDE
            kde_n_vals = safe_kde(normal_vals, x_range)
            if kde_n_vals is not None:
                fig_kde.add_trace(go.Scatter(
                    x=x_range, y=kde_n_vals,
                    mode='lines', name='Normal',
                    line=dict(color='#43a047', width=2.5),
                    fill='tozeroy', fillcolor='rgba(67,160,71,0.12)'
                ))
            else:
                st.caption("Normal KDE: insufficient variation to plot.")

            # Anomaly KDE
            kde_a_vals = safe_kde(anomaly_vals, x_range)
            if kde_a_vals is not None:
                fig_kde.add_trace(go.Scatter(
                    x=x_range, y=kde_a_vals,
                    mode='lines', name='Anomaly',
                    line=dict(color='#e53935', width=2.5),
                    fill='tozeroy', fillcolor='rgba(229,57,53,0.12)'
                ))
            else:
                st.caption("Anomaly KDE: insufficient variation to plot.")

            fig_kde.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
                font_family="DM Sans", title=f"KDE — {kde_feat}",
                title_font_size=14, legend_title_text="",
                margin=dict(t=40, b=20, l=0, r=0), height=320,
                xaxis_title=kde_feat, yaxis_title="Density"
            )
            st.plotly_chart(fig_kde, use_container_width=True)

    st.markdown("<hr class='green-divider'>", unsafe_allow_html=True)

    # ── Descriptive Statistics ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Descriptive Statistics</div>",
                unsafe_allow_html=True)
    desc_df = df[non_internal].describe().T.round(4)
    st.dataframe(desc_df, use_container_width=True)

    st.markdown("<hr class='green-divider'>", unsafe_allow_html=True)

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Correlation Heatmap</div>",
                unsafe_allow_html=True)
    if len(non_internal) >= 2:
        corr   = df[non_internal].corr().round(3)
        fig_hm = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#e53935"], [0.5, "#ffffff"], [1, "#2e7d32"]],
            zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont_size=10,
            hoverongaps=False
        ))
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_family="DM Sans",
            margin=dict(t=20, b=20, l=60, r=20),
            height=max(300, len(non_internal) * 35 + 80)
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for the heatmap.")

# ─── Profile Page ─────────────────────────────────────────────────────────────

def profile_page():
    st.markdown("<div class='page-title'>User Profile</div>", unsafe_allow_html=True)

    uid   = st.session_state.get("user_id", "")
    udata = st.session_state.get("user_data", {})

    tab1, tab2, tab3, tab4 = st.tabs(["Profile", "Access Info", "Audit Trail", "Settings"])

    with tab1:
        st.markdown("<div class='section-header'>Personal Information</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        def info_row(label, value):
            st.markdown(
                f"<div style='margin-bottom:10px;'>"
                f"<span style='font-size:0.75rem;color:#888;text-transform:uppercase;"
                f"letter-spacing:0.06em;'>{label}</span><br>"
                f"<span style='font-size:0.97rem;font-weight:600;color:#1b5e20;'>"
                f"{value or '—'}</span></div>",
                unsafe_allow_html=True
            )
        with c1:
            info_row("Full Name",   udata.get("name"))
            info_row("Email",       udata.get("email"))
            info_row("Department",  udata.get("department"))
        with c2:
            info_row("Role",        udata.get("role"))
            info_row("Employee ID", udata.get("employee_id"))
            info_row("Last Login",  udata.get("last_login"))

    with tab2:
        st.markdown("<div class='section-header'>Access Information</div>",
                    unsafe_allow_html=True)
        perms = ', '.join(udata.get('permissions', ['prediction', 'analysis', 'profile']))
        st.markdown(f"""
        <div class='card'>
            <div style='margin-bottom:8px;'><b>User ID:</b> {uid}</div>
            <div style='margin-bottom:8px;'><b>Role:</b> {udata.get('role','—')}</div>
            <div style='margin-bottom:8px;'><b>Permissions:</b> {perms}</div>
            <div><b>Account Status:</b>
                <span style='color:#2e7d32;font-weight:600;'>Active</span></div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='section-header'>Audit Trail</div>", unsafe_allow_html=True)
        audit_actions = load_audit_actions()
        if not audit_actions:
            st.info("No audit actions recorded yet.")
        else:
            trail_rows = [
                {
                    "Transaction Index": idx,
                    "Action":     a.get("action", "").replace("_", " ").title(),
                    "Status":     a.get("status", "open").title(),
                    "Updated At": a.get("updated_at", ""),
                    "Document":   a.get("doc_name", "")
                }
                for idx, a in audit_actions.items()
                if a.get("updated_by") == udata.get("name")
            ]
            if trail_rows:
                st.dataframe(pd.DataFrame(trail_rows), use_container_width=True)
            else:
                st.info("No actions taken by this user yet.")

    with tab4:
        st.markdown("<div class='section-header'>Change Password</div>",
                    unsafe_allow_html=True)
        with st.form("change_pwd"):
            current  = st.text_input("Current Password",       type="password")
            new_pwd  = st.text_input("New Password",           type="password")
            confirm  = st.text_input("Confirm New Password",   type="password")
            save_btn = st.form_submit_button("Update Password")

        if save_btn:
            users          = load_users()
            stored_pwd     = udata.get("password", "")
            hashed_current = hash_password(current)

            # Accept plain-text OR hashed stored password (mirrors authenticate logic)
            pwd_matches = (stored_pwd == current) or (stored_pwd == hashed_current)

            if not pwd_matches:
                st.error("Current password is incorrect.")
            elif new_pwd != confirm:
                st.error("New passwords do not match.")
            elif len(new_pwd) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                # Store new password as plain text (consistent with users.json format)
                users[uid]["password"] = new_pwd
                save_users(users)
                st.session_state["user_data"]["password"] = new_pwd
                st.success("Password updated successfully.")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    if not st.session_state.get("logged_in"):
        login_page()
        return

    uid = st.session_state.get("user_id")
    if uid and not st.session_state.get("_login_logged"):
        users = load_users()
        if uid in users:
            users[uid]["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_users(users)
            st.session_state["user_data"] = users[uid]
        st.session_state["_login_logged"] = True

    page = render_sidebar()

    if page == "Prediction":
        prediction_page()
    elif page == "Analysis":
        analysis_page()
    elif page == "User Profile":
        profile_page()

if __name__ == "__main__":
    main()
