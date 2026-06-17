"""
AI Work Sentiment Dashboard - Streamlit Deployable App

Dashboard skripsi untuk visualisasi dan deployment modeling analisis sentimen
publik terhadap Artificial Intelligence dalam dunia kerja berbasis pengetahuan
di Indonesia.

Versi ini mengikuti notebook revisi setelah sidang. App memprioritaskan
Dataset Final Semisupervised Binary.csv, menjalankan preprocessing, split data,
feature extraction, training baseline TF-IDF + SVM, evaluasi, dan prediksi
secara langsung saat Streamlit berjalan.

Catatan:
- Dashboard difokuskan untuk visualisasi hasil modeling skripsi.
- Transformer tidak di-fine-tuning ulang di Streamlit Cloud agar aplikasi stabil saat deploy.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Work Sentiment Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# Dataset utama mengikuti notebook revisi setelah sidang.
# Urutan kandidat dibuat agar deploy tetap aman jika file ditempatkan di root repo
# atau di folder data/.
CSV_CANDIDATES = [
    BASE_DIR / "Dataset Final Semisupervised Binary.csv",
    BASE_DIR / "data" / "Dataset Final Semisupervised Binary.csv",
    BASE_DIR / "Dataset Final Semisupervised.csv",
    BASE_DIR / "data" / "Dataset Final Semisupervised.csv",
    BASE_DIR / "Dataset Final.csv",
    BASE_DIR / "data" / "Dataset Final.csv",
]

# Kandidat dataset sumber untuk KPI "Data Sumber".
# Modeling tetap memakai dataset biner 5.034 baris, tetapi sumber data penelitian
# dapat berasal dari dataset penuh 5.226 baris jika file tersebut tersedia di repo.
SOURCE_DATASET_CANDIDATES = [
    BASE_DIR / "Dataset_Skripsi_Finals_Bismillah.csv",
    BASE_DIR / "data" / "Dataset_Skripsi_Finals_Bismillah.csv",
    BASE_DIR / "Dataset Final.csv",
    BASE_DIR / "data" / "Dataset Final.csv",
    BASE_DIR / "Dataset Final Semisupervised.csv",
    BASE_DIR / "data" / "Dataset Final Semisupervised.csv",
    BASE_DIR / "Dataset Final Semisupervised Binary.csv",
    BASE_DIR / "data" / "Dataset Final Semisupervised Binary.csv",
]


def get_default_dataset_path() -> Optional[Path]:
    for candidate in CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def get_source_dataset_path() -> Optional[Path]:
    for candidate in SOURCE_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


DEFAULT_DATASET = get_default_dataset_path()
DEFAULT_SOURCE_DATASET = get_source_dataset_path()

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_FROM_REMAINING = 0.125
SEMI_SUPERVISED_LABELS = ["ancaman", "peluang", "netral"]
MODEL_LABELS = ["ancaman", "peluang"]
LABEL_ORDER = MODEL_LABELS

MODEL_COLORS = {
    "TF-IDF + SVM tanpa stemming": "#4E79A7",
    "TF-IDF + SVM dengan stemming": "#59A14F",
    "IndoBERT": "#E15759",
    "IndoBERTweet": "#F28E2B",
    "IndoBERTweet tanpa normalisasi": "#B07AA1",
}

LABEL_COLORS = {
    "ancaman": "#E15759",
    "peluang": "#59A14F",
    "netral": "#4E79A7",
}

# Hasil modeling Bab IV dari notebook revisi setelah sidang.
# Nilai ini ditampilkan untuk visualisasi dashboard, bukan untuk menjalankan fine-tuning ulang.
RESEARCH_MODEL_RESULTS = {
    "TF-IDF + SVM tanpa stemming": {
        "accuracy": 0.986097,
        "macro_precision": 0.99,
        "macro_recall": 0.98,
        "macro_f1": 0.98,
        "weighted_precision": 0.99,
        "weighted_recall": 0.99,
        "weighted_f1": 0.99,
    },
    "TF-IDF + SVM dengan stemming": {
        "accuracy": 0.986097,
        "macro_precision": 0.99,
        "macro_recall": 0.98,
        "macro_f1": 0.98,
        "weighted_precision": 0.99,
        "weighted_recall": 0.99,
        "weighted_f1": 0.99,
    },
    "IndoBERT": {
        "accuracy": 0.987090,
        "macro_precision": 0.99,
        "macro_recall": 0.98,
        "macro_f1": 0.98,
        "weighted_precision": 0.99,
        "weighted_recall": 0.99,
        "weighted_f1": 0.99,
    },
    "IndoBERTweet": {
        "accuracy": 0.987090,
        "macro_precision": 0.99,
        "macro_recall": 0.98,
        "macro_f1": 0.98,
        "weighted_precision": 0.99,
        "weighted_recall": 0.99,
        "weighted_f1": 0.99,
    },
    "IndoBERTweet tanpa normalisasi": {
        "accuracy": 0.979146,
        "macro_precision": 0.97,
        "macro_recall": 0.98,
        "macro_f1": 0.98,
        "weighted_precision": 0.98,
        "weighted_recall": 0.98,
        "weighted_f1": 0.98,
    },
}

TRANSFORMER_TRAINING = {
    "IndoBERT": [
        {"Epoch": 1, "Training Loss": 0.181766, "Validation Loss": 0.059876, "Validation Accuracy": 0.986111},
        {"Epoch": 2, "Training Loss": 0.068894, "Validation Loss": 0.069323, "Validation Accuracy": 0.992063},
        {"Epoch": 3, "Training Loss": 0.021799, "Validation Loss": 0.064377, "Validation Accuracy": 0.990079},
    ],
    "IndoBERTweet": [
        {"Epoch": 1, "Training Loss": 0.153412, "Validation Loss": 0.063271, "Validation Accuracy": 0.988095},
        {"Epoch": 2, "Training Loss": 0.066158, "Validation Loss": 0.055987, "Validation Accuracy": 0.990079},
        {"Epoch": 3, "Training Loss": 0.025167, "Validation Loss": 0.069481, "Validation Accuracy": 0.992063},
    ],
    "IndoBERTweet tanpa normalisasi": [
        {"Epoch": 1, "Training Loss": 0.149644, "Validation Loss": 0.071791, "Validation Accuracy": 0.978175},
        {"Epoch": 2, "Training Loss": 0.067217, "Validation Loss": 0.054866, "Validation Accuracy": 0.990079},
        {"Epoch": 3, "Training Loss": 0.017818, "Validation Loss": 0.074167, "Validation Accuracy": 0.990079},
    ],
}

TRANSFORMER_CLASS_REPORTS = {
    "IndoBERT": [
        {"Kelas": "Ancaman", "Precision": 0.99, "Recall": 0.99, "F1-score": 0.99, "Support": 707},
        {"Kelas": "Peluang", "Precision": 0.99, "Recall": 0.97, "F1-score": 0.98, "Support": 300},
    ],
    "IndoBERTweet": [
        {"Kelas": "Ancaman", "Precision": 0.99, "Recall": 0.99, "F1-score": 0.99, "Support": 707},
        {"Kelas": "Peluang", "Precision": 0.98, "Recall": 0.97, "F1-score": 0.98, "Support": 300},
    ],
    "IndoBERTweet tanpa normalisasi": [
        {"Kelas": "Ancaman", "Precision": 0.99, "Recall": 0.98, "F1-score": 0.99, "Support": 707},
        {"Kelas": "Peluang", "Precision": 0.96, "Recall": 0.97, "F1-score": 0.97, "Support": 300},
    ],
}

# Matriks menggunakan urutan label: [ancaman, peluang].
TRANSFORMER_CONFUSION_MATRICES = {
    "IndoBERT": np.array([
        [703, 4],
        [9, 291],
    ]),
    "IndoBERTweet": np.array([
        [702, 5],
        [8, 292],
    ]),
    "IndoBERTweet tanpa normalisasi": np.array([
        [695, 12],
        [9, 291],
    ]),
}

SLANG_MAP = {
    "yg": "yang",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "krn": "karena",
    "dr": "dari",
    "utk": "untuk",
    "dgn": "dengan",
    "aja": "saja",
    "udah": "sudah",
    "blm": "belum",
    "bgt": "banget",
    "jd": "jadi",
    "org": "orang",
    "hrs": "harus",
    "thn": "tahun",
    "dpt": "dapat",
}

DEFAULT_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "dalam",
    "atau", "juga", "karena", "agar", "sebagai", "adalah", "itu", "ini",
    "sudah", "telah", "akan", "lebih", "saja", "jadi", "masih", "para",
    "oleh", "seperti", "dapat", "bisa", "buat", "bagi", "secara", "nya",
    "si", "kok", "nih", "deh", "dong", "ya", "yg", "ga", "gak",
}
KEEP_WORDS = {"tidak", "bukan", "belum", "tanpa", "kurang"}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-card: #ffffff;
    --bg-card-hover: #f9fafb;
    --border: #e5e7eb;
    --text-primary: #111827;
    --text-secondary: #6b7280;
    --accent-red: #E15759;
    --accent-blue: #4E79A7;
    --accent-green: #59A14F;
    --accent-orange: #F28E2B;
    --accent-purple: #B07AA1;
    --gold: #F0C040;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: #ffffff;
}

[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.kpi-card {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
    min-height: 132px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--accent-color, #4E79A7);
    border-radius: 14px 14px 0 0;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #d1d5db !important;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.10);
}

.kpi-label {
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280 !important;
    margin-bottom: 0.45rem;
}

.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    line-height: 1.2;
    color: #111827 !important;
}

.kpi-sub {
    font-size: 0.72rem;
    color: #6b7280 !important;
    margin-top: 0.35rem;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #e5e7eb !important;
}

.section-badge {
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280 !important;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #111827 !important;
    margin: 0;
}

.info-card,
.best-model-card,
[data-testid="metric-container"],
details {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

.info-card {
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.info-card-title {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #6b7280 !important;
    margin-bottom: 0.6rem;
}

.best-model-card {
    padding: 1.5rem;
    text-align: center;
    position: relative;
}

.best-badge {
    display: inline-block;
    background: #E15759;
    color: white !important;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

.code-output {
    background: #f8fafc !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #374151 !important;
    overflow-x: auto;
    white-space: pre;
}

.insight-box {
    background: #eff6ff !important;
    border-left: 4px solid #4E79A7;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #1f2937 !important;
}

.insight-box.warning {
    border-left-color: #F0C040;
    background: #fffbeb !important;
}

.insight-box.success {
    border-left-color: #59A14F;
    background: #ecfdf5 !important;
}

.dashboard-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #111827 !important;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.dashboard-subtitle {
    font-size: 0.9rem;
    color: #6b7280 !important;
    font-weight: 400;
    line-height: 1.5;
}

.sidebar-nav-label {
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280 !important;
    padding: 0.5rem 0 0.25rem;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

hr {
    border-color: #e5e7eb !important;
    margin: 1.5rem 0;
}

h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #111827 !important;
    opacity: 1 !important;
}

.stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"] {
    color: #374151 !important;
}

button[data-baseweb="tab"] p {
    color: #6b7280 !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: #E15759 !important;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] label p {
    color: #374151 !important;
    font-size: 0.9rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.15rem 0 !important;
}

.stAlert {
    border-radius: 8px !important;
}


.dual-best-card {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px;
    padding: 1.0rem 1.0rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    min-height: 132px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.dual-best-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--accent-color, #E15759);
    border-radius: 14px 14px 0 0;
}

.dual-best-label {
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280 !important;
    margin-bottom: 0.45rem;
}

.dual-best-stack {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.18rem;
    margin: 0.05rem 0 0.15rem 0;
}

.dual-best-pill {
    display: inline-block;
    min-width: 8.7rem;
    max-width: 100%;
    padding: 0.13rem 0.65rem;
    border-radius: 999px;
    border: 1px solid #f3c2c3;
    background: #fff7f7;
    color: #111827 !important;
    font-family: 'DM Serif Display', serif;
    font-size: 1.0rem;
    line-height: 1.12;
    white-space: nowrap;
}

.dual-best-join {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #6b7280 !important;
    text-transform: uppercase;
    line-height: 1;
}

.dual-best-sub {
    font-size: 0.70rem;
    color: #6b7280 !important;
    margin-top: 0.25rem;
}


.info-card *,
.best-model-card *,
.insight-box *,
.kpi-card *,
.dual-best-card * {
    color: inherit;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# SMALL UI HELPERS
# -----------------------------------------------------------------------------
def section_header(badge: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <span class="section-badge">{badge}</span>
            <h2 class="section-title">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(text: str, style: str = "") -> None:
    st.markdown(
        f'<div class="insight-box {style}">💡 {text}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, color: str) -> None:
    value_text = str(value)
    value_font_size = "1.35rem" if len(value_text) > 18 else "2rem"
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent-color:{color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};font-size:{value_font_size};word-break:normal;overflow-wrap:break-word;">{value_text}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def dual_best_model_card(models: List[str], sub: str = "Accuracy tertinggi setara", color: str = "#E15759") -> None:
    """Render a compact KPI card for tied best models."""
    pills_html = ""
    for i, model in enumerate(models):
        pills_html += f'<div class="dual-best-pill">{model}</div>'
        if i < len(models) - 1:
            pills_html += '<div class="dual-best-join">&</div>'

    st.markdown(
        f"""
        <div class="dual-best-card" style="--accent-color:{color};">
            <div class="dual-best-label">Model Terbaik</div>
            <div class="dual-best-stack">
                {pills_html}
            </div>
            <div class="dual-best-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_dark_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply a clean light dashboard layout to Plotly figures.

    Function name is kept so the rest of the app does not need to change.
    """
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#374151", size=11),
        legend=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(color="#111827"),
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        height=height,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#d1d5db", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#f3f4f6", linecolor="#d1d5db", zerolinecolor="#e5e7eb")
    return fig


# -----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
def read_csv_safely(source: Any) -> pd.DataFrame:
    """Read CSV robustly; notebook files mainly use semicolon separators."""
    attempts = [
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": None, "encoding": "utf-8-sig", "engine": "python"},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df_read = pd.read_csv(source, **kwargs)
            if len(df_read.columns) > 1:
                return df_read
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("Dataset tidak dapat dibaca sebagai CSV.")


@st.cache_data(show_spinner=False)
def load_dataset_from_path(path: str) -> pd.DataFrame:
    return read_csv_safely(path)


@st.cache_data(show_spinner=False)
def get_dataset_row_count(path: str) -> int:
    return int(len(read_csv_safely(path)))


def standardize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Kolom label diprioritaskan sesuai output notebook revisi.
    label_priority = [
        "label_final",
        "label_final_semisupervised",
        "Label Final",
        "manual_labeling",
        "label_manual_awal",
        "sentiment",
        "sentimen",
        "label",
    ]
    label_source = next((col for col in label_priority if col in df.columns), None)
    if label_source is None:
        candidate_cols = [c for c in df.columns if "label" in c.lower()]
        label_source = candidate_cols[0] if candidate_cols else None

    if "full_text" not in df.columns:
        text_candidates = [c for c in df.columns if c.lower() in {"text", "tweet", "content", "clean_text"}]
        if text_candidates:
            df["full_text"] = df[text_candidates[0]]

    missing = []
    if "full_text" not in df.columns:
        missing.append("full_text")
    if label_source is None:
        missing.append("label_final/label_final_semisupervised/manual_labeling")
    if missing:
        raise ValueError(
            "Dataset harus punya kolom teks dan label. "
            f"Kolom yang hilang: {', '.join(missing)}."
        )

    df["manual_labeling"] = df[label_source].astype(str).str.strip().str.lower()
    df.loc[df["manual_labeling"].isin(["", "nan", "none", "null"]), "manual_labeling"] = np.nan
    df["label_final"] = df["manual_labeling"]
    df["label_source_used"] = label_source
    return df

@st.cache_data(show_spinner=False)
def get_stopwords() -> Tuple[set, str]:
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        stopwords = set(StopWordRemoverFactory().get_stop_words())
        source = "Sastrawi"
    except Exception:
        stopwords = set(DEFAULT_STOPWORDS)
        source = "fallback bawaan app"
    return stopwords - KEEP_WORDS, source


def light_normalize_text(text: Any) -> str:
    text = str(text).lower()
    text = text.replace("&amp", " dan ")
    text = text.replace("&lt", " ")
    text = text.replace("&gt", " ")
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = NON_ALNUM_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    return text


def normalize_slang_words(text: Any) -> str:
    tokens = str(text).split()
    normalized_tokens = [SLANG_MAP.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)


def remove_stopwords(text: Any, stopwords: Iterable[str]) -> str:
    stopword_set = set(stopwords)
    tokens = str(text).split()
    return " ".join(token for token in tokens if token not in stopword_set)


@st.cache_data(show_spinner=True)
def preprocess_dataset(df: pd.DataFrame, use_stemming: bool = True) -> Tuple[pd.DataFrame, str]:
    """Run the preprocessing pipeline from the notebook."""
    df = standardize_label_column(df)
    stopwords, stopword_source = get_stopwords()

    df["text_casefold"] = df["full_text"].astype(str).str.lower()
    df["text_light_normalized"] = df["text_casefold"].apply(light_normalize_text)
    df["text_normalized"] = df["text_light_normalized"].apply(normalize_slang_words)
    df["text_stopword_removed"] = df["text_normalized"].apply(lambda x: remove_stopwords(x, stopwords))
    df["text_no_stemming"] = df["text_stopword_removed"]

    stemmer_status = "stemming dilewati"
    if use_stemming:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

            stemmer = StemmerFactory().create_stemmer()
            df["text_stemmed"] = df["text_stopword_removed"].apply(stemmer.stem)
            stemmer_status = "Sastrawi"
        except Exception:
            df["text_stemmed"] = df["text_stopword_removed"]
            stemmer_status = "fallback: Sastrawi tidak tersedia, memakai teks tanpa stemming"
    else:
        df["text_stemmed"] = df["text_stopword_removed"]
        stemmer_status = "dimatikan dari sidebar"

    metadata = f"Stopword: {stopword_source}; Stemming: {stemmer_status}"
    return df, metadata


@st.cache_data(show_spinner=False)
def cleansing_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_label_column(df)
    return pd.DataFrame(
        {
            "pemeriksaan": [
                "Jumlah data sumber",
                "Jumlah data siap modeling biner",
                "full_text kosong",
                "label kosong",
                "Duplikat full_text",
            ],
            "hasil": [
                len(df),
                int(df["label_final"].isin(MODEL_LABELS).sum()),
                int(df["full_text"].isna().sum()),
                int(df["label_final"].isna().sum()),
                int(df["full_text"].duplicated().sum()),
            ],
        }
    )

@st.cache_data(show_spinner=False)
def top_words(series: pd.Series, n: int = 20) -> pd.DataFrame:
    text = " ".join(series.dropna().astype(str))
    tokens = re.findall(r"\b\w+\b", text.lower())
    return pd.DataFrame(Counter(tokens).most_common(n), columns=["kata", "frekuensi"])


@st.cache_data(show_spinner=True)
def split_and_train(df: pd.DataFrame) -> Dict[str, Any]:
    """Split data and train the deployable baseline models from notebook code."""
    df_model = df.dropna(subset=["label_final"]).copy()
    df_model = df_model[df_model["label_final"].isin(MODEL_LABELS)].copy()

    if df_model["label_final"].nunique() < 2:
        raise ValueError("Dataset modeling harus berisi minimal dua label: ancaman dan peluang.")

    train_val_idx, test_idx = train_test_split(
        df_model.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_model["label_final"],
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VALIDATION_SIZE_FROM_REMAINING,
        random_state=RANDOM_STATE,
        stratify=df_model.loc[train_val_idx, "label_final"],
    )

    y_train = df_model.loc[train_idx, "label_final"]
    y_val = df_model.loc[val_idx, "label_final"]
    y_test = df_model.loc[test_idx, "label_final"]

    X_train_no_stem = df_model.loc[train_idx, "text_no_stemming"]
    X_test_no_stem = df_model.loc[test_idx, "text_no_stemming"]
    X_train_stem = df_model.loc[train_idx, "text_stemmed"]
    X_test_stem = df_model.loc[test_idx, "text_stemmed"]

    tfidf_no_stem = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_stem = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_train_no_stem_tfidf = tfidf_no_stem.fit_transform(X_train_no_stem)
    X_test_no_stem_tfidf = tfidf_no_stem.transform(X_test_no_stem)
    X_train_stem_tfidf = tfidf_stem.fit_transform(X_train_stem)
    X_test_stem_tfidf = tfidf_stem.transform(X_test_stem)

    svm_no_stem = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    svm_no_stem.fit(X_train_no_stem_tfidf, y_train)
    y_pred_no_stem = svm_no_stem.predict(X_test_no_stem_tfidf)

    svm_stem = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    svm_stem.fit(X_train_stem_tfidf, y_train)
    y_pred_stem = svm_stem.predict(X_test_stem_tfidf)

    split_distribution = pd.DataFrame(
        {
            "Train": y_train.value_counts(),
            "Validation": y_val.value_counts(),
            "Test": y_test.value_counts(),
        }
    ).fillna(0).astype(int)
    split_distribution = split_distribution.reindex(LABEL_ORDER).fillna(0).astype(int)
    split_distribution.loc["Total"] = split_distribution.sum()

    results = {}
    for model_name, y_pred in [
        ("TF-IDF + SVM tanpa stemming", y_pred_no_stem),
        ("TF-IDF + SVM dengan stemming", y_pred_stem),
    ]:
        results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report_dict": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "report_text": classification_report(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred, labels=LABEL_ORDER),
            "pred": y_pred,
        }

    return {
        "df_model": df_model,
        "train_idx": list(train_idx),
        "val_idx": list(val_idx),
        "test_idx": list(test_idx),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "split_distribution": split_distribution,
        "tfidf_no_stem": tfidf_no_stem,
        "tfidf_stem": tfidf_stem,
        "svm_no_stem": svm_no_stem,
        "svm_stem": svm_stem,
        "results": results,
        "feature_shapes": {
            "tanpa_stemming": X_train_no_stem_tfidf.shape,
            "dengan_stemming": X_train_stem_tfidf.shape,
        },
    }


def report_to_dataframe(report_dict: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label in LABEL_ORDER + ["macro avg", "weighted avg"]:
        if label in report_dict:
            row = report_dict[label]
            rows.append(
                {
                    "Kelas": label.capitalize() if label in LABEL_ORDER else label,
                    "Precision": row.get("precision", np.nan),
                    "Recall": row.get("recall", np.nan),
                    "F1-score": row.get("f1-score", np.nan),
                    "Support": row.get("support", np.nan),
                }
            )
    return pd.DataFrame(rows)


def prediction_margin_to_confidence(decision_scores: np.ndarray) -> float:
    """Convert LinearSVC decision margin to a display-friendly pseudo confidence."""
    scores = np.array(decision_scores, dtype=float).ravel()
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum()
    return float(probs.max())


def predict_text(text: str, df_for_preproc: pd.DataFrame, artifacts: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    stopwords, _ = get_stopwords()
    normalized_light = light_normalize_text(text)
    normalized = normalize_slang_words(normalized_light)
    no_stop = remove_stopwords(normalized, stopwords)

    if model_key == "TF-IDF + SVM dengan stemming":
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

            stemmer = StemmerFactory().create_stemmer()
            model_text = stemmer.stem(no_stop)
        except Exception:
            model_text = no_stop
        vectorizer = artifacts["tfidf_stem"]
        model = artifacts["svm_stem"]
    else:
        model_text = no_stop
        vectorizer = artifacts["tfidf_no_stem"]
        model = artifacts["svm_no_stem"]

    X = vectorizer.transform([model_text])
    label = model.predict(X)[0]
    scores = model.decision_function(X)
    confidence = prediction_margin_to_confidence(scores)

    return {
        "label": label,
        "confidence": confidence,
        "normalized_text": normalized,
        "model_text": model_text,
        "scores": scores.ravel().tolist(),
    }


@st.cache_data(show_spinner=True)
def dominant_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Analisis frasa dominan mengikuti notebook revisi bagian 4.17."""
    df_phrases = standardize_label_column(df)
    df_phrases = (
        df_phrases[df_phrases["label_final"].isin(MODEL_LABELS)]
        .copy()
        .reset_index(drop=True)
    )

    if df_phrases.empty:
        return pd.DataFrame(columns=[
            "label", "rank", "frasa_dominan", "jumlah_tweet",
            "skor_tfidf_rata_rata", "skor_pembeda"
        ])

    # Jika dataset memuat informasi keputusan semi-supervised, gunakan subset label stabil
    # seperti pada notebook agar frasa tidak didominasi pseudo-label yang bertentangan.
    if "label_decision_semisupervised" in df_phrases.columns:
        stable_decisions = [
            "manual_seed_300",
            "pseudo_label_high_confidence",
            "review_low_confidence_use_manual_reference",
        ]
        stable_phrase_mask = df_phrases["label_decision_semisupervised"].isin(stable_decisions)
        if "label_manual_awal" in df_phrases.columns:
            manual_same_mask = (
                df_phrases["label_manual_awal"].astype(str).str.strip().str.lower()
                == df_phrases["label_final"]
            )
            stable_phrase_mask = stable_phrase_mask & (
                manual_same_mask
                | df_phrases["label_decision_semisupervised"].isin([
                    "manual_seed_300",
                    "review_low_confidence_use_manual_reference",
                ])
            )
        if stable_phrase_mask.any():
            df_phrases = df_phrases[stable_phrase_mask].copy().reset_index(drop=True)

    def normalize_phrase_text(text_value: Any) -> str:
        text_value = str(text_value).lower()
        text_value = URL_PATTERN.sub(" ", text_value)
        text_value = MENTION_PATTERN.sub(" ", text_value)
        text_value = HASHTAG_PATTERN.sub(r"\1", text_value)
        text_value = NON_ALNUM_PATTERN.sub(" ", text_value)
        return MULTISPACE_PATTERN.sub(" ", text_value).strip()

    basic_stopwords_phrase = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "dalam", "pada", "ini", "itu",
        "atau", "juga", "karena", "kalau", "akan", "bisa", "ada", "jadi", "lebih", "sudah",
        "saja", "sangat", "sebagai", "para", "aku", "kamu", "kita", "mereka", "dia",
        "the", "to", "of", "in", "is", "are", "a", "an",
        "rt", "amp", "https", "http", "co", "t", "nya", "nih", "sih", "dong", "kok", "lah",
        "ya", "ga", "gak", "nggak", "yg",
    }
    keep_phrase_words = {
        "ai", "kerja", "pekerjaan", "phk", "skill", "karir", "peluang", "ancaman",
        "reskilling", "upskilling", "otomatisasi",
    }

    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stopwords_phrase = set(StopWordRemoverFactory().get_stop_words()) | basic_stopwords_phrase
    except Exception:
        stopwords_phrase = set(basic_stopwords_phrase)
    stopwords_phrase = sorted(stopwords_phrase - keep_phrase_words)

    df_phrases["phrase_text"] = df_phrases["full_text"].apply(normalize_phrase_text)

    phrase_vectorizer = TfidfVectorizer(
        ngram_range=(2, 4),
        min_df=3,
        max_df=0.80,
        stop_words=stopwords_phrase,
        token_pattern=r"(?u)\b\w\w+\b",
        sublinear_tf=True,
    )
    X_phrase_tfidf = phrase_vectorizer.fit_transform(df_phrases["phrase_text"])
    phrase_features = phrase_vectorizer.get_feature_names_out()
    phrase_analyzer = phrase_vectorizer.build_analyzer()

    weak_edge_words = {
        "alasan", "isu", "biasanya", "terus", "yang", "dan", "atau", "karena", "kalau",
        "dengan", "untuk", "pada", "lebih", "saja", "posisi", "tugasnya", "terlalu",
    }
    domain_terms = {
        "ai", "kerja", "pekerjaan", "lapangan", "diganti", "digantikan", "menggantikan",
        "tergantikan", "otomatisasi", "diotomatisasi", "reskilling", "upskilling", "skill",
        "karir", "peluang", "ancaman", "produktif", "produktivitas", "membantu", "bantu",
        "alat", "manusia", "rentan", "efisiensi", "phk", "kreativitas", "inovasi",
    }
    blocked_phrases = {
        "standar biasanya", "biasanya paling", "standar biasanya paling",
        "standar biasanya paling mudah", "ditunda terus", "alasan isu", "alasan isu reskilling",
        "isu reskilling", "isu reskilling ditunda", "biasanya paling mudah", "posisi tugasnya",
        "tugasnya terlalu", "terlalu standar", "efisiensi ruang", "perusahaan mengejar",
        "kerja perusahaan", "ancaman harus", "tambah manusia tidak", "pekerjaan awal",
        "pekerjaan tidak", "juta lapangan", "depan pekerjaan", "manusia makin",
    }
    blocked_substrings = {
        "standar biasanya", "biasanya paling", "posisi tugasnya", "tugasnya terlalu",
        "terlalu standar", "alasan isu", "ditunda terus",
    }

    def is_informative_phrase(phrase: str) -> bool:
        words = phrase.split()
        if len(words) < 2:
            return False
        if phrase in blocked_phrases or any(blocked_part in phrase for blocked_part in blocked_substrings):
            return False
        if words[0] in weak_edge_words or words[-1] in weak_edge_words:
            return False
        if not any(word in domain_terms for word in words):
            return False
        meaningful_words = [word for word in words if word not in weak_edge_words and len(word) > 2]
        return len(meaningful_words) >= 2

    def is_redundant_phrase(phrase: str, selected_phrases: List[str], overlap_threshold: float = 0.75) -> bool:
        phrase_tokens = set(phrase.split())
        for selected_phrase in selected_phrases:
            selected_tokens = set(selected_phrase.split())
            if phrase_tokens.issubset(selected_tokens) or selected_tokens.issubset(phrase_tokens):
                return True
            overlap = len(phrase_tokens & selected_tokens) / max(1, len(phrase_tokens | selected_tokens))
            if overlap >= overlap_threshold:
                return True
        return False

    label_score_map: Dict[str, np.ndarray] = {}
    label_position_map: Dict[str, np.ndarray] = {}
    for label in MODEL_LABELS:
        label_positions = np.flatnonzero(df_phrases["label_final"].to_numpy() == label)
        label_position_map[label] = label_positions
        label_score_map[label] = X_phrase_tfidf[label_positions].mean(axis=0).A1 if len(label_positions) else np.zeros(len(phrase_features))

    rows = []
    for label in MODEL_LABELS:
        label_positions = label_position_map[label]
        if len(label_positions) == 0:
            continue

        other_labels = [other_label for other_label in MODEL_LABELS if other_label != label]
        other_scores = np.mean([label_score_map[other_label] for other_label in other_labels], axis=0)
        label_scores = label_score_map[label]
        dominance_scores = label_scores - other_scores
        candidate_indices = sorted(
            range(len(phrase_features)),
            key=lambda idx: (dominance_scores[idx], label_scores[idx]),
            reverse=True,
        )

        label_texts = df_phrases.iloc[label_positions]["phrase_text"]
        label_doc_ngrams = [set(phrase_analyzer(text_value)) for text_value in label_texts]
        selected_phrases: List[str] = []

        for feature_idx in candidate_indices:
            phrase = phrase_features[feature_idx]
            if dominance_scores[feature_idx] <= 0:
                continue
            if not is_informative_phrase(phrase):
                continue
            if is_redundant_phrase(phrase, selected_phrases):
                continue

            jumlah_tweet = sum(phrase in doc_ngrams for doc_ngrams in label_doc_ngrams)
            selected_phrases.append(phrase)
            rows.append({
                "label": label,
                "rank": len(selected_phrases),
                "frasa_dominan": phrase,
                "jumlah_tweet": int(jumlah_tweet),
                "skor_tfidf_rata_rata": float(label_scores[feature_idx]),
                "skor_pembeda": float(dominance_scores[feature_idx]),
            })
            if len(selected_phrases) == 12:
                break

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def label_distribution_chart(label_counts: pd.Series) -> go.Figure:
    labels = label_counts.index.tolist()
    values = label_counts.values.tolist()
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=[LABEL_COLORS.get(str(lbl).lower(), "#6b7280") for lbl in labels]),
            textfont=dict(color="#111827", size=12),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b7280", size=11),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        annotations=[
            dict(
                text=f"{int(sum(values)):,}<br>Tweet".replace(",", "."),
                x=0.5,
                y=0.5,
                font_size=15,
                font_color="#111827",
                showarrow=False,
            )
        ],
    )
    return fig


def horizontal_bar(df_words: pd.DataFrame, title: str, color: str) -> go.Figure:
    data = df_words.sort_values("frekuensi", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["frekuensi"],
            y=data["kata"],
            orientation="h",
            marker_color=color,
            text=data["frekuensi"],
            textposition="outside",
        )
    )
    fig.update_layout(title=dict(text=title, font=dict(color="#111827", size=14)))
    return plotly_dark_layout(fig, height=430)


def research_model_comparison_dataframe() -> pd.DataFrame:
    rows = []
    for model_name, metrics in RESEARCH_MODEL_RESULTS.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "Weighted Precision": metrics["weighted_precision"],
                "Weighted Recall": metrics["weighted_recall"],
                "Weighted F1": metrics["weighted_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def research_accuracy_chart(height: int = 360) -> go.Figure:
    comparison_df = research_model_comparison_dataframe().sort_values("Accuracy", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=comparison_df["Accuracy"],
            y=comparison_df["Model"],
            orientation="h",
            marker_color=[MODEL_COLORS.get(x, "#4E79A7") for x in comparison_df["Model"]],
            text=[f"{x:.4f} ({x*100:.2f}%)" for x in comparison_df["Accuracy"]],
            textposition="outside",
            textfont=dict(color="#111827", size=11),
        )
    )
    fig.update_layout(title=dict(text="Accuracy Masing-Masing Model", font=dict(color="#111827", size=14)))
    min_acc = float(comparison_df["Accuracy"].min())
    max_acc = float(comparison_df["Accuracy"].max())
    fig.update_xaxes(range=[max(0, min_acc - 0.02), min(1, max_acc + 0.01)], tickformat=".0%")
    return plotly_dark_layout(fig, height=height)


def transformer_training_chart(training_rows: List[Dict[str, float]], title: str) -> go.Figure:
    df = pd.DataFrame(training_rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Epoch"],
            y=df["Training Loss"],
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="#4E79A7", width=2.5),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Epoch"],
            y=df["Validation Loss"],
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="#F28E2B", width=2.5, dash="dash"),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Epoch"],
            y=df["Validation Accuracy"],
            mode="lines+markers",
            name="Validation Accuracy",
            line=dict(color="#59A14F", width=2.5),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#111827")),
        yaxis2=dict(
            overlaying="y",
            side="right",
            title=dict(text="Validation Accuracy", font=dict(color="#59A14F")),
            gridcolor="rgba(0,0,0,0)",
            tickformat=".0%",
        ),
    )
    fig.update_xaxes(tickmode="array", tickvals=[1, 2, 3])
    return plotly_dark_layout(fig, height=330)


def class_metrics_bar(rows: List[Dict[str, Any]], title: str, height: int = 300) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=df["Kelas"], y=df["Precision"], marker_color="#4E79A7"))
    fig.add_trace(go.Bar(name="Recall", x=df["Kelas"], y=df["Recall"], marker_color="#F28E2B"))
    fig.add_trace(go.Bar(name="F1-score", x=df["Kelas"], y=df["F1-score"], marker_color="#59A14F"))
    fig.update_layout(barmode="group", title=dict(text=title, font=dict(color="#111827", size=14)))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return plotly_dark_layout(fig, height=height)


def split_distribution_chart(split_df: pd.DataFrame) -> go.Figure:
    """Create split chart safely even if the label column name changes after reset_index."""
    plot_df = split_df.copy()
    if "Total" in plot_df.index:
        plot_df = plot_df.drop(index="Total", errors="ignore")

    plot_df = plot_df.reset_index()
    first_col = plot_df.columns[0]
    if "label" not in plot_df.columns:
        plot_df = plot_df.rename(columns={first_col: "label"})
    else:
        plot_df["label"] = plot_df["label"].astype(str)

    for col_name in ["Train", "Validation", "Test"]:
        if col_name not in plot_df.columns:
            plot_df[col_name] = 0

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train", x=plot_df["label"], y=plot_df["Train"], marker_color="#59A14F"))
    fig.add_trace(go.Bar(name="Validation", x=plot_df["label"], y=plot_df["Validation"], marker_color="#F28E2B"))
    fig.add_trace(go.Bar(name="Test", x=plot_df["label"], y=plot_df["Test"], marker_color="#E15759"))
    fig.update_layout(barmode="group")
    return plotly_dark_layout(fig, height=340)


def accuracy_chart(results: Dict[str, Any]) -> go.Figure:
    rows = [
        {"Model": name, "Accuracy": item["accuracy"]}
        for name, item in results.items()
    ]
    df_plot = pd.DataFrame(rows).sort_values("Accuracy", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df_plot["Accuracy"],
            y=df_plot["Model"],
            orientation="h",
            marker_color=[MODEL_COLORS.get(x, "#4E79A7") for x in df_plot["Model"]],
            text=[f"{v:.4f} ({v*100:.2f}%)" for v in df_plot["Accuracy"]],
            textposition="outside",
            textfont=dict(color="#111827", size=11),
        )
    )
    fig.update_xaxes(range=[max(0, df_plot["Accuracy"].min() - 0.05), min(1, df_plot["Accuracy"].max() + 0.05)])
    return plotly_dark_layout(fig, height=280)


def confusion_matrix_figure(cm: np.ndarray, title: str, height: int = 360) -> go.Figure:
    cm = np.asarray(cm)
    labels = LABEL_ORDER[: cm.shape[0]]
    if len(labels) != cm.shape[0]:
        labels = [f"Label {i+1}" for i in range(cm.shape[0])]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[label.capitalize() for label in labels],
            y=[label.capitalize() for label in labels],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Jumlah"),
            hovertemplate="Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: %{z}<extra></extra>",
        )
    )

    max_value = float(cm.max()) if cm.size else 0
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            value = int(cm[row_idx, col_idx])
            text_color = "white" if max_value and value > max_value * 0.55 else "#111827"
            fig.add_annotation(
                x=col_idx,
                y=row_idx,
                text=str(value),
                showarrow=False,
                font=dict(color=text_color, size=14),
            )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#111827", size=14)),
        xaxis_title="Prediksi",
        yaxis_title="Label Aktual",
        margin=dict(l=40, r=20, t=50, b=45),
    )
    return plotly_dark_layout(fig, height=height)

def metrics_bar(report_df: pd.DataFrame) -> go.Figure:
    class_rows = report_df[report_df["Kelas"].str.lower().isin(LABEL_ORDER)].copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=class_rows["Kelas"], y=class_rows["Precision"], marker_color="#4E79A7"))
    fig.add_trace(go.Bar(name="Recall", x=class_rows["Kelas"], y=class_rows["Recall"], marker_color="#F28E2B"))
    fig.add_trace(go.Bar(name="F1-score", x=class_rows["Kelas"], y=class_rows["F1-score"], marker_color="#59A14F"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return plotly_dark_layout(fig, height=310)


def issue_chart(top_issues: pd.DataFrame, label: str) -> go.Figure:
    score_col = "skor_pembeda" if "skor_pembeda" in top_issues.columns else "skor_tfidf_rata_rata"
    plot_data = (
        top_issues[top_issues["label"] == label]
        .head(10)
        .sort_values(score_col, ascending=True)
    )
    fig = go.Figure(
        go.Bar(
            x=plot_data[score_col],
            y=plot_data["frasa_dominan"],
            orientation="h",
            marker_color=LABEL_COLORS.get(label, "#4E79A7"),
            text=[f"{x:.4f} | {c} tweet" for x, c in zip(plot_data[score_col], plot_data["jumlah_tweet"])],
            textposition="outside",
            textfont=dict(color="#111827", size=10),
        )
    )
    fig.update_layout(title=dict(text=f"Frasa Dominan: {label.capitalize()}", font=dict(color="#111827", size=14)))
    return plotly_dark_layout(fig, height=390)


# -----------------------------------------------------------------------------
# OPTIONAL TRANSFORMER TRAINING
# -----------------------------------------------------------------------------
def train_transformer_optional(
    model_name: str,
    text_col: str,
    df_model: pd.DataFrame,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Manual Transformer fine-tuning. Not called automatically."""
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    label2id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    id2label = {v: k for k, v in label2id.items()}

    train_df = df_model.loc[train_idx, [text_col, "label_final"]].rename(columns={text_col: "text"}).copy()
    val_df = df_model.loc[val_idx, [text_col, "label_final"]].rename(columns={text_col: "text"}).copy()
    test_df = df_model.loc[test_idx, [text_col, "label_final"]].rename(columns={text_col: "text"}).copy()

    for part in [train_df, val_df, test_df]:
        part["label"] = part["label_final"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = BASE_DIR / "outputs" / re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name)
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    pred_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=1)
    y_pred = [id2label[i] for i in pred_ids]
    y_test = test_df["label_final"].tolist()

    return {
        "model_name": model_name,
        "text_col": text_col,
        "accuracy": accuracy_score(y_test, y_pred),
        "report_dict": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "report_text": classification_report(y_test, y_pred, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred, labels=LABEL_ORDER),
        "output_dir": str(output_dir),
    }


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:1.5rem;">
            <div style="font-size:2rem;">🤖</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#111827;">AI Work Sentiment</div>
            <div style="font-size:0.7rem;color:#6b7280;">Dashboard Skripsi · Deployable</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-nav-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"], label_visibility="collapsed")

    use_stemming = st.toggle("Aktifkan stemming Sastrawi", value=False)
    st.caption("Default dimatikan agar deploy lebih cepat. Aktifkan untuk menjalankan skenario stemming seperti notebook.")

    st.markdown("---")
    st.markdown('<div class="sidebar-nav-label">Navigasi Halaman</div>', unsafe_allow_html=True)

    pages = [
        ("🏠", "Overview"),
        ("📊", "EDA"),
        ("⚙️", "Preprocessing"),
        ("✂️", "Data Split"),
        ("📐", "Baseline TF-IDF + SVM"),
        ("🔮", "Prediksi"),
        ("🧠", "Hasil Transformer"),
        ("🏆", "Model Comparison"),
        ("🧩", "Isu Dominan"),
    ]
    page_labels = [f"{icon} {name}" for icon, name in pages]
    selected = st.radio("", page_labels, label_visibility="collapsed")
    selected_name = selected.split(" ", 1)[1]

    st.markdown("---")
    if uploaded_file is not None:
        st.success("Dataset dari uploader aktif.")
    elif DEFAULT_DATASET is not None:
        st.success(f"{DEFAULT_DATASET.name} ditemukan di repo.")
    else:
        st.error("Dataset belum ditemukan. Upload Dataset Final Semisupervised Binary.csv ke root repo atau folder data/.")

    st.markdown(
        """
        <div style="font-size:0.68rem;color:#d1d5db;text-align:center;line-height:1.6;margin-top:1rem;">
            Streamlit App · Dataset-driven<br/>
            Tanpa HTML export notebook
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# LOAD AND RUN PIPELINE
# -----------------------------------------------------------------------------
try:
    if uploaded_file is not None:
        raw_df = read_csv_safely(uploaded_file)
    else:
        if DEFAULT_DATASET is None:
            raise FileNotFoundError("Dataset Final Semisupervised Binary.csv belum ditemukan di root repo atau folder data/.")
        raw_df = load_dataset_from_path(str(DEFAULT_DATASET))

    with st.spinner("Menjalankan pipeline dari dataset..."):
        processed_df, preprocessing_meta = preprocess_dataset(raw_df, use_stemming=use_stemming)
        artifacts = split_and_train(processed_df)

except Exception as exc:
    st.error("App belum bisa dijalankan karena dataset belum valid.")
    st.exception(exc)
    st.stop()


if uploaded_file is not None:
    source_data_count = int(len(raw_df))
    source_dataset_name = "Dataset upload"
elif DEFAULT_SOURCE_DATASET is not None:
    source_data_count = get_dataset_row_count(str(DEFAULT_SOURCE_DATASET))
    source_dataset_name = DEFAULT_SOURCE_DATASET.name
else:
    source_data_count = int(len(raw_df))
    source_dataset_name = DEFAULT_DATASET.name if DEFAULT_DATASET is not None else "Dataset aktif"

label_counts = artifacts["df_model"]["label_final"].value_counts().reindex(LABEL_ORDER).dropna()
model_results = artifacts["results"]
best_svm_model = max(model_results.items(), key=lambda item: item[1]["accuracy"])[0]
research_comparison_df = research_model_comparison_dataframe()
best_research_accuracy = float(research_comparison_df.loc[0, "Accuracy"])
best_research_models = research_comparison_df.loc[
    research_comparison_df["Accuracy"].eq(best_research_accuracy), "Model"
].tolist()
best_research_model = best_research_models[0]
best_research_model_display = "IndoBERT" if "IndoBERT" in best_research_models else best_research_model
best_research_model_sub = (
    "Setara IndoBERTweet" if {"IndoBERT", "IndoBERTweet"}.issubset(set(best_research_models))
    else "Hasil modeling Bab IV"
)
best_research_model_cards = (
    ["IndoBERT", "IndoBERTweet"]
    if {"IndoBERT", "IndoBERTweet"}.issubset(set(best_research_models))
    else [best_research_model_display]
)


# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------
if selected_name == "Overview":
    st.markdown(
        """
        <div style="margin-bottom:2rem;">
            <div class="dashboard-title">AI Work Sentiment Dashboard</div>
            <div class="dashboard-subtitle">
                Visualisasi hasil modeling analisis sentimen publik terhadap Artificial Intelligence<br/>
                dalam dunia kerja berbasis pengetahuan di Indonesia.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("01", "Ringkasan Penelitian")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Tentang Dashboard</div>
            <p style="color:#374151;font-size:0.9rem;line-height:1.7;margin:0;">
                Dashboard ini menampilkan hasil modeling revisi setelah sidang dengan skenario biner
                <code>ancaman</code> dan <code>peluang</code>. Aplikasi memprioritaskan
                <code>Dataset Final Semisupervised Binary.csv</code> untuk visualisasi data,
                preprocessing, data split, dan prediksi interaktif.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    with cols[0]:
        kpi_card(
            "Total Data Biner",
            f"{len(artifacts['df_model']):,}".replace(",", "."),
            "Siap modeling",
            "#4E79A7",
        )
    with cols[1]:
        kpi_card(
            "Data Sumber",
            f"{source_data_count:,}".replace(",", "."),
            "Baris dataset awal",
            "#59A14F",
        )
    with cols[2]:
        kpi_card(
            "Kategori Label",
            str(len(LABEL_ORDER)),
            "Ancaman · Peluang",
            "#F28E2B",
        )
    with cols[3]:
        dual_best_model_card(
            best_research_model_cards,
            sub="Accuracy tertinggi setara" if len(best_research_model_cards) > 1 else best_research_model_sub,
            color="#E15759",
        )
    with cols[4]:
        kpi_card(
            "Accuracy Terbaik",
            f"{best_research_accuracy*100:.2f}%",
            "Test set",
            "#B07AA1",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.25, 1])
    with col_left:
        section_header("02", "Visualisasi Accuracy Model")
        st.plotly_chart(research_accuracy_chart(height=390), use_container_width=True)
        insight(
            f"Berdasarkan hasil modeling Bab IV, dua model terbaik adalah <strong>IndoBERT</strong> "
            f"dan <strong>IndoBERTweet</strong> dengan accuracy yang sama, yaitu "
            f"<strong>{best_research_accuracy*100:.2f}%</strong> pada test set.",
            "success",
        )

    with col_right:
        section_header("03", "Distribusi Label")
        st.plotly_chart(label_distribution_chart(label_counts), use_container_width=True)


elif selected_name == "EDA":
    section_header("04", "Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "📊 Distribusi Label", "🔤 Top Words"])

    with tab1:
        st.markdown("#### Gambaran Umum Dataset")
        st.dataframe(processed_df.head(10), use_container_width=True)

        st.markdown("#### Struktur Kolom")
        kolom_info = pd.DataFrame(
            {
                "nama_kolom": processed_df.columns,
                "tipe_data": processed_df.dtypes.astype(str).values,
            }
        )
        st.dataframe(kolom_info, use_container_width=True, hide_index=True)

        st.markdown("#### Ringkasan Data Cleansing")
        st.dataframe(cleansing_summary(raw_df), use_container_width=True, hide_index=True)

    with tab2:
        col_tbl, col_chart = st.columns([1, 2])
        label_df = label_counts.rename_axis("Label").reset_index(name="Jumlah")
        label_df["Persentase"] = label_df["Jumlah"] / label_df["Jumlah"].sum() * 100

        with col_tbl:
            st.dataframe(
                label_df.style.format({"Persentase": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
        with col_chart:
            st.plotly_chart(label_distribution_chart(label_counts), use_container_width=True)

    with tab3:
        col_before, col_after = st.columns(2)
        before_words = top_words(processed_df["full_text"], n=20)
        after_words = top_words(processed_df["text_stemmed"], n=20)

        with col_before:
            st.plotly_chart(
                horizontal_bar(before_words, "Top Words Sebelum Preprocessing", "#4E79A7"),
                use_container_width=True,
            )
        with col_after:
            st.plotly_chart(
                horizontal_bar(after_words, "Top Words Setelah Preprocessing", "#59A14F"),
                use_container_width=True,
            )


elif selected_name == "Preprocessing":
    section_header("05", "Preprocessing")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="info-card" style="border-left:3px solid #4E79A7;">
                <div class="info-card-title" style="color:#4E79A7;">Jalur TF-IDF + SVM</div>
                <ul style="color:#374151;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                    <li>Case folding</li>
                    <li>Normalisasi teks</li>
                    <li>Stopword removal</li>
                    <li>Stemming opsional</li>
                    <li>TF-IDF feature extraction</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="info-card" style="border-left:3px solid #E15759;">
                <div class="info-card-title" style="color:#E15759;">Jalur Transformer</div>
                <ul style="color:#374151;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                    <li>Light preprocessing</li>
                    <li>Tanpa stopword removal</li>
                    <li>Tanpa stemming</li>
                    <li>Tokenisasi memakai tokenizer model</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    insight(
        f"Metadata preprocessing aktif: <strong>{preprocessing_meta}</strong>",
        "warning",
    )

    tabs = st.tabs([
        "Case Folding",
        "Light Normalization",
        "Slang Normalization",
        "Stopword Removal",
        "Stemming",
    ])

    columns_to_show = [
        ("Case Folding", ["full_text", "text_casefold"]),
        ("Light Normalization", ["text_casefold", "text_light_normalized"]),
        ("Slang Normalization", ["text_light_normalized", "text_normalized"]),
        ("Stopword Removal", ["text_normalized", "text_stopword_removed"]),
        ("Stemming", ["text_stopword_removed", "text_stemmed"]),
    ]

    for tab, (_, cols_show) in zip(tabs, columns_to_show):
        with tab:
            st.dataframe(processed_df[cols_show].head(15), use_container_width=True)


elif selected_name == "Data Split":
    section_header("06", "Data Splitting")

    col_tbl, col_chart = st.columns([1, 1.5])
    split_df = artifacts["split_distribution"]

    with col_tbl:
        st.markdown("#### Distribusi per Label")
        st.dataframe(split_df.reset_index().rename(columns={"index": "Label"}), use_container_width=True, hide_index=True)

        st.markdown("#### Ringkasan Split")
        for split_name, color in [("Train", "#59A14F"), ("Validation", "#F28E2B"), ("Test", "#E15759")]:
            total = int(split_df.loc["Total", split_name])
            pct = total / len(artifacts["df_model"]) * 100
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:0.5rem 0.8rem;margin-bottom:0.4rem;
                            background:#ffffff;border-radius:8px;border-left:3px solid {color};">
                    <span style="font-weight:600;color:#111827;">{split_name}</span>
                    <span style="color:{color};font-weight:700;">{total:,} tweet ({pct:.1f}%)</span>
                </div>
                """.replace(",", "."),
                unsafe_allow_html=True,
            )

    with col_chart:
        st.markdown("#### Visualisasi Split")
        st.plotly_chart(split_distribution_chart(split_df), use_container_width=True)

    insight(
        "Split dilakukan stratified dengan random_state=42 agar distribusi label pada train, validation, dan test tetap proporsional.",
    )


elif selected_name == "Baseline TF-IDF + SVM":
    section_header("07", "Baseline TF-IDF + SVM")

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Tanpa Stemming", f"{model_results['TF-IDF + SVM tanpa stemming']['accuracy']*100:.2f}%", "Accuracy test set", "#4E79A7")
    with cols[1]:
        kpi_card("Dengan Stemming", f"{model_results['TF-IDF + SVM dengan stemming']['accuracy']*100:.2f}%", "Accuracy test set", "#59A14F")
    with cols[2]:
        kpi_card("Fitur No Stem", f"{artifacts['feature_shapes']['tanpa_stemming'][1]:,}".replace(",", "."), "TF-IDF features", "#F28E2B")
    with cols[3]:
        kpi_card("Fitur Stem", f"{artifacts['feature_shapes']['dengan_stemming'][1]:,}".replace(",", "."), "TF-IDF features", "#B07AA1")

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(accuracy_chart(model_results), use_container_width=True)

    selected_model_eval = st.selectbox(
        "Pilih skenario evaluasi",
        list(model_results.keys()),
        index=0 if best_svm_model == "TF-IDF + SVM tanpa stemming" else 1,
    )
    selected_result = model_results[selected_model_eval]
    report_df = report_to_dataframe(selected_result["report_dict"])

    tab_report, tab_metric, tab_cm = st.tabs(["📋 Classification Report", "📊 Metrik per Kelas", "🧮 Confusion Matrix"])

    with tab_report:
        st.markdown(f'<div class="code-output">{selected_result["report_text"]}</div>', unsafe_allow_html=True)
        st.dataframe(
            report_df.style.format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "Support": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab_metric:
        st.plotly_chart(metrics_bar(report_df), use_container_width=True)

    with tab_cm:
        st.plotly_chart(
            confusion_matrix_figure(selected_result["cm"], f"Confusion Matrix - {selected_model_eval}"),
            use_container_width=True,
        )


elif selected_name == "Prediksi":
    section_header("08", "Prediksi Sentimen Tweet Baru")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Model Deployment Aktif</div>
            <p style="color:#374151;font-size:0.9rem;line-height:1.7;margin:0;">
                Halaman ini memakai model TF-IDF + SVM yang dilatih langsung dari dataset saat app berjalan.
                Ini adalah versi deployable yang ringan untuk GitHub + Streamlit Cloud.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_model_key = st.selectbox("Pilih model prediksi", list(model_results.keys()), index=0)
    input_text = st.text_area(
        "Masukkan teks tweet",
        height=150,
        placeholder="Contoh: AI bisa membuka peluang kerja baru, tapi juga menuntut skill yang lebih tinggi.",
    )

    if st.button("Prediksi Sentimen", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            pred = predict_text(input_text, processed_df, artifacts, pred_model_key)
            color = LABEL_COLORS.get(pred["label"], "#4E79A7")
            st.markdown(
                f"""
                <div class="best-model-card" style="border-color:{color};">
                    <div class="best-badge" style="background:{color};">Hasil Prediksi</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:{color};text-transform:capitalize;">
                        {pred["label"]}
                    </div>
                    <div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem;">
                        Pseudo-confidence berbasis margin SVM: {pred["confidence"]*100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Lihat teks setelah preprocessing"):
                st.write("**Normalized text:**")
                st.code(pred["normalized_text"])
                st.write("**Teks masuk model:**")
                st.code(pred["model_text"])


elif selected_name == "Hasil Transformer":
    section_header("09", "Visualisasi Hasil Modeling Transformer")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Ringkasan</div>
            <p style="color:#374151;font-size:0.9rem;line-height:1.7;margin:0;">
                Halaman ini memvisualisasikan hasil eksperimen Transformer dari notebook skripsi.
                Fine-tuning tidak dijalankan ulang di Streamlit agar dashboard tetap stabil saat deploy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    transformer_models = ["IndoBERT", "IndoBERTweet", "IndoBERTweet tanpa normalisasi"]
    selected_transformer = st.selectbox(
        "Pilih model Transformer",
        transformer_models,
        index=0,
    )

    metrics = RESEARCH_MODEL_RESULTS[selected_transformer]
    rows = TRANSFORMER_CLASS_REPORTS[selected_transformer]

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Model", selected_transformer, "Transformer", MODEL_COLORS.get(selected_transformer, "#E15759"))
    with cols[1]:
        kpi_card("Accuracy", f"{metrics['accuracy']*100:.2f}%", "Test set", "#59A14F")
    with cols[2]:
        kpi_card("Macro F1", f"{metrics['macro_f1']:.2f}", "Rata-rata kelas", "#F28E2B")
    with cols[3]:
        kpi_card("Weighted F1", f"{metrics['weighted_f1']:.2f}", "Support-weighted", "#B07AA1")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.15, 0.85])
    with col_left:
        st.markdown("#### Kurva Training")
        st.plotly_chart(
            transformer_training_chart(
                TRANSFORMER_TRAINING[selected_transformer],
                f"{selected_transformer} — Loss dan Validation Accuracy per Epoch",
            ),
            use_container_width=True,
        )

    with col_right:
        st.markdown("#### Confusion Matrix")
        st.plotly_chart(
            confusion_matrix_figure(
                TRANSFORMER_CONFUSION_MATRICES[selected_transformer],
                f"Confusion Matrix - {selected_transformer}",
                height=330,
            ),
            use_container_width=True,
        )

    st.markdown("#### Metrik per Kelas")
    st.plotly_chart(
        class_metrics_bar(
            rows,
            f"{selected_transformer} — Precision, Recall, F1-score",
            height=360,
        ),
        use_container_width=True,
    )

    st.markdown("#### Classification Report")
    report_df = pd.DataFrame(rows)
    st.dataframe(
        report_df.style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-score": "{:.2f}",
            "Support": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Perbandingan Transformer")
    transformer_compare = research_model_comparison_dataframe()
    transformer_compare = transformer_compare[
        transformer_compare["Model"].isin(transformer_models)
    ].copy()
    st.dataframe(
        transformer_compare.style.format({
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.2f}",
            "Macro Recall": "{:.2f}",
            "Macro F1": "{:.2f}",
            "Weighted Precision": "{:.2f}",
            "Weighted Recall": "{:.2f}",
            "Weighted F1": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )



elif selected_name == "Model Comparison":
    section_header("10", "Perbandingan Semua Model")

    comparison_df = research_model_comparison_dataframe()
    comparison_df.insert(0, "Rank", comparison_df.index + 1)

    cols = st.columns(4)
    with cols[0]:
        dual_best_model_card(
            best_research_model_cards,
            sub="Peringkat #1 bersama" if len(best_research_model_cards) > 1 else best_research_model_sub,
            color="#E15759",
        )
    with cols[1]:
        kpi_card("Best Accuracy", f"{comparison_df.loc[0, 'Accuracy']*100:.2f}%", "Test set", "#59A14F")
    with cols[2]:
        kpi_card("Jumlah Model", str(len(comparison_df)), "Skenario modeling", "#4E79A7")
    with cols[3]:
        kpi_card("Data Biner", f"{len(artifacts['df_model']):,}".replace(",", "."), "Tweet siap model", "#F28E2B")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Ranking Accuracy")
    st.plotly_chart(research_accuracy_chart(height=360), use_container_width=True)

    st.markdown("#### Tabel Evaluasi Utama")
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.2f}",
            "Macro Recall": "{:.2f}",
            "Macro F1": "{:.2f}",
            "Weighted Precision": "{:.2f}",
            "Weighted Recall": "{:.2f}",
            "Weighted F1": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    insight(
        f"Dua model terbaik pada hasil modeling skripsi adalah <strong>IndoBERT</strong> "
        f"dan <strong>IndoBERTweet</strong> dengan accuracy yang sama, yaitu "
        f"<strong>{best_research_accuracy*100:.2f}%</strong>.",
        "success",
    )



elif selected_name == "Frasa Dominan":
    section_header("11", "Analisis Frasa Dominan per Sentimen")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Metode</div>
            <p style="color:#374151;font-size:0.9rem;line-height:1.7;margin:0;">
                Analisis ini memakai TF-IDF n-gram 2 sampai 4 kata untuk mencari frasa dominan
                pada masing-masing label sentimen. Perhitungan dilakukan langsung dari dataset aktif.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        top_issues = dominant_issues(processed_df)
        selected_issue_label = st.selectbox("Pilih kategori sentimen", LABEL_ORDER)
        st.plotly_chart(issue_chart(top_issues, selected_issue_label), use_container_width=True)
        st.dataframe(
            top_issues[top_issues["label"] == selected_issue_label].style.format({"skor_tfidf_rata_rata": "{:.5f}", "skor_pembeda": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.error("Analisis Frasa dominan belum bisa dihitung.")
        st.exception(exc)
