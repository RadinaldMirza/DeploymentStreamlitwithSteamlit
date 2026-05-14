"""
AI Work Sentiment Dashboard - Streamlit Deployable App

Dashboard skripsi untuk visualisasi dan deployment modeling analisis sentimen
publik terhadap Artificial Intelligence dalam dunia kerja berbasis pengetahuan
di Indonesia.

Versi ini tidak lagi membaca file HTML hasil export notebook. App membaca
Dataset Final.csv, menjalankan preprocessing, split data, feature extraction,
training baseline TF-IDF + SVM, evaluasi, dan prediksi secara langsung saat
Streamlit berjalan.

Catatan:
- Fine-tuning IndoBERT/IndoBERTweet disediakan sebagai opsi manual karena proses
  training Transformer berat untuk Streamlit Cloud.
- Untuk deployment ringan, model yang otomatis aktif adalah TF-IDF + SVM.
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
DEFAULT_DATASET = BASE_DIR / "Dataset Final.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_FROM_REMAINING = 0.125
LABEL_ORDER = ["ancaman", "peluang", "netral"]

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
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-card-hover: #1c2128;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
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
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
    min-height: 132px;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-color, #4E79A7);
    border-radius: 12px 12px 0 0;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #484f58;
}

.kpi-label {
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.45rem;
}

.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    line-height: 1.2;
    color: #e6edf3;
}

.kpi-sub {
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 0.35rem;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #30363d;
}

.section-badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #e6edf3;
    margin: 0;
}

.info-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.info-card-title {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.6rem;
}

.best-model-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #1c2030 100%);
    border: 1px solid #E15759;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
}

.best-badge {
    display: inline-block;
    background: #E15759;
    color: white;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

.code-output {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #adbac7;
    overflow-x: auto;
    white-space: pre;
}

.insight-box {
    background: linear-gradient(135deg, #1a2233 0%, #182032 100%);
    border-left: 3px solid #4E79A7;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c9d1d9;
}

.insight-box.warning {
    border-left-color: #F0C040;
    background: linear-gradient(135deg, #1f1e0d 0%, #1e1b00 100%);
}

.insight-box.success {
    border-left-color: #59A14F;
    background: linear-gradient(135deg, #0f1f0d 0%, #0a1a08 100%);
}

.dashboard-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e6edf3;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.dashboard-subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    font-weight: 400;
    line-height: 1.5;
}

.sidebar-nav-label {
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    padding: 0.5rem 0 0.25rem;
}

[data-testid="stDataFrame"] {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
}

details {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

hr {
    border-color: #30363d !important;
    margin: 1.5rem 0;
}

h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #e6edf3 !important;
    opacity: 1 !important;
}

.stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"] {
    color: #c9d1d9 !important;
}

button[data-baseweb="tab"] p {
    color: #8b949e !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: #ff4b4b !important;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] label p {
    color: #c9d1d9 !important;
    font-size: 0.9rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.15rem 0 !important;
}

.stAlert {
    border-radius: 8px !important;
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
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent-color:{color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_dark_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#e6edf3"),
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        height=height,
    )
    fig.update_xaxes(gridcolor="#21262d", linecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", linecolor="#30363d")
    return fig


# -----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
def read_csv_safely(source: Any) -> pd.DataFrame:
    """Read CSV with semicolon first because notebook uses sep=';'."""
    try:
        return pd.read_csv(source, sep=";", encoding="utf-8-sig")
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_dataset_from_path(path: str) -> pd.DataFrame:
    return read_csv_safely(path)


def standardize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "manual_labeling" not in df.columns and "Label Final" in df.columns:
        df["manual_labeling"] = df["Label Final"]
    if "manual_labeling" not in df.columns:
        candidate_cols = [c for c in df.columns if "label" in c.lower()]
        if candidate_cols:
            df["manual_labeling"] = df[candidate_cols[0]]
    if "full_text" not in df.columns:
        text_candidates = [c for c in df.columns if c.lower() in {"text", "tweet", "content"}]
        if text_candidates:
            df["full_text"] = df[text_candidates[0]]
    required_cols = {"full_text", "manual_labeling"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset harus punya kolom full_text dan manual_labeling/Label Final. "
            f"Kolom yang hilang: {', '.join(sorted(missing))}."
        )

    df["manual_labeling"] = df["manual_labeling"].astype(str).str.strip().str.lower()
    df.loc[df["manual_labeling"].isin(["", "nan", "none", "null"]), "manual_labeling"] = np.nan
    df["label_final"] = df["manual_labeling"]
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
                "Jumlah data",
                "full_text kosong",
                "manual_labeling kosong",
                "Duplikat full_text",
            ],
            "hasil": [
                len(df),
                int(df["full_text"].isna().sum()),
                int(df["manual_labeling"].isna().sum()),
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
    """Self-contained issue analysis adapted from notebook section 4.17."""
    df_issues = df.dropna(subset=["label_final"]).copy()

    def normalize_issue_text(text: Any) -> str:
        text = str(text).lower()
        text = URL_PATTERN.sub(" ", text)
        text = MENTION_PATTERN.sub(" ", text)
        text = HASHTAG_PATTERN.sub(r"\1", text)
        text = NON_ALNUM_PATTERN.sub(" ", text)
        return MULTISPACE_PATTERN.sub(" ", text).strip()

    basic_stopwords_issue = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "dalam", "pada", "ini", "itu",
        "atau", "juga", "karena", "kalau", "akan", "bisa", "ada", "jadi", "lebih", "sudah",
        "saja", "sangat", "sebagai", "para", "aku", "kamu", "kita", "mereka", "dia",
        "the", "to", "of", "in", "is", "are", "a", "an",
        "rt", "amp", "https", "http", "co", "t", "nya", "nih", "sih", "dong", "kok", "lah",
        "ya", "ga", "gak", "nggak", "yg",
    }
    keep_issue_words = {"ai", "kerja", "pekerjaan", "phk", "skill", "karir", "peluang", "ancaman"}

    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        stopwords_issue = set(StopWordRemoverFactory().get_stop_words()) | basic_stopwords_issue
    except Exception:
        stopwords_issue = set(basic_stopwords_issue)

    stopwords_issue = sorted(stopwords_issue - keep_issue_words)
    df_issues["issue_text"] = df_issues["full_text"].apply(normalize_issue_text)

    issue_vectorizer = TfidfVectorizer(
        ngram_range=(2, 3),
        min_df=3,
        max_df=0.80,
        stop_words=stopwords_issue,
        token_pattern=r"(?u)\b\w\w+\b",
        sublinear_tf=True,
    )
    X_issue_tfidf = issue_vectorizer.fit_transform(df_issues["issue_text"])
    issue_features = issue_vectorizer.get_feature_names_out()
    issue_analyzer = issue_vectorizer.build_analyzer()

    rows = []
    for label in LABEL_ORDER:
        label_indices = df_issues.index[df_issues["label_final"] == label].to_numpy()
        if len(label_indices) == 0:
            continue
        label_scores = X_issue_tfidf[df_issues.index.get_indexer(label_indices)].mean(axis=0).A1
        top_indices = label_scores.argsort()[::-1][:12]
        label_texts = df_issues.loc[label_indices, "issue_text"]
        label_doc_ngrams = [set(issue_analyzer(text)) for text in label_texts]

        for rank, feature_idx in enumerate(top_indices, start=1):
            phrase = issue_features[feature_idx]
            jumlah_tweet = sum(phrase in doc_ngrams for doc_ngrams in label_doc_ngrams)
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "frasa_dominan": phrase,
                    "jumlah_tweet": int(jumlah_tweet),
                    "skor_tfidf_rata_rata": float(label_scores[feature_idx]),
                }
            )
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
            marker=dict(colors=[LABEL_COLORS.get(str(lbl).lower(), "#8b949e") for lbl in labels]),
            textfont=dict(color="#e6edf3", size=12),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        annotations=[
            dict(
                text=f"{int(sum(values)):,}<br>Tweet".replace(",", "."),
                x=0.5,
                y=0.5,
                font_size=15,
                font_color="#e6edf3",
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
    fig.update_layout(title=dict(text=title, font=dict(color="#e6edf3", size=14)))
    return plotly_dark_layout(fig, height=430)


def split_distribution_chart(split_df: pd.DataFrame) -> go.Figure:
    plot_df = split_df.drop(index="Total", errors="ignore").reset_index().rename(columns={"index": "label"})
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
            textfont=dict(color="#e6edf3", size=11),
        )
    )
    fig.update_xaxes(range=[max(0, df_plot["Accuracy"].min() - 0.05), min(1, df_plot["Accuracy"].max() + 0.05)])
    return plotly_dark_layout(fig, height=280)


def confusion_matrix_figure(cm: np.ndarray, title: str) -> go.Figure:
    fig = px.imshow(
        cm,
        x=LABEL_ORDER,
        y=LABEL_ORDER,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Prediksi", y="Label Aktual", color="Jumlah"),
    )
    fig.update_layout(title=dict(text=title, font=dict(color="#e6edf3", size=14)))
    return plotly_dark_layout(fig, height=380)


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
    plot_data = (
        top_issues[top_issues["label"] == label]
        .head(10)
        .sort_values("skor_tfidf_rata_rata", ascending=True)
    )
    fig = go.Figure(
        go.Bar(
            x=plot_data["skor_tfidf_rata_rata"],
            y=plot_data["frasa_dominan"],
            orientation="h",
            marker_color=LABEL_COLORS.get(label, "#4E79A7"),
            text=[f"{x:.4f} | {c} tweet" for x, c in zip(plot_data["skor_tfidf_rata_rata"], plot_data["jumlah_tweet"])],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=10),
        )
    )
    fig.update_layout(title=dict(text=f"Isu Dominan: {label.capitalize()}", font=dict(color="#e6edf3", size=14)))
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

    label2id = {"ancaman": 0, "peluang": 1, "netral": 2}
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
            <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#e6edf3;">AI Work Sentiment</div>
            <div style="font-size:0.7rem;color:#8b949e;">Dashboard Skripsi · Deployable</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-nav-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Dataset Final.csv", type=["csv"], label_visibility="collapsed")

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
        ("🧠", "Transformer Opsional"),
        ("🏆", "Model Comparison"),
        ("🧩", "Isu Dominan"),
    ]
    page_labels = [f"{icon} {name}" for icon, name in pages]
    selected = st.radio("", page_labels, label_visibility="collapsed")
    selected_name = selected.split(" ", 1)[1]

    st.markdown("---")
    if uploaded_file is not None:
        st.success("Dataset dari uploader aktif.")
    elif DEFAULT_DATASET.exists():
        st.success("Dataset Final.csv ditemukan di repo.")
    else:
        st.error("Dataset Final.csv belum ditemukan.")

    st.markdown(
        """
        <div style="font-size:0.68rem;color:#484f58;text-align:center;line-height:1.6;margin-top:1rem;">
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
        raw_df = load_dataset_from_path(str(DEFAULT_DATASET))

    with st.spinner("Menjalankan pipeline dari dataset..."):
        processed_df, preprocessing_meta = preprocess_dataset(raw_df, use_stemming=use_stemming)
        artifacts = split_and_train(processed_df)

except Exception as exc:
    st.error("App belum bisa dijalankan karena dataset belum valid.")
    st.exception(exc)
    st.stop()


label_counts = processed_df["label_final"].value_counts().reindex(LABEL_ORDER).dropna()
model_results = artifacts["results"]
best_live_model = max(model_results.items(), key=lambda item: item[1]["accuracy"])[0]
best_live_accuracy = model_results[best_live_model]["accuracy"]


# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------
if selected_name == "Overview":
    st.markdown(
        """
        <div style="margin-bottom:2rem;">
            <div class="dashboard-title">AI Work Sentiment Dashboard</div>
            <div class="dashboard-subtitle">
                Visualisasi dan deployment modeling analisis sentimen publik terhadap Artificial Intelligence<br/>
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
            <div class="info-card-title">Tentang Dashboard Baru</div>
            <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;margin:0;">
                Dashboard ini membaca <code>Dataset Final.csv</code> secara langsung, menjalankan preprocessing,
                data splitting, TF-IDF feature extraction, training SVM, evaluasi, dan prediksi saat aplikasi berjalan.
                Dengan begitu, dashboard tidak lagi bergantung pada file HTML export notebook.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    kpis = [
        ("Total Data", f"{len(processed_df):,}".replace(",", "."), "Tweet", "#4E79A7"),
        ("Data Berlabel", f"{len(artifacts['df_model']):,}".replace(",", "."), "Siap model", "#59A14F"),
        ("Kategori Label", f"{processed_df['label_final'].nunique()}", "Ancaman · Peluang · Netral", "#F28E2B"),
        ("Best Live Model", "SVM", best_live_model.replace("TF-IDF + ", ""), "#E15759"),
        ("Best Live Accuracy", f"{best_live_accuracy*100:.2f}%", "Dihitung saat app berjalan", "#B07AA1"),
    ]
    for col, (label, value, sub, color) in zip(cols, kpis):
        with col:
            kpi_card(label, value, sub, color)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.35])
    with col_left:
        section_header("02", "Status Pipeline")
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-title">Pipeline yang Berjalan</div>
                <ul style="color:#c9d1d9;font-size:0.9rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                    <li>Load dataset CSV</li>
                    <li>Case folding</li>
                    <li>Light normalization</li>
                    <li>Slang normalization</li>
                    <li>Stopword removal</li>
                    <li>Stemming opsional</li>
                    <li>Split 70% : 10% : 20%</li>
                    <li>TF-IDF + LinearSVC</li>
                </ul>
                <p style="color:#8b949e;font-size:0.78rem;margin-top:0.8rem;margin-bottom:0;">
                    {preprocessing_meta}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        section_header("03", "Distribusi Label")
        st.plotly_chart(label_distribution_chart(label_counts), use_container_width=True)

    insight(
        f"Model live terbaik pada dataset yang sedang dibaca adalah <strong>{best_live_model}</strong> "
        f"dengan accuracy <strong>{best_live_accuracy*100:.2f}%</strong> pada test set.",
        "success",
    )


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
                <ul style="color:#c9d1d9;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
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
                <ul style="color:#c9d1d9;font-size:0.88rem;line-height:1.9;padding-left:1.2rem;margin:0;">
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
                            background:#161b22;border-radius:8px;border-left:3px solid {color};">
                    <span style="font-weight:600;color:#e6edf3;">{split_name}</span>
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
        index=0 if best_live_model == "TF-IDF + SVM tanpa stemming" else 1,
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
            <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;margin:0;">
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
                    <div style="color:#8b949e;font-size:0.85rem;margin-top:0.3rem;">
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


elif selected_name == "Transformer Opsional":
    section_header("09", "Fine-Tuning Transformer Opsional")

    st.warning(
        "Fine-tuning IndoBERT/IndoBERTweet tidak dijalankan otomatis saat app dibuka karena berat untuk deployment. "
        "Gunakan tombol di bawah hanya jika environment Streamlit punya resource dan dependency yang cukup."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        transformer_choice = st.selectbox(
            "Pilih skenario",
            [
                "IndoBERT",
                "IndoBERTweet",
                "IndoBERTweet tanpa normalisasi",
            ],
        )
    with col_b:
        epochs = st.number_input("Epoch", min_value=1, max_value=5, value=1, step=1)

    batch_size = st.selectbox("Batch size", [4, 8, 16], index=1)

    if transformer_choice == "IndoBERT":
        hf_model_name = "indobenchmark/indobert-base-p1"
        text_col = "text_light_normalized"
    elif transformer_choice == "IndoBERTweet":
        hf_model_name = "indolem/indobertweet-base-uncased"
        text_col = "text_light_normalized"
    else:
        hf_model_name = "indolem/indobertweet-base-uncased"
        processed_df["text_for_indobertweet_raw"] = processed_df["full_text"].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())
        artifacts["df_model"]["text_for_indobertweet_raw"] = processed_df.loc[artifacts["df_model"].index, "text_for_indobertweet_raw"]
        text_col = "text_for_indobertweet_raw"

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">Konfigurasi</div>
            <ul style="color:#c9d1d9;font-size:0.9rem;line-height:1.9;padding-left:1.2rem;margin:0;">
                <li>Model Hugging Face: <code>{hf_model_name}</code></li>
                <li>Input column: <code>{text_col}</code></li>
                <li>Epoch: {epochs}</li>
                <li>Batch size: {batch_size}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Jalankan Fine-Tuning Transformer Sekarang", type="primary", use_container_width=True):
        try:
            with st.spinner("Training Transformer berjalan. Jangan refresh halaman sampai selesai."):
                start = time.time()
                transformer_result = train_transformer_optional(
                    model_name=hf_model_name,
                    text_col=text_col,
                    df_model=artifacts["df_model"],
                    train_idx=artifacts["train_idx"],
                    val_idx=artifacts["val_idx"],
                    test_idx=artifacts["test_idx"],
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                )
                transformer_result["elapsed_seconds"] = time.time() - start
                st.session_state.setdefault("transformer_results", {})
                st.session_state["transformer_results"][transformer_choice] = transformer_result

            st.success(
                f"Training selesai. Accuracy {transformer_choice}: "
                f"{transformer_result['accuracy']*100:.2f}% "
                f"({transformer_result['elapsed_seconds']/60:.1f} menit)."
            )
        except Exception as exc:
            st.error("Training Transformer belum berhasil dijalankan di environment ini.")
            st.exception(exc)

    transformer_results = st.session_state.get("transformer_results", {})
    if transformer_results:
        st.markdown("#### Hasil Transformer pada Session Ini")
        rows = [{"Model": k, "Accuracy": v["accuracy"], "Output Dir": v.get("output_dir", "-")} for k, v in transformer_results.items()]
        st.dataframe(pd.DataFrame(rows).style.format({"Accuracy": "{:.4f}"}), use_container_width=True, hide_index=True)


elif selected_name == "Model Comparison":
    section_header("10", "Perbandingan Model")

    rows = [
        {
            "Model": name,
            "Accuracy": result["accuracy"],
            "Sumber": "Live training saat app berjalan",
            "Catatan": "TF-IDF + LinearSVC",
        }
        for name, result in model_results.items()
    ]

    for name, result in st.session_state.get("transformer_results", {}).items():
        rows.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Sumber": "Fine-tuning Transformer pada session Streamlit ini",
                "Catatan": result.get("model_name", ""),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    comparison_df.insert(0, "Rank", comparison_df.index + 1)

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Best Model", comparison_df.loc[0, "Model"], "Ranking #1", "#E15759")
    with cols[1]:
        kpi_card("Best Accuracy", f"{comparison_df.loc[0, 'Accuracy']*100:.2f}%", "Test set", "#59A14F")
    with cols[2]:
        kpi_card("Model Aktif", str(len(comparison_df)), "Live comparison", "#4E79A7")
    with cols[3]:
        kpi_card("Dataset", f"{len(processed_df):,}".replace(",", "."), "Tweet", "#F28E2B")

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(
        comparison_df.style.format({"Accuracy": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    fig = go.Figure(
        go.Bar(
            x=comparison_df.sort_values("Accuracy")["Accuracy"],
            y=comparison_df.sort_values("Accuracy")["Model"],
            orientation="h",
            marker_color=[MODEL_COLORS.get(x, "#4E79A7") for x in comparison_df.sort_values("Accuracy")["Model"]],
            text=[f"{x:.4f}" for x in comparison_df.sort_values("Accuracy")["Accuracy"]],
            textposition="outside",
            textfont=dict(color="#e6edf3"),
        )
    )
    st.plotly_chart(plotly_dark_layout(fig, height=330), use_container_width=True)

    insight(
        "Perbandingan ini memprioritaskan hasil yang benar-benar dihitung saat aplikasi berjalan. "
        "Jika Transformer belum dijalankan di halaman opsional, tabel hanya menampilkan baseline SVM live.",
        "warning",
    )


elif selected_name == "Isu Dominan":
    section_header("11", "Analisis Isu Dominan per Sentimen")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Metode</div>
            <p style="color:#c9d1d9;font-size:0.9rem;line-height:1.7;margin:0;">
                Analisis ini memakai TF-IDF n-gram 2 sampai 3 kata untuk mencari frasa dominan
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
            top_issues[top_issues["label"] == selected_issue_label].style.format({"skor_tfidf_rata_rata": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.error("Analisis isu dominan belum bisa dihitung.")
        st.exception(exc)
