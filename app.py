import os
from pathlib import Path
import pickle

import mysql.connector
from mysql.connector import Error
import pandas as pd
import streamlit as st
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "imdb_bidirectional_gru.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN = 200
MAX_TEXT_LENGTH = 5000
MAX_CSV_ROWS = 1000
MAX_CSV_COLUMNS = 50
MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def get_db_config():
    """Retrieve MySQL database configuration securely from Streamlit Secrets or Environment Variables.

    Hardcoded credentials are strictly avoided.
    """
    config = {}

    # 1. Try Streamlit Secrets (.streamlit/secrets.toml or Streamlit Cloud Secrets)
    if "mysql" in st.secrets:
        sec = st.secrets["mysql"]
        config = {
            "host": str(sec.get("host", "localhost")),
            "port": int(sec.get("port", 3306)),
            "user": str(sec.get("user", "")),
            "password": str(sec.get("password", "")),
            "database": str(sec.get("database", "imdb_sentiment_db")),
            "autocommit": True,
            "connect_timeout": int(sec.get("connect_timeout", 5)),
        }
        if "ssl_ca" in sec:
            config["ssl_ca"] = sec["ssl_ca"]
        if "ssl_disabled" in sec:
            config["ssl_disabled"] = bool(sec["ssl_disabled"])
    # 2. Fallback to Environment Variables
    else:
        config = {
            "host": os.environ.get("MYSQL_HOST", "localhost"),
            "port": int(os.environ.get("MYSQL_PORT", 3306)),
            "user": os.environ.get("MYSQL_USER", ""),
            "password": os.environ.get("MYSQL_PASSWORD", ""),
            "database": os.environ.get("MYSQL_DATABASE", "imdb_sentiment_db"),
            "autocommit": True,
            "connect_timeout": int(os.environ.get("MYSQL_CONNECT_TIMEOUT", 5)),
        }
        if os.environ.get("MYSQL_SSL_CA"):
            config["ssl_ca"] = os.environ.get("MYSQL_SSL_CA")
        if os.environ.get("MYSQL_SSL_DISABLED"):
            config["ssl_disabled"] = os.environ.get("MYSQL_SSL_DISABLED").lower() in ("true", "1")

    return config


st.set_page_config(page_title="IMDB Sentiment Detector", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #0b1120 !important;
            color: #e5e7eb !important;
        }

        .stMain, .main {
            background: #0b1120 !important;
        }

        div[data-testid="stHorizontalBlock"] > div {
            padding: 0.25rem;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .card {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 1rem;
            color: #e2e8f0;
            box-shadow: none;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox div[role="combobox"],
        .stFileUploaderDropzone,
        .stButton > button,
        .stDownloadButton > button,
        .stDataFrame,
        [data-testid="stDataFrame"] {
            background: #111827 !important;
            color: #f8fafc !important;
            border: 1px solid rgba(96, 165, 250, 0.45) !important;
            border-radius: 10px !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
            color: white !important;
            border: none !important;
            font-weight: 600;
            box-shadow: none !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            filter: brightness(1.08);
        }

        .stAlert {
            background: rgba(15, 23, 42, 0.75) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
        }

        .stAlert p,
        .stAlert div,
        .stAlert strong,
        .stMarkdown {
            color: #f8fafc !important;
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
        }

        p, li, label, h1, h2, h3, h4, h5, h6,
        .stMarkdown {
            color: #f8fafc !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_sentiment(text: str):
    if not text or not text.strip():
        return None, None, None

    # Sanitize & limit max input length consistently
    cleaned_text = text.strip()[:MAX_TEXT_LENGTH]

    model, tokenizer = load_artifacts()
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    raw_score = float(model.predict(padded, verbose=0)[0][0])
    positive_probability = 1.0 - raw_score
    label = "Positive" if positive_probability >= 0.5 else "Negative"
    return label, positive_probability, raw_score


def get_db_connection():
    try:
        config = get_db_config()
        if not config.get("user"):
            st.session_state["db_status"] = "Database configuration missing (Secrets not configured)"
            return None
        return mysql.connector.connect(**config)
    except Error:
        st.session_state["db_status"] = "Database unavailable (Connection failed)"
        return None
    except Exception:
        st.session_state["db_status"] = "Database unavailable (Configuration error)"
        return None


def ensure_database_and_table():
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS review_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    review_text LONGTEXT NOT NULL,
                    predicted_label VARCHAR(20) NOT NULL,
                    positive_probability FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        finally:
            cursor.close()
        st.session_state["db_status"] = "Connected to MySQL"
        return True
    except Error:
        st.session_state["db_status"] = "MySQL available (Table setup pending / restricted)"
        return False
    finally:
        conn.close()


def save_prediction_to_history(review: str, label: str, probability: float):
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        try:
            # Parameterized query to prevent SQL Injection with consistent text length cap
            cursor.execute(
                "INSERT INTO review_history (review_text, predicted_label, positive_probability) VALUES (%s, %s, %s)",
                (review[:MAX_TEXT_LENGTH], label, float(probability)),
            )
        finally:
            cursor.close()
        return True
    except Error:
        st.session_state["db_status"] = "History save failed (Database write error)"
        return False
    finally:
        conn.close()


def load_history():
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame(columns=["id", "review_text", "predicted_label", "positive_probability", "created_at"])

    try:
        query = "SELECT id, review_text, predicted_label, positive_probability, created_at FROM review_history ORDER BY created_at DESC LIMIT 20"
        history_df = pd.read_sql(query, conn)
        return history_df
    except Exception:
        return pd.DataFrame(columns=["id", "review_text", "predicted_label", "positive_probability", "created_at"])
    finally:
        conn.close()


sample_reviews = [
    "This movie was absolutely fantastic and the acting was superb.",
    "I hated this film. The plot was boring and the ending was disappointing.",
    "A decent movie with some good moments, but it was a bit slow in the middle.",
    "I loved every minute of this movie. It was amazing and unforgettable.",
    "This was a terrible film, dull and annoying from start to finish.",
]

st.title("🎬 IMDB Review Sentiment Analyzer")
st.caption("Bi-directional GRU model + tokenizer from your project folder, with MySQL-based prediction history.")

ensure_database_and_table()

st.sidebar.header("Database status")
st.sidebar.write(st.session_state.get("db_status", "Checking database..."))

if st.sidebar.button("Refresh database connection"):
    ensure_database_and_table()

sample_gallery = st.container()
with sample_gallery:
    st.subheader("Sample review gallery")
    cols = st.columns(3)
    for idx, review_text in enumerate(sample_reviews):
        with cols[idx % 3]:
            st.markdown(f"<div class='card'>\n<strong>Sample {idx + 1}</strong><br>{review_text}<br></div>", unsafe_allow_html=True)
            if st.button("Use this sample", key=f"sample_{idx}"):
                st.session_state["sample_review"] = review_text
                st.rerun()

st.markdown("---")

left_col, right_col = st.columns([2, 1])
with left_col:
    review = st.text_area(
        "Enter a movie review",
        value=st.session_state.get("sample_review", ""),
        height=180,
        placeholder="Write a review here...",
    )

    predict_btn = st.button("Predict sentiment", use_container_width=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### Tips")
    st.write("- Use short positive or negative reviews")
    st.write("- Try mixed sentiment to test confidence")
    st.write("- Batch mode accepts CSV format")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    sanitized_review = review.strip()[:MAX_TEXT_LENGTH]
    if len(review) > MAX_TEXT_LENGTH:
        st.info(f"Review exceeds {MAX_TEXT_LENGTH} characters limit and was truncated consistently for analysis.")

    result = predict_sentiment(sanitized_review)
    if result[0] is None:
        st.warning("Please enter a valid review before predicting.")
    else:
        label, positive_probability, raw_score = result
        negative_probability = 1.0 - positive_probability
        save_prediction_to_history(sanitized_review, label, positive_probability)

        st.subheader("Prediction result")
        if label == "Positive":
            st.success(f"This review is predicted as {label}.")
        else:
            st.error(f"This review is predicted as {label}.")

        col_prob, col_raw = st.columns([3, 1])
        with col_prob:
            pos_col, neg_col = st.columns(2)
            with pos_col:
                st.write("Positive probability")
                st.progress(positive_probability)
                st.write(f"{positive_probability * 100:.1f}%")
            with neg_col:
                st.write("Negative probability")
                st.progress(negative_probability)
                st.write(f"{negative_probability * 100:.1f}%")
        with col_raw:
            st.metric("Model output", f"{raw_score:.4f}")

        st.write("### Review text")
        st.write(sanitized_review)

        if st.session_state.get("db_status") and "Connected" in st.session_state["db_status"]:
            st.caption("✅ Prediction saved to MySQL history.")
        else:
            st.caption("⚠️ Model prediction was generated, but MySQL history is unavailable right now.")

st.markdown("---")

st.subheader("CSV batch review analyzer")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # 1. Pre-upload file size check (Limit: 10 MB)
    if uploaded_file.size > MAX_CSV_SIZE_BYTES:
        st.error(f"Uploaded file exceeds maximum allowed size of {MAX_CSV_SIZE_BYTES / (1024 * 1024):.0f}MB.")
    else:
        df = pd.read_csv(uploaded_file)

        # 2. Limit excessive column counts
        if len(df.columns) > MAX_CSV_COLUMNS:
            st.warning(f"CSV contains {len(df.columns)} columns. Truncating to first {MAX_CSV_COLUMNS} columns for security.")
            df = df.iloc[:, :MAX_CSV_COLUMNS]

        # 3. Limit excessive row counts
        if len(df) > MAX_CSV_ROWS:
            st.warning(f"Uploaded CSV has {len(df)} rows. Processed row count has been capped to the first {MAX_CSV_ROWS} rows for security and performance.")
            df = df.head(MAX_CSV_ROWS)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        text_columns = [col for col in df.columns if df[col].dtype == "object"]
        if not text_columns:
            st.warning("The uploaded CSV does not contain a text-like column. Please make sure one column has review text.")
        else:
            review_column = st.selectbox("Choose the review column", text_columns)
            if st.button("Analyze CSV", key="batch_analysis"):
                results = df.copy()
                results["predicted_label"] = ""
                results["positive_probability"] = 0.0
                results["negative_probability"] = 0.0

                for idx, value in results[review_column].items():
                    # 4. Truncate cell text consistently to prevent memory/DB exhaustion
                    text = str(value).strip()[:MAX_TEXT_LENGTH]
                    label, probability, _ = predict_sentiment(text)
                    if label is None:
                        results.at[idx, "predicted_label"] = "Unknown"
                        results.at[idx, "positive_probability"] = 0.0
                        results.at[idx, "negative_probability"] = 0.0
                        continue
                    results.at[idx, "predicted_label"] = label
                    results.at[idx, "positive_probability"] = float(probability)
                    results.at[idx, "negative_probability"] = float(1.0 - probability)
                    save_prediction_to_history(text, label, float(probability))

                st.dataframe(results.head(20))
                csv_download = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predicted CSV",
                    data=csv_download,
                    file_name="imdb_sentiment_predictions.csv",
                    mime="text/csv",
                )

st.markdown("---")

st.subheader("Recent prediction history")
st.caption("Shared global application history (last 20 predictions across all users).")
history_df = load_history()
if history_df.empty:
    st.info("No saved history yet. Predictions will appear here once the MySQL database is available.")
else:
    st.dataframe(history_df, use_container_width=True)
