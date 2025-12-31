# app.py
import os
import io
import re
import pickle
import streamlit as st

# Optional PDF text extraction
try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

import nltk
from nltk.corpus import stopwords

# Ensure required NLTK data is available (quiet)
nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if pkg == "punkt" else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# Utility: extract text from PDF bytes
def extract_text_from_pdf_bytes(file_bytes):
    if not _HAS_PYPDF2:
        raise RuntimeError(
            "PyPDF2 is not installed. Install it or upload a text file instead."
        )
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

# Clean resume text
def clean_resume(resume_text: str) -> str:
    if not isinstance(resume_text, str):
        resume_text = str(resume_text)

    # Lowercase
    text = resume_text.lower()

    # Remove URLs, RT/cc, mentions, hashtags
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\brt\b|\bcc\b", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)

    # Remove non-ascii, punctuation, numbers, extra whitespace
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)

# Try to load models (give helpful UI message if missing)
MODEL_PKL = "clf.pkl"
TFIDF_PKL = "tfidf.pkl"

clf = None
tfidf = None

def load_models():
    global clf, tfidf
    try:
        with open(MODEL_PKL, "rb") as f:
            clf = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_PKL}' not found. Place the trained classifier (pickle) in the project root.")
        return False
    except Exception as e:
        st.error(f"Failed to load '{MODEL_PKL}': {e}")
        return False

    try:
        with open(TFIDF_PKL, "rb") as f:
            tfidf = pickle.load(f)
    except FileNotFoundError:
        st.error(f"TF-IDF file '{TFIDF_PKL}' not found. Place the tfidf vectorizer (pickle) in the project root.")
        return False
    except Exception as e:
        st.error(f"Failed to load '{TFIDF_PKL}': {e}")
        return False

    return True

# Optional image display if exists
if os.path.exists("rrrr.jpg"):
    st.image("rrrr.jpg", width=120)

st.title("Resume Screening — Shortlist vs Reject")
st.markdown(
    "Upload a resume (TXT or PDF). This app will clean the text and run the trained classifier."
)

if load_models():
    uploaded = st.file_uploader("Upload resume file", type=["txt", "pdf"])
    if uploaded is not None:
        try:
            file_bytes = uploaded.read()
            if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                try:
                    raw_text = extract_text_from_pdf_bytes(file_bytes)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
            else:
                # try common encodings
                try:
                    raw_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        raw_text = file_bytes.decode("latin-1")
                    except Exception:
                        raw_text = str(file_bytes)

            cleaned = clean_resume(raw_text)
            if not cleaned:
                st.warning("No extractable text found in the uploaded file.")
            else:
                st.subheader("Cleaned resume (preview)")
                st.write(cleaned[:2000] + ("..." if len(cleaned) > 2000 else ""))

                # Transform and predict
                try:
                    X = tfidf.transform([cleaned])
                    pred = clf.predict(X)[0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                # Category mapping (example)
                category_mapping = {
                    6: "Data Science",
                    12: "HR",
                    0: "Advocate",
                    1: "Arts",
                    24: "Web Designing",
                    16: "Mechanical Engineer",
                    22: "Sales",
                    14: "Health and fitness",
                    5: "Civil Engineer",
                    15: "Java Developer",
                    4: "Business Analyst",
                    21: "SAP Developer",
                    2: "Automation Testing",
                    11: "Electrical Engineering",
                    18: "Operations Manager",
                    20: "Python Developer",
                    8: "DevOps Engineer",
                    17: "Network Security Engineer",
                    19: "PMO",
                    7: "Database",
                    13: "Hadoop",
                    10: "ETL Developer",
                    9: "DotNet Developer",
                    3: "Blockchain",
                    23: "Testing",
                }

                category_name = category_mapping.get(pred, "Unknown")
                st.success(f"Predicted category id: {pred} → {category_name}")

        except Exception as ex:
            st.error(f"Unexpected error reading file: {ex}")
else:
    st.info("Place the required model files (clf.pkl and tfidf.pkl) in the project root to enable predictions.")
