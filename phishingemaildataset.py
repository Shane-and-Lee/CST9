import re
import joblib
from scipy.sparse import hstack, csr_matrix

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phishing Email Detection", layout="wide")

st.title("Phishing Email Detection Using Random Forest")
st.caption("Results summary from the Google Colab notebook (values copied from the notebook run)")

st.header("Try it: classify an email")

@st.cache_resource
def load_models():
    return joblib.load("tfidf.joblib"), joblib.load("rf_model.joblib")

tfidf, rf_model = load_models()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict(raw):
    X_text = tfidf.transform([clean_text(raw)])
    extra = csr_matrix([[len(raw),
                         len(re.findall(r"http\S+|www\.\S+", raw)),
                         sum(c.isdigit() for c in raw)]])
    X = hstack([X_text, extra])
    return rf_model.predict(X)[0], rf_model.predict_proba(X)[0]

email = st.text_area("Paste an email here:", height=180)
if st.button("Classify") and email.strip():
    label, probs = predict(email)
    if label == 1:
        st.error(f"Likely PHISHING ({probs[1]:.1%} confidence)")
    else:
        st.success(f"Likely legitimate ({probs[0]:.1%} confidence)")

# ---------- Dataset ----------
st.header("Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw records", "82,486")
c2.metric("Unique records after cleaning", "80,726")
c3.metric("Training / testing", "64,580 / 16,146")
c4.metric("Features", "5,003")

st.write(
    "Duplicates were removed in two passes: 408 exact duplicates, then 1,352 more "
    "that only became duplicates after the text was normalized."
)

st.subheader("Class balance (before duplicate removal)")
balance = pd.DataFrame(
    {"Emails": [39595, 42891]},
    index=["Legitimate (0)", "Phishing (1)"],
)
st.bar_chart(balance)

# ---------- Model performance ----------
st.header("Random Forest performance (test set)")
m1, m2 = st.columns(2)
m1.metric("Accuracy", "98.48%")
m2.metric("ROC-AUC", "0.9983")

report = pd.DataFrame(
    {
        "Precision": [0.9828, 0.9866],
        "Recall": [0.9850, 0.9847],
        "F1-score": [0.9839, 0.9856],
        "Support": [7602, 8544],
    },
    index=["Legitimate (0)", "Phishing (1)"],
)
st.dataframe(report)

st.subheader("Confusion matrix")
cm = pd.DataFrame(
    [[7488, 114], [131, 8413]],
    index=["Actual: Legitimate", "Actual: Phishing"],
    columns=["Predicted: Legitimate", "Predicted: Phishing"],
)
st.dataframe(cm)

# ---------- Feature importance ----------
st.header("Top 20 most important features")
features = pd.DataFrame(
    {
        "Importance": [
            0.0274, 0.0266, 0.0229, 0.0215, 0.0171,
            0.0152, 0.0141, 0.0141, 0.0119, 0.0088,
            0.0083, 0.0081, 0.0081, 0.0080, 0.0076,
            0.0071, 0.0069, 0.0068, 0.0063, 0.0062,
        ]
    },
    index=[
        "wrote", "aug", "enron", "digit_count", "thanks",
        "text_length", "url_count", "pm", "list", "would",
        "university", "cc", "group", "mailing", "im",
        "money", "vince", "opensuse", "attached", "subject",
    ],
)
st.bar_chart(features)
st.dataframe(features)

st.info(
    "The model partly relies on source-specific vocabulary (for example 'enron' and "
    "'opensuse'), so these results reflect performance on this dataset and still need "
    "to be tested on emails from unseen sources."
)