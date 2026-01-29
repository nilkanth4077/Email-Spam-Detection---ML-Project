import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

st.title("📧 Email Spam Detection System")

email = st.text_area("Enter Email Text")

if st.button("Predict"):
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector).max()

    if pred == 1:
        st.error(f"🚨 Spam | Confidence: {round(prob*100,2)}%")
    else:
        st.success(f"✅ Not Spam | Confidence: {round(prob*100,2)}%")