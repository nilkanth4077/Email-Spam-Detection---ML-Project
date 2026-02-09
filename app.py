import streamlit as st
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ NLTK Setup ------------------ #
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------ Text Cleaning ------------------ #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ------------------ Load Dataset ------------------ #
@st.cache_data
def load_data():
    df = pd.read_csv("final_spam_dataset.csv")
    df['clean_text'] = df['email_text'].apply(clean_text)
    return df

df = load_data()

# ------------------ Train Test Split ------------------ #
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------ Vectorization ------------------ #
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------ Model Training ------------------ #
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ------------------ Evaluation ------------------ #
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# ------------------ Streamlit UI ------------------ #
st.title("📧 Email Spam Detection System")

st.markdown(f"""
**Model:** Multinomial Naive Bayes  
**Vectorization:** TF-IDF (Unigram + Bigram)  
**Accuracy:** `{round(accuracy * 100, 2)}%`
""")

# ------------------ Prediction ------------------ #
st.subheader("🔍 Spam Prediction")

email = st.text_area("Enter Email Text")

if st.button("Predict"):
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector).max()

    if pred == 1:
        st.error(f"🚨 Spam Email | Confidence: {round(prob*100, 2)}%")
    else:
        st.success(f"✅ Not Spam | Confidence: {round(prob*100, 2)}%")

# ------------------ Evaluation Details ------------------ #
with st.expander("📈 Model Evaluation Details"):
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

# ------------------ Dataset Preview ------------------ #
st.subheader("📊 Dataset Preview")

with st.expander("Click to view dataset"):
    st.write("First 10 rows:")
    st.dataframe(df.head(10))

    st.write("Dataset Shape:")
    st.write(df.shape)

    st.write("Class Distribution:")
    st.write(df['label'].value_counts())