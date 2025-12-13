import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt

from PyPDF2 import PdfReader
from textblob import TextBlob
from wordcloud import WordCloud

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# ------------------ NLTK Downloads ------------------
nltk.download("punkt")
nltk.download("stopwords")

# ------------------ Helper Functions ------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'[^a-z0-9\.\?\!\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"


label_map = {"Negative": -1, "Neutral": 0, "Positive": 1}


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Sentiment Analysis App", layout="wide")

st.title("📊 PDF Sentiment Analysis & NLP Dashboard")
st.write("Upload a PDF to analyze sentiment, word frequency, word cloud, and ML models.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # ------------------ Read PDF ------------------
    reader = PdfReader(uploaded_file)
    pages = reader.pages

    all_text = ""
    for page in pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    st.success(f"PDF loaded successfully — {len(pages)} pages")

    # ------------------ Preprocess ------------------
    clean_text = preprocess_text(all_text)

    # ------------------ Sentence Tokenization ------------------
    sentences = sent_tokenize(clean_text)
    df_sentences = pd.DataFrame(sentences, columns=["sentence"])

    # ------------------ Sentiment Analysis ------------------
    df_sentences["sentiment"] = df_sentences["sentence"].apply(analyze_sentiment)

    st.subheader("📈 Sentiment Distribution")
    st.bar_chart(df_sentences["sentiment"].value_counts())

    # ------------------ Tokenization ------------------
    words = word_tokenize(clean_text)
    words = [w for w in words if w.isalnum()]
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words and len(w) > 2]

    # ------------------ Word Frequency ------------------
    freq_dist = FreqDist(words)
    freq_df = pd.DataFrame(freq_dist.most_common(20), columns=["Word", "Frequency"])

    st.subheader("🔤 Top 20 Frequent Words")
    st.dataframe(freq_df)

    # ------------------ Word Cloud ------------------
    st.subheader("☁️ Word Cloud")
    wc = WordCloud(
        width=1000,
        height=500,
        stopwords=stop_words,
        colormap="plasma",
        max_words=200
    ).generate(" ".join(words))

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

    # ------------------ Count Vectorizer ------------------
    cv = CountVectorizer(max_features=30, stop_words="english")
    cv_matrix = cv.fit_transform(df_sentences["sentence"])
    cv_df = pd.DataFrame(cv_matrix.toarray(), columns=cv.get_feature_names_out())

    st.subheader("📊 Document-Term Matrix (CountVectorizer)")
    st.dataframe(cv_df.head())

    # ------------------ TF-IDF ------------------
    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    X_tfidf = tfidf.fit_transform(df_sentences["sentence"])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    # ------------------ ML Preparation ------------------
    y = df_sentences["sentiment"].map(label_map)
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------ Logistic Regression ------------------
    lr = LogisticRegression(max_iter=1000, solver="saga", multi_class="multinomial")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    st.subheader("🤖 Logistic Regression Results")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_lr))
    st.text(classification_report(y_test, y_pred_lr))

    # ------------------ Decision Tree ------------------
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    st.subheader("🌳 Decision Tree Results")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_dt))
    st.text(classification_report(y_test, y_pred_dt))

    # ------------------ Naive Bayes ------------------
    nb = MultinomialNB()
    nb.fit(tfidf_df, y)

    st.subheader("📘 Naive Bayes Results")
    st.write("Training Accuracy:", nb.score(tfidf_df, y))

    st.success("✅ Analysis Complete")
