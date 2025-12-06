import streamlit as st
import pandas as pd
import numpy as np
import io
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import PyPDF2

# Ensure required NLTK packages are available
# Ensure required NLTK packages are available
nltk.download("punkt")
nltk.download("punkt_tab")      # ← NEW FIX
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Initialize sentiment analyzer & stopwords
sia = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words("english"))

st.set_page_config(page_title="PDF NLP & ML App", layout="wide")

st.title("PDF → NLP Pipeline + ML Models (Streamlit)")
st.markdown(
    """
Upload a PDF and the app will:
- Extract text from all pages
- Preprocess text, sentence-tokenize
- Build DataFrame, do sentiment analysis (VADER)
- Word tokenization, frequency counts, wordcloud
- Document-Term Matrix & TF-IDF (max_features=300)
- Train: Logistic Regression, Decision Tree, Naïve Bayes (using sentiment as y)
Bonus:
- Show wordcloud and frequent words (easy)
- Extract paragraphs containing keywords & wordcloud from them (advanced)
"""
)

############################
# Helper functions
############################

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_num": i + 1, "text": text})
    return pages

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove newlines and redundant spaces
    text = re.sub(r"\s+", " ", text)
    # Remove URLs and emails
    text = re.sub(r"http\S+|www\S+|mailto:\S+", " ", text)
    # Remove punctuation (keep apostrophes removed to simplify)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Strip extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_tokenize_pages(pages):
    rows = []
    for p in pages:
        raw = p["text"] or ""
        if not raw.strip():
            continue
        sents = sent_tokenize(raw)
        for s in sents:
            pre = preprocess_text(s)
            if pre.strip():
                rows.append({"page": p["page_num"], "sentence": s.strip(), "preprocessed": pre})
    return pd.DataFrame(rows)

def sentiment_label_from_compound(compound):
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def add_sentiment(df):
    compounds = df["preprocessed"].apply(lambda t: sia.polarity_scores(t)["compound"])
    df = df.copy()
    df["compound"] = compounds
    df["sentiment"] = df["compound"].apply(sentiment_label_from_compound)
    return df

def word_tokenize_text(preprocessed_text):
    tokens = word_tokenize(preprocessed_text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]
    return tokens

def freq_counts_from_tokens(token_list, top_n=30):
    c = Counter(token_list)
    return c.most_common(top_n)

def plot_wordcloud_from_text(text, width=600, height=400):
    wc = WordCloud(width=width, height=height, background_color="white", stopwords=STOPWORDS)
    wc = wc.generate(text)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def plot_freq_bar(freqs, title="Top words"):
    words, counts = zip(*freqs) if freqs else ([], [])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words[::-1], counts[::-1])
    ax.set_title(title)
    ax.set_xlabel("Count")
    return fig

############################
# Streamlit UI
############################

uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded:
    bytes_data = uploaded.read()
    st.info("Extracting pages...")
    pages = extract_text_from_pdf(bytes_data)

    st.success(f"Extracted {len(pages)} pages.")
    if st.checkbox("Show raw text per page (first 3 pages)"):
        for p in pages[:3]:
            st.write(f"**Page {p['page_num']}**")
            st.write(p["text"][:1000] + ("..." if len(p["text"])>1000 else ""))

    # a) Import all pages & b) Extract text - done above
    # c,d,e Build dataframe of sentence-level rows
    with st.spinner("Sentence tokenizing and preprocessing..."):
        df_sent = sentence_tokenize_pages(pages)
    if df_sent.empty:
        st.warning("No sentences extracted from the PDF. Check the PDF or try a different file.")
    else:
        st.success(f"Extracted {len(df_sent)} sentences (rows).")

        # Show sample
        st.subheader("Sample sentences (first 10)")
        st.dataframe(df_sent.head(10))

        # f) Sentiment Analysis
        st.subheader("Sentiment analysis (VADER)")
        df_sent = add_sentiment(df_sent)
        st.dataframe(df_sent[["page", "sentence", "preprocessed", "compound", "sentiment"]].head(10))

        # Show sentiment distribution
        st.write("Sentiment distribution:")
        st.bar_chart(df_sent["sentiment"].value_counts())

        # g) Word Tokenize preprocessed text
        st.subheader("Word tokenization & frequency counts")
        # create token column (list)
        df_sent["tokens"] = df_sent["preprocessed"].apply(lambda t: word_tokenize_text(t))
        all_tokens = [tok for sub in df_sent["tokens"] for tok in sub]
        freqs = freq_counts_from_tokens(all_tokens, top_n=30)
        st.write("Top words:")
        st.table(pd.DataFrame(freqs, columns=["word", "count"]))

        # h) Frequency Count plot
        fig_freq = plot_freq_bar(freqs, title="Top 30 words")
        st.pyplot(fig_freq)

        # i) Wordcloud (full document)
        st.subheader("Wordcloud (entire document)")
        full_preprocessed = " ".join(df_sent["preprocessed"].tolist())
        fig_wc = plot_wordcloud_from_text(full_preprocessed)
        st.pyplot(fig_wc)

        # j) Document Term Matrix (CountVectorizer)
        st.subheader("Document-Term Matrix (CountVectorizer)")
        cv = CountVectorizer(max_features=1000)
        X_counts = cv.fit_transform(df_sent["preprocessed"])
        st.write("DTM shape:", X_counts.shape)
        st.write("Some features (first 30):", cv.get_feature_names_out()[:30].tolist())

        # k) TF-IDF with max_features=300
        st.subheader("TF-IDF Vectorizer (max_features=300)")
        tfidf = TfidfVectorizer(max_features=300)
        X_tfidf = tfidf.fit_transform(df_sent["preprocessed"])
        st.write("TF-IDF matrix shape:", X_tfidf.shape)
        st.write("TF-IDF features (sample 30):", tfidf.get_feature_names_out()[:30].tolist())

        # l) Train models using sentiment as y and TF-IDF as X
        st.subheader("Train ML models (sentiment as y)")

        # Convert y to numeric labels
        y = df_sent["sentiment"]
        label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        y_num = y.map(label_mapping)

        test_size = st.slider("Test size (fraction)", 0.05, 0.5, 0.2)
        random_state = 42

        if st.button("Train models now"):
            with st.spinner("Training models..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_tfidf, y_num, test_size=test_size, random_state=random_state, stratify=y_num
                )

                results = {}
                # Logistic Regression
                lr = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)
                acc_lr = accuracy_score(y_test, y_pred_lr)
                results["Logistic Regression"] = (acc_lr, classification_report(y_test, y_pred_lr, output_dict=True))

                # Decision Tree
                dt = DecisionTreeClassifier(random_state=random_state)
                dt.fit(X_train, y_train)
                y_pred_dt = dt.predict(X_test)
                acc_dt = accuracy_score(y_test, y_pred_dt)
                results["Decision Tree"] = (acc_dt, classification_report(y_test, y_pred_dt, output_dict=True))

                # Naive Bayes (MultinomialNB)
                nb = MultinomialNB()
                nb.fit(X_train, y_train)
                y_pred_nb = nb.predict(X_test)
                acc_nb = accuracy_score(y_test, y_pred_nb)
                results["Naive Bayes"] = (acc_nb, classification_report(y_test, y_pred_nb, output_dict=True))

            st.success("Training complete.")
            # Display results
            for name, (acc, creport) in results.items():
                st.markdown(f"### {name}")
                st.write(f"Accuracy: **{acc:.4f}**")
                cr_df = pd.DataFrame(creport).transpose()
                st.dataframe(cr_df)

            # Confusion matrices
            st.subheader("Confusion Matrices")
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, (name, (acc, _)), y_pred in zip(axes, results.items(), [y_pred_lr, y_pred_dt, y_pred_nb]):
                cm = confusion_matrix(y_test, {"Logistic Regression": y_pred_lr, "Decision Tree": y_pred_dt, "Naive Bayes": y_pred_nb}[name])
                ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                ax.set_title(name)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_xticks([0,1,2])
                ax.set_yticks([0,1,2])
                for (i, j), val in np.ndenumerate(cm):
                    ax.text(j, i, val, ha="center", va="center", color="white" if val>cm.max()/2 else "black")
            st.pyplot(fig)

        # Bonus a) Wordcloud and frequent words when PDF uploaded (low difficulty)
        st.subheader("Bonus A — Wordcloud & Frequent Words (auto)")
        st.write("Displayed above: Wordcloud (entire doc) and frequency bar.")

        # Bonus b) Advanced: extract paragraphs containing multiple keywords & create wordcloud
        st.subheader("Bonus B — Extract paragraphs by keywords & generate wordcloud")
        keywords_input = st.text_input("Enter keywords (comma-separated)", placeholder="e.g. covid, vaccine, lockdown")
        if st.button("Extract paragraphs & make wordcloud"):
            if not keywords_input.strip():
                st.warning("Enter one or more keywords.")
            else:
                keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]
                # Build paragraph list from original pages by splitting on double newline or full stops to get paragraphs
                all_paragraphs = []
                for p in pages:
                    raw = p["text"] or ""
                    # split into paragraphs heuristically
                    paras = [pp.strip() for pp in re.split(r"\n{2,}|\r{2,}", raw) if pp.strip()]
                    if not paras:
                        # fallback: split by sentences but group 3 sentences as paragraph
                        sents = sent_tokenize(raw)
                        paras = [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]
                    for para in paras:
                        all_paragraphs.append({"page": p["page_num"], "paragraph": para, "pre": preprocess_text(para)})

                # filter paragraphs containing any keyword
                matched = []
                for para in all_paragraphs:
                    text_low = para["pre"]
                    if any(re.search(r"\b" + re.escape(k) + r"\b", text_low) for k in keywords):
                        matched.append(para)

                if not matched:
                    st.warning("No paragraphs matched the keywords.")
                else:
                    st.success(f"Found {len(matched)} paragraphs containing the keywords.")
                    # Show matched paragraphs
                    for i, m in enumerate(matched[:10]):
                        st.markdown(f"**Page {m['page']} — Paragraph {i+1}**")
                        st.write(m["paragraph"][:600] + ("..." if len(m["paragraph"])>600 else ""))

                    # Create wordcloud from matched paragraphs
                    combined = " ".join([m["pre"] for m in matched])
                    fig_kw = plot_wordcloud_from_text(combined)
                    st.pyplot(fig_kw)

                    # Frequency of matched paras
                    tokens_matched = [tok for tok in word_tokenize_text(combined)]
                    freqs_matched = freq_counts_from_tokens(tokens_matched, top_n=30)
                    st.table(pd.DataFrame(freqs_matched, columns=["word", "count"]))

        # Provide download of the processed dataframe (CSV)
        csv = df_sent.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed sentences CSV", data=csv, file_name="processed_sentences.csv", mime="text/csv")

else:
    st.info("Upload a PDF to begin. The app will process the file and show results here.")

