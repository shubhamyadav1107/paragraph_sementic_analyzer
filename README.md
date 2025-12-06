# 📄✨ PDF NLP Analyzer & ML Classifier

### 🔍 Extract • 🧹 Clean • 🧠 Analyze • 📊 Visualize • 🤖 Classify

**A Streamlit Web App that transforms any uploaded PDF into insights using NLP + Machine Learning.**

---

## 🚀 What This App Does

Upload a PDF → The app automatically performs:

### 🧩 **Core NLP Pipeline**

* 📥 **Import all pages**
* 📝 **Extract text from each page**
* 🧼 **Preprocess text** (lowercase, clean, remove noise)
* ✂️ **Sentence tokenization**
* 📊 **Convert to a DataFrame**
* 😀😐☹️ **Sentiment Analysis** (VADER)
* 🔤 **Word tokenization**
* 🔢 **Word frequency counts**
* ☁️ **WordCloud generation**
* 📘 **Document-Term Matrix**
* 📗 **TF-IDF Matrix (max_features=300)**

---

## 🤖 Machine Learning Models (Sentiment Classification)

Using **Sentiment** as target `y` and **TF-IDF** as features `X`, the app trains:

* 🔵 **Logistic Regression**
* 🌳 **Decision Tree Classifier**
* 🟡 **Naïve Bayes (MultinomialNB)**

Each model displays:
✔️ Accuracy
✔️ Classification Report
✔️ Confusion Matrix

---

## 🌟 Bonus Features

### 🟢 **Bonus A — Automatic WordCloud & Frequent Words**

Right after PDF upload, the app visualizes:

* Top frequent words
* A beautiful WordCloud

### 🔥 **Bonus B — Extract Paragraphs by Keywords**

Enter **multiple keywords**, and the app will:

* 🔍 Find all paragraphs containing those keywords
* 🎨 Generate a WordCloud from matched paragraphs
* 📑 Display matched text snippets

Perfect for **topic-based extraction** from long PDFs!

---

## 🧰 Tech Stack

| Area          | Tech               |
| ------------- | ------------------ |
| Interface     | 🌐 Streamlit       |
| NLP           | 🧠 NLTK, WordCloud |
| ML            | 🤖 scikit-learn    |
| PDF Parsing   | 📄 PyPDF2          |
| Data Handling | 🐼 Pandas          |

---

## 📂 Repository Structure

```
📁 your-repo/
│── app.py              # Streamlit application
│── requirements.txt    # All Python dependencies
│── README.md           # Documentation (this file)
```

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

This app is designed for **Streamlit Community Cloud**.
Simply link the GitHub repo → Deploy → Upload PDF → Enjoy!

---

## 📸 Screenshots (Optional)

*(Insert screenshots here once your app UI is ready)*

---

## 🤝 Contribution

Feel free to open issues or submit PRs if you want to:

* Add more ML models
* Improve UI
* Add topic modeling (LDA)
* Support DOCX / TXT upload




