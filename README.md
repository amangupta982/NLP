# 🧠 NLP Projects — Python & NLTK

A collection of Natural Language Processing (NLP) projects and beginner codes built using **Python**, **NLTK**, **Scikit-learn**, and **Streamlit**.

---

## 📁 Project Structure

```
NLP/
│
├── 📂 NLP basic Codes/
│   ├── basicOfNLP.py
│   ├── nltk_chunking_parser.py
│   ├── chatbot_with_tokenization.py
│   ├── named_entity_recognition.py
│   ├── named_entity_recognition1.py
│   ├── pos_tagging.py
│   ├── remove_stopwords.py
│   ├── simple_chatbot.py
│   ├── text_preprocessing.py
│   └── README.md
│
└── 📂 NLP Projects/
    ├── 📂 cricket commentary generation/
    ├── 📂 Graph Builder NLP project/
    ├── 📂 Multi-Model-Project/
    │   └── app.py
    ├── 📂 Multi-TaskLogisticRegression/
    │   ├── app.py
    │   └── multi_task_logistic_regression.py
    └── 📂 writing-style-analyzer/
        ├── app.py
        ├── feature_engineering.py
        └── gmm_model.pkl
```

---

## 🚀 Projects

### 1. 🏥 Multi-Task Logistic Regression
> **Folder:** `NLP Projects/Multi-TaskLogisticRegression/`

A clinical NLP system that reads patient symptom notes and simultaneously predicts:
- **Task 1** → Disease (Diabetes, Heart Disease, Flu, Hypertension)
- **Task 2** → Treatment Recommendation

**Tech used:** NLTK · TF-IDF · Scikit-learn · MultiOutputClassifier · Streamlit

**Run:**
```bash
streamlit run app.py
```

**Features:**
- 🔬 Live patient diagnosis from free-text notes
- 📊 Confusion matrix + accuracy metrics
- 📋 Dataset explorer with filters
- 🔑 Top keywords per disease (feature importance)

---

### 2. 📝 Writing Style Analyzer
> **Folder:** `NLP Projects/writing-style-analyzer/`

Analyzes and clusters writing styles using unsupervised machine learning.

**Tech used:** NLTK · Feature Engineering · GMM (Gaussian Mixture Model) · Streamlit

**Files:**
- `app.py` — Streamlit UI
- `feature_engineering.py` — Extracts stylometric features from text
- `gmm_model.pkl` — Pre-trained GMM clustering model

**Run:**
```bash
streamlit run app.py
```

---

### 3. 🏏 Cricket Commentary Generation
> **Folder:** `NLP Projects/cricket commentary generation/`

Generates cricket-style commentary using NLP techniques.

---

### 4. 🕸️ Graph Builder NLP Project
> **Folder:** `NLP Projects/Graph Builder NLP project/`

Builds knowledge graphs from text using Named Entity Recognition and relationship extraction.

---

### 5. 🤖 Multi-Model Project
> **Folder:** `NLP Projects/Multi-Model-Project/`

Combines multiple NLP models into a single pipeline application.

**Run:**
```bash
streamlit run app.py
```

---

## 📚 NLP Basic Codes

| File | What it does |
|------|-------------|
| `basicOfNLP.py` | Introduction to NLTK — tokenization, stemming, lemmatization |
| `nltk_chunking_parser.py` | Phrase chunking and grammar-based parsing |
| `chatbot_with_tokenization.py` | Simple rule-based chatbot using tokenization |
| `named_entity_recognition.py` | Detect names, places, organizations in text |
| `named_entity_recognition1.py` | Extended NER with custom entity types |
| `pos_tagging.py` | Part-of-speech tagging (noun, verb, adjective...) |
| `remove_stopwords.py` | Filter out common words like "the", "is", "and" |
| `simple_chatbot.py` | Basic pattern-matching chatbot |
| `text_preprocessing.py` | Full pipeline: clean → tokenize → stem → lemmatize |

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/NLP.git
cd NLP

# Install all dependencies
pip install nltk scikit-learn pandas numpy matplotlib seaborn streamlit
```

**Download NLTK data (run once in Python):**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| NLTK | Text preprocessing, tokenization, NER, POS tagging |
| Scikit-learn | Machine learning models (Logistic Regression, GMM) |
| Pandas / NumPy | Data handling |
| Matplotlib / Seaborn | Charts and visualizations |
| Streamlit | Web UI for all projects |

---

## 👤 Author

**Your Name**
- GitHub: [@amangupta982](https://github.com/amangupta982)
- LinkedIn: [AmanGupta](https://www.linkedin.com/in/amangupta982/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).