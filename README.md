# Authorship Attribution Using NLP & Machine Learning

> Can a machine identify an author's "fingerprint" in their writing, not from *what* they say, but *how* they say it?

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2D9E6B?style=flat)

---

## Motivation

I've always found stylometry fascinating. The idea that authors leave unconscious statistical fingerprints in their writing through punctuation habits, vocabulary richness, and sentence rhythm, entirely independent of topic or content.

This project started as a straightforward multi-class classification problem: predict which of 10 authors wrote a given passage. But after reading **Fabien et al. (2020) — BertAA**, which demonstrated that combining BERT with explicit stylometric features via a stacked ensemble improved macro F1 by ~2.7% over BERT alone, I wanted to explore whether that held on a literary dataset; and more importantly, *understand why*. Which features capture what? Where does BERT fail that classical methods don't?

Rather than stopping at a single model, I built a full comparison pipeline: hand-crafted stylometric features → classical ML baselines → fine-tuned DistilBERT → a stacked ensemble that combines all three. The goal was less about hitting a number and more about understanding what each approach captures about writing style that the others miss.

**A note on the dataset:** The corpus is small by design: 10 authors, 2 books each, from Project Gutenberg. Public domain texts are clean and well-suited for stylometric analysis, but 20 books is far from production scale. The high accuracy numbers reflect how stylistically distinct these particular authors are: results should be read as a method comparison and proof-of-concept rather than a generalizable authorship detector.

---

## What This Project Does

Given any text passage, the system predicts which of 10 classic authors wrote it, and shows you exactly how confident each model is and why. Five models run simultaneously, each capturing a different dimension of writing style. A **live Streamlit demo** lets you paste any text, toggle between models, and inspect stylometric features in real time.

**Authors in the dataset:**

| Author | Works |
|--------|-------|
| Jane Austen | Sense and Sensibility, Pride and Prejudice |
| Charles Dickens | Oliver Twist, A Tale of Two Cities |
| Mark Twain | Adventures of Tom Sawyer, Huckleberry Finn |
| Ernest Hemingway | A Farewell to Arms, The Sun Also Rises |
| Arthur Conan Doyle | Hound of the Baskervilles, A Study in Scarlet |
| F. Scott Fitzgerald | The Great Gatsby, The Beautiful and Damned |
| Edgar Allan Poe | The Tell-Tale Heart, Fall of the House of Usher |
| Agatha Christie | And Then There Were None, Roger Ackroyd |
| Rudyard Kipling | The Jungle Book, Kim |
| William Shakespeare | Romeo and Juliet, Hamlet |

All texts from [Project Gutenberg](https://www.gutenberg.org/) (public domain).

---

## Tech Stack

| Area | Tools |
|------|-------|
| **Language** | Python 3.11 |
| **Deep Learning** | PyTorch 2.2, HuggingFace Transformers (`distilbert-base-uncased`) |
| **Classical ML** | scikit-learn — LogisticRegression, LinearSVC, ComplementNB, CalibratedClassifierCV |
| **NLP / Features** | NLTK, TF-IDF (word + char n-gram), custom stylometric feature extractor |
| **Data** | pandas, NumPy, SciPy sparse matrices |
| **Visualization** | matplotlib, seaborn, WordCloud |
| **Demo App** | Streamlit |
| **Environment** | conda, Jupyter / VS Code |

---

## Methodology

### Preprocessing
Raw `.txt` files are cleaned, sentence-tokenized with NLTK, then grouped into **5-sentence passages**. Single sentences carry too little stylometric signal and passages give the models enough context to detect style. (~83k sentences → ~16k passages, 80/20 stratified split.)

### Features
- **TF-IDF** — word unigrams/bigrams + character 3–5 grams; captures vocabulary and sub-word punctuation patterns
- **14 stylometric features** — type-token ratio, avg word/sentence length, comma/semicolon/dash/exclamation rates, function word rate, long word rate, and more; captures *how* an author writes independent of *what* they write about

### Models
- **Complement Naive Bayes, Logistic Regression, Linear SVM** — classical baselines on combined TF-IDF + stylometric features
- **DistilBERT** — fine-tuned `distilbert-base-uncased` with a dense + softmax head; 256 tokens, 4 epochs, AdamW lr=2e-5
- **Stacked Ensemble** — probability vectors from Stylometric LR + Calibrated SVM + DistilBERT concatenated and fed into a meta Logistic Regression; train-set probs generated via 5-fold CV to prevent leakage

The ensemble is the key contribution: stylometric features model surface writing habits explicitly; DistilBERT learns deep contextual patterns implicitly. The meta-learner figures out which to trust per author; that's where the gains come from.

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 97.69% | 91.30% |
| Logistic Regression | 98.35% | 95.32% |
| Linear SVM | 98.95% | 96.57% |
| DistilBERT | 98.08% | 96.91% |
| **Stacked Ensemble** | **98.98%** | **98.09%** |


**Key observations:**
- Hemingway, Austen, Shakespeare, Kipling hit perfect or near-perfect F1 — highly distinctive styles
- Poe's 89% F1 is a data size issue (only 27 test samples), not model confusion — more data would close this gap
- Linear SVM and DistilBERT are nearly tied (96.57% vs 96.91%), suggesting classical methods capture most of the signal for stylistically distinct 19th-century authors
- The ensemble's +1.5% F1 gain over DistilBERT confirms the two feature types are genuinely complementary

---

## Setup & Running

```bash
git clone https://github.com/laasya-b/authorship-attribution-ml.git
cd authorship-attribution-ml

conda create -n authorship python=3.11 -y
conda activate authorship
pip install -r requirements.txt
```

Add books as `.txt` files under `input/prose/<AuthorName>/`, then:

```bash
# Step 1 — Preprocessing
jupyter notebook Preprocessing.ipynb

# Step 2 — Train all models
jupyter notebook authorship_attribution.ipynb

# Step 3 — Launch demo
streamlit run demo_app.py
```

> DistilBERT training: ~30–60 min on CPU, ~10 min on Apple Silicon (MPS).

---

## Limitations & Future Work

- **Dataset scale** — 20 books across 10 authors; high accuracy partly reflects how distinctive these authors are from each other
- **Cross-domain generalization** — all texts are 19th–early 20th century prose; performance would degrade on modern or short-form text
- **LLM-generated text** — language models can imitate style convincingly, posing a fundamental challenge to stylometric attribution
- **Future directions** — larger corpora, RoBERTa/DeBERTa backbone, cross-domain evaluation, open-set attribution

---

## Key References

- Fabien et al. (2020). *BertAA: BERT fine-tuning for Authorship Attribution.* ICON 2020.
- Koppel, Schler & Argamon (2009). *Computational methods in authorship attribution.* JASIST, 60(1).
- Neal et al. (2017). *Surveying stylometry techniques and applications.* ACM Computing Surveys, 50(6).
- Burrows, J. (2002). *Delta: A measure of stylistic difference.* Literary and Linguistic Computing, 17(3).

---

*Developed as a course project for Text Mining (graduate level) at the University of Wisconsin–Madison. 
Codebase is modular — each stage of the pipeline can be swapped out independently.*
