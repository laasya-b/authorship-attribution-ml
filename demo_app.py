"""
Authorship Attribution — Live Demo

Run with:  streamlit run demo_app.py
Models auto-downloaded from Hugging Face on first run.
"""

import streamlit as st
import numpy as np
import pickle, os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

#  Auto-download models from Hugging Face if not present 
HF_REPO   = "Laasyab/authorship-attribution-models"
MODEL_DIR = "models"

MODEL_FILES = [
    "lr_model.pkl", "nb_model.pkl", "svm_model.pkl",
    "cal_svm.pkl", "stylo_lr.pkl", "meta_lr.pkl",
    "tfidf_word.pkl", "tfidf_char.pkl",
    "scaler.pkl", "label_encoder.pkl",
    "best_bert.pt"
]

def download_models():
    from huggingface_hub import hf_hub_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename in MODEL_FILES:
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            with st.spinner(f"Downloading {filename} from Hugging Face..."):
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filename,
                    local_dir=MODEL_DIR,
                    repo_type="model"
                )

# Check if any model file is missing and download if so
missing = any(not os.path.exists(os.path.join(MODEL_DIR, f)) for f in MODEL_FILES)
if missing:
    download_models()
    st.success("✅ Models downloaded! Loading...")

# Page Config
st.set_page_config(
    page_title="Authorship Attribution",
    page_icon="✍️",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Serif+4:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Serif 4', serif;
        background-color: #f9f6f1;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #1a1a2e;
    }
    .stTextArea textarea {
        font-family: 'Source Serif 4', serif;
        font-size: 15px;
        border: 1px solid #c8b99a;
        border-radius: 6px;
        background-color: #fffdf8;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 12px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }
    .result-card h2 { color: #f0c040 !important; font-size: 2rem; margin: 0 0 6px 0; }
    .result-card p  { color: #ccc; margin: 0; font-size: 14px; }
    .stButton > button {
        background: #1a1a2e;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 28px;
        font-family: 'Source Serif 4', serif;
        font-size: 15px;
        cursor: pointer;
    }
    .stButton > button:hover { background: #2d2d5e; }
    .metric-box {
        background: white;
        border: 1px solid #e0d8cc;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-box .val { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .metric-box .lbl { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# Constants 
FUNCTION_WORDS = set([
    'the','a','an','and','but','or','nor','for','yet','so','in','on','at',
    'to','of','with','by','from','up','about','into','through','during',
    'is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','shall',
    'must','can','i','he','she','it','they','we','you','my','his','her',
    'its','our','your','their','this','that','these','those','which','who'
])

# Load Models 
@st.cache_resource
def load_classical_models():
    try:
        lr       = pickle.load(open(f'{MODEL_DIR}/lr_model.pkl',      'rb'))
        tfidf_w  = pickle.load(open(f'{MODEL_DIR}/tfidf_word.pkl',    'rb'))
        tfidf_c  = pickle.load(open(f'{MODEL_DIR}/tfidf_char.pkl',    'rb'))
        scaler   = pickle.load(open(f'{MODEL_DIR}/scaler.pkl',        'rb'))
        le       = pickle.load(open(f'{MODEL_DIR}/label_encoder.pkl', 'rb'))
        stylo_lr = pickle.load(open(f'{MODEL_DIR}/stylo_lr.pkl',      'rb'))
        meta_lr  = pickle.load(open(f'{MODEL_DIR}/meta_lr.pkl',       'rb'))
        cal_svm  = pickle.load(open(f'{MODEL_DIR}/cal_svm.pkl',       'rb'))
        return lr, cal_svm, tfidf_w, tfidf_c, scaler, le, stylo_lr, meta_lr, cal_svm
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}")
        return None

@st.cache_resource
def load_bert_model(num_classes):
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available() else 'cpu')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=num_classes
        )
        model.load_state_dict(torch.load(f'{MODEL_DIR}/best_bert.pt', map_location=device))
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except FileNotFoundError:
        return None, None, None

#  Feature Extraction 
def extract_stylometric_features(text):
    text = str(text)
    words = text.split()
    sentences = sent_tokenize(text) if len(text) > 10 else [text]
    num_words = max(len(words), 1)
    num_chars = max(len(text), 1)
    num_sents = max(len(sentences), 1)
    unique_words = set(w.lower() for w in words if w.isalpha())
    ttr            = len(unique_words) / num_words
    avg_word_len   = np.mean([len(w) for w in words if w.isalpha()]) if words else 0
    avg_sent_len   = num_words / num_sents
    comma_rate     = text.count(',')   / num_chars
    semicolon_rate = text.count(';')   / num_chars
    exclaim_rate   = text.count('!')   / num_chars
    question_rate  = text.count('?')   / num_chars
    dash_rate      = (text.count('—') + text.count('–') + text.count('-')) / num_chars
    quote_rate     = (text.count('"')  + text.count("'")) / num_chars
    ellipsis_rate  = text.count('...') / num_chars
    fw_count       = sum(1 for w in words if w.lower() in FUNCTION_WORDS)
    func_word_rate = fw_count / num_words
    upper_rate     = sum(1 for w in words if w.isupper() and len(w) > 1) / num_words
    long_word_rate = sum(1 for w in words if len(w) > 6) / num_words
    short_sent_rate= sum(1 for s in sentences if len(s.split()) < 8) / num_sents
    return np.array([[ttr, avg_word_len, avg_sent_len,
                      comma_rate, semicolon_rate, exclaim_rate, question_rate,
                      dash_rate, quote_rate, ellipsis_rate,
                      func_word_rate, upper_rate, long_word_rate, short_sent_rate]])

def get_readable_features(text):
    feats = extract_stylometric_features(text)[0]
    names = ['Type-Token Ratio', 'Avg Word Length', 'Avg Sentence Length',
             'Comma Rate', 'Semicolon Rate', 'Exclamation Rate', 'Question Rate',
             'Dash Rate', 'Quote Rate', 'Ellipsis Rate',
             'Function Word Rate', 'Uppercase Rate', 'Long Word Rate', 'Short Sentence Rate']
    return dict(zip(names, feats))

def predict_all(text, models_tuple, bert_tuple):
    from scipy.sparse import hstack, csr_matrix
    lr, svm, tfidf_w, tfidf_c, scaler, le, stylo_lr, meta_lr, cal_svm = models_tuple
    tokenizer, bert_model, device = bert_tuple

    X_word   = tfidf_w.transform([text])
    X_char   = tfidf_c.transform([text])
    stylo    = extract_stylometric_features(text)
    stylo_sc = scaler.transform(stylo)
    X_full   = hstack([X_word, X_char, csr_matrix(stylo_sc)])

    results = {}

    lr_probs    = lr.predict_proba(X_full)[0]
    results['Logistic Regression'] = (le.classes_[np.argmax(lr_probs)], lr_probs)

    svm_probs   = cal_svm.predict_proba(X_full)[0]
    results['Linear SVM'] = (le.classes_[np.argmax(svm_probs)], svm_probs)

    stylo_probs = stylo_lr.predict_proba(stylo_sc)[0]
    results['Stylometric Only'] = (le.classes_[np.argmax(stylo_probs)], stylo_probs)

    if tokenizer is not None:
        enc = tokenizer(text, max_length=256, padding='max_length',
                        truncation=True, return_tensors='pt')
        with torch.no_grad():
            out = bert_model(input_ids=enc['input_ids'].to(device),
                             attention_mask=enc['attention_mask'].to(device))
        bert_probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        results['DistilBERT'] = (le.classes_[np.argmax(bert_probs)], bert_probs)

        meta_X    = np.hstack([stylo_probs, svm_probs, bert_probs]).reshape(1, -1)
        ens_probs = meta_lr.predict_proba(meta_X)[0]
        results['Stacked Ensemble'] = (le.classes_[np.argmax(ens_probs)], ens_probs)

    return results, le.classes_

def make_confidence_chart(probs, classes, title, color):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#f9f6f1')
    ax.set_facecolor('#f9f6f1')
    sorted_idx = np.argsort(probs)
    ax.barh([classes[i] for i in sorted_idx],
            [probs[i]   for i in sorted_idx],
            color=color, alpha=0.85, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1a1a2e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    for i, (idx, p) in enumerate(zip(sorted_idx, [probs[j] for j in sorted_idx])):
        ax.text(p + 0.01, i, f'{p:.2%}', va='center', fontsize=8, color='#333')
    plt.tight_layout()
    return fig

# App Layout 
st.markdown("<h1>✍️ Authorship Attribution</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#666; font-size:16px; margin-top:-10px;'>Stylometric Analysis & Machine Learning · UW–Madison</p>", unsafe_allow_html=True)
st.divider()

models_tuple = load_classical_models()
if models_tuple is None:
    st.stop()

_, _, _, _, _, le, _, _, _ = models_tuple
bert_tuple = load_bert_model(len(le.classes_))

# Sidebar 
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    Identifies which of **10 classic authors** wrote a given passage using:
    - **Stylometric features** — punctuation, sentence length, vocabulary richness
    - **TF-IDF + char n-grams** — word & character-level patterns
    - **DistilBERT** — fine-tuned transformer
    - **Stacked Ensemble** — meta-learner combining all three

    **Ensemble accuracy: 98.98% · Macro F1: 98.09%**
    """)
    st.markdown("### Authors")
    for a in sorted(le.classes_):
        st.markdown(f"- {a}")
    st.divider()
    samples = {
        "Jane Austen":      "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.",
        "Ernest Hemingway": "He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a fish. In the first forty days a boy had been with him. But after forty days without a fish the boy's parents had told him that the old man was now definitely and finally salao, which is the worst form of unlucky.",
        "Edgar Allan Poe":  "True! nervous, very, very dreadfully nervous I had been and am; but why will you say that I am mad? The disease had sharpened my senses, not destroyed, not dulled them. Above all was the sense of hearing acute. I heard all things in the heaven and in the earth.",
        "Mark Twain":       "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain't no matter. That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth."
    }
    selected = st.selectbox("Load sample text:", ["(choose one)"] + list(samples.keys()))

#  Main
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Paste a text passage")
    default_text = samples[selected] if selected != "(choose one)" else ""
    user_text = st.text_area(
        label="",
        value=default_text,
        height=220,
        placeholder="Paste any passage of text here (3–4 sentences minimum work best)..."
    )
    model_choice = st.radio(
        "Which model to highlight?",
        ['Stacked Ensemble', 'DistilBERT', 'Linear SVM', 'Logistic Regression', 'Stylometric Only'],
        horizontal=True
    )
    run = st.button("🔍 Predict Author")

with col2:
    if run and user_text.strip():
        with st.spinner("Analysing writing style..."):
            all_results, classes = predict_all(user_text, models_tuple, bert_tuple)

        highlighted = all_results.get(model_choice, list(all_results.values())[0])
        pred_author, pred_probs = highlighted
        confidence = pred_probs.max()

        st.markdown(f"""
        <div class="result-card">
            <p>Predicted Author ({model_choice})</p>
            <h2>{pred_author}</h2>
            <p>Confidence: <strong style="color:#f0c040">{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        all_preds = [v[0] for v in all_results.values()]
        agreement = sum(1 for p in all_preds if p == pred_author)
        st.markdown(f"**Model agreement:** {agreement}/{len(all_results)} models predict *{pred_author}*")

    elif run:
        st.warning("Please enter some text first.")

# Full Results 
if run and user_text.strip():
    st.divider()
    st.markdown("### Confidence Breakdown by Model")

    model_colors = {
        'Logistic Regression': '#4C72B0',
        'Linear SVM':          '#55A868',
        'Stylometric Only':    '#C44E52',
        'DistilBERT':          '#8172B2',
        'Stacked Ensemble':    '#CCB974',
    }

    cols = st.columns(len(all_results))
    for col, (model_name, (pred, probs)) in zip(cols, all_results.items()):
        with col:
            fig = make_confidence_chart(probs, classes, model_name,
                                        model_colors.get(model_name, '#888'))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown(f"<div style='text-align:center;font-size:13px;color:#555'>→ <b>{pred}</b></div>",
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("### Stylometric Features Detected")
    feats = get_readable_features(user_text)
    feat_cols = st.columns(4)
    for i, (name, val) in enumerate(feats.items()):
        with feat_cols[i % 4]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{val:.3f}</div>
                <div class="lbl">{name}</div>
            </div><br>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Text Statistics")
    words = user_text.split()
    sents = sent_tokenize(user_text)
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Words", len(words))
    sc2.metric("Sentences", len(sents))
    sc3.metric("Avg words/sent", f"{len(words)/max(len(sents),1):.1f}")
    sc4.metric("Unique words", len(set(w.lower() for w in words if w.isalpha())))