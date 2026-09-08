import collections
import math
import random
import re
from nltk import ngrams
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Myanmar Agriculture N-gram Predictor", layout="centered"
)
st.title("🌾 Agriculture N-gram Predictor (Train/Test Split)")

csv_file_path = "Myanmar-Agricutlure-1K/Agriculture.csv"


# 1. Dataset Split & Model Training
@st.cache_data

def tokenize_myanmar(text):
        if not isinstance(text, str):
            return []
        pattern = r"[\u1000-\u102A\u103F\u104E\u1050-\u1055]+[\u102B-\u103E\u109E\u109F]*|[a-zA-C0-9]+"
        tokens = re.findall(pattern, text)
        return [t.strip() for t in tokens if t.strip()]

def load_train_and_evaluate():
    trigram_model = collections.defaultdict(collections.Counter)
    bigram_model = collections.defaultdict(collections.Counter)
    unigram_model = collections.Counter()

   
    df = pd.read_csv(csv_file_path, header=None)
    all_lines = []

    for col in df.columns:
        for text in df[col].dropna().astype(str):
            tokens = tokenize_myanmar(text)
            if len(tokens) >= 3:
                all_lines.append(tokens)

    # Train/Test Split (80% Train, 20% Test)
    random.seed(42)
    random.shuffle(all_lines)

    split_idx = int(len(all_lines) * 0.8)
    train_data = all_lines[:split_idx]
    test_data = all_lines[split_idx:]

   
    for tokens in train_data:
        for w in tokens:
            unigram_model[w] += 1
        for w1, w2 in ngrams(tokens, 2):
            bigram_model[w1][w2] += 1
        for w1, w2, w3 in ngrams(tokens, 3):
            trigram_model[(w1, w2)][w3] += 1

   
    correct_top3 = 0
    total_test_cases = 0

    for tokens in test_data:
        for i in range(len(tokens) - 2):
            w1, w2, actual_target = tokens[i], tokens[i + 1], tokens[i + 2]

            
            predicted_words = []
            if (w1, w2) in trigram_model:
                predicted_words = [
                    w for w, c in trigram_model[(w1, w2)].most_common(3)
                ]
            elif w2 in bigram_model:
                predicted_words = [
                    w for w, c in bigram_model[w2].most_common(3)
                ]

            if actual_target in predicted_words:
                correct_top3 += 1
            total_test_cases += 1

    accuracy = (
        (correct_top3 / total_test_cases * 100) if total_test_cases > 0 else 0
    )

    metrics_info = {
        "train_lines": len(train_data),
        "test_lines": len(test_data),
        "total_test_cases": total_test_cases,
        "accuracy": accuracy,
    }

    return (
        trigram_model,
        bigram_model,
        unigram_model,
        tokenize_myanmar,
        metrics_info,
    )


trigram_model, bigram_model, unigram_model, tokenize_myanmar, metrics = (
    load_train_and_evaluate()
)



st.sidebar.header("📊 Dataset Evaluation Metrics")
st.sidebar.metric("Training Data (80%)", f"{metrics['train_lines']} Lines")
st.sidebar.metric("Testing Data (20%)", f"{metrics['test_lines']} Lines")
st.sidebar.metric("Top-3 Test Accuracy", f"{metrics['accuracy']:.2f}%")


# Probability & Perplexity Helper
def get_word_probability(w1, w2, target_word):
    if (w1, w2) in trigram_model:
        context_count = sum(trigram_model[(w1, w2)].values())
        word_count = trigram_model[(w1, w2)][target_word]
        if word_count > 0:
            return word_count / context_count
    if w2 in bigram_model:
        context_count = sum(bigram_model[w2].values())
        word_count = bigram_model[w2][target_word]
        if word_count > 0:
            return (word_count / context_count) * 0.4
    total_unigrams = sum(unigram_model.values())
    word_count = unigram_model[target_word]
    if word_count > 0:
        return (word_count / total_unigrams) * 0.16
    return 1e-6


def calculate_perplexity(tokens):
    if len(tokens) < 3:
        return None
    log_prob_sum = 0
    N = len(tokens) - 2
    for i in range(len(tokens) - 2):
        prob = get_word_probability(tokens[i], tokens[i + 1], tokens[i + 2])
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / N)


# UI Input Section
user_input = st.text_input(
    "Enter Context",
    placeholder="ဥပမာ- စပျစ် စိုက်ပျိုး ရေး",
)

if user_input:
    tokens = tokenize_myanmar(user_input)
    st.write(f"**Tokens:** `{tokens}`")

    results = []
    match_type = ""

    if len(tokens) >= 2:
        w1, w2 = tokens[-2], tokens[-1]
        context = (w1, w2)
        if context in trigram_model:
            total_cnt = sum(trigram_model[context].values())
            results = [
                (w, c, (c / total_cnt) * 100)
                for w, c in trigram_model[context].most_common(5)
            ]
            match_type = f"Trigram Match (`{w1}`, `{w2}`)"

    if not results and len(tokens) >= 1:
        w_last = tokens[-1]
        if w_last in bigram_model:
            total_cnt = sum(bigram_model[w_last].values())
            results = [
                (w, c, (c / total_cnt) * 100)
                for w, c in bigram_model[w_last].most_common(5)
            ]
            match_type = f"Bigram Backoff Match (`{w_last}`)"

    if not results:
        total_uni = sum(unigram_model.values())
        results = [
            (w, c, (c / total_uni) * 100)
            for w, c in unigram_model.most_common(5)
        ]
        match_type = "Unigram Fallback"

    st.subheader(f"Next Word Predictions ({match_type})")
    for rank, (word, count, prob) in enumerate(results, start=1):
        st.success(
            f"**{rank}. {word}** — Probability: **{prob:.2f}%** (Count: {count})"
        )

    st.markdown("---")

    st.subheader("📊 Language Model Evaluation (Perplexity)")
    ppl = calculate_perplexity(tokens)
    if ppl:
        st.metric(label="Sentence Perplexity (PPL)", value=f"{ppl:.2f}")
    else:
        st.info("It is at least 3 words to calculate perplexity")