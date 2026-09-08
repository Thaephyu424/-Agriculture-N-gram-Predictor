# 🌾 Myanmar Agriculture N-gram Word Prediction System

An N-gram Language Model built with Python and Streamlit to predict the next word in Myanmar Unicode text, specifically tuned for the **Myanmar Agricultural domain**. This system incorporates backoff strategies, word probabilities, sentence perplexity evaluation, and train/test dataset splits.

---

## 📌 Features

* **Multi-order N-gram Support:** Implements Unigram, Bigram, and Trigram language modeling.
* **Stupid Backoff Strategy:** Seamlessly falls back from Trigram $\rightarrow$ Bigram $\rightarrow$ Unigram when context matches are not found.
* **Probability Estimation:** Calculates word prediction likelihood percentages.
* **Perplexity Evaluation:** Computes sentence-level Perplexity (PPL) to measure language model fluency and uncertainty.
* **Custom Myanmar Tokenizer:** Robust regex-based Unicode segmentation for Myanmar scripts.
* **Model Validation:** Includes an 80/20 Train/Test split to evaluate top-3 accuracy on unseen data.
* **Streamlit Web Interface:** Interactive and user-friendly Web UI for real-time predictions and evaluation metrics.

---

## 📁 Repository Structure

```text
├── Myanmar-Agricutlure-1K/
│   └── Agriculture.csv       # Myanmar Agriculture Domain Dataset
├── app.py                    # Streamlit Web Application & N-gram Model
├── README.md                 # Project Documentation

Installation & Setup

1. Clone the Repository
Bash
git clone [(https://github.com/Thaephyu424/-Agriculture-N-gram-Predictor.git]
cd -Agriculture-N-gram-Predictor

2. Install Dependencies
Ensure you have Python 3.8+ installed, then install required libraries:

Bash
pip install pandas nltk streamlit

Running the Application
To launch the Streamlit Web Application:

Bash

streamlit run app.py

After running the command, open your browser at http://localhost:8501.

Model Evaluation Metrics
Train/Test Ratio: 80% Training Data | 20% Testing Data

Top-3 Accuracy: Measures whether the actual target word appears within the model's top 3 suggestions.

Sentence Perplexity (PPL): Lower perplexity values indicate higher language model confidence and better contextual fluency.

🛠️ Built With
Language: Python 3

Frontend: Streamlit

NLP Toolkit: NLTK, Regular Expressions (re)

Data Processing: Pandas
