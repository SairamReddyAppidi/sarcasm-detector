# 😏 Sarcasm Detector

A context-aware sarcasm and irony detection web app built with RoBERTa and Streamlit.

## About
This app uses a fine-tuned RoBERTa-base transformer model to detect sarcasm
in social media text. The model was trained on the SARC 2.0 Reddit corpus
and achieves a macro F1 score of 0.7426 on the held-out test set.

The key innovation is context-awareness — the model reads both the parent post
and the reply together, enabling it to detect sarcasm that only makes sense
in conversational context.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
- Base model: roberta-base
- Dataset: SARC 2.0 (50K balanced samples from Reddit)
- Validation F1: 0.7426
- Training: 3 epochs, AdamW optimizer, lr=2e-5
- Model hosted at: https://huggingface.co/rammm18/sarcasm-detector

## Course
CSCI 5922 — Neural Networks and Deep Learning
