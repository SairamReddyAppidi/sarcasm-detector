import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title = "Sarcasm Detector",
    page_icon  = "😏",
    layout     = "centered"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .sarcastic {
        background-color: #fdecea;
        color: #c0392b;
        border: 1.5px solid #e74c3c;
    }
    .not-sarcastic {
        background-color: #eafaf1;
        color: #1e8449;
        border: 1.5px solid #2ecc71;
    }
    .info-box {
        background-color: #f0f4ff;
        border-left: 4px solid #4a6cf7;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #444;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model from HuggingFace (cached after first load) ──────
@st.cache_resource
def load_model():
    MODEL_NAME = "rammm18/sarcasm-detector"
    tokenizer  = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model      = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)
    model.eval()
    return tokenizer, model


# ── Text cleaning (same as training in Colab) ──────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s*/s\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    text = re.sub(r'u/\w+', '[USER]', text)
    text = re.sub(r'r/\w+', '[SUBREDDIT]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Prediction function ────────────────────────────────────────
def predict(context, reply, tokenizer, model):
    context = clean_text(context)
    reply   = clean_text(reply)

    if context:
        text = f"{context} </s> {reply}"
    else:
        text = reply

    inputs = tokenizer(
        text,
        return_tensors = "pt",
        max_length     = 128,
        truncation     = True,
        padding        = True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]
        pred    = torch.argmax(probs).item()

    return pred, probs[1].item(), probs[0].item()


# ── App UI ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">😏 Sarcasm Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Context-aware sarcasm detection · RoBERTa · CSCI 5922</div>',
    unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>How to use:</b> Type the reply you want to analyse below.
    Optionally add the parent post as context — this helps the model
    detect sarcasm that only makes sense in conversation.
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────
with st.spinner("Loading model... (first time only, ~30 seconds)"):
    try:
        tokenizer, model = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Could not load model: {e}")
        model_loaded = False

if model_loaded:

    # ── Example buttons ────────────────────────────────────────
    st.markdown("**Try an example:**")
    col1, col2, col3 = st.columns(3)

    example_context = ""
    example_reply   = ""

    with col1:
        if st.button("😏 Sarcastic", use_container_width=True):
            example_context = "Just got stuck in traffic for 3 hours on a Friday evening."
            example_reply   = "Oh wonderful, sounds like a fantastic start to the weekend!"

    with col2:
        if st.button("😐 Not sarcastic", use_container_width=True):
            example_context = "Anyone have a good recipe for banana bread?"
            example_reply   = "I use this one from Sally's Baking Addiction, it is really simple."

    with col3:
        if st.button("🤔 Tricky one", use_container_width=True):
            example_context = "My flight got cancelled with no warning at all."
            example_reply   = "Oh brilliant. Just what I needed today."

    st.markdown("---")

    # ── Input fields ───────────────────────────────────────────
    context_input = st.text_area(
        "Context — parent post (optional)",
        value       = example_context,
        placeholder = "e.g. My laptop crashed right before the deadline...",
        height      = 80
    )

    reply_input = st.text_area(
        "Reply to analyse ✱",
        value       = example_reply,
        placeholder = "e.g. Oh perfect timing, love when that happens!",
        height      = 100
    )

    # ── Predict button ─────────────────────────────────────────
    if st.button("🔍 Detect Sarcasm", type="primary",
                 use_container_width=True):
        if not reply_input.strip():
            st.warning("Please enter a reply to analyse.")
        else:
            with st.spinner("Analysing..."):
                pred, sarc_prob, not_sarc_prob = predict(
                    context_input, reply_input, tokenizer, model)

            # ── Result ─────────────────────────────────────────
            if pred == 1:
                st.markdown(
                    '<div class="result-box sarcastic">😏 SARCASTIC</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="result-box not-sarcastic">😐 NOT SARCASTIC</div>',
                    unsafe_allow_html=True)

            # ── Confidence ─────────────────────────────────────
            st.markdown("---")
            st.markdown("**Model confidence:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Sarcastic", f"{sarc_prob * 100:.1f}%")
                st.progress(float(sarc_prob))
            with col_b:
                st.metric("Not Sarcastic", f"{not_sarc_prob * 100:.1f}%")
                st.progress(float(not_sarc_prob))

            # ── Context note ───────────────────────────────────
            if context_input.strip():
                st.info(
                    "Context was used — the model read both the "
                    "parent post and the reply together.")
            else:
                st.warning(
                    "No context provided — try adding the parent "
                    "post for better accuracy.")

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; font-size:0.8rem; color:#999;'>
        Built with RoBERTa + HuggingFace Transformers · CSCI 5922 Course Project<br>
        Trained on SARC 2.0 (Reddit, 50K samples) · Val F1 = 0.7426
    </div>
    """, unsafe_allow_html=True)
