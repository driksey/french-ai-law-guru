# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from faq_chatbot.utils import load_faqs, build_embeddings_if_needed, top_k_similar
from faq_chatbot.hf_client import generate_from_model  # version gemma
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

st.set_page_config(page_title="FAQ Chatbot", layout="centered")
st.title("FAQ Chatbot")

with st.sidebar:
    st.markdown("## Settings")
    model = os.getenv("HF_MODEL", "google/gemma-2-2b-it")
    st.write(f"Model: {model}")
    top_k = st.slider("Number of FAQ snippets to include", 1, 5, 3)

faqs = load_faqs()
embeddings = build_embeddings_if_needed(faqs)

question = st.text_input("Ask the FAQ bot a question about the service or product")

if st.button("Ask") and question:
    with st.spinner("Searching FAQs..."):
        hits = top_k_similar(question, faqs, embeddings, k=top_k)

    # Construire le contexte FAQ
    context = "\n\n".join(
        [f"Q: {h[2]['question']}\nA: {h[2].get('answer','')}" for h in hits]
    )

    # Préparer le prompt pour Gemma sous forme de chat
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant answering user questions using only the provided FAQ context."
        },
        {
            "role": "user",
            "content": f"FAQ context:\n{context}\n\nUser question:\n{question}\n\nAnswer concisely and reference relevant FAQ entries."
        }
    ]

    with st.spinner("Generating answer..."):
        answer = generate_from_model(messages=messages)  # hf_client.py doit accepter messages list

    st.markdown("**Answer**")
    st.write(answer)
    st.markdown("**Sources used**")
    for i, sim, faq in hits:
        st.write(f"- (score {sim:.2f}) {faq['question']}")
