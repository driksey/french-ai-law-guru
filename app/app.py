import os
import streamlit as st
from dotenv import load_dotenv
from app.utils import load_faqs, build_embeddings_if_needed, top_k_similar
from app.hf_client import generate_from_model

load_dotenv()

st.set_page_config(page_title="Mini FAQ Chatbot", layout="centered")
st.title("Mini FAQ Chatbot")

with st.sidebar:
    st.markdown("## Settings")
    model = os.getenv("HF_MODEL", "google/flan-t5-small")
    st.write(f"Model: {model}")
    top_k = st.slider("Number of FAQ snippets to include", 1, 5, 3)

faqs = load_faqs()
embeddings = build_embeddings_if_needed(faqs)

question = st.text_input("Ask the FAQ bot a question about the service or product")

if st.button("Ask") and question:
    with st.spinner("Searching FAQs..."):
        hits = top_k_similar(question, faqs, embeddings, k=top_k)
    context = "\n\n".join([f"Q: {h[2]['question']}\nA: {h[2].get('answer','')}" for h in hits])
    prompt = f"""You are a helpful assistant answering user questions using only the provided FAQ context.
FAQ context:
{context}

User question:
{question}

Answer concisely and reference relevant FAQ entries."""
    with st.spinner("Generating answer..."):
        answer = generate_from_model(prompt, max_new_tokens=200)
    st.markdown("**Answer**")
    st.write(answer)
    st.markdown("**Sources used**")
    for i, sim, faq in hits:
        st.write(f"- (score {sim:.2f}) {faq['question']}")
