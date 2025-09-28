# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from faq_chatbot.utils import load_pdfs, preprocess_pdfs, load_or_create_vectorstore, get_cache_info, clear_cache
from faq_chatbot.hf_client import create_huggingface_client
from faq_chatbot.agents import create_rag_agent
from faq_chatbot.chat_handler import process_question_with_agent, get_retrieved_documents
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

st.set_page_config(page_title="AI Regulation Chatbot", layout="centered")
st.title("FAQ Chatbot")

with st.sidebar:
    st.markdown("## Settings")
    model = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    st.write(f"Model: {model}")
    top_k = st.slider("Number of doc snippets to include", 1, 5, 3)
    
    st.markdown("## LangSmith Tracing")
    st.info("LangSmith tracing is enabled. Check your LangSmith dashboard to monitor performance and debug issues.")
    
    # Show LangSmith project info
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "faq-chatbot")
    st.write(f"Project: {langsmith_project}")
    
    # Add link to LangSmith dashboard
    st.markdown(f"[Open LangSmith Dashboard](https://smith.langchain.com/projects)")
    
    st.markdown("## Cache Management")
    cache_info = get_cache_info()
    st.info(f"{cache_info}")
    
    if st.button("Clear Cache"):
        clear_cache()
        st.success("Cache cleared! Restart the app to recreate embeddings.")
        st.rerun()

# Initialize document processing with caching
path = "docs"
docs = load_pdfs(path)
docs_processed = preprocess_pdfs(docs)

# Use cached vectorstore or create new one
with st.spinner("Loading documents and embeddings..."):
    vectorstore = load_or_create_vectorstore(docs_processed)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

# Create RAG agent
chat_model = create_huggingface_client()
agent = create_rag_agent(chat_model, retriever)

question = st.text_input("Ask the chatbot a question about the AI regulations in France")

if st.button("Ask") and question:
    with st.spinner("Generating answer..."):
        # Utiliser l'agent pour générer une réponse
        answer = process_question_with_agent(agent, question)
        
        # Récupérer les documents utilisés
        retrieved_docs = get_retrieved_documents(retriever, question)

    st.markdown("**Answer**")
    st.write(answer)
    st.markdown("**Sources used**")
    for i, doc in enumerate(retrieved_docs):
        st.write(f"- Document {i+1}: {doc.metadata.get('source', 'Unknown source')}")
