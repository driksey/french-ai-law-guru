# app.py
import sys
from pathlib import Path
from typing import cast, Literal

import streamlit as st
from dotenv import load_dotenv

from legal_ai_assistant.agents import create_rag_agent
from legal_ai_assistant.chat_handler import process_question_with_agent
from legal_ai_assistant.config import APP_CONFIG
from legal_ai_assistant.local_models import create_local_llm, get_model_info, get_embedding_info
from legal_ai_assistant.config import LLM_CONFIG, EMBEDDING_CONFIG
from legal_ai_assistant.utils import (
    load_or_create_vectorstore,
    load_pdfs,
    preprocess_pdfs,
    retrieve_documents,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

st.set_page_config(
    page_title=str(APP_CONFIG["title"]), 
    layout=cast(Literal["centered", "wide"], APP_CONFIG["page_layout"]),
    initial_sidebar_state=cast(Literal["auto", "expanded", "collapsed"], APP_CONFIG["initial_sidebar_state"])
)

# Custom CSS to expand sidebar and improve spacing
st.markdown("""
<style>
    .css-1d391kg {
        width: 350px !important;
    }
    .css-1d391kg .css-1v0mbdj {
        width: 350px !important;
    }
    .sidebar .sidebar-content {
        width: 350px !important;
    }
    .main .block-container {
        padding-left: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("French AI Law Assistant")

with st.sidebar:
    st.markdown("## 🏠 AI Law Assistant Settings")

    # Model configuration
    st.markdown("### Model Configuration")
    st.info("🚀 Using Gemma 2 2B via Ollama (~3GB RAM)")
    
    st.markdown("---")  # Separator line

    # Show model information in two columns
    model_info = get_model_info()
    embedding_info = get_embedding_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 LLM Model**")
        st.markdown(f"**Model:** {LLM_CONFIG['model_name']}")
        st.markdown(f"- Size: {model_info['size']}")
        st.markdown(f"- RAM Required: {model_info['ram_required']}")
        st.markdown(f"- Description: {model_info['description']}")
        st.markdown("- Backend: Ollama (Local)")
    
    with col2:
        st.markdown("**🔤 Embedding Model**")
        st.markdown(f"**Model:** {EMBEDDING_CONFIG['model_name']}")
        st.markdown(f"- Size: {embedding_info['size']}")
        st.markdown(f"- Languages: {embedding_info['languages_count']} languages")
        st.markdown(f"- Cross-lingual: FR-EN optimized")
        st.markdown(f"- Description: {embedding_info['description']}")
        st.markdown("- Backend: Hugging Face (Local)")
    
    st.markdown("---")  # Separator line
    
    # Expandable section for supported languages
    with st.expander("🌍 Supported Languages"):
        languages = embedding_info['supported_languages']
        # Group languages for better display
        col1, col2 = st.columns(2)
        with col1:
            for lang in languages[:8]:  # First 8 languages
                st.markdown(f"- {lang.upper()}")
        with col2:
                for lang in languages[8:]:  # Remaining languages
                    st.markdown(f"- {lang.upper()}")

    st.markdown("---")  # Separator line
    
    # Document retrieval settings
    st.markdown("### 📄 Document Retrieval")
    top_k = st.slider(
        "Number of doc snippets to include",
        2,
        cast(int, APP_CONFIG["max_top_k"]),
        cast(int, APP_CONFIG["default_top_k"])
    )


path = "legal_docs"
docs = load_pdfs(path)
docs_processed = preprocess_pdfs(docs)


@st.cache_resource
def get_vectorstore_and_retriever(_docs_processed, top_k):
    """Cache the vectorstore and retriever creation."""
    return load_or_create_vectorstore(_docs_processed), load_or_create_vectorstore(_docs_processed).as_retriever(search_kwargs={"k": top_k})


@st.cache_resource
def get_chat_model():
    """Cache the chat model creation."""
    return create_local_llm()


@st.cache_resource
def get_rag_agent(_chat_model, _retriever, top_k):
    """Cache the RAG agent creation."""
    return create_rag_agent(_chat_model, _retriever)


with st.spinner("Loading documents and embeddings..."):
    vectorstore, retriever = get_vectorstore_and_retriever(docs_processed, top_k)

chat_model = get_chat_model()

if chat_model is None:
    st.error("Failed to load Gemma 2 2B model via Ollama. Please ensure:")
    st.markdown("1. **Ollama is installed and running**")
    st.markdown("2. **Model is installed**: Run `ollama pull gemma2:2b`")
    st.markdown("3. **Check system resources** (≈3GB RAM required)")
    st.stop()

agent = get_rag_agent(chat_model, retriever, top_k)

question = st.text_input("Ask the assistant a question about the AI regulations in France")

if st.button("Ask") and question:
    with st.spinner("Searching documents..."):
        answer, was_rag_used = process_question_with_agent(agent, question)
        if was_rag_used:
            retrieved_docs = retrieve_documents(question, retriever)

    st.markdown("**Answer**")
    st.write(answer)
    
    # Only show sources if tool_rag was actually used
    if was_rag_used:
        st.markdown("**Sources used**")
        for i, doc in enumerate(retrieved_docs):
            st.write(f"- Document {i + 1}: {doc.metadata.get('source', 'Unknown source')}")
