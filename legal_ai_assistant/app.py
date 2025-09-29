# app.py
import streamlit as st
from dotenv import load_dotenv
from legal_ai_assistant.utils import load_pdfs, preprocess_pdfs, load_or_create_vectorstore
from legal_ai_assistant.local_models import create_local_llm, get_model_info
from legal_ai_assistant.agents import create_rag_agent
from legal_ai_assistant.chat_handler import process_question_with_agent
from legal_ai_assistant.utils import retrieve_documents
from legal_ai_assistant.config import APP_CONFIG
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

st.set_page_config(page_title=APP_CONFIG["title"], layout=APP_CONFIG["page_layout"])
st.title("Legal AI Assistant")

with st.sidebar:
    st.markdown("## 🏠 Local Model Settings")
    
    # Model configuration
    st.markdown("### Model Configuration")
    st.info("🚀 Using Llama 3.1 8B via Ollama (~6-8GB RAM)")
    
    # Show model information
    model_info = get_model_info()
    from legal_ai_assistant.config import LLM_CONFIG
    st.markdown(f"**Model:** {LLM_CONFIG['model_name']}")
    st.markdown(f"- Size: {model_info['size']}")
    st.markdown(f"- RAM Required: {model_info['ram_required']}")
    st.markdown(f"- Description: {model_info['description']}")
    st.markdown(f"- Backend: Ollama (Local)")
    
    top_k = st.slider(
        "Number of doc snippets to include", 
        1, 
        APP_CONFIG["max_top_k"], 
        APP_CONFIG["default_top_k"]
    )
    

path = "legal_docs"
docs = load_pdfs(path)
docs_processed = preprocess_pdfs(docs)

@st.cache_resource
def get_vectorstore_and_retriever(docs_processed, top_k):
    """Cache the vectorstore and retriever creation."""
    return load_or_create_vectorstore(docs_processed), load_or_create_vectorstore(docs_processed).as_retriever(search_kwargs={"k": top_k})

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
    st.error("Failed to load Llama 3.1 8B model via Ollama. Please ensure:")
    st.markdown("1. **Ollama is installed and running**")
    st.markdown("2. **Model is installed**: Run `ollama pull llama3.1:8b`")
    st.markdown("3. **Check system resources** (6-8GB RAM required)")
    st.stop()

agent = get_rag_agent(chat_model, retriever, top_k)

question = st.text_input("Ask the chatbot a question about the AI regulations in France")

if st.button("Ask") and question:
    with st.spinner("Generating answer..."):
        answer = process_question_with_agent(agent, question)
        retrieved_docs = retrieve_documents(question, retriever)

    st.markdown("**Answer**")
    st.write(answer)
    st.markdown("**Sources used**")
    for i, doc in enumerate(retrieved_docs):
        st.write(f"- Document {i+1}: {doc.metadata.get('source', 'Unknown source')}")