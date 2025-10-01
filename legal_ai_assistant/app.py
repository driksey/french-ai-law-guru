# app.py
"""
Main Streamlit application for the French AI Law Assistant.

This module contains:
- Application initialization and configuration
- Model loading and management
- Vectorstore initialization
- User interface components
- Chat functionality
"""

import sys
import time
from pathlib import Path
from typing import cast, Literal

import streamlit as st
from dotenv import load_dotenv

from legal_ai_assistant.agents import create_rag_agent
from legal_ai_assistant.chat_handler import process_question_with_agent
from legal_ai_assistant.config import (
    APP_CONFIG,
    LLM_CONFIG,
    EMBEDDING_CONFIG,
    QUESTION_ANALYSIS_CONFIG,
)
from legal_ai_assistant.local_models import (
    create_local_llm,
    get_model_info,
    get_embedding_info,
)
from legal_ai_assistant.utils import (
    load_or_create_vectorstore,
    load_pdfs,
    preprocess_pdfs,
    retrieve_documents,
    extract_document_name,
)

# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title=str(APP_CONFIG["title"]),
    layout=cast(Literal["centered", "wide"], APP_CONFIG["page_layout"]),
    initial_sidebar_state=cast(
        Literal["auto", "expanded", "collapsed"], APP_CONFIG["initial_sidebar_state"]
    ),
)

# =============================================================================
# USER INTERFACE SETUP
# =============================================================================

# Custom CSS to expand sidebar and improve spacing
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

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
        st.markdown("- Cross-lingual: FR-EN optimized")
        st.markdown(f"- Description: {embedding_info['description']}")
        st.markdown("- Backend: Hugging Face (Local)")

    st.markdown("---")  # Separator line

    # Expandable section for supported languages
    with st.expander("🌍 Supported Languages"):
        languages = embedding_info["supported_languages"]
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
        cast(int, APP_CONFIG["default_top_k"]),
    )


# =============================================================================
# MODEL AND VECTORSTORE INITIALIZATION
# =============================================================================

# Load and process documents
path = "legal_docs"
docs = load_pdfs(path)
docs_processed = preprocess_pdfs(docs)


@st.cache_resource(show_spinner=False)
def get_vectorstore(_docs_processed):
    """Cache the vectorstore creation."""
    return load_or_create_vectorstore(_docs_processed)


def get_retriever(vectorstore, top_k):
    """Create retriever with dynamic top_k parameter."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# Model loading functions - optimized caching
@st.cache_resource(show_spinner=False)
def get_main_model():
    """Cache the main model creation (gemma2:2b for analysis and final answers)."""
    return create_local_llm()


@st.cache_resource(show_spinner=False)
def get_tool_model():
    """Cache the tool model creation (uses gemma3:270m for tool calls)."""
    return create_local_llm(QUESTION_ANALYSIS_CONFIG)  # Keep gemma3:270m for tool calls


@st.cache_resource(show_spinner=False)
def get_rag_agent(
    _main_model, _retriever, top_k, _tool_model=None, _progress_callback=None
):
    """Cache the RAG agent creation."""
    return create_rag_agent(
        _main_model, _retriever, _main_model, _tool_model, _progress_callback
    )


# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

# Initialize models and vectorstore - optimized loading
if "models_loaded" not in st.session_state:
    with st.spinner("🚀 Loading documents and embeddings..."):
        vectorstore = get_vectorstore(docs_processed)

    with st.spinner("🤖 Loading AI models..."):
        main_model = get_main_model()  # gemma2:2b for analysis and final answers
        tool_model = get_tool_model()  # gemma3:270m for tool calls

    # Store in session state to avoid reloading
    st.session_state.models_loaded = True
    st.session_state.vectorstore = vectorstore
    st.session_state.main_model = main_model
    st.session_state.tool_model = tool_model
else:
    # Use cached models from session state
    vectorstore = st.session_state.vectorstore
    main_model = st.session_state.main_model
    tool_model = st.session_state.tool_model

# Create retriever with current top_k value (dynamic, not cached)
retriever = get_retriever(vectorstore, top_k)

# Model validation
if main_model is None:
    st.error("❌ Failed to load Gemma 2 2B model via Ollama. Please ensure:")
    st.markdown("1. **Ollama is installed and running**")
    st.markdown("2. **Model is installed**: Run `ollama pull gemma2:2b`")
    st.markdown("3. **Check system resources** (≈3GB RAM required)")
    st.stop()

if tool_model is None:
    st.warning("⚠️ Tool model not available. Using main model for tool calls.")
    tool_model = main_model


# Create agent with progress callback support
def create_agent_with_callback(progress_callback=None):
    """Create agent with optional progress callback."""
    return get_rag_agent(main_model, retriever, top_k, tool_model, progress_callback)


# =============================================================================
# CHAT INTERFACE
# =============================================================================

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create a form to prevent automatic reruns and enable Enter key support
with st.form("chat_form", clear_on_submit=False):
    st.markdown("### 💬 Ask a Legal Question")
    question = st.text_input(
        "Ask the assistant a question about the AI regulations in France",
        placeholder="e.g., What are the GDPR requirements for AI systems?",
        help="Press Enter or click Ask to submit your question",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        submitted = st.form_submit_button("Ask", use_container_width=True)
    with col2:
        st.markdown("*💡 Tip: Press **Enter** or click **Ask** to submit*")

# Process the question when submitted (either by Enter key or button click)
if submitted and question.strip():
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Define progress callback function
    def update_progress(status: str, progress_percent: int):
        status_text.text(status)
        progress_bar.progress(progress_percent)

    try:
        # Create agent with progress callback
        agent_with_callback = create_agent_with_callback(update_progress)

        # Use the agent with progress callback
        answer, was_rag_used, retrieved_docs = process_question_with_agent(
            agent_with_callback, question.strip(), progress_callback=update_progress
        )

        # Clear progress indicators after completion
        progress_bar.empty()
        status_text.empty()

        # Add to chat history
        st.session_state.chat_history.append(
            {
                "question": question.strip(),
                "answer": answer,
                "was_rag_used": was_rag_used,
                "retrieved_docs": retrieved_docs,
                "timestamp": time.time(),
            }
        )

        # Display the answer
        st.markdown("### 📋 Answer")
        st.write(answer)

        # Only show sources if tool_rag was actually used
        if was_rag_used and retrieved_docs:
            st.markdown("### 📚 Documents Used by AI")
            for i, doc_content in enumerate(retrieved_docs):
                doc_name = extract_document_name(doc_content, i)
                st.markdown(f"📄 **{doc_name}**")

    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        progress_bar.empty()
        status_text.empty()

# Display chat history if there are previous questions
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📜 Chat History")

    # Show only the last 5 conversations to avoid clutter
    recent_history = st.session_state.chat_history[-5:]

    for i, chat in enumerate(reversed(recent_history)):
        with st.expander(
            f"Q: {chat['question'][:50]}{'...' if len(chat['question']) > 50 else ''}"
        ):
            st.markdown("**Question:**")
            st.write(chat["question"])
            st.markdown("**Answer:**")
            st.write(chat["answer"])
            if chat["was_rag_used"] and chat.get("retrieved_docs"):
                st.markdown("**📚 Documents Used:**")
                for j, doc_content in enumerate(chat["retrieved_docs"]):
                    doc_name = extract_document_name(doc_content, j)
                    st.markdown(f"📄 {doc_name}")
            elif chat["was_rag_used"]:
                st.markdown("*📚 Legal documents were consulted*")
            else:
                st.markdown("*💬 General response*")

    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
