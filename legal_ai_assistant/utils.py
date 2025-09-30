import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_CONFIG, VECTORSTORE_CONFIG, LLM_CONFIG

EMB_MODEL = EMBEDDING_CONFIG["model_name"]


def load_pdfs(path=str):
    pdfs = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(path, filename)
            loader = PyPDFLoader(full_path)
            pdfs.extend(loader.load())
    return pdfs


def preprocess_pdfs(pdfs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(pdfs)
    for doc in docs:
        # Extract the parent directory name from the file path
        source_path = doc.metadata["source"]

        # Handle both Windows and Unix paths
        if "/" in source_path:
            path_parts = source_path.split("/")
        else:
            path_parts = source_path.split("\\")

        # Get the parent directory (the folder containing the PDF)
        # If the path is like "docs/file.pdf", we want "docs"
        if len(path_parts) >= 2:
            category = path_parts[-2]  # Parent directory
        else:
            category = "unknown"  # Fallback if path structure is unexpected

        doc.metadata = {
            "source": source_path,
            "category": category
        }
    return docs


def store_docs_with_embeddings(docs):
    model = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": EMBEDDING_CONFIG["batch_size"],
            "normalize_embeddings": EMBEDDING_CONFIG["normalize_embeddings"]
        },
        show_progress=EMBEDDING_CONFIG["show_progress"]
    )
    vectorstore = Chroma.from_documents(
        docs,
        embedding=model,
        collection_name=VECTORSTORE_CONFIG["collection_name"],
        persist_directory=VECTORSTORE_CONFIG["persist_directory"]
    )
    return vectorstore


def retrieve_documents(query: str, retriever):
    """Retrieve documents with token optimization."""
    docs_retrieved = retriever.invoke(query)
    
    # Limit document content length to prevent token overflow
    max_doc_length = 300  # Reduced for token efficiency
    for doc in docs_retrieved:
        if len(doc.page_content) > max_doc_length:
            doc.page_content = doc.page_content[:max_doc_length] + "..."
    
    return docs_retrieved


def load_or_create_vectorstore(docs, cache_key=None):
    """
    Load vectorstore from persistent directory or create new one.

    Args:
        docs: List of documents to embed
        cache_key: Optional cache key, if None will generate from docs content

    Returns:
        Chroma vectorstore
    """
    from langchain_chroma import Chroma

    persist_dir = VECTORSTORE_CONFIG["persist_directory"]

    # Try to load existing persistent vectorstore
    try:
        if Path(persist_dir).exists():
            print("Loading existing vectorstore from persistent directory...")
            model = HuggingFaceEmbeddings(
                model_name=EMB_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "batch_size": EMBEDDING_CONFIG["batch_size"],
                    "normalize_embeddings": EMBEDDING_CONFIG["normalize_embeddings"]
                },
                show_progress=False  # No progress bar for loading
            )
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=model,
                collection_name=VECTORSTORE_CONFIG["collection_name"]
            )

            # Test if the vectorstore has documents
            test_results = vectorstore.similarity_search("test", k=1)
            if test_results:
                print("Existing vectorstore loaded successfully!")
                return vectorstore
            else:
                print("Empty vectorstore found, recreating...")
    except Exception as e:
        print(f"Error loading persistent vectorstore: {e}, creating new one...")

    # Create new vectorstore
    print("Creating new vectorstore (this may take a while)...")
    vectorstore = store_docs_with_embeddings(docs)

    return vectorstore


# ---------- TOKEN CALCULATION UTILITIES ----------
def estimate_tokens_from_chars(text: str) -> int:
    """Estimate token count from character count (rough approximation: 1 token ≈ 4 chars)."""
    return len(text) // 4


def calculate_max_response_tokens(context_length: int, num_docs: int, workflow_overhead: int = 100) -> int:
    """Calculate maximum response tokens based on available context."""
    
    # Base context window
    total_context = LLM_CONFIG["context_window"]
    
    # Estimate tokens for different components
    system_prompt_tokens = 50  # Estimated system prompt tokens
    user_question_tokens = 30  # Estimated user question tokens
    doc_content_tokens = context_length // 4  # Document content (chars to tokens)
    prompt_formatting_tokens = 80  # Prompt structure and formatting
    
    # Total reserved tokens
    reserved_tokens = (system_prompt_tokens + user_question_tokens + 
                      doc_content_tokens + prompt_formatting_tokens + workflow_overhead)
    
    # Calculate available space for response
    available_tokens = total_context - reserved_tokens
    
    # Apply safety margin (25% buffer for safety)
    max_response_tokens = int(available_tokens * 0.75)
    
    # Ensure minimum response length
    min_response_tokens = 60
    max_response_tokens = max(max_response_tokens, min_response_tokens)
    
    # Cap at model's max_new_tokens setting
    model_max_tokens = LLM_CONFIG["max_new_tokens"]
    max_response_tokens = min(max_response_tokens, model_max_tokens)
    
    # Debug info (can be removed in production)
    print(f"[TOKEN_CALC] Context: {total_context}, Reserved: {reserved_tokens}, Available: {available_tokens}, Max response: {max_response_tokens}")
    
    return max_response_tokens
