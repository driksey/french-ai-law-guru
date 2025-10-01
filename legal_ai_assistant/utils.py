import os
from pathlib import Path
from typing import cast

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
    """Retrieve documents from vectorstore."""
    docs_retrieved = retriever.invoke(query)
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
    """Estimate token count from character count with improved accuracy for multilingual content."""
    # More accurate estimation considering French/legal text complexity
    # French text typically has higher token density than English
    if not text:
        return 0

    # Base ratio: ~3.5 chars per token for French legal text
    # Adjust based on text characteristics
    char_count = len(text)
    base_ratio = 3.5

    # Adjust for French-specific patterns (accents, longer words)
    french_chars = sum(1 for c in text if c in 'àâäéèêëïîôöùûüÿçñ')
    if french_chars > 0:
        # French text typically needs more tokens
        base_ratio = 3.2

    # Adjust for legal terminology (longer words, complex sentences)
    avg_word_length = char_count / max(text.count(' ') + text.count('\n') + 1, 1)
    if avg_word_length > 8:  # Legal text tends to have longer words
        base_ratio *= 0.9  # More tokens needed

    return int(char_count / base_ratio)


def calculate_max_response_tokens(doc_content: str, user_question: str = "", workflow_overhead: int = 100) -> int:
    """Calculate maximum response tokens based on actual content length."""

    # Base context window - increased since we removed document truncation
    total_context: int = cast(int, LLM_CONFIG["context_window"])

    # Use improved token estimation for actual content
    doc_content_tokens = estimate_tokens_from_chars(doc_content)
    user_question_tokens = estimate_tokens_from_chars(user_question) if user_question else 50

    # More realistic estimates based on actual prompt structure
    system_prompt_tokens = 120  # More accurate for legal assistant prompt
    prompt_formatting_tokens = 150  # Includes structure, instructions, formatting

    # Dynamic overhead based on number of documents (more docs = more structure)
    doc_count_overhead = min(50, len(doc_content.split('\n\n')) * 5)  # Per document overhead

    # Total reserved tokens
    reserved_tokens = (system_prompt_tokens + user_question_tokens +
                       doc_content_tokens + prompt_formatting_tokens +
                       workflow_overhead + doc_count_overhead)

    # Calculate available space for response
    available_tokens = total_context - reserved_tokens

    # Apply smaller safety margin since we need more comprehensive responses
    max_response_tokens = int(available_tokens * 0.85)  # Reduced from 0.75 to 0.85

    # Ensure minimum response length for legal analysis
    min_response_tokens = 100  # Increased from 60 to 100
    max_response_tokens = max(max_response_tokens, min_response_tokens)

    # Cap at model's max_new_tokens setting
    model_max_tokens: int = cast(int, LLM_CONFIG["max_new_tokens"])
    max_response_tokens = min(max_response_tokens, model_max_tokens)

    # Debug info with more details
    print(f"[TOKEN_CALC] Context: {total_context}, Doc tokens: {doc_content_tokens}, "
          f"User tokens: {user_question_tokens}, Reserved: {reserved_tokens}, "
          f"Available: {available_tokens}, Max response: {max_response_tokens}")

    return max_response_tokens
