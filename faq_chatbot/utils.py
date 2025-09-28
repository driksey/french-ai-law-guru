import os
import pickle
import hashlib
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import random

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_pdfs(path=str):
    pdfs = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(path, filename)
            loader = PyPDFLoader(full_path)
            pdfs.extend(loader.load())
    return pdfs


def preprocess_pdfs(pdfs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pdfs)
    for doc in docs:
        category = doc.metadata["source"].split("/")[-2]
        doc.metadata = {
            "source": doc.metadata["source"],
            "category": category
        }
    return docs

def store_docs_with_embeddings(docs):
    model = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu"}  
    )
    vectorstore = Chroma.from_documents(docs, embedding=model, collection_name =f"docs_{random.random()}")
    return vectorstore


def retrieve_documents(query: str, retriever):
    docs_retrieved = retriever.invoke(query)
    return docs_retrieved


def get_documents_hash(docs):
    """Generate a hash based on document content and metadata."""
    content = ""
    for doc in docs:
        content += doc.page_content + str(doc.metadata)
    return hashlib.md5(content.encode()).hexdigest()


def load_or_create_vectorstore(docs, cache_key=None):
    """
    Load vectorstore from cache or create new one if cache is invalid.
    
    Args:
        docs: List of documents to embed
        cache_key: Optional cache key, if None will generate from docs content
        
    Returns:
        Chroma vectorstore
    """
    if cache_key is None:
        cache_key = get_documents_hash(docs)
    
    cache_file = CACHE_DIR / f"vectorstore_{cache_key}.pkl"
    
    # Check if cache exists and is valid
    if cache_file.exists():
        try:
            print("Loading vectorstore from cache...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify the cached data is still valid
            if cached_data.get('hash') == cache_key:
                print("Cache loaded successfully!")
                return cached_data['vectorstore']
            else:
                print("Cache invalid, recreating...")
        except Exception as e:
            print(f"Error loading cache: {e}, recreating...")
    
    # Create new vectorstore
    print("Creating new vectorstore (this may take a while)...")
    vectorstore = store_docs_with_embeddings(docs)
    
    # Save to cache
    try:
        cache_data = {
            'vectorstore': vectorstore,
            'hash': cache_key
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Vectorstore cached successfully!")
    except Exception as e:
        print(f"Could not save cache: {e}")
    
    return vectorstore


def clear_cache():
    """Clear all cached vectorstores."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        print("Cache cleared successfully!")
    else:
        print("No cache to clear.")


def get_cache_info():
    """Get information about cached files."""
    if not CACHE_DIR.exists():
        return "No cache directory found."
    
    cache_files = list(CACHE_DIR.glob("vectorstore_*.pkl"))
    if not cache_files:
        return "No cached vectorstores found."
    
    total_size = sum(f.stat().st_size for f in cache_files)
    size_mb = total_size / (1024 * 1024)
    
    return f"Found {len(cache_files)} cached vectorstore(s), total size: {size_mb:.2f} MB"

