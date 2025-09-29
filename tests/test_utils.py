import tempfile

from legal_ai_assistant.utils import load_pdfs, preprocess_pdfs


def test_load_pdfs_empty_directory():
    """Test loading PDFs from an empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pdfs = load_pdfs(temp_dir)
        assert len(pdfs) == 0


def test_preprocess_pdfs_empty_list():
    """Test preprocessing an empty list of PDFs."""
    docs = preprocess_pdfs([])
    assert len(docs) == 0


def test_preprocess_pdfs_with_mock_docs():
    """Test preprocessing with mock document data."""
    # Create mock documents
    class MockDoc:
        def __init__(self, content, source):
            self.page_content = content
            self.metadata = {"source": source}

    mock_docs = [
        MockDoc("Test content 1", "legal_docs/test1.pdf"),
        MockDoc("Test content 2", "legal_docs/test2.pdf")
    ]

    # Test preprocessing
    processed_docs = preprocess_pdfs(mock_docs)

    # Verify that documents have been processed
    assert len(processed_docs) > 0

    # Check that metadata has been updated
    for doc in processed_docs:
        assert "category" in doc.metadata
        assert doc.metadata["category"] == "legal_docs"
