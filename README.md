# French AI Law Assistant 🤖

![CI - main branch](https://github.com/driksey/french-ai-law-guru/actions/workflows/ci.yml/badge.svg?branch=main&label=main) ![CI - develop branch](https://github.com/driksey/french-ai-law-guru/actions/workflows/ci.yml/badge.svg?branch=develop&label=develop)

A sophisticated RAG (Retrieval-Augmented Generation) Q&A Assistant built with **Streamlit**, **LangChain**, and **LangGraph** for answering questions about AI regulations in France.  
It processes PDF documents, creates embeddings, and uses **Llama 3.2 1B** via Ollama to generate contextual answers with ultra-fast CPU-optimized inference and tool calling support.

---

## 🚀 Features
- **PDF Document Processing**: Automatically loads and processes PDF documents from the `legal_docs/` folder
- **Advanced RAG Architecture**: Uses LangChain and LangGraph for sophisticated document retrieval and generation
- **Vectorstore Caching**: Intelligent caching system to speed up document loading and embedding creation
- **LangSmith Integration**: Built-in tracing and monitoring for performance optimization and debugging
- **Ollama Integration**: Uses Ollama with Llama 3.2 1B for ultra-fast CPU inference and tool calling
- **Local LLM**: Runs entirely locally with Llama 3.2 1B (~1.3GB) - no external API dependencies
- **Agent-Based Processing**: Uses LangGraph agents for intelligent question answering
- **Stateless Q&A System**: Each question is processed independently for optimal performance
- **User-friendly Interface**: Clean Streamlit interface with cache management and settings
- **CI/CD Pipeline**: Includes linting, testing, and automated workflows  

---

## 🤖 Model Specifications

**Llama 3.2 1B Quantifié via Ollama**
- **Model Size**: ~1.1GB (quantized, optimized by Ollama)
- **RAM Required**: ≈2GB
- **Features**: Ultra-fast inference, tool calling support, RAG optimization, quantized Q4_K_M for speed
- **Performance**: Excellent for legal document analysis and Q&A on CPU with optimized memory usage
- **Local Processing**: No external API calls required

**Embeddings Multilingues**
- **Model**: `distiluse-base-multilingual-cased`
- **Size**: ~135MB
- **Languages**: French, English, German, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Turkish, Arabic, Chinese, Japanese, Korean, Hindi
- **Cross-lingual Performance**: Excellent FR-EN semantic matching
- **Features**: Optimized for multilingual document retrieval and cross-language Q&A

> **⚡ Performance Optimized**: This quantized model prioritizes **speed and efficiency for optimal Q&A performance**. 
> - **Speed**: Ultra-fast inference with optimized response times
> - **Memory**: Efficient RAM usage with quantized model
> - **Quality**: Excellent for legal document analysis and Q&A
> - **Multilingual**: Support for French-English cross-lingual queries
> - **Use Case**: Perfect for quick legal document queries and fast responses

---

## 📋 How It Works

This is a **stateless Q&A system**, meaning each question is processed independently without memory of previous interactions. This design choice provides:

- **Optimal Performance**: No context accumulation means faster response times
- **Consistent Results**: Each answer is based solely on the documents, not conversation history
- **Resource Efficiency**: Lower memory usage and computational overhead
- **Reliability**: No risk of context confusion or drift over time

**Note**: This is not a conversational chatbot but rather an intelligent document-based Q&A assistant.

---

## 🛠️ Installation

### Prerequisites

**Option 1: Local Development**
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull the model**: `ollama pull llama3.2:1b-instruct-q4_K_M`

**Option 2: Docker Deployment (Recommended)**
1. **Install Docker**: Download from [docker.com](https://docker.com)
2. **No additional setup needed** - Ollama and model are included in the container

### Setup

### Clone the repo
```bash
git clone https://github.com/<USER>/french-ai-law-guru.git
cd french-ai-law-guru
```

### Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```bash
# Hugging Face Configuration
HF_TOKEN=your_huggingface_token
# Ollama configuration (no API key needed for local models)
OLLAMA_MODEL=llama3.2:1b-instruct-q4_K_M

# LangSmith Configuration (Optional - for tracing and monitoring)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=faq-chatbot
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## 🌍 Multilingual Support

This application now supports **French-English cross-lingual queries**:

- **Questions in French** → **Documents in English** ✅
- **Questions in English** → **Documents in French** ✅
- **Mixed language documents** ✅
- **Automatic language matching** → **Responses in the same language as questions** ✅

### Migration to Multilingual Embeddings

If upgrading from the previous version, you need to regenerate the vectorstore:

```bash
# Delete old vectorstore to force re-embedding with multilingual model
rm -rf chroma_db/
```

The new embedding model will be downloaded automatically on first run (~135MB).

### Example Usage

**French Question with English Documents:**
- Question: "Quelles sont les exigences GDPR pour les applications d'IA ?"
- Answer: "Les exigences GDPR pour les applications d'IA incluent..." (in French)

**English Question with French Documents:**
- Question: "What are the AI Act requirements for transparency?"
- Answer: "The AI Act requirements for transparency include..." (in English)

**Automatic Language Detection:**
- The system automatically detects the language of your question and responds in the same language
- Works with French, English, and other supported languages

### Getting API Keys

**Hugging Face Token:**
1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Add it to your `.env` file

**LangSmith API Key (Optional):**
1. Go to [LangSmith](https://smith.langchain.com/)
2. Sign up or log in
3. Go to Settings > API Keys
4. Create a new API key
5. Add it to your `.env` file

> **Note**: LangSmith integration is optional but recommended for monitoring performance and debugging issues.

## ▶️ Usage

### Quick Start
Run the app using the provided launcher:
```bash
python run_streamlit.py
```

### Alternative Method
Or run directly with Streamlit:
```bash
streamlit run legal_ai_assistant/app.py
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone and run
git clone https://github.com/<USER>/french-ai-law-guru.git
cd french-ai-law-guru
docker-compose up --build
```

The application will be available at `http://localhost:8501`

### Docker Commands

```bash
# Start the application
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Container Features

- ✅ **Ollama pre-installed** with Llama 3.2 1B
- ✅ **Model persistence** between container restarts
- ✅ **Automatic model download** on first run
- ✅ **Health checks** for monitoring
- ✅ **Resource limits** configured (8GB RAM)
- ✅ **Full project mount** to `/work` for live development
- ✅ **Data persistence** with named volumes

### Access the App
Open your browser and go to: http://localhost:8501

### First Run
- The app will automatically process PDF documents in the `legal_docs/` folder
- Embeddings will be created and cached for faster subsequent runs
- If LangSmith is configured, you'll see tracing information in the sidebar

### Document Management
- Place your PDF documents in the `legal_docs/` folder
- The app supports multiple PDF files
- Documents are automatically chunked and processed
- Use the "Clear Cache" button in the sidebar to refresh embeddings when documents change

## 🧪 Testing & Linting

Run tests:
```bash
pytest
```

Run linter:
```bash
ruff check .
```

## 📂 Project Structure
```
french-ai-law-guru/
├── legal_ai_assistant/          # Main application package
│   ├── __init__.py             # Package initialization
│   ├── app.py                  # Main Streamlit application
│   ├── agents.py               # LangGraph agent definitions and workflow
│   ├── chat_handler.py         # Question processing and answer generation
│   ├── local_models.py         # Ollama model client configuration
│   ├── config.py               # Application configuration
│   └── utils.py                # Document processing, embeddings, and caching utilities
│
├── legal_docs/                  # PDF documents for processing
│   ├── AI ACT.pdf             # AI Act regulation document
│   ├── GDPR.pdf               # GDPR regulation document
│   └── faqs.json              # FAQ data (legacy format)
│
├── chroma_db/                  # ChromaDB persistent storage (auto-created)
│
├── tests/                      # Test suite
│   ├── __init__.py            # Test package initialization
│   └── test_utils.py          # Unit tests for utility functions
│
├── notebooks/                  # Jupyter notebooks
│   └── PDFembedding.ipynb     # PDF embedding exploration notebook
│
├── .env                        # Environment variables (create this)
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
├── run_streamlit.py            # Streamlit app launcher
└── README.md                   # This file
```

### Key Components

- **`app.py`**: Main Streamlit interface with document processing and chat functionality
- **`agents.py`**: LangGraph agent implementation for RAG workflow
- **`chat_handler.py`**: Handles question processing and answer generation
- **`utils.py`**: Core utilities for PDF processing, embeddings, and caching
- **`local_models.py`**: Ollama model client configuration for Llama 3.2 1B
- **`config.py`**: Centralized configuration including LLM and embedding settings
- **`chroma_db/`**: Stores processed vectorstores for faster loading
- **`legal_docs/`**: Contains PDF documents that the chatbot processes

## 🔍 LangSmith Integration

This chatbot includes built-in LangSmith integration for monitoring, tracing, and debugging. LangSmith provides valuable insights into the RAG pipeline performance.

### Benefits
- **Performance Monitoring**: Track execution times for each step in the RAG pipeline
- **Token Usage Tracking**: Monitor API costs and usage patterns
- **Error Debugging**: Identify where issues occur in the processing chain
- **Optimization Insights**: Find bottlenecks and optimization opportunities

### Setup
1. **Get LangSmith API Key**: Visit [LangSmith](https://smith.langchain.com/) and create an API key
2. **Configure Environment**: Add LangSmith variables to your `.env` file (see Environment Variables section)
3. **Monitor Performance**: Check the LangSmith dashboard for real-time traces and metrics

### Dashboard Access
- **URL**: https://smith.langchain.com/projects
- **Project**: `faq-chatbot` (configurable via `LANGCHAIN_PROJECT`)
- **Features**: View traces, metrics, token usage, and error logs

### Advanced Configuration
For detailed setup instructions, see the configuration files in the project.

> **💡 Tip**: The app automatically uses the `llama3.2:1b-instruct-q4_K_M` model configured in `legal_ai_assistant/config.py`. You can modify the model settings there if needed.

## ⚡ Performance Optimization

This application is optimized for **ultra-fast inference** using the Llama 3.2 1B model:

### Current Configuration

| Aspect | Llama 3.2 1B Quantified |
|--------|-------------------------|
| **Inference Time** | 8-15 seconds |
| **Model Size** | 1.1GB |
| **RAM Required** | ≈2.5GB |
| **Context Window** | 2048 tokens |
| **Response Quality** | Excellent for Q&A |
| **Use Case** | Fast legal document analysis with memory |

### Performance Features

- **Fast responses**: Optimized for Q&A interactions with context memory
- **Memory efficient**: Quantized model with optimal RAM usage
- **CPU optimized**: Configured for maximum performance on CPU-only systems
- **Context awareness**: Remembers previous tool_rag calls within conversation
- **Document retrieval**: Enhanced context window for better document analysis

## 🐳 Docker Support

The project includes Docker configuration for easy deployment:

```bash
# Build the Docker image
docker build -t french-ai-law-guru .

# Run the container
docker run -p 8501:8501 --env-file .env french-ai-law-guru
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Developed by drikseyy 🚀
