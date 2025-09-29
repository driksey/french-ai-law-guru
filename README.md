# French AI Law Guru 🤖

![CI - main](https://github.com/driksey/french-ai-law-guru/actions/workflows/ci.yml/badge.svg?branch=main) ![CI - develop](https://github.com/driksey/french-ai-law-guru/actions/workflows/ci.yml/badge.svg?branch=develop)

A sophisticated RAG (Retrieval-Augmented Generation) chatbot built with **Streamlit**, **LangChain**, and **LangGraph** for answering questions about AI regulations in France.  
It processes PDF documents, creates embeddings, and uses Hugging Face models to generate contextual answers.

---

## 🚀 Features
- **PDF Document Processing**: Automatically loads and processes PDF documents from the `legal_docs/` folder
- **Advanced RAG Architecture**: Uses LangChain and LangGraph for sophisticated document retrieval and generation
- **Vectorstore Caching**: Intelligent caching system to speed up document loading and embedding creation
- **LangSmith Integration**: Built-in tracing and monitoring for performance optimization and debugging
- **Ollama Integration**: Uses Ollama with Llama 3.1 8B for advanced reasoning and tool calling
- **Agent-Based Processing**: Uses LangGraph agents for intelligent question answering
- **User-friendly Interface**: Clean Streamlit interface with cache management and settings
- **CI/CD Pipeline**: Includes linting, testing, and automated workflows  

---

## 🛠️ Installation

### Prerequisites

**Option 1: Local Development**
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull the model**: `ollama pull llama3.1:8b`

**Option 2: Docker Deployment (Recommended)**
1. **Install Docker**: Download from [docker.com](https://docker.com)
2. **No additional setup needed** - Ollama and model are included in the container

### Setup

# Clone the repo
```
git clone https://github.com/<USER>/french-ai-law-guru.git
cd french-ai-law-guru
```

# Create virtual environment
```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

# Install dependencies
```
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
OLLAMA_MODEL=deepseek-r1:7b

# LangSmith Configuration (Optional - for tracing and monitoring)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=faq-chatbot
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

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
streamlit run faq_chatbot/app.py
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone and run
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
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

- ✅ **Ollama pre-installed** with DeepSeek R1 7B
- ✅ **Model persistence** between container restarts
- ✅ **Automatic model download** on first run
- ✅ **Health checks** for monitoring
- ✅ **Resource limits** configured (8GB RAM)
- ✅ **Full project mount** to `/work` for live development
- ✅ **Data persistence** with named volumes

### Access the App
Open your browser and go to: http://localhost:8501

### First Run
- The app will automatically process PDF documents in the `docs/` folder
- Embeddings will be created and cached for faster subsequent runs
- If LangSmith is configured, you'll see tracing information in the sidebar

### Document Management
- Place your PDF documents in the `docs/` folder
- The app supports multiple PDF files
- Documents are automatically chunked and processed
- Use the "Clear Cache" button in the sidebar to refresh embeddings when documents change

## 🧪 Testing & Linting

Run tests:

pytest


Run linter:

ruff check .

## 📂 Project Structure
```
faq-chatbot/
├── faq_chatbot/                 # Main application package
│   ├── __init__.py             # Package initialization
│   ├── app.py                  # Main Streamlit application
│   ├── agents.py               # LangGraph agent definitions and workflow
│   ├── chat_handler.py         # Question processing and answer generation
│   ├── local_models.py         # Hugging Face model client configuration
│   ├── config.py               # Application configuration
│   └── utils.py                # Document processing, embeddings, and caching utilities
│
├── docs/                       # PDF documents for processing
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
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup configuration
├── run_streamlit.py            # Streamlit app launcher
├── LOCAL_MODELS_GUIDE.md       # Local models configuration guide
├── LANGSMITH_SETUP.md          # LangSmith configuration guide
└── README.md                   # This file
```

### Key Components

- **`app.py`**: Main Streamlit interface with document processing and chat functionality
- **`agents.py`**: LangGraph agent implementation for RAG workflow
- **`chat_handler.py`**: Handles question processing and answer generation
- **`utils.py`**: Core utilities for PDF processing, embeddings, and caching
- **`local_models.py`**: Hugging Face model client configuration
- **`chroma_db/`**: Stores processed vectorstores for faster loading
- **`docs/`**: Contains PDF documents that the chatbot processes

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
For detailed setup instructions, see [LANGSMITH_SETUP.md](LANGSMITH_SETUP.md).

## 🐳 Docker Support

The project includes Docker configuration for easy deployment:

```bash
# Build the Docker image
docker build -t faq-chatbot .

# Run the container
docker run -p 8501:8501 --env-file .env faq-chatbot
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
