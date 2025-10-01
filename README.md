# French AI Law Assistant 🤖

![CI - main branch](https://github.com/driksey/french-ai-law-guru/actions/workflows/ci.yml/badge.svg?branch=main&label=main) ![CI - develop branch](https://github.com/driksey/french-ai-law-guru/actions/workflows/ci.yml/badge.svg?branch=develop&label=develop)

A sophisticated RAG (Retrieval-Augmented Generation) Q&A Assistant built with **Streamlit**, **LangChain**, and **LangGraph** for answering questions about AI regulations in France and Europe.  
It processes PDF documents, creates embeddings, and uses **Gemma 2 2B** and **Gemma 3 270M** via Ollama to generate contextual answers with ultra-fast CPU-optimized inference, structured output, and intelligent question analysis.

---

## 🚀 Features
- **PDF Document Processing**: Automatically loads and processes PDF documents from the `legal_docs/` folder
- **Advanced RAG Architecture**: Uses LangChain and LangGraph for sophisticated document retrieval and generation
- **Intelligent Question Analysis**: Reformulates questions and determines legal relevance before processing
- **Multi-Model Architecture**: Uses Gemma 2 2B for analysis and final answers, Gemma 3 270M for tool calls
- **Structured Output**: Pydantic models ensure robust JSON parsing and validation
- **Vectorstore Caching**: Intelligent caching system to speed up document loading and embedding creation
- **LangSmith Integration**: Built-in tracing and monitoring for performance optimization and debugging
- **Ollama Integration**: Uses Ollama with specialized models for ultra-fast CPU inference and tool calling
- **Local LLM**: Runs entirely locally with Gemma models - no external API dependencies
- **Agent-Based Processing**: Uses LangGraph agents with conditional routing for intelligent question answering
- **Multilingual Support**: French-English cross-lingual queries with automatic language detection
- **Enhanced Legal Responses**: Structured legal answers with direct responses, legal basis, conditions, and consequences
- **User-friendly Interface**: Clean Streamlit interface with cache management and settings
- **CI/CD Pipeline**: Includes linting, testing, and automated workflows  

---

## 🤖 Model Specifications

**Gemma 2 2B via Ollama** (Primary Model)
- **Model Size**: ~1.6GB (optimized by Ollama)
- **RAM Required**: ≈3GB
- **Features**: Ultra-fast inference, excellent JSON generation, RAG optimization, Google's latest architecture
- **Performance**: Superior quality and speed for legal document analysis and Q&A on CPU
- **Usage**: Question analysis, reformulation, and final answer generation
- **Local Processing**: No external API calls required

**Gemma 3 270M via Ollama** (Tool Model)
- **Model Size**: ~291MB (ultra-lightweight)
- **RAM Required**: ≈1GB
- **Features**: Ultra-fast inference, optimized for tool calls and document retrieval
- **Performance**: Lightning-fast processing for structured output and JSON generation
- **Usage**: Tool calls and document retrieval queries
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

This is a **stateless Q&A system** with intelligent question analysis and routing:

### Workflow Architecture

1. **Question Analysis** (Gemma 2 2B)
   - Reformulates questions for optimal document retrieval
   - Determines legal relevance and scope
   - Identifies specific legal domains

2. **Conditional Routing**
   - Legal questions → Document retrieval and analysis
   - Non-legal questions → General response

3. **Document Retrieval** (Gemma 3 270M)
   - Structured tool calls with Pydantic validation
   - Multi-question support for comprehensive search
   - Optimized query generation

4. **Final Answer Generation** (Gemma 2 2B)
   - Structured legal responses with:
     - Direct answer (Légal/Illégal/Partiellement légal)
     - Legal basis with specific references
     - Conditions and requirements
     - Practical consequences
     - Recommendations

### Design Benefits

- **Optimal Performance**: Intelligent routing optimizes model usage
- **Consistent Results**: Each answer is based solely on the documents, not conversation history
- **Resource Efficiency**: Specialized models for different tasks
- **Reliability**: Structured output prevents parsing errors
- **Enhanced Accuracy**: Multi-step analysis improves response quality

**Note**: This is not a conversational chatbot but rather an intelligent document-based legal analysis assistant.

---

## 🛠️ Installation

### Prerequisites

**Option 1: Local Development**
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull the models**: 
   ```bash
   ollama pull gemma2:2b
   ollama pull gemma3:270m
   ```

**Option 2: Docker Deployment (Recommended)**
1. **Install Docker**: Download from [docker.com](https://docker.com)
2. **No additional setup needed** - Ollama and models are included in the container

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
OLLAMA_MODEL_MAIN=gemma2:2b
OLLAMA_MODEL_TOOL=gemma3:270m

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

- ✅ **Ollama pre-installed** with Gemma models
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
│   ├── CELEX_32001L0029_EN_TXT.pdf  # EU Directives and Regulations
│   ├── CELEX_32016R0679_EN_TXT.pdf  # GDPR Regulation
│   ├── CELEX_32019L0790_EN_TXT.pdf  # EU AI Act
│   ├── CELEX_32022R1925_EN_TXT.pdf  # Additional EU Regulations
│   ├── CELEX_32022R2065_EN_TXT.pdf  # EU Regulations
│   ├── CELEX_32024L2853_EN_TXT.pdf  # Latest EU AI Regulations
│   ├── CELEX_52022PC0165_EN_TXT.pdf # EU Proposals
│   ├── CELEX_52022PC0496_EN_TXT.pdf # EU Proposals
│   ├── joe_*.pdf              # French Official Journal documents
│   └── OJ_L_*.pdf             # Official Journal L series documents
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
- **`agents.py`**: LangGraph agent implementation with question analysis, routing, and structured output
- **`chat_handler.py`**: Handles question processing and answer generation
- **`utils.py`**: Core utilities for PDF processing, embeddings, caching, and token calculation
- **`local_models.py`**: Ollama model client configuration for Gemma models
- **`config.py`**: Centralized configuration including multi-model LLM and embedding settings
- **`chroma_db/`**: Stores processed vectorstores for faster loading
- **`legal_docs/`**: Contains EU and French legal PDF documents for processing

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

> **💡 Tip**: The app automatically uses the multi-model configuration in `legal_ai_assistant/config.py`. You can modify the model settings there if needed.

## ⚡ Performance Optimization

This application is optimized for **high-quality inference** using a multi-model architecture:

### Current Configuration

| Aspect | Gemma 2 2B | Gemma 3 270M |
|--------|------------|--------------|
| **Inference Time** | 15-25 seconds | 2-5 seconds |
| **Model Size** | 1.6GB | 291MB |
| **RAM Required** | ≈3GB | ≈1GB |
| **Context Window** | 2048 tokens | 512 tokens |
| **Response Quality** | Superior for Q&A with citations | Optimized for tool calls |
| **Use Case** | Analysis & final answers | Tool calls & retrieval |

### Performance Features

- **Intelligent routing**: Questions are analyzed and routed to appropriate models
- **Structured output**: Pydantic models ensure robust JSON parsing and validation
- **Comprehensive responses**: Optimized for detailed Q&A with full legal citations
- **Memory efficient**: Multi-model architecture with optimized RAM usage
- **CPU optimized**: Configured for maximum performance on CPU-only systems
- **Context awareness**: Dynamic token calculation for optimal response length
- **Enhanced legal analysis**: Structured legal responses with direct answers, legal basis, and consequences
- **Multi-question support**: Tool calls can handle multiple reformulated questions simultaneously

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
