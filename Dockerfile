# Dockerfile minimal optimisé avec LangSmith
FROM python:3.11-slim

WORKDIR /work

# Installer uniquement les dépendances essentielles
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Installer Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Installer les dépendances Python optimisées
RUN pip install --no-cache-dir \
    streamlit \
    python-dotenv \
    langchain \
    langchain-community \
    langchain-core \
    langchain-chroma \
    langchain-ollama \
    langchain-huggingface \
    langgraph \
    chromadb \
    sentence-transformers \
    scikit-learn \
    numpy \
    pypdf \
    ollama \
    langsmith

# Copier l'application
COPY . /work

# Créer le script de démarrage avec une méthode plus robuste
RUN cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting Ollama service..."
ollama serve &

echo "Waiting for Ollama to start..."
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is ready!"
    break
  fi
  echo "Attempt $i/30: Ollama not ready yet, waiting..."
  sleep 2
done

if ollama list | grep -q "llama3.2:1b-instruct-q4_K_M"; then
  echo "Llama 3.2:1b-instruct-q4_K_M model already installed"
else
  echo "Pulling Llama 3.2:1b-instruct-q4_K_M model..."
  ollama pull llama3.2:1b-instruct-q4_K_M
fi

echo "Starting Streamlit Q&A Assistant application..."
streamlit run run_streamlit.py --server.port=8501 --server.address=0.0.0.0
EOF

RUN chmod +x start.sh

ENV PORT=8501
EXPOSE 8501

CMD ["./start.sh"]