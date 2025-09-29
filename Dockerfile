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

# Créer le script de démarrage
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
\n\
echo "Waiting for Ollama to start..."\n\
for i in {1..30}; do\n\
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then\n\
    echo "Ollama is ready!"\n\
    break\n\
  fi\n\
  echo "Attempt $i/30: Ollama not ready yet, waiting..."\n\
  sleep 2\n\
done\n\
\n\
if ollama list | grep -q "llama3.1:8b"; then\n\
  echo "Llama 3.1:8b model already installed"\n\
else\n\
  echo "Pulling Llama 3.1:8b model..."\n\
  ollama pull llama3.1:8b\n\
fi\n\
\n\
echo "Starting Streamlit application..."\n\
streamlit run run_streamlit.py --server.port=8501 --server.address=0.0.0.0\n\
' > start.sh && chmod +x start.sh

ENV PORT=8501
EXPOSE 8501

CMD ["./start.sh"]