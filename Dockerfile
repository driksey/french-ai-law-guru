ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /work

# Install system dependencies including Ollama
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application
COPY . /work

# Create a startup script that pulls the model and starts the app
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start Ollama in background\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to be ready\n\
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
# Check if model is already installed\n\
if ollama list | grep -q "llama3.1:8b"; then\n\
  echo "Llama 3.1:8b model already installed"\n\
else\n\
  echo "Pulling Llama 3.1:8b model..."\n\
  ollama pull llama3.1:8b\n\
fi\n\
\n\
# Start Streamlit\n\
echo "Starting Streamlit application..."\n\
streamlit run run_streamlit.py --server.port=8501 --server.address=0.0.0.0\n\
' > start.sh && chmod +x start.sh

ENV PORT=8501
EXPOSE 8501

# Run the startup script
CMD ["./start.sh"]
