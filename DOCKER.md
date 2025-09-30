# 🐳 FAQ Chatbot - Docker Deployment

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least 8GB RAM available
- 10GB free disk space

### Start the Application
```bash
# Using Docker Compose (Recommended)
docker-compose up --build

# Start in background
docker-compose up -d --build
```

### Access the Application
- **URL**: http://localhost:8501
- **Model**: Gemma 2:2b (1.6GB)
- **Memory**: 8-10GB RAM required

## 📋 Available Commands

### Using Docker Compose
```bash
# Start the application
docker-compose up --build

# Start in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart
docker-compose restart

# Clean everything (containers, images, volumes)
docker-compose down -v --rmi all
```

## 🔧 Configuration

### Environment Variables
```yaml
environment:
  - OLLAMA_MODEL=gemma2:2b
  - LANGCHAIN_TRACING_V2=false
```

### Resource Requirements
```yaml
deploy:
  resources:
    limits:
      memory: 10G
    reservations:
      memory: 8G
```

### Persistent Volumes
- `ollama_data`: Ollama models (~807MB)
- `chroma_data`: Vector database (~100MB)
- `cache_data`: Application cache (~50MB)

## 🏗️ Architecture

### Container Components
1. **Ollama Service**: Runs Gemma 2:2b model
2. **Streamlit Q&A Assistant**: Web interface on port 8501
3. **ChromaDB**: Vector database for document storage
4. **Python Dependencies**: All required packages

### Startup Process
1. Install Ollama in container
2. Start Ollama service
3. Wait for Ollama to be ready
4. Pull Gemma 2:2b model (if not cached)
5. Start Streamlit Q&A Assistant application

## 🔍 Monitoring

### Health Check
The container includes a health check that monitors:
- Streamlit application availability
- Port 8501 accessibility
- Application responsiveness

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f faq-chatbot
```

### Volume Management
```bash
# List volumes
docker volume ls

# Show volume sizes
docker run --rm -v ollama_data:/source alpine du -sh /source
docker run --rm -v chroma_data:/source alpine du -sh /source
docker run --rm -v cache_data:/source alpine du -sh /source

# Backup volumes
mkdir -p backup/$(date +%Y%m%d_%H%M%S)
docker run --rm -v ollama_data:/source -v $(pwd)/backup:/backup alpine tar czf /backup/ollama_data.tar.gz -C /source .

# Clean volumes (WARNING: destroys data)
docker-compose down -v
```

## 🛠️ Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check Docker is running
docker info

# Check available memory
docker system df

# Clean up space
docker system prune -a
```

#### Model Download Fails
```bash
# Check internet connection
docker-compose exec faq-chatbot ping google.com

# Manual model pull
docker-compose exec faq-chatbot ollama pull gemma2:2b
```

#### Memory Issues
```bash
# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 12G
```

#### Port Conflicts
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 instead
```

### Performance Optimization

#### For Better Performance
- Use SSD storage
- Allocate more RAM (10GB+)
- Close unnecessary applications
- Use Docker Desktop with WSL2 backend (Windows)

#### For Lower Resource Usage
- Reduce memory limits to 6GB minimum
- Use smaller model (if available)
- Disable health checks

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Startup Time | 2-5 minutes (first run) |
| Model Load Time | 30-60 seconds |
| Response Time | 3-8 seconds |
| Memory Usage | 6-8GB |
| Disk Usage | ~5GB |

## 🔄 Updates

### Update Model
```bash
# Pull latest Gemma 2:2b
docker-compose exec faq-chatbot ollama pull gemma2:2b
```

### Update Application
```bash
# Rebuild with latest code
docker-compose up --build

# Or pull latest image
docker-compose pull && docker-compose up
```

## 🎯 Production Deployment

### Security Considerations
- Use non-root user in container
- Limit container capabilities
- Use secrets management
- Enable HTTPS in production

### Scaling
- Use Docker Swarm or Kubernetes
- Implement load balancing
- Use external database for ChromaDB
- Implement health checks

### Monitoring
- Set up logging aggregation
- Monitor resource usage
- Implement alerting
- Track performance metrics

## 📝 Notes

- First startup takes longer due to model download
- Model is cached in `ollama_data` volume
- ChromaDB data persists between restarts
- All configuration is in `docker-compose.yml`
- Use standard Docker Compose commands for management
