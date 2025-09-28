# LangSmith Setup Guide

## Overview
LangSmith has been integrated into the FAQ chatbot to provide tracing and debugging capabilities. This helps identify performance bottlenecks and debug issues.

## Setup Instructions

### 1. Install LangSmith
```bash
pip install langsmith
```

### 2. Get LangSmith API Key
1. Go to [LangSmith](https://smith.langchain.com/)
2. Sign up or log in
3. Go to Settings > API Keys
4. Create a new API key

### 3. Configure Environment Variables

#### Option A: Using .env file (Recommended)
Create a `.env` file in the project root and add:

```bash
# LangSmith Configuration
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=faq-chatbot
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

#### Option B: Using environment variables directly
Set the environment variable in your terminal:

**Windows (PowerShell):**
```powershell
$env:LANGCHAIN_API_KEY="your_langsmith_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

**Linux/Mac:**
```bash
export LANGCHAIN_API_KEY="your_langsmith_api_key_here"
```

#### Option C: Using Streamlit secrets
Create `.streamlit/secrets.toml` file:
```toml
[langsmith]
api_key = "your_langsmith_api_key_here"
project = "faq-chatbot"
```

### 4. Verify Configuration
To check if LangSmith is properly configured, you can run this Python script:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Check if environment variables are set
api_key = os.getenv("LANGCHAIN_API_KEY")
project = os.getenv("LANGCHAIN_PROJECT")
tracing = os.getenv("LANGCHAIN_TRACING_V2")

print(f"API Key set: {'✅' if api_key else '❌'}")
print(f"Project: {project}")
print(f"Tracing enabled: {tracing}")

if api_key:
    print("🎉 LangSmith is properly configured!")
else:
    print("⚠️  Please set LANGCHAIN_API_KEY in your .env file")
```

### 5. Usage
- The app automatically traces all LangChain operations
- Check the LangSmith dashboard to see:
  - Execution times for each step
  - Token usage
  - Error traces
  - Performance metrics

### 6. Dashboard Access
- Visit: https://smith.langchain.com/projects
- Select your "faq-chatbot" project
- View traces, metrics, and debug information

## Benefits
- **Performance Monitoring**: See which operations take the longest
- **Debugging**: Identify where errors occur in the chain
- **Token Usage**: Track API costs
- **Optimization**: Identify bottlenecks for improvement
