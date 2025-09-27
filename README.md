# Mini FAQ Chatbot 🤖

![CI - main](https://github.com/driksey/faq-chatbot/actions/workflows/ci.yml/badge.svg?branch=main) ![CI - develop](https://github.com/driksey/faq-chatbot/actions/workflows/ci.yml/badge.svg?branch=develop)



A lightweight FAQ chatbot built with **Streamlit** and Hugging Face models (`google/gemma-2-2b-it` by default).  
It retrieves the most relevant FAQ entries and generates answers based on context.

---

## 🚀 Features
- Loads FAQs from a simple `faq.json` file  
- Finds similar questions using **embeddings**  
- Generates answers via Hugging Face Inference API  
- User-friendly **Streamlit** interface  
- Includes **CI/CD** with linting and tests  

---

## 🛠️ Installation

# Clone the repo
```
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
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

Create a .env file in the project root:
```
HF_TOKEN=your_huggingface_token
HF_MODEL=google/gemma-2-2b-it
```

## ▶️ Usage

Run the app locally:
```
streamlit run app/main.py
```

Then open http://localhost:8501
.

## 🧪 Testing & Linting

Run tests:

pytest


Run linter:

ruff check .

## 📂 Project Structure
```
faq-chatbot/
│── faq_chatbot/
│   ├── __init__.py      # Makes the folder a Python package
│   ├── app.py           # Main Streamlit app
│   ├── embeddings.py    # Functions to compute/load embeddings
│   ├── faqs.json        # FAQ data
│   ├── utils.py         # Helper functions: FAQ loading, similarity search
│   ├── hf_client.py     # Hugging Face model API client
│
│── tests/
│   ├── test_utils.py    # Unit tests for utils.py
│   ├── __init__.py      # Makes the folder a Python package
│
│── .gitignore            # Git ignore rules
│── Dockerfile            # Docker configuration
│── requirements.txt      # Python dependencies
│── setup.py              # Package setup
│── test_hf_token.py      # Script to verify Hugging Face token
│── run_streamlit.py      # Shortcut to launch the Streamlit app
│── .env.example          # Sample environment variables
│── .github/workflows/ci.yml  # CI pipeline configuration
│── .pre-commit-config.yaml   # Pre-commit hooks configuration
```

## 👨‍💻 Author

Developed by drikseyy 🚀
