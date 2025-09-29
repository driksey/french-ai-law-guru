import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "faq_chatbot/app.py"])