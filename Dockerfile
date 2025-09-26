ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV PORT=8501
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
