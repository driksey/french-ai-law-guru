ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV PORT=8888
EXPOSE 8888

CMD ["streamlit", "run", "app/app.py", "--server.port=8888", "--server.address=0.0.0.0"]
