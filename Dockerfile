FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir --default-timeout=300 --retries=10 --index-url https://pypi.org/simple -r requirements.txt

COPY app.py .
COPY models/ ./models/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]