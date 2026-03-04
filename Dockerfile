FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

COPY requirements.deploy.txt /app/requirements.deploy.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.deploy.txt

COPY . /app

EXPOSE 8501

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true
