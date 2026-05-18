FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias primero (optimiza caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar stopwords de NLTK durante el build
RUN python -m nltk.downloader stopwords

# Copiar el resto del proyecto
COPY app/       ./app/
COPY models/    ./models/
COPY artifacts/ ./artifacts/

# Puerto expuesto
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]