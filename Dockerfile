# Basisimage met CUDA
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Installeer basisvereisten
RUN apt update && apt install -y python3 python3-pip git

# Zet werkdirectory
WORKDIR /app

# Installeer vereiste Python-pakketten
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers pillow fastapi uvicorn

# Kopieer de API
COPY server.py /app/server.py

# Start de API
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
