# RunPod Serverless Dockerfile for Statement Extractor
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Upgrade PyTorch to 2.6+ (required for T5Gemma2 masking_utils)
RUN pip install --no-cache-dir --upgrade \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
# NOTE: T5Gemma2 requires the dev version of transformers from GitHub
RUN pip install --no-cache-dir \
    runpod \
    accelerate \
    safetensors \
    sentencepiece \
    pydantic \
    sentence-transformers \
    gliner2 \
    "transformers @ git+https://github.com/huggingface/transformers.git"


# Pre-download embedding model (downloads on CPU during build, will use GPU at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2'); \
    print('Embedding model downloaded')"

# Pre-download GLiNER2 model for faster cold starts
RUN python -c "from gliner2 import GLiNER2; \
    model = GLiNER2.from_pretrained('fastino/gliner2-base-v1'); \
    print('GLiNER2 model downloaded')"

# Install corp-extractor library
RUN pip install --no-cache-dir --no-deps corp-extractor==0.4.1

# Download models at build time for faster cold starts
RUN python -c "from statement_extractor import StatementExtractor; \
    extractor = StatementExtractor(); \
    print(f'T5-Gemma2 model loaded on {extractor.device}')"

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

# Run the handler
CMD ["python", "-u", "handler.py"]
