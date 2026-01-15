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
    spacy \
    "transformers @ git+https://github.com/huggingface/transformers.git"

# Install corp-extractor library
RUN pip install --no-cache-dir --no-deps corp-extractor==0.3.0

# Download models at build time for faster cold starts
RUN python -c "from statement_extractor import StatementExtractor; \
    extractor = StatementExtractor(); \
    print(f'T5-Gemma model loaded on {extractor.device}')"

# Pre-download embedding model (downloads on CPU during build, will use GPU at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2'); \
    print('Embedding model downloaded')"

#Download spacy files
RUN python -m spacy download en_core_web_sm

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

# Run the handler
CMD ["python", "-u", "handler.py"]
