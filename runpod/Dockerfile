# RunPod Serverless Dockerfile for Statement Extractor
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# HuggingFace token for gated models (pass with --build-arg HF_TOKEN=...)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

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
    huggingface_hub \
    "transformers @ git+https://github.com/huggingface/transformers.git"

# Install llama-cpp-python with CUDA support for GGUF models
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-cache-dir llama-cpp-python

# Pre-download embedding models (downloads on CPU during build, will use GPU at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2'); \
    print('MiniLM embedding model downloaded')"

# Pre-download EmbeddingGemma model for taxonomy classification
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('google/embeddinggemma-300m'); \
    print('EmbeddingGemma-300m model downloaded')"

# Pre-download GLiNER2 model for faster cold starts
RUN python -c "from gliner2 import GLiNER2; \
    model = GLiNER2.from_pretrained('fastino/gliner2-base-v1'); \
    print('GLiNER2 model downloaded')"

# Pre-download Gemma3 GGUF model for person qualification
RUN python -c "from huggingface_hub import hf_hub_download; \
    path = hf_hub_download(repo_id='google/gemma-3-12b-it-qat-q4_0-gguf', filename='gemma-3-12b-it-q4_0.gguf'); \
    print(f'Gemma3 GGUF model downloaded to {path}')"

# Install corp-extractor library
RUN pip install --no-cache-dir --no-deps corp-extractor==0.5.0

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
