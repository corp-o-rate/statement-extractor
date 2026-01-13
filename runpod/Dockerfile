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
    "transformers @ git+https://github.com/huggingface/transformers.git"

# Download model at build time for faster cold starts
RUN python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Corp-o-Rate-Community/statement-extractor', trust_remote_code=True); \
    AutoModelForSeq2SeqLM.from_pretrained('Corp-o-Rate-Community/statement-extractor', trust_remote_code=True)"

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

# Run the handler
CMD ["python", "-u", "handler.py"]
