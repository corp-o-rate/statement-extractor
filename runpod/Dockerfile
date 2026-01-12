# RunPod Serverless Dockerfile for Statement Extractor
# Base image with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers>=4.35.0 \
    accelerate \
    safetensors \
    sentencepiece

# Copy handler
COPY handler.py /app/handler.py

# Download model at build time for faster cold starts
RUN python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Corp-o-Rate-Community/statement-extractor', trust_remote_code=True); \
    AutoModelForSeq2SeqLM.from_pretrained('Corp-o-Rate-Community/statement-extractor', trust_remote_code=True)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

# Run the handler
CMD ["python", "-u", "handler.py"]
