# RunPod Serverless Dockerfile for Statement Extractor
# Use a newer base image with compatible torch/transformers versions
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python dependencies
# NOTE: T5Gemma2 requires the dev version of transformers from GitHub
RUN pip install --no-cache-dir \
    runpod \
    accelerate \
    safetensors \
    sentencepiece \
    "transformers @ git+https://github.com/huggingface/transformers.git"

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
