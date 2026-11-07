# LAF AI Studios - lean ACE-Step 1.5 serverless worker
# No Gradio UI process, no API server, no always-on pod.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    ACESTEP_PROJECT_ROOT=/opt/ace-step

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Pin a tag here once a build works (e.g. --branch v0.1.6) so updates never surprise you.
ARG ACESTEP_REF=main
RUN git clone --depth 1 --branch ${ACESTEP_REF} https://github.com/ace-step/ACE-Step-1.5.git /opt/ace-step

RUN pip install --no-cache-dir uv \
    && cd /opt/ace-step \
    && uv pip install --system --no-cache -e . \
    && uv pip install --system --no-cache runpod boto3 huggingface_hub hf_transfer

# Optional: bake the main model into the image (bigger image, no download on first start).
# Leave commented if you attach a network volume instead (recommended) or the build times out.
# RUN cd /opt/ace-step && python -c "from acestep.model_downloader import ensure_main_model; ensure_main_model()"

WORKDIR /opt/ace-step
COPY handler.py /opt/ace-step/laf_handler.py

CMD ["python", "-u", "/opt/ace-step/laf_handler.py"]
