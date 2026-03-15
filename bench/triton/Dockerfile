FROM nvcr.io/nvidia/tritonserver:25.08-py3

WORKDIR /app

# System dependencies
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl ffmpeg jq locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONUNBUFFERED=1

# Copy echo-whisperX source and install it
# Build context is repo root (../../) so this copies the whole project
COPY pyproject.toml README.md /opt/echo-whisperx/
COPY whisperx/ /opt/echo-whisperx/whisperx/
RUN pip install --no-cache-dir /opt/echo-whisperx/ && \
    rm -rf /usr/local/lib/python3.12/dist-packages/numpy-1.26.4.dist-info && \
    pip install --force-reinstall --no-cache-dir "numpy>=2.1.0"

# Assets (pytorch_model.bin, mel_filters.npz) may not be included by pip install
# when building outside a git repo. Copy them explicitly.
RUN cp -r /opt/echo-whisperx/whisperx/assets/ \
    /usr/local/lib/python3.12/dist-packages/whisperx/assets/

ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

COPY bench/triton/Makefile /app/Makefile

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

ENTRYPOINT ["make", "run"]
