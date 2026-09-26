# RFQ Router demo (Jev) as a container, for any cloud host that runs Docker images.
# The server is standard library only. For the RFQ details beta it also needs Tesseract (OCR for
# scans, faxes, photos, and screenshots) and poppler (pdftoppm rasterizes PDF pages) from apt, and
# pypdf, Pillow, numpy, and onnxruntime (the YOLO region detector) from requirements.txt.
#
#   docker build -t rfq-router .
#   docker run --rm -p 8765:8765 -e AI_GATEWAY_API_KEY=vck_... -e RFQ_DEMO_PASSWORD=pick-one rfq-router
#
# Pass the key at run time (or as a secret on your cloud host). Never bake it into the image.
# Pinned to Debian 13 (trixie), whose Tesseract 5.5.0 was measured with ocr.py --evaluate (81/81 key
# fields, the same as 5.3.4). The bare 3.12-slim tag moves to new Debian releases on its own.
FROM python:3.12-slim-trixie

# MALLOC_ARENA_MAX: every request thread that decodes a page image keeps its own malloc arena, so
# 24 page requests at once left the server at 560 to 760 MB for good; with 2 arenas, 250 MB.
# ORT_DISABLE_TELEMETRY: onnxruntime otherwise sends usage events to Microsoft (layout.py sets it too).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RFQ_DEMO_HOST=0.0.0.0 \
    MALLOC_ARENA_MAX=2 \
    ORT_DISABLE_TELEMETRY=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Compiled once here: the app user cannot write /app, so otherwise every cold start (every wake on
# the free plan) compiles the code again, about 2 s at 0.1 CPU.
RUN python -m compileall -q /app

# Run as a normal user. Only cache/ (saved Jev answers and OCR results for new files) is writable.
# Committed saved answers in data/saved_results.json are read at startup, so replays stay instant
# after a restart.
RUN useradd --uid 10001 --no-create-home --home-dir /app rfq \
    && mkdir -p cache \
    && chown rfq cache
USER rfq

# Listens on $PORT when the host sets it (Cloud Run, Render, Railway), otherwise 8765.
EXPOSE 8765
CMD ["python", "server.py", "--no-browser"]
