# RFQ Router demo (Jev) as a container, for any cloud host that runs Docker images.
# The server is standard library only. For the RFQ details beta it also needs Tesseract (OCR for
# scans, faxes, photos, and screenshots) and poppler (pdftoppm rasterizes PDF pages) from apt, and
# pypdf, Pillow, numpy, and onnxruntime (the YOLO region detector) from requirements.txt.
#
#   docker build -t rfq-router .
#   docker run --rm -p 8765:8765 -e AI_GATEWAY_API_KEY=vck_... -e RFQ_DEMO_PASSWORD=pick-one rfq-router
#
# Pass the key at run time (or as a secret on your cloud host). Never bake it into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RFQ_DEMO_HOST=0.0.0.0

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Run as a normal user. Only cache/ (saved Jev answers) is writable. Committed saved answers in
# data/saved_results.json are read at startup, so replays stay instant after a restart.
RUN useradd --uid 10001 --no-create-home --home-dir /app rfq \
    && mkdir -p cache \
    && chown rfq cache
USER rfq

# Listens on $PORT when the host sets it (Cloud Run, Render, Railway), otherwise 8765.
EXPOSE 8765
CMD ["python", "server.py", "--no-browser"]
