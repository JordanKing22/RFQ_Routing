# RFQ Router demo (Jev) as a container, for any cloud host that runs Docker images.
# Standard library only, so there is no pip install step.
#
#   docker build -t rfq-router .
#   docker run --rm -p 8765:8765 -e AI_GATEWAY_API_KEY=vck_... -e RFQ_DEMO_PASSWORD=pick-one rfq-router
#
# Pass the key at run time (or as a secret on your cloud host). Never bake it into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RFQ_DEMO_HOST=0.0.0.0

WORKDIR /app
COPY . .

# Run as a normal user. Only cache/ (saved Jev answers) is writable.
RUN useradd --uid 10001 --no-create-home --home-dir /app rfq \
    && mkdir -p cache \
    && chown rfq cache
USER rfq

# Listens on $PORT when the host sets it (Cloud Run, Render, Railway), otherwise 8765.
EXPOSE 8765
CMD ["python", "server.py", "--no-browser"]
