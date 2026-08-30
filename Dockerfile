# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.15 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN uv export --frozen --no-dev --extra media --no-emit-project \
    --prune torch --prune torchvision \
    --output-file /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt

COPY src ./src
RUN uv pip install --system --no-cache --no-deps .

COPY migrations ./migrations
COPY alembic.ini ./
COPY fixtures ./fixtures
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin brandfit \
    && chown -R brandfit:brandfit /app /home/brandfit

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/brandfit \
    XDG_CACHE_HOME=/home/brandfit/.cache
USER 10001:10001
CMD ["uvicorn", "brandfit_core.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
