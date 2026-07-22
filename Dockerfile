# syntax=docker/dockerfile:1
# Live Web 控制台 + 打包策略的可部署镜像。
# 构建：docker compose build
# 启动：docker compose up -d --build

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    LIVE_CONSOLE_HOST=0.0.0.0 \
    LIVE_CONSOLE_PORT=8787 \
    LIVE_DATA_DIR=/app/data/okx \
    LIVE_RUNTIME_DIR=/app/runtime/live

WORKDIR /app

# curl：健康检查；gosu：入口降权
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY pyproject.toml README.md ./
COPY crypto_lab ./crypto_lab
COPY configs ./configs
COPY data/okx ./data/okx
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN pip install --upgrade pip \
    && pip install . \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/runtime/live /app/configs \
    && chown -R appuser:appuser /app

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${LIVE_CONSOLE_PORT:-8787}/login" >/dev/null || exit 1

# 以 root 进入 entrypoint，修正卷权限后 gosu 到 appuser
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["live-console"]
