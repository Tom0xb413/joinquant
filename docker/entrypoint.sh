#!/bin/sh
# Docker 入口：准备配置与运行时目录，再启动 live-console。
# 以 root 启动时会修正挂载卷属主，再降权为 appuser。
set -eu

cd /app

RUNTIME_DIR="${LIVE_RUNTIME_DIR:-/app/runtime/live}"
DATA_DIR="${LIVE_DATA_DIR:-/app/data/okx}"
CONFIG_PATH="${LIVE_CONFIG_PATH:-/app/configs/live_console.json}"

mkdir -p "${RUNTIME_DIR}" "${DATA_DIR}" configs

if [ ! -f "${CONFIG_PATH}" ]; then
  cp /app/configs/live_console.docker.json "${CONFIG_PATH}"
  echo "[entrypoint] seeded ${CONFIG_PATH} from live_console.docker.json"
fi

if [ "$(id -u)" = "0" ]; then
  chown -R appuser:appuser /app/runtime /app/configs || true
  exec gosu appuser /usr/local/bin/entrypoint.sh "$@"
fi

HOST="${LIVE_CONSOLE_HOST:-0.0.0.0}"
PORT="${LIVE_CONSOLE_PORT:-8787}"

case "${1:-live-console}" in
  live-console)
    shift || true
    exec python3 -m crypto_lab.cli live-console \
      --config "${CONFIG_PATH}" \
      --host "${HOST}" \
      --port "${PORT}" \
      "$@"
    ;;
  trade-book)
    shift || true
    exec python3 -m crypto_lab.cli trade-book "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
