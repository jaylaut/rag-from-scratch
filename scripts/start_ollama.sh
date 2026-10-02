#!/usr/bin/env bash
# 启动 Ollama 后台服务(macOS)
#
# 为什么要这个脚本:brew 装的 Ollama 靠 ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist
# 实现登录自启,但该 LaunchAgent 只在「下次登录」时才会被 launchd 载入。
# 因此在下一次重启/重新登录之前,用这个脚本按需拉起服务。
#
# 用法:
#   bash scripts/start_ollama.sh          # 已在跑就什么都不做
#   bash scripts/start_ollama.sh --force  # 强制重启
set -euo pipefail

OLLAMA_BIN="$(command -v ollama || echo /opt/homebrew/opt/ollama/bin/ollama)"
HOST="${OLLAMA_HOST:-http://localhost:11434}"

is_up() { curl -sf -m 3 "${HOST}/api/version" >/dev/null 2>&1; }

if [[ "${1:-}" == "--force" ]]; then
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 1
fi

if is_up; then
  echo "✅ Ollama 服务已在运行:$(curl -sf "${HOST}/api/version")"
  exit 0
fi

echo "启动 Ollama 服务 ..."
nohup "$OLLAMA_BIN" serve > /tmp/ollama.log 2>&1 &

for _ in $(seq 1 30); do
  sleep 1
  if is_up; then
    echo "✅ Ollama 已就绪:$(curl -sf "${HOST}/api/version")"
    exit 0
  fi
done

echo "❌ 启动失败,查看日志:tail -50 /tmp/ollama.log" >&2
exit 1
