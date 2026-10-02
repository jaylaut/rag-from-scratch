#!/usr/bin/env bash
# 一键验收:激活 venv → 确保 Ollama 在跑 → 跑 scripts/verify_setup.py
#
# 用法:
#   bash scripts/verify.sh                # 全量验收(不花钱)
#   bash scripts/verify.sh --skip-embed   # 快速验收
#   bash scripts/verify.sh --live         # 含 Qwen 真实调用(约花几分钱)
#
# 参数会原样透传给 verify_setup.py。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # 切到项目根:verify_setup.py 按自身位置推算路径,但 pytest 需要在根跑

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "❌ 未找到 .venv。先执行: python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

bash scripts/start_ollama.sh

exec python scripts/verify_setup.py "$@"
