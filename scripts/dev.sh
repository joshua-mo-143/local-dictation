#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/dev.sh server
  scripts/dev.sh dictate <command> [arguments...]

Runs the STT server and the dictate client from this checkout with their own
runtime directory, socket, configuration, and virtual environment. The
installed local-dictation service is never started, stopped, or reconfigured.

Environment:
  DICTATION_DEV_EXTRA  engine extra for .dev/venv: cpu (default) or cuda
  PYTHON_VERSION       Python for .dev/venv (default 3.11)
EOF
}

command="${1:-}"
case "$command" in
  server)
    shift
    if (($# != 0)); then
      usage
      exit 2
    fi
    ;;
  dictate)
    shift
    if (($# == 0)); then
      usage
      exit 2
    fi
    if [[ "$1" == "services" ]]; then
      echo "scripts/dev.sh: the development launcher never manages the installed service" >&2
      exit 2
    fi
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

extra="${DICTATION_DEV_EXTRA:-cpu}"
if [[ "$extra" != cpu && "$extra" != cuda ]]; then
  echo "scripts/dev.sh: DICTATION_DEV_EXTRA must be cpu or cuda" >&2
  exit 2
fi
if [[ "${XDG_RUNTIME_DIR:-}" != /* ]]; then
  echo "scripts/dev.sh: XDG_RUNTIME_DIR must be an absolute path" >&2
  exit 1
fi

export DICTATION_RUNTIME_DIR="$XDG_RUNTIME_DIR/local-dictation-dev"
export XDG_CONFIG_HOME="$PROJECT_DIR/.dev/config"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.dev/venv"
mkdir -p -- "$XDG_CONFIG_HOME"
(umask 077 && mkdir -p -- "$DICTATION_RUNTIME_DIR")
chmod 700 -- "$DICTATION_RUNTIME_DIR"

uv_run=(
  uv run
  --project "$PROJECT_DIR"
  --python "${PYTHON_VERSION:-3.11}"
  --extra "$extra"
)
if [[ "$command" == server ]]; then
  exec "${uv_run[@]}" local-dictation-server
fi
exec "${uv_run[@]}" dictate "$@"
