#!/usr/bin/env bash
# One-shot installer for local-dictation on Arch, Ubuntu/Debian, and Fedora.
#
# Installs the system packages dictation needs, syncs the service environment
# (.venv-service), links `dictate` into ~/.local/bin, and installs, audits, and
# starts the local-dictation systemd user service. Safe to re-run: each step is
# idempotent.
#
# Usage:
#   ./scripts/install.sh [--device auto|cpu|cuda]
#
# Overridable via environment:
#   PYTHON_VERSION           Python for the service environment (default 3.11)
#   DICTATION_DEVICE         Speech-to-text compute: auto, cpu, or cuda (default
#                            auto); --device takes precedence
#   SKIP_SYSTEM_PACKAGES=1   Skip distro package installation
#
# `auto` installs the cuda extra only when a bounded nvidia-smi probe confirms a
# usable NVIDIA driver; otherwise it installs the cpu extra. `cuda` is a strict
# request and installs the cuda extra without probing.
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# The script lives in the checkout, so derive the project root from its path
# rather than cloning again.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# The shipped unit runs %h/local-dictation/.venv-service/bin/local-dictation-server.
SERVICE_VENV="$PROJECT_DIR/.venv-service"

# Remember the caller's PATH so the hint about ~/.local/bin reflects their shell,
# then make a freshly bootstrapped uv visible to the rest of this run.
CALLER_PATH="$PATH"
export PATH="$HOME/.local/bin:$PATH"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

require() {
  local missing=0 c
  for c in "$@"; do
    if ! have "$c"; then
      warn "required command not found: $c"
      missing=1
    fi
  done
  return "$missing"
}

python_planner_available() {
  have python3 &&
    python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
      >/dev/null 2>&1
}

as_root() {
  if [[ "$EUID" -eq 0 ]]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    warn "sudo is required to install the Python planning prerequisite."
    return 1
  fi
}

bootstrap_installer_python() {
  if python_planner_available; then
    return 0
  fi
  if [[ "${SKIP_SYSTEM_PACKAGES:-0}" == 1 ]]; then
    warn "Python 3.11 or newer is required to resolve the installation plan."
    return 1
  fi
  local release="${OS_RELEASE_PATH:-/etc/os-release}" identity="" key value
  if [[ ! -r "$release" ]]; then
    warn "Cannot read $release to install the Python planning prerequisite."
    return 1
  fi
  while IFS="=" read -r key value; do
    case "$key" in
      ID|ID_LIKE)
        value="${value#\"}"
        value="${value%\"}"
        identity+=" ${value,,}"
        ;;
    esac
  done <"$release"
  step "Installing Python planning prerequisite"
  case "$identity" in
    *arch*|*cachyos*|*manjaro*|*endeavouros*)
      if ! have paru; then
        warn "paru is required to install Python on Arch-family systems."
        return 1
      fi
      paru -S --needed --noconfirm python
      ;;
    *ubuntu*|*debian*|*pop*|*linuxmint*|*elementary*)
      as_root apt-get update
      as_root apt-get install -y --no-install-recommends python3
      ;;
    *fedora*|*rhel*|*centos*|*nobara*)
      as_root dnf install -y python3
      ;;
    *)
      warn "Unsupported distribution; install Python 3.11 or newer manually."
      return 1
      ;;
  esac
  if ! python_planner_available; then
    warn "Python 3.11 or newer is required to resolve the installation plan."
    return 1
  fi
}

resolve_install_plan() {
  PYTHONPATH="$PROJECT_DIR/src" python3 -m local_dictation.install_distro "$@"
}

cuda_capability_args() {
  if have nvidia-smi && timeout 10 nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "--cuda-available"
  fi
}

install_distro_packages() {
  local -a command
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  read -r -a command <<<"$INSTALL_PACKAGE_COMMAND"
  "${command[@]}" "$@"
}

# The shipped unit hardcodes %h/local-dictation, so a checkout elsewhere is
# reached through a symlink at that path.
link_service_checkout() {
  local link="$1" checkout="$2" current
  current="$(readlink -f -- "$link" 2>/dev/null || true)"
  if [[ "$current" == "$checkout" ]]; then
    return 0
  fi
  if [[ -L "$link" ]]; then
    ln -sfn -- "$checkout" "$link"
    info "Updated $link -> $checkout for the shipped user unit"
  elif [[ -e "$link" ]]; then
    warn "$link exists and is not a symlink; refusing to replace it."
    warn "Move that path aside or run the installer from $link."
    return 1
  else
    ln -s -- "$checkout" "$link"
    info "Linked $link -> $checkout for the shipped user unit"
  fi
}

# Retarget only links into a local-dictation service environment, such as one
# left by a moved checkout; anything else at that path belongs to another tool.
link_dictate() {
  local link="$1" target="$2" current
  if [[ -L "$link" ]]; then
    current="$(readlink -- "$link")"
    if [[ "$current" != "$target" && "$current" != */.venv-service/bin/dictate ]]; then
      warn "$link points to $current; refusing to replace it."
      warn "Remove that link, then re-run the installer."
      return 1
    fi
  elif [[ -e "$link" && ! "$link" -ef "$target" ]]; then
    warn "$link exists and is not a symlink; refusing to replace it."
    warn "Move it aside, then re-run the installer."
    return 1
  fi
  mkdir -p -- "$(dirname -- "$link")"
  ln -sfn -- "$target" "$link"
  info "Linked $link -> $target"
}

main() {
  local device="${DICTATION_DEVICE:-auto}" plan
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --device)
        if [[ $# -lt 2 ]]; then
          warn "--device requires auto, cpu, or cuda."
          exit 1
        fi
        device="$2"
        shift 2
        ;;
      --device=*)
        device="${1#*=}"
        shift
        ;;
      *)
        warn "unknown installer argument: $1"
        exit 1
        ;;
    esac
  done

  cd "$PROJECT_DIR"

  bold "local-dictation installer"
  info "Project directory: $PROJECT_DIR"
  bootstrap_installer_python

  step "Resolving the installation plan"
  local -a plan_args cuda_args
  mapfile -t cuda_args < <(cuda_capability_args)
  plan_args=(
    --format env
    --checkout "$PROJECT_DIR"
    --os-release "${OS_RELEASE_PATH:-/etc/os-release}"
    --device "$device"
    "${cuda_args[@]}"
  )
  if [[ "${SKIP_SYSTEM_PACKAGES:-0}" == 1 ]]; then
    plan_args+=(--skip-system-packages)
  fi
  plan="$(resolve_install_plan "${plan_args[@]}")" || exit 1
  eval "$plan"
  info "Device: $INSTALL_DEVICE (installs the $INSTALL_EXTRA extra)"
  info "Distribution: $INSTALL_DISTRO_ID ($INSTALL_DISTRO_FAMILY family, $INSTALL_PACKAGE_MANAGER)"
  info "Checkout: $INSTALL_CHECKOUT"
  info "User services: $INSTALL_SYSTEMD_USER_DIR"
  info "Configuration: $INSTALL_CONFIG_DIR"
  link_service_checkout "$INSTALL_SERVICE_CHECKOUT" "$INSTALL_CHECKOUT"

  # --- 1. System packages ---------------------------------------------------
  if [[ "${SKIP_SYSTEM_PACKAGES:-0}" == 1 ]]; then
    step "Skipping system packages (SKIP_SYSTEM_PACKAGES=1)"
    info "Make sure these are installed: $INSTALL_DISTRO_PACKAGES $INSTALL_DISTRO_CUDA_PACKAGES $INSTALL_SKIPPED_PACKAGES"
  else
    if [[ "$INSTALL_DISTRO_FAMILY" == "arch" ]] && ! have paru; then
      warn "paru is required on Arch to install system packages."
      warn "Install paru, or rerun with SKIP_SYSTEM_PACKAGES=1 after installing dependencies."
      exit 1
    fi
    if [[ "$INSTALL_DISTRO_FAMILY" == "debian" ]] && ! have apt-get; then
      warn "apt-get is required on Ubuntu/Debian."
      exit 1
    fi
    if [[ "$INSTALL_DISTRO_FAMILY" == "fedora" ]] && ! have dnf; then
      warn "dnf is required on Fedora."
      exit 1
    fi
    step "Installing system packages ($INSTALL_PACKAGE_MANAGER)"
    # shellcheck disable=SC2086
    install_distro_packages $INSTALL_DISTRO_PACKAGES
    if [[ -n "$INSTALL_DISTRO_CUDA_PACKAGES" ]]; then
      step "Installing CUDA runtime packages"
      # shellcheck disable=SC2086
      install_distro_packages $INSTALL_DISTRO_CUDA_PACKAGES
    else
      info "Skipping CUDA packages ($INSTALL_EXTRA extra)"
    fi
    if [[ -n "$INSTALL_SKIPPED_PACKAGES" ]]; then
      info "Packages handled outside the distro package manager: $INSTALL_SKIPPED_PACKAGES"
    fi
    if [[ "$INSTALL_UV_BOOTSTRAP" == 1 ]] && ! have uv; then
      step "Installing uv"
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
  fi

  require uv || {
    warn "Install the missing commands above, then re-run."
    exit 1
  }

  # --- 2. Service environment -----------------------------------------------
  # A separate environment, because `uv sync` in the checkout manages .venv
  # exactly and would drop the cpu/cuda extra the service needs.
  step "Syncing the service environment (.venv-service, $INSTALL_EXTRA extra)"
  UV_PROJECT_ENVIRONMENT="$SERVICE_VENV" \
    uv sync --python "$PYTHON_VERSION" --extra "$INSTALL_EXTRA" --no-dev

  # --- 3. dictate command ---------------------------------------------------
  step "Linking dictate into $INSTALL_USER_BIN"
  link_dictate "$INSTALL_DICTATE" "$SERVICE_VENV/bin/dictate"
  case ":$CALLER_PATH:" in
    *":$INSTALL_USER_BIN:"*) : ;;
    *) warn "Add $INSTALL_USER_BIN to your PATH to use the dictate command." ;;
  esac

  # --- 4. Configuration -----------------------------------------------------
  step "Configuring the speech-to-text device"
  "$INSTALL_DICTATE" config set stt.device "$INSTALL_DEVICE"

  # --- 5. Service -----------------------------------------------------------
  step "Installing and starting the local-dictation user service"
  "$INSTALL_DICTATE" services install
  "$INSTALL_DICTATE" services audit
  "$INSTALL_DICTATE" services start

  # --- Done -----------------------------------------------------------------
  step "Install complete"
  bold "Next steps:"
  info "1. Pick a microphone (list PipeWire capture sources with: wpctl status):"
  info "     dictate config set dictation.source <source-name>"
  info "2. Verify the setup:"
  info "     dictate doctor"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
