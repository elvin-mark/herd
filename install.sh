#!/usr/bin/env bash
# ==============================================================================
# Herd CLI One-Line Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/elvin-mark/herd/main/install.sh | bash
# ==============================================================================

set -e

# Styling & Colors
BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
echo "    __  ______  ____  ____ "
echo "   / / / / __ \/ __ \/ __ \\"
echo "  / /_/ / /_/ / /_/ / / / /"
echo " / __  / ____/ _, _/ /_/ / "
echo "/_/ /_/_/   /_/ |_/_____/  "
echo "                           "
echo "Herd Local AI Model Servicing & Gateway"
echo -e "${RESET}"

# 1. Check Dependencies (Python 3.10+)
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}[!] Error: python3 is required but not installed.${RESET}"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo -e "${RED}[!] Error: Python 3.10 or higher is required. Found Python ${PY_VERSION}.${RESET}"
    exit 1
fi

echo -e "${GREEN}[✓] Detected Python ${PY_VERSION}${RESET}"

# 2. Setup isolated Herd directory and virtualenv (~/.herd/venv)
HERD_DIR="${HERD_HOME:-$HOME/.herd}"
VENV_DIR="${HERD_DIR}/venv"
BIN_DIR="$HOME/.local/bin"

echo -e "${CYAN}[*] Setting up isolated Herd environment in ${VENV_DIR}...${RESET}"
mkdir -p "${HERD_DIR}"
mkdir -p "${BIN_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi

# 3. Upgrade pip & install herd-cli into venv
echo -e "${CYAN}[*] Installing Herd CLI...${RESET}"
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install --upgrade "git+https://github.com/elvin-mark/herd.git" --quiet

# 4. Create symlink in ~/.local/bin
echo -e "${CYAN}[*] Creating symlink in ${BIN_DIR}/herd...${RESET}"
ln -sf "${VENV_DIR}/bin/herd" "${BIN_DIR}/herd"

# 5. Check PATH environment variable
PATH_UPDATED=0
case ":$PATH:" in
    *":${BIN_DIR}:"*) ;;
    *)
        PATH_UPDATED=1
        SHELL_CONFIG=""
        if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_CONFIG="$HOME/.bashrc"
        fi

        if [ -n "$SHELL_CONFIG" ]; then
            if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_CONFIG"; then
                echo '' >> "$SHELL_CONFIG"
                echo '# Added by Herd CLI installer' >> "$SHELL_CONFIG"
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
                echo -e "${YELLOW}[!] Added ${BIN_DIR} to PATH in ${SHELL_CONFIG}${RESET}"
            fi
        fi
        ;;
esac

echo -e "\n${GREEN}${BOLD}✨ Herd successfully installed!${RESET}\n"

if [ "$PATH_UPDATED" -eq 1 ]; then
    echo -e "${YELLOW}To start using herd immediately, run:${RESET}"
    echo -e "  ${BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}\n"
fi

echo -e "Try running:"
echo -e "  ${CYAN}herd --help${RESET}"
echo -e "  ${CYAN}herd run smollm2${RESET}"
echo -e "  ${CYAN}herd serve${RESET}"
