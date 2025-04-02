#!/bin/bash

# Set _dir to directory of the script being run
_here_dir=$(pwd)
_dir="$(cd "$(dirname "$0")" && pwd)"
_topdir="$(cd "${_dir}/.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NO_COLOR='\033[0m'

# Official trusted install script
UV_INSTALL_URL=https://astral.sh/uv/install.sh

VENV=".venv"

# used for step numbering
step=1

yellow() {
    echo -ne "${YELLOW}${1}${NO_COLOR}"
}

green() {
    echo -ne "${GREEN}${1}${NO_COLOR}"
}

red() {
    echo -ne "${RED}${1}${NO_COLOR}"
}

write_step() {
    yellow "[$step] "
    echo -ne "${1}\n"
    step=$((step + 1))
}

red_and_exit() {
    red "${1}\n"
    exit 1
}

opt_force=0
while getopts 'Fh' OPTION; do
    case "$OPTION" in
    F)
        opt_force=1
        ;;
    h | ?)
        printf "Usage: %s [-F]\n" "$(basename "$0")" >&2
        printf "Options:\n" >&2
        printf "  -F: Clear the environment folder and replace with new environment.\n" >&2
        printf "  -h: Show this help message.\n" >&2
        printf "The %s directory is created where the script is run.\n" "${VENV}" >&2
        exit 1
        ;;
    esac
done

if [ "$opt_force" -eq 1 ] && [ -d "./${VENV}" ]; then
    write_step "Deleting the extant \"${VENV}\" folder to clear way for fresh environment..."
    rm -rf -- "${VENV}" || red_and_exit "Failed to delete ${VENV}"
    green "Deleted.\n"
fi

if [ -d "./${VENV}" ]; then
    yellow "${VENV} folder already exists! (ignoring and trying to sync...)\n"
    if [ ! -f "./${VENV}/bin/activate" ]; then
        red_and_exit "The ${VENV} folder exists but no virtual environment was found. Exiting...\n"
    fi
    echo -ne "Pass option -F to clear and replace the ${VENV} folder.\n"
fi

# check if uv is installed
which uv >/dev/null 2>&1 || {
    write_step "Installing uv in order to use it to install everything else..."
    curl -LsSf "${UV_INSTALL_URL}" | sh || red_and_exit "Failed to install uv via curl\n"
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        red_and_exit "No uv binary found in ${HOME}/.local/bin; uv install failed!"
    fi
}

# At this point, we should have uv installed
write_step "Updating or creating .venv/ and packages..."
uv sync --group dev || red_and_exit "Failed to sync the environment! Exiting...\n"

printf "\n\nSetup script complete!\n\n"
printf "You can now run \n"
printf "source %s/bin/activate\n" "${VENV}"
printf "to activate the virtual environment.\n\n"
exit 0
