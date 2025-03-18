#!/bin/bash

# Set _dir to directory of the script being run
_here_dir=$(pwd)
_dir="$(cd "$(dirname "$0")" && pwd)"
_topdir="$(cd "${_dir}/.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NO_COLOR='\033[0m'

BOOTSTRAP="${_topdir}/requirements-bootstrap.txt" # only "uv"
REQ_MAIN="${_topdir}/requirements.txt"            # main requirements

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

yellow_and_exit() {
    yellow "${1}\n"
    exit 1
}

red_and_exit() {
    red "${1}\n"
    exit 1
}

pip_install_requirement() {
    # args: $1 - Message to print
    #       $2 - $n - the rest of the args to "uv pip install"
    #
    msg="$1"
    shift
    write_step "${msg}\nRunning uv pip install ${*}...\n"
    uv pip install "$@" || red_and_exit "Failed to install ${msg} requirements! Exiting...\n"
    green "Done.\n"
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

# create virtual environment if it doesn't exist
if [ ! -d "./${VENV}" ]; then
    write_step "Creating virtual environment in the ${PWD}/${VENV} directory..."
    uv venv "${VENV}" || red_and_exit "Failed to create virtual environment! Exiting...\n"
    green "Done.\n"
fi

if [ ! -r "$BOOTSTRAP" ]; then
    yellow_and_exit "No bootstrap requirements ($BOOTSTRAP) file found! Exiting...\n"
fi
if [ ! -r "$REQ_MAIN" ]; then
    yellow_and_exit "No main requirements ($REQ_MAIN) file found! Exiting...\n"
fi

pip_install_requirement "Bootstrap" -r "${BOOTSTRAP}"
# shellcheck source=/dev/null
. "${VENV}/bin/activate"
pip_install_requirement "Main Install" --exact -r "${BOOTSTRAP}" -r "${REQ_MAIN}"

printf "\n\nSetup script complete!\n\n"
printf "You can now run \n"
printf "source %s/bin/activate\n" "${VENV}"
printf "to activate the newly created virtual environment.\n\n"
exit 0
