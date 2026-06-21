#!/usr/bin/env bash
# Source this to enter the project environment:  . activate.sh
# The venv lives OFF the SSD because the SSD is exFAT (no symlink support, which
# `python -m venv` requires). Code stays on the SSD; only the venv is elsewhere.
export EDH_VENV="${EDH_VENV:-/home/neuraldyn/.venvs/edh}"
# shellcheck disable=SC1091
. "$EDH_VENV/bin/activate"
export PIP_USER=0          # a global PIP_USER=1 otherwise forces --user installs
unset PYTHONUSERBASE
echo "edh env active: $(python --version), venv=$EDH_VENV"
