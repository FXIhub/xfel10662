#!/bin/bash


# print commands and run them
set -o xtrace

# Guard against an empty destination: without REPO_ON_MAXWELL the rsync target
# collapses to max-exfl-display:/ (the filesystem root). Source
# source_at_maxwell.sh first.
: "${REPO_ON_MAXWELL:?source source_at_maxwell.sh first}"

rsync -vrtlpzh --progress --exclude=__pycache__ --exclude='*.pyc' \
    $(pwd)/ \
    max-exfl-display:$REPO_ON_MAXWELL/
