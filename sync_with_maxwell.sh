#!/bin/bash


# print commands and run them
set -o xtrace

rsync -vrtlpzh --progress --exclude=__pycache__ --exclude='*.pyc' \
    $(pwd)/ \
    max-exfl-display:$REPO_ON_MAXWELL/
