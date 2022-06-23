#!/usr/bin/env bash
mkdir .venv
python3 -m venv .venv/
source .venv/bin/activate && pip3 install -r requirements-dev.txt
