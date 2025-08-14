#!/bin/bash
echo ">>> Updating pip and setuptools"
python -m pip install --upgrade pip setuptools wheel

echo ">>> Installing dependencies"
pip install -r requirements.txt --no-cache-dir

echo ">>> Installation completed"
