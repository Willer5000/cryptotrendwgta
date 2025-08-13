#!/bin/bash
echo ">>> Actualizando pip y setuptools"
python -m pip install --upgrade pip setuptools wheel

echo ">>> Instalando dependencias"
pip install -r requirements.txt --no-cache-dir

echo ">>> Instalación completada"
