#!/bin/bash
echo ">>> Actualizando pip y setuptools"
python -m pip install --upgrade pip setuptools

echo ">>> Instalando dependencias"
pip install -r requirements.txt

echo ">>> Instalación completada"
