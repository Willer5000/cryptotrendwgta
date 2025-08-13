#!/bin/bash
echo ">>> Actualizando pip"
python -m pip install --upgrade pip

echo ">>> Instalando dependencias"
pip install -r requirements.txt --no-cache-dir

echo ">>> Instalación completada"
