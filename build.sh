#!/bin/bash
echo ">>> Actualizando pip y setuptools"
pip install --upgrade pip setuptools wheel

echo ">>> Instalando dependencias"
pip install -r requirements.txt --no-cache-dir

echo ">>> Permisos de ejecución"
chmod +x app.py

echo ">>> Instalación completada"
