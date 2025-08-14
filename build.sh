#!/bin/bash
echo "Instalando dependencias..."
pip install -r requirements.txt

# Instalar TA-Lib
echo "Instalando TA-Lib..."
sudo apt-get update
sudo apt-get install -y build-essential libta-lib-dev
pip install TA-Lib
