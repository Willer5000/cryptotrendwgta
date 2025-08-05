#!/bin/bash
set -e

# Instalar dependencias del sistema
sudo apt-get update
sudo apt-get install -y build-essential

# Descargar y compilar TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# Instalar dependencias de Python
pip install -r requirements.txt
