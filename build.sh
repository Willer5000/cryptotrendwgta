#!/bin/bash
echo "Instalando dependencias..."
pip install -r requirements.txt

echo "Iniciando base de datos..."
python -c "import sqlite3; \
conn = sqlite3.connect('historical_signals.db'); \
c = conn.cursor(); \
c.execute('''CREATE TABLE IF NOT EXISTS signals
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              symbol TEXT,
              signal_type TEXT,
              entry REAL,
              sl REAL,
              tp1 REAL,
              tp2 REAL,
              volume TEXT,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)'''); \
conn.commit(); \
conn.close()"

echo "Build completado"
