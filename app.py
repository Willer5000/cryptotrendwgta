from flask import Flask, render_template
import sqlite3
import os
import json
from datetime import datetime

app = Flask(__name__)
DATABASE = 'signals.db'

# Inicializar la base de datos al inicio
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        timeframe TEXT,
        signal_type TEXT,
        entry REAL,
        sl REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        timestamp DATETIME
    )
    ''')
    conn.commit()
    conn.close()

# Llamar a la inicialización al arrancar
init_db()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    
    # Obtener señales LONG
    long_signals = conn.execute(
        "SELECT * FROM signals WHERE signal_type = 'LONG' ORDER BY timestamp DESC"
    ).fetchall()
    
    # Obtener señales SHORT
    short_signals = conn.execute(
        "SELECT * FROM signals WHERE signal_type = 'SHORT' ORDER BY timestamp DESC"
    ).fetchall()
    
    conn.close()
    
    # Obtener última actualización
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template(
        'index.html',
        long_signals=long_signals,
        short_signals=short_signals,
        update_time=update_time
    )

if __name__ == '__main__':
    app.run(debug=True)
