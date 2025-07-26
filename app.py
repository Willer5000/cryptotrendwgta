from flask import Flask, render_template
import sqlite3
import os
import json
from datetime import datetime

app = Flask(__name__)
DATABASE = 'signals.db'

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
