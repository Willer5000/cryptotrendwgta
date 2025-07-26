from apscheduler.schedulers.blocking import BlockingScheduler
from data_fetcher import update_all_data
from analyzer import analyze_all
import os
import sqlite3

# Inicializar base de datos de señales
def init_signals_db():
    conn = sqlite3.connect('signals.db')
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

# Ejecutar análisis inmediatamente al iniciar
print('Ejecutando primera actualización...')
init_signals_db()
update_all_data()
analyze_all()

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=15)
def scheduled_job():
    print('Iniciando actualización periódica...')
    update_all_data()
    analyze_all()
    print('Análisis completado!')

if __name__ == '__main__':
    sched.start()
