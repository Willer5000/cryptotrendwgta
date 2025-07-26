import os
import time
import threading
from flask import Flask, render_template
from utils.kucoin_api import fetch_crypto_data
from utils.technical_analysis import analyze_crypto
from utils.telegram_alerts import send_telegram_alert

app = Flask(__name__)

# Configuración por defecto (basada en experiencia profesional)
DEFAULT_SETTINGS = {
    "trend_period": 20,
    "sr_period": 14,
    "adx_period": 14,
    "adx_level": 25,
    "ema_fast": 9,
    "ema_slow": 20,
    "divergence_period": 14,
    "rsi_period": 14,
    "volume_thresholds": {
        "muy alto": 2.5,
        "alto": 1.8,
        "medio": 1.2,
        "bajo": 0.8,
        "muy bajo": 0.5
    }
}

# Leer lista de criptomonedas
with open('cryptos.txt', 'r') as f:
    CRYPTOS = [line.strip() for line in f]

# Almacenamiento de señales
long_signals = []
short_signals = []

def update_signals():
    global long_signals, short_signals
    while True:
        new_long = []
        new_short = []
        
        for crypto in CRYPTOS:
            try:
                data = fetch_crypto_data(crypto)
                analysis = analyze_crypto(data, DEFAULT_SETTINGS)
                
                if analysis['long_signal']:
                    new_long.append({
                        "crypto": crypto,
                        "entry": analysis['entry_long'],
                        "sl": analysis['sl_long'],
                        "tp1": analysis['tp1_long'],
                        "tp2": analysis['tp2_long'],
                        "tp3": analysis['tp3_long']
                    })
                    # Enviar alerta si hay nueva señal
                    send_telegram_alert(analysis, 'LONG')
                
                if analysis['short_signal']:
                    new_short.append({
                        "crypto": crypto,
                        "entry": analysis['entry_short'],
                        "sl": analysis['sl_short'],
                        "tp1": analysis['tp1_short'],
                        "tp2": analysis['tp2_short'],
                        "tp3": analysis['tp3_short']
                    })
                    # Enviar alerta si hay nueva señal
                    send_telegram_alert(analysis, 'SHORT')
            
            except Exception as e:
                print(f"Error analyzing {crypto}: {str(e)}")
        
        long_signals = new_long
        short_signals = new_short
        time.sleep(900)  # Actualizar cada 15 minutos

@app.route('/')
def index():
    return render_template('index.html', 
                           long_signals=long_signals,
                           short_signals=short_signals)

if __name__ == '__main__':
    # Iniciar hilo para actualización de señales
    thread = threading.Thread(target=update_signals)
    thread.daemon = True
    thread.start()
    app.run(port=5000)
