import os
import time
import threading
import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Configuración inicial
CRYPTO_FILE = 'cryptos.txt'
KUCOIN_API_URL = "https://api.kucoin.com/api/v1/market/candles"
TIME_FRAMES = ['30min', '1h', '2h', '4h', '1d', '1w']
TIME_FRAME_MAP = {
    '30min': 1800,
    '1h': 3600,
    '2h': 7200,
    '4h': 14400,
    '1d': 86400,
    '1w': 604800
}
VOLUME_CLASSES = {
    'Muy Alto': 3.0,
    'Alto': 2.0,
    'Medio': 1.0,
    'Bajo': 0.5,
    'Muy Bajo': 0.2
}

# Variables globales para almacenar datos
crypto_data = {'long': [], 'short': []}
last_update = "Nunca"
crypto_list = []

# Cargar lista de criptomonedas
def load_cryptos():
    global crypto_list
    with open(CRYPTO_FILE, 'r') as f:
        crypto_list = [line.strip().upper() + '-USDT' for line in f.readlines() if line.strip()]

# Obtener datos históricos de Kucoin
def get_historical_data(symbol, timeframe, limit=200):
    try:
        response = requests.get(
            KUCOIN_API_URL,
            params={
                'symbol': symbol,
                'type': timeframe,
                'startAt': int((datetime.now() - timedelta(days=60)).timestamp()),
                'endAt': int(datetime.now().timestamp())
            }
        )
        data = response.json()
        if data['code'] == '200000' and data['data']:
            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.iloc[::-1].reset_index(drop=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df[['open', 'close', 'high', 'low', 'volume']] = df[['open', 'close', 'high', 'low', 'volume']].astype(float)
            return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
    return pd.DataFrame()

# Calcular indicadores técnicos
def calculate_indicators(df, params):
    if df.empty or len(df) < 50:
        return None
    
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=params['ema_fast'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['ema_slow'], adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=params['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ADX
    df['tr'] = df['high'].combine(df['low'].shift(), lambda x1, x2: max(x1 - df['low'].shift(), abs(x1 - df['close'].shift()), abs(df['low'].shift() - df['close'].shift()))
    df['atr'] = df['tr'].rolling(window=params['adx_period']).mean()
    
    # Soportes y resistencias
    df['resistance'] = df['high'].rolling(window=params['sr_period']).max()
    df['support'] = df['low'].rolling(window=params['sr_period']).min()
    
    # Volumen
    avg_volume = df['volume'].rolling(window=params['volume_period']).mean()
    df['volume_ratio'] = df['volume'] / avg_volume
    
    return df

# Detectar divergencias
def detect_divergence(df, lookback=5):
    if df.empty or len(df) < lookback + 10:
        return False, False
    
    # Buscar divergencias alcistas (bullish)
    if (df['rsi'].iloc[-1] > 30 and 
        df['close'].iloc[-1] < df['close'].iloc[-lookback] and 
        df['rsi'].iloc[-1] > df['rsi'].iloc[-lookback]):
        return True, False
    
    # Buscar divergencias bajistas (bearish)
    if (df['rsi'].iloc[-1] < 70 and 
        df['close'].iloc[-1] > df['close'].iloc[-lookback] and 
        df['rsi'].iloc[-1] < df['rsi'].iloc[-lookback]):
        return False, True
    
    return False, False

# Clasificar volumen
def classify_volume(volume_ratio):
    for cls, threshold in VOLUME_CLASSES.items():
        if volume_ratio >= threshold:
            return cls
    return "Muy Bajo"

# Calcular niveles de entrada, SL y TP
def calculate_levels(df, signal_type, params):
    if df.empty:
        return None
    
    last_close = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    
    if signal_type == 'long':
        entry = df['resistance'].iloc[-1] * 0.99
        sl = entry - (atr * params['sl_atr_multiplier'])
        tp1 = entry + (atr * params['tp1_atr_multiplier'])
        tp2 = entry + (atr * params['tp2_atr_multiplier'])
        tp3 = entry + (atr * params['tp3_atr_multiplier'])
    else:  # short
        entry = df['support'].iloc[-1] * 1.01
        sl = entry + (atr * params['sl_atr_multiplier'])
        tp1 = entry - (atr * params['tp1_atr_multiplier'])
        tp2 = entry - (atr * params['tp2_atr_multiplier'])
        tp3 = entry - (atr * params['tp3_atr_multiplier'])
    
    return {
        'entry': round(entry, 4),
        'sl': round(sl, 4),
        'tp1': round(tp1, 4),
        'tp2': round(tp2, 4),
        'tp3': round(tp3, 4)
    }

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_historical_data(symbol, params['timeframe'])
        if df.empty:
            return None
        
        df = calculate_indicators(df, params)
        if df is None:
            return None
        
        last_row = df.iloc[-1]
        volume_class = classify_volume(last_row['volume_ratio'])
        bullish_div, bearish_div = detect_divergence(df, params['divergence_period'])
        
        result = {
            'symbol': symbol.replace('-USDT', ''),
            'price': last_row['close'],
            'volume_class': volume_class,
            'bullish_div': bullish_div,
            'bearish_div': bearish_div,
            'ema_fast': last_row['ema_fast'],
            'ema_slow': last_row['ema_slow'],
            'rsi': last_row['rsi'],
            'adx': last_row.get('atr', 0),  # Usamos ATR como proxy para ADX
            'support': last_row['support'],
            'resistance': last_row['resistance'],
            'long_signal': False,
            'short_signal': False,
            'levels': None
        }
        
        # Condiciones para LONG
        if (bullish_div and
            last_row['close'] > last_row['ema_fast'] > last_row['ema_slow'] and
            last_row['close'] < last_row['resistance'] * 1.05 and
            volume_class in ['Alto', 'Muy Alto'] and
            last_row['rsi'] < 70):
            result['long_signal'] = True
            result['levels'] = calculate_levels(df, 'long', params)
        
        # Condiciones para SHORT
        if (bearish_div and
            last_row['close'] < last_row['ema_fast'] < last_row['ema_slow'] and
            last_row['close'] > last_row['support'] * 0.95 and
            volume_class in ['Alto', 'Muy Alto'] and
            last_row['rsi'] > 30):
            result['short_signal'] = True
            result['levels'] = calculate_levels(df, 'short', params)
        
        return result
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

# Tarea programada para actualizar datos
def update_crypto_data():
    global crypto_data, last_update
    
    while True:
        try:
            params = {
                'timeframe': '4h',
                'sr_period': 20,
                'adx_period': 14,
                'adx_level': 25,
                'ema_fast': 9,
                'ema_slow': 20,
                'rsi_period': 14,
                'divergence_period': 5,
                'volume_period': 20,
                'sl_atr_multiplier': 1.5,
                'tp1_atr_multiplier': 1.0,
                'tp2_atr_multiplier': 2.0,
                'tp3_atr_multiplier': 3.0
            }
            
            long_signals = []
            short_signals = []
            
            for symbol in crypto_list:
                result = analyze_crypto(symbol, params)
                if result:
                    if result['long_signal'] and result['levels']:
                        long_signals.append({
                            'symbol': result['symbol'],
                            'entry': result['levels']['entry'],
                            'sl': result['levels']['sl'],
                            'tp1': result['levels']['tp1'],
                            'tp2': result['levels']['tp2'],
                            'tp3': result['levels']['tp3']
                        })
                    if result['short_signal'] and result['levels']:
                        short_signals.append({
                            'symbol': result['symbol'],
                            'entry': result['levels']['entry'],
                            'sl': result['levels']['sl'],
                            'tp1': result['levels']['tp1'],
                            'tp2': result['levels']['tp2'],
                            'tp3': result['levels']['tp3']
                        })
            
            crypto_data = {
                'long': sorted(long_signals, key=lambda x: x['symbol']),
                'short': sorted(short_signals, key=lambda x: x['symbol'])
            }
            
            last_update = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"Data updated at {last_update}")
            
        except Exception as e:
            print(f"Update error: {str(e)}")
        
        time.sleep(900)  # Esperar 15 minutos

# Ruta principal
@app.route('/')
def index():
    return render_template(
        'index.html',
        long_signals=crypto_data['long'],
        short_signals=crypto_data['short'],
        last_update=last_update
    )

# Iniciar la aplicación
if __name__ == '__main__':
    load_cryptos()
    
    # Iniciar hilo de actualización de datos
    update_thread = threading.Thread(target=update_crypto_data)
    update_thread.daemon = True
    update_thread.start()
    
    # Ejecutar Flask
    app.run(host='0.0.0.0', port=5000, debug=False)
