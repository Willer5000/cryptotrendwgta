import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime
from flask import Flask, render_template, request
from flask_caching import Cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback

app = Flask(__name__)
app.logger.setLevel('INFO')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 900  # 15 minutos
DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 50,
    'divergence_lookback': 20,
    'max_risk_percent': 1.5,
    'price_distance_threshold': 1.0
}

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
        app.logger.info(f"Cargadas {len(cryptos)} criptomonedas")
        return cryptos
    except Exception as e:
        app.logger.error(f"Error cargando criptos: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'TRX']

# Obtener datos de KuCoin
def get_kucoin_data(symbol, timeframe):
    tf_mapping = {
        '30m': '30min',
        '1h': '1hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': '1day',
        '1w': '1week'
    }
    kucoin_tf = tf_mapping.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                candles.reverse()
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir a numérico
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                if len(df) > 0:
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    return df
        return None
    except Exception as e:
        app.logger.error(f"Error API {symbol}: {str(e)}")
        return None

# Implementación manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    if len(high) < window:
        return pd.Series([0] * len(high)), pd.Series([0] * len(high)), pd.Series([0] * len(high))
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    minus_dm = -minus_dm
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    try:
        df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
        df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
        
        resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
        supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
        
        return supports, resistances
    except:
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if not avg_vol or avg_vol == 0:
        return 'Muy Bajo'
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Calcular distancia al nivel más cercano
def calculate_distance_to_level(price, levels, threshold_percent=1.0):
    if not levels:
        return False
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    return min_distance <= threshold

# Analizar una criptomoneda y calcular probabilidades
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return None, None, 0, 0, 'Muy Bajo'
        
        # Calcular EMAs
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # Calcular RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # Calcular ADX
        df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # Encontrar soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG
        if trend == 'up': 
            long_prob += 30
        if last['adx'] > params['adx_level']: 
            long_prob += 20
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): 
            long_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: 
            long_prob += 15
        if last['rsi'] < 40:  # RSI no sobrecomprado
            long_prob += 10
        
        # Criterios para SHORT
        if trend == 'down': 
            short_prob += 30
        if last['adx'] > params['adx_level']: 
            short_prob += 20
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): 
            short_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: 
            short_prob += 15
        if last['rsi'] > 60:  # RSI sobrecomprado
            short_prob += 10
        
        # Limitar probabilidades a 100%
        long_prob = min(long_prob, 100)
        short_prob = min(short_prob, 100)
        
        # Señales LONG (solo si hay volumen suficiente)
        long_signal = None
        if long_prob >= 65 and volume_class in ['Alto', 'Muy Alto']:
            long_signal = {
                'symbol': symbol,
                'long_prob': long_prob,
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 4),
                'volume': volume_class
            }
        
        # Señales SHORT (solo si hay volumen suficiente)
        short_signal = None
        if short_prob >= 65 and volume_class in ['Alto', 'Muy Alto']:
            short_signal = {
                'symbol': symbol,
                'short_prob': short_prob,
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 4),
                'volume': volume_class
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        app.logger.error(f"Error analizando {symbol}: {str(e)}")
        traceback.print_exc()
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        with app.app_context():
            try:
                app.logger.info("Iniciando actualización de datos...")
                cryptos = load_cryptos()
                long_signals = []
                short_signals = []
                scatter_data = []
                
                for crypto in cryptos:
                    long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                    
                    scatter_data.append({
                        'symbol': crypto,
                        'long_prob': long_prob,
                        'short_prob': short_prob,
                        'volume': volume_class
                    })
                    
                    if long_signal:
                        long_signals.append(long_signal)
                    if short_signal:
                        short_signals.append(short_signal)
                
                # Ordenar por fuerza de tendencia (ADX)
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar caché
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                
                app.logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT")
            except Exception as e:
                app.logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()

@app.route('/')
def index():
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Estadísticas para gráficos
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=json.dumps(scatter_data))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
