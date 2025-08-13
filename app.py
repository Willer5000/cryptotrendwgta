import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import math
import logging
from functools import lru_cache

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300  # 5 minutos
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

# Almacenamiento en memoria
crypto_data = {
    'long_signals': [],
    'short_signals': [],
    'scatter_data': [],
    'last_update': datetime.now(),
    'update_in_progress': False
}
data_lock = threading.Lock()

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        app.logger.error(f"Error loading cryptos: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP']  # Fallback básico

# Obtener datos de KuCoin con reintentos
def get_kucoin_data(symbol, timeframe, retries=3):
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
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    candles.reverse()  # Invertir para orden cronológico
                    
                    if len(candles) < 50:
                        app.logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    # Convertir a tipos numéricos
                    numeric_cols = ['open', 'close', 'high', 'low', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Eliminar filas con valores NaN
                    df = df.dropna()
                    
                    if len(df) < 50:
                        app.logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                        return None
                    
                    # Convertir timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    return df
            else:
                app.logger.warning(f"Intento {attempt+1} para {symbol} falló: HTTP {response.status_code}")
        except Exception as e:
            app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        time.sleep(1)  # Esperar antes de reintentar
    
    app.logger.error(f"No se pudieron obtener datos para {symbol} después de {retries} intentos")
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window):
    if len(series) < window + 1:
        return pd.Series([np.nan] * len(series))
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Evitar división por cero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close))
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm <= 0] = 0
    minus_dm[minus_dm <= 0] = 0
    
    # Calcular True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window).mean() / (atr + 1e-10))
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # Eliminar filas con NaN resultantes de los cálculos
        df = df.dropna()
        
        return df
    except Exception as e:
        app.logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias simplificado
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Usar solo los últimos 100 datos para mejor rendimiento
        recent = df.tail(100)
        
        # Identificar máximos y mínimos locales
        highs = recent['high'].values
        lows = recent['low'].values
        
        supports = []
        resistances = []
        
        # Buscar niveles clave
        for i in range(1, len(highs)-1):
            # Resistencia: punto más alto que sus vecinos
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                resistances.append(highs[i])
            # Soporte: punto más bajo que sus vecinos
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                supports.append(lows[i])
        
        # Eliminar duplicados y ordenar
        supports = sorted(set(supports))
        resistances = sorted(set(resistances))
        
        return supports, resistances
    except Exception as e:
        app.logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol == 0 or current_vol is None or avg_vol is None:
            return 'muy-bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.0: return 'muy-alto'
        if ratio > 1.5: return 'alto'
        if ratio > 1.0: return 'medio'
        if ratio > 0.5: return 'bajo'
        return 'muy-bajo'
    except:
        return 'muy-bajo'

# Analizar una criptomoneda y calcular probabilidades
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        return None, None, 0, 0, 'muy-bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'muy-bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades básicas
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if trend == 'up': long_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: long_prob += 20
        if volume_class in ['alto', 'muy-alto']: long_prob += 15
        if supports and last['close'] > min(supports): long_prob += 10
        
        # Factores para SHORT
        if trend == 'down': short_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: short_prob += 20
        if volume_class in ['alto', 'muy-alto']: short_prob += 15
        if resistances and last['close'] < max(resistances): short_prob += 10
        
        # Asegurar que las probabilidades estén en rango
        long_prob = max(0, min(100, long_prob))
        short_prob = max(0, min(100, short_prob))
        
        # Señales LONG
        long_signal = None
        if long_prob >= 70 and volume_class in ['alto', 'muy-alto']:
            # Cálculo simplificado de niveles
            entry = last['close'] * 1.005
            sl = last['close'] * (1 - params['max_risk_percent']/100)
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 70 and volume_class in ['alto', 'muy-alto']:
            # Cálculo simplificado de niveles
            entry = last['close'] * 0.995
            sl = last['close'] * (1 + params['max_risk_percent']/100)
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'muy-bajo'

# Tarea para actualizar datos
def update_crypto_data(params):
    global crypto_data
    
    start_time = time.time()
    app.logger.info("Iniciando actualización de datos...")
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    scatter_data = []
    
    # Procesar en lotes para reducir memoria
    batch_size = 20
    for i in range(0, len(cryptos), batch_size):
        batch = cryptos[i:i+batch_size]
        for crypto in batch:
            long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, params)
            if long_signal:
                long_signals.append(long_signal)
            if short_signal:
                short_signals.append(short_signal)
            
            # Datos para el gráfico de dispersión
            scatter_data.append({
                'symbol': crypto,
                'long_prob': long_prob,
                'short_prob': short_prob,
                'volume': volume_class
            })
        
        # Pequeña pausa entre lotes
        time.sleep(1)
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    # Actualizar datos globales
    with data_lock:
        crypto_data = {
            'long_signals': long_signals,
            'short_signals': short_signals,
            'scatter_data': scatter_data,
            'last_update': datetime.now(),
            'update_in_progress': False
        }
    
    elapsed = time.time() - start_time
    app.logger.info(f"Actualización completada en {elapsed:.2f}s: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} puntos")
    return True

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        try:
            # Actualizar con los parámetros por defecto
            update_crypto_data(DEFAULTS)
        except Exception as e:
            app.logger.error(f"Error en actualización de fondo: {str(e)}")
        
        # Esperar hasta la próxima actualización
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
try:
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    app.logger.info("Hilo de actualización iniciado")
except Exception as e:
    app.logger.error(f"No se pudo iniciar el hilo de actualización: {str(e)}")

@app.route('/')
def index():
    with data_lock:
        data = crypto_data.copy()
    
    # Estadísticas para gráficos
    signal_count = len(data['long_signals']) + len(data['short_signals'])
    avg_adx_long = np.mean([s['adx'] for s in data['long_signals']]) if data['long_signals'] else 0
    avg_adx_short = np.mean([s['adx'] for s in data['short_signals']]) if data['short_signals'] else 0
    
    # Preparar datos para el gráfico de dispersión
    scatter_data = []
    for item in data['scatter_data']:
        # Asegurar que los datos sean serializables
        scatter_data.append({
            'symbol': item['symbol'],
            'long_prob': int(item['long_prob']),
            'short_prob': int(item['short_prob']),
            'volume': item['volume']
        })
    
    return render_template('index.html', 
                           long_signals=data['long_signals'][:50], 
                           short_signals=data['short_signals'][:50],
                           last_update=data['last_update'],
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=json.dumps(scatter_data))

@app.route('/update', methods=['POST'])
def manual_update():
    if not crypto_data['update_in_progress']:
        with data_lock:
            crypto_data['update_in_progress'] = True
        
        # Ejecutar actualización en un hilo separado
        def update_task():
            try:
                update_crypto_data(DEFAULTS)
            except Exception as e:
                app.logger.error(f"Error en actualización manual: {str(e)}")
        
        threading.Thread(target=update_task, daemon=True).start()
        return jsonify({'status': 'update_started'}), 202
    else:
        return jsonify({'status': 'update_already_in_progress'}), 200

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    # Esta función se simplifica para el ejemplo
    return render_template('chart.html', symbol=symbol, signal_type=signal_type)

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
