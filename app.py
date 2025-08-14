import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import math
import logging

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuración mejorada
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Cargadas {len(cryptos)} criptomonedas")
            return cryptos
    except Exception as e:
        logger.error(f"Error al cargar criptomonedas: {str(e)}")
        return ['BTC', 'ETH', 'BNB']  # Fallback básico

# Mapeo de timeframes
TIMEFRAME_MAP = {
    '30m': '30min',
    '1h': '1hour',
    '2h': '2hour',
    '4h': '4hour',
    '1d': '1day',
    '1w': '1week'
}

# Obtener datos de KuCoin (versión mejorada)
def get_kucoin_data(symbol, timeframe):
    kucoin_tf = TIMEFRAME_MAP.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    try:
        logger.info(f"Obteniendo datos para {symbol} en {timeframe}")
        response = requests.get(url, timeout=25)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                candles.reverse()  # Invertir para orden cronológico
                
                if len(candles) < 100:
                    logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir a numérico con manejo de errores
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                if len(df) < 50:
                    logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                    return None
                
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
        else:
            logger.error(f"Error en API KuCoin para {symbol}: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementación manual de EMA (mejorada)
def calculate_ema(series, window):
    try:
        if len(series) < window:
            return pd.Series([np.nan] * len(series))
        return series.ewm(span=window, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error en EMA: {str(e)}")
        return pd.Series([np.nan] * len(series))

# Implementación manual de RSI (mejorada)
def calculate_rsi(series, window):
    try:
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
    except Exception as e:
        logger.error(f"Error en RSI: {str(e)}")
        return pd.Series([50] * len(series))

# Implementación manual de ADX (mejorada)
def calculate_adx(high, low, close, window):
    try:
        if len(close) < window * 2:
            return pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close))
        
        up = high.diff()
        down = -low.diff()
        
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = np.maximum.reduce([tr1, tr2, tr3])
        
        atr = tr.rolling(window).mean()
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window).mean()
        
        return adx, plus_di, minus_di
    except Exception as e:
        logger.error(f"Error en ADX: {str(e)}")
        return pd.Series([0]*len(high)), pd.Series([0]*len(high)), pd.Series([0]*len(high))

# Calcular indicadores manualmente (versión robusta)
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'].astype(float), 
            df['low'].astype(float), 
            df['close'].astype(float), 
            params['adx_period']
        )
        
        # Eliminar filas con NaN
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias (versión mejorada)
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Usar los últimos 200 puntos para S/R
        df_sub = df.iloc[-200:] if len(df) > 200 else df
        
        # Encontrar máximos y mínimos locales
        highs = df_sub['high'].values
        lows = df_sub['low'].values
        supports = []
        resistances = []
        
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                resistances.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                supports.append(lows[i])
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen (versión mejorada)
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol == 0 or current_vol is None or avg_vol is None:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.5: return 'Muy Alto'
        if ratio > 1.8: return 'Alto'
        if ratio > 1.2: return 'Medio'
        if ratio > 0.7: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Analizar una criptomoneda (versión robusta)
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(50).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades (versión mejorada)
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG
        if trend == 'up': 
            long_prob += 30
        if last['adx'] > params['adx_level']: 
            long_prob += 20
        if last['rsi'] < 40: 
            long_prob += 15
        if volume_class in ['Alto', 'Muy Alto']: 
            long_prob += 15
        if supports and last['close'] <= min(supports) * 1.03: 
            long_prob += 20
        
        # Criterios para SHORT
        if trend == 'down': 
            short_prob += 30
        if last['adx'] > params['adx_level']: 
            short_prob += 20
        if last['rsi'] > 60: 
            short_prob += 15
        if volume_class in ['Alto', 'Muy Alto']: 
            short_prob += 15
        if resistances and last['close'] >= max(resistances) * 0.97: 
            short_prob += 20
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = int((long_prob / total) * 100)
            short_prob = int((short_prob / total) * 100)
        else:
            long_prob = 50
            short_prob = 50
        
        # Señales LONG
        long_signal = None
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            # Lógica de entrada y stop loss
            entry = last['close'] * 1.005
            stop_loss = last['close'] * 0.985
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 6),
                'sl': round(stop_loss, 6),
                'tp1': round(entry * 1.015, 6),
                'tp2': round(entry * 1.03, 6),
                'volume': volume_class,
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 6),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            # Lógica de entrada y stop loss
            entry = last['close'] * 0.995
            stop_loss = last['close'] * 1.015
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 6),
                'sl': round(stop_loss, 6),
                'tp1': round(entry * 0.985, 6),
                'tp2': round(entry * 0.97, 6),
                'volume': volume_class,
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 6),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, 50, 50, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        with app.app_context():
            try:
                logger.info("Iniciando actualización de datos...")
                cryptos = load_cryptos()
                long_signals = []
                short_signals = []
                scatter_data = []
                
                for crypto in cryptos:
                    try:
                        long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                        if long_signal:
                            long_signals.append(long_signal)
                        if short_signal:
                            short_signals.append(short_signal)
                        
                        scatter_data.append({
                            'symbol': crypto,
                            'long_prob': long_prob,
                            'short_prob': short_prob,
                            'volume': volume_class
                        })
                    except Exception as e:
                        logger.error(f"Error procesando {crypto}: {str(e)}")
                
                # Ordenar por fuerza de tendencia (ADX)
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar caché
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                
                logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT")
            except Exception as e:
                logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
try:
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    logger.info("Hilo de actualización iniciado")
except Exception as e:
    logger.error(f"No se pudo iniciar el hilo de actualización: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Manejar actualización de parámetros
    if request.method == 'POST':
        new_params = DEFAULTS.copy()
        for key in DEFAULTS:
            if key in request.form:
                try:
                    # Convertir a número si es necesario
                    if key in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                        new_params[key] = int(request.form[key])
                    elif key in ['max_risk_percent', 'price_distance_threshold']:
                        new_params[key] = float(request.form[key])
                    else:
                        new_params[key] = request.form[key]
                except Exception as e:
                    logger.error(f"Error actualizando parámetro {key}: {str(e)}")
        
        # Actualizar los valores por defecto
        DEFAULTS.update(new_params)
        logger.info(f"Parámetros actualizados: {DEFAULTS}")
    
    # Obtener datos de caché
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Estadísticas
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=json.dumps(scatter_data),
                           cryptos_count=len(scatter_data))

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Datos no disponibles", 404
    
    df = calculate_indicators(df, DEFAULTS)
    if df is None or len(df) < 20:
        return "Datos insuficientes para generar gráfico", 404
    
    # Buscar señal correspondiente
    signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
    if signals is None:
        return "Señales no disponibles", 404
    
    signal = next((s for s in signals if s['symbol'] == symbol), None)
    
    if not signal:
        return "Señal no encontrada", 404
    
    plot_url = generate_chart(df, signal, signal_type)
    if not plot_url:
        return "Error generando gráfico", 500
        
    return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
