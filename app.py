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
import traceback

app = Flask(__name__)
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(cryptos)} cryptos from file")
            return cryptos
    except Exception as e:
        logger.error(f"Error loading cryptos: {str(e)}")
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
        logger.info(f"Fetching data for {symbol} ({kucoin_tf})")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                # KuCoin devuelve las velas en orden descendente, invertimos para tener ascendente
                candles.reverse()
                
                # Validar que hay suficientes velas
                if len(candles) < 100:
                    logger.warning(f"Insufficient data for {symbol}: {len(candles)} candles")
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir a tipos numéricos
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Eliminar filas con valores NaN
                df = df.dropna()
                
                # Validar que aún tenemos suficientes datos
                if len(df) < 50:
                    logger.warning(f"Insufficient data after cleaning for {symbol}: {len(df)} candles")
                    return None
                
                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
        else:
            logger.error(f"API error for {symbol}: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
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
    
    try:
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
        
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).replace([np.inf, -np.inf], 0)
        adx = dx.rolling(window).mean()
        return adx, plus_di, minus_di
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        return pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close))

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
        logger.error(f"Error calculating indicators: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Detectar soportes y resistencias (versión simplificada y robusta)
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Calcular máximos y mínimos en ventana
        highs = df['high'].rolling(window=window, min_periods=1).max()
        lows = df['low'].rolling(window=window, min_periods=1).min()
        
        # Identificar puntos donde el precio es igual al máximo o mínimo de la ventana
        resistances = df[df['high'] == highs]['high'].unique().tolist()
        supports = df[df['low'] == lows]['low'].unique().tolist()
        
        # Eliminar duplicados y ordenar
        supports = sorted(set(supports))
        resistances = sorted(set(resistances))
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error finding S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol == 0 or current_vol is None or avg_vol is None:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.0: return 'Muy Alto'
        if ratio > 1.5: return 'Alto'
        if ratio > 1.0: return 'Medio'
        if ratio > 0.5: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Analizar una criptomoneda y calcular probabilidades
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        # Obtener el último dato
        last = df.iloc[-1]
        
        # Calcular volumen promedio
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Detectar soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades (versión simplificada)
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if trend == 'up': long_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: long_prob += 20
        if last['rsi'] < 40: long_prob += 20  # RSI en zona de sobreventa
        
        # Factores para SHORT
        if trend == 'down': short_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: short_prob += 20
        if last['rsi'] > 60: short_prob += 20  # RSI en zona de sobrecompra
        
        # Ajustar probabilidades para que sumen 100
        total = long_prob + short_prob
        if total > 0:
            long_prob = int((long_prob / total) * 100)
            short_prob = int((short_prob / total) * 100)
        else:
            long_prob = 50
            short_prob = 50
        
        # Solo generar señales si la probabilidad es alta y hay volumen
        long_signal = None
        short_signal = None
        
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            # Calcular parámetros de riesgo básicos
            atr = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
            entry = last['close'] * 1.005
            sl = entry * (1 - params['max_risk_percent']/100)
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
        
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            # Calcular parámetros de riesgo básicos
            atr = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
            entry = last['close'] * 0.995
            sl = entry * (1 + params['max_risk_percent']/100)
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
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        with app.app_context():
            try:
                logger.info("Starting data update...")
                cryptos = load_cryptos()
                long_signals = []
                short_signals = []
                scatter_data = []
                
                # Procesar cada cripto
                for i, crypto in enumerate(cryptos):
                    try:
                        long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                        
                        if long_signal:
                            long_signals.append(long_signal)
                        if short_signal:
                            short_signals.append(short_signal)
                        
                        # Siempre añadir al scatter plot
                        scatter_data.append({
                            'symbol': crypto,
                            'long_prob': long_prob,
                            'short_prob': short_prob,
                            'volume': volume_class
                        })
                        
                        # Pausa para no sobrecargar la API
                        if (i+1) % 5 == 0:
                            time.sleep(1)
                            
                    except Exception as e:
                        logger.error(f"Error processing {crypto}: {str(e)}")
                
                # Ordenar por fuerza de tendencia (ADX)
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar caché
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                
                logger.info(f"Update completed: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} points")
            except Exception as e:
                logger.error(f"Error in background update: {str(e)}")
                logger.error(traceback.format_exc())
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
try:
    logger.info("Starting update thread...")
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    logger.info("Update thread started")
except Exception as e:
    logger.error(f"Failed to start update thread: {str(e)}")

@app.route('/')
def index():
    try:
        long_signals = cache.get('long_signals') or []
        short_signals = cache.get('short_signals') or []
        scatter_data = cache.get('scatter_data') or []
        last_update = cache.get('last_update') or datetime.now()
        
        # Estadísticas para gráficos
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
                            scatter_data=json.dumps(scatter_data))
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html', message="Error loading data")

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
        if df is None:
            return "Data not available", 404
        
        df = calculate_indicators(df, DEFAULTS)
        if df is None or len(df) < 20:
            return "Insufficient data", 404
        
        # Buscar señal correspondiente
        signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
        if signals is None:
            return "Signals not available", 404
        
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Signal not found", 404        
        return render_template('chart.html', symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error in chart route: {str(e)}")
        return "Internal server error", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
