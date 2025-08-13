import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import math
import logging
from threading import Lock
import random

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Mapeo de timeframes
TIMEFRAME_MAP = {
    '30m': '30min',
    '1h': '1hour',
    '2h': '2hour',
    '4h': '4hour',
    '1d': '1day',
    '1w': '1week'
}

# Datos en memoria
crypto_data = {}
last_updated = datetime.utcnow()
next_update = last_updated + timedelta(seconds=CACHE_TIME)
data_lock = Lock()

# Leer lista de criptomonedas
def load_cryptos():
    cryptos = []
    with open(CRYPTOS_FILE, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol:
                cryptos.append(symbol)
    return cryptos

# Obtener datos de KuCoin
def get_kucoin_data(symbol, timeframe):
    kucoin_tf = TIMEFRAME_MAP.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                candles.reverse()  # Invertir para orden cronológico
                
                if len(candles) < 100:
                    logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir a tipos numéricos
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Eliminar filas con valores NaN
                df = df.dropna()
                
                if len(df) < 50:
                    logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                    return None
                
                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
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
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).replace([np.inf, -np.inf], 0)
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
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'], df['low'], df['close'], params['adx_period'])
        
        # Eliminar filas con NaN resultantes de los cálculos
        df = df.dropna()
        
        return df
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
        df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
        
        resistances = df[df['high'] >= df['high_roll']]['high'].unique().tolist()
        supports = df[df['low'] <= df['low_roll']]['low'].unique().tolist()
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
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
        if trend == 'up': long_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: long_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 15
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): long_prob += 10
        
        # Criterios para SHORT
        if trend == 'down': short_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: short_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 15
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): short_prob += 10
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = int((long_prob / total) * 100)
            short_prob = int((short_prob / total) * 100)
        else:
            long_prob = 50
            short_prob = 50
        
        return long_prob, short_prob, volume_class, last['close'], last['adx'] if not pd.isna(last['adx']) else 0
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return 0, 0, 'Muy Bajo', 0, 0

# Calcular distancia al nivel más cercano
def calculate_distance_to_level(price, levels, threshold_percent):
    try:
        if not levels or price is None:
            return False
        min_distance = min(abs(price - level) for level in levels)
        threshold = price * threshold_percent / 100
        return min_distance <= threshold
    except:
        return False

# Tarea en segundo plano para actualizar datos
def background_update():
    global crypto_data, last_updated, next_update
    
    while True:
        try:
            logger.info("Iniciando actualización de datos...")
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            
            # Procesar en lotes para reducir memoria
            batch_size = 20
            for i in range(0, len(cryptos), batch_size):
                batch = cryptos[i:i+batch_size]
                for crypto in batch:
                    try:
                        long_prob, short_prob, volume_class, price, adx = analyze_crypto(crypto, DEFAULTS)
                        
                        # Datos para el gráfico de dispersión
                        scatter_data.append({
                            'symbol': crypto,
                            'long_prob': long_prob,
                            'short_prob': short_prob,
                            'volume': volume_class,
                            'price': price,
                            'adx': adx
                        })
                        
                        # Señal LONG si probabilidad > 70%
                        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
                            long_signals.append({
                                'symbol': crypto,
                                'long_prob': long_prob,
                                'volume': volume_class,
                                'price': price,
                                'adx': adx
                            })
                        
                        # Señal SHORT si probabilidad > 70%
                        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
                            short_signals.append({
                                'symbol': crypto,
                                'short_prob': short_prob,
                                'volume': volume_class,
                                'price': price,
                                'adx': adx
                            })
                    
                    except Exception as e:
                        logger.error(f"Error procesando {crypto}: {str(e)}")
                
                # Liberar memoria entre lotes
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
                    'crypto_count': len(cryptos)
                }
                last_updated = datetime.now()
                next_update = last_updated + timedelta(seconds=CACHE_TIME)
            
            logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} puntos")
        except Exception as e:
            logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

@app.route('/')
def index():
    with data_lock:
        long_signals = crypto_data.get('long_signals', [])
        short_signals = crypto_data.get('short_signals', [])
        scatter_data = crypto_data.get('scatter_data', [])
        last_update = crypto_data.get('last_update', datetime.now())
        crypto_count = crypto_data.get('crypto_count', 0)
    
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
                           crypto_count=crypto_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=json.dumps(scatter_data))

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Datos no disponibles", 404
    
    df = calculate_indicators(df, DEFAULTS)
    if df is None or len(df) < 20:
        return "Datos insuficientes para generar gráfico", 404
    
    # Buscar señal correspondiente
    with data_lock:
        if signal_type == 'long':
            signals = crypto_data.get('long_signals', [])
        else:
            signals = crypto_data.get('short_signals', [])
    
    signal = next((s for s in signals if s['symbol'] == symbol), None)
    
    if not signal:
        return "Señal no encontrada", 404
    
    # Generar gráfico (simplificado para este ejemplo)
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(df['close'], label='Precio')
        plt.title(f'Gráfico de {symbol}')
        plt.legend()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/data')
def get_data():
    with data_lock:
        long_signals = crypto_data.get('long_signals', [])
        short_signals = crypto_data.get('short_signals', [])
        scatter_data = crypto_data.get('scatter_data', [])
        last_update = crypto_data.get('last_update', datetime.now())
    
    return jsonify({
        'long_signals': long_signals,
        'short_signals': short_signals,
        'scatter_data': scatter_data,
        'last_update': last_update.isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
