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
import plotly
import plotly.graph_objs as go
from threading import Lock
from collections import defaultdict
import talib

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

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
    'price_distance_threshold': 1.0,
    'volume_threshold': 1000000
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Obtener datos de Binance (más confiable)
def get_binance_data(symbol, timeframe):
    tf_mapping = {
        '30m': '30m',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1w'
    }
    binance_tf = tf_mapping.get(timeframe, '1h')
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval={binance_tf}&limit=300"
    
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if data:
                # Procesar datos
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = pd.DataFrame(data, columns=columns)
                
                # Convertir a tipos numéricos
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Eliminar filas con valores NaN
                df = df.dropna()
                
                # Validar que tenemos suficientes datos
                if len(df) < 100:
                    logger.warning(f"Datos insuficientes para {symbol}: {len(df)} velas")
                    return None
                
                return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementación mejorada de EMA
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Implementación mejorada de RSI
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# Implementación mejorada de ADX
def calculate_adx(high, low, close, window):
    try:
        adx = talib.ADX(high, low, close, timeperiod=window)
        return adx
    except:
        return pd.Series([0] * len(close))

# Calcular indicadores
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'] = calculate_adx(
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

# Detectar soportes y resistencias mejorado
def find_support_resistance(df, window):
    try:
        # Identificar máximos y mínimos locales
        df['min'] = df['low'].rolling(window=window, center=True).min()
        df['max'] = df['high'].rolling(window=window, center=True).max()
        
        # Filtrar niveles significativos
        supports = df[df['low'] == df['min']]['low'].unique().tolist()
        resistances = df[df['high'] == df['max']]['high'].unique().tolist()
        
        # Agrupar niveles cercanos
        def cluster_levels(levels, threshold=0.01):
            clusters = []
            for level in sorted(levels):
                found = False
                for cluster in clusters:
                    if abs(level - cluster['mean']) / cluster['mean'] < threshold:
                        cluster['levels'].append(level)
                        cluster['mean'] = sum(cluster['levels']) / len(cluster['levels'])
                        found = True
                        break
                if not found:
                    clusters.append({'mean': level, 'levels': [level]})
            return [cluster['mean'] for cluster in clusters]
        
        supports = cluster_levels(supports)
        resistances = cluster_levels(resistances)
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen mejorado
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0 or current_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 3.0: return 'Muy Alto'
        if ratio > 1.8: return 'Alto'
        if ratio > 1.2: return 'Medio'
        if ratio > 0.7: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias mejorado
def detect_divergence(df, lookback):
    try:
        # Buscar divergencias en los últimos N períodos
        recent = df.iloc[-lookback:]
        
        # Divergencia alcista: precio hace mínimo más bajo, RSI hace mínimo más alto
        min_idx = recent['close'].idxmin()
        rsi_min_idx = recent['rsi'].idxmin()
        if min_idx != rsi_min_idx and min_idx > rsi_min_idx:
            price_lows = df.loc[[rsi_min_idx, min_idx], 'close']
            rsi_lows = df.loc[[rsi_min_idx, min_idx], 'rsi']
            if price_lows[0] > price_lows[1] and rsi_lows[0] < rsi_lows[1]:
                return 'bullish'
        
        # Divergencia bajista: precio hace máximo más alto, RSI hace máximo más bajo
        max_idx = recent['close'].idxmax()
        rsi_max_idx = recent['rsi'].idxmax()
        if max_idx != rsi_max_idx and max_idx > rsi_max_idx:
            price_highs = df.loc[[rsi_max_idx, max_idx], 'close']
            rsi_highs = df.loc[[rsi_max_idx, max_idx], 'rsi']
            if price_highs[0] < price_highs[1] and rsi_highs[0] > rsi_highs[1]:
                return 'bearish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Calcular distancia al nivel más cercano
def calculate_distance_to_level(price, levels, threshold_percent):
    if not levels or price is None:
        return False
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    return min_distance <= threshold

# Analizar una criptomoneda mejorado
def analyze_crypto(symbol, params):
    df = get_binance_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(50).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Filtrar por volumen mínimo
        if last['volume'] < params['volume_threshold']:
            return None, None, 0, 0, volume_class
        
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG
        if trend == 'up': long_prob += 30
        if last['adx'] > params['adx_level']: long_prob += 20
        if is_breakout or divergence == 'bullish': long_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 15
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): long_prob += 10
        
        # Criterios para SHORT
        if trend == 'down': short_prob += 30
        if last['adx'] > params['adx_level']: short_prob += 20
        if is_breakdown or divergence == 'bearish': short_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 15
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): short_prob += 10
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Señales LONG
        long_signal = None
        if long_prob >= 65 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar resistencia más cercana
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            
            # Encontrar soporte más cercano para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * 0.98
            
            risk = entry - sl
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 65 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar soporte más cercano
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            
            # Encontrar resistencia más cercana para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * 1.02
            
            risk = sl - entry
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano mejorada
def background_update():
    while True:
        with app.app_context():
            try:
                logger.info("Iniciando actualización de datos...")
                cryptos = load_cryptos()
                long_signals = []
                short_signals = []
                scatter_data = []
                stats = defaultdict(int)
                
                for i, crypto in enumerate(cryptos):
                    try:
                        long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                        
                        if long_signal:
                            long_signals.append(long_signal)
                            stats['long_signals'] += 1
                        if short_signal:
                            short_signals.append(short_signal)
                            stats['short_signals'] += 1
                        
                        scatter_data.append({
                            'symbol': crypto,
                            'long_prob': round(long_prob, 1),
                            'short_prob': round(short_prob, 1),
                            'volume': volume_class
                        })
                        
                        # Pequeña pausa cada 10 cryptos
                        if i % 10 == 0:
                            time.sleep(1)
                    except Exception as e:
                        logger.error(f"Error procesando {crypto}: {str(e)}")
                
                # Ordenar señales por fuerza
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

# Iniciar hilo de actualización
update_thread = None
if not update_thread:
    try:
        update_thread = threading.Thread(target=background_update, daemon=True)
        update_thread.start()
        logger.info("Hilo de actualización iniciado")
    except Exception as e:
        logger.error(f"No se pudo iniciar el hilo: {str(e)}")

@app.route('/')
def index():
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Estadísticas
    cryptos_analyzed = len(scatter_data)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    # Preparar datos para gráficos
    market_data = {
        'labels': ['LONG', 'SHORT', 'Neutral'],
        'data': [
            len(long_signals), 
            len(short_signals), 
            cryptos_analyzed - len(long_signals) - len(short_signals)
        ]
    }
    
    long_stats = {
        'distance': np.mean([s['distance'] for s in long_signals]) if long_signals else 0
    }
    
    short_stats = {
        'distance': np.mean([s['distance'] for s in short_signals]) if short_signals else 0
    }
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=DEFAULTS,
                           market_data=market_data,
                           long_stats=long_stats,
                           short_stats=short_stats,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=scatter_data,
                           cryptos_analyzed=cryptos_analyzed)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    df = get_binance_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Datos no disponibles", 404
    
    df = calculate_indicators(df, DEFAULTS)
    if df is None or len(df) < 50:
        return "Datos insuficientes", 404
    
    # Buscar señal
    signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
    if not signals:
        return "Señales no disponibles", 404
    
    signal = next((s for s in signals if s['symbol'] == symbol), None)
    if not signal:
        return "Señal no encontrada", 404
    
    try:
        plt.figure(figsize=(14, 10))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA{DEFAULTS["ema_fast"]}', color='orange', linestyle='--')
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA{DEFAULTS["ema_slow"]}', color='green', linestyle='--')
        
        # Líneas de entrada y SL
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='-', label='Entrada')
            plt.axhline(y=signal['sl'], color='red', linestyle='-', label='Stop Loss')
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='-', label='Entrada')
            plt.axhline(y=signal['sl'], color='green', linestyle='-', label='Stop Loss')
        
        plt.title(f'{symbol}-USDT - Señal {signal_type.upper()}')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        plt.bar(df['timestamp'], df['volume'], color=colors)
        plt.title('Volumen')
        plt.grid(True)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        plt.title('Indicadores')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
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

@app.route('/update_params', methods=['POST'])
def update_params():
    try:
        # Actualizar parámetros
        for param in DEFAULTS:
            if param in request.form:
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    DEFAULTS[param] = int(request.form[param])
                elif param in ['max_risk_percent', 'price_distance_threshold', 'volume_threshold']:
                    DEFAULTS[param] = float(request.form[param])
                else:
                    DEFAULTS[param] = request.form[param]
        
        # Forzar actualización
        cache.set('last_update', datetime.now() - timedelta(seconds=CACHE_TIME))
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados',
            'params': DEFAULTS
        })
    except Exception as e:
        logger.error(f"Error actualizando parámetros: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
