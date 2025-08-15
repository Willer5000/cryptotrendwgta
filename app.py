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
import plotly
import plotly.graph_objs as go
from threading import Lock
from scipy.signal import argrelextrema

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300  # 5 minutos
DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 20,  # Umbral más bajo
    'rsi_period': 14,
    'sr_window': 50,
    'divergence_lookback': 10,  # Lookback más corto
    'max_risk_percent': 1.5,
    'price_distance_threshold': 1.5,  # Umbral más flexible
    'min_volume_ratio': 1.2  # Nuevo parámetro
}

# Almacenamiento global con bloqueo
global_data = {
    'long_signals': [],
    'short_signals': [],
    'prev_signals': [],
    'scatter_data': [],
    'last_update': datetime.now()
}
data_lock = Lock()

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            # Filtrar símbolos válidos para KuCoin
            valid_cryptos = []
            for crypto in cryptos:
                if crypto.isalpha() and crypto.isupper():
                    valid_cryptos.append(crypto)
            return valid_cryptos[:100]  # Limitar a 100 para pruebas
    except Exception as e:
        logger.error(f"Error cargando criptos: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'LINK']

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
                    candles.reverse()
                    
                    if len(candles) < 100:
                        logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) < 50:
                        logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                        return None
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                    return df
            else:
                logger.warning(f"Intento {attempt+1} para {symbol} falló: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error en intento {attempt+1} para {symbol}: {str(e)}")
        time.sleep(1)
    
    logger.error(f"No se pudieron obtener datos para {symbol} después de {retries} intentos")
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window=14):
    if len(series) < window + 1:
        return pd.Series([50] * len(series))
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        # Calcular +DM y -DM
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
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = dx.rolling(window).mean()
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))

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
            df['high'], 
            df['low'], 
            df['close'], 
            params['adx_period']
        )
        
        # Eliminar filas con NaN resultantes de los cálculos
        df = df.dropna()
        
        return df
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias usando máximos/minimos locales
def find_support_resistance(df, window=10):
    try:
        if len(df) < window * 2:
            return [], []
        
        # Identificar máximos locales
        high = df['high'].values
        max_idx = argrelextrema(high, np.greater, order=window)[0]
        resistances = [high[i] for i in max_idx if i < len(high) - 1]  # Excluir último punto
        
        # Identificar mínimos locales
        low = df['low'].values
        min_idx = argrelextrema(low, np.less, order=window)[0]
        supports = [low[i] for i in min_idx if i < len(low) - 1]  # Excluir último punto
        
        # Consolidar niveles cercanos
        def consolidate_levels(levels, threshold=0.015):
            if not levels:
                return []
                
            levels.sort()
            consolidated = [levels[0]]
            
            for level in levels[1:]:
                if level > consolidated[-1] * (1 + threshold):
                    consolidated.append(level)
                    
            return consolidated
        
        supports = consolidate_levels(supports)
        resistances = consolidate_levels(resistances)
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen basado en relación con el promedio
def classify_volume(current_vol, avg_vol, params):
    try:
        if avg_vol <= 0 or current_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 3.0: return 'Muy Alto'
        if ratio > 2.0: return 'Alto'
        if ratio > params['min_volume_ratio']: return 'Medio'
        if ratio > 0.8: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias con RSI
def detect_divergence(df, lookback):
    try:
        if len(df) < lookback + 10:
            return None
            
        # Seleccionar los últimos datos
        recent = df.iloc[-lookback:]
        
        # Buscar máximos/mínimos recientes en precio y RSI
        price_peaks = recent['close'][recent['close'] == recent['close'].rolling(5, center=True).max()].dropna()
        rsi_peaks = recent['rsi'][recent['rsi'] == recent['rsi'].rolling(5, center=True).max()].dropna()
        price_valleys = recent['close'][recent['close'] == recent['close'].rolling(5, center=True).min()].dropna()
        rsi_valleys = recent['rsi'][recent['rsi'] == recent['rsi'].rolling(5, center=True).min()].dropna()
        
        # Divergencia bajista: precio hace máximos más altos, RSI hace máximos más bajos
        if len(price_peaks) > 1 and len(rsi_peaks) > 1:
            last_price_peak = price_peaks.iloc[-1]
            prev_price_peak = price_peaks.iloc[-2]
            last_rsi_peak = rsi_peaks.iloc[-1]
            prev_rsi_peak = rsi_peaks.iloc[-2]
            
            if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
                return 'bearish'
        
        # Divergencia alcista: precio hace mínimos más bajos, RSI hace mínimos más altos
        if len(price_valleys) > 1 and len(rsi_valleys) > 1:
            last_price_valley = price_valleys.iloc[-1]
            prev_price_valley = price_valleys.iloc[-2]
            last_rsi_valley = rsi_valleys.iloc[-1]
            prev_rsi_valley = rsi_valleys.iloc[-2]
            
            if last_price_valley < prev_price_valley and last_rsi_valley > prev_rsi_valley:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Calcular distancia al nivel más cercano
def calculate_distance_to_level(price, levels, threshold_percent):
    try:
        if not levels or price is None:
            return False, None
        
        min_distance = float('inf')
        closest_level = None
        
        for level in levels:
            distance = abs(price - level)
            if distance < min_distance:
                min_distance = distance
                closest_level = level
        
        threshold = price * threshold_percent / 100
        return min_distance <= threshold, closest_level
    except:
        return False, None

# Detectar quiebres de estructura
def detect_structure_break(df, supports, resistances):
    try:
        if len(df) < 3:
            return False, False
        
        # Últimas 3 velas
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev_prev = df.iloc[-3]
        
        # Quiebre alcista: cierre por encima de resistencia después de tocar soporte
        breakout = False
        for res in resistances:
            if (prev['low'] < res and 
                current['close'] > res and 
                current['close'] > current['open'] and 
                current['volume'] > prev['volume']):
                breakout = True
                break
        
        # Quiebre bajista: cierre por debajo de soporte después de tocar resistencia
        breakdown = False
        for sup in supports:
            if (prev['high'] > sup and 
                current['close'] < sup and 
                current['close'] < current['open'] and 
                current['volume'] > prev['volume']):
                breakdown = True
                break
        
        return breakout, breakdown
    except Exception as e:
        logger.error(f"Error detectando quiebres: {str(e)}")
        return False, False

# Analizar una criptomoneda y calcular señales
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            logger.warning(f"Datos insuficientes para {symbol}")
            return None, None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            logger.warning(f"Indicadores insuficientes para {symbol}")
            return None, None, None, 0, 0, 'Muy Bajo'
        
        # Calcular soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Datos de la última vela
        last = df.iloc[-1]
        prev = df.iloc[-2]  # Vela anterior
        
        # Clasificar volumen
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        volume_class = classify_volume(last['volume'], avg_vol, params)
        
        # Detectar divergencias
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres de estructura
        breakout, breakdown = detect_structure_break(df, supports, resistances)
        
        # Determinar tendencia
        trend_strength = last['adx'] > params['adx_level']
        trend_direction = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if trend_direction == 'up': long_prob += 25
        if trend_strength: long_prob += 20
        if breakout or divergence == 'bullish': long_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 20
        near_support, _ = calculate_distance_to_level(last['close'], supports, params['price_distance_threshold'])
        if near_support: long_prob += 10
        if last['rsi'] < 40: long_prob += 10
        
        # Factores para SHORT
        if trend_direction == 'down': short_prob += 25
        if trend_strength: short_prob += 20
        if breakdown or divergence == 'bearish': short_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 20
        near_resistance, _ = calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold'])
        if near_resistance: short_prob += 10
        if last['rsi'] > 60: short_prob += 10
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Generar señales
        long_signal = None
        short_signal = None
        prev_signal = None
        
        # Señal LONG (última vela)
        if long_prob >= 60 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar soporte más cercano para SL
            _, closest_support = calculate_distance_to_level(last['close'], supports, 5)
            sl = closest_support * 0.99 if closest_support else last['close'] * (1 - params['max_risk_percent']/100)
            
            # Encontrar resistencia más cercana para TP
            _, closest_resistance = calculate_distance_to_level(last['close'], resistances, 5)
            entry = last['close']
            tp1 = closest_resistance if closest_resistance else entry * 1.02
            tp2 = tp1 * 1.02 if closest_resistance else entry * 1.04
            
            risk = entry - sl
            reward1 = tp1 - entry
            reward2 = tp2 - entry
            
            if risk > 0 and reward1 > risk:
                long_signal = {
                    'symbol': symbol,
                    'price': round(last['close'], 4),
                    'entry': round(entry, 4),
                    'sl': round(sl, 4),
                    'tp1': round(tp1, 4),
                    'tp2': round(tp2, 4),
                    'volume': volume_class,
                    'adx': round(last['adx'], 2),
                    'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                    'type': 'LONG'
                }
        
        # Señal SHORT (última vela)
        if short_prob >= 60 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar resistencia más cercana para SL
            _, closest_resistance = calculate_distance_to_level(last['close'], resistances, 5)
            sl = closest_resistance * 1.01 if closest_resistance else last['close'] * (1 + params['max_risk_percent']/100)
            
            # Encontrar soporte más cercano para TP
            _, closest_support = calculate_distance_to_level(last['close'], supports, 5)
            entry = last['close']
            tp1 = closest_support if closest_support else entry * 0.98
            tp2 = tp1 * 0.98 if closest_support else entry * 0.96
            
            risk = sl - entry
            reward1 = entry - tp1
            reward2 = entry - tp2
            
            if risk > 0 and reward1 > risk:
                short_signal = {
                    'symbol': symbol,
                    'price': round(last['close'], 4),
                    'entry': round(entry, 4),
                    'sl': round(sl, 4),
                    'tp1': round(tp1, 4),
                    'tp2': round(tp2, 4),
                    'volume': volume_class,
                    'adx': round(last['adx'], 2),
                    'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                    'type': 'SHORT'
                }
        
        # Detectar señal en vela anterior
        if prev is not None:
            prev_volume_class = classify_volume(prev['volume'], avg_vol, params)
            prev_breakout, prev_breakdown = detect_structure_break(df.iloc[:-1], supports, resistances)
            prev_divergence = detect_divergence(df.iloc[:-1], params['divergence_lookback'])
            
            # Señal en vela anterior
            if (prev_breakout or prev_divergence == 'bullish') and prev_volume_class in ['Medio', 'Alto', 'Muy Alto']:
                prev_signal = {
                    'symbol': symbol,
                    'entry': round(prev['close'], 4),
                    'sl': round(prev['close'] * 0.99, 4),
                    'tp1': round(prev['close'] * 1.02, 4),
                    'tp2': round(prev['close'] * 1.04, 4),
                    'volume': prev_volume_class,
                    'type': 'LONG'
                }
            elif (prev_breakdown or prev_divergence == 'bearish') and prev_volume_class in ['Medio', 'Alto', 'Muy Alto']:
                prev_signal = {
                    'symbol': symbol,
                    'entry': round(prev['close'], 4),
                    'sl': round(prev['close'] * 1.01, 4),
                    'tp1': round(prev['close'] * 0.98, 4),
                    'tp2': round(prev['close'] * 0.96, 4),
                    'volume': prev_volume_class,
                    'type': 'SHORT'
                }
        
        return long_signal, short_signal, prev_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        start_time = time.time()
        try:
            logger.info("Iniciando actualización de datos...")
            cryptos = load_cryptos()
            total_cryptos = len(cryptos)
            long_signals = []
            short_signals = []
            prev_signals = []
            scatter_data = []
            
            # Procesar criptomonedas
            for i, crypto in enumerate(cryptos):
                try:
                    long_signal, short_signal, prev_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                    
                    if long_signal:
                        long_signals.append(long_signal)
                    if short_signal:
                        short_signals.append(short_signal)
                    if prev_signal:
                        prev_signals.append(prev_signal)
                    
                    scatter_data.append({
                        'symbol': crypto,
                        'long_prob': long_prob,
                        'short_prob': short_prob,
                        'volume': volume_class
                    })
                    
                    # Log de progreso
                    progress = (i + 1) / total_cryptos * 100
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progreso: {i+1}/{total_cryptos} ({progress:.1f}%)")
                    
                    # Pausa para no sobrecargar la API
                    time.sleep(0.2)
                except Exception as e:
                    logger.error(f"Error procesando {crypto}: {str(e)}")
            
            # Ordenar por fuerza de tendencia (ADX)
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            prev_signals.sort(key=lambda x: x['entry'], reverse=True)
            
            # Actualizar datos globales
            with data_lock:
                global_data['long_signals'] = long_signals
                global_data['short_signals'] = short_signals
                global_data['prev_signals'] = prev_signals
                global_data['scatter_data'] = scatter_data
                global_data['last_update'] = datetime.now()
            
            elapsed = time.time() - start_time
            logger.info(f"Actualización completada en {elapsed:.1f}s")
            logger.info(f"Señales LONG: {len(long_signals)}")
            logger.info(f"Señales SHORT: {len(short_signals)}")
            logger.info(f"Señales previas: {len(prev_signals)}")
            logger.info(f"Criptos analizadas: {len(scatter_data)}")
        except Exception as e:
            logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

@app.route('/')
def index():
    with data_lock:
        long_signals = global_data['long_signals']
        short_signals = global_data['short_signals']
        prev_signals = global_data['prev_signals']
        scatter_data = global_data['scatter_data']
        last_update = global_data['last_update']
    
    # Estadísticas
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    # Información de criptomonedas analizadas
    cryptos_analyzed = len(scatter_data)
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           prev_signals=prev_signals,
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=scatter_data,
                           cryptos_analyzed=cryptos_analyzed)

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
        signals = global_data['long_signals'] if signal_type == 'long' else global_data['short_signals']
    
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
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', linestyle='--', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', linestyle='--', alpha=0.8)
        
        # Marcar niveles clave
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='-', label='Entrada', alpha=0.7)
            plt.axhline(y=signal['sl'], color='red', linestyle='-', label='Stop Loss', alpha=0.7)
            plt.axhline(y=signal['tp1'], color='blue', linestyle='--', label='TP1', alpha=0.5)
            plt.axhline(y=signal['tp2'], color='purple', linestyle='--', label='TP2', alpha=0.5)
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='-', label='Entrada', alpha=0.7)
            plt.axhline(y=signal['sl'], color='green', linestyle='-', label='Stop Loss', alpha=0.7)
            plt.axhline(y=signal['tp1'], color='blue', linestyle='--', label='TP1', alpha=0.5)
            plt.axhline(y=signal['tp2'], color='purple', linestyle='--', label='TP2', alpha=0.5)
        
        plt.title(f'{signal["symbol"]} - Precio y EMAs', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        colors = ['green' if close >= open else 'red' for close, open in zip(df['close'], df['open'])]
        plt.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        plt.title('Volumen', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown', linewidth=1.5)
        plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        plt.title('Indicadores', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        
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
                value = request.form[param]
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    DEFAULTS[param] = int(value)
                elif param in ['max_risk_percent', 'price_distance_threshold', 'min_volume_ratio']:
                    DEFAULTS[param] = float(value)
                else:
                    DEFAULTS[param] = value
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados correctamente',
            'params': DEFAULTS
        })
    except Exception as e:
        logger.error(f"Error actualizando parámetros: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error actualizando parámetros: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
