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
from scipy.signal import argrelextrema

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300  # 5 minutos para actualizaciones más frecuentes
DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 20,  # Umbral reducido
    'rsi_period': 14,
    'sr_window': 50,
    'divergence_lookback': 10,  # Lookback reducido
    'max_risk_percent': 1.5,
    'price_distance_threshold': 1.5  # Umbral aumentado
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        cryptos = [line.strip() for line in f.readlines() if line.strip()]
    return cryptos[:50]  # Limitar a 50 para pruebas

# Obtener datos de KuCoin mejorado
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
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data.get('code') != '200000' or not data.get('data'):
            logger.warning(f"Respuesta inesperada de KuCoin para {symbol}: {data}")
            return None
            
        candles = data['data']
        if len(candles) < 100:
            logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
            return None
        
        # Procesar velas
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df = df.iloc[::-1].reset_index(drop=True)  # Invertir orden
        
        # Convertir tipos
        numeric_cols = ['open', 'close', 'high', 'low', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 50:
            logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementación mejorada de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación mejorada de RSI
def calculate_rsi(series, window=14):
    if len(series) < window + 1:
        return pd.Series([50] * len(series))
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / (avg_loss.replace(0, 0.001))  # Evitar división por cero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación mejorada de ADX
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        # Calcular +DM y -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm <= 0] = 0
        minus_dm[minus_dm <= 0] = 0
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Suavizar
        atr = tr.rolling(window).mean()
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window).mean()
        
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))

# Detectar soportes y resistencias mejorado
def find_support_resistance(df, window=5):
    try:
        # Encontrar mínimos locales (soportes)
        min_idx = argrelextrema(df['low'].values, np.less, order=window)[0]
        supports = df.iloc[min_idx]['low'].values.tolist()
        
        # Encontrar máximos locales (resistencias)
        max_idx = argrelextrema(df['high'].values, np.greater, order=window)[0]
        resistances = df.iloc[max_idx]['high'].values.tolist()
        
        # Consolidar niveles cercanos
        def consolidate_levels(levels, threshold=0.02):
            if not levels: return []
            levels.sort()
            consolidated = [levels[0]]
            for level in levels[1:]:
                if abs(level - consolidated[-1]) > threshold * consolidated[-1]:
                    consolidated.append(level)
            return consolidated
        
        return consolidate_levels(supports), consolidate_levels(resistances)
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen mejorado
def classify_volume(current_vol, historical_vol):
    try:
        if historical_vol <= 0: return 'Muy Bajo'
        
        ratio = current_vol / historical_vol
        if ratio > 2.5: return 'Muy Alto'
        if ratio > 1.8: return 'Alto'
        if ratio > 1.2: return 'Medio'
        if ratio > 0.7: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias mejorado
def detect_divergence(df, lookback=14):
    try:
        if len(df) < lookback + 10:
            return None
            
        # Seleccionar los últimos datos
        recent = df.iloc[-lookback:]
        
        # Buscar máximos/mínimos recientes
        price_peaks = recent[recent['high'] == recent['high'].rolling(5, center=True).max()]
        price_valleys = recent[recent['low'] == recent['low'].rolling(5, center=True).min()]
        
        # Divergencia bajista
        if len(price_peaks) > 1:
            last_peak = price_peaks.iloc[-1]
            prev_peak = price_peaks.iloc[-2]
            
            if (last_peak['high'] > prev_peak['high'] and 
                last_peak['rsi'] < prev_peak['rsi'] and
                last_peak['rsi'] > 65):
                return 'bearish'
        
        # Divergencia alcista
        if len(price_valleys) > 1:
            last_valley = price_valleys.iloc[-1]
            prev_valley = price_valleys.iloc[-2]
            
            if (last_valley['low'] < prev_valley['low'] and 
                last_valley['rsi'] > prev_valley['rsi'] and
                last_valley['rsi'] < 35):
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Analizar una criptomoneda mejorado
def analyze_crypto(symbol, params):
    try:
        logger.info(f"Analizando {symbol}...")
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            logger.warning(f"No se pudieron obtener datos para {symbol}")
            return None, None, 0, 0, 'Muy Bajo'
        
        # Calcular indicadores
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # Filtrar datos recientes
        df = df.dropna()
        if len(df) < 50:
            logger.warning(f"Datos insuficientes para análisis técnico en {symbol}")
            return None, None, 0, 0, 'Muy Bajo'
        
        # Obtener últimos datos
        last = df.iloc[-1]
        avg_vol = df['volume'].mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Detectar niveles clave
        supports, resistances = find_support_resistance(df, params['sr_window'])
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Determinar tendencia
        trend_strength = 1 if last['adx'] > params['adx_level'] else 0.5
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades
        long_score = 0
        short_score = 0
        
        # Factores para LONG
        if trend == 'up': long_score += 30 * trend_strength
        if last['rsi'] < 45: long_score += 20
        if any(last['close'] > r * 1.01 for r in resistances): long_score += 25  # Breakout
        if volume_class in ['Alto', 'Muy Alto']: long_score += 15
        if divergence == 'bullish': long_score += 25
        
        # Factores para SHORT
        if trend == 'down': short_score += 30 * trend_strength
        if last['rsi'] > 55: short_score += 20
        if any(last['close'] < s * 0.99 for s in supports): short_score += 25  # Breakdown
        if volume_class in ['Alto', 'Muy Alto']: short_score += 15
        if divergence == 'bearish': short_score += 25
        
        # Normalizar
        total = long_score + short_score
        long_prob = (long_score / total) * 100 if total > 0 else 50
        short_prob = (short_score / total) * 100 if total > 0 else 50
        
        # Generar señales (umbral reducido)
        long_signal = None
        if long_prob > 55 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar resistencia más cercana
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.01 if next_resistances else last['close'] * 1.015
            
            # Encontrar soporte más cercano para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.99 if next_supports else entry * 0.98
            
            risk = entry - sl
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        short_signal = None
        if short_prob > 55 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar soporte más cercano
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.99 if next_supports else last['close'] * 0.985
            
            # Encontrar resistencia más cercana para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.01 if next_resistances else entry * 1.02
            
            risk = sl - entry
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}", exc_info=True)
        return None, None, 0, 0, 'Muy Bajo'

# Tarea de actualización en segundo plano
def background_update():
    while True:
        start_time = time.time()
        try:
            logger.info("====== INICIANDO ACTUALIZACIÓN DE DATOS ======")
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            
            total = len(cryptos)
            for i, crypto in enumerate(cryptos):
                try:
                    long_sig, short_sig, long_prob, short_prob, vol = analyze_crypto(crypto, DEFAULTS)
                    if long_sig:
                        long_signals.append(long_sig)
                    if short_sig:
                        short_signals.append(short_sig)
                    
                    scatter_data.append({
                        'symbol': crypto,
                        'long_prob': round(long_prob, 1),
                        'short_prob': round(short_prob, 1),
                        'volume': vol
                    })
                    
                    # Progreso
                    progress = (i + 1) / total * 100
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progreso: {i+1}/{total} ({progress:.1f}%)")
                except Exception as e:
                    logger.error(f"Error procesando {crypto}: {str(e)}")
            
            # Ordenar y almacenar
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            cache.set('long_signals', long_signals)
            cache.set('short_signals', short_signals)
            cache.set('scatter_data', scatter_data)
            cache.set('last_update', datetime.now())
            
            elapsed = time.time() - start_time
            logger.info(f"Actualización completada en {elapsed:.1f}s")
            logger.info(f"Señales LONG: {len(long_signals)}")
            logger.info(f"Señales SHORT: {len(short_signals)}")
            logger.info(f"Criptos analizadas: {len(scatter_data)}")
        except Exception as e:
            logger.error(f"Error en actualización de fondo: {str(e)}", exc_info=True)
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

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
    
    return render_template('index.html', 
                           long_signals=long_signals,
                           short_signals=short_signals,
                           last_update=last_update,
                           params=DEFAULTS,
                           avg_adx_long=round(avg_adx_long, 1),
                           avg_adx_short=round(avg_adx_short, 1),
                           scatter_data=scatter_data,
                           cryptos_analyzed=cryptos_analyzed)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
        if df is None or len(df) < 50:
            return "Datos no disponibles", 404
        
        # Calcular indicadores
        df['ema_fast'] = calculate_ema(df['close'], DEFAULTS['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], DEFAULTS['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], DEFAULTS['rsi_period'])
        df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], DEFAULTS['adx_period'])
        
        # Obtener señal
        signals = long_signals if signal_type == 'long' else short_signals
        signals = cache.get('long_signals' if signal_type == 'long' else 'short_signals') or []
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        # Crear gráfico
        plt.figure(figsize=(14, 10))
        
        # Precio y EMAs
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA{DEFAULTS["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA{DEFAULTS["ema_slow"]}', color='green', alpha=0.8)
        
        # Niveles de trading
        plt.axhline(y=signal['entry'], color='lime' if signal_type == 'long' else 'red', 
                    linestyle='--', label='Entrada')
        plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
        plt.axhline(y=signal['tp1'], color='blue', linestyle=':', label='TP1')
        plt.axhline(y=signal['tp2'], color='purple', linestyle=':', label='TP2')
        
        plt.title(f'{symbol} - Análisis Técnico')
        plt.legend()
        plt.grid(True)
        
        # Volumen
        plt.subplot(3, 1, 2)
        colors = np.where(df['close'] > df['open'], 'green', 'red')
        plt.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        plt.title('Volumen')
        plt.grid(True)
        
        # Indicadores
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
        
        # Convertir a imagen
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}", exc_info=True)
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
                elif param in ['max_risk_percent', 'price_distance_threshold']:
                    DEFAULTS[param] = float(value)
                else:
                    DEFAULTS[param] = value
        
        # Forzar actualización
        cache.set('last_update', datetime.now() - timedelta(seconds=CACHE_TIME))
        
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
