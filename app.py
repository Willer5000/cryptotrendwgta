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
import talib  # Añadido para mejores cálculos de indicadores

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 600  # 10 minutos
DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 50,
    'divergence_lookback': 14,
    'max_risk_percent': 1.5,
    'price_distance_threshold': 1.0
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
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # Convertir a tipos numéricos
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Eliminar filas con valores NaN
                df = df.dropna()
                
                # Validar que tenemos suficientes datos
                if len(df) > 50:
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Calcular indicadores usando TA-Lib (más preciso)
def calculate_indicators(df, params):
    try:
        # Convertir a arrays de tipo float
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        
        # EMA
        df['ema_fast'] = talib.EMA(close, timeperiod=params['ema_fast'])
        df['ema_slow'] = talib.EMA(close, timeperiod=params['ema_slow'])
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=params['rsi_period'])
        
        # ADX
        df['adx'] = talib.ADX(high, low, close, timeperiod=params['adx_period'])
        
        # Eliminar filas con NaN resultantes de los cálculos
        df = df.dropna()
        
        return df
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias mejorado
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Identificar pivots de precio
        df['pivot'] = df['low'].rolling(window=window, center=True).min()
        df['resistance'] = df['high'].rolling(window=window, center=True).max()
        
        # Filtrar niveles significativos
        supports = df[df['low'] == df['pivot']]['low'].dropna().unique().tolist()
        resistances = df[df['high'] == df['resistance']]['high'].dropna().unique().tolist()
        
        # Filtrar niveles cercanos
        filtered_supports = []
        filtered_resistances = []
        
        for s in supports:
            if not filtered_supports or min([abs(s - fs) for fs in filtered_supports]) > s * 0.01:
                filtered_supports.append(s)
        
        for r in resistances:
            if not filtered_resistances or min([abs(r - fr) for fr in filtered_resistances]) > r * 0.01:
                filtered_resistances.append(r)
        
        return filtered_supports, filtered_resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen mejorado
def classify_volume(current_vol, vol_series):
    try:
        if current_vol is None or vol_series.empty:
            return 'Muy Bajo'
        
        # Calcular percentiles
        p25 = vol_series.quantile(0.25)
        p50 = vol_series.quantile(0.50)
        p75 = vol_series.quantile(0.75)
        p90 = vol_series.quantile(0.90)
        
        if current_vol > p90 * 1.5: return 'Muy Alto'
        if current_vol > p75: return 'Alto'
        if current_vol > p50: return 'Medio'
        if current_vol > p25: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias mejorado
def detect_divergence(df, lookback):
    try:
        if len(df) < lookback + 1:
            return None
        
        # Seleccionar los últimos datos
        recent = df.iloc[-lookback:]
        
        # Buscar máximos/mínimos recientes
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        # Divergencia bajista
        if not pd.isna(high_idx):
            price_high = df.loc[high_idx, 'high']
            rsi_high = df.loc[high_idx, 'rsi']
            current_price = df.iloc[-1]['high']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price > price_high and current_rsi < rsi_high - 5 and current_rsi > 65:
                return 'bearish'
        
        # Divergencia alcista
        if not pd.isna(low_idx):
            price_low = df.loc[low_idx, 'low']
            rsi_low = df.loc[low_idx, 'rsi']
            current_price = df.iloc[-1]['low']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price < price_low and current_rsi > rsi_low + 5 and current_rsi < 35:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

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

# Analizar una criptomoneda mejorado
def analyze_crypto(symbol, params):
    df = get_binance_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        vol_series = df['volume'].tail(100)
        volume_class = classify_volume(last['volume'], vol_series)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        adx_strength = last['adx'] > params['adx_level']
        
        # Calcular probabilidades mejoradas
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG
        if trend == 'up': long_prob += 25
        if adx_strength: long_prob += 20
        if divergence == 'bullish': long_prob += 20
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): long_prob += 15
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 20
        
        # Criterios para SHORT
        if trend == 'down': short_prob += 25
        if adx_strength: short_prob += 20
        if divergence == 'bearish': short_prob += 20
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): short_prob += 15
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 20
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        else:
            long_prob = 50
            short_prob = 50
        
        # Señales LONG
        long_signal = None
        if long_prob >= 60 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar soporte más cercano para SL
            next_supports = [s for s in supports if s < last['close']]
            sl = max(next_supports) * 0.995 if next_supports else last['close'] * (1 - params['max_risk_percent']/100)
            
            # Calcular entrada y objetivos
            entry = last['close'] * 1.002
            risk = entry - sl
            tp1 = entry + risk * 1.5
            tp2 = entry + risk * 3
            
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 60 and volume_class in ['Medio', 'Alto', 'Muy Alto']:
            # Encontrar resistencia más cercana para SL
            next_resistances = [r for r in resistances if r > last['close']]
            sl = min(next_resistances) * 1.005 if next_resistances else last['close'] * (1 + params['max_risk_percent']/100)
            
            # Calcular entrada y objetivos
            entry = last['close'] * 0.998
            risk = sl - entry
            tp1 = entry - risk * 1.5
            tp2 = entry - risk * 3
            
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano optimizada
def background_update():
    while True:
        start_time = time.time()
        try:
            logger.info("Iniciando actualización de datos...")
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            
            # Procesar en paralelo con ThreadPool
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_crypto(crypto):
                try:
                    long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(
                        crypto, DEFAULTS)
                    
                    result = {
                        'symbol': crypto,
                        'long_signal': long_signal,
                        'short_signal': short_signal,
                        'long_prob': long_prob,
                        'short_prob': short_prob,
                        'volume': volume_class
                    }
                    return result
                except Exception as e:
                    logger.error(f"Error procesando {crypto}: {str(e)}")
                    return None
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_crypto, crypto) for crypto in cryptos]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        if result['long_signal']:
                            long_signals.append(result['long_signal'])
                        if result['short_signal']:
                            short_signals.append(result['short_signal'])
                        
                        scatter_data.append({
                            'symbol': result['symbol'],
                            'long_prob': result['long_prob'],
                            'short_prob': result['short_prob'],
                            'volume': result['volume']
                        })
            
            # Ordenar por fuerza de tendencia (ADX)
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar caché
            cache.set('long_signals', long_signals)
            cache.set('short_signals', short_signals)
            cache.set('scatter_data', scatter_data)
            cache.set('last_update', datetime.now())
            
            elapsed = time.time() - start_time
            logger.info(f"Actualización completada en {elapsed:.1f}s: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} cryptos")
        except Exception as e:
            logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
if not hasattr(app, 'update_thread') or not app.update_thread.is_alive():
    try:
        app.update_thread = threading.Thread(target=background_update, daemon=True)
        app.update_thread.start()
        logger.info("Hilo de actualización iniciado")
    except Exception as e:
        logger.error(f"No se pudo iniciar el hilo de actualización: {str(e)}")

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
    
    # Información de criptomonedas analizadas
    cryptos_analyzed = len(scatter_data)
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
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
    if df is None or len(df) < 20:
        return "Datos insuficientes para generar gráfico", 404
    
    # Buscar señal correspondiente
    signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
    signal = None
    if signals:
        for s in signals:
            if s['symbol'] == symbol:
                signal = s
                break
    
    if not signal:
        return "Señal no encontrada", 404
    
    try:
        plt.figure(figsize=(14, 10))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', alpha=0.8)
        
        # Marcar niveles clave
        plt.axhline(y=signal['entry'], color='lime', linestyle='-', linewidth=1.5, alpha=0.7, label='Entrada')
        plt.axhline(y=signal['sl'], color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Stop Loss')
        plt.axhline(y=signal['tp1'], color='cyan', linestyle='--', linewidth=1.2, alpha=0.7, label='TP1')
        plt.axhline(y=signal['tp2'], color='blue', linestyle='--', linewidth=1.2, alpha=0.7, label='TP2')
        
        plt.title(f'{signal["symbol"]} - Precio y EMAs')
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
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        plt.ylim(0, 100)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown', linewidth=1.5)
        plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.7)
        
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
                elif param in ['max_risk_percent', 'price_distance_threshold']:
                    DEFAULTS[param] = float(request.form[param])
                else:
                    DEFAULTS[param] = request.form[param]
        
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
