import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
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
app.logger.setLevel(logging.INFO)
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
            app.logger.info(f"Loaded {len(cryptos)} cryptos from file")
            return cryptos
    except Exception as e:
        app.logger.error(f"Error loading cryptos: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL']  # Default cryptos

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
        response = requests.get(url, timeout=30)
        app.logger.info(f"API response for {symbol}-USDT: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                candles.reverse()
                
                if len(candles) < 50:
                    app.logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir a tipos numéricos
                numeric_cols = ['open', 'close', 'high', 'low', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                if len(df) < 50:
                    app.logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                    return None
                
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
        else:
            app.logger.error(f"Error fetching data for {symbol}: HTTP {response.status_code}")
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}\n{traceback.format_exc()}")
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
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # Eliminar filas con NaN
        df = df.dropna()
        return df
    except Exception as e:
        app.logger.error(f"Error calculando indicadores: {str(e)}\n{traceback.format_exc()}")
        return None

# Detectar soportes y resistencias (versión simplificada)
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Usar los precios de cierre para simplificar
        rolling_high = df['close'].rolling(window=window).max()
        rolling_low = df['close'].rolling(window=window).min()
        
        # Identificar puntos clave
        resistances = rolling_high.drop_duplicates().tolist()
        supports = rolling_low.drop_duplicates().tolist()
        
        return supports, resistances
    except Exception as e:
        app.logger.error(f"Error buscando S/R: {str(e)}")
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
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Calcular probabilidades básicas
        trend_strength = last['adx'] / 100 if not pd.isna(last['adx']) else 0.5
        rsi_value = last['rsi'] / 100 if not pd.isna(last['rsi']) else 50
        volume_factor = {
            'Muy Alto': 0.9,
            'Alto': 0.7,
            'Medio': 0.5,
            'Bajo': 0.3,
            'Muy Bajo': 0.1
        }.get(volume_class, 0.5)
        
        long_prob = min(100, int((
            (1 if last['ema_fast'] > last['ema_slow'] else 0.2) * 40 +
            trend_strength * 30 +
            (1 - (rsi_value / 100)) * 20 +
            volume_factor * 10
        ))
        
        short_prob = min(100, int((
            (1 if last['ema_fast'] < last['ema_slow'] else 0.2) * 40 +
            trend_strength * 30 +
            (rsi_value / 100) * 20 +
            volume_factor * 10
        ))
        
        # Señales solo si probabilidad > 70% y volumen suficiente
        long_signal = None
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = last['close'] * 1.01
            sl = entry * (1 - params['max_risk_percent']/100)
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry * 1.02, 4),
                'tp2': round(entry * 1.04, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(1.0, 2)
            }
        
        short_signal = None
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = last['close'] * 0.99
            sl = entry * (1 + params['max_risk_percent']/100)
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry * 0.98, 4),
                'tp2': round(entry * 0.96, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(1.0, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}\n{traceback.format_exc()}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        try:
            with app.app_context():
                app.logger.info("Iniciando actualización de datos...")
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
                        
                        # Datos para el gráfico de dispersión
                        scatter_data.append({
                            'symbol': crypto,
                            'long_prob': long_prob,
                            'short_prob': short_prob,
                            'volume': volume_class
                        })
                        
                        time.sleep(0.5)  # Evitar sobrecarga
                    except Exception as e:
                        app.logger.error(f"Error processing {crypto}: {str(e)}")
                
                # Ordenar por ADX
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar caché
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                
                app.logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} puntos scatter")
        except Exception as e:
            app.logger.error(f"Error en actualización de fondo: {str(e)}\n{traceback.format_exc()}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
def start_update_thread():
    try:
        update_thread = threading.Thread(target=background_update, daemon=True)
        update_thread.start()
        app.logger.info("Hilo de actualización iniciado")
    except Exception as e:
        app.logger.error(f"No se pudo iniciar el hilo de actualización: {str(e)}")

start_update_thread()

@app.route('/', methods=['GET', 'POST'])
def index():
    # Manejar actualización de parámetros
    params = DEFAULTS.copy()
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                try:
                    if key in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                        params[key] = int(request.form[key])
                    elif key in ['max_risk_percent', 'price_distance_threshold']:
                        params[key] = float(request.form[key])
                    else:
                        params[key] = request.form[key]
                except:
                    pass
    
    # Obtener datos de caché
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Estadísticas
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    app.logger.info(f"Enviando a frontend: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} puntos")
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=params,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=scatter_data)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
        if df is None:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, DEFAULTS)
        if df is None or len(df) < 20:
            return "Datos insuficientes", 404
        
        # Buscar señal en caché
        signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        # Generar gráfico básico
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['close'], label='Precio')
        plt.title(f'{symbol} - Precio')
        plt.grid(True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        app.logger.error(f"Error generating chart: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
