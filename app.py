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
            app.logger.info(f"Loaded {len(cryptos)} cryptocurrencies")
            return cryptos
    except Exception as e:
        app.logger.error(f"Error loading cryptos: {str(e)}")
        return []

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
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data:
                # Crear DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # Convertir tipos
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # Validar datos
                if len(df) < 100:
                    app.logger.warning(f"Insuficient data for {symbol}: {len(df)} candles")
                    return None
                
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        else:
            app.logger.warning(f"Binance API error for {symbol}: HTTP {response.status_code}")
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window=14):
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
def calculate_adx(high, low, close, window=14):
    if len(close) < window * 2:
        return pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close))
    
    try:
        # Calcular +DM y -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Calcular True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Suavizar con EMA
        atr = calculate_ema(tr, window)
        
        # Calcular indicadores direccionales
        plus_di = 100 * calculate_ema(pd.Series(plus_dm), window) / atr
        minus_di = 100 * calculate_ema(pd.Series(minus_dm), window) / atr
        
        # Calcular DX y ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = calculate_ema(dx, window)
        
        return adx, plus_di, minus_di
    except Exception as e:
        app.logger.error(f"Error calculating ADX: {str(e)}")
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
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'], df['low'], df['close'], params['adx_period']
        )
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        return df
    except Exception as e:
        app.logger.error(f"Error calculating indicators: {str(e)}")
        return None

# Detectar soportes y resistencias simplificado
def find_support_resistance(df, window=50):
    try:
        if len(df) < window:
            return [], []
        
        # Identificar máximos y mínimos locales
        highs = df['high'].values
        lows = df['low'].values
        
        supports = []
        resistances = []
        
        # Buscar pivotes en los últimos datos
        for i in range(10, len(highs)-10):
            # Máximo local
            if highs[i] == max(highs[i-10:i+11]):
                resistances.append(highs[i])
            # Mínimo local
            if lows[i] == min(lows[i-10:i+11]):
                supports.append(lows[i])
        
        return supports, resistances
    except Exception as e:
        app.logger.error(f"Error finding S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0 or current_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.5: return 'Muy Alto'
        if ratio > 1.8: return 'Alto'
        if ratio > 1.2: return 'Medio'
        if ratio > 0.8: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias simplificado
def detect_divergence(df, lookback=20):
    try:
        if len(df) < lookback + 1:
            return None
            
        # Seleccionar los últimos datos
        recent = df.iloc[-lookback:]
        
        # Encontrar máximos y mínimos
        max_idx = recent['high'].idxmax()
        min_idx = recent['low'].idxmin()
        
        # Divergencia bajista
        if not pd.isna(max_idx):
            price_high = df.loc[max_idx, 'high']
            rsi_high = df.loc[max_idx, 'rsi']
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            
            if current_price > price_high and current_rsi < rsi_high - 5 and current_rsi > 60:
                return 'bearish'
        
        # Divergencia alcista
        if not pd.isna(min_idx):
            price_low = df.loc[min_idx, 'low']
            rsi_low = df.loc[min_idx, 'rsi']
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            
            if current_price < price_low and current_rsi > rsi_low + 5 and current_rsi < 40:
                return 'bullish'
        
        return None
    except Exception as e:
        app.logger.error(f"Error detecting divergence: {str(e)}")
        return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_binance_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            app.logger.warning(f"Insuficient data for {symbol}")
            return None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            app.logger.warning(f"Insuficient indicators for {symbol}")
            return None, None, 0, 0, 'Muy Bajo'
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calcular volumen
        avg_vol = df['volume'].tail(50).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Detectar soportes/resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Determinar tendencia
        trend_up = last['ema_fast'] > last['ema_slow'] and last['close'] > last['ema_slow']
        trend_down = last['ema_fast'] < last['ema_slow'] and last['close'] < last['ema_slow']
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if trend_up: 
            long_prob += 35
        if last['adx'] > params['adx_level']:
            long_prob += 20
        if last['rsi'] < 45 and last['rsi'] > prev['rsi']:
            long_prob += 15
        if detect_divergence(df) == 'bullish':
            long_prob += 20
        if volume_class in ['Alto', 'Muy Alto']:
            long_prob += 10
            
        # Factores para SHORT
        if trend_down: 
            short_prob += 35
        if last['adx'] > params['adx_level']:
            short_prob += 20
        if last['rsi'] > 55 and last['rsi'] < prev['rsi']:
            short_prob += 15
        if detect_divergence(df) == 'bearish':
            short_prob += 20
        if volume_class in ['Alto', 'Muy Alto']:
            short_prob += 10
            
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
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto'] and trend_up:
            # Encontrar niveles de entrada y stop loss
            entry = last['close'] * 1.005
            sl = min(supports[-3:]) if supports else entry * 0.98
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 6),
                'sl': round(sl, 6),
                'tp1': round(entry + risk, 6),
                'tp2': round(entry + risk * 2, 6),
                'volume': volume_class,
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 6),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto'] and trend_down:
            # Encontrar niveles de entrada y stop loss
            entry = last['close'] * 0.995
            sl = max(resistances[-3:]) if resistances else entry * 1.02
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 6),
                'sl': round(sl, 6),
                'tp1': round(entry - risk, 6),
                'tp2': round(entry - risk * 2, 6),
                'volume': volume_class,
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 6),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}")
        traceback.print_exc()
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        try:
            app.logger.info("Starting background update...")
            start_time = time.time()
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            processed = 0
            
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
                    
                    processed += 1
                    if processed % 10 == 0:
                        app.logger.info(f"Processed {processed}/{len(cryptos)} cryptos")
                    
                    time.sleep(0.1)  # Evitar sobrecarga
                    
                except Exception as e:
                    app.logger.error(f"Error processing {crypto}: {str(e)}")
            
            # Ordenar señales
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar caché
            with app.app_context():
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
            
            elapsed = time.time() - start_time
            app.logger.info(f"Update completed in {elapsed:.2f}s: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} points")
        except Exception as e:
            app.logger.error(f"Background update error: {str(e)}")
            traceback.print_exc()
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
try:
    app.logger.info("Starting update thread...")
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
except Exception as e:
    app.logger.error(f"Failed to start update thread: {str(e)}")

@app.route('/')
def index():
    try:
        long_signals = cache.get('long_signals') or []
        short_signals = cache.get('short_signals') or []
        scatter_data = cache.get('scatter_data') or []
        last_update = cache.get('last_update') or datetime.now()
        
        # Estadísticas
        avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
        avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
        
        return render_template('index.html', 
                              long_signals=long_signals[:50], 
                              short_signals=short_signals[:50],
                              last_update=last_update,
                              params=DEFAULTS,
                              avg_adx_long=round(avg_adx_long, 2),
                              avg_adx_short=round(avg_adx_short, 2),
                              scatter_data=json.dumps(scatter_data))
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html', message="Error loading data")

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        df = get_binance_data(symbol, DEFAULTS['timeframe'])
        if df is None:
            return "Data not available", 404
        
        # Buscar señal
        signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Signal not found", 404
        
        # Generar gráfico
        plt.figure(figsize=(12, 8))
        plt.plot(df['timestamp'], df['close'], label='Price')
        
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='g', linestyle='--', label='Entry')
            plt.axhline(y=signal['sl'], color='r', linestyle='--', label='Stop Loss')
        else:
            plt.axhline(y=signal['entry'], color='r', linestyle='--', label='Entry')
            plt.axhline(y=signal['sl'], color='g', linestyle='--', label='Stop Loss')
        
        plt.title(f"{symbol} - {signal_type.upper()} Signal")
        plt.legend()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        app.logger.error(f"Chart error: {str(e)}")
        return "Error generating chart", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
