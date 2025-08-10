import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Desactivar caché para desarrollo
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
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        app.logger.error(f"Error loading cryptos: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC']

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
    
    if timeframe not in tf_mapping:
        app.logger.error(f"Invalid timeframe: {timeframe}")
        return None
        
    kucoin_tf = tf_mapping[timeframe]
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                if not candles:
                    app.logger.warning(f"No data for {symbol}-USDT on {timeframe}")
                    return None
                    
                candles.reverse()
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir tipos de datos
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Eliminar filas con valores nulos
                df = df.dropna()
                
                if len(df) < 50:
                    app.logger.warning(f"Insufficient data for {symbol}-USDT: {len(df)} rows")
                    return None
                
                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
        else:
            app.logger.error(f"API error for {symbol}: {response.status_code}")
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    return None

# Implementación manual de EMA
def calculate_ema(prices, window):
    if len(prices) < window:
        return [np.nan] * len(prices)
    
    ema = []
    k = 2 / (window + 1)
    current_ema = prices[0]
    
    for price in prices:
        current_ema = price * k + current_ema * (1 - k)
        ema.append(current_ema)
    
    return ema

# Implementación manual de RSI
def calculate_rsi(prices, window):
    if len(prices) < window + 1:
        return [np.nan] * len(prices)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    rsi = [np.nan] * window
    for i in range(window, len(prices)-1):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        
        if avg_loss == 0:
            rs = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            
        rsi.append(100 - (100 / (1 + rs)))
    
    return rsi

# Implementación manual de ADX
def calculate_adx(highs, lows, closes, window):
    n = len(highs)
    if n < window * 2:
        return [np.nan] * n, [np.nan] * n, [np.nan] * n
    
    # Calcular +DM y -DM
    plus_dm = [0] * n
    minus_dm = [0] * n
    
    for i in range(1, n):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Calcular TR
    tr = [0] * n
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    
    # Suavizar los valores
    plus_di = [np.nan] * n
    minus_di = [np.nan] * n
    dx = [np.nan] * n
    adx = [np.nan] * n
    
    # Primeros valores
    plus_di[window] = 100 * (sum(plus_dm[1:window+1]) / sum(tr[1:window+1]))
    minus_di[window] = 100 * (sum(minus_dm[1:window+1]) / sum(tr[1:window+1]))
    
    for i in range(window+1, n):
        plus_di[i] = (plus_di[i-1] * (window - 1) + plus_dm[i]) / window
        minus_di[i] = (minus_di[i-1] * (window - 1) + minus_dm[i]) / window
        
        di_diff = abs(plus_di[i] - minus_di[i])
        di_sum = plus_di[i] + minus_di[i]
        
        if di_sum > 0:
            dx[i] = 100 * (di_diff / di_sum)
    
    # Calcular ADX
    adx[window*2-1] = np.mean(dx[window:window*2])
    
    for i in range(window*2, n):
        adx[i] = (adx[i-1] * (window - 1) + dx[i]) / window
    
    return adx, plus_di, minus_di

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # EMA
    df['ema_fast'] = calculate_ema(prices, params['ema_fast'])
    df['ema_slow'] = calculate_ema(prices, params['ema_slow'])
    
    # RSI
    df['rsi'] = calculate_rsi(prices, params['rsi_period'])
    
    # ADX
    adx, plus_di, minus_di = calculate_adx(highs, lows, prices, params['adx_period'])
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # Eliminar filas con NaN
    df = df.dropna()
    
    return df

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    highs = df['high'].values
    lows = df['low'].values
    
    resistances = []
    supports = []
    
    for i in range(window, len(df)-window):
        # Resistencia: máximo local
        if highs[i] == max(highs[i-window:i+window]):
            resistances.append(highs[i])
        
        # Soporte: mínimo local
        if lows[i] == min(lows[i-window:i+window]):
            supports.append(lows[i])
    
    return list(set(supports)), list(set(resistances))

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol <= 0 or current_vol <= 0:
        return 'Muy Bajo'
        
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback):
    if len(df) < lookback + 1:
        return None
        
    prices = df['close'].values
    rsi = df['rsi'].values
    
    # Buscar máximos/mínimos recientes
    recent_prices = prices[-lookback:]
    recent_rsi = rsi[-lookback:]
    
    price_high_idx = np.argmax(recent_prices)
    price_low_idx = np.argmin(recent_prices)
    
    # Divergencia bajista
    if price_high_idx > 0 and price_high_idx < len(recent_prices) - 1:
        current_price = prices[-1]
        current_rsi = rsi[-1]
        price_high = recent_prices[price_high_idx]
        rsi_high = recent_rsi[price_high_idx]
        
        if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if price_low_idx > 0 and price_low_idx < len(recent_prices) - 1:
        current_price = prices[-1]
        current_rsi = rsi[-1]
        price_low = recent_prices[price_low_idx]
        rsi_low = recent_rsi[price_low_idx]
        
        if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
            return 'bullish'
    
    return None

# Calcular distancia al nivel más cercano
def calculate_distance_to_level(price, levels, threshold_percent):
    if not levels or price <= 0:
        return False
        
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    return min_distance <= threshold

# Analizar una criptomoneda y calcular probabilidades
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        app.logger.warning(f"Insufficient data for {symbol}")
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df.empty:
            return None, None, 0, 0, 'Muy Bajo'
            
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Obtener soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Detectar quiebres
        is_breakout = False
        is_breakdown = False
        
        if resistances:
            is_breakout = any(last['close'] > r * 1.005 for r in resistances)
        
        if supports:
            is_breakdown = any(last['close'] < s * 0.995 for s in supports)
        
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
        
        # Señales LONG
        long_signal = None
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar la resistencia más cercana por encima
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            # Encontrar el soporte más cercano por debajo para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * (1 - params['max_risk_percent']/100)
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'tp3': round(entry + risk * 3, 4),
                'volume': volume_class,
                'divergence': divergence == 'bullish',
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 4),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar el soporte más cercano por debajo
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            # Encontrar la resistencia más cercana por encima para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * (1 + params['max_risk_percent']/100)
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'tp3': round(entry - risk * 3, 4),
                'volume': volume_class,
                'divergence': divergence == 'bearish',
                'adx': round(last['adx'], 2),
                'price': round(last['close'], 4),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        with app.app_context():
            try:
                app.logger.info("Iniciando actualización de datos...")
                cryptos = load_cryptos()
                long_signals = []
                short_signals = []
                scatter_data = []
                
                for crypto in cryptos:
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
                
                # Ordenar por fuerza de tendencia (ADX)
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar caché
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                
                app.logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT")
            except Exception as e:
                app.logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()

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
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=json.dumps(scatter_data))

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Datos no disponibles", 404
    
    df = calculate_indicators(df, DEFAULTS)
    
    # Buscar señal correspondiente
    signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
    signal = next((s for s in signals if s['symbol'] == symbol), None) if signals else None
    
    if not signal:
        return "Señal no encontrada", 404
    
    # Generar gráfico
    plt.figure(figsize=(12, 8))
    
    # Gráfico de precio
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='Precio', color='blue')
    plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', alpha=0.7)
    plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', alpha=0.7)
    
    # Marcar entrada y SL
    if signal_type == 'long':
        plt.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada')
        plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
    else:
        plt.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada')
        plt.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss')
    
    plt.title(f'{symbol}-USDT - Precio y EMAs')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de volumen
    plt.subplot(3, 1, 2)
    plt.bar(df['timestamp'], df['volume'], color=np.where(df['close'] > df['open'], 'green', 'red'))
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
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
