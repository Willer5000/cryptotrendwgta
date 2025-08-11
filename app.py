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

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    'price_distance_threshold': 1.0  # 1% para considerar "cerca" de soporte/resistencia
}

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(cryptos)} cryptocurrencies")
        return cryptos
    except Exception as e:
        logger.error(f"Error loading cryptos: {str(e)}")
        return []

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
                    # KuCoin devuelve las velas en orden descendente, invertimos para tener ascendente
                    candles.reverse()
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    # Verificar si tenemos datos suficientes
                    if len(df) < 50:
                        logger.warning(f"Insufficient data for {symbol}-USDT: only {len(df)} records")
                        continue
                        
                    df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                    # Convertir timestamp a entero para evitar el warning
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    logger.info(f"Successfully fetched data for {symbol}-USDT: {len(df)} records")
                    return df
            else:
                logger.warning(f"API error for {symbol}-USDT: Status {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}-USDT (attempt {attempt+1}): {str(e)}")
            time.sleep(2)  # Esperar antes de reintentar
    
    logger.error(f"Failed to fetch data for {symbol}-USDT after {retries} attempts")
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX simplificada
def calculate_adx(high, low, close, window):
    # Cálculo simplificado para evitar errores de memoria
    up_move = high.diff()
    down_move = low.diff().abs()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / atr
    minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / atr
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di

# Implementación manual de ATR
def calculate_atr(high, low, close, window):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    # EMA
    df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
    df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
    
    # ADX
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
    
    # ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    return df

# Detectar soportes y resistencias simplificado
def find_support_resistance(df, window):
    # Enfoque simplificado para ahorrar recursos
    highs = df['high'].values
    lows = df['low'].values
    
    supports = []
    resistances = []
    
    # Identificar máximos locales para resistencias
    for i in range(window, len(highs)-window):
        if highs[i] == max(highs[i-window:i+window]):
            resistances.append(highs[i])
    
    # Identificar mínimos locales para soportes
    for i in range(window, len(lows)-window):
        if lows[i] == min(lows[i-window:i+window]):
            supports.append(lows[i])
    
    # Eliminar duplicados y valores extremos
    supports = sorted(set(supports))
    resistances = sorted(set(resistances))
    
    return supports, resistances

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol <= 0:
        return 'Muy Bajo'
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias simplificado
def detect_divergence(df, lookback):
    if len(df) < lookback + 1:
        return None
    
    # Tomar los últimos datos
    prices = df['close'].values[-lookback:]
    rsi = df['rsi'].values[-lookback:]
    
    # Encontrar máximos y mínimos
    price_high_idx = np.argmax(prices)
    price_low_idx = np.argmin(prices)
    
    # Divergencia bajista: precio hace nuevo alto pero RSI no
    if price_high_idx > 0 and price_high_idx < len(prices)-1:
        if prices[-1] > prices[price_high_idx] and rsi[-1] < rsi[price_high_idx]:
            return 'bearish'
    
    # Divergencia alcista: precio hace nuevo bajo pero RSI no
    if price_low_idx > 0 and price_low_idx < len(prices)-1:
        if prices[-1] < prices[price_low_idx] and rsi[-1] > rsi[price_low_idx]:
            return 'bullish'
    
    return None

# Calcular distancia al nivel más cercano
def calculate_distance_to_level(price, levels, threshold_percent):
    if not levels or price <= 0:
        return False
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    return min_distance <= threshold

# Generar gráfico para una señal
def generate_chart(df, signal, signal_type):
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
    
    return plot_url

# Analizar una criptomoneda y calcular probabilidades
def analyze_crypto(symbol, params):
    logger.info(f"Analyzing {symbol}-USDT")
    
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        logger.warning(f"Insufficient data for {symbol}-USDT")
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        if pd.isna(avg_vol) or avg_vol <= 0:
            avg_vol = df['volume'].mean()
            
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres
        is_breakout = False
        is_breakdown = False
        
        if resistances:
            is_breakout = last['close'] > max(resistances) * 1.005
        if supports:
            is_breakdown = last['close'] < min(supports) * 0.995
        
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
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2) if last['close'] > 0 else 0
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
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2) if last['close'] > 0 else 0
            }
        
        logger.info(f"Analysis for {symbol}-USDT completed: LONG {long_prob}%, SHORT {short_prob}%")
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analyzing {symbol}-USDT: {str(e)}", exc_info=True)
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        try:
            logger.info("Starting background data update...")
            start_time = time.time()
            
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            
            for i, crypto in enumerate(cryptos):
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
                
                # Progreso cada 10 criptos
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(cryptos)} cryptos")
            
            # Ordenar por fuerza de tendencia (ADX)
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar caché
            cache.set('long_signals', long_signals)
            cache.set('short_signals', short_signals)
            cache.set('scatter_data', scatter_data)
            cache.set('last_update', datetime.now())
            
            elapsed = time.time() - start_time
            logger.info(f"Update completed in {elapsed:.2f}s: {len(long_signals)} LONG, {len(short_signals)} SHORT")
        except Exception as e:
            logger.error(f"Background update error: {str(e)}", exc_info=True)
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()

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
                               long_signals=long_signals, 
                               short_signals=short_signals,
                               last_update=last_update,
                               params=DEFAULTS,
                               signal_count=signal_count,
                               avg_adx_long=round(avg_adx_long, 2),
                               avg_adx_short=round(avg_adx_short, 2),
                               scatter_data=json.dumps(scatter_data))
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Error loading data. Please try again later.")

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
        if df is None:
            return render_template('error.html', message="Data not available for this cryptocurrency")
        
        df = calculate_indicators(df, DEFAULTS)
        
        # Buscar señal correspondiente
        signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
        if signals is None:
            return render_template('error.html', message="Signals not loaded yet")
            
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return render_template('error.html', message="Signal not found")
        
        plot_url = generate_chart(df, signal, signal_type)
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error in chart route: {str(e)}", exc_info=True)
        return render_template('error.html', message="Error generating chart")

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/update')
def manual_update():
    # Forzar actualización manual
    cache.clear()
    background_update()
    return "Manual update triggered", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
