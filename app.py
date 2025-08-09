import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import threading
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_FILE = 'cache.json'
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
    'candle_count': 100
}

# Configurar logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Obtener datos de KuCoin
def get_kucoin_data(symbol, timeframe, candle_count=100):
    tf_mapping = {
        '30m': '30min',
        '1h': '1hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': '1day',
        '1w': '1week'
    }
    kucoin_tf = tf_mapping[timeframe]
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('code') == '200000' and data.get('data'):
            candles = data['data']
            # KuCoin devuelve las velas en orden descendente, invertimos para tener ascendente
            candles.reverse()
            if len(candles) > candle_count:
                candles = candles[-candle_count:]
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
            # Convertir timestamp a entero para evitar el warning
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            return df
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    # Evitar división por cero
    avg_loss = avg_loss.replace(0, 1e-10)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm <= 0] = 0
    minus_dm[minus_dm <= 0] = 0
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window).mean()
    return adx

# Implementación manual de ATR
def calculate_atr(high, low, close, window):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
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
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
    
    # ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    return df

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df['high_roll'] = df['high'].rolling(window=window).max()
    df['low_roll'] = df['low'].rolling(window=window).min()
    
    resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
    
    return supports, resistances

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol == 0:
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
        
    # Seleccionar los últimos datos
    recent = df.iloc[-lookback:]
    
    # Buscar máximos/mínimos recientes
    max_idx = recent['close'].idxmax()
    min_idx = recent['close'].idxmin()
    
    # Divergencia bajista
    if not pd.isna(max_idx):
        price_high = df.loc[max_idx, 'close']
        rsi_high = df.loc[max_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if not pd.isna(min_idx):
        price_low = df.loc[min_idx, 'close']
        rsi_low = df.loc[min_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'], params['candle_count'])
        if df is None or len(df) < 100:
            app.logger.warning(f"No hay suficientes datos para {symbol}")
            return None, None, None
        
        df = calculate_indicators(df, params)
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Preparar datos para el gráfico
        chart_data = {
            'dates': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'close': df['close'].tolist(),
            'ema_fast': df['ema_fast'].tolist(),
            'ema_slow': df['ema_slow'].tolist(),
            'supports': supports,
            'resistances': resistances
        }
        
        # Señales LONG
        long_signal = None
        if (trend == 'up' and last['adx'] > params['adx_level'] and 
            (is_breakout or divergence == 'bullish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar la resistencia más cercana por encima
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            # Encontrar el soporte más cercano por debajo para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * 0.98
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
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar el soporte más cercano por debajo
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            # Encontrar la resistencia más cercana por encima para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * 1.02
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
        
        return long_signal, short_signal, chart_data
    except Exception as e:
        app.logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, None

# Función para actualizar la caché en segundo plano
def update_cache_background(params):
    app.logger.info("Iniciando actualización de caché en segundo plano...")
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    chart_data = {}
    
    for i, crypto in enumerate(cryptos):
        long_signal, short_signal, crypto_chart = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
            chart_data[crypto] = crypto_chart
        if short_signal:
            short_signals.append(short_signal)
            chart_data[crypto] = crypto_chart
        
        # Registrar progreso cada 10 criptos
        if (i + 1) % 10 == 0:
            app.logger.info(f"Procesadas {i+1}/{len(cryptos)} criptomonedas")
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    cache = {
        'last_updated': datetime.utcnow().timestamp(),
        'long_signals': long_signals,
        'short_signals': short_signals,
        'chart_data': chart_data
    }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    
    app.logger.info("Caché actualizada exitosamente")
    return cache

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    cache = None
    cache_age = float('inf')
    
    # Cargar caché si existe
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            cache_age = datetime.utcnow().timestamp() - cache['last_updated']
    
    # Si la caché está vacía o es muy antigua, iniciar actualización en segundo plano
    if cache is None or cache_age > CACHE_TIME:
        app.logger.info("Iniciando actualización de caché en segundo plano")
        thread = threading.Thread(target=update_cache_background, args=(params,))
        thread.start()
        
        # Si no hay caché, crear una vacía temporalmente
        if cache is None:
            cache = {
                'long_signals': [],
                'short_signals': [],
                'chart_data': {},
                'last_updated': 0
            }
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                # Convertir a int si es numérico, de lo contrario dejar como está
                if key != 'timeframe':
                    params[key] = int(request.form[key])
                else:
                    params[key] = request.form[key]
    
    # Estadísticas para gráficos adicionales
    signal_count = len(cache['long_signals']) + len(cache['short_signals'])
    avg_adx_long = np.mean([s['adx'] for s in cache['long_signals']]) if cache['long_signals'] else 0
    avg_adx_short = np.mean([s['adx'] for s in cache['short_signals']]) if cache['short_signals'] else 0
    
    return render_template('index.html', 
                           long_signals=cache['long_signals'], 
                           short_signals=cache['short_signals'],
                           params=params,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           chart_data=json.dumps(cache['chart_data']),
                           last_updated=datetime.utcfromtimestamp(cache['last_updated']).strftime('%Y-%m-%d %H:%M:%S UTC'))

@app.route('/chart/<symbol>')
def get_chart_data(symbol):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            if symbol in cache['chart_data']:
                return jsonify(cache['chart_data'][symbol])
    return jsonify({})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
