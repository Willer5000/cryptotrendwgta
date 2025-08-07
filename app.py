import os
import time
import requests
import pandas as pd
import numpy as np
import math
from flask import Flask, render_template, request
from datetime import datetime, timedelta

app = Flask(__name__)

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
    'div_lookback': 20
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]

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
    kucoin_tf = tf_mapping[timeframe]
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('code') == '200000' and data.get('data'):
            candles = data['data']
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            return df.sort_values('timestamp')
    return None

# Cálculo manual de EMA
def calculate_ema(prices, window):
    ema = [prices[0]]
    multiplier = 2 / (window + 1)
    
    for i in range(1, len(prices)):
        ema_val = (prices[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_val)
    
    return ema

# Cálculo manual de RSI
def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = [100 - (100 / (1 + rs))]
    
    for i in range(window, len(deltas)):
        delta = deltas[i]
        if delta > 0:
            up_val = delta
            down_val = 0
        else:
            up_val = 0
            down_val = -delta
            
        up = (up * (window - 1) + up_val) / window
        down = (down * (window - 1) + down_val) / window
        rs = up / down
        rsi.append(100 - (100 / (1 + rs)))
    
    # Rellenar los primeros valores con NaN
    prefix = [np.nan] * (window)
    return prefix + rsi

# Cálculo manual de ADX
def calculate_adx(high, low, close, window=14):
    # Calcular movimiento direccional
    plus_dm = []
    minus_dm = []
    tr = []
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)
            
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)
            
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    
    # Suavizar valores
    plus_dm_smooth = [sum(plus_dm[:window]) / window]
    minus_dm_smooth = [sum(minus_dm[:window]) / window]
    tr_smooth = [sum(tr[:window]) / window]
    
    for i in range(window, len(plus_dm)):
        plus_dm_smooth.append((plus_dm_smooth[-1] * (window - 1) + plus_dm[i]) / window)
        minus_dm_smooth.append((minus_dm_smooth[-1] * (window - 1) + minus_dm[i]) / window)
        tr_smooth.append((tr_smooth[-1] * (window - 1) + tr[i]) / window)
    
    # Calcular indicadores direccionales
    plus_di = [100 * (p / t) for p, t in zip(plus_dm_smooth, tr_smooth)]
    minus_di = [100 * (m / t) for m, t in zip(minus_dm_smooth, tr_smooth)]
    
    # Calcular DX
    dx = [100 * abs(p - m) / (p + m) for p, m in zip(plus_di, minus_di)]
    
    # Calcular ADX
    adx = [sum(dx[:window]) / window]
    for i in range(window, len(dx)):
        adx.append((adx[-1] * (window - 1) + dx[i]) / window)
    
    # Rellenar los primeros valores con NaN
    prefix = [np.nan] * (window + 1)
    return prefix + adx

# Cálculo manual de ATR
def calculate_atr(high, low, close, window=14):
    tr = [max(high[0] - low[0], abs(high[0] - close[0]), abs(low[0] - close[0]))]
    
    for i in range(1, len(high)):
        tr_val = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
        tr.append(tr_val)
    
    atr = [sum(tr[:window]) / window]
    
    for i in range(window, len(tr)):
        atr_val = (atr[-1] * (window - 1) + tr[i]) / window
        atr.append(atr_val)
    
    # Rellenar los primeros valores con NaN
    prefix = [np.nan] * (window - 1)
    return prefix + atr

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    high_roll = df['high'].rolling(window=window).max()
    low_roll = df['low'].rolling(window=window).min()
    
    resistances = df[df['high'] == high_roll]['high'].unique().tolist()
    supports = df[df['low'] == low_roll]['low'].unique().tolist()
    
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
def detect_divergence(df, lookback=20):
    if len(df) < lookback + 5:
        return None
    
    # Tomar los últimos datos
    close_prices = df['close'].values[-lookback:]
    rsi_values = df['rsi'].values[-lookback:]
    
    # Encontrar máximos y mínimos en precios
    price_high_idx = np.argmax(close_prices)
    price_low_idx = np.argmin(close_prices)
    
    # Encontrar máximos y mínimos en RSI
    rsi_high_idx = np.argmax(rsi_values)
    rsi_low_idx = np.argmin(rsi_values)
    
    # Divergencia bajista
    if (price_high_idx < len(close_prices) - 2 and 
        rsi_high_idx < len(rsi_values) - 2 and
        close_prices[-1] > close_prices[price_high_idx] and
        rsi_values[-1] < rsi_values[rsi_high_idx] and
        rsi_values[-1] > 70):
        return 'bearish'
    
    # Divergencia alcista
    if (price_low_idx < len(close_prices) - 2 and 
        rsi_low_idx < len(rsi_values) - 2 and
        close_prices[-1] < close_prices[price_low_idx] and
        rsi_values[-1] > rsi_values[rsi_low_idx] and
        rsi_values[-1] < 30):
        return 'bullish'
    
    return None

# Calcular todos los indicadores
def calculate_indicators(df, params):
    # Calcular EMAs
    df['ema_fast'] = calculate_ema(df['close'].tolist(), params['ema_fast'])
    df['ema_slow'] = calculate_ema(df['close'].tolist(), params['ema_slow'])
    
    # Calcular RSI
    df['rsi'] = calculate_rsi(df['close'].tolist(), params['rsi_period'])
    
    # Calcular ADX
    df['adx'] = calculate_adx(
        df['high'].tolist(), 
        df['low'].tolist(), 
        df['close'].tolist(), 
        params['adx_period']
    )
    
    # Calcular ATR
    df['atr'] = calculate_atr(
        df['high'].tolist(), 
        df['low'].tolist(), 
        df['close'].tolist(), 
        14
    )
    
    return df

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return None, None
        
        df = calculate_indicators(df, params)
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['div_lookback'])
        
        # Filtrar valores NaN
        if any(pd.isna([last['ema_fast'], last['ema_slow'], last['adx'], last['rsi']])):
            return None, None
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
        
        # Señales LONG
        long_signal = None
        if (trend == 'up' and last['adx'] > params['adx_level'] and 
            (is_breakout or divergence == 'bullish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Calcular entrada y SL basados en soportes/resistencias
            if supports:
                support_below = max([s for s in supports if s < last['close']], default=None)
                if support_below:
                    entry = last['close'] * 1.002  # Entrada ligeramente arriba del precio actual
                    sl = support_below * 0.995
                    risk = entry - sl
                    
                    # Calcular objetivos de ganancia
                    if resistances:
                        next_resistance = min([r for r in resistances if r > entry], default=entry + risk * 3)
                        tp1 = min(entry + risk, next_resistance * 0.98)
                        tp2 = min(entry + risk * 2, next_resistance * 0.99) if next_resistance > tp1 else tp1 * 1.5
                        tp3 = next_resistance
                    else:
                        tp1 = entry + risk
                        tp2 = entry + risk * 2
                        tp3 = entry + risk * 3
                    
                    long_signal = {
                        'symbol': symbol,
                        'entry': round(entry, 4),
                        'sl': round(sl, 4),
                        'tp1': round(tp1, 4),
                        'tp2': round(tp2, 4),
                        'tp3': round(tp3, 4),
                        'volume': volume_class,
                        'divergence': divergence == 'bullish',
                        'adx': round(last['adx'], 2),
                        'rsi': round(last['rsi'], 2)
                    }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Calcular entrada y SL basados en soportes/resistencias
            if resistances:
                resistance_above = min([r for r in resistances if r > last['close']], default=None)
                if resistance_above:
                    entry = last['close'] * 0.998  # Entrada ligeramente debajo del precio actual
                    sl = resistance_above * 1.005
                    risk = sl - entry
                    
                    # Calcular objetivos de ganancia
                    if supports:
                        next_support = max([s for s in supports if s < entry], default=entry - risk * 3)
                        tp1 = max(entry - risk, next_support * 1.02)
                        tp2 = max(entry - risk * 2, next_support * 1.01) if next_support < tp1 else tp1 * 0.5
                        tp3 = next_support
                    else:
                        tp1 = entry - risk
                        tp2 = entry - risk * 2
                        tp3 = entry - risk * 3
                    
                    short_signal = {
                        'symbol': symbol,
                        'entry': round(entry, 4),
                        'sl': round(sl, 4),
                        'tp1': round(tp1, 4),
                        'tp2': round(tp2, 4),
                        'tp3': round(tp3, 4),
                        'volume': volume_class,
                        'divergence': divergence == 'bearish',
                        'adx': round(last['adx'], 2),
                        'rsi': round(last['rsi'], 2)
                    }
        
        return long_signal, short_signal
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                try:
                    # Convertir a número excepto para temporalidad
                    if key != 'timeframe':
                        params[key] = int(request.form[key])
                    else:
                        params[key] = request.form[key]
                except:
                    pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Analizar las primeras 20 criptos para ahorrar recursos
    for crypto in cryptos[:20]:
        long_signal, short_signal = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
        if short_signal:
            short_signals.append(short_signal)
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    # Preparar datos para gráficos adicionales
    market_status = {
        'long_count': len(long_signals),
        'short_count': len(short_signals),
        'neutral_count': 20 - len(long_signals) - len(short_signals)
    }
    
    # Gráfico de fuerza de tendencia
    trend_strength = {
        'labels': [sig['symbol'] for sig in long_signals[:5]] + [sig['symbol'] for sig in short_signals[:5]],
        'long_adx': [sig['adx'] for sig in long_signals[:5]],
        'short_adx': [sig['adx'] for sig in short_signals[:5]]
    }
    
    # Gráfico de RSI
    rsi_data = {
        'labels': [sig['symbol'] for sig in long_signals[:5]] + [sig['symbol'] for sig in short_signals[:5]],
        'long_rsi': [sig['rsi'] for sig in long_signals[:5]],
        'short_rsi': [sig['rsi'] for sig in short_signals[:5]]
    }
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           market_status=market_status,
                           trend_strength=trend_strength,
                           rsi_data=rsi_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
