import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

app = Flask(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 900  # 15 minutos
DEFAULTS = {
    'timeframe': '4h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 50
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
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['code'] == '200000':
            candles = data['data']
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df.sort_values('timestamp')
    return None

# Calcular indicadores
def calculate_indicators(df, params):
    # EMAs
    ema_fast = EMAIndicator(df['close'], window=params['ema_fast'])
    df['ema_fast'] = ema_fast.ema_indicator()
    
    ema_slow = EMAIndicator(df['close'], window=params['ema_slow'])
    df['ema_slow'] = ema_slow.ema_indicator()
    
    # ADX
    adx_ind = ADXIndicator(df['high'], df['low'], df['close'], window=params['adx_period'])
    df['adx'] = adx_ind.adx()
    
    # RSI
    rsi = RSIIndicator(df['close'], window=params['rsi_period'])
    df['rsi'] = rsi.rsi()
    
    # ATR
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    
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
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=20):
    last_close = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    # Buscar máximos/mínimos recientes
    max_idx = df['close'].tail(lookback).idxmax()
    min_idx = df['close'].tail(lookback).idxmin()
    
    # Divergencia bajista
    if max_idx != -1:
        prev_high = df['close'].iloc[max_idx-1] if max_idx > 0 else 0
        if last_close > df['close'].iloc[max_idx] and last_rsi < df['rsi'].iloc[max_idx] and last_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx != -1:
        prev_low = df['close'].iloc[min_idx-1] if min_idx > 0 else 0
        if last_close < df['close'].iloc[min_idx] and last_rsi > df['rsi'].iloc[min_idx] and last_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(f"{symbol}-USDT", params['timeframe'])
    if df is None or len(df) < 100:
        return None, None
    
    df = calculate_indicators(df, params)
    supports, resistances = find_support_resistance(df, params['sr_window'])
    
    last = df.iloc[-1]
    avg_vol = df['volume'].tail(20).mean()
    volume_class = classify_volume(last['volume'], avg_vol)
    divergence = detect_divergence(df)
    
    # Detectar quiebres
    is_breakout = any(last['close'] > r * 1.005 for r in resistances)
    is_breakdown = any(last['close'] < s * 0.995 for s in supports)
    
    # Determinar tendencia
    trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
    
    # Señales LONG
    long_signal = None
    if (trend == 'up' and last['adx'] > params['adx_level'] and 
        (is_breakout or divergence == 'bullish') and 
        volume_class in ['Alto', 'Muy Alto']):
        
        entry = min(r for r in resistances if r > last['close']) * 1.005 if resistances else last['close'] * 1.01
        sl = max(s for s in supports if s < entry) * 0.995 if supports else entry * 0.98
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
            'adx': round(last['adx'], 2)
        }
    
    # Señales SHORT
    short_signal = None
    if (trend == 'down' and last['adx'] > params['adx_level'] and 
        (is_breakdown or divergence == 'bearish') and 
        volume_class in ['Alto', 'Muy Alto']):
        
        entry = max(s for s in supports if s < last['close']) * 0.995 if supports else last['close'] * 0.99
        sl = min(r for r in resistances if r > entry) * 1.005 if resistances else entry * 1.02
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
            'adx': round(last['adx'], 2)
        }
    
    return long_signal, short_signal

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                params[key] = int(request.form[key])
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    for crypto in cryptos[:20]:  # Limitar a 20 por recursos
        long_signal, short_signal = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
        if short_signal:
            short_signals.append(short_signal)
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
