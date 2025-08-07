import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from datetime import datetime, timedelta

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
        if data['code'] == '200000' and 'data' in data and data['data']:
            candles = data['data']
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            return df.sort_values('timestamp')
    return None

# Calcular EMA manualmente
def calculate_ema(series, window):
    alpha = 2 / (window + 1)
    ema = [series.iloc[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series.iloc[i] + (1 - alpha) * ema[i-1])
    return pd.Series(ema, index=series.index)

# Calcular ADX manualmente
def calculate_adx(high, low, close, window):
    # Calcular +DM y -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Suavizar con EMA
    plus_dm_smooth = calculate_ema(pd.Series(plus_dm), window)
    minus_dm_smooth = calculate_ema(pd.Series(minus_dm), window)
    
    # Calcular True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = calculate_ema(tr, window)
    
    # Calcular los índices direccionales
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # Calcular el DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = calculate_ema(dx, window)
    
    return adx, plus_di, minus_di

# Calcular RSI manualmente
def calculate_rsi(series, window):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Para los primeros valores donde no hay suficientes datos
    for i in range(window, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window - 1) + loss.iloc[i]) / window
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# Calcular ATR manualmente
def calculate_atr(high, low, close, window):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

# Calcular indicadores
def calculate_indicators(df, params):
    # EMAs
    df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
    df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
    
    # ADX
    df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
    
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
        return 'Muy Alto' if current_vol > 0 else 'Muy Bajo'
    
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback):
    if len(df) < lookback:
        return None
        
    last_close = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    # Buscar máximos/mínimos recientes
    max_idx = df['close'].tail(lookback).idxmax()
    min_idx = df['close'].tail(lookback).idxmin()
    
    # Divergencia bajista
    if not pd.isna(max_idx):
        prev_high = df['close'].iloc[max_idx]
        prev_rsi = df['rsi'].iloc[max_idx]
        
        if last_close > prev_high and last_rsi < prev_rsi and last_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if not pd.isna(min_idx):
        prev_low = df['close'].iloc[min_idx]
        prev_rsi = df['rsi'].iloc[min_idx]
        
        if last_close < prev_low and last_rsi > prev_rsi and last_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None
    
    df = calculate_indicators(df, params)
    supports, resistances = find_support_resistance(df, params['sr_window'])
    
    last = df.iloc[-1]
    avg_vol = df['volume'].tail(20).mean()
    volume_class = classify_volume(last['volume'], avg_vol)
    divergence = detect_divergence(df, params['div_lookback'])
    
    # Detectar quiebres
    is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
    is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
    
    # Determinar tendencia
    trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
    
    # Señales LONG
    long_signal = None
    if (trend == 'up' and last['adx'] > params['adx_level'] and 
        (is_breakout or divergence == 'bullish') and 
        volume_class in ['Alto', 'Muy Alto']):
        
        # Buscar la resistencia más cercana por encima
        next_resistances = [r for r in resistances if r > last['close']]
        if next_resistances:
            entry = min(next_resistances) * 1.005
        else:
            entry = last['close'] * 1.01
            
        # Buscar el soporte más cercano por debajo para SL
        next_supports = [s for s in supports if s < entry]
        if next_supports:
            sl = max(next_supports) * 0.995
        else:
            sl = entry * 0.98
            
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
        
        # Buscar el soporte más cercano por debajo
        next_supports = [s for s in supports if s < last['close']]
        if next_supports:
            entry = max(next_supports) * 0.995
        else:
            entry = last['close'] * 0.99
            
        # Buscar la resistencia más cercana por encima para SL
        next_resistances = [r for r in resistances if r > entry]
        if next_resistances:
            sl = min(next_resistances) * 1.005
        else:
            sl = entry * 1.02
            
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
                # Convertir a int excepto para timeframe
                if key == 'timeframe':
                    params[key] = request.form[key]
                else:
                    params[key] = int(request.form[key])
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 20 criptos por recursos (en Render gratuito)
    for crypto in cryptos[:20]:
        try:
            long_signal, short_signal = analyze_crypto(crypto, params)
            if long_signal:
                long_signals.append(long_signal)
            if short_signal:
                short_signals.append(short_signal)
        except Exception as e:
            app.logger.error(f"Error analyzing {crypto}: {str(e)}")
    
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
