import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import json
from datetime import datetime

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
    'max_cryptos': 30  # Límite para Render gratuito
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        cryptos = [line.strip() for line in f.readlines()]
    return cryptos

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
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['code'] == '200000' and data['data']:
                candles = data['data']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir tipos de datos
                numeric_cols = ['open', 'close', 'high', 'low', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # Convertir timestamp correctamente
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                
                return df.sort_values('timestamp')
    except Exception as e:
        print(f"Error obteniendo datos para {symbol}: {str(e)}")
    
    return None

# Calcular EMA manualmente
def calculate_ema(prices, window):
    return prices.ewm(span=window, adjust=False).mean()

# Calcular RSI manualmente
def calculate_rsi(prices, window=14):
    delta = prices.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcular ADX manualmente
def calculate_adx(high, low, close, window=14):
    # Calcular +DM y -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Calcular True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calcular indicadores direccionales
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
    
    # Calcular DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/window).mean()
    
    return adx

# Calcular ATR
def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

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
        return 'Muy Alto'
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=20):
    if len(df) < lookback + 10:
        return None
        
    last_close = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    # Buscar máximos/mínimos recientes
    recent_df = df.tail(lookback)
    max_idx = recent_df['close'].idxmax()
    min_idx = recent_df['close'].idxmin()
    
    # Divergencia bajista
    if max_idx != -1:
        prev_high = df['close'].loc[max_idx]
        prev_rsi_high = df['rsi'].loc[max_idx]
        if last_close > prev_high and last_rsi < prev_rsi_high and last_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx != -1:
        prev_low = df['close'].loc[min_idx]
        prev_rsi_low = df['rsi'].loc[min_idx]
        if last_close < prev_low and last_rsi > prev_rsi_low and last_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(f"{symbol}-USDT", params['timeframe'])
        if df is None or len(df) < 100:
            return None, None
        
        # Calcular indicadores manualmente
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
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
            
            # Buscar la resistencia más cercana por encima
            above_resistances = [r for r in resistances if r > last['close']]
            if above_resistances:
                entry = min(above_resistances) * 1.005
            else:
                entry = last['close'] * 1.01
                
            # Buscar soporte más cercano por debajo para SL
            below_supports = [s for s in supports if s < entry]
            if below_supports:
                sl = max(below_supports) * 0.995
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
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0
            }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Buscar el soporte más cercano por debajo
            below_supports = [s for s in supports if s < last['close']]
            if below_supports:
                entry = max(below_supports) * 0.995
            else:
                entry = last['close'] * 0.99
                
            # Buscar resistencia más cercana por encima para SL
            above_resistances = [r for r in resistances if r > entry]
            if above_resistances:
                sl = min(above_resistances) * 1.005
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
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0
            }
        
        return long_signal, short_signal
    
    except Exception as e:
        print(f"Error analizando {symbol}: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                # Asegurar que los valores numéricos sean enteros
                if key != 'timeframe':
                    params[key] = int(request.form[key])
                else:
                    params[key] = request.form[key]
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    max_cryptos = params.pop('max_cryptos', 30)
    
    # Procesar criptos en grupos para evitar sobrecarga
    for i, crypto in enumerate(cryptos):
        if i >= max_cryptos:
            break
            
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
