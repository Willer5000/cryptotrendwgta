import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pandas_ta as ta

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
        return [line.strip() for line in f.readlines() if line.strip()]

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
    kucoin_tf = tf_mapping.get(timeframe, '4hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('code') == '200000':
            candles = data.get('data', [])
            if not candles:
                return None
                
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df.sort_values('timestamp').reset_index(drop=True)
    return None

# Calcular indicadores
def calculate_indicators(df, params):
    # EMAs
    df['ema_fast'] = ta.ema(df['close'], length=params['ema_fast'])
    df['ema_slow'] = ta.ema(df['close'], length=params['ema_slow'])
    
    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=params['adx_period'])
    df['adx'] = adx[f'ADX_{params["adx_period"]}']
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=params['rsi_period'])
    
    # ATR
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr'] = atr
    
    return df

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df['high_roll'] = df['high'].rolling(window=window).max()
    df['low_roll'] = df['low'].rolling(window=window).min()
    
    resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
    
    # Filtrar niveles cercanos
    resistances = sorted(set(r for r in resistances if not any(abs(r - r2)/r2 < 0.01 for r2 in resistances if r != r2)))
    supports = sorted(set(s for s in supports if not any(abs(s - s2)/s2 < 0.01 for s2 in supports if s != s2)))
    
    return supports, resistances

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol == 0:  # Evitar división por cero
        return 'Muy Alto'
        
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=30):
    if len(df) < lookback + 10:
        return None
        
    # Obtener los últimos datos
    close_prices = df['close'].values
    rsi_values = df['rsi'].values
    
    # Buscar máximos/mínimos recientes
    max_idx = df['close'].tail(lookback).idxmax()
    min_idx = df['close'].tail(lookback).idxmin()
    
    # Divergencia bajista
    if max_idx != -1 and max_idx > 0:
        if (close_prices[-1] > close_prices[max_idx] and 
            rsi_values[-1] < rsi_values[max_idx] and 
            rsi_values[-1] > 70):
            return 'bearish'
    
    # Divergencia alcista
    if min_idx != -1 and min_idx > 0:
        if (close_prices[-1] < close_prices[min_idx] and 
            rsi_values[-1] > rsi_values[min_idx] and 
            rsi_values[-1] < 30):
            return 'bullish'
    
    return None

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
        divergence = detect_divergence(df)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Señales LONG
        long_signal = None
        if trend == 'up' and last['adx'] > params['adx_level']:
            # Verificar si está cerca de soporte o rompiendo resistencia
            near_support = any(last['close'] < s * 1.005 and last['close'] > s * 0.995 for s in supports)
            breakout = any(last['close'] > r * 1.005 for r in resistances)
            
            if (near_support or breakout or divergence == 'bullish') and volume_class in ['Alto', 'Muy Alto']:
                # Calcular niveles
                entry = last['close']
                if supports:
                    entry = max(s for s in supports if s < entry) * 1.005
                
                if resistances:
                    sl = max(s for s in supports if s < entry) * 0.995 if supports else entry * 0.98
                else:
                    sl = entry * 0.98
                
                risk = entry - sl
                if risk <= 0:  # Evitar riesgo negativo
                    risk = entry * 0.02
                
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
        if trend == 'down' and last['adx'] > params['adx_level']:
            # Verificar si está cerca de resistencia o rompiendo soporte
            near_resistance = any(last['close'] < r * 1.005 and last['close'] > r * 0.995 for r in resistances)
            breakdown = any(last['close'] < s * 0.995 for s in supports)
            
            if (near_resistance or breakdown or divergence == 'bearish') and volume_class in ['Alto', 'Muy Alto']:
                # Calcular niveles
                entry = last['close']
                if resistances:
                    entry = min(r for r in resistances if r > entry) * 0.995
                
                if supports:
                    sl = min(r for r in resistances if r > entry) * 1.005 if resistances else entry * 1.02
                else:
                    sl = entry * 1.02
                
                risk = sl - entry
                if risk <= 0:  # Evitar riesgo negativo
                    risk = entry * 0.02
                
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
    except Exception as e:
        print(f"Error analizando {symbol}: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                try:
                    # Manejar temporalidad diferente
                    if key == 'timeframe':
                        params[key] = request.form[key]
                    else:
                        params[key] = int(request.form[key])
                except:
                    pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 30 criptos para reducir carga
    for crypto in cryptos[:30]:
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
    app.run(host='0.0.0.0', port=port, debug=False)
