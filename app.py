import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import json
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
    'divergence_lookback': 20
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r', encoding='utf-8') as f:
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
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                return df.sort_values('timestamp').reset_index(drop=True)
        return None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Calcular EMA manualmente
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Calcular RSI manualmente
def calculate_rsi(series, window):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcular ADX manualmente
def calculate_adx(high, low, close, window):
    # Calcular +DM y -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Suavizar con EMA
    plus_dm_smoothed = calculate_ema(plus_dm, window)
    minus_dm_smoothed = calculate_ema(minus_dm, window)
    
    # Calcular True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = calculate_ema(tr, window)
    
    # Calcular DI+ y DI-
    plus_di = (plus_dm_smoothed / atr) * 100
    minus_di = (minus_dm_smoothed / atr) * 100
    
    # Calcular DX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = calculate_ema(dx, window)
    return adx

# Calcular ATR manualmente
def calculate_atr(high, low, close, window):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df = df.copy()
    df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
    df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
    
    resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
    
    # Filtrar niveles cercanos
    resistances = sorted(set([round(r, 4) for r in resistances]), reverse=True)
    supports = sorted(set([round(s, 4) for s in supports]))
    
    return supports, resistances

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol == 0:  # Evitar división por cero
        return 'Muy Bajo'
    
    ratio = current_vol / avg_vol
    if ratio > 2.5: return 'Muy Alto'
    if ratio > 1.8: return 'Alto'
    if ratio > 1.2: return 'Medio'
    if ratio > 0.8: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=20):
    if len(df) < lookback + 5:
        return None
    
    # Considerar solo el último lookback
    df_sub = df.tail(lookback).copy()
    
    # Buscar máximos y mínimos en el periodo
    max_idx = df_sub['close'].idxmax()
    min_idx = df_sub['close'].idxmin()
    
    # Divergencia bajista: Precio hace nuevo máximo, RSI no
    if max_idx == df_sub.index[-1]:
        # Comparar con el máximo anterior
        prev_max_idx = df_sub['close'].iloc[:-1].idxmax()
        if prev_max_idx != max_idx:
            if df_sub.loc[max_idx, 'close'] > df_sub.loc[prev_max_idx, 'close'] and \
               df_sub.loc[max_idx, 'rsi'] < df_sub.loc[prev_max_idx, 'rsi']:
                return 'bearish'
    
    # Divergencia alcista: Precio hace nuevo mínimo, RSI no
    if min_idx == df_sub.index[-1]:
        prev_min_idx = df_sub['close'].iloc[:-1].idxmin()
        if prev_min_idx != min_idx:
            if df_sub.loc[min_idx, 'close'] < df_sub.loc[prev_min_idx, 'close'] and \
               df_sub.loc[min_idx, 'rsi'] > df_sub.loc[prev_min_idx, 'rsi']:
                return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None
    
    # Calcular indicadores manualmente
    df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
    df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
    df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
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
    
    # Señales LONG
    long_signal = None
    if (trend == 'up' and last['adx'] > params['adx_level'] and 
        (is_breakout or divergence == 'bullish') and 
        volume_class in ['Alto', 'Muy Alto']):
        
        # Encontrar siguiente resistencia
        next_res = min([r for r in resistances if r > last['close']], default=last['close'] * 1.05)
        entry = next_res * 1.005  # Entrar 0.5% arriba de la resistencia
        
        # Encontrar soporte más cercano para SL
        closest_support = max([s for s in supports if s < entry], default=entry * 0.95)
        sl = closest_support * 0.995  # SL 0.5% abajo del soporte
        
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
        
        # Encontrar siguiente soporte
        next_sup = max([s for s in supports if s < last['close']], default=last['close'] * 0.95)
        entry = next_sup * 0.995  # Entrar 0.5% abajo del soporte
        
        # Encontrar resistencia más cercana para SL
        closest_res = min([r for r in resistances if r > entry], default=entry * 1.05)
        sl = closest_res * 1.005  # SL 0.5% arriba de la resistencia
        
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
                # Convertir a entero excepto para timeframe
                if key != 'timeframe':
                    params[key] = int(request.form[key])
                else:
                    params[key] = request.form[key]
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 20 criptos para no exceder recursos
    for crypto in cryptos[:20]:
        try:
            long_signal, short_signal = analyze_crypto(crypto, params)
            if long_signal:
                long_signals.append(long_signal)
            if short_signal:
                short_signals.append(short_signal)
        except Exception as e:
            print(f"Error analyzing {crypto}: {str(e)}")
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    # Preparar datos para el gráfico
    chart_data = {
        'labels': ['Señales LONG', 'Señales SHORT', 'Sin señal'],
        'data': [len(long_signals), len(short_signals), 20 - (len(long_signals) + len(short_signals))]
    }
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           chart_data=chart_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
