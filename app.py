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
    'timeframe': '1h',  # 1h por defecto como solicitado
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 20,
    'max_cryptos': 50  # Límite para evitar sobrecarga
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        cryptos = [line.strip() for line in f.readlines()]
        return cryptos[:DEFAULTS['max_cryptos']]  # Limitar al máximo configurado

# Cálculos manuales de indicadores
def calculate_ema(prices, window):
    return prices.ewm(span=window, adjust=False).mean()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, window=14):
    # Calcular +DM y -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Suavizar DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=window).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=window).mean()
    
    # Calcular True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=window).mean()
    
    # Calcular DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calcular DX y ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

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
            if data['code'] == '200000' and data['data']:
                candles = data['data']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                return df.sort_values('timestamp').iloc[-100:]  # Últimas 100 velas
    except Exception as e:
        print(f"Error obteniendo datos para {symbol}: {str(e)}")
    
    return None

# Detectar soportes y resistencias
def find_support_resistance(df, window=20):
    df['high_roll'] = df['high'].rolling(window=window).max()
    df['low_roll'] = df['low'].rolling(window=window).min()
    
    resistances = []
    supports = []
    
    # Encontrar niveles clave
    for i in range(len(df)):
        if df['high'].iloc[i] == df['high_roll'].iloc[i]:
            resistances.append(df['high'].iloc[i])
        if df['low'].iloc[i] == df['low_roll'].iloc[i]:
            supports.append(df['low'].iloc[i])
    
    # Eliminar duplicados y niveles cercanos
    resistances = sorted(list(set(resistances)))
    supports = sorted(list(set(supports)))
    
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
    
    last_close = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    # Buscar máximos/mínimos recientes
    max_idx = df['close'].tail(lookback).idxmax()
    min_idx = df['close'].tail(lookback).idxmin()
    
    # Divergencia bajista
    if max_idx != -1:
        if last_close > df['close'].loc[max_idx] and last_rsi < df['rsi'].loc[max_idx] and last_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx != -1:
        if last_close < df['close'].loc[min_idx] and last_rsi > df['rsi'].loc[min_idx] and last_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        return None, None
    
    try:
        # Calcular indicadores manualmente
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df)
        
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
            
            # Encontrar la resistencia más cercana como TP
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            
            # Encontrar soporte más cercano como SL
            prev_supports = [s for s in supports if s < entry]
            sl = max(prev_supports) * 0.995 if prev_supports else entry * 0.97
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
                'rsi': round(last['rsi'], 2)
            }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar el soporte más cercano como TP
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            
            # Encontrar resistencia más cercana como SL
            prev_resistances = [r for r in resistances if r > entry]
            sl = min(prev_resistances) * 1.005 if prev_resistances else entry * 1.03
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
                'rsi': round(last['rsi'], 2)
            }
        
        return long_signal, short_signal
    except Exception as e:
        print(f"Error analizando {symbol}: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    form_params = {k: v for k, v in request.form.items()}
    
    for key in params:
        if key in form_params and form_params[key].isdigit():
            params[key] = int(form_params[key])
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    for crypto in cryptos:
        long_signal, short_signal = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
        if short_signal:
            short_signals.append(short_signal)
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    # Estadísticas para gráficos adicionales
    volume_stats = {
        'Muy Alto': 0,
        'Alto': 0,
        'Medio': 0,
        'Bajo': 0,
        'Muy Bajo': 0
    }
    
    for signal in long_signals + short_signals:
        volume_stats[signal['volume']] += 1
    
    rsi_stats = {
        'Sobrecompra (>70)': 0,
        'Neutral (30-70)': 0,
        'Sobreventa (<30)': 0
    }
    
    for signal in long_signals:
        if signal['rsi'] < 30:
            rsi_stats['Sobreventa (<30)'] += 1
        elif signal['rsi'] > 70:
            rsi_stats['Sobrecompra (>70)'] += 1
        else:
            rsi_stats['Neutral (30-70)'] += 1
    
    for signal in short_signals:
        if signal['rsi'] < 30:
            rsi_stats['Sobreventa (<30)'] += 1
        elif signal['rsi'] > 70:
            rsi_stats['Sobrecompra (>70)'] += 1
        else:
            rsi_stats['Neutral (30-70)'] += 1
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           volume_stats=volume_stats,
                           rsi_stats=rsi_stats)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
