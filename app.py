import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

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
    'lookback': 20
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
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000':
                candles = data.get('data', [])
                if not candles:
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                # Convertir a float
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = df[col].astype(float)
                # Convertir timestamp a entero y luego a datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df.sort_values('timestamp')
    except Exception as e:
        app.logger.error(f"Error getting data for {symbol}: {str(e)}")
    
    return None

# Calcular EMA manualmente
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Calcular ADX manualmente
def calculate_adx(high, low, close, window):
    # Calcular +DM y -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Suavizar con EMA
    atr = calculate_ema(calculate_tr(high, low, close), window)
    plus_di = 100 * calculate_ema(pd.Series(plus_dm), window) / atr
    minus_di = 100 * calculate_ema(pd.Series(minus_dm), window) / atr
    
    # Calcular DX y ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = calculate_ema(pd.Series(dx), window)
    return adx

# Calcular True Range (para ATR y ADX)
def calculate_tr(high, low, close):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = np.max(np.array([tr1, tr2, tr3]), axis=0)
    return tr

# Calcular ATR manualmente
def calculate_atr(high, low, close, window):
    tr = calculate_tr(high, low, close)
    return calculate_ema(pd.Series(tr), window)

# Calcular RSI manualmente
def calculate_rsi(series, window):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcular indicadores
def calculate_indicators(df, params):
    # EMAs
    df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
    df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
    
    # ADX
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
    
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
    recent_data = df.tail(lookback)
    max_idx = recent_data['close'].idxmax()
    min_idx = recent_data['close'].idxmin()
    
    # Divergencia bajista
    if pd.notna(max_idx):
        if last_close > df.loc[max_idx, 'close'] and last_rsi < df.loc[max_idx, 'rsi'] and last_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if pd.notna(min_idx):
        if last_close < df.loc[min_idx, 'close'] and last_rsi > df.loc[min_idx, 'rsi'] and last_rsi < 30:
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
        divergence = detect_divergence(df, params['lookback'])
        
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
            
            # Encontrar la resistencia más cercana por encima
            above_resistances = [r for r in resistances if r > last['close']]
            if above_resistances:
                entry = min(above_resistances) * 1.005
            else:
                entry = last['close'] * 1.01
            
            # Encontrar el soporte más cercano por debajo
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
                'adx': round(last['adx'], 2) if not np.isnan(last['adx']) else 0
            }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar el soporte más cercano por debajo
            below_supports = [s for s in supports if s < last['close']]
            if below_supports:
                entry = max(below_supports) * 0.995
            else:
                entry = last['close'] * 0.99
            
            # Encontrar la resistencia más cercana por encima
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
                'adx': round(last['adx'], 2) if not np.isnan(last['adx']) else 0
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
                # Convertir a int si es numérico, excepto timeframe
                if key != 'timeframe':
                    params[key] = int(request.form[key])
                else:
                    params[key] = request.form[key]
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 20 criptos para no sobrecargar en la versión gratuita
    for crypto in cryptos[:20]:
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
