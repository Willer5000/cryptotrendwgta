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
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 50,
    'atr_period': 14
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        cryptos = [line.strip() for line in f.readlines()]
    return cryptos[:50]  # Limitar a 50 para evitar sobrecarga

# Cálculo manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Cálculo manual de RSI
def calculate_rsi(series, window):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Cálculo manual de ATR
def calculate_atr(df, window):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

# Cálculo manual de ADX
def calculate_adx(df, window):
    # Calcular DM+ y DM-
    df['up_move'] = df['high'].diff()
    df['down_move'] = -df['low'].diff()
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Suavizar DM+ y DM-
    df['plus_dm_smoothed'] = df['plus_dm'].rolling(window=window).mean()
    df['minus_dm_smoothed'] = df['minus_dm'].rolling(window=window).mean()
    
    # Calcular TR y DI
    tr = calculate_atr(df, window)
    df['plus_di'] = (df['plus_dm_smoothed'] / tr) * 100
    df['minus_di'] = (df['minus_dm_smoothed'] / tr) * 100
    
    # Calcular DX y ADX
    df['dx'] = (np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    adx = df['dx'].rolling(window=window).mean()
    return adx

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
    kucoin_tf = tf_mapping.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000':
                candles = data.get('data', [])
                if len(candles) == 0:
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df.sort_values('timestamp').iloc[-100:]  # Últimas 100 velas
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    if len(df) < window:
        return [], []
    
    df = df.copy()
    df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
    df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
    
    resistances = df[df['high'] >= df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] <= df['low_roll']]['low'].unique().tolist()
    
    return supports[:10], resistances[:10]  # Limitar a 10 niveles

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol == 0:
        return 'Muy Bajo'
    
    ratio = current_vol / avg_vol
    if ratio > 3.0: return 'Extremo'
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=20):
    if len(df) < lookback + 5:
        return None
    
    prices = df['close'].values
    rsi_values = df['rsi'].values
    
    # Buscar máximos/mínimos en precios
    price_max_idx = np.argmax(prices[-lookback:]) + len(prices) - lookback
    price_min_idx = np.argmin(prices[-lookback:]) + len(prices) - lookback
    
    # Buscar máximos/mínimos en RSI
    rsi_max_idx = np.argmax(rsi_values[-lookback:]) + len(rsi_values) - lookback
    rsi_min_idx = np.argmin(rsi_values[-lookback:]) + len(rsi_values) - lookback
    
    # Divergencia bajista
    if (price_max_idx > rsi_max_idx and 
        prices[price_max_idx] > prices[rsi_max_idx] and 
        rsi_values[price_max_idx] < rsi_values[rsi_max_idx] and
        rsi_values[-1] > 70):
        return 'bearish'
    
    # Divergencia alcista
    if (price_min_idx > rsi_min_idx and 
        prices[price_min_idx] < prices[rsi_min_idx] and 
        rsi_values[price_min_idx] > rsi_values[rsi_min_idx] and
        rsi_values[-1] < 30):
        return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(f"{symbol}-USDT", params['timeframe'])
        if df is None or len(df) < 50:
            return None, None
        
        # Calcular indicadores manualmente
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'] = calculate_adx(df, params['adx_period'])
        df['atr'] = calculate_atr(df, params['atr_period'])
        
        # Obtener últimos valores
        last = df.iloc[-1]
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Calcular volumen promedio
        avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Señales LONG
        long_signal = None
        if trend == 'up' and last['adx'] > params['adx_level']:
            # Encontrar resistencia más cercana
            next_resistance = min([r for r in resistances if r > last['close']], default=None)
            
            if next_resistance:
                entry = next_resistance * 1.005  # Entrada en quiebre
                sl = max([s for s in supports if s < last['close']], default=entry * 0.95)
                
                # Si no hay soportes cercanos, usar ATR
                if sl >= entry:
                    sl = last['close'] - (last['atr'] * 1.5)
                
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
        if trend == 'down' and last['adx'] > params['adx_level']:
            # Encontrar soporte más cercano
            next_support = max([s for s in supports if s < last['close']], default=None)
            
            if next_support:
                entry = next_support * 0.995  # Entrada en quiebre
                sl = min([r for r in resistances if r > last['close']], default=entry * 1.05)
                
                # Si no hay resistencias cercanas, usar ATR
                if sl <= entry:
                    sl = last['close'] + (last['atr'] * 1.5)
                
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
        app.logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                try:
                    if key == 'timeframe':
                        params[key] = request.form[key]
                    else:
                        params[key] = int(request.form[key])
                except:
                    pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Analizar criptos
    for crypto in cryptos:
        long_signal, short_signal = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
        if short_signal:
            short_signals.append(short_signal)
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    # Estadísticas para gráficos
    market_stats = {
        'total_cryptos': len(cryptos),
        'long_signals': len(long_signals),
        'short_signals': len(short_signals),
        'strong_trends': len([s for s in long_signals + short_signals if s['adx'] > 40]),
        'high_volume': len([s for s in long_signals + short_signals if 'Alto' in s['volume'] or 'Extremo' in s['volume']])
    }
    
    return render_template('index.html', 
                           long_signals=long_signals[:15], 
                           short_signals=short_signals[:15],
                           params=params,
                           market_stats=market_stats)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
