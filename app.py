import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import math

app = Flask(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 900  # 15 minutos
DEFAULTS = {
    'timeframe': '4h',
    'ema_fast': 9,
    'ema_slow': 20,
    'ts_period': 14,
    'ts_level': 0.5,
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
            if not candles:
                return None
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            return df.sort_values('timestamp')
    return None

# Calcular EMA manualmente
def calculate_ema(series, window):
    if len(series) < window:
        return None
    ema = [series[:window].mean()]
    multiplier = 2 / (window + 1)
    
    for i in range(window, len(series)):
        ema_val = (series[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_val)
    
    # Rellenar los primeros valores con NaN
    prefix = [np.nan] * (window - 1)
    return pd.Series(prefix + ema)

# Calcular RSI manualmente
def calculate_rsi(series, window):
    if len(series) < window + 1:
        return None
    
    deltas = series.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcular fuerza de tendencia (reemplazo para ADX)
def calculate_trend_strength(high, low, close, window):
    if len(close) < window + 1:
        return None
    
    # Calcular la diferencia de precios
    price_diff = abs(close.diff())
    
    # Calcular el movimiento direccional
    up_move = high.diff()
    down_move = -low.diff()
    
    # Suavizar los movimientos
    up_avg = up_move.rolling(window).mean()
    down_avg = down_move.rolling(window).mean()
    
    # Calcular fuerza relativa
    strength = abs(up_avg - down_avg) / (0.5 * (up_avg + down_avg))
    return strength * 100

# Calcular ATR manualmente
def calculate_atr(high, low, close, window):
    if len(close) < window + 1:
        return None
    
    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr.max(axis=1)
    
    atr = tr['tr'].rolling(window).mean()
    return atr

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    if len(df) < window:
        return [], []
    
    df['high_roll'] = df['high'].rolling(window=window).max()
    df['low_roll'] = df['low'].rolling(window=window).min()
    
    resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
    
    # Filtrar niveles cercanos
    resistances = [r for i, r in enumerate(resistances) 
                 if all(abs(r - r2) > r * 0.005 for r2 in resistances[:i])]
    supports = [s for i, s in enumerate(supports) 
              if all(abs(s - s2) > s * 0.005 for s2 in supports[:i])]
    
    return supports, resistances

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol <= 0:
        return 'Muy Alto'
    
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=20):
    if len(df) < lookback + 1:
        return None
    
    last_close = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1] if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]) else 50
    
    # Buscar máximos/mínimos recientes
    max_idx = df['close'].tail(lookback).idxmax()
    min_idx = df['close'].tail(lookback).idxmin()
    
    # Divergencia bajista
    if max_idx != -1 and max_idx < len(df) - 1:
        if last_close > df['close'].iloc[max_idx] and last_rsi < df['rsi'].iloc[max_idx] and last_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx != -1 and min_idx < len(df) - 1:
        if last_close < df['close'].iloc[min_idx] and last_rsi > df['rsi'].iloc[min_idx] and last_rsi < 30:
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
        df['ts'] = calculate_trend_strength(df['high'], df['low'], df['close'], params['ts_period'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        
        # Obtener último dato válido
        df = df.dropna(subset=['ema_fast', 'ema_slow', 'ts'])
        if len(df) < 2:
            return None, None
            
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Calcular volumen promedio
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
        
        # Señales LONG
        long_signal = None
        if (trend == 'up' and last['ts'] > params['ts_level'] and 
            (is_breakout or divergence == 'bullish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar próximo nivel de resistencia
            next_res = min((r for r in resistances if r > last['close']), default=last['close'] * 1.05)
            entry = next_res * 1.005
            
            # Encontrar soporte más cercano para SL
            closest_support = max((s for s in supports if s < entry), default=entry * 0.95)
            sl = closest_support * 0.995
            
            # Calcular TP basado en ATR si está disponible
            atr_val = last['atr'] if 'atr' in last and not pd.isna(last['atr']) else (entry - sl) * 2
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + atr_val * 0.5, 4),
                'tp2': round(entry + atr_val, 4),
                'tp3': round(entry + atr_val * 1.5, 4),
                'volume': volume_class,
                'divergence': divergence == 'bullish',
                'ts': round(last['ts'], 2)
            }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['ts'] > params['ts_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar próximo nivel de soporte
            next_sup = max((s for s in supports if s < last['close']), default=last['close'] * 0.95)
            entry = next_sup * 0.995
            
            # Encontrar resistencia más cercana para SL
            closest_res = min((r for r in resistances if r > entry), default=entry * 1.05)
            sl = closest_res * 1.005
            
            # Calcular TP basado en ATR si está disponible
            atr_val = last['atr'] if 'atr' in last and not pd.isna(last['atr']) else (sl - entry) * 2
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - atr_val * 0.5, 4),
                'tp2': round(entry - atr_val, 4),
                'tp3': round(entry - atr_val * 1.5, 4),
                'volume': volume_class,
                'divergence': divergence == 'bearish',
                'ts': round(last['ts'], 2)
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
                    params[key] = int(request.form[key])
                except ValueError:
                    # Mantener valor por defecto si no es entero
                    pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 20 criptos para reducir carga
    for crypto in cryptos[:20]:
        long_signal, short_signal = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
        if short_signal:
            short_signals.append(short_signal)
    
    # Ordenar por fuerza de tendencia
    long_signals.sort(key=lambda x: x['ts'], reverse=True)
    short_signals.sort(key=lambda x: x['ts'], reverse=True)
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
