import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 900  # 15 minutos
DEFAULTS = {
    'timeframe': '1h',  # Temporalidad por defecto: 1 hora
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
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
        return cryptos
    except Exception as e:
        app.logger.error(f"Error loading cryptos: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK']

# Cálculo manual de EMA
def calculate_ema(prices, window):
    ema = [prices[0]]
    k = 2 / (window + 1)
    for i in range(1, len(prices)):
        ema.append(prices[i] * k + ema[-1] * (1 - k))
    return ema

# Cálculo manual de RSI
def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down if down != 0 else 0
    rsi = [100 - (100 / (1 + rs))] if down != 0 else [100]
    
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
        
        rs = up / down if down != 0 else 0
        rsi.append(100 - (100 / (1 + rs)) if down != 0 else 100)
    
    return [np.nan] * window + rsi

# Cálculo manual de ADX
def calculate_adx(high, low, close, window):
    plus_dm = [0]
    minus_dm = [0]
    tr = [0]
    
    for i in range(1, len(high)):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        
        if h_diff > l_diff and h_diff > 0:
            plus_dm.append(h_diff)
        else:
            plus_dm.append(0)
            
        if l_diff > h_diff and l_diff > 0:
            minus_dm.append(l_diff)
        else:
            minus_dm.append(0)
        
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    
    tr_smooth = [sum(tr[1:window+1]) / window]
    plus_dm_smooth = [sum(plus_dm[1:window+1]) / window]
    minus_dm_smooth = [sum(minus_dm[1:window+1]) / window]
    
    for i in range(window+1, len(tr)):
        tr_smooth.append((tr_smooth[-1] * (window - 1) + tr[i]) / window)
        plus_dm_smooth.append((plus_dm_smooth[-1] * (window - 1) + plus_dm[i]) / window)
        minus_dm_smooth.append((minus_dm_smooth[-1] * (window - 1) + minus_dm[i]) / window)
    
    plus_di = [100 * (x / y) if y != 0 else 0 for x, y in zip(plus_dm_smooth, tr_smooth)]
    minus_di = [100 * (x / y) if y != 0 else 0 for x, y in zip(minus_dm_smooth, tr_smooth)]
    dx = [100 * abs(x - y) / (x + y) if x + y != 0 else 0 for x, y in zip(plus_di, minus_di)]
    
    adx = [sum(dx[:window]) / window]
    for i in range(window, len(dx)):
        adx.append((adx[-1] * (window - 1) + dx[i]) / window)
    
    return adx, plus_di, minus_di

# Cálculo manual de ATR
def calculate_atr(high, low, close, window):
    tr = [0]
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    
    atr = [sum(tr[1:window+1]) / window]
    for i in range(window+1, len(tr)):
        atr.append((atr[-1] * (window - 1) + tr[i]) / window)
    
    return [np.nan] * window + atr

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
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000':
                candles = data.get('data', [])
                if candles:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    # Convertir a numérico
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Convertir timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    return df.sort_values('timestamp')
        return None
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
    df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
    
    resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
    
    return supports, resistances

# Clasificar volumen
def classify_volume(current_vol, df_vol):
    if len(df_vol) < 20:
        return 'Medio'
    
    avg_vol = df_vol.tail(20).mean()
    if avg_vol == 0:
        return 'Medio'
    
    ratio = current_vol / avg_vol
    if ratio > 3.0: return 'Muy Alto'
    if ratio > 2.0: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=20):
    if len(df) < lookback + 10:
        return None
        
    prices = df['close'].values
    rsi = df['rsi'].values
    
    # Buscar máximos/mínimos recientes
    max_idx = prices[-lookback:].argmax() + len(prices) - lookback
    min_idx = prices[-lookback:].argmin() + len(prices) - lookback
    
    # Divergencia bajista
    if prices[-1] > prices[max_idx] and rsi[-1] < rsi[max_idx] and rsi[-1] < 70:
        return 'bearish'
    
    # Divergencia alcista
    if prices[-1] < prices[min_idx] and rsi[-1] > rsi[min_idx] and rsi[-1] > 30:
        return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None
    
    # Calcular indicadores manualmente
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # EMA
    df['ema_fast'] = calculate_ema(close_prices, params['ema_fast'])
    df['ema_slow'] = calculate_ema(close_prices, params['ema_slow'])
    
    # RSI
    df['rsi'] = calculate_rsi(close_prices, params['rsi_period'])
    
    # ADX
    adx, _, _ = calculate_adx(high_prices, low_prices, close_prices, params['adx_period'])
    df['adx'] = adx + [np.nan] * (len(df) - len(adx))  # Ajustar longitud
    
    # ATR
    df['atr'] = calculate_atr(high_prices, low_prices, close_prices, params['atr_period'])
    
    # Soportes y resistencias
    supports, resistances = find_support_resistance(df, params['sr_window'])
    
    last = df.iloc[-1]
    volume_class = classify_volume(last['volume'], df['volume'])
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
        
        # Encontrar la resistencia más cercana por encima
        above_resistances = [r for r in resistances if r > last['close']]
        entry = min(above_resistances) * 1.005 if above_resistances else last['close'] * 1.01
        
        # Calcular SL basado en soporte más cercano o ATR
        below_supports = [s for s in supports if s < entry]
        if below_supports:
            sl = max(below_supports) * 0.995
        else:
            sl = entry - 2 * last['atr'] if not np.isnan(last['atr']) else entry * 0.98
        
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
        entry = max(below_supports) * 0.995 if below_supports else last['close'] * 0.99
        
        # Calcular SL basado en resistencia más cercana o ATR
        above_resistances = [r for r in resistances if r > entry]
        if above_resistances:
            sl = min(above_resistances) * 1.005
        else:
            sl = entry + 2 * last['atr'] if not np.isnan(last['atr']) else entry * 1.02
        
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

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                # Convertir a int si es numérico
                if key != 'timeframe':
                    params[key] = int(request.form[key])
                else:
                    params[key] = request.form[key]
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 20 criptos para no sobrecargar en el plan gratuito
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
