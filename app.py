import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pandas_ta as ta
from datetime import datetime, timedelta

app = Flask(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_DIR = 'cache'
CACHE_TIME = 900  # 15 minutos
DEFAULTS = {
    'timeframe': '4h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 50,
    'max_cryptos': 30
}

# Crear directorio de caché si no existe
os.makedirs(CACHE_DIR, exist_ok=True)

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Obtener datos de KuCoin con caché
def get_kucoin_data(symbol, timeframe):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{timeframe}.csv")
    
    # Verificar caché
    if os.path.exists(cache_file):
        mod_time = os.path.getmtime(cache_file)
        if time.time() - mod_time < CACHE_TIME:
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    
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
        if data['code'] == '200000' and data['data']:
            candles = data['data']
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            
            # Guardar en caché
            df.to_csv(cache_file, index=False)
            return df
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
    
    return df.dropna()

# Detectar soportes y resistencias (método optimizado)
def find_support_resistance(df, window=50):
    # Simplificado para ahorrar memoria
    resistances = []
    supports = []
    
    # Buscar pivots
    for i in range(window, len(df)-window):
        high_window = df['high'].iloc[i-window:i+window]
        low_window = df['low'].iloc[i-window:i+window]
        
        if df['high'].iloc[i] == high_window.max():
            resistances.append(df['high'].iloc[i])
        if df['low'].iloc[i] == low_window.min():
            supports.append(df['low'].iloc[i])
    
    # Eliminar duplicados y devolver los 5 más recientes
    return sorted(set(supports))[-5:], sorted(set(resistances))[-5:]

# Clasificar volumen
def classify_volume(current_vol, df):
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    ratio = current_vol / avg_vol if avg_vol > 0 else 1
    
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias (optimizado)
def detect_divergence(df):
    # Solo últimos 50 puntos para ahorrar recursos
    df = df.tail(50).reset_index(drop=True)
    last_close = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    # Buscar máximos/mínimos en precio y RSI
    price_max_idx = df['close'].idxmax()
    price_min_idx = df['close'].idxmin()
    rsi_max_idx = df['rsi'].idxmax()
    rsi_min_idx = df['rsi'].idxmin()
    
    # Divergencia bajista
    if price_max_idx > rsi_max_idx and last_close > df['close'].iloc[price_max_idx] and last_rsi < df['rsi'].iloc[rsi_max_idx]:
        return 'bearish'
    
    # Divergencia alcista
    if price_min_idx > rsi_min_idx and last_close < df['close'].iloc[price_min_idx] and last_rsi > df['rsi'].iloc[rsi_min_idx]:
        return 'bullish'
    
    return None

# Analizar una criptomoneda (optimizado)
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return None, None
        
        df = calculate_indicators(df, params)
        supports, resistances = find_support_resistance(df.tail(200), params['sr_window'])
        
        last = df.iloc[-1]
        volume_class = classify_volume(last['volume'], df)
        divergence = detect_divergence(df)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Señales LONG
        long_signal = None
        if (trend == 'up' and last['adx'] > params['adx_level'] and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Calcular niveles
            entry = last['close'] * 1.005
            sl = min(supports) if supports else last['close'] * 0.98
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'tp3': round(entry + risk * 3, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2)
            }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Calcular niveles
            entry = last['close'] * 0.995
            sl = max(resistances) if resistances else last['close'] * 1.02
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'tp3': round(entry - risk * 3, 4),
                'volume': volume_class,
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
                if key == 'timeframe':
                    params[key] = request.form[key]
                else:
                    try:
                        params[key] = int(request.form[key])
                    except:
                        pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar número de criptos a analizar
    max_cryptos = min(params['max_cryptos'], len(cryptos))
    
    for i, crypto in enumerate(cryptos[:max_cryptos]):
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
                           params=params,
                           last_update=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
