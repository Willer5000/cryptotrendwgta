import os
import time
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
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
    'max_risk_percent': 1.0
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
            if data['code'] == '200000' and data['data']:
                candles = data['data']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                return df.sort_values('timestamp').iloc[-300:]  # Limitar a 300 velas
    except Exception as e:
        print(f"Error obteniendo datos para {symbol}: {str(e)}")
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
    df['atr'] = atr['ATRr_14']
    
    return df.dropna()

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
    df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
    
    resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
    
    return sorted(supports), sorted(resistances)

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
    if len(df) < lookback + 5:
        return None
    
    # Obtener los últimos precios e RSI
    prices = df['close'].values[-lookback:]
    rsis = df['rsi'].values[-lookback:]
    
    # Encontrar máximos y mínimos
    price_peaks = [i for i in range(1, len(prices)-1) if prices[i] > prices[i-1] and prices[i] > prices[i+1]]
    rsi_peaks = [i for i in range(1, len(rsis)-1) if rsis[i] > rsis[i-1] and rsis[i] > rsis[i+1]]
    
    # Buscar divergencia bajista
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        last_rsi_peak = rsi_peaks[-1]
        prev_rsi_peak = rsi_peaks[-2]
        
        if prices[last_price_peak] > prices[prev_price_peak] and rsis[last_rsi_peak] < rsis[prev_rsi_peak]:
            return 'bearish'
    
    # Buscar divergencia alcista
    price_valleys = [i for i in range(1, len(prices)-1) if prices[i] < prices[i-1] and prices[i] < prices[i+1]]
    rsi_valleys = [i for i in range(1, len(rsis)-1) if rsis[i] < rsis[i-1] and rsis[i] < rsis[i+1]]
    
    if len(price_valleys) >= 2 and len(rsi_valleys) >= 2:
        last_price_valley = price_valleys[-1]
        prev_price_valley = price_valleys[-2]
        last_rsi_valley = rsi_valleys[-1]
        prev_rsi_valley = rsi_valleys[-2]
        
        if prices[last_price_valley] < prices[prev_price_valley] and rsis[last_rsi_valley] > rsis[prev_rsi_valley]:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return None, None
        
        df = calculate_indicators(df, params)
        if len(df) < 50:
            return None, None
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df)
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.005 for r in resistances[-3:])
        is_breakdown = any(last['close'] < s * 0.995 for s in supports[-3:])
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Señales LONG
        long_signal = None
        if (trend == 'up' and last['adx'] > params['adx_level'] and 
            (is_breakout or divergence == 'bullish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Calcular entrada, SL y TP
            entry = min(r for r in resistances if r > last['close']) if resistances else last['close'] * 1.01
            sl = max(s for s in supports if s < entry) if supports else entry * 0.98
            risk = entry - sl
            
            # Ajustar riesgo máximo
            max_risk = last['close'] * (params['max_risk_percent'] / 100)
            if risk > max_risk > 0:
                risk = max_risk
                sl = entry - risk
            
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
            
            entry = max(s for s in supports if s < last['close']) if supports else last['close'] * 0.99
            sl = min(r for r in resistances if r > entry) if resistances else entry * 1.02
            risk = sl - entry
            
            # Ajustar riesgo máximo
            max_risk = last['close'] * (params['max_risk_percent'] / 100)
            if risk > max_risk > 0:
                risk = max_risk
                sl = entry + risk
            
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
                if key == 'timeframe':
                    params[key] = request.form[key]
                else:
                    try:
                        params[key] = float(request.form[key])
                    except:
                        pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 20 criptos por recursos
    for crypto in cryptos[:20]:
        long_signal, short_signal = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
        if short_signal:
            short_signals.append(short_signal)
        time.sleep(0.1)  # Evitar sobrecarga
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           last_update=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
