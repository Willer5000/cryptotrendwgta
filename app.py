import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pandas_ta as ta  # Cambio a pandas_ta

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
    'max_analyze': 30  # Límite para Render gratuito
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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get('code') == '200000':
            candles = data['data']
            if not candles:
                return None
                
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df.sort_values('timestamp').iloc[-200:]  # Últimas 200 velas
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
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

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
    df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
    
    resistances = df[df['high'] >= 0.99 * df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] <= 1.01 * df['low_roll']]['low'].unique().tolist()
    
    # Filtrar niveles cercanos
    resistances = sorted(set(round(r, 4) for r in resistances))[-5:]
    supports = sorted(set(round(s, 4) for s in supports))[:5]
    
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
def detect_divergence(df, lookback=30):
    try:
        # Buscar máximos en precio e indicador
        price_highs = df['close'].rolling(5, center=True).max().dropna()
        rsi_highs = df['rsi'].rolling(5, center=True).max().dropna()
        
        # Buscar mínimos en precio e indicador
        price_lows = df['close'].rolling(5, center=True).min().dropna()
        rsi_lows = df['rsi'].rolling(5, center=True).min().dropna()
        
        # Divergencia bajista: precio hace máximos más altos, RSI máximos más bajos
        if len(price_highs) > 1 and len(rsi_highs) > 1:
            if price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]:
                if df['rsi'].iloc[-1] < 70:
                    return 'bearish'
        
        # Divergencia alcista: precio hace mínimos más bajos, RSI mínimos más altos
        if len(price_lows) > 1 and len(rsi_lows) > 1:
            if price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
                if df['rsi'].iloc[-1] > 30:
                    return 'bullish'
    except:
        pass
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        return None, None
    
    df = calculate_indicators(df, params)
    supports, resistances = find_support_resistance(df, params['sr_window'])
    
    last = df.iloc[-1]
    avg_vol = df['volume'].tail(20).mean()
    volume_class = classify_volume(last['volume'], avg_vol)
    divergence = detect_divergence(df)
    
    # Determinar tendencia
    trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
    adx_strong = last['adx'] > params['adx_level'] if not pd.isna(last['adx']) else False
    
    # Señales LONG
    long_signal = None
    if trend == 'up' and adx_strong:
        # Punto de entrada: primer resistencia por encima
        entry = next((r for r in resistances if r > last['close'] * 1.01), last['close'] * 1.01)
        
        # Stop Loss: soporte más cercano
        sl_candidates = [s for s in supports if s < entry]
        sl = max(sl_candidates) if sl_candidates else entry * 0.98
        
        # Calcular niveles TP basados en riesgo
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
    if trend == 'down' and adx_strong:
        # Punto de entrada: primer soporte por debajo
        entry = next((s for s in supports if s < last['close'] * 0.99), last['close'] * 0.99)
        
        # Stop Loss: resistencia más cercana
        sl_candidates = [r for r in resistances if r > entry]
        sl = min(sl_candidates) if sl_candidates else entry * 1.02
        
        # Calcular niveles TP basados en riesgo
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

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                try:
                    params[key] = int(request.form[key])
                except:
                    pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar análisis por rendimiento
    max_analyze = min(params['max_analyze'], 30)  # Máximo 30 para Render gratuito
    
    for i, crypto in enumerate(cryptos):
        if i >= max_analyze:
            break
            
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
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           analyzed_count=min(len(cryptos), max_analyze))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
