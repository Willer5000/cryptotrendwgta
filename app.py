import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import logging
from datetime import datetime, timedelta

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
    
    # KuCoin usa símbolos como BTC-USDT
    kucoin_symbol = f"{symbol}-USDT"
    
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={kucoin_symbol}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                if not candles:
                    return None
                    
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                return df.sort_values('timestamp')
            else:
                app.logger.warning(f"No data for {symbol}: {data.get('msg')}")
        else:
            app.logger.error(f"API error for {symbol}: {response.status_code}")
    except Exception as e:
        app.logger.error(f"Error fetching {symbol}: {str(e)}")
    
    return None

# Calcular indicadores
def calculate_indicators(df, params):
    # EMAs
    df['ema_fast'] = EMAIndicator(df['close'], window=params['ema_fast']).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=params['ema_slow']).ema_indicator()
    
    # ADX
    df['adx'] = ADXIndicator(
        df['high'], df['low'], df['close'], window=params['adx_period']
    ).adx()
    
    # RSI
    df['rsi'] = RSIIndicator(df['close'], window=params['rsi_period']).rsi()
    
    # ATR
    df['atr'] = AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    
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
    if avg_vol <= 0:  # Evitar división por cero
        return 'Muy Bajo'
        
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback=20):
    if len(df) < lookback:
        return None
        
    last_close = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    # Buscar máximos/mínimos recientes
    max_idx = df['close'].iloc[-lookback:].idxmax()
    min_idx = df['close'].iloc[-lookback:].idxmin()
    
    # Divergencia bajista
    if pd.notna(max_idx):
        max_rsi = df.loc[max_idx, 'rsi']
        if last_close > df.loc[max_idx, 'close'] and last_rsi < max_rsi and last_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if pd.notna(min_idx):
        min_rsi = df.loc[min_idx, 'rsi']
        if last_close < df.loc[min_idx, 'close'] and last_rsi > min_rsi and last_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
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
        entry = None
        sl = None
        
        # Encontrar resistencia más cercana
        next_res = min((r for r in resistances if r > last['close']), default=None)
        
        if next_res:
            entry = next_res * 1.005  # Entrar un 0.5% arriba de la resistencia
            # Encontrar soporte más cercano para SL
            prev_support = max((s for s in supports if s < entry), default=entry * 0.95)
            sl = prev_support * 0.995
        else:
            entry = last['close'] * 1.01
            sl = entry * 0.97
            
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
    if trend == 'down' and last['adx'] > params['adx_level']:
        entry = None
        sl = None
        
        # Encontrar soporte más cercano
        next_support = max((s for s in supports if s < last['close']), default=None)
        
        if next_support:
            entry = next_support * 0.995  # Entrar un 0.5% abajo del soporte
            # Encontrar resistencia más cercana para SL
            prev_res = min((r for r in resistances if r > entry), default=entry * 1.05)
            sl = prev_res * 1.005
        else:
            entry = last['close'] * 0.99
            sl = entry * 1.03
            
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
                # Manejar timeframe como string
                if key == 'timeframe':
                    params[key] = request.form[key]
                else:
                    try:
                        params[key] = int(request.form[key])
                    except ValueError:
                        pass  # Mantener el valor por defecto
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Limitar a 20 criptos para optimizar recursos
    for crypto in cryptos[:20]:
        try:
            long_signal, short_signal = analyze_crypto(crypto, params)
            if long_signal:
                long_signals.append(long_signal)
            if short_signal:
                short_signals.append(short_signal)
        except Exception as e:
            app.logger.error(f"Error analizando {crypto}: {str(e)}")
            continue
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           last_update=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
