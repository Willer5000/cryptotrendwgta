import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, render_template, request
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    'divergence_lookback': 20
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
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                # Invertir orden para tener cronológico
                candles.reverse()
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                # Convertir tipos
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                # Convertir timestamp (evitando FutureWarning)
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    return None

# EMA manual
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# RSI manual
def calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ADX manual
def calculate_adx(high, low, close, window):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    minus_dm = -minus_dm
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window).mean()
    return adx

# ATR manual
def calculate_atr(high, low, close, window):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

# Generar gráfico
def generate_chart(df, symbol, signal_type):
    plt.figure(figsize=(10, 4))
    plt.plot(df['timestamp'], df['close'], label='Precio', color='#1f77b4')
    
    if signal_type == 'long':
        plt.axhline(y=df['close'].iloc[-1], color='green', linestyle='--', label='Entrada')
    else:
        plt.axhline(y=df['close'].iloc[-1], color='red', linestyle='--', label='Entrada')
    
    plt.title(f'Análisis: {symbol} ({signal_type.upper()})')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio (USDT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Convertir a base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Calcular indicadores
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # ATR
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        
        return df
    except Exception as e:
        app.logger.error(f"Error calculating indicators: {str(e)}")
        return None

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    try:
        df['high_roll'] = df['high'].rolling(window=window).max()
        df['low_roll'] = df['low'].rolling(window=window).min()
        
        resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
        supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
        
        return supports, resistances
    except Exception as e:
        app.logger.error(f"Error finding S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol == 0 or current_vol == 0:
        return 'Muy Bajo'
    ratio = current_vol / avg_vol
    if ratio > 3.0: return 'Extremo'
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback):
    if len(df) < lookback + 1:
        return None
        
    # Seleccionar los últimos datos
    recent = df.iloc[-lookback:]
    
    # Buscar máximos/mínimos recientes
    max_idx = recent['close'].idxmax()
    min_idx = recent['close'].idxmin()
    
    # Divergencia bajista
    if pd.notna(max_idx):
        price_high = df.loc[max_idx, 'close']
        rsi_high = df.loc[max_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if pd.notna(min_idx):
        price_low = df.loc[min_idx, 'close']
        rsi_low = df.loc[min_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return None, None, None, None
        
        df = calculate_indicators(df, params)
        if df is None:
            return None, None, None, None
            
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
        long_chart = None
        if (trend == 'up' and last['adx'] > params['adx_level'] and 
            (is_breakout or divergence == 'bullish') and 
            volume_class in ['Alto', 'Muy Alto', 'Extremo']):
            
            # Encontrar la resistencia más cercana por encima
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            
            # Encontrar el soporte más cercano por debajo para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * 0.98
            
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
            
            # Generar gráfico
            long_chart = generate_chart(df, symbol, 'long')
        
        # Señales SHORT
        short_signal = None
        short_chart = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto', 'Extremo']):
            
            # Encontrar el soporte más cercano por debajo
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            
            # Encontrar la resistencia más cercana por encima para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * 1.02
            
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
            
            # Generar gráfico
            short_chart = generate_chart(df, symbol, 'short')
        
        return long_signal, short_signal, long_chart, short_chart
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    start_time = time.time()
    
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
    
    # Usar ThreadPool para procesamiento paralelo
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_crypto, crypto, params): crypto for crypto in cryptos}
        
        for future in as_completed(futures):
            crypto = futures[future]
            long_signal, short_signal, long_chart, short_chart = future.result()
            
            if long_signal:
                long_signal['chart'] = long_chart
                long_signals.append(long_signal)
            if short_signal:
                short_signal['chart'] = short_chart
                short_signals.append(short_signal)
    
    # Ordenar por fuerza de tendencia (ADX)
    long_signals.sort(key=lambda x: x['adx'], reverse=True)
    short_signals.sort(key=lambda x: x['adx'], reverse=True)
    
    # Estadísticas para gráficos
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    avg_rsi_long = np.mean([s['rsi'] for s in long_signals]) if long_signals else 0
    avg_rsi_short = np.mean([s['rsi'] for s in short_signals]) if short_signals else 0
    
    processing_time = round(time.time() - start_time, 2)
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           avg_rsi_long=round(avg_rsi_long, 2),
                           avg_rsi_short=round(avg_rsi_short, 2),
                           total_cryptos=len(cryptos),
                           processing_time=processing_time)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
