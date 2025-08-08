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
    'divergence_lookback': 20
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Cálculo manual de EMA
def calculate_ema(series, window):
    ema = [series[0]]
    alpha = 2 / (window + 1)
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[i-1])
    return ema

# Cálculo manual de RSI
def calculate_rsi(series, window):
    deltas = series.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Cálculo manual de ADX
def calculate_adx(high, low, close, window):
    # Calcular +DM y -DM
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    
    # Calcular True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Suavizar valores
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    # Calcular DX y ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di

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
                
                # Convertir tipos y ordenar
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    return None

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
    df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
    
    resistances = df[df['high'] >= df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] <= df['low_roll']]['low'].unique().tolist()
    
    # Filtrar niveles cercanos
    resistances = sorted(set(resistances), reverse=True)[:10]
    supports = sorted(set(supports))[:10]
    
    return supports, resistances

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol == 0: 
        return 'Muy Alto' if current_vol > 0 else 'Muy Bajo'
        
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback):
    if len(df) < lookback + 10:
        return None
        
    prices = df['close'].values
    rsi = df['rsi'].values
    
    # Buscar máximos/mínimos recientes
    max_idx = prices[-lookback:].argmax() + len(prices) - lookback
    min_idx = prices[-lookback:].argmin() + len(prices) - lookback
    
    # Divergencia bajista
    if max_idx > 0:
        if prices[-1] > prices[max_idx] and rsi[-1] < rsi[max_idx] and rsi[-1] > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx > 0:
        if prices[-1] < prices[min_idx] and rsi[-1] > rsi[min_idx] and rsi[-1] < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            return None, None
        
        # Calcular indicadores manualmente
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # Calcular volumen promedio
        avg_vol = df['volume'].tail(20).mean()
        last = df.iloc[-1]
        
        # Detectar soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Clasificar volumen
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Detectar divergencia
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        strong_trend = last['adx'] > params['adx_level'] if not pd.isna(last['adx']) else False
        
        # Señales LONG
        long_signal = None
        if trend == 'up' and strong_trend and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar resistencia más cercana
            next_resistance = next((r for r in resistances if r > last['close']), None)
            
            if next_resistance:
                entry = next_resistance * 1.005  # Entrada al superar resistencia
                # Encontrar soporte más cercano para SL
                closest_support = next((s for s in supports if s < entry), last['close'] * 0.98)
                sl = closest_support * 0.995
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
                    'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                    'price': round(last['close'], 4),
                    'change': round((last['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100, 2)
                }
        
        # Señales SHORT
        short_signal = None
        if trend == 'down' and strong_trend and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar soporte más cercano
            next_support = next((s for s in supports if s < last['close']), None)
            
            if next_support:
                entry = next_support * 0.995  # Entrada al romper soporte
                # Encontrar resistencia más cercana para SL
                closest_resistance = next((r for r in resistances if r > entry), last['close'] * 1.02)
                sl = closest_resistance * 1.005
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
                    'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                    'price': round(last['close'], 4),
                    'change': round((last['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100, 2)
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
                    # Manejar diferentes tipos de parámetros
                    if key == 'timeframe':
                        params[key] = request.form[key]
                    else:
                        params[key] = int(request.form[key])
                except:
                    pass
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    
    # Analizar solo las primeras 25 criptos para mantener el rendimiento
    for crypto in cryptos[:25]:
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
        'strong_trends': sum(1 for s in long_signals + short_signals if s['adx'] > 30),
        'high_volume': sum(1 for s in long_signals + short_signals if s['volume'] in ['Alto', 'Muy Alto'])
    }
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           market_stats=market_stats,
                           now=datetime.utcnow())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
