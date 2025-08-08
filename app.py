import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from datetime import datetime

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
    'max_cryptos': 20
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        cryptos = [line.strip() for line in f.readlines()]
        return cryptos[:int(DEFAULTS['max_cryptos'])]

# Cálculo manual de EMA
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Cálculo manual de RSI
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Cálculo manual de ADX
def calculate_adx(high, low, close, window=14):
    # Calcular +DM y -DM
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Suavizado de Wilder
    def wilder_smooth(series, window):
        return series.ewm(alpha=1/window, adjust=False).mean()
    
    atr = wilder_smooth(np.maximum(high - low, 
                                  np.maximum(np.abs(high - close.shift(1)), 
                                  np.abs(low - close.shift(1))))), window)
    plus_di = 100 * wilder_smooth(plus_dm, window) / atr
    minus_di = 100 * wilder_smooth(minus_dm, window) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, window)
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
    kucoin_tf = tf_mapping[timeframe]
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['code'] == '200000' and data['data']:
                candles = data['data']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                # Convertir tipos de datos
                df = df.apply(pd.to_numeric, errors='ignore')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                return df.iloc[-100:]  # Últimos 100 períodos
    except Exception as e:
        app.logger.error(f"Error obteniendo datos para {symbol}: {str(e)}")
    return None

# Detectar soportes y resistencias
def find_support_resistance(df, window=50):
    supports = []
    resistances = []
    
    # Encontrar mínimos locales para soportes
    for i in range(window, len(df)-window):
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            supports.append(df['low'].iloc[i])
    
    # Encontrar máximos locales para resistencias
    for i in range(window, len(df)-window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            resistances.append(df['high'].iloc[i])
    
    # Eliminar duplicados y valores cercanos
    supports = sorted(list(set(round(s, 4) for s in supports)))
    resistances = sorted(list(set(round(r, 4) for r in resistances)))
    
    return supports[-5:], resistances[-5:]  # Últimos 5 niveles significativos

# Clasificar volumen
def classify_volume(current_vol, df):
    avg_vol = df['volume'].tail(20).mean()
    ratio = current_vol / avg_vol if avg_vol > 0 else 1
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, period=14):
    try:
        # Calcular RSI
        df['rsi'] = calculate_rsi(df['close'], period)
        
        # Buscar máximos y mínimos en precio y RSI
        price_peaks = df['close'].iloc[-20:].nlargest(2)
        rsi_peaks = df['rsi'].iloc[-20:].nlargest(2)
        price_valleys = df['close'].iloc[-20:].nsmallest(2)
        rsi_valleys = df['rsi'].iloc[-20:].nsmallest(2)
        
        # Divergencia bajista: precio hace nuevo alto, RSI no lo confirma
        if len(price_peaks) > 1 and len(rsi_peaks) > 1:
            if price_peaks[0] > price_peaks[1] and rsi_peaks[0] < rsi_peaks[1]:
                return 'bearish'
        
        # Divergencia alcista: precio hace nuevo bajo, RSI no lo confirma
        if len(price_valleys) > 1 and len(rsi_valleys) > 1:
            if price_valleys[0] < price_valleys[1] and rsi_valleys[0] > rsi_valleys[1]:
                return 'bullish'
    except:
        pass
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 30:
            return None, None
        
        # Calcular indicadores manualmente
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # Calcular ADX solo si hay suficientes datos
        if len(df) > params['adx_period'] * 2:
            df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        else:
            df['adx'] = 0
        
        last = df.iloc[-1]
        current_price = last['close']
        
        # Clasificar volumen
        volume_class = classify_volume(last['volume'], df)
        
        # Detectar soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Detectar divergencias
        divergence = detect_divergence(df)
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        strong_trend = last['adx'] > params['adx_level'] if 'adx' in last else False
        
        # Señales LONG
        long_signal = None
        if trend == 'up' and strong_trend and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar resistencia más cercana
            next_resistance = next((r for r in resistances if r > current_price), None)
            if next_resistance:
                entry = current_price * 1.002  # Entrar ligeramente arriba del precio actual
                sl = min(supports) if supports else current_price * 0.97
                risk = entry - sl
                
                # Calcular objetivos de ganancia
                tp1 = entry + risk * 1
                tp2 = entry + risk * 2
                tp3 = entry + risk * 3
                
                long_signal = {
                    'symbol': symbol,
                    'entry': round(entry, 4),
                    'sl': round(sl, 4),
                    'tp1': round(tp1, 4),
                    'tp2': round(tp2, 4),
                    'tp3': round(tp3, 4),
                    'volume': volume_class,
                    'divergence': divergence == 'bullish',
                    'adx': round(last['adx'], 2) if 'adx' in last else 0,
                    'price': round(current_price, 4),
                    'trend_strength': round(last['adx'], 2) if 'adx' in last else 0
                }
        
        # Señales SHORT
        short_signal = None
        if trend == 'down' and strong_trend and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar soporte más cercano
            next_support = next((s for s in supports if s < current_price), None)
            if next_support:
                entry = current_price * 0.998  # Entrar ligeramente abajo del precio actual
                sl = max(resistances) if resistances else current_price * 1.03
                risk = sl - entry
                
                # Calcular objetivos de ganancia
                tp1 = entry - risk * 1
                tp2 = entry - risk * 2
                tp3 = entry - risk * 3
                
                short_signal = {
                    'symbol': symbol,
                    'entry': round(entry, 4),
                    'sl': round(sl, 4),
                    'tp1': round(tp1, 4),
                    'tp2': round(tp2, 4),
                    'tp3': round(tp3, 4),
                    'volume': volume_class,
                    'divergence': divergence == 'bearish',
                    'adx': round(last['adx'], 2) if 'adx' in last else 0,
                    'price': round(current_price, 4),
                    'trend_strength': round(last['adx'], 2) if 'adx' in last else 0
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
                except:
                    params[key] = request.form[key]
    
    cryptos = load_cryptos()
    long_signals = []
    short_signals = []
    trend_strengths = []
    
    for crypto in cryptos:
        long_signal, short_signal = analyze_crypto(crypto, params)
        if long_signal:
            long_signals.append(long_signal)
            trend_strengths.append({
                'symbol': crypto,
                'strength': long_signal['trend_strength'],
                'direction': 'up'
            })
        if short_signal:
            short_signals.append(short_signal)
            trend_strengths.append({
                'symbol': crypto,
                'strength': short_signal['trend_strength'],
                'direction': 'down'
            })
    
    # Ordenar por fuerza de tendencia
    long_signals.sort(key=lambda x: x['trend_strength'], reverse=True)
    short_signals.sort(key=lambda x: x['trend_strength'], reverse=True)
    trend_strengths.sort(key=lambda x: x['strength'], reverse=True)
    
    # Preparar datos para gráficos
    trend_data = {
        'labels': [f"{ts['symbol']} ({'↑' if ts['direction'] == 'up' else '↓'})" for ts in trend_strengths[:5]],
        'strengths': [ts['strength'] for ts in trend_strengths[:5]]
    }
    
    volume_dist = {
        'Muy Alto': sum(1 for s in long_signals + short_signals if s['volume'] == 'Muy Alto'),
        'Alto': sum(1 for s in long_signals + short_signals if s['volume'] == 'Alto'),
        'Medio': sum(1 for s in long_signals + short_signals if s['volume'] == 'Medio'),
        'Bajo': sum(1 for s in long_signals + short_signals if s['volume'] == 'Bajo'),
        'Muy Bajo': sum(1 for s in long_signals + short_signals if s['volume'] == 'Muy Bajo')
    }
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           params=params,
                           trend_data=trend_data,
                           volume_dist=volume_dist,
                           updated=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
