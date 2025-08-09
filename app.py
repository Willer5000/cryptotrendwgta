import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

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
    'divergence_lookback': 20,
    'max_risk_percent': 1.5,
    'support_resistance_percent': 2.0
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
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                candles.reverse()
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Valor neutro cuando no hay datos suficientes

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm <= 0] = 0
    minus_dm[minus_dm <= 0] = 0
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
    adx = dx.rolling(window).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

# Implementación manual de ATR
def calculate_atr(high, low, close, window):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().fillna(0)
    return atr

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    # EMA
    df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
    df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
    
    # ADX
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
    
    # ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    return df

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
    df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
    
    resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
    supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
    
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

# Obtener nivel de volumen para el gráfico
def get_volume_level(volume_class):
    if volume_class == 'Muy Alto': return 4
    if volume_class == 'Alto': return 3
    if volume_class == 'Medio': return 2
    if volume_class == 'Bajo': return 1
    return 0

# Detectar divergencias
def detect_divergence(df, lookback):
    if len(df) < lookback + 1:
        return None
        
    # Seleccionar los últimos datos
    recent = df.iloc[-lookback:]
    
    # Buscar máximos/mínimos recientes
    max_idx = recent['close'].idxmax() if len(recent) > 0 else None
    min_idx = recent['close'].idxmin() if len(recent) > 0 else None
    
    # Divergencia bajista
    if max_idx is not None and not pd.isna(max_idx):
        price_high = df.loc[max_idx, 'close']
        rsi_high = df.loc[max_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx is not None and not pd.isna(min_idx):
        price_low = df.loc[min_idx, 'close']
        rsi_low = df.loc[min_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
            return 'bullish'
    
    return None

# Verificar si el precio está cerca de un nivel
def is_price_near_level(price, levels, percent=1.0):
    if not levels:
        return False
    for level in levels:
        if abs(price - level) / level <= percent / 100.0:
            return True
    return False

# Calcular puntuación para una cripto (probabilidad de señales)
def calculate_probability(df, last, supports, resistances, volume_class, divergence, params):
    long_prob = 0
    short_prob = 0
    
    # LONG: Tendencia alcista
    if last['ema_fast'] > last['ema_slow']:
        long_prob += 20
    
    # SHORT: Tendencia bajista
    if last['ema_fast'] < last['ema_slow']:
        short_prob += 20
    
    # ADX fuerte
    if last['adx'] > params['adx_level']:
        long_prob += 20
        short_prob += 20
    
    # Quiebre o divergencia para LONG
    breakout_long = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
    if breakout_long or divergence == 'bullish':
        long_prob += 30
    
    # Quiebre o divergencia para SHORT
    breakdown_short = any(last['close'] < s * 0.995 for s in supports) if supports else False
    if breakdown_short or divergence == 'bearish':
        short_prob += 30
    
    # Volumen
    if volume_class in ['Alto', 'Muy Alto']:
        long_prob += 20
        short_prob += 20
    
    # Precio cerca de soporte (LONG) o resistencia (SHORT)
    if is_price_near_level(last['close'], supports, params['support_resistance_percent']):
        long_prob += 10
    if is_price_near_level(last['close'], resistances, params['support_resistance_percent']):
        short_prob += 10
    
    return min(long_prob, 100), min(short_prob, 100)

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Calcular probabilidades
        long_prob, short_prob = calculate_probability(df, last, supports, resistances, volume_class, divergence, params)
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Señales LONG
        long_signal = None
        if (trend == 'up' and last['adx'] > params['adx_level'] and 
            (is_breakout or divergence == 'bullish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar la resistencia más cercana por encima
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            # Encontrar el soporte más cercano por debajo para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * (1 - params['max_risk_percent']/100)
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
                'price': round(last['close'], 4),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'probability': long_prob
            }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar el soporte más cercano por debajo
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            # Encontrar la resistencia más cercana por encima para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * (1 + params['max_risk_percent']/100)
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
                'price': round(last['close'], 4),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'probability': short_prob
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        with app.app_context():
            try:
                app.logger.info("Iniciando actualización de datos...")
                cryptos = load_cryptos()
                long_signals = []
                short_signals = []
                scatter_data = []
                
                for crypto in cryptos:
                    long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                    
                    volume_level = get_volume_level(volume_class)
                    
                    scatter_data.append({
                        'symbol': crypto,
                        'long_prob': long_prob,
                        'short_prob': short_prob,
                        'volume_class': volume_class,
                        'volume_level': volume_level
                    })
                    
                    if long_signal:
                        long_signals.append(long_signal)
                    if short_signal:
                        short_signals.append(short_signal)
                
                # Ordenar por fuerza de tendencia (ADX)
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar caché
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                
                app.logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT")
            except Exception as e:
                app.logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()

@app.route('/')
def index():
    params = DEFAULTS.copy()
    
    # Si se envió el formulario de filtros
    if request.method == 'GET' and request.args:
        for key in params:
            if key in request.args:
                # Convertir a int si es numérico, de lo contrario dejar como está
                if key != 'timeframe':
                    try:
                        params[key] = int(request.args[key])
                    except:
                        pass
                else:
                    params[key] = request.args[key]
    
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Filtrar por volumen si se especifica
    volume_filter = request.args.get('volume_filter', 'all')
    if volume_filter != 'all':
        scatter_data = [d for d in scatter_data if d['volume_class'] == volume_filter]
    
    # Estadísticas para gráficos
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    # Preparar datos para el gráfico de dispersión
    scatter_labels = [d['symbol'] for d in scatter_data]
    scatter_long = [d['long_prob'] for d in scatter_data]
    scatter_short = [d['short_prob'] for d in scatter_data]
    scatter_volume = [d['volume_level'] for d in scatter_data]
    scatter_volume_class = [d['volume_class'] for d in scatter_data]
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=params,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_labels=scatter_labels,
                           scatter_long=scatter_long,
                           scatter_short=scatter_short,
                           scatter_volume=scatter_volume,
                           scatter_volume_class=scatter_volume_class,
                           volume_filter=volume_filter)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Datos no disponibles", 404
    
    df = calculate_indicators(df, DEFAULTS)
    
    # Buscar señal correspondiente
    signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
    signal = next((s for s in signals if s['symbol'] == symbol), None)
    
    if not signal:
        return "Señal no encontrada", 404
    
    plot_url = generate_chart(df, signal, signal_type)
    return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
