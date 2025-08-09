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
import math

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

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
    'volume_filter': 'all'
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
                # KuCoin devuelve las velas en orden descendente, invertimos para tener ascendente
                candles.reverse()
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                # Convertir timestamp a entero para evitar el warning
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
    return rsi

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
    return adx, plus_di, minus_di

# Implementación manual de ATR
def calculate_atr(high, low, close, window):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
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
    if max_idx is not None:
        price_high = df.loc[max_idx, 'close']
        rsi_high = df.loc[max_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx is not None:
        price_low = df.loc[min_idx, 'close']
        rsi_low = df.loc[min_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
            return 'bullish'
    
    return None

# Calcular probabilidades LONG/SHORT
def calculate_probabilities(df, last, supports, resistances, volume_class):
    long_prob = 0
    short_prob = 0
    
    # Probabilidad LONG
    # Tendencia alcista (20%)
    if last['ema_fast'] > last['ema_slow']:
        long_prob += 20
    
    # ADX > 25 (20%)
    if last['adx'] > DEFAULTS['adx_level']:
        long_prob += 20
    
    # Quiebre o divergencia alcista (30%)
    is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
    divergence = detect_divergence(df, DEFAULTS['divergence_lookback'])
    if is_breakout or divergence == 'bullish':
        long_prob += 30
    
    # Volumen Alto/Muy Alto (20%)
    if volume_class in ['Alto', 'Muy Alto']:
        long_prob += 20
    
    # Precio cerca de soporte (10%)
    if supports:
        closest_support = max([s for s in supports if s < last['close']], default=None)
        if closest_support and (last['close'] - closest_support) / last['close'] <= 0.02:  # 2%
            long_prob += 10
    
    # Probabilidad SHORT
    # Tendencia bajista (20%)
    if last['ema_fast'] < last['ema_slow']:
        short_prob += 20
    
    # ADX > 25 (20%)
    if last['adx'] > DEFAULTS['adx_level']:
        short_prob += 20
    
    # Quiebre o divergencia bajista (30%)
    is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
    if is_breakdown or divergence == 'bearish':
        short_prob += 30
    
    # Volumen Alto/Muy Alto (20%)
    if volume_class in ['Alto', 'Muy Alto']:
        short_prob += 20
    
    # Precio cerca de resistencia (10%)
    if resistances:
        closest_resistance = min([r for r in resistances if r > last['close']], default=None)
        if closest_resistance and (closest_resistance - last['close']) / last['close'] <= 0.02:  # 2%
            short_prob += 10
    
    return min(long_prob, 100), min(short_prob, 100)

# Generar gráfico para una señal
def generate_chart(df, signal, signal_type):
    plt.figure(figsize=(12, 8))
    
    # Gráfico de precio
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='Precio', color='blue')
    plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', alpha=0.7)
    plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', alpha=0.7)
    
    # Marcar entrada y SL
    if signal_type == 'long':
        plt.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada')
        plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
    else:
        plt.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada')
        plt.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss')
    
    plt.title(f'{signal["symbol"]} - Precio y EMAs')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de volumen
    plt.subplot(3, 1, 2)
    plt.bar(df['timestamp'], df['volume'], color=np.where(df['close'] > df['open'], 'green', 'red'))
    plt.title('Volumen')
    plt.grid(True)
    
    # Gráfico de indicadores
    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    
    plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
    plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.5)
    
    plt.title('Indicadores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Convertir a base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Calcular probabilidades
        long_prob, short_prob = calculate_probabilities(df, last, supports, resistances, volume_class)
        
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
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
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
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
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
                probability_data = []
                
                for crypto in cryptos:
                    long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                    if long_signal:
                        long_signals.append(long_signal)
                    if short_signal:
                        short_signals.append(short_signal)
                    
                    # Guardar datos para el gráfico de probabilidades
                    probability_data.append({
                        'symbol': crypto,
                        'long_prob': long_prob,
                        'short_prob': short_prob,
                        'volume': volume_class
                    })
                
                # Ordenar por fuerza de tendencia (ADX)
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar caché
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('probability_data', probability_data)
                cache.set('last_update', datetime.now())
                
                app.logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT")
            except Exception as e:
                app.logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Actualizar parámetros desde el formulario
        for key in DEFAULTS:
            if key in request.form:
                # Convertir a int si es numérico, de lo contrario dejar como está
                if key not in ['timeframe', 'volume_filter']:
                    DEFAULTS[key] = int(request.form[key])
                else:
                    DEFAULTS[key] = request.form[key]
    
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    probability_data = cache.get('probability_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Filtrar datos de probabilidad según el filtro de volumen
    filtered_prob_data = []
    if DEFAULTS['volume_filter'] == 'all':
        filtered_prob_data = probability_data
    else:
        filtered_prob_data = [item for item in probability_data if item['volume'] == DEFAULTS['volume_filter']]
    
    # Preparar datos para el gráfico de dispersión
    scatter_data = []
    for item in filtered_prob_data:
        scatter_data.append({
            'x': item['short_prob'],
            'y': item['long_prob'],
            'symbol': item['symbol'],
            'volume': item['volume']
        })
    
    # Estadísticas para gráficos
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           scatter_data=scatter_data,
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2))

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
