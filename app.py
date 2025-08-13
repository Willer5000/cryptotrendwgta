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
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300  # 5 minutos
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
    'price_distance_threshold': 1.0
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        cryptos = [line.strip() for line in f.readlines() if line.strip()]
        app.logger.info(f"Cargadas {len(cryptos)} criptomonedas")
        return cryptos

# Obtener datos de KuCoin con reintentos
def get_kucoin_data(symbol, timeframe, retries=3):
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
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    candles.reverse()  # Invertir para orden ascendente
                    
                    if len(candles) < 50:
                        app.logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    # Convertir a tipos numéricos
                    numeric_cols = ['open', 'close', 'high', 'low', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Eliminar filas con valores NaN
                    df = df.dropna()
                    
                    if len(df) < 30:
                        app.logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                        return None
                    
                    # Convertir timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    return df
            elif response.status_code == 429:
                wait_time = (2 ** attempt) + random.random()
                app.logger.warning(f"Rate limit alcanzado para {symbol}, reintento {attempt+1} en {wait_time}s")
                time.sleep(wait_time)
        except Exception as e:
            app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            time.sleep(1)
    
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Evitar división por cero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window=14):
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
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(window).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'], df['low'], df['close'], params['adx_period']
        )
        
        return df.dropna()
    except Exception as e:
        app.logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias simplificado
def find_support_resistance(df, window=50):
    try:
        # Usar máximos y mínimos locales
        df['min'] = df['low'].rolling(window, center=True).min()
        df['max'] = df['high'].rolling(window, center=True).max()
        
        supports = df[df['low'] <= df['min']]['low'].unique().tolist()
        resistances = df[df['high'] >= df['max']]['high'].unique().tolist()
        
        return supports[-5:], resistances[-5:]  # Solo los últimos 5 niveles
    except Exception as e:
        app.logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0:
            return 'Muy Bajo'
            
        ratio = current_vol / avg_vol
        if ratio > 2.0: return 'Muy Alto'
        if ratio > 1.5: return 'Alto'
        if ratio > 1.0: return 'Medio'
        if ratio > 0.5: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Analizar una criptomoneda y calcular probabilidades
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 30:
            return {
                'symbol': symbol,
                'long_signal': None,
                'short_signal': None,
                'long_prob': 0,
                'short_prob': 0,
                'volume': 'Muy Bajo'
            }
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return {
                'symbol': symbol,
                'long_signal': None,
                'short_signal': None,
                'long_prob': 0,
                'short_prob': 0,
                'volume': 'Muy Bajo'
            }
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        current_price = last['close']
        
        # Calcular probabilidades base
        long_prob = 0
        short_prob = 0
        
        # Tendencia
        if last['ema_fast'] > last['ema_slow']:
            long_prob += 30
        else:
            short_prob += 30
        
        # Fuerza de tendencia
        if last['adx'] > params['adx_level']:
            if last['plus_di'] > last['minus_di']:
                long_prob += 25
            else:
                short_prob += 25
        
        # Soportes/resistencias
        if supports and current_price <= min(supports) * 1.01:
            long_prob += 20
        if resistances and current_price >= max(resistances) * 0.99:
            short_prob += 20
        
        # Volumen
        if volume_class in ['Alto', 'Muy Alto']:
            long_prob += 15
            short_prob += 15
        
        # Asegurar no exceder 100%
        long_prob = min(long_prob, 100)
        short_prob = min(short_prob, 100)
        
        # Generar señales solo si probabilidad > 70%
        long_signal = None
        short_signal = None
        
        # Señal LONG
        if long_prob > 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = current_price * 1.005  # Entrada 0.5% arriba
            sl = current_price * 0.98      # Stop loss 2% abajo
            risk = entry - sl
            
            long_signal = {
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'price': round(current_price, 4),
                'distance': round(((entry - current_price) / current_price) * 100, 2),
                'adx': round(last['adx'], 2)
            }
        
        # Señal SHORT
        if short_prob > 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = current_price * 0.995  # Entrada 0.5% abajo
            sl = current_price * 1.02      # Stop loss 2% arriba
            risk = sl - entry
            
            short_signal = {
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'price': round(current_price, 4),
                'distance': round(((current_price - entry) / current_price) * 100, 2),
                'adx': round(last['adx'], 2)
            }
        
        return {
            'symbol': symbol,
            'long_signal': long_signal,
            'short_signal': short_signal,
            'long_prob': int(long_prob),
            'short_prob': int(short_prob),
            'volume': volume_class
        }
        
    except Exception as e:
        app.logger.error(f"Error analizando {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'long_signal': None,
            'short_signal': None,
            'long_prob': 0,
            'short_prob': 0,
            'volume': 'Muy Bajo'
        }

# Tarea para actualizar datos
def update_crypto_data():
    while True:
        try:
            start_time = time.time()
            app.logger.info("Iniciando actualización de datos...")
            cryptos = load_cryptos()
            
            long_signals = []
            short_signals = []
            scatter_data = []
            
            # Procesar en paralelo
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(analyze_crypto, crypto, DEFAULTS) for crypto in cryptos]
                
                for future in as_completed(futures):
                    result = future.result()
                    scatter_data.append({
                        'symbol': result['symbol'],
                        'long_prob': result['long_prob'],
                        'short_prob': result['short_prob'],
                        'volume': result['volume']
                    })
                    
                    if result['long_signal']:
                        long_signals.append({
                            'symbol': result['symbol'],
                            **result['long_signal']
                        })
                    
                    if result['short_signal']:
                        short_signals.append({
                            'symbol': result['symbol'],
                            **result['short_signal']
                        })
            
            # Ordenar por ADX
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar caché
            cache.set('long_signals', long_signals)
            cache.set('short_signals', short_signals)
            cache.set('scatter_data', scatter_data)
            cache.set('last_update', datetime.now())
            
            elapsed = time.time() - start_time
            app.logger.info(f"Actualización completada en {elapsed:.2f}s: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} puntos")
        except Exception as e:
            app.logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
def start_update_thread():
    try:
        thread = threading.Thread(target=update_crypto_data, daemon=True)
        thread.start()
        app.logger.info("Hilo de actualización iniciado")
    except Exception as e:
        app.logger.error(f"No se pudo iniciar hilo: {str(e)}")

start_update_thread()

@app.route('/')
def index():
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Estadísticas
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    app.logger.info(f"Enviando a frontend: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} puntos")
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=DEFAULTS,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=json.dumps(scatter_data))

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
        if df is None:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, DEFAULTS)
        if df is None or len(df) < 20:
            return "Datos insuficientes", 404
        
        # Buscar señal en caché
        signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        # Generar gráfico
        plt.figure(figsize=(12, 8))
        
        # Precio y EMAs
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue')
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', alpha=0.7)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', alpha=0.7)
        
        # Líneas de entrada y stop
        plt.axhline(y=signal['entry'], color='green' if signal_type == 'long' else 'red', 
                    linestyle='--', label='Entrada')
        plt.axhline(y=signal['sl'], color='red' if signal_type == 'long' else 'green', 
                    linestyle='--', label='Stop Loss')
        
        plt.title(f'{symbol}-USDT - Señal {signal_type.upper()}')
        plt.legend()
        plt.grid(True)
        
        # Volumen
        plt.subplot(3, 1, 2)
        plt.bar(df['timestamp'], df['volume'], color='blue', alpha=0.7)
        plt.title('Volumen')
        plt.grid(True)
        
        # Indicadores
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
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
        
    except Exception as e:
        app.logger.error(f"Error generando gráfico: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/update', methods=['POST'])
def update_params():
    try:
        # Actualizar parámetros desde el formulario
        for key in DEFAULTS.keys():
            if key in request.form:
                # Convertir a número si es posible
                try:
                    DEFAULTS[key] = float(request.form[key])
                except:
                    DEFAULTS[key] = request.form[key]
        
        return jsonify({'status': 'success', 'params': DEFAULTS})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
