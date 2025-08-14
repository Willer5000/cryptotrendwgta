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
import random

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuración mejorada
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
    'price_distance_threshold': 1.0
}

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip().split(':')[0] for line in f.readlines() if line.strip()]
            return cryptos
    except Exception as e:
        logger.error(f"Error loading cryptos: {str(e)}")
        return []

# Obtener datos de KuCoin (versión robusta)
def get_kucoin_data(symbol, timeframe):
    TIMEFRAME_MAP = {
        '30m': '30min',
        '1h': '1hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': '1day',
        '1w': '1week'
    }
    
    kucoin_tf = TIMEFRAME_MAP.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    try:
        response = requests.get(url, timeout=25)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                
                # Validar suficientes datos
                if len(candles) < 100:
                    logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                    return None
                
                # Crear DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.iloc[::-1]  # Invertir orden
                
                # Convertir tipos
                numeric_cols = ['open', 'close', 'high', 'low', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                df = df.dropna()
                
                if len(df) < 50:
                    logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                    return None
                
                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
        else:
            logger.error(f"Error API KuCoin para {symbol}: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementaciones manuales robustas de indicadores
def calculate_ema(series, window):
    try:
        return series.ewm(span=window, adjust=False).mean()
    except:
        return series

def calculate_rsi(series, window=14):
    try:
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series([50] * len(series))

def calculate_adx(high, low, close, window):
    try:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        
        up_move = high.diff()
        down_move = low.diff().abs() * -1
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window).mean()
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
        return pd.Series([0]*len(high)), pd.Series([0]*len(high)), pd.Series([0]*len(high))

# Detectar soportes/resistencias mejorado
def find_support_resistance(df, window=50):
    try:
        highs = df['high'].rolling(window, min_periods=1).max()
        lows = df['low'].rolling(window, min_periods=1).min()
        
        resistances = df[df['high'] >= highs]['high'].unique()
        supports = df[df['low'] <= lows]['low'].unique()
        
        return sorted(supports), sorted(resistances, reverse=True)
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen robusto
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol == 0: return 'Muy Bajo'
        ratio = current_vol / avg_vol
        
        if ratio > 2.0: return 'Muy Alto'
        if ratio > 1.5: return 'Alto'
        if ratio > 1.0: return 'Medio'
        if ratio > 0.5: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias mejorada
def detect_divergence(df, lookback=14):
    try:
        # Solo últimos 50 datos para eficiencia
        recent = df.iloc[-50:]
        
        # Encontrar máximos y mínimos locales
        highs = recent['high'].values
        lows = recent['low'].values
        rsi = recent['rsi'].values
        
        max_idx = np.argmax(highs)
        min_idx = np.argmin(lows)
        
        # Divergencia bajista: precio hace nuevo alto pero RSI no
        if highs[-1] > highs[max_idx] and rsi[-1] < rsi[max_idx]:
            return 'bearish'
        
        # Divergencia alcista: precio hace nuevo bajo pero RSI no
        if lows[-1] < lows[min_idx] and rsi[-1] > rsi[min_idx]:
            return 'bullish'
            
        return None
    except:
        return None

# Analizar una criptomoneda (versión optimizada)
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        return None, None, 50, 50, 'Muy Bajo'
    
    try:
        # Calcular indicadores
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # Datos actuales
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['divergence_lookback'])
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Calcular probabilidades
        long_prob = 50
        short_prob = 50
        
        # Factores para LONG
        if last['ema_fast'] > last['ema_slow']:
            long_prob += 20
        if last['adx'] > params['adx_level']:
            long_prob += 15
        if divergence == 'bullish':
            long_prob += 15
        if volume_class in ['Alto', 'Muy Alto']:
            long_prob += 10
        if any(abs(last['close'] - s) < (last['close'] * 0.01) for s in supports):
            long_prob += 10
            
        # Factores para SHORT
        if last['ema_fast'] < last['ema_slow']:
            short_prob += 20
        if last['adx'] > params['adx_level']:
            short_prob += 15
        if divergence == 'bearish':
            short_prob += 15
        if volume_class in ['Alto', 'Muy Alto']:
            short_prob += 10
        if any(abs(last['close'] - r) < (last['close'] * 0.01) for r in resistances):
            short_prob += 10
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        long_prob = int((long_prob / total) * 100)
        short_prob = int((short_prob / total) * 100)
        
        # Generar señales solo si superan umbral
        long_signal = None
        short_signal = None
        
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = min([r for r in resistances if r > last['close']] or [last['close'] * 1.01])
            sl = max([s for s in supports if s < entry] or [entry * 0.99])
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'volume': volume_class,
                'adx': round(last['adx'], 1)
            }
        
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = max([s for s in supports if s < last['close']] or [last['close'] * 0.99])
            sl = min([r for r in resistances if r > entry] or [entry * 1.01])
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'volume': volume_class,
                'adx': round(last['adx'], 1)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
        
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, 50, 50, 'Muy Bajo'

# Tarea de actualización en segundo plano
def background_update():
    while True:
        start_time = time.time()
        try:
            logger.info("Iniciando actualización de datos...")
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            
            # Procesar cada cripto
            for i, crypto in enumerate(cryptos):
                try:
                    long_sig, short_sig, long_prob, short_prob, vol_class = analyze_crypto(crypto, DEFAULTS)
                    
                    if long_sig:
                        long_signals.append(long_sig)
                    if short_sig:
                        short_signals.append(short_sig)
                    
                    scatter_data.append({
                        'symbol': crypto,
                        'long_prob': long_prob,
                        'short_prob': short_prob,
                        'volume': vol_class
                    })
                    
                    # Pequeña pausa cada 10 criptos para no saturar
                    if i % 10 == 0:
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error procesando {crypto}: {str(e)}")
            
            # Ordenar por ADX
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar caché
            cache.set('long_signals', long_signals)
            cache.set('short_signals', short_signals)
            cache.set('scatter_data', scatter_data)
            cache.set('last_update', datetime.now())
            
            elapsed = time.time() - start_time
            logger.info(f"Actualización completada en {elapsed:.2f}s: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} puntos")
            
        except Exception as e:
            logger.error(f"Error grave en actualización: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
try:
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    logger.info("Hilo de actualización iniciado")
except Exception as e:
    logger.error(f"No se pudo iniciar hilo de actualización: {str(e)}")

@app.route('/')
def index():
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    
    # Estadísticas
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
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
        if df is None or len(df) < 50:
            return "Datos no disponibles", 404
        
        # Buscar señal en caché
        signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        # Generar gráfico simple
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['close'], label='Precio')
        plt.axhline(y=signal['entry'], color='g', linestyle='--', label='Entrada')
        plt.axhline(y=signal['sl'], color='r', linestyle='--', label='Stop Loss')
        plt.title(f"{symbol} - Señal {signal_type.upper()}")
        plt.legend()
        plt.grid(True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
        
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        return "Error interno", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
