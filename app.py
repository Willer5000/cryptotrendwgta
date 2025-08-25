import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import logging
import traceback
from threading import Lock, Event
from collections import deque
import pytz
import calendar
from dateutil.relativedelta import relativedelta

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Deshabilitar caché

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300  # 5 minutos
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 5  # Reducido para mejor rendimiento en Render

# Zona horaria de Nueva York (UTC-4 o UTC-5 según horario de verano)
try:
    NY_TZ = pytz.timezone('America/New_York')
except:
    NY_TZ = pytz.timezone('UTC')

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

# Estado global con bloqueo
analysis_state = {
    'long_signals': [],
    'short_signals': [],
    'scatter_data': [],
    'historical_signals': deque(maxlen=100),
    'current_signals': deque(maxlen=100),
    'last_update': datetime.now(),
    'cryptos_analyzed': 0,
    'is_updating': False,
    'update_progress': 0,
    'params': DEFAULTS.copy(),
    'lock': Lock(),
    'timeframe_data': {},
    'update_event': Event(),
    'force_update': False
}

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Cargadas {len(cryptos)} criptomonedas desde archivo")
            return cryptos
    except Exception as e:
        logger.error(f"Error cargando criptomonedas: {str(e)}")
        # Lista reducida para pruebas
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'LINK']

# Obtener datos de KuCoin con reintentos
def get_kucoin_data(symbol, timeframe):
    tf_mapping = {
        '15m': '15min',
        '30m': '30min',
        '1h': '1hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': '1day',
        '1w': '1week'
    }
    kucoin_tf = tf_mapping.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=30)  # Aumentado timeout
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    candles.reverse()
                    
                    if len(candles) < 50:  # Reducido el mínimo requerido
                        logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) < 30:  # Reducido el mínimo después de limpieza
                        logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                        return None
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
                    df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                    
                    return df
                else:
                    logger.warning(f"Respuesta no válida de KuCoin para {symbol}: {data.get('msg')}")
                    return None
            else:
                logger.warning(f"Error HTTP {response.status_code} para {symbol}, reintento {attempt+1}/{MAX_RETRIES}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} ({timeframe}): {str(e)}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    return None

# Implementación optimizada de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación optimizada de RSI
def calculate_rsi(series, window=14):
    if len(series) < window + 1:
        return pd.Series([50] * len(series))
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación optimizada de ADX
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window).mean().fillna(tr)
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = dx.rolling(window).mean().fillna(0)
        return adx, plus_di, minus_di
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))

# Calcular indicadores
def calculate_indicators(df, params):
    try:
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'], _, _ = calculate_adx(
            df['high'], 
            df['low'], 
            df['close'], 
            params['adx_period']
        )
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias optimizado
def find_support_resistance(df, window=50):
    try:
        if len(df) < window:
            return [], []
        
        # Usar un método más simple para S/R
        supports = df['low'].rolling(window=window, center=True).min().dropna().unique().tolist()
        resistances = df['high'].rolling(window=window, center=True).max().dropna().unique().tolist()
        
        # Consolidar niveles cercanos
        def consolidate(levels, threshold=0.01):
            if len(levels) == 0:
                return []
                
            levels.sort()
            consolidated = []
            current = levels[0]
            
            for level in levels[1:]:
                if abs(level - current) / current <= threshold:
                    current = (current + level) / 2
                else:
                    consolidated.append(current)
                    current = level
                    
            consolidated.append(current)
            return consolidated
        
        return consolidate(supports), consolidate(resistances)
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0 or current_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.0: return 'Muy Alto'
        if ratio > 1.5: return 'Alto'
        if ratio > 1.0: return 'Medio'
        if ratio > 0.5: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias (simplificado)
def detect_divergence(df, lookback=14):
    try:
        if len(df) < lookback + 1:
            return None
            
        # Simplificar la detección de divergencia
        recent = df.iloc[-lookback:]
        rsi_recent = recent['rsi']
        
        # Buscar divergencias simples
        price_high_idx = recent['high'].idxmax()
        rsi_high_idx = rsi_recent.idxmax()
        
        price_low_idx = recent['low'].idxmin()
        rsi_low_idx = rsi_recent.idxmin()
        
        # Divergencia bajista: precio hace nuevo alto pero RSI no
        if price_high_idx > rsi_high_idx and recent.loc[price_high_idx, 'high'] > recent['high'].mean():
            return 'bearish'
        
        # Divergencia alcista: precio hace nuevo bajo pero RSI no
        if price_low_idx > rsi_low_idx and recent.loc[price_low_idx, 'low'] < recent['low'].mean():
            return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Calcular distancia al nivel más cercano
def near_level(price, levels, threshold_percent=1.0):
    if not levels or price <= 0:
        return False
    
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    return min_distance <= threshold

# Obtener timestamp de inicio de vela anterior
def get_previous_candle_start(timeframe):
    now = datetime.now(NY_TZ)
    
    if timeframe == '15m':
        minutes = (now.minute // 15) * 15
        current_candle_start = now.replace(minute=minutes, second=0, microsecond=0)
        return current_candle_start - timedelta(minutes=15)
    elif timeframe == '30m':
        minutes = (now.minute // 30) * 30
        current_candle_start = now.replace(minute=minutes, second=0, microsecond=0)
        return current_candle_start - timedelta(minutes=30)
    elif timeframe == '1h':
        current_candle_start = now.replace(minute=0, second=0, microsecond=0)
        return current_candle_start - timedelta(hours=1)
    elif timeframe == '2h':
        hours = (now.hour // 2) * 2
        current_candle_start = now.replace(hour=hours, minute=0, second=0, microsecond=0)
        return current_candle_start - timedelta(hours=2)
    elif timeframe == '4h':
        hours = (now.hour // 4) * 4
        current_candle_start = now.replace(hour=hours, minute=0, second=0, microsecond=0)
        return current_candle_start - timedelta(hours=4)
    elif timeframe == '1d':
        current_candle_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return current_candle_start - timedelta(days=1)
    elif timeframe == '1w':
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_week - timedelta(weeks=1)
    
    return now - timedelta(hours=1)

# Formatear eje X según timeframe
def format_xaxis_by_timeframe(timeframe, ax):
    try:
        if timeframe == '15m' or timeframe == '30m':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        elif timeframe == '1h' or timeframe == '2h':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        elif timeframe == '4h':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        elif timeframe == '1d':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        elif timeframe == '1w':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    except Exception as e:
        logger.error(f"Error formateando eje X: {str(e)}")

# Analizar una criptomoneda (versión simplificada)
def analyze_crypto(symbol, params, analyze_previous=False):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 30:
            return None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        # Determinar qué vela analizar
        if analyze_previous and len(df) > 1:
            last = df.iloc[-2]  # Penúltima vela para histórico
        else:
            last = df.iloc[-1]  # Última vela para actual
        
        # Calcular volumen promedio
        avg_vol = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Simplificar cálculo de probabilidades
        trend_up = last['ema_fast'] > last['ema_slow']
        trend_down = last['ema_fast'] < last['ema_slow']
        strong_trend = last['adx'] > params['adx_level']
        
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if trend_up: 
            long_prob += 30
            if strong_trend: long_prob += 20
        if last['rsi'] < 40: long_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 20
        
        # Factores para SHORT
        if trend_down: 
            short_prob += 30
            if strong_trend: short_prob += 20
        if last['rsi'] > 60: short_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 20
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Solo generar señales si hay suficiente probabilidad y volumen
        long_signal = None
        short_signal = None
        
        if long_prob >= 60 and volume_class in ['Alto', 'Muy Alto']:
            # Calcular niveles de entrada, SL y TP de forma simplificada
            entry = last['close'] * 1.005  # Entrada ligeramente arriba del precio actual
            sl = last['close'] * 0.98      # Stop loss al 2%
            risk = entry - sl
            tp1 = entry + risk             # TP1 con ratio 1:1
            tp2 = entry + risk * 2         # TP2 con ratio 1:2
            
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'LONG',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'] if 'timestamp' in last else datetime.now(NY_TZ)
            }
        
        if short_prob >= 60 and volume_class in ['Alto', 'Muy Alto']:
            # Calcular niveles de entrada, SL y TP de forma simplificada
            entry = last['close'] * 0.995  # Entrada ligeramente abajo del precio actual
            sl = last['close'] * 1.02      # Stop loss al 2%
            risk = sl - entry
            tp1 = entry - risk             # TP1 con ratio 1:1
            tp2 = entry - risk * 2         # TP2 con ratio 1:2
            
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'SHORT',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'] if 'timestamp' in last else datetime.now(NY_TZ)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
        
    except Exception as e:
        logger.error(f"Error analizando {symbol} ({params['timeframe']}): {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea de actualización
def update_task():
    while True:
        try:
            with analysis_state['lock']:
                analysis_state['is_updating'] = True
                analysis_state['update_progress'] = 0
                
                cryptos = load_cryptos()
                total = len(cryptos)
                processed = 0
                
                params = analysis_state['params']
                current_timeframe = params['timeframe']
                
                logger.info(f"Iniciando análisis de {total} criptomonedas para timeframe {current_timeframe}...")
                
                # Inicializar estructuras para este timeframe
                if current_timeframe not in analysis_state['timeframe_data']:
                    analysis_state['timeframe_data'][current_timeframe] = {
                        'long_signals': [],
                        'short_signals': [],
                        'scatter_data': [],
                        'historical_signals': deque(maxlen=100)
                    }
                
                timeframe_data = analysis_state['timeframe_data'][current_timeframe]
                long_signals = []
                short_signals = []
                scatter_data = []
                
                # Obtener timestamp de inicio de vela anterior
                previous_candle_start = get_previous_candle_start(current_timeframe)
                
                # Procesar criptomonedas en lotes
                for i in range(0, total, BATCH_SIZE):
                    batch = cryptos[i:i+BATCH_SIZE]
                    
                    for crypto in batch:
                        try:
                            # Analizar vela actual
                            long_sig, short_sig, long_prob, short_prob, vol = analyze_crypto(crypto, params, analyze_previous=False)
                            
                            if long_sig:
                                long_signals.append(long_sig)
                            
                            if short_sig:
                                short_signals.append(short_sig)
                            
                            scatter_data.append({
                                'symbol': crypto,
                                'long_prob': long_prob,
                                'short_prob': short_prob,
                                'volume': vol
                            })
                            
                            # Analizar vela anterior para señales históricas
                            long_sig_prev, short_sig_prev, _, _, _ = analyze_crypto(crypto, params, analyze_previous=True)
                            
                            if long_sig_prev:
                                timeframe_data['historical_signals'].append(long_sig_prev)
                            
                            if short_sig_prev:
                                timeframe_data['historical_signals'].append(short_sig_prev)
                            
                            processed += 1
                            progress = int((processed / total) * 100)
                            analysis_state['update_progress'] = progress
                            
                        except Exception as e:
                            logger.error(f"Error procesando {crypto}: {str(e)}")
                            processed += 1
                            progress = int((processed / total) * 100)
                            analysis_state['update_progress'] = progress
                    
                    # Pequeña pausa entre lotes para no saturar la API
                    time.sleep(1)
                
                # Ordenar señales por fuerza de tendencia (ADX)
                long_signals.sort(key=lambda x: x['adx'] if x else 0, reverse=True)
                short_signals.sort(key=lambda x: x['adx'] if x else 0, reverse=True)
                
                # Actualizar datos del timeframe
                timeframe_data['long_signals'] = long_signals
                timeframe_data['short_signals'] = short_signals
                timeframe_data['scatter_data'] = scatter_data
                
                # Actualizar estado global
                analysis_state['cryptos_analyzed'] = total
                analysis_state['last_update'] = datetime.now()
                analysis_state['is_updating'] = False
                
                logger.info(f"Análisis completado para {current_timeframe}: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(timeframe_data['historical_signals'])} históricas")
                
        except Exception as e:
            logger.error(f"Error crítico en actualización: {str(e)}")
            traceback.print_exc()
            analysis_state['is_updating'] = False
        
        # Esperar hasta la próxima actualización (5 minutos) o hasta que se fuerce una actualización
        analysis_state['update_event'].clear()
        analysis_state['update_event'].wait(CACHE_TIME)
        
        # Verificar si se necesita una actualización forzada
        if analysis_state.get('force_update', False):
            analysis_state['force_update'] = False
            continue

# Iniciar hilo de actualización
update_thread = threading.Thread(target=update_task, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

@app.route('/')
def index():
    with analysis_state['lock']:
        params = analysis_state['params']
        current_timeframe = params['timeframe']
        
        # Obtener datos del timeframe actual
        if current_timeframe in analysis_state['timeframe_data']:
            timeframe_data = analysis_state['timeframe_data'][current_timeframe]
            long_signals = timeframe_data['long_signals'][:50]
            short_signals = timeframe_data['short_signals'][:50]
            scatter_data = timeframe_data['scatter_data']
            historical_signals = list(timeframe_data['historical_signals'])[-20:]  # Últimas 20 señales
        else:
            # Si no hay datos para este timeframe, usar estructuras vacías
            long_signals = []
            short_signals = []
            scatter_data = []
            historical_signals = []
        
        last_update = analysis_state['last_update']
        cryptos_analyzed = analysis_state['cryptos_analyzed']
        
        # Estadísticas
        avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
        avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
        
        # Preparar datos para gráfico de dispersión
        scatter_ready = []
        for item in scatter_data:
            # Asegurar que las probabilidades sean números válidos
            long_prob = max(0, min(100, item.get('long_prob', 0)))
            short_prob = max(0, min(100, item.get('short_prob', 0)))
            scatter_ready.append({
                'symbol': item['symbol'],
                'long_prob': long_prob,
                'short_prob': short_prob,
                'volume': item['volume']
            })
        
        # Ordenar señales históricas por timestamp
        historical_signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return render_template('index.html', 
                               long_signals=long_signals, 
                               short_signals=short_signals,
                               historical_signals=historical_signals,
                               last_update=last_update,
                               params=params,
                               avg_adx_long=round(avg_adx_long, 1),
                               avg_adx_short=round(avg_adx_short, 1),
                               scatter_data=scatter_ready,
                               cryptos_analyzed=cryptos_analyzed,
                               is_updating=analysis_state['is_updating'],
                               update_progress=analysis_state['update_progress'])

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        params = analysis_state['params']
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 30:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return "Datos insuficientes", 404
        
        # Buscar señal en el timeframe actual
        current_timeframe = params['timeframe']
        signal = None
        
        if current_timeframe in analysis_state['timeframe_data']:
            if signal_type == 'long':
                signals = analysis_state['timeframe_data'][current_timeframe]['long_signals']
            else:
                signals = analysis_state['timeframe_data'][current_timeframe]['short_signals']
                
            signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            # Si no encontramos la señal, crear una básica para mostrar el gráfico
            last = df.iloc[-1]
            signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(last['close'] * 1.01, 4) if signal_type == 'long' else round(last['close'] * 0.99, 4),
                'sl': round(last['close'] * 0.98, 4) if signal_type == 'long' else round(last['close'] * 1.02, 4),
                'tp1': round(last['close'] * 1.03, 4) if signal_type == 'long' else round(last['close'] * 0.97, 4),
                'tp2': round(last['close'] * 1.05, 4) if signal_type == 'long' else round(last['close'] * 0.95, 4),
                'type': signal_type.upper()
            }
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', alpha=0.8)
        
        # Marcar niveles clave
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        
        # Formatear eje X según timeframe
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        
        plt.title(f'{signal["symbol"]} - Precio y EMAs ({params["timeframe"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        plt.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        
        # Formatear eje X según timeframe
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        
        plt.title('Volumen')
        plt.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        # Formatear eje X según timeframe
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        
        plt.title('Indicadores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/historical_chart/<symbol>/<signal_type>')
def get_historical_chart(symbol, signal_type):
    try:
        params = analysis_state['params']
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 30:
            return "Datos insuficientes", 404
        
        # Buscar señal histórica en el timeframe actual
        current_timeframe = params['timeframe']
        historical_signals = []
        
        if current_timeframe in analysis_state['timeframe_data']:
            timeframe_data = analysis_state['timeframe_data'][current_timeframe]
            
            for signal in timeframe_data['historical_signals']:
                if signal['symbol'] == symbol and signal['type'].lower() == signal_type:
                    historical_signals.append(signal)
        
        if not historical_signals:
            return "Señal histórica no encontrada", 404
        
        signal = historical_signals[-1]  # La más reciente
        
        # Crear gráfico histórico
        plt.figure(figsize=(12, 8))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', alpha=0.8)
        
        # Marcar niveles clave
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        
        # Marcar la vela anterior
        prev_candle_start = get_previous_candle_start(params['timeframe'])
        prev_candle_idx = df[df['timestamp'] >= prev_candle_start].index
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                plt.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.5, label='Vela Anterior')
        
        # Formatear eje X según timeframe
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        
        plt.title(f'{signal["symbol"]} - Señal Histórica {signal_type.upper()} ({params["timeframe"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        plt.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        
        # Marcar volumen de la vela anterior
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                plt.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.5)
        
        # Formatear eje X según timeframe
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        
        plt.title('Volumen')
        plt.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        # Marcar la vela anterior
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                plt.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.5)
        
        # Formatear eje X según timeframe
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        
        plt.title('Indicadores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('historical_chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico histórico: {str(e)}")
        return "Error generando gráfico histórico", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/update_params', methods=['POST'])
def update_params():
    try:
        # Leer datos del formulario
        data = request.form.to_dict()
        
        # Actualizar parámetros
        new_params = analysis_state['params'].copy()
        
        for param in new_params:
            if param in data:
                value = data[param]
                # Conversión de tipos
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    new_params[param] = int(value)
                elif param in ['max_risk_percent', 'price_distance_threshold']:
                    new_params[param] = float(value)
                else:
                    new_params[param] = value
        
        with analysis_state['lock']:
            analysis_state['params'] = new_params
            # Forzar una actualización inmediata
            analysis_state['force_update'] = True
            analysis_state['update_event'].set()
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados correctamente. El sistema comenzará a analizar con el nuevo timeframe.',
            'params': new_params
        })
    except Exception as e:
        logger.error(f"Error actualizando parámetros: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error actualizando parámetros: {str(e)}"
        }), 500

@app.route('/status')
def status():
    with analysis_state['lock']:
        return jsonify({
            'last_update': analysis_state['last_update'].isoformat(),
            'is_updating': analysis_state['is_updating'],
            'progress': analysis_state['update_progress'],
            'long_signals': len(analysis_state['timeframe_data'].get(analysis_state['params']['timeframe'], {}).get('long_signals', [])),
            'short_signals': len(analysis_state['timeframe_data'].get(analysis_state['params']['timeframe'], {}).get('short_signals', [])),
            'historical_signals': len(analysis_state['timeframe_data'].get(analysis_state['params']['timeframe'], {}).get('historical_signals', [])),
            'cryptos_analyzed': analysis_state['cryptos_analyzed'],
            'params': analysis_state['params']
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
