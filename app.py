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
BATCH_SIZE = 5

# Zona horaria de Nueva York
try:
    NY_TZ = pytz.timezone('America/New_York')
except:
    NY_TZ = pytz.timezone('UTC')

DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 20,
    'ema_trend': 200,  # Nueva EMA para tendencia mayor
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 50,
    'divergence_lookback': 20,
    'max_risk_percent': 1.5,
    'price_distance_threshold': 1.0,
    'volume_ma_window': 20  # Nueva: ventana para MA de volumen
}

# Estado global con bloqueo
analysis_state = {
    'timeframe_data': {},  # Almacena datos por timeframe: {'1h': {data}, '4h': {data}}
    'last_update': datetime.now(),
    'is_updating': False,
    'update_progress': 0,
    'params': DEFAULTS.copy(),
    'lock': Lock(),
    'update_event': Event(),
    'current_timeframe': DEFAULTS['timeframe']
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
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    candles.reverse()
                    
                    if len(candles) < 100:
                        logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) < 50:
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
        df['ema_trend'] = calculate_ema(df['close'], params['ema_trend'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'], _, _ = calculate_adx(
            df['high'], 
            df['low'], 
            df['close'], 
            params['adx_period']
        )
        
        # Calcular MA de volumen para volumen relativo
        df['volume_ma'] = df['volume'].rolling(window=params['volume_ma_window']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias optimizado
def find_support_resistance(df, window=50):
    try:
        if len(df) < window:
            return [], []
        
        df['min'] = df['low'].rolling(window=3, center=True).min()
        df['max'] = df['high'].rolling(window=3, center=True).max()
        
        supports = df[df['low'] == df['min']]['low'].values
        resistances = df[df['high'] == df['max']]['high'].values
        
        def consolidate(levels, threshold=0.01):
            if len(levels) == 0:
                return []
                
            levels.sort()
            consolidated = []
            current = levels[0]
            
            for level in levels[1:]:
                if level <= current * (1 + threshold):
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

# Clasificar volumen con volumen relativo
def classify_volume(current_vol, avg_vol, volume_ratio):
    try:
        if avg_vol <= 0 or current_vol <= 0:
            return 'Muy Bajo', 0
        
        ratio = current_vol / avg_vol
        
        # Considerar tanto el ratio absoluto como el relativo
        volume_score = (ratio + volume_ratio) / 2
        
        if volume_score > 2.5: return 'Muy Alto', volume_score
        if volume_score > 1.8: return 'Alto', volume_score
        if volume_score > 1.2: return 'Medio', volume_score
        if volume_score > 0.8: return 'Bajo', volume_score
        return 'Muy Bajo', volume_score
    except:
        return 'Muy Bajo', 0

# Detectar divergencias
def detect_divergence(df, lookback=14):
    try:
        if len(df) < lookback + 1:
            return None
            
        recent = df.iloc[-lookback:]
        
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        current_high = recent['high'].iloc[-1]
        current_low = recent['low'].iloc[-1]
        current_rsi = recent['rsi'].iloc[-1]
        
        if high_idx != recent.index[-1]:
            high_rsi = df.loc[high_idx, 'rsi']
            if current_high > df.loc[high_idx, 'high'] and current_rsi < high_rsi:
                return 'bearish'
        
        if low_idx != recent.index[-1]:
            low_rsi = df.loc[low_idx, 'rsi']
            if current_low < df.loc[low_idx, 'low'] and current_rsi > low_rsi:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Detectar patrones de velas
def detect_candle_patterns(df):
    try:
        if len(df) < 2:
            return None
            
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Martillo alcista
        hammer = (last['close'] > last['open'] and 
                 (last['close'] - last['low']) > 2 * (last['high'] - last['close']) and
                 (last['close'] - last['low']) > 3 * (last['open'] - last['low']))
        
        # Estrella fugaz bajista
        shooting_star = (last['open'] > last['close'] and 
                        (last['high'] - last['open']) > 2 * (last['open'] - last['close']) and
                        (last['high'] - last['open']) > 3 * (last['close'] - last['low']))
        
        # Engulfing alcista
        bullish_engulfing = (prev['open'] > prev['close'] and 
                            last['close'] > last['open'] and
                            last['open'] < prev['close'] and
                            last['close'] > prev['open'])
        
        # Engulfing bajista
        bearish_engulfing = (prev['close'] > prev['open'] and 
                            last['open'] > last['close'] and
                            last['open'] > prev['close'] and
                            last['close'] < prev['open'])
        
        if hammer or bullish_engulfing:
            return 'bullish'
        elif shooting_star or bearish_engulfing:
            return 'bearish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando patrones de velas: {str(e)}")
        return None

# Calcular distancia al nivel más cercano
def near_level(price, levels, threshold_percent=1.0):
    if not levels or price <= 0:
        return False, 0
    
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    return min_distance <= threshold, min_distance

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

# Analizar una criptomoneda
def analyze_crypto(symbol, params, analyze_previous=False):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None, 0, 0, 'Muy Bajo', 0
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo', 0
        
        if analyze_previous:
            if len(df) > 1:
                last = df.iloc[-2]
                prev = df.iloc[-3] if len(df) > 2 else df.iloc[-2]
            else:
                last = df.iloc[-1]
                prev = df.iloc[-1]
        else:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        volume_class, volume_score = classify_volume(last['volume'], avg_vol, last.get('volume_ratio', 1))
        
        divergence = detect_divergence(df, params['divergence_lookback'])
        candle_pattern = detect_candle_patterns(df)
        is_breakout = any(last['close'] > r * 1.01 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.99 for s in supports) if supports else False
        
        # Verificar tendencia mayor (EMA tendencia)
        trend_major = last['close'] > last['ema_trend']
        
        trend_up = last['ema_fast'] > last['ema_slow'] and last['adx'] > params['adx_level']
        trend_down = last['ema_fast'] < last['ema_slow'] and last['adx'] > params['adx_level']
        
        long_prob = 0
        short_prob = 0
        
        # Cálculo de probabilidades LONG
        if trend_up: long_prob += 25
        if trend_major: long_prob += 10  # Bonus por tendencia mayor alcista
        
        near_support, support_dist = near_level(last['close'], supports, params['price_distance_threshold'])
        if near_support: long_prob += 20
        
        if last['rsi'] < 40: long_prob += 15
        if is_breakout or divergence == 'bullish' or candle_pattern == 'bullish': long_prob += 20
        
        # Añadir puntos por volumen (usando score de volumen)
        long_prob += min(20, volume_score * 10)
        
        # Cálculo de probabilidades SHORT
        if trend_down: short_prob += 25
        if not trend_major: short_prob += 10  # Bonus por tendencia mayor bajista
        
        near_resistance, resistance_dist = near_level(last['close'], resistances, params['price_distance_threshold'])
        if near_resistance: short_prob += 20
        
        if last['rsi'] > 60: short_prob += 15
        if is_breakdown or divergence == 'bearish' or candle_pattern == 'bearish': short_prob += 20
        
        # Añadir puntos por volumen (usando score de volumen)
        short_prob += min(20, volume_score * 10)
        
        # Ajustar probabilidades para que sumen 100%
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        long_signal = None
        if long_prob >= 60 and volume_class in ['Alto', 'Muy Alto']:
            next_supports = [s for s in supports if s < last['close']]
            sl = max(next_supports) * 0.99 if next_supports else last['close'] * (1 - params['max_risk_percent']/100)
            
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.01 if next_resistances else last['close'] * 1.02
            
            risk = entry - sl
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'volume': volume_class,
                'volume_score': round(volume_score, 2),
                'adx': round(last['adx'], 1),
                'rsi': round(last['rsi'], 1),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'LONG',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'],
                'trend_major': trend_major,
                'candle_pattern': candle_pattern,
                'divergence': divergence
            }
        
        short_signal = None
        if short_prob >= 60 and volume_class in ['Alto', 'Muy Alto']:
            next_resistances = [r for r in resistances if r > last['close']]
            sl = min(next_resistances) * 1.01 if next_resistances else last['close'] * (1 + params['max_risk_percent']/100)
            
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.99 if next_supports else last['close'] * 0.98
            
            risk = sl - entry
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'volume': volume_class,
                'volume_score': round(volume_score, 2),
                'adx': round(last['adx'], 1),
                'rsi': round(last['rsi'], 1),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'SHORT',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'],
                'trend_major': trend_major,
                'candle_pattern': candle_pattern,
                'divergence': divergence
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class, volume_score
    except Exception as e:
        logger.error(f"Error analizando {symbol} ({params['timeframe']}): {str(e)}")
        traceback.print_exc()
        return None, None, 0, 0, 'Muy Bajo', 0

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
                
                # Inicializar datos para este timeframe si no existen
                if current_timeframe not in analysis_state['timeframe_data']:
                    analysis_state['timeframe_data'][current_timeframe] = {
                        'long_signals': [],
                        'short_signals': [],
                        'scatter_data': [],
                        'historical_signals': [],
                        'last_updated': datetime.now()
                    }
                
                timeframe_data = analysis_state['timeframe_data'][current_timeframe]
                long_signals = []
                short_signals = []
                scatter_data = []
                
                previous_candle_start = get_previous_candle_start(current_timeframe)
                
                for i in range(0, total, BATCH_SIZE):
                    batch = cryptos[i:i+BATCH_SIZE]
                    
                    for crypto in batch:
                        try:
                            # Analizar vela actual
                            long_sig, short_sig, long_prob, short_prob, vol, vol_score = analyze_crypto(crypto, params, analyze_previous=False)
                            
                            if long_sig:
                                long_signals.append(long_sig)
                            
                            if short_sig:
                                short_signals.append(short_sig)
                            
                            scatter_data.append({
                                'symbol': crypto,
                                'long_prob': long_prob,
                                'short_prob': short_prob,
                                'volume': vol,
                                'volume_score': vol_score
                            })
                            
                            # Analizar vela anterior para señales históricas
                            long_sig_prev, short_sig_prev, _, _, _, _ = analyze_crypto(crypto, params, analyze_previous=True)
                            
                            if long_sig_prev:
                                candle_time = long_sig_prev.get('candle_timestamp')
                                if candle_time and candle_time == previous_candle_start:
                                    timeframe_data['historical_signals'].append(long_sig_prev)
                            
                            if short_sig_prev:
                                candle_time = short_sig_prev.get('candle_timestamp')
                                if candle_time and candle_time == previous_candle_start:
                                    timeframe_data['historical_signals'].append(short_sig_prev)
                            
                            processed += 1
                            progress = int((processed / total) * 100)
                            analysis_state['update_progress'] = progress
                        except Exception as e:
                            logger.error(f"Error procesando {crypto}: {str(e)}")
                            traceback.print_exc()
                    
                    time.sleep(1)  # Pequeña pausa entre batches
                
                # Ordenar señales por ADX (mayor primero)
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Actualizar datos del timeframe
                timeframe_data['long_signals'] = long_signals
                timeframe_data['short_signals'] = short_signals
                timeframe_data['scatter_data'] = scatter_data
                timeframe_data['last_updated'] = datetime.now()
                
                analysis_state['last_update'] = datetime.now()
                analysis_state['is_updating'] = False
                
                logger.info(f"Análisis completado para {current_timeframe}: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(timeframe_data['historical_signals'])} históricas")
        except Exception as e:
            logger.error(f"Error crítico en actualización: {str(e)}")
            traceback.print_exc()
            analysis_state['is_updating'] = False
        
        # Esperar hasta la próxima actualización
        analysis_state['update_event'].clear()
        analysis_state['update_event'].wait(CACHE_TIME)

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
            long_signals = timeframe_data['long_signals'][:50]  # Limitar a 50 señales
            short_signals = timeframe_data['short_signals'][:50]
            scatter_data = timeframe_data['scatter_data']
            historical_signals = list(timeframe_data['historical_signals'])[-20:]  # Últimas 20 señales históricas
        else:
            # No hay datos para este timeframe aún
            long_signals = []
            short_signals = []
            scatter_data = []
            historical_signals = []
        
        last_update = analysis_state['last_update']
        cryptos_analyzed = len(load_cryptos())
        
        avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
        avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
        
        # Preparar datos para el gráfico de dispersión
        scatter_ready = []
        for item in scatter_data:
            long_prob = max(0, min(100, item.get('long_prob', 0)))
            short_prob = max(0, min(100, item.get('short_prob', 0)))
            scatter_ready.append({
                'symbol': item['symbol'],
                'long_prob': long_prob,
                'short_prob': short_prob,
                'volume': item['volume'],
                'volume_score': item.get('volume_score', 0)
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
        if df is None or len(df) < 50:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return "Datos insuficientes", 404
        
        current_timeframe = params['timeframe']
        signal = None
        
        # Buscar la señal en los datos del timeframe actual
        if current_timeframe in analysis_state['timeframe_data']:
            signals = analysis_state['timeframe_data'][current_timeframe]['long_signals'] if signal_type == 'long' else analysis_state['timeframe_data'][current_timeframe]['short_signals']
            signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        plt.figure(figsize=(14, 10))
        
        # Gráfico de precio
        plt.subplot(4, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_trend'], label=f'EMA {params["ema_trend"]} (Tendencia)', color='red', alpha=0.8)
        
        # Líneas de entrada, SL y TP
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
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title(f'{signal["symbol"]} - Precio y EMAs ({params["timeframe"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        plt.subplot(4, 1, 2)
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        plt.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7, label='Volumen')
        plt.plot(df['timestamp'], df['volume_ma'], color='blue', linewidth=1.5, label='MA Volumen')
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title('Volumen')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de RSI
        plt.subplot(4, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Sobrecomprado')
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Sobrevendido')
        plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Gráfico de ADX
        plt.subplot(4, 1, 4)
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5, label='Umbral ADX')
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title('ADX - Fuerza de Tendencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        traceback.print_exc()
        return "Error generando gráfico", 500

@app.route('/historical_chart/<symbol>/<signal_type>')
def get_historical_chart(symbol, signal_type):
    try:
        params = analysis_state['params']
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return "Datos insuficientes", 404
        
        current_timeframe = params['timeframe']
        historical_signals = []
        
        # Buscar señales históricas para este símbolo y tipo
        if current_timeframe in analysis_state['timeframe_data']:
            timeframe_data = analysis_state['timeframe_data'][current_timeframe]
            previous_candle_start = get_previous_candle_start(current_timeframe)
            
            for signal in timeframe_data['historical_signals']:
                if signal['symbol'] == symbol and signal['type'].lower() == signal_type:
                    candle_time = signal.get('candle_timestamp')
                    if candle_time and candle_time == previous_candle_start:
                        historical_signals.append(signal)
        
        if not historical_signals:
            return "Señal histórica no encontrada", 404
        
        signal = historical_signals[-1]  # Tomar la más reciente
        
        plt.figure(figsize=(14, 10))
        
        # Gráfico de precio
        plt.subplot(4, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_trend'], label=f'EMA {params["ema_trend"]} (Tendencia)', color='red', alpha=0.8)
        
        # Líneas de entrada, SL y TP
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
        
        # Marcar la vela anterior analizada
        prev_candle_idx = df[df['timestamp'] == get_previous_candle_start(params['timeframe'])].index
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                plt.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.7, linewidth=2, label='Vela Anterior Analizada')
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title(f'{signal["symbol"]} - Señal Histórica {signal_type.upper()} ({params["timeframe"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        plt.subplot(4, 1, 2)
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        plt.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7, label='Volumen')
        plt.plot(df['timestamp'], df['volume_ma'], color='blue', linewidth=1.5, label='MA Volumen')
        
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                plt.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.7, linewidth=2)
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title('Volumen')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de RSI
        plt.subplot(4, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Sobrecomprado')
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Sobrevendido')
        plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                plt.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.7, linewidth=2)
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Gráfico de ADX
        plt.subplot(4, 1, 4)
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5, label='Umbral ADX')
        
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                plt.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.7, linewidth=2)
        
        format_xaxis_by_timeframe(params['timeframe'], plt.gca())
        plt.title('ADX - Fuerza de Tendencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('historical_chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico histórico: {str(e)}")
        traceback.print_exc()
        return "Error generando gráfico histórico", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/update_params', methods=['POST'])
def update_params():
    try:
        data = request.form.to_dict()
        
        new_params = analysis_state['params'].copy()
        
        for param in new_params:
            if param in data:
                value = data[param]
                if param in ['ema_fast', 'ema_slow', 'ema_trend', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback', 'volume_ma_window']:
                    new_params[param] = int(value)
                elif param in ['max_risk_percent', 'price_distance_threshold']:
                    new_params[param] = float(value)
                else:
                    new_params[param] = value
        
        with analysis_state['lock']:
            analysis_state['params'] = new_params
        
        # Forzar una actualización inmediata
        analysis_state['update_event'].set()
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados correctamente',
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
        current_timeframe = analysis_state['params']['timeframe']
        timeframe_data = analysis_state['timeframe_data'].get(current_timeframe, {})
        
        return jsonify({
            'last_update': analysis_state['last_update'].isoformat(),
            'is_updating': analysis_state['is_updating'],
            'progress': analysis_state['update_progress'],
            'long_signals': len(timeframe_data.get('long_signals', [])),
            'short_signals': len(timeframe_data.get('short_signals', [])),
            'historical_signals': len(timeframe_data.get('historical_signals', [])),
            'timeframes_available': list(analysis_state['timeframe_data'].keys()),
            'current_timeframe': current_timeframe,
            'params': analysis_state['params']
        })

@app.route('/switch_timeframe/<timeframe>')
def switch_timeframe(timeframe):
    try:
        with analysis_state['lock']:
            if timeframe in ['15m', '30m', '1h', '2h', '4h', '1d', '1w']:
                analysis_state['params']['timeframe'] = timeframe
                analysis_state['current_timeframe'] = timeframe
                
                # Forzar actualización si no hay datos para este timeframe
                if timeframe not in analysis_state['timeframe_data']:
                    analysis_state['update_event'].set()
                
                return jsonify({
                    'status': 'success',
                    'message': f'Timeframe cambiado a {timeframe}',
                    'timeframe': timeframe
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Timeframe no válido'
                }), 400
    except Exception as e:
        logger.error(f"Error cambiando timeframe: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error cambiando timeframe: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
