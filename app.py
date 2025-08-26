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
from threading import Lock, Event, Thread
from collections import deque
import pytz
import calendar
from dateutil.relativedelta import relativedelta
import talib  # Vamos a usar TA-Lib con una implementación alternativa

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 3  # Reducido para evitar timeouts

# Zona horaria
try:
    NY_TZ = pytz.timezone('America/New_York')
except:
    NY_TZ = pytz.timezone('UTC')
    logger.warning("No se pudo cargar zona horaria de NY, usando UTC")

# Implementación alternativa de TA-Lib para Render
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def ADX(high, low, close, period):
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(period).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 21,  # Cambiado a 21 (más común)
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 20,  # Reducido para mejor rendimiento
    'divergence_lookback': 10,  # Reducido
    'max_risk_percent': 1.0,  # Más conservador
    'price_distance_threshold': 0.8,  # Más estricto
    'min_volume_ratio': 1.2  # Nuevo: filtro de volumen mínimo
}

# Estado global
class AnalysisState:
    def __init__(self):
        self.lock = Lock()
        self.update_event = Event()
        self.timeframe_data = {}
        self.last_update = datetime.now()
        self.cryptos_analyzed = 0
        self.is_updating = False
        self.update_progress = 0
        self.params = DEFAULTS.copy()
        self.market_status = {
            'btc_dominance': 0,
            'total_marketcap': 0,
            'fear_greed': 50,
            'market_trend': 'neutral'
        }
    
    def get_timeframe_data(self, timeframe):
        if timeframe not in self.timeframe_data:
            self.timeframe_data[timeframe] = {
                'long_signals': [],
                'short_signals': [],
                'scatter_data': [],
                'historical_signals': deque(maxlen=50),
                'last_analysis': None
            }
        return self.timeframe_data[timeframe]

analysis_state = AnalysisState()

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Cargadas {len(cryptos)} criptomonedas")
            return cryptos[:50]  # Limitar a 50 para mejor rendimiento
    except Exception as e:
        logger.error(f"Error cargando criptomonedas: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'DOGE']

# Obtener datos con caché
def get_crypto_data(symbol, timeframe, force_update=False):
    cache_key = f"{symbol}_{timeframe}"
    cache_file = f"cache/{cache_key}.csv"
    
    # Crear directorio cache si no existe
    os.makedirs("cache", exist_ok=True)
    
    # Verificar si hay datos en caché (menos de 5 minutos)
    if not force_update and os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < CACHE_TIME:
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(NY_TZ)
                logger.info(f"Datos de {symbol} cargados desde caché")
                return df
            except Exception as e:
                logger.error(f"Error leyendo caché: {str(e)}")
    
    # Obtener datos de KuCoin
    tf_mapping = {
        '15m': '15min', '30m': '30min', '1h': '1hour', 
        '2h': '2hour', '4h': '4hour', '1d': '1day', '1w': '1week'
    }
    kucoin_tf = tf_mapping.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        logger.warning(f"No hay datos para {symbol}")
                        return None
                    
                    candles.reverse()
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    # Convertir tipos de datos
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) < 50:
                        logger.warning(f"Datos insuficientes para {symbol}: {len(df)} velas")
                        return None
                    
                    # Procesar timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
                    df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                    
                    # Guardar en caché
                    try:
                        df.to_csv(cache_file, index=False)
                    except Exception as e:
                        logger.error(f"Error guardando en caché: {str(e)}")
                    
                    return df
        except Exception as e:
            logger.error(f"Error obteniendo datos de {symbol}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    return None

# Calcular todos los indicadores
def calculate_indicators(df, params):
    try:
        # EMAs
        df['ema_fast'] = EMA(df['close'], params['ema_fast'])
        df['ema_slow'] = EMA(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = RSI(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = ADX(
            df['high'], df['low'], df['close'], params['adx_period']
        )
        
        # Volumen promedio
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Encontrar soportes y resistencias optimizado
def find_support_resistance(df, window=20):
    try:
        if len(df) < window:
            return [], []
        
        # Usar los últimos N períodos para S/R
        recent = df.iloc[-window:]
        
        # Encontrar mínimos y máximos locales
        supports = []
        resistances = []
        
        for i in range(2, len(recent)-2):
            if recent['low'].iloc[i] < recent['low'].iloc[i-1] and \
               recent['low'].iloc[i] < recent['low'].iloc[i-2] and \
               recent['low'].iloc[i] < recent['low'].iloc[i+1] and \
               recent['low'].iloc[i] < recent['low'].iloc[i+2]:
                supports.append(recent['low'].iloc[i])
            
            if recent['high'].iloc[i] > recent['high'].iloc[i-1] and \
               recent['high'].iloc[i] > recent['high'].iloc[i-2] and \
               recent['high'].iloc[i] > recent['high'].iloc[i+1] and \
               recent['high'].iloc[i] > recent['high'].iloc[i+2]:
                resistances.append(recent['high'].iloc[i])
        
        # Consolidar niveles cercanos
        def consolidate_levels(levels, threshold=0.02):
            if not levels:
                return []
            
            levels.sort()
            consolidated = [levels[0]]
            
            for level in levels[1:]:
                if level > consolidated[-1] * (1 + threshold):
                    consolidated.append(level)
            
            return consolidated
        
        return consolidate_levels(supports), consolidate_levels(resistances)
    except Exception as e:
        logger.error(f"Error encontrando S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.5: return 'Muy Alto'
        if ratio > 1.8: return 'Alto'
        if ratio > 1.2: return 'Medio'
        if ratio > 0.8: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias mejorado
def detect_divergence(df, lookback=10):
    try:
        if len(df) < lookback + 5:
            return None
        
        recent = df.iloc[-lookback:]
        
        # Buscar máximos y mínimos en precio y RSI
        price_highs = recent['high'].nlargest(3).index
        price_lows = recent['low'].nsmallest(3).index
        
        rsi_highs = recent['rsi'].nlargest(3).index
        rsi_lows = recent['rsi'].nsmallest(3).index
        
        # Divergencia bajista: precio hace máximos más altos, RSI hace máximos más bajos
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if price_highs[0] > price_highs[1] and rsi_highs[0] < rsi_highs[1]:
                return 'bearish'
        
        # Divergencia alcista: precio hace mínimos más bajos, RSI hace mínimos más altos
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if price_lows[0] < price_lows[1] and rsi_lows[0] > rsi_lows[1]:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Calcular distancia al nivel más cercano
def distance_to_level(price, levels, threshold_percent=1.0):
    if not levels:
        return float('inf'), None
    
    distances = [abs(price - level) for level in levels]
    min_distance = min(distances)
    min_level = levels[distances.index(min_distance)]
    
    threshold = price * threshold_percent / 100
    return min_distance, min_level if min_distance <= threshold else (float('inf'), None)

# Obtener timestamp de inicio de vela anterior
def get_previous_candle_start(timeframe):
    now = datetime.now(NY_TZ)
    
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        current_minute = (now.minute // minutes) * minutes
        current_candle_start = now.replace(minute=current_minute, second=0, microsecond=0)
        return current_candle_start - timedelta(minutes=minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        current_hour = (now.hour // hours) * hours
        current_candle_start = now.replace(hour=current_hour, minute=0, second=0, microsecond=0)
        return current_candle_start - timedelta(hours=hours)
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
        if timeframe in ['15m', '30m']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        elif timeframe in ['1h', '2h']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        elif timeframe == '4h':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        elif timeframe == '1d':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        elif timeframe == '1w':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    except Exception as e:
        logger.error(f"Error formateando eje X: {str(e)}")

# Analizar una criptomoneda
def analyze_crypto(symbol, params, analyze_previous=False):
    try:
        df = get_crypto_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        # Determinar qué vela analizar
        if analyze_previous and len(df) > 1:
            analysis_index = -2  # Vela anterior
        else:
            analysis_index = -1  # Vela actual
        
        last = df.iloc[analysis_index]
        prev = df.iloc[analysis_index-1] if analysis_index > 0 else last
        
        # Calcular soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Clasificar volumen
        avg_vol = df['volume_ma'].iloc[analysis_index] if 'volume_ma' in df else df['volume'].mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Detectar divergencia
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if last['ema_fast'] > last['ema_slow']:
            long_prob += 25  # Tendencia alcista
        
        if last['adx'] > params['adx_level']:
            long_prob += 15  # Tendencia fuerte
        
        dist_to_support, nearest_support = distance_to_level(last['close'], supports, params['price_distance_threshold'])
        if nearest_support is not None:
            long_prob += 20  # Cerca de soporte
        
        if last['rsi'] < 40:
            long_prob += 15  # RSI oversold
        
        if divergence == 'bullish':
            long_prob += 15  # Divergencia alcista
        
        if volume_class in ['Alto', 'Muy Alto'] and last['volume'] > avg_vol * params['min_volume_ratio']:
            long_prob += 10  # Volumen alto
        
        # Factores para SHORT
        if last['ema_fast'] < last['ema_slow']:
            short_prob += 25  # Tendencia bajista
        
        if last['adx'] > params['adx_level']:
            short_prob += 15  # Tendencia fuerte
        
        dist_to_resistance, nearest_resistance = distance_to_level(last['close'], resistances, params['price_distance_threshold'])
        if nearest_resistance is not None:
            short_prob += 20  # Cerca de resistencia
        
        if last['rsi'] > 60:
            short_prob += 15  # RSI overbought
        
        if divergence == 'bearish':
            short_prob += 15  # Divergencia bajista
        
        if volume_class in ['Alto', 'Muy Alto'] and last['volume'] > avg_vol * params['min_volume_ratio']:
            short_prob += 10  # Volumen alto
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Generar señales si superan el threshold
        long_signal = None
        short_signal = None
        
        if long_prob >= 65 and volume_class in ['Alto', 'Muy Alto']:
            # Calcular entrada, SL y TP para LONG
            if nearest_support is not None:
                sl = nearest_support * 0.99
            else:
                sl = last['close'] * (1 - params['max_risk_percent']/100)
            
            if nearest_resistance is not None:
                entry = nearest_resistance * 1.005
            else:
                entry = last['close'] * 1.01
            
            risk = entry - sl
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'tp3': round(entry + risk * 3, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'rsi': round(last['rsi'], 1),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'LONG',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'],
                'probability': round(long_prob, 1)
            }
        
        if short_prob >= 65 and volume_class in ['Alto', 'Muy Alto']:
            # Calcular entrada, SL y TP para SHORT
            if nearest_resistance is not None:
                sl = nearest_resistance * 1.01
            else:
                sl = last['close'] * (1 + params['max_risk_percent']/100)
            
            if nearest_support is not None:
                entry = nearest_support * 0.995
            else:
                entry = last['close'] * 0.99
            
            risk = sl - entry
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'tp3': round(entry - risk * 3, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'rsi': round(last['rsi'], 1),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'SHORT',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'],
                'probability': round(short_prob, 1)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
        
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Obtener estado del mercado
def get_market_status():
    try:
        # Obtener dominio de BTC
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            btc_dominance = data['data']['market_cap_percentage']['btc']
            total_marketcap = data['data']['total_market_cap']['usd']
            
            # Obtener índice Fear & Greed (approximación)
            # En una implementación real, usarías una API específica
            fear_greed = 50  # Placeholder
            
            analysis_state.market_status = {
                'btc_dominance': round(btc_dominance, 2),
                'total_marketcap': round(total_marketcap / 1e12, 2),  # Trillones
                'fear_greed': fear_greed,
                'market_trend': 'bullish' if btc_dominance > 48 else 'bearish' if btc_dominance < 42 else 'neutral'
            }
    except Exception as e:
        logger.error(f"Error obteniendo estado del mercado: {str(e)}")

# Tarea de actualización
def update_task():
    while True:
        try:
            with analysis_state.lock:
                analysis_state.is_updating = True
                analysis_state.update_progress = 0
                
                # Actualizar estado del mercado
                get_market_status()
                
                cryptos = load_cryptos()
                total = len(cryptos)
                processed = 0
                
                params = analysis_state.params
                current_timeframe = params['timeframe']
                logger.info(f"Iniciando análisis de {total} criptomonedas en {current_timeframe}")
                
                # Obtener datos del timeframe actual
                timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
                long_signals = []
                short_signals = []
                scatter_data = []
                
                # Obtener timestamp de la vela anterior
                previous_candle_start = get_previous_candle_start(current_timeframe)
                
                # Procesar criptomonedas en batches
                for crypto in cryptos:
                    try:
                        # Analizar vela actual
                        long_sig, short_sig, long_prob, short_prob, vol = analyze_crypto(crypto, params, False)
                        
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
                        long_sig_prev, short_sig_prev, _, _, _ = analyze_crypto(crypto, params, True)
                        
                        if long_sig_prev:
                            candle_time = long_sig_prev.get('candle_timestamp')
                            if candle_time and candle_time >= previous_candle_start:
                                timeframe_data['historical_signals'].append(long_sig_prev)
                        
                        if short_sig_prev:
                            candle_time = short_sig_prev.get('candle_timestamp')
                            if candle_time and candle_time >= previous_candle_start:
                                timeframe_data['historical_signals'].append(short_sig_prev)
                        
                        processed += 1
                        progress = int((processed / total) * 100)
                        analysis_state.update_progress = progress
                        
                        # Pequeña pausa para no saturar la API
                        if processed % 5 == 0:
                            time.sleep(0.5)
                            
                    except Exception as e:
                        logger.error(f"Error procesando {crypto}: {str(e)}")
                        processed += 1
                
                # Ordenar señales por probabilidad/calidad
                long_signals.sort(key=lambda x: (x['probability'], x['adx']), reverse=True)
                short_signals.sort(key=lambda x: (x['probability'], x['adx']), reverse=True)
                
                # Actualizar datos del timeframe
                timeframe_data['long_signals'] = long_signals
                timeframe_data['short_signals'] = short_signals
                timeframe_data['scatter_data'] = scatter_data
                timeframe_data['last_analysis'] = datetime.now()
                
                analysis_state.cryptos_analyzed = total
                analysis_state.last_update = datetime.now()
                analysis_state.is_updating = False
                
                logger.info(f"Análisis completado: {len(long_signals)} LONG, {len(short_signals)} SHORT")
                
        except Exception as e:
            logger.error(f"Error crítico en actualización: {str(e)}")
            analysis_state.is_updating = False
        
        # Esperar hasta la próxima actualización
        analysis_state.update_event.clear()
        analysis_state.update_event.wait(CACHE_TIME)

# Iniciar hilo de actualización
update_thread = threading.Thread(target=update_task, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

# Rutas de la aplicación
@app.route('/')
def index():
    with analysis_state.lock:
        params = analysis_state.params
        current_timeframe = params['timeframe']
        
        timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
        long_signals = timeframe_data['long_signals'][:30]  # Limitar a 30 señales
        short_signals = timeframe_data['short_signals'][:30]
        scatter_data = timeframe_data['scatter_data']
        historical_signals = list(timeframe_data['historical_signals'])[-15:]  # Últimas 15 señales históricas
        
        last_update = analysis_state.last_update
        cryptos_analyzed = analysis_state.cryptos_analyzed
        
        # Calcular promedios
        avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
        avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
        avg_prob_long = np.mean([s['probability'] for s in long_signals]) if long_signals else 0
        avg_prob_short = np.mean([s['probability'] for s in short_signals]) if short_signals else 0
        
        # Preparar datos para el gráfico de dispersión
        scatter_ready = []
        for item in scatter_data:
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
                               avg_prob_long=round(avg_prob_long, 1),
                               avg_prob_short=round(avg_prob_short, 1),
                               scatter_data=scatter_ready,
                               cryptos_analyzed=cryptos_analyzed,
                               is_updating=analysis_state.is_updating,
                               update_progress=analysis_state.update_progress,
                               market_status=analysis_state.market_status)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        params = analysis_state.params
        df = get_crypto_data(symbol, params['timeframe'])
        if df is None or len(df) < 50:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return "Datos insuficientes", 404
        
        # Obtener la señal específica
        current_timeframe = params['timeframe']
        timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
        
        if signal_type == 'long':
            signals = timeframe_data['long_signals']
        else:
            signals = timeframe_data['short_signals']
        
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        # Encontrar soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Crear gráfico
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Gráfico de precios
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='white', linewidth=1.5)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA{params["ema_fast"]}', color='orange', alpha=0.8)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA{params["ema_slow"]}', color='cyan', alpha=0.8)
        
        # Dibujar soportes y resistencias
        for support in supports:
            ax1.axhline(y=support, color='green', linestyle='--', alpha=0.5)
        
        for resistance in resistances:
            ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5)
        
        # Dibujar niveles de entrada, SL y TP
        if signal_type == 'long':
            ax1.axhline(y=signal['entry'], color='lime', linestyle='-', label='Entrada', linewidth=2)
            ax1.axhline(y=signal['sl'], color='red', linestyle='-', label='Stop Loss', linewidth=2)
            ax1.axhline(y=signal['tp1'], color='yellow', linestyle='--', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='orange', linestyle='--', label='TP2', alpha=0.7)
            ax1.axhline(y=signal['tp3'], color='magenta', linestyle='--', label='TP3', alpha=0.7)
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='-', label='Entrada', linewidth=2)
            ax1.axhline(y=signal['sl'], color='lime', linestyle='-', label='Stop Loss', linewidth=2)
            ax1.axhline(y=signal['tp1'], color='yellow', linestyle='--', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='orange', linestyle='--', label='TP2', alpha=0.7)
            ax1.axhline(y=signal['tp3'], color='magenta', linestyle='--', label='TP3', alpha=0.7)
        
        format_xaxis_by_timeframe(params['timeframe'], ax1)
        ax1.set_title(f'{symbol} - Precio y EMAs ({params["timeframe"]}) - Señal {signal_type.upper()}')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        ax2.plot(df['timestamp'], df['volume_ma'], color='white', alpha=0.7, label='MA Volumen')
        
        format_xaxis_by_timeframe(params['timeframe'], ax2)
        ax2.set_title('Volumen')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax3.axhline(y=50, color='white', linestyle=':', alpha=0.5)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['timestamp'], df['adx'], label='ADX', color='cyan', linewidth=1.5)
        ax3_twin.axhline(y=params['adx_level'], color='yellow', linestyle='--', alpha=0.7)
        
        format_xaxis_by_timeframe(params['timeframe'], ax3)
        ax3.set_title('Indicadores')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, facecolor='#0E1117')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/historical_chart/<symbol>/<signal_type>')
def get_historical_chart(symbol, signal_type):
    try:
        params = analysis_state.params
        df = get_crypto_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return "Datos insuficientes", 404
        
        # Obtener la señal histórica
        current_timeframe = params['timeframe']
        timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
        historical_signals = list(timeframe_data['historical_signals'])
        
        # Buscar la señal específica
        signal = next((s for s in historical_signals if s['symbol'] == symbol and s['type'].lower() == signal_type), None)
        
        if not signal:
            return "Señal histórica no encontrada", 404
        
        # Encontrar soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Encontrar el índice de la vela anterior
        previous_candle_start = get_previous_candle_start(params['timeframe'])
        candle_idx = df[df['timestamp'] >= previous_candle_start].index[0] if not df[df['timestamp'] >= previous_candle_start].empty else -2
        
        # Crear gráfico
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Gráfico de precios
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='white', linewidth=1.5)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA{params["ema_fast"]}', color='orange', alpha=0.8)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA{params["ema_slow"]}', color='cyan', alpha=0.8)
        
        # Dibujar soportes y resistencias
        for support in supports:
            ax1.axhline(y=support, color='green', linestyle='--', alpha=0.5)
        
        for resistance in resistances:
            ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5)
        
        # Dibujar niveles de entrada, SL y TP
        if signal_type == 'long':
            ax1.axhline(y=signal['entry'], color='lime', linestyle='-', label='Entrada', linewidth=2)
            ax1.axhline(y=signal['sl'], color='red', linestyle='-', label='Stop Loss', linewidth=2)
            ax1.axhline(y=signal['tp1'], color='yellow', linestyle='--', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='orange', linestyle='--', label='TP2', alpha=0.7)
            ax1.axhline(y=signal['tp3'], color='magenta', linestyle='--', label='TP3', alpha=0.7)
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='-', label='Entrada', linewidth=2)
            ax1.axhline(y=signal['sl'], color='lime', linestyle='-', label='Stop Loss', linewidth=2)
            ax1.axhline(y=signal['tp1'], color='yellow', linestyle='--', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='orange', linestyle='--', label='TP2', alpha=0.7)
            ax1.axhline(y=signal['tp3'], color='magenta', linestyle='--', label='TP3', alpha=0.7)
        
        # Marcar la vela anterior analizada
        if candle_idx < len(df):
            ax1.axvline(x=df.iloc[candle_idx]['timestamp'], color='gray', linestyle='-', alpha=0.7, label='Vela Analizada')
        
        format_xaxis_by_timeframe(params['timeframe'], ax1)
        ax1.set_title(f'{symbol} - Señal Histórica {signal_type.upper()} ({params["timeframe"]})')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        ax2.plot(df['timestamp'], df['volume_ma'], color='white', alpha=0.7, label='MA Volumen')
        
        if candle_idx < len(df):
            ax2.axvline(x=df.iloc[candle_idx]['timestamp'], color='gray', linestyle='-', alpha=0.7)
        
        format_xaxis_by_timeframe(params['timeframe'], ax2)
        ax2.set_title('Volumen')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax3.axhline(y=50, color='white', linestyle=':', alpha=0.5)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['timestamp'], df['adx'], label='ADX', color='cyan', linewidth=1.5)
        ax3_twin.axhline(y=params['adx_level'], color='yellow', linestyle='--', alpha=0.7)
        
        if candle_idx < len(df):
            ax3.axvline(x=df.iloc[candle_idx]['timestamp'], color='gray', linestyle='-', alpha=0.7)
        
        format_xaxis_by_timeframe(params['timeframe'], ax3)
        ax3.set_title('Indicadores')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, facecolor='#0E1117')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
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
        data = request.form.to_dict()
        
        new_params = analysis_state.params.copy()
        
        for param in new_params:
            if param in data:
                value = data[param]
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    new_params[param] = int(value)
                elif param in ['max_risk_percent', 'price_distance_threshold', 'min_volume_ratio']:
                    new_params[param] = float(value)
                else:
                    new_params[param] = value
        
        with analysis_state.lock:
            analysis_state.params = new_params
        
        # Forzar una actualización inmediata
        analysis_state.update_event.set()
        
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
    with analysis_state.lock:
        return jsonify({
            'last_update': analysis_state.last_update.isoformat(),
            'is_updating': analysis_state.is_updating,
            'progress': analysis_state.update_progress,
            'params': analysis_state.params,
            'market_status': analysis_state.market_status
        })

@app.route('/force_update')
def force_update():
    analysis_state.update_event.set()
    return jsonify({'status': 'success', 'message': 'Actualización forzada iniciada'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
