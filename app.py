import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response, session
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
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.secret_key = 'wgta_crypto_trading_secret_key_2024'

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300
MAX_RETRIES = 2
RETRY_DELAY = 1
MAX_WORKERS = 3

# Zona horaria
try:
    NY_TZ = pytz.timezone('America/New_York')
except:
    NY_TZ = pytz.timezone('UTC')
    logger.warning("No se pudo cargar la zona horaria de NY, usando UTC")

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
    'price_distance_threshold': 1.0,
    'atr_period': 14,
    'min_volume_ratio': 1.2
}

# Mapeo de timeframes de KuCoin
KUCOIN_TIMEFRAMES = {
    '15m': '15min',
    '30m': '30min', 
    '1h': '1hour',
    '2h': '2hour',
    '4h': '4hour',
    '1d': '1day',
    '1w': '1week'
}

# Duración de señales guardadas por timeframe (en horas)
SIGNAL_DURATIONS = {
    '15m': 4,    # 4 horas (16 velas)
    '30m': 8,    # 8 horas (16 velas)
    '1h': 16,    # 16 horas (16 velas)
    '2h': 32,    # 32 horas (16 velas)
    '4h': 64,    # 64 horas (16 velas)
    '1d': 384,   # 16 días (16 velas)
    '1w': 2688   # 16 semanas (16 velas)
}

# Estado global
class AnalysisState:
    def __init__(self):
        self.lock = Lock()
        self.long_signals = []
        self.short_signals = []
        self.scatter_data = []
        self.historical_signals = deque(maxlen=50)
        self.current_signals = deque(maxlen=50)
        self.last_update = datetime.now()
        self.cryptos_analyzed = 0
        self.is_updating = False
        self.update_progress = 0
        self.params = DEFAULTS.copy()
        self.timeframe_data = {}
        self.update_event = Event()
        self.current_timeframe = DEFAULTS['timeframe']
        self.symbols = []
        
    def to_dict(self):
        return {
            'long_signals': self.long_signals,
            'short_signals': self.short_signals,
            'scatter_data': self.scatter_data,
            'historical_signals': list(self.historical_signals),
            'current_signals': list(self.current_signals),
            'last_update': self.last_update,
            'cryptos_analyzed': self.cryptos_analyzed,
            'is_updating': self.is_updating,
            'update_progress': self.update_progress,
            'params': self.params,
            'current_timeframe': self.current_timeframe
        }

analysis_state = AnalysisState()

# Inicializar base de datos
def init_db():
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Tabla de usuarios
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            winrate REAL DEFAULT 0
        )
    ''')
    
    # Tabla de señales guardadas
    c.execute('''
        CREATE TABLE IF NOT EXISTS saved_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            sl_price REAL NOT NULL,
            tp1_price REAL NOT NULL,
            tp2_price REAL NOT NULL,
            tp3_price REAL NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            expiration DATETIME NOT NULL,
            status TEXT DEFAULT 'active',
            result TEXT,
            recommendation TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insertar usuarios predeterminados si no existen
    users = [
        ('Willer', '1234'),
        ('Danilo', '1234'),
        ('Damir', '1234')
    ]
    
    for username, password in users:
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        if c.fetchone() is None:
            password_hash = generate_password_hash(password)
            c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                     (username, password_hash))
    
    conn.commit()
    conn.close()

init_db()

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Cargadas {len(cryptos)} criptomonedas")
            return cryptos
    except Exception as e:
        logger.error(f"Error cargando criptomonedas: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC']

# Obtener datos de KuCoin optimizado
def get_kucoin_data(symbol, timeframe):
    if timeframe not in KUCOIN_TIMEFRAMES:
        logger.error(f"Timeframe {timeframe} no soportado")
        return None
        
    kucoin_tf = KUCOIN_TIMEFRAMES[timeframe]
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        return None
                    
                    candles.reverse()
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    # Convertir tipos de datos
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) < 100:
                        logger.warning(f"Datos insuficientes para {symbol}: {len(df)} velas")
                        return None
                    
                    # Convertir timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
                    df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                    
                    return df
                else:
                    logger.warning(f"Respuesta API no válida para {symbol}: {data.get('msg', 'Unknown error')}")
                    return None
            else:
                logger.warning(f"Error HTTP {response.status_code} para {symbol}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout obteniendo datos para {symbol}, intento {attempt+1}")
        except Exception as e:
            logger.error(f"Error obteniendo datos para {symbol}: {str(e)}")
        
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
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# Implementación optimizada de ADX
def calculate_adx(high, low, close, window=14):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        
        # Directional Movement
        up = high - high.shift()
        down = low.shift() - low
        
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        # Suavizado
        plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / atr)
        
        # ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        adx = dx.rolling(window).mean()
        
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))

# Calcular ATR
def calculate_atr(high, low, close, window=14):
    if len(close) < window + 1:
        return pd.Series([0] * len(close))
    
    try:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        return atr.fillna(tr.mean() if not tr.empty else 0)
    except Exception as e:
        logger.error(f"Error calculando ATR: {str(e)}")
        return pd.Series([0] * len(close))

# Calcular todos los indicadores
def calculate_indicators(df, params):
    try:
        # Calcular EMAs
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # Calcular RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # Calcular ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'], df['low'], df['close'], params['adx_period']
        )
        
        # Calcular ATR
        df['atr'] = calculate_atr(
            df['high'], df['low'], df['close'], params.get('atr_period', 14)
        )
        
        # Calcular volumen promedio
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias optimizado
def find_support_resistance(df, window=50):
    try:
        if len(df) < window:
            return [], []
        
        # Encontrar pivotes
        highs = df['high'].rolling(window=5, center=True).max()
        lows = df['low'].rolling(window=5, center=True).min()
        
        pivot_highs = df[df['high'] == highs]['high'].values
        pivot_lows = df[df['low'] == lows]['low'].values
        
        # Consolidar niveles cercanos
        def consolidate_levels(levels, threshold=0.01):
            if len(levels) == 0:
                return []
            
            levels = sorted(set(levels))
            consolidated = []
            current_group = [levels[0]]
            
            for i in range(1, len(levels)):
                if abs(levels[i] - current_group[-1]) / current_group[-1] <= threshold:
                    current_group.append(levels[i])
                else:
                    consolidated.append(sum(current_group) / len(current_group))
                    current_group = [levels[i]]
            
            if current_group:
                consolidated.append(sum(current_group) / len(current_group))
            
            return consolidated
        
        return consolidate_levels(pivot_lows), consolidate_levels(pivot_highs)
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen mejorado
def classify_volume(current_vol, avg_vol, min_ratio=1.2):
    try:
        if avg_vol <= 0 or current_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 3.0: return 'Extremo'
        if ratio > 2.0: return 'Muy Alto'
        if ratio > 1.5: return 'Alto'
        if ratio > min_ratio: return 'Medio'
        if ratio > 0.8: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias mejorado
def detect_divergence(df, lookback=20):
    try:
        if len(df) < lookback + 10:
            return None, 0
            
        # Buscar divergencias en el RSI
        recent = df.iloc[-lookback:]
        
        # Encontrar máximos y mínimos recientes
        price_highs = recent['high'].nlargest(3)
        price_lows = recent['low'].nsmallest(3)
        rsi_highs = recent['rsi'].nlargest(3)
        rsi_lows = recent['rsi'].nsmallest(3)
        
        # Divergencia bajista: precio hace máximos más altos, RSI hace máximos más bajos
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if (price_highs.iloc[0] > price_highs.iloc[1] and 
                rsi_highs.iloc[0] < rsi_highs.iloc[1]):
                return 'bearish', 25
        
        # Divergencia alcista: precio hace mínimos más bajos, RSI hace mínimos más altos
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if (price_lows.iloc[0] < price_lows.iloc[1] and 
                rsi_lows.iloc[0] > rsi_lows.iloc[1]):
                return 'bullish', 25
        
        return None, 0
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None, 0

# Calcular distancia al nivel más cercano
def near_level(price, levels, threshold_percent=1.0):
    if not levels or price <= 0:
        return False, 0
    
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    distance_ratio = min_distance / price * 100
    
    return min_distance <= threshold, distance_ratio

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
        if timeframe in ['15m', '30m']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        elif timeframe in ['1h', '2h']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        elif timeframe == '4h':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        elif timeframe in ['1d', '1w']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    except Exception as e:
        logger.error(f"Error formateando eje X: {str(e)}")

# Analizar una criptomoneda (versión optimizada)
def analyze_crypto(symbol, params, analyze_previous=False):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        # Seleccionar vela actual o anterior
        if analyze_previous and len(df) > 1:
            last_idx = -2
        else:
            last_idx = -1
            
        last = df.iloc[last_idx]
        prev = df.iloc[last_idx-1] if len(df) > abs(last_idx) else last
        
        # Encontrar soportes y resistencias
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Clasificar volumen
        avg_vol = df['volume_ma'].iloc[last_idx] if 'volume_ma' in df else df['volume'].rolling(20).mean().iloc[last_idx]
        volume_class = classify_volume(last['volume'], avg_vol, params.get('min_volume_ratio', 1.2))
        
        # Detectar divergencias
        divergence, divergence_score = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar rupturas
        is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
        
        # Determinar tendencia
        trend_up = last['ema_fast'] > last['ema_slow'] and last['adx'] > params['adx_level'] and last['plus_di'] > last['minus_di']
        trend_down = last['ema_fast'] < last['ema_slow'] and last['adx'] > params['adx_level'] and last['minus_di'] > last['plus_di']
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if trend_up: 
            long_prob += 30
        near_support, distance = near_level(last['close'], supports, params['price_distance_threshold'])
        if near_support: 
            long_prob += 20 - min(distance * 2, 15)  # Más cerca = más probabilidad
        if last['rsi'] < 40: 
            long_prob += 15
        if is_breakout or divergence == 'bullish': 
            long_prob += 20 + divergence_score
        if volume_class in ['Alto', 'Muy Alto', 'Extremo']: 
            long_prob += 15
        
        # Factores para SHORT
        if trend_down: 
            short_prob += 30
        near_resistance, distance = near_level(last['close'], resistances, params['price_distance_threshold'])
        if near_resistance: 
            short_prob += 20 - min(distance * 2, 15)
        if last['rsi'] > 60: 
            short_prob += 15
        if is_breakdown or divergence == 'bearish': 
            short_prob += 20 + divergence_score
        if volume_class in ['Alto', 'Muy Alto', 'Extremo']: 
            short_prob += 15
        
        # Ajustar probabilidades para que sumen 100%
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Generar señales si cumplen criterios
        long_signal = None
        if long_prob >= 65 and volume_class in ['Alto', 'Muy Alto', 'Extremo'] and trend_up:
            # Calcular niveles con ATR (prioridad) o S/R (fallback)
            atr = last.get('atr', 0)
            if atr > 0:
                # Usar ATR para calcular SL y entrada
                sl = last['low'] - 2 * atr
                entry = last['close'] + 0.5 * atr
            else:
                # Fallback a método S/R
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
                'tp3': round(entry + risk * 3, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'rsi': round(last['rsi'], 1),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'LONG',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'],
                'atr': round(atr, 4) if atr > 0 else 0
            }
        
        short_signal = None
        if short_prob >= 65 and volume_class in ['Alto', 'Muy Alto', 'Extremo'] and trend_down:
            atr = last.get('atr', 0)
            if atr > 0:
                # Usar ATR para calcular SL y entrada
                sl = last['high'] + 2 * atr
                entry = last['close'] - 0.5 * atr
            else:
                # Fallback a método S/R
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
                'tp3': round(entry - risk * 3, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'rsi': round(last['rsi'], 1),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'SHORT',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp'],
                'atr': round(atr, 4) if atr > 0 else 0
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol} ({params['timeframe']}): {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Procesar un lote de criptomonedas
def process_crypto_batch(batch, params, previous_candle_start, timeframe_data):
    long_signals = []
    short_signals = []
    scatter_data = []
    
    for crypto in batch:
        try:
            # Analizar señal actual
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
            
            # Analizar señal histórica (vela anterior)
            long_sig_prev, short_sig_prev, _, _, _ = analyze_crypto(crypto, params, True)
            
            if long_sig_prev:
                candle_time = long_sig_prev.get('candle_timestamp')
                if candle_time and candle_time == previous_candle_start:
                    timeframe_data['historical_signals'].append(long_sig_prev)
            
            if short_sig_prev:
                candle_time = short_sig_prev.get('candle_timestamp')
                if candle_time and candle_time == previous_candle_start:
                    timeframe_data['historical_signals'].append(short_sig_prev)
                    
        except Exception as e:
            logger.error(f"Error procesando {crypto}: {str(e)}")
    
    return long_signals, short_signals, scatter_data

# Obtener señales guardadas para un usuario
def get_saved_signals(user_id):
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Obtener señales activas
    c.execute('''
        SELECT * FROM saved_signals 
        WHERE user_id = ? AND status = 'active' 
        ORDER BY timestamp DESC
    ''', (user_id,))
    
    signals = []
    for row in c.fetchall():
        signals.append({
            'id': row[0],
            'symbol': row[2],
            'signal_type': row[3],
            'entry_price': row[4],
            'sl_price': row[5],
            'tp1_price': row[6],
            'tp2_price': row[7],
            'tp3_price': row[8],
            'timeframe': row[9],
            'timestamp': row[10],
            'expiration': row[11],
            'result': row[13],
            'recommendation': row[14]
        })
    
    conn.close()
    return signals

# Guardar una señal para un usuario
def save_signal_for_user(user_id, signal):
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Verificar si ya existe una señal similar activa
    c.execute('''
        SELECT COUNT(*) FROM saved_signals 
        WHERE user_id = ? AND symbol = ? AND signal_type = ? AND status = 'active'
    ''', (user_id, signal['symbol'], signal['type']))
    
    if c.fetchone()[0] > 0:
        conn.close()
        return False, "Ya existe una señal activa para este par"
    
    # Verificar límite de 10 señales por usuario
    c.execute('''
        SELECT COUNT(*) FROM saved_signals 
        WHERE user_id = ? AND status = 'active'
    ''', (user_id,))
    
    if c.fetchone()[0] >= 10:
        conn.close()
        return False, "Límite de 10 señales activas alcanzado"
    
    # Calcular fecha de expiración
    timeframe = signal['timeframe']
    duration_hours = SIGNAL_DURATIONS.get(timeframe, 16)  # Default 16 horas
    expiration = datetime.now() + timedelta(hours=duration_hours)
    
    # Insertar señal
    c.execute('''
        INSERT INTO saved_signals 
        (user_id, symbol, signal_type, entry_price, sl_price, tp1_price, tp2_price, tp3_price, timeframe, expiration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, 
        signal['symbol'], 
        signal['type'], 
        signal['entry'], 
        signal['sl'], 
        signal['tp1'], 
        signal['tp2'], 
        signal['tp3'], 
        timeframe,
        expiration
    ))
    
    conn.commit()
    conn.close()
    return True, "Señal guardada correctamente"

# Eliminar una señal guardada
def remove_saved_signal(signal_id, user_id):
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    c.execute('''
        DELETE FROM saved_signals 
        WHERE id = ? AND user_id = ?
    ''', (signal_id, user_id))
    
    conn.commit()
    conn.close()
    return c.rowcount > 0

# Actualizar recomendaciones para señales guardadas
def update_saved_signals_recommendations():
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Obtener todas las señales activas
    c.execute('SELECT * FROM saved_signals WHERE status = "active"')
    signals = c.fetchall()
    
    for signal in signals:
        signal_id, user_id, symbol, signal_type, entry, sl, tp1, tp2, tp3, timeframe, timestamp, expiration, status, result, recommendation = signal
        
        # Obtener datos actuales del mercado
        df = get_kucoin_data(symbol, timeframe)
        if df is None or len(df) < 10:
            continue
        
        current_price = df['close'].iloc[-1]
        previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        
        # Determinar la recomendación basada en el precio actual
        new_recommendation = None
        new_result = result
        
        # Verificar si la señal ha expirado
        if datetime.now() > datetime.strptime(expiration, '%Y-%m-%d %H:%M:%S'):
            if result is None:
                new_result = 'expired'
                new_recommendation = "Señal expirada sin resultado claro"
            status = 'inactive'
        
        # Verificar SL
        elif (signal_type == 'LONG' and current_price <= sl) or (signal_type == 'SHORT' and current_price >= sl):
            new_result = 'sl'
            new_recommendation = "Toco Stop Loss, esta fue una falsa señal. Lo siento"
            status = 'inactive'
        
        # Verificar TP1
        elif (signal_type == 'LONG' and current_price >= tp1) or (signal_type == 'SHORT' and current_price <= tp1):
            if result != 'tp1':
                new_result = 'tp1'
                new_recommendation = "Toco Take Profit 1, coloca tu SL a break even"
        
        # Verificar TP2
        elif (signal_type == 'LONG' and current_price >= tp2) or (signal_type == 'SHORT' and current_price <= tp2):
            if result != 'tp2':
                new_result = 'tp2'
                new_recommendation = "Toco Take Profit 2, si gustas salte de la operación con buenas ganancias o coloca SL en Take Profit 1"
        
        # Verificar TP3
        elif (signal_type == 'LONG' and current_price >= tp3) or (signal_type == 'SHORT' and current_price <= tp3):
            new_result = 'tp3'
            new_recommendation = "Toco Take Profit 3, Felicidades la operación fue todo un éxito"
            status = 'inactive'
        
        # Verificar si la operación está durando demasiado
        else:
            signal_age = datetime.now() - datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            half_duration = timedelta(hours=SIGNAL_DURATIONS.get(timeframe, 16) / 2)
            
            if signal_age > half_duration and result is None:
                new_recommendation = "Esta operación está durando mucho, sal de esta con discreción"
        
        # Actualizar la señal en la base de datos
        update_query = '''
            UPDATE saved_signals 
            SET recommendation = ?, result = ?, status = ?
            WHERE id = ?
        '''
        c.execute(update_query, (new_recommendation, new_result, status, signal_id))
    
    conn.commit()
    conn.close()

# Calcular winrate de un usuario
def calculate_user_winrate(user_id):
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Contar señales con resultado
    c.execute('''
        SELECT result, COUNT(*) FROM saved_signals 
        WHERE user_id = ? AND result IS NOT NULL
        GROUP BY result
    ''', (user_id,))
    
    results = c.fetchall()
    total_signals = 0
    winning_signals = 0
    
    for result, count in results:
        total_signals += count
        if result in ['tp1', 'tp2', 'tp3']:
            winning_signals += count
    
    winrate = (winning_signals / total_signals * 100) if total_signals > 0 else 0
    
    # Actualizar winrate del usuario
    c.execute('UPDATE users SET winrate = ? WHERE id = ?', (winrate, user_id))
    conn.commit()
    conn.close()
    
    return winrate

# Tarea de actualización optimizada
def update_task():
    while True:
        try:
            with analysis_state.lock:
                analysis_state.is_updating = True
                analysis_state.update_progress = 0
                
                cryptos = load_cryptos()
                analysis_state.symbols = cryptos
                total = len(cryptos)
                processed = 0
                
                params = analysis_state.params
                current_timeframe = params['timeframe']
                logger.info(f"Iniciando análisis de {total} criptomonedas para timeframe {current_timeframe}...")
                
                if current_timeframe not in analysis_state.timeframe_data:
                    analysis_state.timeframe_data[current_timeframe] = {
                        'long_signals': [],
                        'short_signals': [],
                        'scatter_data': [],
                        'historical_signals': deque(maxlen=50)
                    }
                
                timeframe_data = analysis_state.timeframe_data[current_timeframe]
                all_long_signals = []
                all_short_signals = []
                all_scatter_data = []
                
                previous_candle_start = get_previous_candle_start(current_timeframe)
                
                # Procesar en lotes secuenciales (más estable que paralelo)
                batch_size = 10
                batches = [cryptos[i:i+batch_size] for i in range(0, len(cryptos), batch_size)]
                
                for batch in batches:
                    long_signals, short_signals, scatter_data = process_crypto_batch(
                        batch, params, previous_candle_start, timeframe_data
                    )
                    
                    all_long_signals.extend(long_signals)
                    all_short_signals.extend(short_signals)
                    all_scatter_data.extend(scatter_data)
                    
                    processed += len(batch)
                    progress = int((processed / total) * 100)
                    analysis_state.update_progress = progress
                    
                    # Pequeña pausa entre lotes para evitar sobrecarga
                    time.sleep(0.5)
                
                # Ordenar señales por fuerza (ADX)
                all_long_signals.sort(key=lambda x: x['adx'], reverse=True)
                all_short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                timeframe_data['long_signals'] = all_long_signals
                timeframe_data['short_signals'] = all_short_signals
                timeframe_data['scatter_data'] = all_scatter_data
                
                analysis_state.cryptos_analyzed = total
                analysis_state.last_update = datetime.now()
                analysis_state.is_updating = False
                
                logger.info(f"Análisis completado: {len(all_long_signals)} LONG, {len(all_short_signals)} SHORT, {len(timeframe_data['historical_signals'])} históricas")
                
                # Actualizar recomendaciones de señales guardadas
                update_saved_signals_recommendations()
                
        except Exception as e:
            logger.error(f"Error crítico en actualización: {str(e)}")
            traceback.print_exc()
            analysis_state.is_updating = False
        
        # Esperar hasta la próxima actualización
        analysis_state.update_event.clear()
        analysis_state.update_event.wait(CACHE_TIME)

# Iniciar hilo de actualización
update_thread = Thread(target=update_task, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

# Rutas de Flask
@app.route('/')
def index():
    with analysis_state.lock:
        params = analysis_state.params
        current_timeframe = params['timeframe']
        
        if current_timeframe in analysis_state.timeframe_data:
            timeframe_data = analysis_state.timeframe_data[current_timeframe]
            long_signals = timeframe_data['long_signals'][:50]
            short_signals = timeframe_data['short_signals'][:50]
            scatter_data = timeframe_data['scatter_data']
            historical_signals = list(timeframe_data['historical_signals'])[-20:]
        else:
            long_signals = []
            short_signals = []
            scatter_data = []
            historical_signals = []
        
        last_update = analysis_state.last_update
        cryptos_analyzed = analysis_state.cryptos_analyzed
        
        avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
        avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
        
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
        
        historical_signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Obtener señales guardadas si el usuario está logueado
        saved_signals = []
        user_winrate = 0
        if 'user_id' in session:
            saved_signals = get_saved_signals(session['user_id'])
            user_winrate = calculate_user_winrate(session['user_id'])
        
        return render_template('index.html', 
                               long_signals=long_signals, 
                               short_signals=short_signals,
                               historical_signals=historical_signals,
                               saved_signals=saved_signals,
                               last_update=last_update,
                               params=params,
                               avg_adx_long=round(avg_adx_long, 1),
                               avg_adx_short=round(avg_adx_short, 1),
                               scatter_data=scatter_ready,
                               cryptos_analyzed=cryptos_analyzed,
                               is_updating=analysis_state.is_updating,
                               update_progress=analysis_state.update_progress,
                               user_winrate=user_winrate)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        session['user_id'] = user[0]
        session['username'] = username
        return jsonify({'status': 'success', 'message': 'Login exitoso'})
    else:
        return jsonify({'status': 'error', 'message': 'Credenciales inválidas'})

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({'status': 'success', 'message': 'Logout exitoso'})

@app.route('/save_signal', methods=['POST'])
def save_signal():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Debe iniciar sesión para guardar señales'})
    
    signal_data = request.get_json()
    if not signal_data:
        return jsonify({'status': 'error', 'message': 'Datos de señal no válidos'})
    
    success, message = save_signal_for_user(session['user_id'], signal_data)
    
    if success:
        return jsonify({'status': 'success', 'message': message})
    else:
        return jsonify({'status': 'error', 'message': message})

@app.route('/remove_signal/<int:signal_id>', methods=['POST'])
def remove_signal(signal_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Debe iniciar sesión para eliminar señales'})
    
    success = remove_saved_signal(signal_id, session['user_id'])
    
    if success:
        return jsonify({'status': 'success', 'message': 'Señal eliminada correctamente'})
    else:
        return jsonify({'status': 'error', 'message': 'Error al eliminar la señal'})

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        params = analysis_state.params
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return "Datos insuficientes", 404
        
        current_timeframe = params['timeframe']
        signal = None
        
        if current_timeframe in analysis_state.timeframe_data:
            signals = (analysis_state.timeframe_data[current_timeframe]['long_signals'] 
                      if signal_type == 'long' else 
                      analysis_state.timeframe_data[current_timeframe]['short_signals'])
            signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            # Intentar generar la señal si no está en cache
            long_sig, short_sig, _, _, _ = analyze_crypto(symbol, params, False)
            signal = long_sig if signal_type == 'long' else short_sig
            if not signal:
                return "Señal no encontrada", 404
        
        # Crear gráfico
        plt.style.use('default')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Gráfico de precios
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5, alpha=0.7)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', linewidth=1.2)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', linewidth=1.2)
        
        # Dibujar niveles de entrada, SL y TP
        if signal_type == 'long':
            ax1.axhline(y=signal['entry'], color='green', linestyle='--', alpha=0.8, label='Entrada')
            ax1.axhline(y=signal['sl'], color='red', linestyle='--', alpha=0.8, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
            ax1.axhline(y=signal['tp3'], color='brown', linestyle=':', alpha=0.7, label='TP3')
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='--', alpha=0.8, label='Entrada')
            ax1.axhline(y=signal['sl'], color='green', linestyle='--', alpha=0.8, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
            ax1.axhline(y=signal['tp3'], color='brown', linestyle=':', alpha=0.7, label='TP3')
        
        format_xaxis_by_timeframe(params['timeframe'], ax1)
        
        ax1.set_title(f'{signal["symbol"]} - Precio y EMAs ({params["timeframe"]})')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        colors = ['green' if close >= open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        
        format_xaxis_by_timeframe(params['timeframe'], ax2)
        
        ax2.set_title('Volumen')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.2)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI', color='purple')
        
        ax3b = ax3.twinx()
        ax3b.plot(df['timestamp'], df['adx'], label='ADX', color='brown', linewidth=1.2)
        ax3b.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5, label='Umbral ADX')
        ax3b.set_ylabel('ADX', color='brown')
        
        ax3.set_title('Indicadores')
        ax3.grid(True, alpha=0.3)
        format_xaxis_by_timeframe(params['timeframe'], ax3)
        
        # Combinar leyendas
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type, 
                              signal=signal, timeframe=params['timeframe'])
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/historical_chart/<symbol>/<signal_type>')
def get_historical_chart(symbol, signal_type):
    try:
        params = analysis_state.params
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return "Datos insuficientes", 404
        
        current_timeframe = params['timeframe']
        previous_candle_start = get_previous_candle_start(current_timeframe)
        historical_signals = []
        
        if current_timeframe in analysis_state.timeframe_data:
            timeframe_data = analysis_state.timeframe_data[current_timeframe]
            for signal in timeframe_data['historical_signals']:
                if signal['symbol'] == symbol and signal['type'].lower() == signal_type:
                    candle_time = signal.get('candle_timestamp')
                    if candle_time and candle_time == previous_candle_start:
                        historical_signals.append(signal)
        
        if not historical_signals:
            # Intentar generar la señal histórica
            long_sig, short_sig, _, _, _ = analyze_crypto(symbol, params, True)
            signal = long_sig if signal_type == 'long' else short_sig
            if signal:
                historical_signals.append(signal)
            else:
                return "Señal histórica no encontrada", 404
        
        signal = historical_signals[-1]
        
        # Crear gráfico histórico
        plt.style.use('default')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Gráfico de precios
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5, alpha=0.7)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', linewidth=1.2)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', linewidth=1.2)
        
        # Dibujar niveles de entrada, SL y TP
        if signal_type == 'long':
            ax1.axhline(y=signal['entry'], color='green', linestyle='--', alpha=0.8, label='Entrada')
            ax1.axhline(y=signal['sl'], color='red', linestyle='--', alpha=0.8, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
            ax1.axhline(y=signal['tp3'], color='brown', linestyle=':', alpha=0.7, label='TP3')
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='--', alpha=0.8, label='Entrada')
            ax1.axhline(y=signal['sl'], color='green', linestyle='--', alpha=0.8, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
            ax1.axhline(y=signal['tp3'], color='brown', linestyle=':', alpha=0.7, label='TP3')
        
        # Marcar la vela anterior
        prev_candle_idx = df[df['timestamp'] == previous_candle_start].index
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                ax1.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.7, label='Vela Anterior')
        
        format_xaxis_by_timeframe(params['timeframe'], ax1)
        
        ax1.set_title(f'{signal["symbol"]} - Señal Histórica {signal_type.upper()} ({params["timeframe"]})')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        colors = ['green' if close >= open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                ax2.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.7)
        
        format_xaxis_by_timeframe(params['timeframe'], ax2)
        
        ax2.set_title('Volumen')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.2)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI', color='purple')
        
        ax3b = ax3.twinx()
        ax3b.plot(df['timestamp'], df['adx'], label='ADX', color='brown', linewidth=1.2)
        ax3b.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5, label='Umbral ADX')
        ax3b.set_ylabel('ADX', color='brown')
        
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                ax3.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.7)
        
        ax3.set_title('Indicadores')
        ax3.grid(True, alpha=0.3)
        format_xaxis_by_timeframe(params['timeframe'], ax3)
        
        # Combinar leyendas
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        return render_template('historical_chart.html', plot_url=plot_url, symbol=symbol, 
                              signal_type=signal_type, signal=signal, timeframe=params['timeframe'])
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
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback', 'atr_period']:
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
            'long_signals': len(analysis_state.timeframe_data.get(analysis_state.params['timeframe'], {}).get('long_signals', [])),
            'short_signals': len(analysis_state.timeframe_data.get(analysis_state.params['timeframe'], {}).get('short_signals', [])),
            'historical_signals': len(analysis_state.timeframe_data.get(analysis_state.params['timeframe'], {}).get('historical_signals', [])),
            'cryptos_analyzed': analysis_state.cryptos_analyzed,
            'params': analysis_state.params
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
