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
from collections import deque, defaultdict
import pytz
import concurrent.futures
from scipy import stats

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 10
MAX_WORKERS = 5

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
    'price_distance_threshold': 1.0,
    'atr_period': 14
}

class CryptoDataCache:
    def __init__(self):
        self.cache = defaultdict(dict)
        self.lock = Lock()
        self.last_update = defaultdict(dict)
    
    def get_data(self, symbol, timeframe):
        with self.lock:
            if symbol in self.cache and timeframe in self.cache[symbol]:
                if time.time() - self.last_update[symbol].get(timeframe, 0) < CACHE_TIME:
                    return self.cache[symbol][timeframe].copy()
        return None
    
    def set_data(self, symbol, timeframe, data):
        with self.lock:
            self.cache[symbol][timeframe] = data.copy()
            self.last_update[symbol][timeframe] = time.time()

crypto_cache = CryptoDataCache()

analysis_state = {
    'timeframe_data': defaultdict(lambda: {
        'long_signals': [],
        'short_signals': [],
        'scatter_data': [],
        'historical_signals': deque(maxlen=100),
        'last_update': datetime.now(),
        'cryptos_analyzed': 0
    }),
    'is_updating': False,
    'update_progress': 0,
    'params': DEFAULTS.copy(),
    'lock': Lock(),
    'update_event': Event(),
    'current_timeframe': DEFAULTS['timeframe']
}

def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Cargadas {len(cryptos)} criptomonedas")
            return cryptos
    except Exception as e:
        logger.error(f"Error cargando criptomonedas: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'LINK']

def get_kucoin_data(symbol, timeframe):
    cached_data = crypto_cache.get_data(symbol, timeframe)
    if cached_data is not None:
        return cached_data
    
    tf_mapping = {
        '15m': '15min', '30m': '30min', '1h': '1hour', '2h': '2hour',
        '4h': '4hour', '1d': '1day', '1w': '1week'
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
                        return None
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
                    df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                    
                    crypto_cache.set_data(symbol, timeframe, df)
                    return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} ({timeframe}): {str(e)}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    return None

def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    if len(series) < window + 1:
        return pd.Series([50] * len(series))
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
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
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))

def calculate_atr(high, low, close, window=14):
    try:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean().fillna(tr)
    except Exception as e:
        logger.error(f"Error calculando ATR: {str(e)}")
        return pd.Series([0] * len(close))

def calculate_indicators(df, params):
    try:
        df = df.copy()
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'], _, _ = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], params['atr_period'])
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

def find_support_resistance(df, window=50):
    try:
        if len(df) < window:
            return [], []
        
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['low'].iloc[i])
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['high'].iloc[i])
        
        def merge_levels(levels, threshold=0.005):
            if not levels:
                return []
            
            levels.sort()
            merged = []
            current = levels[0]
            
            for level in levels[1:]:
                if abs(level - current) / current <= threshold:
                    current = (current + level) / 2
                else:
                    merged.append(current)
                    current = level
            
            merged.append(current)
            return merged
        
        return merge_levels(support_levels), merge_levels(resistance_levels)
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

def classify_volume(current_vol, historical_vol):
    try:
        if historical_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / historical_vol
        if ratio > 2.5: return 'Muy Alto'
        if ratio > 1.8: return 'Alto'
        if ratio > 1.2: return 'Medio'
        if ratio > 0.8: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

def detect_divergence(df, lookback=20):
    try:
        if len(df) < lookback + 10:
            return None, None
        
        price = df['close'].values
        rsi = df['rsi'].values
        
        bull_divergence = False
        bear_divergence = False
        
        for i in range(len(df) - lookback, len(df) - 5):
            for j in range(i + 5, len(df)):
                if price[i] > price[j] and rsi[i] < rsi[j]:
                    bull_divergence = True
                if price[i] < price[j] and rsi[i] > rsi[j]:
                    bear_divergence = True
        
        return bull_divergence, bear_divergence
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None, None

def detect_candle_patterns(df):
    try:
        if len(df) < 3:
            return []
        
        patterns = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Hammer
        if (latest['close'] > latest['open'] and 
            (latest['close'] - latest['low']) > 2 * (latest['high'] - latest['close']) and
            (latest['close'] - latest['low']) > 1.5 * abs(latest['close'] - latest['open'])):
            patterns.append('hammer')
        
        # Engulfing alcista
        if (prev['close'] < prev['open'] and 
            latest['close'] > latest['open'] and
            latest['open'] < prev['close'] and 
            latest['close'] > prev['open']):
            patterns.append('bullish_engulfing')
        
        # Engulfing bajista
        if (prev['close'] > prev['open'] and 
            latest['close'] < latest['open'] and
            latest['open'] > prev['close'] and 
            latest['close'] < prev['open']):
            patterns.append('bearish_engulfing')
        
        return patterns
    except Exception as e:
        logger.error(f"Error detectando patrones de velas: {str(e)}")
        return []

def calculate_probabilities(df, params, supports, resistances, volume_class, candle_patterns):
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        long_prob = 0
        short_prob = 0
        
        # Tendencia y momentum (40%)
        if last['ema_fast'] > last['ema_slow']:
            long_prob += 20
        else:
            short_prob += 20
            
        if last['adx'] > params['adx_level']:
            if last['ema_fast'] > last['ema_slow']:
                long_prob += 10
            else:
                short_prob += 10
                
        if last['rsi'] < 35:
            long_prob += 10
        elif last['rsi'] > 65:
            short_prob += 10
            
        # Soporte y resistencia (30%)
        close_to_support = any(abs(last['close'] - s) / s < params['price_distance_threshold'] / 100 for s in supports)
        close_to_resistance = any(abs(last['close'] - r) / r < params['price_distance_threshold'] / 100 for r in resistances)
        
        if close_to_support:
            long_prob += 20
        if close_to_resistance:
            short_prob += 20
            
        # Patrones de velas y volumen (20%)
        if 'hammer' in candle_patterns or 'bullish_engulfing' in candle_patterns:
            long_prob += 10
        if 'bearish_engulfing' in candle_patterns:
            short_prob += 10
            
        if volume_class in ['Alto', 'Muy Alto']:
            long_prob += 5
            short_prob += 5
        elif volume_class in ['Muy Bajo']:
            long_prob -= 10
            short_prob -= 10
            
        # Ajuste final
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
            
        return min(100, max(0, long_prob)), min(100, max(0, short_prob))
    except Exception as e:
        logger.error(f"Error calculando probabilidades: {str(e)}")
        return 0, 0

def analyze_crypto_batch(crypto_batch, params, previous_candle_start):
    results = []
    for crypto in crypto_batch:
        try:
            df = get_kucoin_data(crypto, params['timeframe'])
            if df is None or len(df) < 100:
                continue
                
            df = calculate_indicators(df, params)
            if df is None or len(df) < 50:
                continue
                
            # Análisis de la vela actual
            last = df.iloc[-1]
            avg_vol = df['volume'].rolling(20).mean().iloc[-1]
            volume_class = classify_volume(last['volume'], avg_vol)
            
            supports, resistances = find_support_resistance(df, params['sr_window'])
            candle_patterns = detect_candle_patterns(df)
            bull_div, bear_div = detect_divergence(df, params['divergence_lookback'])
            
            long_prob, short_prob = calculate_probabilities(
                df, params, supports, resistances, volume_class, candle_patterns
            )
            
            long_signal, short_signal = None, None
            
            # Señal LONG
            if (long_prob >= 60 and last['ema_fast'] > last['ema_slow'] and 
                last['adx'] > params['adx_level'] and volume_class in ['Alto', 'Muy Alto']):
                
                stop_distance = last['atr'] * 2
                entry = last['close'] * 1.002
                sl = entry - stop_distance
                
                long_signal = {
                    'symbol': crypto,
                    'price': round(last['close'], 4),
                    'entry': round(entry, 4),
                    'sl': round(sl, 4),
                    'tp1': round(entry + stop_distance, 4),
                    'tp2': round(entry + stop_distance * 2, 4),
                    'volume': volume_class,
                    'adx': round(last['adx'], 1),
                    'rsi': round(last['rsi'], 1),
                    'atr': round(last['atr'], 4),
                    'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                    'timestamp': datetime.now().isoformat(),
                    'type': 'LONG',
                    'timeframe': params['timeframe'],
                    'candle_timestamp': last['timestamp'],
                    'probability': round(long_prob, 1),
                    'patterns': candle_patterns,
                    'divergence': bull_div
                }
            
            # Señal SHORT
            if (short_prob >= 60 and last['ema_fast'] < last['ema_slow'] and 
                last['adx'] > params['adx_level'] and volume_class in ['Alto', 'Muy Alto']):
                
                stop_distance = last['atr'] * 2
                entry = last['close'] * 0.998
                sl = entry + stop_distance
                
                short_signal = {
                    'symbol': crypto,
                    'price': round(last['close'], 4),
                    'entry': round(entry, 4),
                    'sl': round(sl, 4),
                    'tp1': round(entry - stop_distance, 4),
                    'tp2': round(entry - stop_distance * 2, 4),
                    'volume': volume_class,
                    'adx': round(last['adx'], 1),
                    'rsi': round(last['rsi'], 1),
                    'atr': round(last['atr'], 4),
                    'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                    'timestamp': datetime.now().isoformat(),
                    'type': 'SHORT',
                    'timeframe': params['timeframe'],
                    'candle_timestamp': last['timestamp'],
                    'probability': round(short_prob, 1),
                    'patterns': candle_patterns,
                    'divergence': bear_div
                }
            
            # Análisis de la vela anterior para señales históricas
            if len(df) > 1:
                prev_candle = df.iloc[-2]
                prev_avg_vol = df['volume'].rolling(20).mean().iloc[-2]
                prev_volume_class = classify_volume(prev_candle['volume'], prev_avg_vol)
                
                prev_long_prob, prev_short_prob = calculate_probabilities(
                    df.iloc[:-1], params, supports, resistances, prev_volume_class, candle_patterns
                )
                
                historical_signal = None
                if (prev_long_prob >= 60 and prev_candle['ema_fast'] > prev_candle['ema_slow'] and 
                    prev_candle['adx'] > params['adx_level'] and prev_volume_class in ['Alto', 'Muy Alto']):
                    
                    historical_signal = {
                        'symbol': crypto,
                        'price': round(prev_candle['close'], 4),
                        'entry': round(prev_candle['close'] * 1.002, 4),
                        'sl': round(prev_candle['close'] * 0.98, 4),
                        'tp1': round(prev_candle['close'] * 1.02, 4),
                        'tp2': round(prev_candle['close'] * 1.04, 4),
                        'volume': prev_volume_class,
                        'type': 'LONG',
                        'timestamp': datetime.now().isoformat(),
                        'candle_timestamp': prev_candle['timestamp']
                    }
                
                elif (prev_short_prob >= 60 and prev_candle['ema_fast'] < prev_candle['ema_slow'] and 
                      prev_candle['adx'] > params['adx_level'] and prev_volume_class in ['Alto', 'Muy Alto']):
                    
                    historical_signal = {
                        'symbol': crypto,
                        'price': round(prev_candle['close'], 4),
                        'entry': round(prev_candle['close'] * 0.998, 4),
                        'sl': round(prev_candle['close'] * 1.02, 4),
                        'tp1': round(prev_candle['close'] * 0.98, 4),
                        'tp2': round(prev_candle['close'] * 0.96, 4),
                        'volume': prev_volume_class,
                        'type': 'SHORT',
                        'timestamp': datetime.now().isoformat(),
                        'candle_timestamp': prev_candle['timestamp']
                    }
            else:
                historical_signal = None
            
            results.append({
                'symbol': crypto,
                'long_signal': long_signal,
                'short_signal': short_signal,
                'historical_signal': historical_signal,
                'long_prob': long_prob,
                'short_prob': short_prob,
                'volume': volume_class
            })
            
        except Exception as e:
            logger.error(f"Error analizando {crypto}: {str(e)}")
            continue
            
    return results

def update_timeframe_data(timeframe, params):
    try:
        cryptos = load_cryptos()
        total = len(cryptos)
        
        timeframe_data = {
            'long_signals': [],
            'short_signals': [],
            'scatter_data': [],
            'historical_signals': deque(maxlen=100),
            'last_update': datetime.now(),
            'cryptos_analyzed': total
        }
        
        previous_candle_start = get_previous_candle_start(params['timeframe'])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, total, BATCH_SIZE):
                batch = cryptos[i:i+BATCH_SIZE]
                futures.append(executor.submit(analyze_crypto_batch, batch, params, previous_candle_start))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    for result in batch_results:
                        if result['long_signal']:
                            timeframe_data['long_signals'].append(result['long_signal'])
                        if result['short_signal']:
                            timeframe_data['short_signals'].append(result['short_signal'])
                        if result['historical_signal']:
                            timeframe_data['historical_signals'].append(result['historical_signal'])
                            
                        timeframe_data['scatter_data'].append({
                            'symbol': result['symbol'],
                            'long_prob': result['long_prob'],
                            'short_prob': result['short_prob'],
                            'volume': result['volume']
                        })
                except Exception as e:
                    logger.error(f"Error procesando lote: {str(e)}")
        
        timeframe_data['long_signals'].sort(key=lambda x: x['probability'], reverse=True)
        timeframe_data['short_signals'].sort(key=lambda x: x['probability'], reverse=True)
        
        with analysis_state['lock']:
            analysis_state['timeframe_data'][timeframe] = timeframe_data
            
        logger.info(f"Actualización completada para {timeframe}: {len(timeframe_data['long_signals'])} LONG, {len(timeframe_data['short_signals'])} SHORT")
        
    except Exception as e:
        logger.error(f"Error crítico en actualización de {timeframe}: {str(e)}")
        traceback.print_exc()

def update_task():
    while True:
        try:
            with analysis_state['lock']:
                analysis_state['is_updating'] = True
                analysis_state['update_progress'] = 0
                params = analysis_state['params'].copy()
                current_timeframe = params['timeframe']
            
            update_timeframe_data(current_timeframe, params)
            
            with analysis_state['lock']:
                analysis_state['is_updating'] = False
                analysis_state['last_update'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error en update_task: {str(e)}")
            with analysis_state['lock']:
                analysis_state['is_updating'] = False
        
        analysis_state['update_event'].clear()
        analysis_state['update_event'].wait(CACHE_TIME)

update_thread = Thread(target=update_task, daemon=True)
update_thread.start()

@app.route('/')
def index():
    with analysis_state['lock']:
        params = analysis_state['params']
        current_timeframe = params['timeframe']
        
        if current_timeframe in analysis_state['timeframe_data']:
            timeframe_data = analysis_state['timeframe_data'][current_timeframe]
            long_signals = timeframe_data['long_signals'][:50]
            short_signals = timeframe_data['short_signals'][:50]
            scatter_data = timeframe_data['scatter_data']
            historical_signals = list(timeframe_data['historical_signals'])[-20:]
            cryptos_analyzed = timeframe_data['cryptos_analyzed']
            last_update = timeframe_data['last_update']
        else:
            long_signals = []
            short_signals = []
            scatter_data = []
            historical_signals = []
            cryptos_analyzed = 0
            last_update = datetime.now()
        
        avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
        avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
        
        scatter_ready = []
        for item in scatter_data:
            scatter_ready.append({
                'symbol': item['symbol'],
                'long_prob': max(0, min(100, item.get('long_prob', 0))),
                'short_prob': max(0, min(100, item.get('short_prob', 0))),
                'volume': item['volume']
            })
        
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
        
        if current_timeframe in analysis_state['timeframe_data']:
            signals = (analysis_state['timeframe_data'][current_timeframe]['long_signals'] 
                       if signal_type == 'long' else 
                       analysis_state['timeframe_data'][current_timeframe]['short_signals'])
            signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        
        # Gráfico de precio
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', alpha=0.8)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', alpha=0.8)
        
        if signal_type == 'long':
            ax1.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada', alpha=0.8)
            ax1.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss', alpha=0.8)
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', label='TP2', alpha=0.7)
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada', alpha=0.8)
            ax1.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss', alpha=0.8)
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', label='TP2', alpha=0.7)
        
        format_xaxis_by_timeframe(params['timeframe'], ax1)
        ax1.set_title(f'{symbol} - Precio y EMAs ({params["timeframe"]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        format_xaxis_by_timeframe(params['timeframe'], ax2)
        ax2.set_title('Volumen')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylim(0, 100)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['timestamp'], df['adx'], label='ADX', color='brown', linewidth=1.5)
        ax3_twin.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5)
        ax3_twin.set_ylim(0, 100)
        
        format_xaxis_by_timeframe(params['timeframe'], ax3)
        ax3.set_title('Indicadores')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
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
        params = analysis_state['params']
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return "Datos insuficientes", 404
        
        current_timeframe = params['timeframe']
        historical_signals = []
        
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
        
        signal = historical_signals[-1]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        
        # Gráfico de precio
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', alpha=0.8)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', alpha=0.8)
        
        if signal_type == 'long':
            ax1.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada', alpha=0.8)
            ax1.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss', alpha=0.8)
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', label='TP2', alpha=0.7)
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada', alpha=0.8)
            ax1.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss', alpha=0.8)
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', label='TP1', alpha=0.7)
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', label='TP2', alpha=0.7)
        
        # Marcar vela anterior
        prev_candle_start = get_previous_candle_start(params['timeframe'])
        prev_candle_idx = df[df['timestamp'] == prev_candle_start].index
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                ax1.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.5, label='Vela Anterior')
        
        format_xaxis_by_timeframe(params['timeframe'], ax1)
        ax1.set_title(f'{symbol} - Señal Histórica {signal_type.upper()} ({params["timeframe"]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                ax2.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.5)
        
        format_xaxis_by_timeframe(params['timeframe'], ax2)
        ax2.set_title('Volumen')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylim(0, 100)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['timestamp'], df['adx'], label='ADX', color='brown', linewidth=1.5)
        ax3_twin.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5)
        ax3_twin.set_ylim(0, 100)
        
        if len(prev_candle_idx) > 0:
            idx = prev_candle_idx[0]
            if idx < len(df):
                ax3.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.5)
        
        format_xaxis_by_timeframe(params['timeframe'], ax3)
        ax3.set_title('Indicadores')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
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
        new_params = analysis_state['params'].copy()
        
        for param in new_params:
            if param in data:
                value = data[param]
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback', 'atr_period']:
                    new_params[param] = int(value)
                elif param in ['max_risk_percent', 'price_distance_threshold']:
                    new_params[param] = float(value)
                else:
                    new_params[param] = value
        
        with analysis_state['lock']:
            analysis_state['params'] = new_params
        
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
        params = analysis_state['params']
        current_timeframe = params['timeframe']
        timeframe_data = analysis_state['timeframe_data'].get(current_timeframe, {})
        
        return jsonify({
            'last_update': timeframe_data.get('last_update', datetime.now()).isoformat(),
            'is_updating': analysis_state['is_updating'],
            'progress': analysis_state['update_progress'],
            'long_signals': len(timeframe_data.get('long_signals', [])),
            'short_signals': len(timeframe_data.get('short_signals', [])),
            'historical_signals': len(timeframe_data.get('historical_signals', [])),
            'cryptos_analyzed': timeframe_data.get('cryptos_analyzed', 0),
            'params': params
        })

@app.route('/force_update')
def force_update():
    analysis_state['update_event'].set()
    return jsonify({'status': 'success', 'message': 'Actualización forzada iniciada'})

def get_previous_candle_start(timeframe):
    now = datetime.now(NY_TZ)
    
    if timeframe == '15m':
        minutes = (now.minute // 15) * 15
        return now.replace(minute=minutes, second=0, microsecond=0) - timedelta(minutes=15)
    elif timeframe == '30m':
        minutes = (now.minute // 30) * 30
        return now.replace(minute=minutes, second=0, microsecond=0) - timedelta(minutes=30)
    elif timeframe == '1h':
        return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    elif timeframe == '2h':
        hours = (now.hour // 2) * 2
        return now.replace(hour=hours, minute=0, second=0, microsecond=0) - timedelta(hours=2)
    elif timeframe == '4h':
        hours = (now.hour // 4) * 4
        return now.replace(hour=hours, minute=0, second=0, microsecond=0) - timedelta(hours=4)
    elif timeframe == '1d':
        return now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    elif timeframe == '1w':
        start_of_week = now - timedelta(days=now.weekday())
        return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(weeks=1)
    
    return now - timedelta(hours=1)

def format_xaxis_by_timeframe(timeframe, ax):
    try:
        if timeframe in ['15m', '30m']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=NY_TZ))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        elif timeframe in ['1h', '2h']:
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
