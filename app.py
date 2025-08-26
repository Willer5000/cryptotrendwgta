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
from threading import Lock, Event, Timer
from collections import deque
import pytz
from functools import lru_cache
import hashlib

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 10

try:
    NY_TZ = pytz.timezone('America/New_York')
except:
    NY_TZ = pytz.timezone('UTC')
    logger.warning("No se pudo cargar la zona horaria de NY, usando UTC")

DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 21,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 20,
    'divergence_lookback': 14,
    'max_risk_percent': 1.5,
    'price_distance_threshold': 1.0,
    'atr_period': 14,
    'volume_multiplier': 1.5
}

class AnalysisState:
    def __init__(self):
        self.lock = Lock()
        self.timeframe_data = {}
        self.last_update = datetime.now()
        self.is_updating = False
        self.update_progress = 0
        self.params = DEFAULTS.copy()
        self.update_event = Event()
        self.symbols = self.load_cryptos()
        self.data_cache = {}
        self.cache_expiry = {}
        
    def load_cryptos(self):
        try:
            with open(CRYPTOS_FILE, 'r') as f:
                cryptos = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Cargadas {len(cryptos)} criptomonedas")
                return cryptos
        except Exception as e:
            logger.error(f"Error cargando criptomonedas: {str(e)}")
            return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC']
    
    def get_cached_data(self, symbol, timeframe):
        cache_key = f"{symbol}_{timeframe}"
        now = time.time()
        
        if cache_key in self.data_cache and now - self.cache_expiry.get(cache_key, 0) < CACHE_TIME:
            return self.data_cache[cache_key]
        return None
    
    def set_cached_data(self, symbol, timeframe, data):
        cache_key = f"{symbol}_{timeframe}"
        self.data_cache[cache_key] = data
        self.cache_expiry[cache_key] = time.time()

analysis_state = AnalysisState()

def get_kucoin_data(symbol, timeframe, retry=0):
    cached_data = analysis_state.get_cached_data(symbol, timeframe)
    if cached_data is not None:
        return cached_data.copy()

    tf_mapping = {
        '15m': '15min', '30m': '30min', '1h': '1hour', 
        '2h': '2hour', '4h': '4hour', '1d': '1day', '1w': '1week'
    }
    
    kucoin_tf = tf_mapping.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
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
                
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                if len(df) < 50:
                    return None
                
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
                df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                
                analysis_state.set_cached_data(symbol, timeframe, df)
                return df
        else:
            logger.warning(f"HTTP {response.status_code} for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
    
    if retry < MAX_RETRIES:
        time.sleep(RETRY_DELAY)
        return get_kucoin_data(symbol, timeframe, retry + 1)
    
    return None

def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_adx(high, low, close, period):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.fillna(0)

def calculate_indicators(df, params):
    try:
        df = df.copy()
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'], df['low'], df['close'], params['adx_period']
        )
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], params['atr_period'])
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

def find_support_resistance(df, window=20):
    try:
        rolling_high = df['high'].rolling(window=window, center=True).max()
        rolling_low = df['low'].rolling(window=window, center=True).min()
        
        resistance_levels = df[df['high'] == rolling_high]['high'].unique()
        support_levels = df[df['low'] == rolling_low]['low'].unique()
        
        return support_levels.tolist(), resistance_levels.tolist()
    except Exception as e:
        logger.error(f"Error finding S/R: {str(e)}")
        return [], []

def classify_volume(current_vol, historical_vol):
    if historical_vol == 0:
        return "Muy Bajo"
    
    ratio = current_vol / historical_vol
    if ratio > 3.0: return "Muy Alto"
    if ratio > 2.0: return "Alto"
    if ratio > 1.5: return "Medio"
    if ratio > 1.0: return "Bajo"
    return "Muy Bajo"

def detect_divergence(price, indicator, lookback=14):
    if len(price) < lookback + 5:
        return None
    
    price = price[-lookback:]
    indicator = indicator[-lookback:]
    
    price_max_idx = price.idxmax()
    price_min_idx = price.idxmin()
    indicator_max_idx = indicator.idxmax()
    indicator_min_idx = indicator.idxmin()
    
    if price_max_idx != indicator_max_idx and price[-1] > price.iloc[-2]:
        return "bearish"
    if price_min_idx != indicator_min_idx and price[-1] < price.iloc[-2]:
        return "bullish"
    
    return None

def calculate_distance_percent(price1, price2):
    return abs(price1 - price2) / price1 * 100

def get_previous_candle_start(timeframe):
    now = datetime.now(NY_TZ)
    
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        current_candle_start = now.replace(second=0, microsecond=0)
        current_candle_start = current_candle_start - timedelta(
            minutes=current_candle_start.minute % minutes,
            seconds=current_candle_start.second
        )
        return current_candle_start - timedelta(minutes=minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        current_candle_start = now.replace(minute=0, second=0, microsecond=0)
        current_candle_start = current_candle_start - timedelta(
            hours=current_candle_start.hour % hours
        )
        return current_candle_start - timedelta(hours=hours)
    elif timeframe.endswith('d'):
        return (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1))
    elif timeframe.endswith('w'):
        start_of_week = now - timedelta(days=now.weekday())
        return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(weeks=1)
    
    return now - timedelta(hours=1)

def analyze_crypto(symbol, params, analyze_previous=False):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        current_idx = -2 if analyze_previous else -1
        last = df.iloc[current_idx]
        prev = df.iloc[current_idx - 1] if len(df) > abs(current_idx) else last
        
        avg_volume = df['volume'].rolling(20).mean().iloc[current_idx]
        volume_class = classify_volume(last['volume'], avg_volume)
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        trend_up = last['ema_fast'] > last['ema_slow'] and last['adx'] > params['adx_level']
        trend_down = last['ema_fast'] < last['ema_slow'] and last['adx'] > params['adx_level']
        
        divergence = detect_divergence(
            df['close'].iloc[-params['divergence_lookback']:], 
            df['rsi'].iloc[-params['divergence_lookback']:],
            params['divergence_lookback']
        )
        
        long_prob = 0
        short_prob = 0
        
        if trend_up: long_prob += 30
        if last['rsi'] < 40: long_prob += 20
        if divergence == 'bullish': long_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 25
        
        if trend_down: short_prob += 30
        if last['rsi'] > 60: short_prob += 20
        if divergence == 'bearish': short_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 25
        
        long_signal = None
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            atr = last['atr']
            entry = last['close']
            sl = entry - (2 * atr)
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'price': round(entry, 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + (2 * risk), 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(calculate_distance_percent(entry, last['close']), 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'LONG',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp']
            }
        
        short_signal = None
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            atr = last['atr']
            entry = last['close']
            sl = entry + (2 * atr)
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'price': round(entry, 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - (2 * risk), 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(calculate_distance_percent(entry, last['close']), 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'SHORT',
                'timeframe': params['timeframe'],
                'candle_timestamp': last['timestamp']
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

def update_timeframe_data(timeframe):
    with analysis_state.lock:
        if timeframe not in analysis_state.timeframe_data:
            analysis_state.timeframe_data[timeframe] = {
                'long_signals': [],
                'short_signals': [],
                'scatter_data': [],
                'historical_signals': deque(maxlen=50),
                'last_updated': datetime.now()
            }
        return analysis_state.timeframe_data[timeframe]

def update_task():
    while True:
        try:
            with analysis_state.lock:
                analysis_state.is_updating = True
                analysis_state.update_progress = 0
                
                params = analysis_state.params.copy()
                timeframe = params['timeframe']
                symbols = analysis_state.symbols
                total = len(symbols)
                
                timeframe_data = update_timeframe_data(timeframe)
                long_signals = []
                short_signals = []
                scatter_data = []
                
                previous_candle_start = get_previous_candle_start(timeframe)
                
                for i, symbol in enumerate(symbols):
                    try:
                        long_sig, short_sig, long_prob, short_prob, vol = analyze_crypto(symbol, params, False)
                        
                        if long_sig:
                            long_signals.append(long_sig)
                        if short_sig:
                            short_signals.append(short_sig)
                            
                        scatter_data.append({
                            'symbol': symbol,
                            'long_prob': long_prob,
                            'short_prob': short_prob,
                            'volume': vol
                        })
                        
                        long_hist, short_hist, _, _, _ = analyze_crypto(symbol, params, True)
                        
                        if long_hist and long_hist.get('candle_timestamp') == previous_candle_start:
                            timeframe_data['historical_signals'].append(long_hist)
                        if short_hist and short_hist.get('candle_timestamp') == previous_candle_start:
                            timeframe_data['historical_signals'].append(short_hist)
                            
                    except Exception as e:
                        logger.error(f"Error procesando {symbol}: {str(e)}")
                    
                    analysis_state.update_progress = int((i + 1) / total * 100)
                
                timeframe_data['long_signals'] = sorted(long_signals, key=lambda x: x['adx'], reverse=True)
                timeframe_data['short_signals'] = sorted(short_signals, key=lambda x: x['adx'], reverse=True)
                timeframe_data['scatter_data'] = scatter_data
                timeframe_data['last_updated'] = datetime.now()
                
                analysis_state.last_update = datetime.now()
                analysis_state.is_updating = False
                
                logger.info(f"Actualización completada para {timeframe}. "
                           f"LONG: {len(long_signals)}, SHORT: {len(short_signals)}")
                
        except Exception as e:
            logger.error(f"Error crítico en update_task: {str(e)}")
            analysis_state.is_updating = False
        
        analysis_state.update_event.clear()
        analysis_state.update_event.wait(CACHE_TIME)

update_thread = threading.Thread(target=update_task, daemon=True)
update_thread.start()

@app.route('/')
def index():
    with analysis_state.lock:
        params = analysis_state.params
        timeframe = params['timeframe']
        
        if timeframe in analysis_state.timeframe_data:
            data = analysis_state.timeframe_data[timeframe]
            long_signals = data['long_signals'][:100]
            short_signals = data['short_signals'][:100]
            scatter_data = data['scatter_data']
            historical_signals = list(data['historical_signals'])[-20:]
        else:
            long_signals = []
            short_signals = []
            scatter_data = []
            historical_signals = []
        
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
        
        return render_template('index.html', 
                               long_signals=long_signals, 
                               short_signals=short_signals,
                               historical_signals=historical_signals,
                               last_update=analysis_state.last_update,
                               params=params,
                               avg_adx_long=round(avg_adx_long, 1),
                               avg_adx_short=round(avg_adx_short, 1),
                               scatter_data=scatter_ready,
                               cryptos_analyzed=len(analysis_state.symbols),
                               is_updating=analysis_state.is_updating,
                               update_progress=analysis_state.update_progress)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        params = analysis_state.params
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None:
            return "Datos insuficientes", 404
        
        timeframe_data = analysis_state.timeframe_data.get(params['timeframe'], {})
        signals = timeframe_data.get(f'{signal_type.lower()}_signals', [])
        signal = next((s for s in signals if s['symbol'] == symbol), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Precio y EMAs
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='white', linewidth=1.5)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA{params["ema_fast"]}', color='cyan', linewidth=1)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA{params["ema_slow"]}', color='magenta', linewidth=1)
        
        if signal_type.lower() == 'long':
            ax1.axhline(y=signal['entry'], color='green', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal['sl'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal['sl'], color='green', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        
        ax1.set_title(f'{symbol} - {signal_type.upper()} Signal ({params["timeframe"]})', color='white', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volumen
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        ax2.set_title('Volumen', color='white')
        ax2.grid(True, alpha=0.3)
        
        # Indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='yellow', linewidth=1)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['timestamp'], df['adx'], label='ADX', color='cyan', linewidth=1)
        ax3_twin.axhline(y=params['adx_level'], color='white', linestyle='--', alpha=0.5)
        
        ax3.set_title('Indicadores', color='white')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
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
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None:
            return "Datos insuficientes", 404
        
        timeframe_data = analysis_state.timeframe_data.get(params['timeframe'], {})
        historical_signals = [s for s in timeframe_data.get('historical_signals', []) 
                             if s['symbol'] == symbol and s['type'].lower() == signal_type.lower()]
        
        if not historical_signals:
            return "Señal histórica no encontrada", 404
        
        signal = historical_signals[-1]
        prev_candle_start = get_previous_candle_start(params['timeframe'])
        
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Precio y EMAs
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='white', linewidth=1.5)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA{params["ema_fast"]}', color='cyan', linewidth=1)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA{params["ema_slow"]}', color='magenta', linewidth=1)
        
        if signal_type.lower() == 'long':
            ax1.axhline(y=signal['entry'], color='green', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal['sl'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        else:
            ax1.axhline(y=signal['entry'], color='red', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal['sl'], color='green', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            ax1.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        
        # Marcar vela anterior
        prev_idx = df[df['timestamp'] == prev_candle_start].index
        if len(prev_idx) > 0:
            idx = prev_idx[0]
            if idx < len(df):
                ax1.axvline(x=df.iloc[idx]['timestamp'], color='gray', linestyle='-', alpha=0.5, label='Vela Anterior')
        
        ax1.set_title(f'{symbol} - Señal Histórica {signal_type.upper()} ({params["timeframe"]})', color='white', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volumen
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        ax2.set_title('Volumen', color='white')
        ax2.grid(True, alpha=0.3)
        
        # Indicadores
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='yellow', linewidth=1)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['timestamp'], df['adx'], label='ADX', color='cyan', linewidth=1)
        ax3_twin.axhline(y=params['adx_level'], color='white', linestyle='--', alpha=0.5)
        
        ax3.set_title('Indicadores', color='white')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
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
        
        for key in new_params:
            if key in data:
                if key in ['ema_fast', 'ema_slow', 'adx_period', 'rsi_period', 'sr_window', 'divergence_lookback', 'atr_period']:
                    new_params[key] = int(data[key])
                elif key in ['adx_level', 'max_risk_percent', 'price_distance_threshold', 'volume_multiplier']:
                    new_params[key] = float(data[key])
                else:
                    new_params[key] = data[key]
        
        with analysis_state.lock:
            analysis_state.params = new_params
        
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
        timeframe = analysis_state.params['timeframe']
        timeframe_data = analysis_state.timeframe_data.get(timeframe, {})
        
        return jsonify({
            'last_update': analysis_state.last_update.isoformat(),
            'is_updating': analysis_state.is_updating,
            'progress': analysis_state.update_progress,
            'long_signals': len(timeframe_data.get('long_signals', [])),
            'short_signals': len(timeframe_data.get('short_signals', [])),
            'historical_signals': len(timeframe_data.get('historical_signals', [])),
            'cryptos_analyzed': len(analysis_state.symbols),
            'params': analysis_state.params
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
