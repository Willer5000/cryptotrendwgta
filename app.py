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
from matplotlib import gridspec
import io
import base64
import logging
import traceback
from threading import Lock, Event, Thread
from collections import deque
import pytz
import calendar
from dateutil.relativedelta import relativedelta
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300
MAX_RETRIES = 5
RETRY_DELAY = 1
BATCH_SIZE = 10
MAX_CACHE_SIZE = 50

try:
    NY_TZ = pytz.timezone('America/New_York')
except:
    NY_TZ = pytz.timezone('UTC')
    logger.warning("Using UTC timezone as fallback")

DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 21,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 20,
    'divergence_lookback': 14,
    'max_risk_percent': 1.0,
    'price_distance_threshold': 0.8,
    'volume_multiplier': 1.2,
    'min_adx_strength': 20,
    'btc_correlation_threshold': 0.7
}

class AdvancedIndicatorEngine:
    @staticmethod
    def exponential_moving_average(series, window):
        if len(series) < window:
            return pd.Series([np.nan] * len(series))
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def super_rsi(price, period=14):
        if len(price) < period:
            return pd.Series([50] * len(price))
        
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def advanced_adx(high, low, close, period=14):
        if len(close) < period * 2:
            return pd.Series([0] * len(close))
        
        try:
            return talib.ADX(high, low, close, timeperiod=period)
        except:
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
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
            return dx.rolling(period).mean().fillna(0)

    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        if len(series) < slow + signal:
            return pd.Series([0] * len(series)), pd.Series([0] * len(series)), pd.Series([0] * len(series))
        
        try:
            macd, macd_signal, macd_hist = talib.MACD(series, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd, macd_signal, macd_hist
        except:
            ema_fast = AdvancedIndicatorEngine.exponential_moving_average(series, fast)
            ema_slow = AdvancedIndicatorEngine.exponential_moving_average(series, slow)
            macd_line = ema_fast - ema_slow
            signal_line = AdvancedIndicatorEngine.exponential_moving_average(macd_line, signal)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

    @staticmethod
    def find_pivot_points(high, low, close, window=5):
        pivots_high = []
        pivots_low = []
        
        for i in range(window, len(high) - window):
            if high[i] == high[i-window:i+window].max():
                pivots_high.append((i, high[i]))
            if low[i] == low[i-window:i+window].min():
                pivots_low.append((i, low[i]))
                
        return pivots_high, pivots_low

    @staticmethod
    def calculate_support_resistance(high, low, close, window=20, consolidation_threshold=0.02):
        sr_levels = []
        
        for i in range(window, len(close)):
            recent_high = high[i-window:i].max()
            recent_low = low[i-window:i].min()
            
            if abs(recent_high - recent_low) / recent_low < consolidation_threshold:
                sr_levels.append((recent_high + recent_low) / 2)
            else:
                if close[i] > (recent_high + recent_low) / 2:
                    sr_levels.append(recent_low)
                else:
                    sr_levels.append(recent_high)
                    
        return sr_levels

    @staticmethod
    def calculate_volume_profile(high, low, volume, bins=20):
        range_min, range_max = low.min(), high.max()
        bin_size = (range_max - range_min) / bins
        
        if bin_size == 0:
            return [], []
            
        volume_per_price = {}
        
        for i in range(len(volume)):
            price_bin = int((high[i] + low[i]) / 2 / bin_size) * bin_size
            volume_per_price[price_bin] = volume_per_price.get(price_bin, 0) + volume[i]
            
        prices = list(volume_per_price.keys())
        volumes = list(volume_per_price.values())
        
        return prices, volumes

class CryptoDataManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CryptoDataManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
            
    def _initialize(self):
        self.data_cache = {}
        self.last_fetch_time = {}
        self.symbols = self._load_symbols()
        self.timeframe_map = {
            '15m': '15min', '30m': '30min', '1h': '1hour',
            '2h': '2hour', '4h': '4hour', '1d': '1day', '1w': '1week'
        }
        
    def _load_symbols(self):
        try:
            with open(CRYPTOS_FILE, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except:
            return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'DOGE']
            
    def get_crypto_data(self, symbol, timeframe, force_update=False):
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        
        if not force_update and cache_key in self.data_cache:
            if current_time - self.last_fetch_time.get(cache_key, 0) < CACHE_TIME:
                return self.data_cache[cache_key]
                
        kucoin_tf = self.timeframe_map.get(timeframe, '1hour')
        url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == '200000' and data.get('data'):
                        df = self._process_candle_data(data['data'])
                        self.data_cache[cache_key] = df
                        self.last_fetch_time[cache_key] = current_time
                        
                        if len(self.data_cache) > MAX_CACHE_SIZE:
                            oldest_key = min(self.last_fetch_time, key=self.last_fetch_time.get)
                            del self.data_cache[oldest_key]
                            del self.last_fetch_time[oldest_key]
                            
                        return df
                elif response.status_code == 429:
                    sleep_time = RETRY_DELAY * (attempt + 1)
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error fetching {symbol} {timeframe}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(RETRY_DELAY)
                
        return None
        
    def _process_candle_data(self, candles):
        candles.reverse()
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        
        if len(df) < 50:
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
        
        return df

class TradingSignalEngine:
    def __init__(self, params):
        self.params = params
        self.data_manager = CryptoDataManager()
        self.indicator_engine = AdvancedIndicatorEngine()
        
    def analyze_symbol(self, symbol, analyze_previous=False):
        df = self.data_manager.get_crypto_data(symbol, self.params['timeframe'])
        if df is None or len(df) < 100:
            return None, None, 0, 0, 'Muy Bajo'
            
        try:
            df = self._calculate_technical_indicators(df)
            if df is None or len(df) < 50:
                return None, None, 0, 0, 'Muy Bajo'
                
            if analyze_previous:
                analysis_index = -2 if len(df) > 1 else -1
            else:
                analysis_index = -1
                
            last = df.iloc[analysis_index]
            prev = df.iloc[analysis_index - 1] if analysis_index > 0 else last
            
            supports, resistances = self._find_key_levels(df)
            volume_class = self._classify_volume(df)
            market_regime = self._determine_market_regime(df)
            divergence_type = self._detect_divergence(df)
            
            long_prob, short_prob = self._calculate_probabilities(
                last, prev, supports, resistances, volume_class, market_regime, divergence_type
            )
            
            long_signal = self._generate_signal(symbol, last, long_prob, 'LONG', supports, resistances) if long_prob >= 65 else None
            short_signal = self._generate_signal(symbol, last, short_prob, 'SHORT', supports, resistances) if short_prob >= 65 else None
            
            return long_signal, short_signal, long_prob, short_prob, volume_class
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None, None, 0, 0, 'Muy Bajo'
            
    def _calculate_technical_indicators(self, df):
        try:
            df['ema_fast'] = self.indicator_engine.exponential_moving_average(df['close'], self.params['ema_fast'])
            df['ema_slow'] = self.indicator_engine.exponential_moving_average(df['close'], self.params['ema_slow'])
            df['rsi'] = self.indicator_engine.super_rsi(df['close'], self.params['rsi_period'])
            df['adx'] = self.indicator_engine.advanced_adx(df['high'], df['low'], df['close'], self.params['adx_period'])
            
            df['macd'], df['macd_signal'], df['macd_hist'] = self.indicator_engine.calculate_macd(
                df['close'], fast=12, slow=26, signal=9
            )
            
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None
            
    def _find_key_levels(self, df):
        try:
            pivots_high, pivots_low = self.indicator_engine.find_pivot_points(
                df['high'], df['low'], df['close'], window=3
            )
            
            recent_pivot_high = sorted([p[1] for p in pivots_high[-5:]], reverse=True)
            recent_pivot_low = sorted([p[1] for p in pivots_low[-5:]])
            
            sr_levels = self.indicator_engine.calculate_support_resistance(
                df['high'], df['low'], df['close'], window=self.params['sr_window']
            )
            
            key_resistances = list(set(recent_pivot_high + [max(sr_levels[-10:])] if sr_levels else []))
            key_supports = list(set(recent_pivot_low + [min(sr_levels[-10:])] if sr_levels else []))
            
            return key_supports, key_resistances
        except:
            return [], []
            
    def _classify_volume(self, df):
        try:
            if len(df) < 20:
                return 'Muy Bajo'
                
            current_vol = df['volume'].iloc[-1]
            avg_vol = df['volume'].rolling(20).mean().iloc[-1]
            
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
            
    def _determine_market_regime(self, df):
        if len(df) < 20:
            return 'neutral'
            
        adx_value = df['adx'].iloc[-1]
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        
        if adx_value > self.params['min_adx_strength']:
            return 'bullish' if ema_fast > ema_slow else 'bearish'
        else:
            return 'neutral'
            
    def _detect_divergence(self, df):
        if len(df) < self.params['divergence_lookback'] + 5:
            return None
            
        lookback = self.params['divergence_lookback']
        recent = df.iloc[-lookback:]
        
        price_highs = recent['high'].values
        price_lows = recent['low'].values
        rsi_highs = recent['rsi'].values
        rsi_lows = recent['rsi'].values
        
        try:
            price_slope, _, _, _, _ = stats.linregress(range(len(price_highs)), price_highs)
            rsi_slope, _, _, _, _ = stats.linregress(range(len(rsi_highs)), rsi_highs)
            
            if price_slope > 0 and rsi_slope < 0:
                return 'bearish_divergence'
            elif price_slope < 0 and rsi_slope > 0:
                return 'bullish_divergence'
                
        except:
            pass
            
        return None
        
    def _calculate_probabilities(self, last, prev, supports, resistances, volume_class, market_regime, divergence_type):
        long_prob = 0
        short_prob = 0
        
        # Trend factors
        if market_regime == 'bullish':
            long_prob += 25
        elif market_regime == 'bearish':
            short_prob += 25
            
        # Price action factors
        if last['close'] > last['ema_fast'] > last['ema_slow']:
            long_prob += 15
        elif last['close'] < last['ema_fast'] < last['ema_slow']:
            short_prob += 15
            
        # Support/Resistance factors
        near_support = any(abs(last['close'] - s) / s < self.params['price_distance_threshold'] / 100 for s in supports)
        near_resistance = any(abs(last['close'] - r) / r < self.params['price_distance_threshold'] / 100 for r in resistances)
        
        if near_support:
            long_prob += 20
        if near_resistance:
            short_prob += 20
            
        # Momentum factors
        if last['rsi'] < 35:
            long_prob += 15
        elif last['rsi'] > 65:
            short_prob += 15
            
        if divergence_type == 'bullish_divergence':
            long_prob += 20
        elif divergence_type == 'bearish_divergence':
            short_prob += 20
            
        # Volume confirmation
        if volume_class in ['Alto', 'Muy Alto']:
            if market_regime == 'bullish':
                long_prob += 10
            elif market_regime == 'bearish':
                short_prob += 10
                
        # Ensure probabilities are within bounds
        long_prob = min(100, max(0, long_prob))
        short_prob = min(100, max(0, short_prob))
        
        return long_prob, short_prob
        
    def _generate_signal(self, symbol, last_data, probability, signal_type, supports, resistances):
        price = last_data['close']
        atr = last_data.get('atr', price * 0.02)
        
        if signal_type == 'LONG':
            entry = price * 1.002
            stop_loss = min(supports) if supports else price * (1 - self.params['max_risk_percent'] / 100)
            risk = entry - stop_loss
            
            # Ensure risk is reasonable
            if risk / entry > self.params['max_risk_percent'] / 100:
                stop_loss = entry * (1 - self.params['max_risk_percent'] / 100)
                risk = entry - stop_loss
                
            take_profit1 = entry + risk * 1.5
            take_profit2 = entry + risk * 2.5
            
        else:  # SHORT
            entry = price * 0.998
            stop_loss = max(resistances) if resistances else price * (1 + self.params['max_risk_percent'] / 100)
            risk = stop_loss - entry
            
            # Ensure risk is reasonable
            if risk / entry > self.params['max_risk_percent'] / 100:
                stop_loss = entry * (1 + self.params['max_risk_percent'] / 100)
                risk = stop_loss - entry
                
            take_profit1 = entry - risk * 1.5
            take_profit2 = entry - risk * 2.5
            
        return {
            'symbol': symbol,
            'price': round(price, 4),
            'entry': round(entry, 4),
            'sl': round(stop_loss, 4),
            'tp1': round(take_profit1, 4),
            'tp2': round(take_profit2, 4),
            'volume': volume_class,
            'adx': round(last_data['adx'], 1),
            'rsi': round(last_data['rsi'], 1),
            'distance': round(abs(entry - price) / price * 100, 2),
            'timestamp': datetime.now().isoformat(),
            'type': signal_type,
            'timeframe': self.params['timeframe'],
            'probability': probability,
            'atr': round(atr, 4),
            'atr_percent': round(atr / price * 100, 2)
        }

class AnalysisState:
    def __init__(self):
        self.lock = Lock()
        self.timeframe_data = {}
        self.params = DEFAULTS.copy()
        self.is_updating = False
        self.update_progress = 0
        self.last_update = datetime.now()
        self.update_event = Event()
        self.data_manager = CryptoDataManager()
        
    def get_timeframe_data(self, timeframe):
        with self.lock:
            if timeframe not in self.timeframe_data:
                self.timeframe_data[timeframe] = {
                    'long_signals': [],
                    'short_signals': [],
                    'scatter_data': [],
                    'historical_signals': deque(maxlen=50),
                    'last_updated': None
                }
            return self.timeframe_data[timeframe]
            
    def update_timeframe_data(self, timeframe, data):
        with self.lock:
            self.timeframe_data[timeframe] = data
            self.timeframe_data[timeframe]['last_updated'] = datetime.now()

analysis_state = AnalysisState()

def update_analysis_task():
    while True:
        try:
            analysis_state.is_updating = True
            analysis_state.update_progress = 0
            
            cryptos = CryptoDataManager().symbols
            total = len(cryptos)
            processed = 0
            
            params = analysis_state.params.copy()
            current_timeframe = params['timeframe']
            
            logger.info(f"Iniciando análisis de {total} criptomonedas en timeframe {current_timeframe}")
            
            timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
            signal_engine = TradingSignalEngine(params)
            
            long_signals = []
            short_signals = []
            scatter_data = []
            
            previous_candle_start = get_previous_candle_start(current_timeframe)
            
            for i in range(0, total, BATCH_SIZE):
                batch = cryptos[i:i+BATCH_SIZE]
                
                for crypto in batch:
                    try:
                        long_sig, short_sig, long_prob, short_prob, vol_class = signal_engine.analyze_symbol(crypto)
                        
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
                        
                        long_sig_prev, short_sig_prev, _, _, _ = signal_engine.analyze_symbol(crypto, analyze_previous=True)
                        
                        if long_sig_prev:
                            timeframe_data['historical_signals'].append(long_sig_prev)
                        if short_sig_prev:
                            timeframe_data['historical_signals'].append(short_sig_prev)
                            
                    except Exception as e:
                        logger.error(f"Error processing {crypto}: {str(e)}")
                    
                    processed += 1
                    progress = int((processed / total) * 100)
                    analysis_state.update_progress = progress
                
                time.sleep(0.5)
            
            long_signals.sort(key=lambda x: (x['probability'], x['adx']), reverse=True)
            short_signals.sort(key=lambda x: (x['probability'], x['adx']), reverse=True)
            
            timeframe_data['long_signals'] = long_signals[:100]
            timeframe_data['short_signals'] = short_signals[:100]
            timeframe_data['scatter_data'] = scatter_data
            
            analysis_state.update_timeframe_data(current_timeframe, timeframe_data)
            analysis_state.last_update = datetime.now()
            analysis_state.is_updating = False
            
            logger.info(f"Análisis completado: {len(long_signals)} LONG, {len(short_signals)} SHORT")
            
        except Exception as e:
            logger.error(f"Error en tarea de análisis: {str(e)}")
            analysis_state.is_updating = False
            traceback.print_exc()
        
        analysis_state.update_event.wait(CACHE_TIME)
        analysis_state.update_event.clear()

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

def create_chart_image(df, signals, symbol, timeframe, signal_type, historical=False):
    try:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
        
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])
        
        # Plot price and EMAs
        ax1.plot(df['timestamp'], df['close'], label='Precio', color='white', linewidth=1.5)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA{params["ema_fast"]}', color='orange', linewidth=1)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA{params["ema_slow"]}', color='purple', linewidth=1)
        
        # Plot signals
        if signals:
            signal = signals[0]
            if signal_type == 'long':
                ax1.axhline(y=signal['entry'], color='green', linestyle='--', alpha=0.7, label='Entrada')
                ax1.axhline(y=signal['sl'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
                ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
                ax1.axhline(y=signal['tp2'], color='cyan', linestyle=':', alpha=0.7, label='TP2')
            else:
                ax1.axhline(y=signal['entry'], color='red', linestyle='--', alpha=0.7, label='Entrada')
                ax1.axhline(y=signal['sl'], color='green', linestyle='--', alpha=0.7, label='Stop Loss')
                ax1.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
                ax1.axhline(y=signal['tp2'], color='cyan', linestyle=':', alpha=0.7, label='TP2')
        
        if historical:
            prev_candle_start = get_previous_candle_start(timeframe)
            prev_candle = df[df['timestamp'] == prev_candle_start]
            if not prev_candle.empty:
                ax1.axvline(x=prev_candle_start, color='gray', linestyle='-', alpha=0.5, label='Vela Anterior')
        
        ax1.set_title(f'{symbol} - {timeframe} - Señal {signal_type.upper()}')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot volume
        colors = ['green' if close >= open else 'red' for close, open in zip(df['close'], df['open'])]
        ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volumen')
        ax2.grid(True, alpha=0.3)
        
        # Plot RSI
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='yellow', linewidth=1)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # Plot ADX and MACD
        ax4.plot(df['timestamp'], df['adx'], label='ADX', color='white', linewidth=1)
        ax4.axhline(y=25, color='blue', linestyle='--', alpha=0.5, label='Umbral 25')
        ax4.set_ylabel('ADX')
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=NY_TZ))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, facecolor='#121212')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        return plot_url
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return None

# Iniciar hilo de actualización
update_thread = Thread(target=update_analysis_task, daemon=True)
update_thread.start()

@app.route('/')
def index():
    params = analysis_state.params
    current_timeframe = params['timeframe']
    
    timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
    
    long_signals = timeframe_data.get('long_signals', [])
    short_signals = timeframe_data.get('short_signals', [])
    scatter_data = timeframe_data.get('scatter_data', [])
    historical_signals = list(timeframe_data.get('historical_signals', []))[-20:]
    
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    scatter_ready = []
    for item in scatter_data:
        scatter_ready.append({
            'symbol': item['symbol'],
            'long_prob': max(0, min(100, item.get('long_prob', 0))),
            'short_prob': max(0, min(100, item.get('short_prob', 0))),
            'volume': item.get('volume', 'Muy Bajo')
        })
    
    historical_signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           historical_signals=historical_signals,
                           last_update=analysis_state.last_update,
                           params=params,
                           avg_adx_long=round(avg_adx_long, 1),
                           avg_adx_short=round(avg_adx_short, 1),
                           scatter_data=scatter_ready,
                           cryptos_analyzed=len(CryptoDataManager().symbols),
                           is_updating=analysis_state.is_updating,
                           update_progress=analysis_state.update_progress)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        params = analysis_state.params
        df = CryptoDataManager().get_crypto_data(symbol, params['timeframe'])
        
        if df is None or len(df) < 50:
            return "Datos no disponibles", 404
            
        current_timeframe = params['timeframe']
        timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
        
        signals = timeframe_data['long_signals'] if signal_type == 'long' else timeframe_data['short_signals']
        symbol_signals = [s for s in signals if s['symbol'] == symbol]
        
        plot_url = create_chart_image(df, symbol_signals, symbol, params['timeframe'], signal_type)
        
        if not plot_url:
            return "Error generando gráfico", 500
            
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
        
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/historical_chart/<symbol>/<signal_type>')
def get_historical_chart(symbol, signal_type):
    try:
        params = analysis_state.params
        df = CryptoDataManager().get_crypto_data(symbol, params['timeframe'])
        
        if df is None or len(df) < 100:
            return "Datos no disponibles", 404
            
        current_timeframe = params['timeframe']
        timeframe_data = analysis_state.get_timeframe_data(current_timeframe)
        
        historical_signals = [s for s in timeframe_data['historical_signals'] 
                             if s['symbol'] == symbol and s['type'].lower() == signal_type]
        
        plot_url = create_chart_image(df, historical_signals, symbol, params['timeframe'], signal_type, historical=True)
        
        if not plot_url:
            return "Error generando gráfico histórico", 500
            
        return render_template('historical_chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
        
    except Exception as e:
        logger.error(f"Error generating historical chart: {str(e)}")
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
                elif param in ['max_risk_percent', 'price_distance_threshold', 'volume_multiplier', 'btc_correlation_threshold', 'min_adx_strength']:
                    new_params[param] = float(value)
                else:
                    new_params[param] = value
        
        analysis_state.params = new_params
        analysis_state.update_event.set()
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados correctamente',
            'params': new_params
        })
    except Exception as e:
        logger.error(f"Error updating parameters: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error actualizando parámetros: {str(e)}"
        }), 500

@app.route('/status')
def status():
    return jsonify({
        'last_update': analysis_state.last_update.isoformat(),
        'is_updating': analysis_state.is_updating,
        'progress': analysis_state.update_progress,
        'params': analysis_state.params
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
