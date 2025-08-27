import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import logging
from threading import Lock, Event
from collections import deque
import pytz

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 3

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
    'volume_filter': 1.0
}

class TradingSystem:
    def __init__(self):
        self.lock = Lock()
        self.update_event = Event()
        self.timeframe_data = {}
        self.params = DEFAULTS.copy()
        self.last_update = datetime.now()
        self.is_updating = False
        self.update_progress = 0
        self.cryptos_analyzed = 0
        
    def load_cryptos(self):
        try:
            with open(CRYPTOS_FILE, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except:
            return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK']

trading_system = TradingSystem()

def get_kucoin_data(symbol, timeframe, limit=100):
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
                    candles = data['data'][-limit:] if limit else data['data']
                    candles.reverse()
                    
                    if len(candles) < 20:
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) < 20:
                        return None
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
                    df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                    
                    return df
        except Exception as e:
            logger.warning(f"Intento {attempt+1} para {symbol} falló: {str(e)}")
            time.sleep(RETRY_DELAY)
    
    return None

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_adx(high, low, close, window):
    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = (high - close.shift()).abs()
    tr['l-pc'] = (low - close.shift()).abs()
    tr['tr'] = tr.max(axis=1)
    
    atr = tr['tr'].rolling(window).mean()
    
    up = high - high.shift()
    down = low.shift() - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(window).mean()
    
    return adx, plus_di, minus_di

def calculate_indicators(df, params):
    try:
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        adx, plus_di, minus_di = calculate_adx(
            df['high'], df['low'], df['close'], params['adx_period']
        )
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

def find_support_resistance(df, window=20):
    try:
        pivot_range = 3
        df['pivot_low'] = df['low'].rolling(window=pivot_range*2+1, center=True).min()
        df['pivot_high'] = df['high'].rolling(window=pivot_range*2+1, center=True).max()
        
        supports = df[df['low'] == df['pivot_low']]['low'].tail(5).values
        resistances = df[df['high'] == df['pivot_high']]['high'].tail(5).values
        
        return supports, resistances
    except:
        return [], []

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

def detect_divergence(df, lookback=14):
    try:
        if len(df) < lookback + 5:
            return None
            
        price_highs = df['high'].rolling(5).max().dropna()
        price_lows = df['low'].rolling(5).min().dropna()
        rsi_highs = df['rsi'].rolling(5).max().dropna()
        rsi_lows = df['rsi'].rolling(5).min().dropna()
        
        if len(price_highs) < 2 or len(rsi_highs) < 2:
            return None
        
        bearish_div = (price_highs.iloc[-1] > price_highs.iloc[-2] and 
                      rsi_highs.iloc[-1] < rsi_highs.iloc[-2])
        bullish_div = (price_lows.iloc[-1] < price_lows.iloc[-2] and 
                      rsi_lows.iloc[-1] > rsi_lows.iloc[-2])
        
        if bearish_div:
            return 'bearish'
        elif bullish_div:
            return 'bullish'
        return None
    except:
        return None

def analyze_crypto(symbol, params, analyze_previous=False):
    df = get_kucoin_data(symbol, params['timeframe'], 100)
    if df is None or len(df) < 50:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        current_idx = -2 if analyze_previous else -1
        last = df.iloc[current_idx]
        
        avg_vol = df['volume'].rolling(20).mean().iloc[current_idx]
        volume_class = classify_volume(last['volume'], avg_vol)
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        trend_up = (last['ema_fast'] > last['ema_slow'] and 
                   last['adx'] > params['adx_level'] and
                   last['plus_di'] > last['minus_di'])
        trend_down = (last['ema_fast'] < last['ema_slow'] and 
                     last['adx'] > params['adx_level'] and
                     last['minus_di'] > last['plus_di'])
        
        long_prob = 0
        short_prob = 0
        
        if trend_up: long_prob += 40
        if last['rsi'] < 40: long_prob += 20
        if divergence == 'bullish': long_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 15
        
        if trend_down: short_prob += 40
        if last['rsi'] > 60: short_prob += 20
        if divergence == 'bearish': short_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 15
        
        long_signal = None
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = last['close'] * 1.005
            sl = min(supports) if len(supports) > 0 else last['close'] * 0.98
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'LONG'
            }
        
        short_signal = None
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            entry = last['close'] * 0.995
            sl = max(resistances) if len(resistances) > 0 else last['close'] * 1.02
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'SHORT'
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

def update_task():
    while True:
        try:
            with trading_system.lock:
                trading_system.is_updating = True
                trading_system.update_progress = 0
                
                cryptos = trading_system.load_cryptos()
                total = len(cryptos)
                processed = 0
                
                params = trading_system.params
                timeframe = params['timeframe']
                
                if timeframe not in trading_system.timeframe_data:
                    trading_system.timeframe_data[timeframe] = {
                        'long_signals': [],
                        'short_signals': [],
                        'scatter_data': [],
                        'historical_signals': []
                    }
                
                tf_data = trading_system.timeframe_data[timeframe]
                long_signals = []
                short_signals = []
                scatter_data = []
                
                for crypto in cryptos:
                    try:
                        long_sig, short_sig, long_prob, short_prob, vol = analyze_crypto(crypto, params)
                        
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
                        
                        long_prev, short_prev, _, _, _ = analyze_crypto(crypto, params, True)
                        if long_prev:
                            tf_data['historical_signals'].append(long_prev)
                        if short_prev:
                            tf_data['historical_signals'].append(short_prev)
                            
                    except Exception as e:
                        logger.error(f"Error procesando {crypto}: {str(e)}")
                    
                    processed += 1
                    progress = int((processed / total) * 100)
                    trading_system.update_progress = progress
                    time.sleep(0.5)
                
                tf_data['long_signals'] = sorted(long_signals, key=lambda x: x['adx'], reverse=True)
                tf_data['short_signals'] = sorted(short_signals, key=lambda x: x['adx'], reverse=True)
                tf_data['scatter_data'] = scatter_data
                tf_data['historical_signals'] = tf_data['historical_signals'][-50:]
                
                trading_system.cryptos_analyzed = total
                trading_system.last_update = datetime.now()
                trading_system.is_updating = False
                
        except Exception as e:
            logger.error(f"Error en update_task: {str(e)}")
            trading_system.is_updating = False
        
        trading_system.update_event.clear()
        trading_system.update_event.wait(CACHE_TIME)

update_thread = threading.Thread(target=update_task, daemon=True)
update_thread.start()

@app.route('/')
def index():
    with trading_system.lock:
        params = trading_system.params
        timeframe = params['timeframe']
        
        if timeframe in trading_system.timeframe_data:
            tf_data = trading_system.timeframe_data[timeframe]
            long_signals = tf_data['long_signals'][:50]
            short_signals = tf_data['short_signals'][:50]
            scatter_data = tf_data['scatter_data']
            historical_signals = tf_data['historical_signals'][-20:]
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
                             last_update=trading_system.last_update,
                             params=params,
                             avg_adx_long=round(avg_adx_long, 1),
                             avg_adx_short=round(avg_adx_short, 1),
                             scatter_data=scatter_ready,
                             cryptos_analyzed=trading_system.cryptos_analyzed,
                             is_updating=trading_system.is_updating,
                             update_progress=trading_system.update_progress)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        params = trading_system.params
        df = get_kucoin_data(symbol, params['timeframe'], 100)
        if df is None:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None:
            return "Datos insuficientes", 404
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA{params["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA{params["ema_slow"]}', color='green', alpha=0.8)
        plt.title(f'{symbol} - Price & EMAs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 2)
        plt.bar(df['timestamp'], df['volume'], color=['green' if c > o else 'red' for c, o in zip(df['close'], df['open'])])
        plt.title('Volume')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        plt.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        plt.title('RSI')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 4)
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.plot(df['timestamp'], df['plus_di'], label='+DI', color='green', alpha=0.7)
        plt.plot(df['timestamp'], df['minus_di'], label='-DI', color='red', alpha=0.7)
        plt.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.7)
        plt.title('ADX & DI')
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
        return "Error generando gráfico", 500

@app.route('/historical_chart/<symbol>/<signal_type>')
def get_historical_chart(symbol, signal_type):
    try:
        params = trading_system.params
        df = get_kucoin_data(symbol, params['timeframe'], 100)
        if df is None:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None:
            return "Datos insuficientes", 404        
            
        plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA{params["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA{params["ema_slow"]}', color='green', alpha=0.8)
        
        if len(df) > 1:
            plt.axvline(x=df['timestamp'].iloc[-2], color='gray', linestyle='--', alpha=0.7, label='Vela Anterior')
        
        plt.title(f'{symbol} - Historical Signal ({signal_type.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 2)
        plt.bar(df['timestamp'], df['volume'], color=['green' if c > o else 'red' for c, o in zip(df['close'], df['open'])])
        if len(df) > 1:
            plt.axvline(x=df['timestamp'].iloc[-2], color='gray', linestyle='--', alpha=0.7)
        plt.title('Volume')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        if len(df) > 1:
            plt.axvline(x=df['timestamp'].iloc[-2], color='gray', linestyle='--', alpha=0.7)
        plt.title('RSI')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 4)
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.7)
        if len(df) > 1:
            plt.axvline(x=df['timestamp'].iloc[-2], color='gray', linestyle='--', alpha=0.7)
        plt.title('ADX')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('historical_chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        return "Error generando gráfico histórico", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/update_params', methods=['POST'])
def update_params():
    try:
        data = request.form.to_dict()
        new_params = trading_system.params.copy()
        
        for key in new_params:
            if key in data:
                if key in ['ema_fast', 'ema_slow', 'adx_period', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    new_params[key] = int(data[key])
                elif key in ['adx_level', 'max_risk_percent', 'price_distance_threshold', 'volume_filter']:
                    new_params[key] = float(data[key])
                else:
                    new_params[key] = data[key]
        
        with trading_system.lock:
            trading_system.params = new_params
        
        trading_system.update_event.set()
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados correctamente'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/status')
def status():
    return jsonify({
        'is_updating': trading_system.is_updating,
        'progress': trading_system.update_progress,
        'last_update': trading_system.last_update.isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
