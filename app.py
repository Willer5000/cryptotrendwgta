import os
import time
import json
import requests
import pandas as pd
import numpy as np
import ta
import redis
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta

app = Flask(__name__)

# Configuración Redis
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
r = redis.Redis.from_url(redis_url)

# Mapeo de timeframes
TIMEFRAME_MAP = {
    '15m': 15*60,
    '30m': 30*60,
    '1h': 60*60,
    '2h': 2*60*60,
    '4h': 4*60*60,
    '1d': 24*60*60,
    '1w': 7*24*60*60
}

# 1. Funciones de indicadores
def ema_macro_signal(df):
    ema100 = ta.trend.ema_indicator(df['close'], 100)
    ema200 = ta.trend.ema_indicator(df['close'], 200)
    return 1 if ema100.iloc[-1] > ema200.iloc[-1] else -1

def volume_profile_signal(df, bins=20):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    min_tp, max_tp = typical_price.min(), typical_price.max()
    bin_size = (max_tp - min_tp) / bins
    bin_edges = [min_tp + i * bin_size for i in range(bins+1)]
    
    hist, _ = np.histogram(typical_price, bins=bin_edges, weights=df['volume'])
    max_vol_index = np.argmax(hist)
    poc = bin_edges[max_vol_index] + bin_size/2
    
    return 1 if df['close'].iloc[-1] > poc else -1

def adaptive_rsi_value(df):
    rsi = ta.momentum.rsi(df['close'], 14)
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    mean_close = df['close'].mean()
    
    if pd.isna(rsi.iloc[-1]) or pd.isna(atr.iloc[-1]):
        return 50
    
    return rsi.iloc[-1] * (1 + (atr.iloc[-1] / mean_close))

# 2. Sistema de puntuación
def calculate_score(symbol, df):
    ema_score = ema_macro_signal(df) * 0.4
    vp_score = volume_profile_signal(df) * 0.3
    
    ar = adaptive_rsi_value(df)
    rsi_score = 1 if ar > 60 else -1 if ar < 40 else 0
    rsi_score *= 0.3
    
    total = ema_score + vp_score + rsi_score
    direction = 'LONG' if total > 0.5 else 'SHORT'
    
    return {
        'symbol': symbol,
        'total_score': total,
        'direction': direction,
        'confidence': min(100, abs(total * 100))
    }

# 3. Gestión de riesgo
def calculate_risk_parameters(df, direction):
    atr = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], 14).iloc[-1]
    close = df['close'].iloc[-1]
    
    if direction == 'LONG':
        entry = close * 1.001
        sl = min(df['low'].iloc[-5:].min(), close - (atr * 1.5))
        tp1 = close + (atr * 1)
        tp2 = close + (atr * 2)
        tp3 = close + (atr * 3)
        rr = abs((entry - sl) / (tp3 - entry))
    else:  # SHORT
        entry = close * 0.999
        sl = max(df['high'].iloc[-5:].max(), close + (atr * 1.5))
        tp1 = close - (atr * 1)
        tp2 = close - (atr * 2)
        tp3 = close - (atr * 3)
        rr = abs((entry - sl) / (entry - tp3))
    
    return {
        'entry': round(entry, 4),
        'stop_loss': round(sl, 4),
        'tp1': round(tp1, 4),
        'tp2': round(tp2, 4),
        'tp3': round(tp3, 4),
        'risk_reward': round(rr, 1)
    }

# 4. Fuente de datos
def get_top_symbols():
    cache_key = "top_symbols"
    cached = r.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    url = "https://api.kucoin.com/api/v1/market/allTickers"
    response = requests.get(url)
    data = response.json()
    
    if data['code'] != '200000':
        return []
    
    tickers = data['data']['ticker']
    sorted_tickers = sorted(
        tickers, 
        key=lambda x: float(x['vol']), 
        reverse=True
    )[:50]
    
    symbols = [ticker['symbol'] for ticker in sorted_tickers]
    r.setex(cache_key, 3600, json.dumps(symbols))  # Cache 1 hora
    return symbols

def fetch_ohlcv(symbol, timeframe, limit=500):
    cache_key = f"{symbol}_{timeframe}"
    cached = r.get(cache_key)
    
    if cached:
        return pd.DataFrame(json.loads(cached))
    
    kucoin_tf = {
        '15m': '15min',
        '30m': '30min',
        '1h': '1hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': '1day',
        '1w': '1week'
    }[timeframe]
    
    end_time = int(time.time())
    start_time = end_time - (TIMEFRAME_MAP[timeframe] * limit * 1.2)
    
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}&startAt={start_time}&endAt={end_time}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None
    
    data = response.json().get('data', [])
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Convertir tipos de datos
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    for col in ['open', 'close', 'high', 'low', 'volume']:
        df[col] = df[col].astype(float)
    
    if len(df) > limit:
        df = df.tail(limit)
    
    r.setex(cache_key, 600, df.to_json())  # Cache 10 minutos
    return df

# 5. Procesamiento principal
def generate_recommendations(timeframe):
    symbols = get_top_symbols()
    recommendations = []
    
    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol, timeframe)
            if df is None or len(df) < 100:
                continue
                
            score_data = calculate_score(symbol, df)
            risk_data = calculate_risk_parameters(df, score_data['direction'])
            
            recommendations.append({
                **score_data,
                **risk_data,
                'timeframe': timeframe
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:10]

# 6. Endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations')
def recommendations():
    timeframe = request.args.get('timeframe', '1h')
    min_confidence = float(request.args.get('min_confidence', 0))
    direction = request.args.get('direction', 'ALL')
    min_rr = float(request.args.get('min_rr', 1))
    
    cache_key = f"recs_{timeframe}"
    cached = r.get(cache_key)
    
    if cached:
        all_recs = json.loads(cached)
    else:
        all_recs = generate_recommendations(timeframe)
        r.setex(cache_key, 600, json.dumps(all_recs))  # Cache 10 minutos
    
    # Filtrado
    filtered = [
        rec for rec in all_recs 
        if rec['confidence'] >= min_confidence and
           (direction == 'ALL' or rec['direction'] == direction) and
           rec['risk_reward'] >= min_rr
    ]
    
    return jsonify(filtered)

@app.route('/heatmap')
def heatmap_data():
    timeframes = ['15m', '30m', '1h', '4h', '1d', '1w']
    data = []
    
    for tf in timeframes:
        recs = generate_recommendations(tf)
        for rec in recs:
            data.append({
                'symbol': rec['symbol'],
                'timeframe': tf,
                'score': rec['total_score'],
                'confidence': rec['confidence']
            })
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
