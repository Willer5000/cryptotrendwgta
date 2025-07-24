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

# Configuración Redis mejorada para Render
def get_redis_connection():
    redis_url = os.getenv('REDIS_URL')
    
    # Intenta obtener de variables de entorno alternativas
    if not redis_url:
        redis_url = os.getenv('REDISCLOUD_URL')
    
    if not redis_url:
        print("⚠️ REDIS_URL no configurada. Usando modo sin Redis.")
        return None
    
    print(f"🔗 Conectando a Redis: {redis_url}")
    try:
        r = redis.Redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=False,
            ssl_cert_reqs=None  # Importante para conexiones SSL
        )
        r.ping()  # Test de conexión
        print("✅ Conexión Redis exitosa")
        return r
    except Exception as e:
        print(f"❌ Error conectando a Redis: {str(e)}")
        return None

# Inicializar conexión Redis
r = get_redis_connection()

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

# 1. Funciones de indicadores corregidas
def ema_macro_signal(df):
    if len(df) < 100:
        return 0
    
    try:
        ema100 = ta.trend.ema_indicator(df['close'], window=100)
        ema200 = ta.trend.ema_indicator(df['close'], window=200)
        
        if pd.isna(ema100.iloc[-1]) or pd.isna(ema200.iloc[-1]):
            return 0
            
        return 1 if ema100.iloc[-1] > ema200.iloc[-1] else -1
    except Exception:
        return 0

def volume_profile_signal(df, bins=20):
    if len(df) < 50:
        return 0
        
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        min_tp = typical_price.min()
        max_tp = typical_price.max()
        
        if min_tp >= max_tp:
            return 0
            
        bin_size = (max_tp - min_tp) / bins
        bin_edges = np.linspace(min_tp, max_tp, bins + 1)
        
        hist, _ = np.histogram(typical_price, bins=bin_edges, weights=df['volume'])
        max_vol_index = np.argmax(hist)
        poc = bin_edges[max_vol_index]
        
        return 1 if df['close'].iloc[-1] > poc else -1
    except Exception:
        return 0

def adaptive_rsi_value(df):
    if len(df) < 20:
        return 50
        
    try:
        rsi = ta.momentum.rsi(df['close'], window=14)
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        if pd.isna(rsi.iloc[-1]) or pd.isna(atr.iloc[-1]):
            return 50
            
        mean_close = df['close'].mean()
        
        # Evitar división por cero
        if mean_close == 0:
            return rsi.iloc[-1]
            
        volatility_factor = atr.iloc[-1] / mean_close
        
        # Limitar factor de volatilidad
        volatility_factor = min(volatility_factor, 0.5)  # Máximo 50% de ajuste
        
        return rsi.iloc[-1] * (1 + volatility_factor)
    except Exception:
        return 50

# 2. Sistema de puntuación mejorado
def calculate_score(symbol, df):
    try:
        ema_score = ema_macro_signal(df) * 0.4
        vp_score = volume_profile_signal(df) * 0.3
        
        ar = adaptive_rsi_value(df)
        rsi_score = 1 if ar > 60 else -1 if ar < 40 else 0
        rsi_score *= 0.3
        
        total = ema_score + vp_score + rsi_score
        direction = 'LONG' if total > 0.5 else 'SHORT' if total < -0.5 else 'NEUTRAL'
        confidence = min(100, max(0, abs(total * 100)))
        
        return {
            'symbol': symbol,
            'total_score': total,
            'direction': direction,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error calculating score for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'total_score': 0,
            'direction': 'NEUTRAL',
            'confidence': 0
        }

# 3. Gestión de riesgo robusta
def calculate_risk_parameters(df, direction):
    try:
        # Calcular ATR con ventana más pequeña si es necesario
        atr_window = min(14, len(df) - 1)
        if atr_window < 5:
            atr_value = df['high'].iloc[-1] - df['low'].iloc[-1]
        else:
            atr = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=atr_window)
            atr_value = atr.iloc[-1]
        
        close = df['close'].iloc[-1]
        
        # Asegurar mínimo 5 velas disponibles
        lookback = min(5, len(df))
        
        if direction == 'LONG':
            entry = close * 1.001
            sl = min(df['low'].iloc[-lookback:].min(), close - (atr_value * 1.5))
            tp1 = close + (atr_value * 1)
            tp2 = close + (atr_value * 2)
            tp3 = close + (atr_value * 3)
            rr = abs((entry - sl) / (tp3 - entry)) if (tp3 - entry) > 0 else 1
        else:  # SHORT
            entry = close * 0.999
            sl = max(df['high'].iloc[-lookback:].max(), close + (atr_value * 1.5))
            tp1 = close - (atr_value * 1)
            tp2 = close - (atr_value * 2)
            tp3 = close - (atr_value * 3)
            rr = abs((entry - sl) / (entry - tp3)) if (entry - tp3) > 0 else 1
        
        return {
            'entry': round(entry, 4),
            'stop_loss': round(sl, 4),
            'tp1': round(tp1, 4),
            'tp2': round(tp2, 4),
            'tp3': round(tp3, 4),
            'risk_reward': round(rr, 1)
        }
    except Exception as e:
        print(f"Error calculating risk for {direction}: {str(e)}")
        return {
            'entry': close,
            'stop_loss': 0,
            'tp1': 0,
            'tp2': 0,
            'tp3': 0,
            'risk_reward': 0
        }

# 4. Fuente de datos con manejo de errores mejorado
def get_top_symbols():
    cache_key = "top_symbols"
    
    # Intento obtener de Redis si está disponible
    if r:
        try:
            cached = r.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {str(e)}")
    
    # Lista de símbolos de respaldo
    backup_symbols = [
        "BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "ADA-USDT",
        "SOL-USDT", "DOGE-USDT", "DOT-USDT", "TRX-USDT", "MATIC-USDT"
    ]
    
    try:
        url = "https://api.kucoin.com/api/v1/market/allTickers"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"KuCoin API error: {response.status_code}")
            return backup_symbols
            
        data = response.json()
        
        if data.get('code') != '200000':
            print(f"KuCoin API error: {data.get('msg')}")
            return backup_symbols
        
        tickers = data['data']['ticker']
        sorted_tickers = sorted(
            tickers, 
            key=lambda x: float(x.get('vol', 0)), 
            reverse=True
        )[:50]
        
        symbols = [ticker['symbol'] for ticker in sorted_tickers if '-' in ticker['symbol']]
        
        if not symbols:
            return backup_symbols
            
        if r:
            try:
                r.setex(cache_key, 3600, json.dumps(symbols))
            except Exception as e:
                print(f"Redis set error: {str(e)}")
        
        return symbols
    except Exception as e:
        print(f"Error getting top symbols: {str(e)}")
        return backup_symbols

def fetch_ohlcv(symbol, timeframe, limit=500):
    cache_key = f"{symbol}_{timeframe}"
    
    # Intento obtener de Redis si está disponible
    if r:
        try:
            cached = r.get(cache_key)
            if cached:
                df = pd.read_json(cached)
                if not df.empty:
                    return df
        except Exception as e:
            print(f"Redis get error: {str(e)}")
    
    # Mapeo de timeframes
    kucoin_tf_map = {
        '15m': '15min',
        '30m': '30min',
        '1h': '1hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': '1day',
        '1w': '1week'
    }
    
    kucoin_tf = kucoin_tf_map.get(timeframe, '1hour')
    
    end_time = int(time.time())
    start_time = end_time - (TIMEFRAME_MAP[timeframe] * limit * 1.5)
    
    try:
        url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}"
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            print(f"KuCoin API error for {symbol}: {response.status_code}")
            return None
        
        data = response.json().get('data', [])
        if not data:
            print(f"No data for {symbol} {timeframe}")
            return None
        
        # Crear DataFrame con manejo de datos vacíos
        columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
        df = pd.DataFrame(data, columns=columns)
        
        # Revertir orden y resetear índice
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas con valores nulos
        df = df.dropna()
        
        # Limitar a 500 velas máximo
        if len(df) > limit:
            df = df.tail(limit)
        
        # Guardar en caché solo si tenemos datos válidos
        if not df.empty and r:
            try:
                r.setex(cache_key, 600, df.to_json())
            except Exception as e:
                print(f"Redis set error: {str(e)}")
                
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
        return None

# 5. Procesamiento principal con redundancia
def generate_recommendations(timeframe):
    symbols = get_top_symbols()
    recommendations = []
    
    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol, timeframe)
            if df is None or len(df) < 30:
                continue
                
            score_data = calculate_score(symbol, df)
            
            # Solo considerar señales válidas
            if score_data['direction'] in ['LONG', 'SHORT']:
                risk_data = calculate_risk_parameters(df, score_data['direction'])
                
                # Solo agregar si tenemos datos de riesgo válidos
                if risk_data['entry'] > 0 and risk_data['risk_reward'] > 0:
                    recommendations.append({
                        **score_data,
                        **risk_data,
                        'timeframe': timeframe
                    })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Ordenar por confianza y limitar a 10
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    return recommendations[:10]

# 6. Endpoints con manejo de errores mejorado
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations')
def recommendations():
    try:
        timeframe = request.args.get('timeframe', '1h')
        min_confidence = float(request.args.get('min_confidence', 0))
        direction = request.args.get('direction', 'ALL')
        min_rr = float(request.args.get('min_rr', 1))
        
        # Generar recomendaciones directamente sin caché si Redis falla
        all_recs = generate_recommendations(timeframe)
        
        # Filtrado seguro
        filtered = []
        for rec in all_recs:
            try:
                if (rec['confidence'] >= min_confidence and
                    (direction == 'ALL' or rec['direction'] == direction) and
                    rec['risk_reward'] >= min_rr):
                    filtered.append(rec)
            except:
                continue
        
        return jsonify(filtered)
    
    except Exception as e:
        print(f"Error in recommendations endpoint: {str(e)}")
        return jsonify([])

@app.route('/heatmap')
def heatmap_data():
    try:
        timeframes = ['15m', '30m', '1h', '4h', '1d', '1w']
        data = []
        
        for tf in timeframes:
            recs = generate_recommendations(tf) or []
            for rec in recs:
                try:
                    data.append({
                        'symbol': rec['symbol'],
                        'timeframe': tf,
                        'score': rec['total_score'],
                        'confidence': rec['confidence']
                    })
                except:
                    continue
        
        return jsonify(data)
    except Exception as e:
        print(f"Error in heatmap endpoint: {str(e)}")
        return jsonify([])

@app.route('/status')
def status():
    try:
        redis_status = "active" if r and r.ping() else "inactive"
        return jsonify({
            "status": "online",
            "redis": redis_status,
            "timestamp": datetime.utcnow().isoformat()
        })
    except:
        return jsonify({
            "status": "online",
            "redis": "inactive",
            "timestamp": datetime.utcnow().isoformat()
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
