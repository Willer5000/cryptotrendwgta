import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import logging

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.logger.setLevel(logging.INFO)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 900  # 15 minutos
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

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Obtener datos de KuCoin
def get_kucoin_data(symbol, timeframe):
    tf_mapping = {
        '30m': '30min',
        '1h': '1hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': '1day',
        '1w': '1week'
    }
    kucoin_tf = tf_mapping.get(timeframe, '1hour')
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                candles.reverse()  # Invertir para orden cronológico
                
                if len(candles) < 100:
                    app.logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir a tipos numéricos
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                if len(df) < 50:
                    app.logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                    return None
                
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                return df
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window):
    if len(series) < window + 1:
        return pd.Series([np.nan] * len(series))
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Evitar división por cero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close)), pd.Series([np.nan] * len(close))
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm <= 0] = 0
    minus_dm[minus_dm <= 0] = 0
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).replace([np.inf, -np.inf], 0)
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di

# Implementación manual de ATR
def calculate_atr(high, low, close, window):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # ATR
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        return df
    except Exception as e:
        app.logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
        df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
        
        resistances = df[df['high'] >= df['high_roll']]['high'].unique().tolist()
        supports = df[df['low'] <= df['low_roll']]['low'].unique().tolist()
        
        return supports, resistances
    except Exception as e:
        app.logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol == 0 or current_vol is None or avg_vol is None:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.0: return 'Muy Alto'
        if ratio > 1.5: return 'Alto'
        if ratio > 1.0: return 'Medio'
        if ratio > 0.5: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback):
    try:
        if len(df) < lookback + 1:
            return None
            
        recent = df.iloc[-lookback:]
        max_idx = recent['close'].idxmax()
        min_idx = recent['close'].idxmin()
        
        # Divergencia bajista
        if not pd.isna(max_idx):
            price_high = df.loc[max_idx, 'close']
            rsi_high = df.loc[max_idx, 'rsi']
            current_price = df.iloc[-1]['close']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
                return 'bearish'
        
        # Divergencia alcista
        if not pd.isna(min_idx):
            price_low = df.loc[min_idx, 'close']
            rsi_low = df.loc[min_idx, 'rsi']
            current_price = df.iloc[-1]['close']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
                return 'bullish'
        
        return None
    except Exception as e:
        app.logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Calcular distancia al nivel más cercano
def calculate_distance_to_level(price, levels, threshold_percent):
    try:
        if not levels or price is None:
            return False
        min_distance = min(abs(price - level) for level in levels)
        threshold = price * threshold_percent / 100
        return min_distance <= threshold
    except:
        return False

# Analizar una criptomoneda y calcular probabilidades
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 50:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.005 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.995 for s in supports) if supports else False
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG
        if trend == 'up': long_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: long_prob += 20
        if is_breakout or divergence == 'bullish': long_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 15
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): long_prob += 10
        
        # Criterios para SHORT
        if trend == 'down': short_prob += 30
        if not pd.isna(last['adx']) and last['adx'] > params['adx_level']: short_prob += 20
        if is_breakdown or divergence == 'bearish': short_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 15
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): short_prob += 10
        
        # Señales LONG
        long_signal = None
        if long_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * (1 - params['max_risk_percent']/100)
            risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'tp3': round(entry + risk * 3, 4),
                'volume': volume_class,
                'divergence': divergence == 'bullish',
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 70 and volume_class in ['Alto', 'Muy Alto']:
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * (1 + params['max_risk_percent']/100)
            risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'tp3': round(entry - risk * 3, 4),
                'volume': volume_class,
                'divergence': divergence == 'bearish',
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'price': round(last['close'], 4),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        with app.app_context():
            try:
                app.logger.info("Iniciando actualización de datos...")
                cryptos = load_cryptos()
                long_signals = []
                short_signals = []
                scatter_data = []
                
                batch_size = 20
                for i in range(0, len(cryptos), batch_size):
                    batch = cryptos[i:i+batch_size]
                    for crypto in batch:
                        long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                        if long_signal:
                            long_signals.append(long_signal)
                        if short_signal:
                            short_signals.append(short_signal)
                        
                        scatter_data.append({
                            'symbol': crypto,
                            'long_prob': long_prob,
                            'short_prob': short_prob,
                            'volume': volume_class
                        })
                    time.sleep(1)
                
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                cache.set('cryptos_list', cryptos)
                
                app.logger.info(f"Actualización completada: {len(long_signals)} LONG, {len(short_signals)} SHORT")
            except Exception as e:
                app.logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
try:
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    app.logger.info("Hilo de actualización iniciado")
except Exception as e:
    app.logger.error(f"No se pudo iniciar el hilo de actualización: {str(e)}")

@app.route('/')
def index():
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    cryptos_list = cache.get('cryptos_list') or []
    
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=DEFAULTS,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=json.dumps(scatter_data),
                           cryptos_list=cryptos_list,
                           cryptos_count=len(cryptos_list),
                           long_count=len(long_signals),
                           short_count=len(short_signals))

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    return "Gráfico temporalmente deshabilitado", 503

@app.route('/manual')
def manual():
    return render_template('manual.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
