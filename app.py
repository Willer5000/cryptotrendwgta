import os
import time
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import json
import math
import logging

app = Flask(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_FILE = 'signals_cache.json'
CACHE_TIME = 900  # 15 minutos
DEFAULTS = {
    'timeframe': '1h',
    'ema_fast': 9,
    'ema_slow': 20,
    'adx_period': 14,
    'adx_level': 25,
    'rsi_period': 14,
    'sr_window': 50,
    'divergence_lookback': 20
}

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        logging.error(f"Error loading cryptos: {str(e)}")
        return []

# Cache de señales
long_signals_cache = []
short_signals_cache = []
last_update_time = None

# Implementación manual de EMA
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm <= 0] = 0
    minus_dm[minus_dm >= 0] = 0
    minus_dm = -minus_dm
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window).mean()
    return adx

# Implementación manual de ATR
def calculate_atr(high, low, close, window):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

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
    kucoin_tf = tf_mapping[timeframe]
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={symbol}-USDT"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                # KuCoin devuelve las velas en orden descendente, invertimos para tener ascendente
                candles.reverse()
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})
                # Convertir timestamp a entero para evitar el warning
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                return df
        return None
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'], params['adx_period'])
        
        # ATR
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        return df

# Detectar soportes y resistencias
def find_support_resistance(df, window):
    try:
        df['high_roll'] = df['high'].rolling(window=window).max()
        df['low_roll'] = df['low'].rolling(window=window).min()
        
        resistances = df[df['high'] == df['high_roll']]['high'].unique().tolist()
        supports = df[df['low'] == df['low_roll']]['low'].unique().tolist()
        
        return supports, resistances
    except Exception as e:
        logging.error(f"Error finding S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    if avg_vol == 0 or math.isnan(avg_vol):
        return 'Muy Bajo'
    ratio = current_vol / avg_vol
    if ratio > 2.0: return 'Muy Alto'
    if ratio > 1.5: return 'Alto'
    if ratio > 1.0: return 'Medio'
    if ratio > 0.5: return 'Bajo'
    return 'Muy Bajo'

# Detectar divergencias
def detect_divergence(df, lookback):
    if len(df) < lookback + 1:
        return None
        
    # Seleccionar los últimos datos
    recent = df.iloc[-lookback:]
    
    # Buscar máximos/mínimos recientes
    max_idx = recent['close'].idxmax() if not recent['close'].empty else -1
    min_idx = recent['close'].idxmin() if not recent['close'].empty else -1
    
    # Divergencia bajista
    if max_idx != -1:
        price_high = df.loc[max_idx, 'close']
        rsi_high = df.loc[max_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
            return 'bearish'
    
    # Divergencia alcista
    if min_idx != -1:
        price_low = df.loc[min_idx, 'close']
        rsi_low = df.loc[min_idx, 'rsi']
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
            return 'bullish'
    
    return None

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        logging.info(f"Not enough data for {symbol}")
        return None, None
    
    try:
        df = calculate_indicators(df, params)
        if df is None or 'close' not in df.columns:
            return None, None
            
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
        
        # Señales LONG
        long_signal = None
        if (trend == 'up' and last['adx'] > params['adx_level'] and 
            (is_breakout or divergence == 'bullish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar la resistencia más cercana por encima
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            # Encontrar el soporte más cercano por debajo para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * 0.98
            risk = entry - sl
            
            # Preparar datos para el gráfico
            chart_data = {
                'prices': df['close'].tail(100).tolist(),
                'ema_fast': df['ema_fast'].tail(100).tolist(),
                'ema_slow': df['ema_slow'].tail(100).tolist(),
                'supports': supports,
                'resistances': resistances,
                'entry': entry,
                'sl': sl,
                'tp1': entry + risk,
                'tp2': entry + risk * 2,
                'tp3': entry + risk * 3,
                'timestamps': df['timestamp'].tail(100).dt.strftime('%Y-%m-%d %H:%M').tolist()
            }
            
            long_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'tp3': round(entry + risk * 3, 4),
                'volume': volume_class,
                'divergence': divergence == 'bullish',
                'adx': round(last['adx'], 2),
                'chart_data': chart_data
            }
        
        # Señales SHORT
        short_signal = None
        if (trend == 'down' and last['adx'] > params['adx_level'] and 
            (is_breakdown or divergence == 'bearish') and 
            volume_class in ['Alto', 'Muy Alto']):
            
            # Encontrar el soporte más cercano por debajo
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            # Encontrar la resistencia más cercana por encima para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * 1.02
            risk = sl - entry
            
            # Preparar datos para el gráfico
            chart_data = {
                'prices': df['close'].tail(100).tolist(),
                'ema_fast': df['ema_fast'].tail(100).tolist(),
                'ema_slow': df['ema_slow'].tail(100).tolist(),
                'supports': supports,
                'resistances': resistances,
                'entry': entry,
                'sl': sl,
                'tp1': entry - risk,
                'tp2': entry - risk * 2,
                'tp3': entry - risk * 3,
                'timestamps': df['timestamp'].tail(100).dt.strftime('%Y-%m-%d %H:%M').tolist()
            }
            
            short_signal = {
                'symbol': symbol,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'tp3': round(entry - risk * 3, 4),
                'volume': volume_class,
                'divergence': divergence == 'bearish',
                'adx': round(last['adx'], 2),
                'chart_data': chart_data
            }
        
        return long_signal, short_signal
    except Exception as e:
        logging.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None

# Actualizar señales en caché
def update_signals_cache():
    global long_signals_cache, short_signals_cache, last_update_time
    try:
        start_time = time.time()
        logging.info("Starting signals cache update...")
        cryptos = load_cryptos()
        long_signals = []
        short_signals = []
        
        params = DEFAULTS.copy()
        
        for crypto in cryptos:
            long_signal, short_signal = analyze_crypto(crypto, params)
            if long_signal:
                long_signals.append(long_signal)
            if short_signal:
                short_signals.append(short_signal)
            # Pequeña pausa para evitar sobrecargar la API
            time.sleep(0.1)
        
        # Ordenar por fuerza de tendencia (ADX)
        long_signals.sort(key=lambda x: x['adx'], reverse=True)
        short_signals.sort(key=lambda x: x['adx'], reverse=True)
        
        long_signals_cache = long_signals
        short_signals_cache = short_signals
        last_update_time = datetime.now()
        
        # Guardar en archivo
        with open(CACHE_FILE, 'w') as f:
            json.dump({
                'long_signals': long_signals,
                'short_signals': short_signals,
                'last_update_time': last_update_time.isoformat()
            }, f)
        
        logging.info(f"Cache updated in {time.time()-start_time:.2f}s. Found {len(long_signals)} LONG and {len(short_signals)} SHORT signals.")
    except Exception as e:
        logging.error(f"Error updating cache: {str(e)}")

# Cargar caché desde archivo
def load_cache_from_file():
    global long_signals_cache, short_signals_cache, last_update_time
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                long_signals_cache = data.get('long_signals', [])
                short_signals_cache = data.get('short_signals', [])
                last_update_time = datetime.fromisoformat(data.get('last_update_time', datetime.now().isoformat()))
                logging.info("Cache loaded from file")
    except Exception as e:
        logging.error(f"Error loading cache: {str(e)}")

# Programar actualizaciones
def schedule_updates():
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_signals_cache, 'interval', minutes=15)
    scheduler.start()
    logging.info("Scheduler started")

# Inicializar la aplicación
@app.before_first_request
def initialize():
    load_cache_from_file()
    if not last_update_time or (datetime.now() - last_update_time).total_seconds() > CACHE_TIME:
        update_signals_cache()
    schedule_updates()

@app.route('/', methods=['GET', 'POST'])
def index():
    params = DEFAULTS.copy()
    
    if request.method == 'POST':
        for key in params:
            if key in request.form:
                # Convertir a int si es numérico, de lo contrario dejar como está
                if key != 'timeframe':
                    params[key] = int(request.form[key])
                else:
                    params[key] = request.form[key]
    
    # Estadísticas para gráficos adicionales
    signal_count = len(long_signals_cache) + len(short_signals_cache)
    avg_adx_long = np.mean([s['adx'] for s in long_signals_cache]) if long_signals_cache else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals_cache]) if short_signals_cache else 0
    
    # Calcular distribución por fuerza de tendencia
    strength_distribution = {
        'Fuerte (ADX > 40)': len([s for s in long_signals_cache + short_signals_cache if s['adx'] > 40]),
        'Moderada (ADX 25-40)': len([s for s in long_signals_cache + short_signals_cache if 25 <= s['adx'] <= 40]),
        'Débil (ADX < 25)': len([s for s in long_signals_cache + short_signals_cache if s['adx'] < 25])
    }
    
    return render_template('index.html', 
                           long_signals=long_signals_cache[:50],  # Mostrar top 50
                           short_signals=short_signals_cache[:50],  # Mostrar top 50
                           params=params,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2) if avg_adx_long else 0,
                           avg_adx_short=round(avg_adx_short, 2) if avg_adx_short else 0,
                           strength_distribution=strength_distribution,
                           last_update_time=last_update_time)

@app.route('/get_chart_data/<symbol>/<signal_type>')
def get_chart_data(symbol, signal_type):
    try:
        if signal_type == 'long':
            signal = next((s for s in long_signals_cache if s['symbol'] == symbol), None)
        else:
            signal = next((s for s in short_signals_cache if s['symbol'] == symbol), None)
        
        if signal:
            return jsonify(signal['chart_data'])
        return jsonify({'error': 'Signal not found'}), 404
    except Exception as e:
        logging.error(f"Error getting chart data: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
