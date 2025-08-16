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
import io
import base64
import math
import logging
import plotly
import plotly.graph_objs as go
from threading import Lock, Event

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
UPDATE_INTERVAL = 300  # 5 minutos
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

# Estado global con bloqueo
class AppState:
    def __init__(self):
        self.lock = Lock()
        self.long_signals = []
        self.short_signals = []
        self.scatter_data = []
        self.last_update = datetime.now()
        self.cryptos_analyzed = 0
        self.historical_signals = []
        self.is_ready = Event()
        self.is_ready.set()  # Inicialmente listo

app_state = AppState()

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(cryptos)} cryptocurrencies")
            return cryptos
    except Exception as e:
        logger.error(f"Error loading cryptos: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA']  # Lista de respaldo

# Obtener datos de KuCoin con reintentos
def get_kucoin_data(symbol, timeframe, retries=3):
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
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        logger.warning(f"No data for {symbol} on {timeframe}")
                        return None
                    
                    candles.reverse()
                    
                    if len(candles) < 100:
                        logger.warning(f"Insufficient data for {symbol}: {len(candles)} candles")
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    # Convertir a tipos numéricos
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Eliminar filas con valores NaN
                    df = df.dropna()
                    
                    if len(df) < 50:
                        logger.warning(f"Insufficient data after cleaning for {symbol}: {len(df)} candles")
                        return None
                    
                    # Convertir timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                    return df
                else:
                    logger.warning(f"Unexpected response for {symbol}: {data.get('msg')}")
            else:
                logger.warning(f"HTTP error for {symbol}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} (attempt {attempt+1}): {str(e)}")
            time.sleep(2)
    
    return None

# Implementación manual de EMA
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI
def calculate_rsi(series, window=14):
    if len(series) < window + 1:
        return pd.Series([50] * len(series))
    
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
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm <= 0] = 0
        minus_dm[minus_dm <= 0] = 0
        
        # Calcular True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
        adx = dx.rolling(window).mean()
        return adx, plus_di, minus_di
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))

# Calcular indicadores manualmente
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'], 
            df['low'], 
            df['close'], 
            params['adx_period']
        )
        
        # Eliminar filas con NaN resultantes de los cálculos
        df = df.dropna()
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return None

# Detectar soportes y resistencias mejorado
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Identificar pivots usando ventana móvil
        highs = df['high'].values
        lows = df['low'].values
        
        resistances = []
        supports = []
        
        # Buscar resistencias (máximos locales)
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window]):
                resistances.append(highs[i])
        
        # Buscar soportes (mínimos locales)
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i - window:i + window]):
                supports.append(lows[i])
        
        # Consolidar niveles cercanos (dentro del 0.5%)
        resistances = sorted(set(resistances))
        supports = sorted(set(supports))
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error finding S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0 or current_vol is None or avg_vol is None:
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
            
        # Seleccionar los últimos datos
        recent = df.iloc[-lookback:]
        
        # Buscar máximos/mínimos recientes
        max_idx = recent['high'].idxmax()
        min_idx = recent['low'].idxmin()
        
        # Divergencia bajista
        if pd.notna(max_idx):
            price_high = df.loc[max_idx, 'high']
            rsi_high = df.loc[max_idx, 'rsi']
            current_price = df['high'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            
            if current_price > price_high and current_rsi < rsi_high and current_rsi < 70:
                return 'bearish'
        
        # Divergencia alcista
        if pd.notna(min_idx):
            price_low = df.loc[min_idx, 'low']
            rsi_low = df.loc[min_idx, 'rsi']
            current_price = df['low'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            
            if current_price < price_low and current_rsi > rsi_low and current_rsi > 30:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detecting divergence: {str(e)}")
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

# Analizar una criptomoneda
def analyze_crypto(symbol, params, last_signals):
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
        is_breakout = any(last['close'] > r * 1.01 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.99 for s in supports) if supports else False
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG
        if trend == 'up': long_prob += 30
        if last['adx'] > params['adx_level']: long_prob += 20
        if is_breakout or divergence == 'bullish': long_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 15
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): long_prob += 10
        
        # Criterios para SHORT
        if trend == 'down': short_prob += 30
        if last['adx'] > params['adx_level']: short_prob += 20
        if is_breakdown or divergence == 'bearish': short_prob += 25
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 15
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): short_prob += 10
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Señales LONG
        long_signal = None
        if long_prob >= 65:
            # Encontrar soporte más cercano para SL
            next_supports = [s for s in supports if s < last['close']]
            sl = max(next_supports) * 0.995 if next_supports else last['close'] * (1 - params['max_risk_percent']/100)
            
            # Encontrar resistencia más cercana para TP
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.01
            
            risk = entry - sl
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'distance': round(((entry - last['close']) / last['close']) * 100, 2)
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= 65:
            # Encontrar resistencia más cercana para SL
            next_resistances = [r for r in resistances if r > last['close']]
            sl = min(next_resistances) * 1.005 if next_resistances else last['close'] * (1 + params['max_risk_percent']/100)
            
            # Encontrar soporte más cercano para TP
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.99
            
            risk = sl - entry
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'volume': volume_class,
                'adx': round(last['adx'], 2) if not pd.isna(last['adx']) else 0,
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2)
            }
        
        # Registrar señal histórica si es diferente a la anterior
        prev_signal = next((s for s in last_signals if s['symbol'] == symbol), None)
        current_signal = long_signal or short_signal
        
        if current_signal and (not prev_signal or 
           (current_signal['entry'] != prev_signal['entry'] and 
            current_signal['sl'] != prev_signal['sl'])):
            historical_signal = current_signal.copy()
            historical_signal['timestamp'] = datetime.now().isoformat()
            historical_signal['type'] = 'LONG' if long_signal else 'SHORT'
            app_state.historical_signals.append(historical_signal)
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano para actualizar datos
def background_update():
    while True:
        try:
            app_state.is_ready.clear()
            logger.info("Starting data update...")
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            
            # Mantener solo las últimas 100 señales históricas
            with app_state.lock:
                if len(app_state.historical_signals) > 100:
                    app_state.historical_signals = app_state.historical_signals[-100:]
                
                last_signals = app_state.long_signals + app_state.short_signals
            
            # Procesar en lotes para reducir memoria
            batch_size = 20
            total = len(cryptos)
            processed = 0
            
            for i in range(0, total, batch_size):
                batch = cryptos[i:i+batch_size]
                for crypto in batch:
                    try:
                        long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(
                            crypto, DEFAULTS, last_signals)
                        
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
                        processed += 1
                        logger.info(f"Progress: {processed}/{total} ({processed/total*100:.1f}%)")
                    except Exception as e:
                        logger.error(f"Error processing {crypto}: {str(e)}")
                
                # Liberar memoria entre lotes
                time.sleep(1)
            
            # Ordenar por fuerza de tendencia (ADX)
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar estado global
            with app_state.lock:
                app_state.long_signals = long_signals
                app_state.short_signals = short_signals
                app_state.scatter_data = scatter_data
                app_state.cryptos_analyzed = len(scatter_data)
                app_state.last_update = datetime.now()
            
            logger.info(f"Update completed: {len(long_signals)} LONG, {len(short_signals)} SHORT, {len(scatter_data)} cryptos analyzed")
            app_state.is_ready.set()
        except Exception as e:
            logger.error(f"Error in background update: {str(e)}")
            app_state.is_ready.set()
        
        time.sleep(UPDATE_INTERVAL)

# Iniciar hilo de actualización en segundo plano
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()
logger.info("Background update thread started")

@app.route('/')
def index():
    # Esperar hasta que los datos estén listos
    app_state.is_ready.wait()
    
    with app_state.lock:
        long_signals = app_state.long_signals
        short_signals = app_state.short_signals
        scatter_data = app_state.scatter_data
        last_update = app_state.last_update
        cryptos_analyzed = app_state.cryptos_analyzed
    
    # Estadísticas para gráficos
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=scatter_data,
                           cryptos_analyzed=cryptos_analyzed)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Data not available", 404
    
    df = calculate_indicators(df, DEFAULTS)
    if df is None or len(df) < 20:
        return "Insufficient data to generate chart", 404
    
    # Buscar señal correspondiente
    with app_state.lock:
        signals = app_state.long_signals if signal_type == 'long' else app_state.short_signals
    
    signal = next((s for s in signals if s['symbol'] == symbol), None)
    
    if not signal:
        return "Signal not found", 404
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Price', color='blue')
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', alpha=0.7)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', alpha=0.7)
        
        # Marcar entrada y SL
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='--', label='Entry')
            plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='--', label='Entry')
            plt.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss')
        
        plt.title(f'{signal["symbol"]} - Price and EMAs')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        plt.bar(df['timestamp'], df['volume'], color=np.where(df['close'] > df['open'], 'green', 'red'))
        plt.title('Volume')
        plt.grid(True)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        plt.title('Indicators')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return "Error generating chart", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/update_params', methods=['POST'])
def update_params():
    try:
        # Actualizar parámetros
        for param in DEFAULTS:
            if param in request.form:
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    DEFAULTS[param] = int(request.form[param])
                elif param in ['max_risk_percent', 'price_distance_threshold']:
                    DEFAULTS[param] = float(request.form[param])
                else:
                    DEFAULTS[param] = request.form[param]
        
        return jsonify({
            'status': 'success',
            'message': 'Parameters updated successfully',
            'params': DEFAULTS
        })
    except Exception as e:
        logger.error(f"Error updating parameters: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error updating parameters: {str(e)}"
        }), 500

@app.route('/status')
def status():
    with app_state.lock:
        return jsonify({
            'last_update': app_state.last_update.isoformat(),
            'cryptos_analyzed': app_state.cryptos_analyzed,
            'long_signals': len(app_state.long_signals),
            'short_signals': len(app_state.short_signals),
            'is_ready': app_state.is_ready.is_set()
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
