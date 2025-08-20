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
import io
import base64
import logging
import traceback
from threading import Lock
from collections import deque
import pytz

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Deshabilitar caché

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300  # 5 minutos
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 15

# Zona horaria de Nueva York (UTC-4/5)
NY_TZ = pytz.timezone('America/New_York')

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
analysis_state = {
    'long_signals': [],
    'short_signals': [],
    'scatter_data': [],
    'historical_signals': deque(maxlen=100),  # Señales históricas por timeframe
    'last_update': datetime.now(),
    'cryptos_analyzed': 0,
    'is_updating': False,
    'update_progress': 0,
    'params': DEFAULTS.copy(),
    'lock': Lock(),
    'timeframe_data': {}  # Almacenar datos por timeframe
}

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Cargadas {len(cryptos)} criptomonedas desde archivo")
            return cryptos
    except Exception as e:
        logger.error(f"Error cargando criptomonedas: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'LINK']

# Obtener datos de KuCoin con reintentos
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
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    candles.reverse()
                    
                    if len(candles) < 100:  # Necesitamos más datos para análisis histórico
                        logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                        return None
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    
                    # Convertir a tipos numéricos
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Eliminar filas con valores NaN
                    df = df.dropna()
                    
                    if len(df) < 100:
                        logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                        return None
                    
                    # Convertir timestamp a datetime con zona horaria UTC
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
                    
                    # Convertir a zona horaria de Nueva York
                    df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
                    
                    return df
                else:
                    logger.warning(f"Respuesta no válida de KuCoin para {symbol}: {data.get('msg')}")
                    return None
            else:
                logger.warning(f"Error HTTP {response.status_code} para {symbol}, reintento {attempt+1}/{MAX_RETRIES}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
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
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación optimizada de ADX
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        # Calcular +DM y -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calcular True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Suavizar valores
        atr = tr.rolling(window).mean().fillna(tr)
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        # Calcular ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = dx.rolling(window).mean().fillna(0)
        return adx, plus_di, minus_di
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))

# Calcular indicadores
def calculate_indicators(df, params):
    try:
        # EMA
        df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
        df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
        
        # ADX
        df['adx'], _, _ = calculate_adx(
            df['high'], 
            df['low'], 
            df['close'], 
            params['adx_period']
        )
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias optimizado
def find_support_resistance(df, window=50):
    try:
        if len(df) < window:
            return [], []
        
        # Identificar pivots locales
        df['min'] = df['low'].rolling(window=3, center=True).min()
        df['max'] = df['high'].rolling(window=3, center=True).max()
        
        supports = df[df['low'] == df['min']]['low'].values
        resistances = df[df['high'] == df['max']]['high'].values
        
        # Consolidar niveles cercanos
        def consolidate(levels, threshold=0.01):
            if len(levels) == 0:
                return []
                
            levels.sort()
            consolidated = []
            current = levels[0]
            
            for level in levels[1:]:
                if level <= current * (1 + threshold):
                    current = (current + level) / 2
                else:
                    consolidated.append(current)
                    current = level
                    
            consolidated.append(current)
            return consolidated
        
        return consolidate(supports), consolidate(resistances)
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0 or current_vol <= 0:
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
def detect_divergence(df, lookback=14):
    try:
        if len(df) < lookback + 1:
            return None
            
        # Seleccionar los últimos datos
        recent = df.iloc[-lookback:]
        
        # Encontrar máximos y mínimos
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        current_high = recent['high'].iloc[-1]
        current_low = recent['low'].iloc[-1]
        current_rsi = recent['rsi'].iloc[-1]
        
        # Divergencia bajista
        if high_idx != recent.index[-1]:
            high_rsi = df.loc[high_idx, 'rsi']
            if current_high > df.loc[high_idx, 'high'] and current_rsi < high_rsi:
                return 'bearish'
        
        # Divergencia alcista
        if low_idx != recent.index[-1]:
            low_rsi = df.loc[low_idx, 'rsi']
            if current_low < df.loc[low_idx, 'low'] and current_rsi > low_rsi:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Calcular distancia al nivel más cercano
def near_level(price, levels, threshold_percent=1.0):
    if not levels or price <= 0:
        return False
    
    min_distance = min(abs(price - level) for level in levels)
    threshold = price * threshold_percent / 100
    return min_distance <= threshold

# Analizar una criptomoneda
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return None, None, 0, 0, 'Muy Bajo'
        
        # Obtener datos de la vela anterior (penúltima vela)
        last = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        current = df.iloc[-1]
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        # Calcular volumen promedio
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        volume_class = classify_volume(last['volume'], avg_vol)
        
        # Detectar eventos
        divergence = detect_divergence(df, params['divergence_lookback'])
        is_breakout = any(last['close'] > r * 1.01 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.99 for s in supports) if supports else False
        
        # Determinar tendencia
        trend_up = last['ema_fast'] > last['ema_slow'] and last['adx'] > params['adx_level']
        trend_down = last['ema_fast'] < last['ema_slow'] and last['adx'] > params['adx_level']
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Factores para LONG
        if trend_up: long_prob += 35
        if near_level(last['close'], supports, params['price_distance_threshold']): long_prob += 25
        if last['rsi'] < 40: long_prob += 15
        if is_breakout or divergence == 'bullish': long_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 20
        
        # Factores para SHORT
        if trend_down: short_prob += 35
        if near_level(last['close'], resistances, params['price_distance_threshold']): short_prob += 25
        if last['rsi'] > 60: short_prob += 15
        if is_breakdown or divergence == 'bearish': short_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 20
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 100:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Generar señal LONG
        long_signal = None
        if long_prob >= 60 and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar soporte más cercano para SL
            next_supports = [s for s in supports if s < last['close']]
            sl = max(next_supports) * 0.99 if next_supports else last['close'] * (1 - params['max_risk_percent']/100)
            
            # Encontrar resistencia más cercana para TP
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
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'LONG',
                'timeframe': params['timeframe'],
                'current_price': round(current['close'], 4)  # Precio actual para referencia
            }
        
        # Generar señal SHORT
        short_signal = None
        if short_prob >= 60 and volume_class in ['Alto', 'Muy Alto']:
            # Encontrar resistencia más cercana para SL
            next_resistances = [r for r in resistances if r > last['close']]
            sl = min(next_resistances) * 1.01 if next_resistances else last['close'] * (1 + params['max_risk_percent']/100)
            
            # Encontrar soporte más cercano para TP
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
                'volume': volume_class,
                'adx': round(last['adx'], 1),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'type': 'SHORT',
                'timeframe': params['timeframe'],
                'current_price': round(current['close'], 4)  # Precio actual para referencia
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea de actualización
def update_task():
    while True:
        try:
            with analysis_state['lock']:
                analysis_state['is_updating'] = True
                analysis_state['update_progress'] = 0
                
                cryptos = load_cryptos()
                total = len(cryptos)
                long_signals = []
                short_signals = []
                scatter_data = []
                processed = 0
                
                logger.info(f"Iniciando análisis de {total} criptomonedas...")
                
                # Obtener parámetros actuales
                params = analysis_state['params']
                
                for i in range(0, total, BATCH_SIZE):
                    batch = cryptos[i:i+BATCH_SIZE]
                    
                    for crypto in batch:
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
                            
                            processed += 1
                            progress = int((processed / total) * 100)
                            analysis_state['update_progress'] = progress
                        except Exception as e:
                            logger.error(f"Error procesando {crypto}: {str(e)}")
                    
                    # Pausa entre lotes
                    time.sleep(1)
                
                # Ordenar por fuerza de tendencia
                long_signals.sort(key=lambda x: x['adx'], reverse=True)
                short_signals.sort(key=lambda x: x['adx'], reverse=True)
                
                # Almacenar datos por timeframe
                timeframe = params['timeframe']
                analysis_state['timeframe_data'][timeframe] = {
                    'long_signals': long_signals,
                    'short_signals': short_signals,
                    'scatter_data': scatter_data,
                    'last_update': datetime.now()
                }
                
                # Actualizar estado global con datos del timeframe actual
                analysis_state['long_signals'] = long_signals
                analysis_state['short_signals'] = short_signals
                analysis_state['scatter_data'] = scatter_data
                analysis_state['cryptos_analyzed'] = total
                analysis_state['last_update'] = datetime.now()
                analysis_state['is_updating'] = False
                
                # Actualizar señales históricas
                now = datetime.now()
                for signal in long_signals + short_signals:
                    signal['historical_timestamp'] = now
                    analysis_state['historical_signals'].append(signal)
                
                logger.info(f"Análisis completado: {len(long_signals)} LONG, {len(short_signals)} SHORT")
        except Exception as e:
            logger.error(f"Error crítico en actualización: {str(e)}")
            traceback.print_exc()
        
        # Esperar hasta la próxima actualización
        next_run = datetime.now() + timedelta(seconds=CACHE_TIME)
        logger.info(f"Próxima actualización a las {next_run.strftime('%H:%M:%S')}")
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
update_thread = threading.Thread(target=update_task, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

@app.route('/')
def index():
    with analysis_state['lock']:
        params = analysis_state['params']
        timeframe = params['timeframe']
        
        # Obtener datos del timeframe actual
        if timeframe in analysis_state['timeframe_data']:
            data = analysis_state['timeframe_data'][timeframe]
            long_signals = data['long_signals'][:50]
            short_signals = data['short_signals'][:50]
            scatter_data = data['scatter_data']
            last_update = data['last_update']
        else:
            long_signals = []
            short_signals = []
            scatter_data = []
            last_update = datetime.now()
        
        # Filtrar señales históricas por timeframe actual y eliminar expiradas
        current_time = datetime.now()
        timeframe_durations = {
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1)
        }
        
        duration = timeframe_durations.get(timeframe, timedelta(hours=1))
        historical_signals = []
        
        for signal in analysis_state['historical_signals']:
            if (signal['timeframe'] == timeframe and 
                current_time - signal['historical_timestamp'] <= duration):
                historical_signals.append(signal)
        
        # Ordenar por timestamp
        historical_signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Estadísticas
        avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
        avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
        
        # Preparar datos para gráfico de dispersión
        scatter_ready = []
        for item in scatter_data:
            # Asegurar que las probabilidades sean números válidos
            long_prob = max(0, min(100, item.get('long_prob', 0)))
            short_prob = max(0, min(100, item.get('short_prob', 0)))
            scatter_ready.append({
                'symbol': item['symbol'],
                'long_prob': long_prob,
                'short_prob': short_prob,
                'volume': item['volume']
            })
        
        return render_template('index.html', 
                               long_signals=long_signals, 
                               short_signals=short_signals,
                               historical_signals=historical_signals[-20:],  # Mostrar solo las últimas 20
                               last_update=last_update,
                               params=params,
                               avg_adx_long=round(avg_adx_long, 1),
                               avg_adx_short=round(avg_adx_short, 1),
                               scatter_data=scatter_ready,
                               cryptos_analyzed=analysis_state['cryptos_analyzed'],
                               is_updating=analysis_state['is_updating'],
                               update_progress=analysis_state['update_progress'])

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    try:
        params = analysis_state['params']
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            return "Datos no disponibles", 404
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 20:
            return "Datos insuficientes", 404
        
        # Buscar señal
        signals = analysis_state['long_signals'] if signal_type == 'long' else analysis_state['short_signals']
        signal = next((s for s in signals if s['symbol'] == symbol and s['timeframe'] == params['timeframe']), None)
        
        if not signal:
            return "Señal no encontrada", 404
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {params["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {params["ema_slow"]}', color='green', alpha=0.8)
        
        # Marcar niveles clave
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.7, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.7, label='TP2')
        
        plt.title(f'{signal["symbol"]} - Precio y EMAs ({params["timeframe"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
        plt.bar(df['timestamp'], df['volume'], color=colors)
        plt.title('Volumen')
        plt.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=params['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        plt.title('Indicadores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('chart.html', plot_url=plot_url, symbol=symbol, signal_type=signal_type)
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        return "Error generando gráfico", 500

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/update_params', methods=['POST'])
def update_params():
    try:
        # Actualizar parámetros
        new_params = analysis_state['params'].copy()
        
        for param in new_params:
            if param in request.form:
                value = request.form[param]
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    new_params[param] = int(value)
                elif param in ['max_risk_percent', 'price_distance_threshold']:
                    new_params[param] = float(value)
                else:
                    new_params[param] = value
        
        with analysis_state['lock']:
            analysis_state['params'] = new_params
        
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
        return jsonify({
            'last_update': analysis_state['last_update'].isoformat(),
            'is_updating': analysis_state['is_updating'],
            'progress': analysis_state['update_progress'],
            'long_signals': len(analysis_state['long_signals']),
            'short_signals': len(analysis_state['short_signals']),
            'cryptos_analyzed': analysis_state['cryptos_analyzed'],
            'params': analysis_state['params']
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
