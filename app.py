+import os
import time
import requests
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import math
import logging
import plotly
import plotly.graph_objs as go
from collections import deque

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
CRYPTOS_FILE = 'cryptos.txt'
CACHE_TIME = 300  # 5 minutos
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
    'min_signal_strength': 60
}

# Historial de señales
signal_history = deque(maxlen=100)

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        cryptos = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"Cargadas {len(cryptos)} criptomonedas")
    return cryptos

# Obtener datos de KuCoin mejorado
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
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        
        data = response.json()
        if data.get('code') == '200000' and data.get('data'):
            candles = data['data']
            if not candles:
                logger.warning(f"No hay datos para {symbol}")
                return None
                
            candles.reverse()
            
            if len(candles) < 100:
                logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            
            # Convertir a tipos numéricos
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Eliminar filas con valores NaN
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                return None
            
            # Convertir timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
            return df
            
    except Exception as e:
        logger.error(f"Error obteniendo datos para {symbol}: {str(e)}")
    
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
    
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Implementación manual de ADX
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm[plus_dm <= 0] = 0
        minus_dm[minus_dm <= 0] = 0
        
        # Calcular True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        return adx, plus_di, minus_di
    except Exception as e:
        logger.error(f"Error calculando ADX: {str(e)}")
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
        df['adx'], _, _ = calculate_adx(
            df['high'], 
            df['low'], 
            df['close'], 
            params['adx_period']
        )
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        return df
    except Exception as e:
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias mejorado
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Identificar pivots
        high_pivots = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        low_pivots = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        resistances = df[high_pivots]['high'].unique().tolist()
        supports = df[low_pivots]['low'].unique().tolist()
        
        # Consolidar niveles cercanos
        def consolidate_levels(levels, threshold=0.005):
            if not levels:
                return []
                
            levels.sort()
            consolidated = []
            current_group = [levels[0]]
            
            for level in levels[1:]:
                if level <= current_group[-1] * (1 + threshold):
                    current_group.append(level)
                else:
                    consolidated.append(sum(current_group) / len(current_group))
                    current_group = [level]
                    
            consolidated.append(sum(current_group) / len(current_group))
            return consolidated
        
        supports = consolidate_levels(supports)
        resistances = consolidate_levels(resistances)
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen mejorado
def classify_volume(current_vol, avg_vol):
    try:
        if avg_vol <= 0 or current_vol <= 0:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 2.5: return 'Muy Alto'
        if ratio > 1.8: return 'Alto'
        if ratio > 1.2: return 'Medio'
        if ratio > 0.7: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias mejorado
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
        if pd.notnull(max_idx):
            price_high = df.loc[max_idx, 'high']
            rsi_high = df.loc[max_idx, 'rsi']
            current_price = df.iloc[-1]['high']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price > price_high and current_rsi < rsi_high - 5:
                return 'bearish'
        
        # Divergencia alcista
        if pd.notnull(min_idx):
            price_low = df.loc[min_idx, 'low']
            rsi_low = df.loc[min_idx, 'rsi']
            current_price = df.iloc[-1]['low']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price < price_low and current_rsi > rsi_low + 5:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
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

# Analizar una criptomoneda mejorado
def analyze_crypto(symbol, params):
    try:
        df = get_kucoin_data(symbol, params['timeframe'])
        if df is None or len(df) < 100:
            logger.warning(f"Datos insuficientes para {symbol}")
            return None, None, 0, 0, 'Muy Bajo'
        
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            logger.warning(f"Indicadores incompletos para {symbol}")
            return None, None, 0, 0, 'Muy Bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        volume_class = classify_volume(last['volume'], avg_vol)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.01 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.99 for s in supports) if supports else False
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        trend_strength = 1.5 if last['adx'] > params['adx_level'] else 1.0
        
        # Calcular probabilidades mejoradas
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG
        if trend == 'up': 
            long_prob += 35 * trend_strength
        if last['close'] > last['open']: 
            long_prob += 15
        if is_breakout: 
            long_prob += 20
        if divergence == 'bullish':
            long_prob += 15
        if volume_class in ['Alto', 'Muy Alto']: 
            long_prob += 20
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): 
            long_prob += 15
        if last['rsi'] < 40: 
            long_prob += 10
        
        # Criterios para SHORT
        if trend == 'down': 
            short_prob += 35 * trend_strength
        if last['close'] < last['open']: 
            short_prob += 15
        if is_breakdown: 
            short_prob += 20
        if divergence == 'bearish':
            short_prob += 15
        if volume_class in ['Alto', 'Muy Alto']: 
            short_prob += 20
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): 
            short_prob += 15
        if last['rsi'] > 60: 
            short_prob += 10
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Señales LONG
        long_signal = None
        min_strength = params['min_signal_strength']
        if long_prob >= min_strength and volume_class in ['Alto', 'Muy Alto']:
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
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'timestamp': datetime.now()
            }
        
        # Señales SHORT
        short_signal = None
        if short_prob >= min_strength and volume_class in ['Alto', 'Muy Alto']:
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
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'timestamp': datetime.now()
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analizando {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano optimizada
def background_update():
    cache.set('long_signals', [])
    cache.set('short_signals', [])
    cache.set('scatter_data', [])
    cache.set('last_update', datetime.now())
    
    while True:
        start_time = time.time()
        try:
            logger.info("Iniciando actualización de datos...")
            cryptos = load_cryptos()
            long_signals = []
            short_signals = []
            scatter_data = []
            
            total = len(cryptos)
            analyzed = 0
            
            for i, crypto in enumerate(cryptos):
                try:
                    long_signal, short_signal, long_prob, short_prob, volume_class = analyze_crypto(crypto, DEFAULTS)
                    
                    if long_signal:
                        long_signals.append(long_signal)
                        # Guardar en historial
                        signal_history.append({
                            'symbol': crypto,
                            'type': 'LONG',
                            'timestamp': datetime.now(),
                            'entry': long_signal['entry'],
                            'sl': long_signal['sl'],
                            'tp1': long_signal['tp1'],
                            'tp2': long_signal['tp2'],
                            'volume': volume_class
                        })
                    
                    if short_signal:
                        short_signals.append(short_signal)
                        # Guardar en historial
                        signal_history.append({
                            'symbol': crypto,
                            'type': 'SHORT',
                            'timestamp': datetime.now(),
                            'entry': short_signal['entry'],
                            'sl': short_signal['sl'],
                            'tp1': short_signal['tp1'],
                            'tp2': short_signal['tp2'],
                            'volume': volume_class
                        })
                    
                    scatter_data.append({
                        'symbol': crypto,
                        'long_prob': long_prob,
                        'short_prob': short_prob,
                        'volume': volume_class
                    })
                    
                    analyzed += 1
                    
                    # Log intermedio
                    if (i+1) % 10 == 0:
                        logger.info(f"Progreso: {i+1}/{total} cryptos analizadas")
                    
                except Exception as e:
                    logger.error(f"Error procesando {crypto}: {str(e)}")
                
                time.sleep(0.2)  # Evitar saturar la API
            
            # Ordenar y limitar señales
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar caché
            cache.set('long_signals', long_signals[:100])
            cache.set('short_signals', short_signals[:100])
            cache.set('scatter_data', scatter_data)
            cache.set('last_update', datetime.now())
            cache.set('signal_history', list(signal_history))
            
            elapsed = time.time() - start_time
            logger.info(f"Actualización completada en {elapsed:.2f}s")
            logger.info(f"Señales LONG: {len(long_signals)}")
            logger.info(f"Señales SHORT: {len(short_signals)}")
            logger.info(f"Cryptos analizadas: {analyzed}/{total}")
            
        except Exception as e:
            logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

@app.route('/')
def index():
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    signal_history = cache.get('signal_history') or []
    
    # Estadísticas
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    # Información de criptomonedas analizadas
    cryptos_analyzed = len(scatter_data)
    
    # Filtrar señales históricas recientes
    recent_history = [s for s in signal_history if (datetime.now() - s['timestamp']).total_seconds() < 3600]
    
    return render_template('index.html', 
                           long_signals=long_signals, 
                           short_signals=short_signals,
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=scatter_data,
                           cryptos_analyzed=cryptos_analyzed,
                           signal_history=recent_history)

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Datos no disponibles", 404
    
    df = calculate_indicators(df, DEFAULTS)
    if df is None or len(df) < 50:
        return "Datos insuficientes para generar gráfico", 404
    
    # Buscar señal correspondiente
    signals = cache.get('long_signals') if signal_type == 'long' else cache.get('short_signals')
    if signals is None:
        return "Señales no disponibles", 404
    
    signal = next((s for s in signals if s['symbol'] == symbol), None)
    
    if not signal:
        return "Señal no encontrada", 404
    
    try:
        plt.figure(figsize=(14, 10))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', alpha=0.8)
        
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
        
        plt.title(f'{signal["symbol"]} - Precio y EMAs', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        plt.bar(df['timestamp'], df['volume'], color=np.where(df['close'] > df['open'], 'green', 'red'), alpha=0.7)
        plt.title('Volumen', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown', linewidth=1.5)
        plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        plt.title('Indicadores', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        
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
        for param in DEFAULTS:
            if param in request.form:
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    DEFAULTS[param] = int(request.form[param])
                elif param in ['max_risk_percent', 'price_distance_threshold', 'min_signal_strength']:
                    DEFAULTS[param] = float(request.form[param])
                else:
                    DEFAULTS[param] = request.form[param]
        
        # Forzar actualización
        cache.set('last_update', datetime.now() - timedelta(seconds=CACHE_TIME))
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados correctamente',
            'params': DEFAULTS
        })
    except Exception as e:
        logger.error(f"Error actualizando parámetros: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error actualizando parámetros: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
