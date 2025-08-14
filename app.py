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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import math
import logging
import plotly
import plotly.graph_objs as go
from threading import Lock
import concurrent.futures

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
    'divergence_lookback': 14,
    'max_risk_percent': 1.5,
    'price_distance_threshold': 1.0,
    'min_probability': 60,  # Nuevo parámetro
    'volume_filter': 'medium'  # Nuevo parámetro
}

# Leer lista de criptomonedas
def load_cryptos():
    with open(CRYPTOS_FILE, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Obtener datos de KuCoin (optimizado)
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
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                candles = data['data']
                # KuCoin devuelve las velas en orden descendente, invertimos para tener ascendente
                candles.reverse()
                
                # Validar que hay suficientes velas
                if len(candles) < 100:
                    logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                    return None
                
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                
                # Convertir a tipos numéricos
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Eliminar filas con valores NaN
                df = df.dropna()
                
                # Validar que aún tenemos suficientes datos
                if len(df) < 50:
                    logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                    return None
                
                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

# Implementación manual de EMA (optimizada)
def calculate_ema(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.ewm(span=window, adjust=False).mean()

# Implementación manual de RSI (optimizada)
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

# Implementación manual de ADX (optimizada)
def calculate_adx(high, low, close, window):
    if len(close) < window * 2:
        return pd.Series([0] * len(close)), pd.Series([0] * len(close)), pd.Series([0] * len(close))
    
    try:
        # Calcular +DM y -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm[plus_dm <= 0] = 0
        plus_dm[(plus_dm > 0) & (plus_dm < minus_dm)] = 0
        
        minus_dm[minus_dm <= 0] = 0
        minus_dm[(minus_dm > 0) & (minus_dm < plus_dm)] = 0
        
        # Calcular True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window).mean()
        
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
        adx = dx.rolling(window).mean()
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
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
        logger.error(f"Error calculando indicadores: {str(e)}")
        return None

# Detectar soportes y resistencias (mejorado)
def find_support_resistance(df, window):
    try:
        if len(df) < window:
            return [], []
        
        # Identificar pivots
        df['high_pivot'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['low_pivot'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        resistances = df[df['high_pivot']]['high'].drop_duplicates().nlargest(5).tolist()
        supports = df[df['low_pivot']]['low'].drop_duplicates().nsmallest(5).tolist()
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen (mejorado)
def classify_volume(current_vol, volume_data):
    try:
        if volume_data['mean'] == 0 or current_vol is None:
            return 'Muy Bajo'
        
        ratio = current_vol / volume_data['mean']
        std_dev = volume_data['std']
        
        if ratio > volume_data['mean'] + 2 * std_dev: return 'Muy Alto'
        if ratio > volume_data['mean'] + std_dev: return 'Alto'
        if ratio > volume_data['mean']: return 'Medio'
        if ratio > volume_data['mean'] - std_dev: return 'Bajo'
        return 'Muy Bajo'
    except:
        return 'Muy Bajo'

# Detectar divergencias (mejorado)
def detect_divergence(df, lookback):
    try:
        if len(df) < lookback + 1:
            return None
            
        # Buscar máximos/mínimos en el lookback
        recent = df.iloc[-lookback:]
        
        # Divergencia bajista
        max_idx = recent['high'].idxmax()
        if pd.notna(max_idx):
            price_high = df.loc[max_idx, 'high']
            rsi_high = df.loc[max_idx, 'rsi']
            current_price = df.iloc[-1]['close']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price > price_high and current_rsi < rsi_high and current_rsi < 70:
                return 'bearish'
        
        # Divergencia alcista
        min_idx = recent['low'].idxmin()
        if pd.notna(min_idx):
            price_low = df.loc[min_idx, 'low']
            rsi_low = df.loc[min_idx, 'rsi']
            current_price = df.iloc[-1]['close']
            current_rsi = df.iloc[-1]['rsi']
            
            if current_price < price_low and current_rsi > rsi_low and current_rsi > 30:
                return 'bullish'
        
        return None
    except Exception as e:
        logger.error(f"Error detectando divergencia: {str(e)}")
        return None

# Calcular distancia al nivel más cercano (optimizado)
def calculate_distance_to_level(price, levels, threshold_percent):
    try:
        if not levels or price is None:
            return False, 100.0
        
        min_distance = min(abs(price - level) for level in levels)
        threshold = price * threshold_percent / 100
        distance_percent = (min_distance / price) * 100
        return min_distance <= threshold, distance_percent
    except:
        return False, 100.0

# Analizar una criptomoneda y calcular probabilidades (completamente revisado)
def analyze_crypto(symbol, params):
    df = get_kucoin_data(symbol, params['timeframe'])
    if df is None or len(df) < 100:
        return None, None, 0, 0, 'Muy Bajo'
    
    try:
        df = calculate_indicators(df, params)
        if df is None or len(df) < 50:
            return None, None, 0, 0, 'Muy Bajo'
        
        supports, resistances = find_support_resistance(df, params['sr_window'])
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Estadísticas de volumen
        volume_data = {
            'mean': df['volume'].mean(),
            'std': df['volume'].std(),
            'current': last['volume']
        }
        volume_class = classify_volume(last['volume'], volume_data)
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres
        is_breakout = any(last['close'] > r * 1.01 for r in resistances) if resistances else False
        is_breakdown = any(last['close'] < s * 0.99 for s in supports) if supports else False
        
        # Determinar tendencia
        trend_strength = 0
        if last['ema_fast'] > last['ema_slow']:
            trend = 'up'
            trend_strength = ((last['ema_fast'] - last['ema_slow']) / last['ema_slow']) * 100
        else:
            trend = 'down'
            trend_strength = ((last['ema_slow'] - last['ema_fast']) / last['ema_fast']) * 100
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # --- Criterios para LONG ---
        if trend == 'up': 
            long_prob += 30 + min(trend_strength, 20)  # Máximo 50 por tendencia
        
        if last['adx'] > params['adx_level']: 
            adx_strength = min((last['adx'] - params['adx_level']) / 20 * 30, 30)
            long_prob += adx_strength
        
        if divergence == 'bullish': 
            long_prob += 25
        
        if is_breakout: 
            long_prob += 20
        
        near_support, support_dist = calculate_distance_to_level(
            last['close'], supports, params['price_distance_threshold']
        )
        if near_support:
            long_prob += 20 - min(support_dist, 15)
        
        # Volumen - impacto variable
        if volume_class == 'Muy Alto': long_prob += 20
        elif volume_class == 'Alto': long_prob += 15
        elif volume_class == 'Medio': long_prob += 10
        
        # RSI
        if last['rsi'] < 40: long_prob += 10
        elif last['rsi'] < 30: long_prob += 15
        
        # --- Criterios para SHORT ---
        if trend == 'down': 
            short_prob += 30 + min(trend_strength, 20)  # Máximo 50 por tendencia
        
        if last['adx'] > params['adx_level']: 
            adx_strength = min((last['adx'] - params['adx_level']) / 20 * 30, 30)
            short_prob += adx_strength
        
        if divergence == 'bearish': 
            short_prob += 25
        
        if is_breakdown: 
            short_prob += 20
        
        near_resistance, resistance_dist = calculate_distance_to_level(
            last['close'], resistances, params['price_distance_threshold']
        )
        if near_resistance:
            short_prob += 20 - min(resistance_dist, 15)
        
        # Volumen - impacto variable
        if volume_class == 'Muy Alto': short_prob += 20
        elif volume_class == 'Alto': short_prob += 15
        elif volume_class == 'Medio': short_prob += 10
        
        # RSI
        if last['rsi'] > 60: short_prob += 10
        elif last['rsi'] > 70: short_prob += 15
        
        # Limitar probabilidades máximas
        long_prob = min(long_prob, 100)
        short_prob = min(short_prob, 100)
        
        # Normalizar si suma más de 100
        total_prob = long_prob + short_prob
        if total_prob > 100:
            long_prob = (long_prob / total_prob) * 100
            short_prob = (short_prob / total_prob) * 100
        
        # Asegurar mínimo de 5% para mantener activos en el gráfico
        long_prob = max(long_prob, 5)
        short_prob = max(short_prob, 5)
        
        # Generar señales si superan el umbral mínimo
        min_prob = params['min_probability']
        long_signal = None
        short_signal = None
        
        if long_prob >= min_prob:
            # Encontrar resistencia más cercana para entrada
            next_resistances = [r for r in resistances if r > last['close']]
            entry = min(next_resistances) * 1.005 if next_resistances else last['close'] * 1.015
            
            # Encontrar soporte más cercano para SL
            next_supports = [s for s in supports if s < entry]
            sl = max(next_supports) * 0.995 if next_supports else entry * 0.98
            risk = entry - sl
            
            # Si el riesgo es demasiado grande, ajustar a % máximo
            risk_percent = (risk / entry) * 100
            if risk_percent > params['max_risk_percent']:
                sl = entry * (1 - params['max_risk_percent']/100)
                risk = entry - sl
            
            long_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry + risk, 4),
                'tp2': round(entry + risk * 2, 4),
                'volume': volume_class,
                'divergence': divergence == 'bullish',
                'adx': round(last['adx'], 2),
                'rsi': round(last['rsi'], 2),
                'distance': round(((entry - last['close']) / last['close']) * 100, 2),
                'risk_reward': round((risk * 2) / risk, 1),
                'probability': round(long_prob, 1)
            }
        
        if short_prob >= min_prob:
            # Encontrar soporte más cercano para entrada
            next_supports = [s for s in supports if s < last['close']]
            entry = max(next_supports) * 0.995 if next_supports else last['close'] * 0.985
            
            # Encontrar resistencia más cercana para SL
            next_resistances = [r for r in resistances if r > entry]
            sl = min(next_resistances) * 1.005 if next_resistances else entry * 1.02
            risk = sl - entry
            
            # Si el riesgo es demasiado grande, ajustar a % máximo
            risk_percent = (risk / entry) * 100
            if risk_percent > params['max_risk_percent']:
                sl = entry * (1 + params['max_risk_percent']/100)
                risk = sl - entry
            
            short_signal = {
                'symbol': symbol,
                'price': round(last['close'], 4),
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(entry - risk, 4),
                'tp2': round(entry - risk * 2, 4),
                'volume': volume_class,
                'divergence': divergence == 'bearish',
                'adx': round(last['adx'], 2),
                'rsi': round(last['rsi'], 2),
                'distance': round(((last['close'] - entry) / last['close']) * 100, 2),
                'risk_reward': round((risk * 2) / risk, 1),
                'probability': round(short_prob, 1)
            }
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Analizar cripto en paralelo
def analyze_crypto_batch(cryptos, params):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_crypto = {
            executor.submit(analyze_crypto, crypto, params): crypto 
            for crypto in cryptos
        }
        
        for future in concurrent.futures.as_completed(future_to_crypto):
            crypto = future_to_crypto[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error processing {crypto}: {str(e)}")
                results.append((None, None, 0, 0, 'Muy Bajo'))
    return results

# Tarea en segundo plano para actualizar datos (completamente revisado)
def background_update():
    while True:
        try:
            start_time = time.time()
            logger.info("Iniciando actualización de datos...")
            
            cryptos = load_cryptos()
            if not cryptos:
                logger.error("No se encontraron criptomonedas para analizar")
                time.sleep(CACHE_TIME)
                continue
            
            # Usar parámetros actuales
            current_params = DEFAULTS.copy()
            
            # Analizar en paralelo
            results = analyze_crypto_batch(cryptos, current_params)
            
            long_signals = []
            short_signals = []
            scatter_data = []
            
            for result in results:
                long_signal, short_signal, long_prob, short_prob, volume_class = result
                
                if long_signal:
                    long_signals.append(long_signal)
                if short_signal:
                    short_signals.append(short_signal)
                
                scatter_data.append({
                    'symbol': long_signal['symbol'] if long_signal else (
                        short_signal['symbol'] if short_signal else 'UNKNOWN'),
                    'long_prob': long_prob,
                    'short_prob': short_prob,
                    'volume': volume_class
                })
            
            # Ordenar por probabilidad
            long_signals.sort(key=lambda x: x['probability'], reverse=True)
            short_signals.sort(key=lambda x: x['probability'], reverse=True)
            
            # Actualizar caché
            with app.app_context():
                cache.set('long_signals', long_signals)
                cache.set('short_signals', short_signals)
                cache.set('scatter_data', scatter_data)
                cache.set('last_update', datetime.now())
                cache.set('params', current_params)
            
            elapsed = time.time() - start_time
            logger.info(f"Actualización completada en {elapsed:.2f}s: "
                       f"{len(long_signals)} LONG, {len(short_signals)} SHORT, "
                       f"{len(scatter_data)} puntos")
        except Exception as e:
            logger.error(f"Error en actualización de fondo: {str(e)}")
        
        time.sleep(CACHE_TIME)

# Iniciar hilo de actualización en segundo plano
update_thread = None
if not update_thread or not update_thread.is_alive():
    try:
        update_thread = threading.Thread(target=background_update, daemon=True)
        update_thread.start()
        logger.info("Hilo de actualización iniciado")
    except Exception as e:
        logger.error(f"No se pudo iniciar el hilo de actualización: {str(e)}")

@app.route('/')
def index():
    long_signals = cache.get('long_signals') or []
    short_signals = cache.get('short_signals') or []
    scatter_data = cache.get('scatter_data') or []
    last_update = cache.get('last_update') or datetime.now()
    current_params = cache.get('params') or DEFAULTS
    
    # Estadísticas para gráficos
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    # Información de criptomonedas analizadas
    cryptos_analyzed = len(scatter_data)
    
    # Calcular probabilidades promedio
    avg_long_prob = np.mean([s['probability'] for s in long_signals]) if long_signals else 0
    avg_short_prob = np.mean([s['probability'] for s in short_signals]) if short_signals else 0
    
    # Filtrar scatter_data para mostrar todas las cryptos
    filtered_scatter = scatter_data
    
    return render_template('index.html', 
                           long_signals=long_signals[:100], 
                           short_signals=short_signals[:100],
                           last_update=last_update,
                           params=current_params,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=filtered_scatter,
                           cryptos_analyzed=cryptos_analyzed,
                           avg_long_prob=round(avg_long_prob, 1),
                           avg_short_prob=round(avg_short_prob, 1))

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
    signal = None
    if signals:
        for s in signals:
            if s['symbol'] == symbol:
                signal = s
                break
    
    if not signal:
        return "Señal no encontrada", 404
    
    try:
        plt.figure(figsize=(14, 10))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='black', linewidth=1.5)
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='blue', alpha=0.8)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='red', alpha=0.8)
        
        # Marcar niveles importantes
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='-', linewidth=2, label='Entrada')
            plt.axhline(y=signal['sl'], color='red', linestyle='-', linewidth=2, label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle='--', label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle='--', label='TP2')
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='-', linewidth=2, label='Entrada')
            plt.axhline(y=signal['sl'], color='green', linestyle='-', linewidth=2, label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle='--', label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle='--', label='TP2')
        
        plt.title(f'{signal["symbol"]} - Precio y EMAs', fontsize=14)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        colors = np.where(df['close'] > df['open'], 'green', 'red')
        plt.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7)
        plt.title('Volumen', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=1.5)
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        plt.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='teal', linewidth=1.5)
        plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.7)
        
        plt.title('Indicadores', fontsize=14)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
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
        current_params = cache.get('params') or DEFAULTS.copy()
        
        for param in current_params:
            if param in request.form:
                if param in ['ema_fast', 'ema_slow', 'adx_period', 'adx_level', 'rsi_period', 'sr_window', 'divergence_lookback']:
                    current_params[param] = int(request.form[param])
                elif param in ['max_risk_percent', 'price_distance_threshold', 'min_probability']:
                    current_params[param] = float(request.form[param])
                else:
                    current_params[param] = request.form[param]
        
        # Actualizar caché
        cache.set('params', current_params)
        
        # Forzar actualización
        cache.set('last_update', datetime.now() - timedelta(seconds=CACHE_TIME))
        
        return jsonify({
            'status': 'success',
            'message': 'Parámetros actualizados correctamente',
            'params': current_params
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
