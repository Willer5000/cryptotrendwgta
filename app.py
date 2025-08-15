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
from threading import Lock, RLock
from collections import deque

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
    'price_distance_threshold': 1.0,
    'min_volume_ratio': 1.2
}

# Almacenamiento global con bloqueo
class DataStore:
    def __init__(self):
        self.lock = RLock()
        self.long_signals = []
        self.short_signals = []
        self.scatter_data = []
        self.historical_signals = deque(maxlen=100)
        self.last_update = datetime.now()
        self.cryptos_analyzed = 0
        self.is_updating = False
        self.update_progress = 0

    def update_data(self, long_signals, short_signals, scatter_data):
        with self.lock:
            self.long_signals = long_signals
            self.short_signals = short_signals
            self.scatter_data = scatter_data
            self.cryptos_analyzed = len(scatter_data)
            self.last_update = datetime.now()
            
            # Registrar señales actuales en histórico
            for signal in long_signals:
                self.historical_signals.append({
                    'type': 'LONG',
                    'symbol': signal['symbol'],
                    'timestamp': datetime.now(),
                    'entry': signal['entry'],
                    'sl': signal['sl'],
                    'tp1': signal['tp1'],
                    'tp2': signal['tp2'],
                    'volume': signal['volume'],
                    'price': signal['price']
                })
                
            for signal in short_signals:
                self.historical_signals.append({
                    'type': 'SHORT',
                    'symbol': signal['symbol'],
                    'timestamp': datetime.now(),
                    'entry': signal['entry'],
                    'sl': signal['sl'],
                    'tp1': signal['tp1'],
                    'tp2': signal['tp2'],
                    'volume': signal['volume'],
                    'price': signal['price']
                })

    def get_data(self):
        with self.lock:
            return {
                'long_signals': self.long_signals,
                'short_signals': self.short_signals,
                'scatter_data': self.scatter_data,
                'historical_signals': list(self.historical_signals),
                'last_update': self.last_update,
                'cryptos_analyzed': self.cryptos_analyzed,
                'is_updating': self.is_updating,
                'update_progress': self.update_progress
            }

# Instancia global de almacenamiento de datos
data_store = DataStore()

# Leer lista de criptomonedas
def load_cryptos():
    try:
        with open(CRYPTOS_FILE, 'r') as f:
            cryptos = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Cargadas {len(cryptos)} criptomonedas desde archivo")
            return cryptos
    except Exception as e:
        logger.error(f"Error al cargar criptomonedas: {str(e)}")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'DOGE', 'TRX']

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
        response.raise_for_status()
        
        data = response.json()
        if data.get('code') == '200000' and data.get('data'):
            candles = data['data']
            candles.reverse()
            
            if len(candles) < 100:
                logger.warning(f"Datos insuficientes para {symbol}: {len(candles)} velas")
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Datos insuficientes después de limpieza para {symbol}: {len(df)} velas")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
            return df
        else:
            logger.warning(f"Respuesta inesperada de KuCoin para {symbol}: {data.get('msg', 'Sin mensaje')}")
            return None
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
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
        # Calcular +DM y -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[(plus_dm <= 0) | (plus_dm <= minus_dm)] = 0
        minus_dm[(minus_dm <= 0) | (minus_dm <= plus_dm)] = 0
        
        # Calcular True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Suavizar valores
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
        
        # Calcular ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
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
        
        # Identificar pivots altos y bajos
        df['high_roll'] = df['high'].rolling(window=window, min_periods=1).max()
        df['low_roll'] = df['low'].rolling(window=window, min_periods=1).min()
        
        resistances = df[df['high'] >= df['high_roll']]['high'].unique().tolist()
        supports = df[df['low'] <= df['low_roll']]['low'].unique().tolist()
        
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
                    consolidated.append(np.mean(current_group))
                    current_group = [level]
                    
            consolidated.append(np.mean(current_group))
            return consolidated
        
        supports = consolidate_levels(supports)
        resistances = consolidate_levels(resistances)
        
        return supports, resistances
    except Exception as e:
        logger.error(f"Error buscando S/R: {str(e)}")
        return [], []

# Clasificar volumen
def classify_volume(current_vol, avg_vol, min_ratio=1.2):
    try:
        if avg_vol == 0 or current_vol is None or avg_vol is None:
            return 'Muy Bajo'
        
        ratio = current_vol / avg_vol
        if ratio > 3.0: return 'Muy Alto'
        if ratio > 2.0: return 'Alto'
        if ratio > min_ratio: return 'Medio'
        if ratio > 0.8: return 'Bajo'
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
        if pd.notna(max_idx):
            price_high = df.loc[max_idx, 'high']
            rsi_high = df.loc[max_idx, 'rsi']
            current_price = df['high'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            
            if current_price > price_high and current_rsi < rsi_high and current_rsi > 70:
                return 'bearish'
        
        # Divergencia alcista
        if pd.notna(min_idx):
            price_low = df.loc[min_idx, 'low']
            rsi_low = df.loc[min_idx, 'rsi']
            current_price = df['low'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            
            if current_price < price_low and current_rsi > rsi_low and current_rsi < 30:
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
        volume_class = classify_volume(last['volume'], avg_vol, params['min_volume_ratio'])
        divergence = detect_divergence(df, params['divergence_lookback'])
        
        # Detectar quiebres con confirmación de volumen
        is_breakout = False
        is_breakdown = False
        
        if resistances:
            breakout_candidates = [r for r in resistances if last['close'] > r * 1.005]
            if breakout_candidates:
                # Confirmar con volumen
                recent_vol_avg = df['volume'].tail(5).mean()
                if last['volume'] > recent_vol_avg * 1.5:
                    is_breakout = True
        
        if supports:
            breakdown_candidates = [s for s in supports if last['close'] < s * 0.995]
            if breakdown_candidates:
                # Confirmar con volumen
                recent_vol_avg = df['volume'].tail(5).mean()
                if last['volume'] > recent_vol_avg * 1.5:
                    is_breakdown = True
        
        # Determinar tendencia
        trend = 'up' if last['ema_fast'] > last['ema_slow'] else 'down'
        trend_strength = last['adx'] / 100 if not pd.isna(last['adx']) else 0.5
        
        # Calcular probabilidades
        long_prob = 0
        short_prob = 0
        
        # Criterios para LONG (Estrategia basada en S/R, tendencia y divergencias)
        if trend == 'up': long_prob += 30 * trend_strength
        if is_breakout: long_prob += 25
        if divergence == 'bullish': long_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: long_prob += 15
        if calculate_distance_to_level(last['close'], supports, params['price_distance_threshold']): long_prob += 10
        if last['rsi'] < 40: long_prob += 10
        
        # Criterios para SHORT
        if trend == 'down': short_prob += 30 * trend_strength
        if is_breakdown: short_prob += 25
        if divergence == 'bearish': short_prob += 20
        if volume_class in ['Alto', 'Muy Alto']: short_prob += 15
        if calculate_distance_to_level(last['close'], resistances, params['price_distance_threshold']): short_prob += 10
        if last['rsi'] > 60: short_prob += 10
        
        # Normalizar probabilidades
        total = long_prob + short_prob
        if total > 0:
            long_prob = (long_prob / total) * 100
            short_prob = (short_prob / total) * 100
        
        # Señales LONG
        long_signal = None
        if long_prob >= 65 and volume_class in ['Alto', 'Muy Alto'] and trend == 'up':
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
        if short_prob >= 65 and volume_class in ['Alto', 'Muy Alto'] and trend == 'down':
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
        
        return long_signal, short_signal, long_prob, short_prob, volume_class
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None, None, 0, 0, 'Muy Bajo'

# Tarea en segundo plano optimizada
def background_update():
    data_store.is_updating = True
    
    while True:
        start_time = time.time()
        try:
            cryptos = load_cryptos()
            total_cryptos = len(cryptos)
            logger.info(f"Iniciando actualización para {total_cryptos} criptomonedas...")
            
            long_signals = []
            short_signals = []
            scatter_data = []
            
            # Procesar en lotes
            batch_size = 10
            for i in range(0, total_cryptos, batch_size):
                batch = cryptos[i:i+batch_size]
                for j, crypto in enumerate(batch):
                    try:
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
                    except Exception as e:
                        logger.error(f"Error procesando {crypto}: {str(e)}")
                    
                    # Actualizar progreso
                    progress = ((i + j + 1) / total_cryptos) * 100
                    data_store.update_progress = round(progress, 1)
                
                # Pausa entre lotes
                time.sleep(1)
            
            # Ordenar por fuerza de tendencia (ADX)
            long_signals.sort(key=lambda x: x['adx'], reverse=True)
            short_signals.sort(key=lambda x: x['adx'], reverse=True)
            
            # Actualizar almacenamiento de datos
            data_store.update_data(long_signals, short_signals, scatter_data)
            
            elapsed = time.time() - start_time
            logger.info(f"Actualización completada en {elapsed:.2f}s: {len(long_signals)} LONG, {len(short_signals)} SHORT")
        except Exception as e:
            logger.error(f"Error en actualización de fondo: {str(e)}")
        
        data_store.is_updating = False
        time.sleep(UPDATE_INTERVAL)

# Iniciar hilo de actualización
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()
logger.info("Hilo de actualización iniciado")

@app.route('/')
def index():
    data = data_store.get_data()
    long_signals = data['long_signals']
    short_signals = data['short_signals']
    scatter_data = data['scatter_data']
    last_update = data['last_update']
    historical_signals = data['historical_signals']
    
    # Estadísticas
    signal_count = len(long_signals) + len(short_signals)
    avg_adx_long = np.mean([s['adx'] for s in long_signals]) if long_signals else 0
    avg_adx_short = np.mean([s['adx'] for s in short_signals]) if short_signals else 0
    
    # Información de criptomonedas analizadas
    cryptos_analyzed = data['cryptos_analyzed']
    
    # Filtrar solo las últimas 50 señales históricas
    recent_historical = historical_signals[-50:] if historical_signals else []
    
    return render_template('index.html', 
                           long_signals=long_signals[:50], 
                           short_signals=short_signals[:50],
                           historical_signals=recent_historical,
                           last_update=last_update,
                           params=DEFAULTS,
                           signal_count=signal_count,
                           avg_adx_long=round(avg_adx_long, 2),
                           avg_adx_short=round(avg_adx_short, 2),
                           scatter_data=scatter_data,
                           cryptos_analyzed=cryptos_analyzed,
                           is_updating=data['is_updating'],
                           update_progress=data['update_progress'])

@app.route('/chart/<symbol>/<signal_type>')
def get_chart(symbol, signal_type):
    data = data_store.get_data()
    signals = data['long_signals'] if signal_type == 'long' else data['short_signals']
    signal = next((s for s in signals if s['symbol'] == symbol), None)
    
    if not signal:
        return "Señal no encontrada", 404
    
    df = get_kucoin_data(symbol, DEFAULTS['timeframe'])
    if df is None:
        return "Datos no disponibles", 404
    
    df = calculate_indicators(df, DEFAULTS)
    if df is None or len(df) < 20:
        return "Datos insuficientes para generar gráfico", 404
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Gráfico de precio
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Precio', color='blue')
        plt.plot(df['timestamp'], df['ema_fast'], label=f'EMA {DEFAULTS["ema_fast"]}', color='orange', alpha=0.7)
        plt.plot(df['timestamp'], df['ema_slow'], label=f'EMA {DEFAULTS["ema_slow"]}', color='green', alpha=0.7)
        
        # Marcar niveles clave
        if signal_type == 'long':
            plt.axhline(y=signal['entry'], color='green', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='red', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.5, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.5, label='TP2')
        else:
            plt.axhline(y=signal['entry'], color='red', linestyle='--', label='Entrada')
            plt.axhline(y=signal['sl'], color='green', linestyle='--', label='Stop Loss')
            plt.axhline(y=signal['tp1'], color='blue', linestyle=':', alpha=0.5, label='TP1')
            plt.axhline(y=signal['tp2'], color='purple', linestyle=':', alpha=0.5, label='TP2')
        
        plt.title(f'{signal["symbol"]} - Precio y EMAs')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de volumen
        plt.subplot(3, 1, 2)
        plt.bar(df['timestamp'], df['volume'], color=np.where(df['close'] > df['open'], 'green', 'red'))
        plt.title('Volumen')
        plt.grid(True)
        
        # Gráfico de indicadores
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        plt.plot(df['timestamp'], df['adx'], label='ADX', color='brown')
        plt.axhline(y=DEFAULTS['adx_level'], color='blue', linestyle='--', alpha=0.5)
        
        plt.title('Indicadores')
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
                elif param in ['max_risk_percent', 'price_distance_threshold', 'min_volume_ratio']:
                    DEFAULTS[param] = float(request.form[param])
                else:
                    DEFAULTS[param] = request.form[param]
        
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

@app.route('/status')
def status():
    data = data_store.get_data()
    return jsonify({
        'last_update': data['last_update'].strftime('%Y-%m-%d %H:%M:%S'),
        'is_updating': data['is_updating'],
        'update_progress': data['update_progress'],
        'cryptos_analyzed': data['cryptos_analyzed'],
        'long_signals': len(data['long_signals']),
        'short_signals': len(data['short_signals'])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
