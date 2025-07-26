import pandas as pd
import numpy as np
import sqlite3
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import traceback

# Parámetros por defecto (ajustables)
DEFAULT_PARAMS = {
    'rsi_period': 14,
    'adx_period': 14,
    'adx_level': 25,
    'ema_fast': 9,
    'ema_slow': 20,
    'sr_lookback': 100,
    'divergence_lookback': 14,
    'volume_levels': [0.5, 1.0, 2.0, 3.0]  # Muy Bajo, Bajo, Medio, Alto, Muy Alto
}

def calculate_support_resistance(df, lookback=100):
    try:
        # Identificar pivots para soportes/resistencias
        df['pivot_low'] = df['low'].rolling(lookback, center=True).min()
        df['pivot_high'] = df['high'].rolling(lookback, center=True).max()
        
        # Últimos soportes/resistencias significativos
        supports = df[df['low'] == df['pivot_low']]['low'].tail(3).values
        resistances = df[df['high'] == df['pivot_high']]['high'].tail(3).values
        
        return {
            'supports': supports.tolist(),
            'resistances': resistances.tolist(),
            'current_support': supports[-1] if len(supports) > 0 else None,
            'current_resistance': resistances[-1] if len(resistances) > 0 else None
        }
    except:
        return {'supports': [], 'resistances': []}

def detect_divergence(df, lookback=14):
    try:
        # Calcular RSI
        rsi = RSIIndicator(pd.to_numeric(df['close']), window=DEFAULT_PARAMS['rsi_period']).rsi()
        
        # Buscar divergencias en el lookback
        current_low = df['low'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Bullish divergence (precio hace menor bajo, RSI hace bajo más alto)
        if current_low < df['low'].iloc[-lookback] and current_rsi > rsi.iloc[-lookback]:
            return 'bullish'
        
        # Bearish divergence (precio hace mayor alto, RSI hace alto más bajo)
        if current_high > df['high'].iloc[-lookback] and current_rsi < rsi.iloc[-lookback]:
            return 'bearish'
        
        return None
    except:
        return None

def classify_volume(current_volume, avg_volume):
    ratios = [current_volume / avg_volume for level in DEFAULT_PARAMS['volume_levels']]
    
    if ratios[0] < DEFAULT_PARAMS['volume_levels'][0]:
        return 'Muy Bajo'
    elif ratios[0] < DEFAULT_PARAMS['volume_levels'][1]:
        return 'Bajo'
    elif ratios[0] < DEFAULT_PARAMS['volume_levels'][2]:
        return 'Medio'
    elif ratios[0] < DEFAULT_PARAMS['volume_levels'][3]:
        return 'Alto'
    else:
        return 'Muy Alto'

def analyze_symbol(symbol, timeframe):
    conn = sqlite3.connect('ohlcv.db')
    query = f"SELECT * FROM ohlcv_data WHERE symbol = '{symbol}' AND timeframe = '{timeframe}' ORDER BY timestamp DESC LIMIT 200"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty or len(df) < 50:
        return None
    
    # Convertir tipos de datos correctamente
    numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Calcular indicadores
    df['ema_fast'] = EMAIndicator(df['close'], window=DEFAULT_PARAMS['ema_fast']).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=DEFAULT_PARAMS['ema_slow']).ema_indicator()
    
    adx_ind = ADXIndicator(df['high'], df['low'], df['close'], window=DEFAULT_PARAMS['adx_period'])
    df['adx'] = adx_ind.adx()
    
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    # Soporte/resistencia
    sr_data = calculate_support_resistance(df, DEFAULT_PARAMS['sr_lookback'])
    
    # Divergencias
    divergence = detect_divergence(df, DEFAULT_PARAMS['divergence_lookback'])
    
    # Volumen
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    volume_class = classify_volume(df['volume'].iloc[-1], avg_volume)
    
    # Señales de trading
    signals = []
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Condiciones para LONG
    long_conditions = (
        current['close'] > current['ema_slow'] and
        current['ema_fast'] > current['ema_slow'] and
        current['adx'] > DEFAULT_PARAMS['adx_level'] and
        divergence == 'bullish' and
        volume_class in ['Alto', 'Muy Alto'] and
        sr_data['current_support'] and
        abs(current['close'] - sr_data['current_support']) / sr_data['current_support'] < 0.01
    )
    
    # Condiciones para SHORT
    short_conditions = (
        current['close'] < current['ema_slow'] and
        current['ema_fast'] < current['ema_slow'] and
        current['adx'] > DEFAULT_PARAMS['adx_level'] and
        divergence == 'bearish' and
        volume_class in ['Alto', 'Muy Alto'] and
        sr_data['current_resistance'] and
        abs(current['close'] - sr_data['current_resistance']) / sr_data['current_resistance'] < 0.01
    )
    
    # Generar señal LONG
    if long_conditions:
        entry = max(current['close'], sr_data['current_support'])
        sl = entry - 2 * atr.iloc[-1]
        tp1 = entry + 1 * atr.iloc[-1]
        tp2 = entry + 2 * atr.iloc[-1]
        tp3 = entry + 3 * atr.iloc[-1]
        
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': 'LONG',
            'entry': round(entry, 4),
            'sl': round(sl, 4),
            'tp1': round(tp1, 4),
            'tp2': round(tp2, 4),
            'tp3': round(tp3, 4),
            'timestamp': current['timestamp']
        })
    
    # Generar señal SHORT
    if short_conditions:
        entry = min(current['close'], sr_data['current_resistance'])
        sl = entry + 2 * atr.iloc[-1]
        tp1 = entry - 1 * atr.iloc[-1]
        tp2 = entry - 2 * atr.iloc[-1]
        tp3 = entry - 3 * atr.iloc[-1]
        
        signals.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': 'SHORT',
            'entry': round(entry, 4),
            'sl': round(sl, 4),
            'tp1': round(tp1, 4),
            'tp2': round(tp2, 4),
            'tp3': round(tp3, 4),
            'timestamp': current['timestamp']
        })
    
    return signals

def analyze_all():
    # Leer lista de criptos
    with open('cryptos.txt', 'r') as f:
        cryptos = [line.strip() for line in f if line.strip()]
    
    timeframes = ['30m', '1h', '2h', '4h', '1d', '1w']
    
    all_signals = []
    
    for crypto in cryptos:
        for tf in timeframes:
            try:
                signals = analyze_symbol(crypto, tf)
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                print(f"Error analyzing {crypto} {tf}: {e}")
                traceback.print_exc()
    
    # Guardar en base de datos
    conn = sqlite3.connect('signals.db')
    cursor = conn.cursor()
    
    # Crear tabla si no existe
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        timeframe TEXT,
        signal_type TEXT,
        entry REAL,
        sl REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        timestamp DATETIME
    )
    ''')
    
    # Insertar nuevas señales
    for signal in all_signals:
        cursor.execute('''
        INSERT INTO signals (symbol, timeframe, signal_type, entry, sl, tp1, tp2, tp3, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'],
            signal['timeframe'],
            signal['signal_type'],
            signal['entry'],
            signal['sl'],
            signal['tp1'],
            signal['tp2'],
            signal['tp3'],
            signal['timestamp']
        ))
    
    conn.commit()
    conn.close()
