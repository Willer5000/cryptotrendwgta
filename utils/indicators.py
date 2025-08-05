import pandas as pd
import numpy as np
import talib

class TradingSignals:
    def __init__(self):
        self.params = {
            'ema_fast': 9,
            'ema_slow': 20,
            'adx_period': 14,
            'adx_level': 25,
            'rsi_period': 14,
            's_r_period': 20
        }
    
    def update_params(self, params):
        self.params = params
    
    def calculate_volume_class(self, current_volume, avg_volume):
        ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if ratio > 2.5:
            return 'Muy Alto'
        elif ratio > 1.5:
            return 'Alto'
        elif ratio > 0.8:
            return 'Medio'
        else:
            return 'Bajo'
    
    def find_support_resistance(self, df, period=20):
        df = df.copy().tail(period)
        
        # Identificar máximos y mínimos locales
        df['max'] = df['high'].rolling(5, center=True).max()
        df['min'] = df['low'].rolling(5, center=True).min()
        
        resistances = df[df['high'] == df['max']]
        supports = df[df['low'] == df['min']]
        
        # Agrupar niveles cercanos
        def cluster_levels(levels, threshold=0.02):
            if levels.empty:
                return []
                
            levels = levels.sort_values(ascending=True)
            clusters = []
            current_cluster = []
            
            for level in levels:
                if not current_cluster:
                    current_cluster.append(level)
                else:
                    if abs(level - np.mean(current_cluster)) < np.mean(current_cluster) * threshold:
                        current_cluster.append(level)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [level]
            
            if current_cluster:
                clusters.append(current_cluster)
            
            return [np.mean(cluster) for cluster in clusters]
        
        resistance_levels = cluster_levels(resistances['high'])
        support_levels = cluster_levels(supports['low'])
        
        # Obtener niveles más relevantes
        return {
            'support': min(support_levels) if support_levels else None,
            'resistance': max(resistance_levels) if resistance_levels else None
        }
    
    def calculate_risk_levels(self, df, signal_type, close_price):
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        atr_multiplier = 1.5
        
        if signal_type == 'LONG':
            entry = close_price
            sl = entry - atr * atr_multiplier
            tp1 = entry + atr * atr_multiplier
            tp2 = entry + atr * atr_multiplier * 2
            tp3 = entry + atr * atr_multiplier * 3
        else:  # SHORT
            entry = close_price
            sl = entry + atr * atr_multiplier
            tp1 = entry - atr * atr_multiplier
            tp2 = entry - atr * atr_multiplier * 2
            tp3 = entry - atr * atr_multiplier * 3
        
        return entry, sl, tp1, tp2, tp3
    
    def detect_divergence(self, df, rsi_period=14):
        # Detectar divergencias regulares
        df = df.copy().tail(50)
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
        
        # Encontrar máximos y mínimos
        df['max'] = df['high'].rolling(5, center=True).max()
        df['min'] = df['low'].rolling(5, center=True).min()
        
        # Buscar divergencias alcistas
        min_idx = df[df['low'] == df['min']].index
        if len(min_idx) > 2:
            last_min = df.loc[min_idx[-1]]
            prev_min = df.loc[min_idx[-2]]
            
            if (last_min['low'] < prev_min['low']) and (last_min['rsi'] > prev_min['rsi']):
                return 'bullish'
        
        # Buscar divergencias bajistas
        max_idx = df[df['high'] == df['max']].index
        if len(max_idx) > 2:
            last_max = df.loc[max_idx[-1]]
            prev_max = df.loc[max_idx[-2]]
            
            if (last_max['high'] > prev_max['high']) and (last_max['rsi'] < prev_max['rsi']):
                return 'bearish'
        
        return None
    
    def generate(self, df, crypto, timeframe):
        if len(df) < 50:
            return None
            
        close = df['close'].iloc[-1]
        volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        # Calcular indicadores
        ema_fast = talib.EMA(df['close'], timeperiod=self.params['ema_fast']).iloc[-1]
        ema_slow = talib.EMA(df['close'], timeperiod=self.params['ema_slow']).iloc[-1]
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.params['adx_period']).iloc[-1]
        rsi = talib.RSI(df['close'], timeperiod=self.params['rsi_period']).iloc[-1]
        
        # Clasificación de volumen
        volume_class = self.calculate_volume_class(volume, avg_volume)
        
        # Detectar divergencias
        divergence = self.detect_divergence(df, self.params['rsi_period'])
        
        # Encontrar soportes y resistencias
        s_r = self.find_support_resistance(df, self.params['s_r_period'])
        support = s_r.get('support')
        resistance = s_r.get('resistance')
        
        # Determinar señales
        signal = None
        signal_strength = 0
        
        # Condiciones para LONG
        long_conditions = [
            close > ema_slow,
            ema_fast > ema_slow,
            adx > self.params['adx_level'],
            volume_class in ['Alto', 'Muy Alto'],
            divergence == 'bullish',
            support and (close <= support * 1.01)
        ]
        
        # Condiciones para SHORT
        short_conditions = [
            close < ema_slow,
            ema_fast < ema_slow,
            adx > self.params['adx_level'],
            volume_class in ['Alto', 'Muy Alto'],
            divergence == 'bearish',
            resistance and (close >= resistance * 0.99)
        ]
        
        if sum(long_conditions) >= 4:
            signal = 'LONG'
            signal_strength = sum(long_conditions)
        elif sum(short_conditions) >= 4:
            signal = 'SHORT'
            signal_strength = sum(short_conditions)
        
        if not signal:
            return None
        
        # Calcular niveles de riesgo
        entry, sl, tp1, tp2, tp3 = self.calculate_risk_levels(df, signal, close)
        
        return {
            'crypto': crypto,
            'timeframe': timeframe,
            'signal': signal,
            'strength': signal_strength,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'volume_class': volume_class,
            'current_price': close,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'adx': adx,
            'rsi': rsi
        }
