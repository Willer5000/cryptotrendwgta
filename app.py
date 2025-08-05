import os
import time
import threading
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template
from datetime import datetime, timedelta

app = Flask(__name__)

# Configuración
TIME_INTERVALS = {
    '30m': 1800,
    '1h': 3600,
    '2h': 7200,
    '4h': 14400,
    '1d': 86400,
    '1w': 604800
}
DEFAULT_CRYPTOS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK']
DATA_REFRESH = 900  # 15 minutos

class CryptoAnalyzer:
    def __init__(self):
        self.cryptos = self.load_cryptos()
        self.signals = {'LONG': [], 'SHORT': []}
        self.last_update = None
        self.lock = threading.Lock()
        
    def load_cryptos(self):
        if os.path.exists('cryptos.txt'):
            with open('cryptos.txt', 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return DEFAULT_CRYPTOS
    
    def fetch_data(self, symbol, interval):
        endpoint = "https://api.kucoin.com/api/v1/market/candles"
        now = int(time.time())
        start = now - TIME_INTERVALS[interval] * 200
        params = {
            'type': interval,
            'symbol': f'{symbol}-USDT',
            'startAt': start,
            'endAt': now
        }
        try:
            response = requests.get(endpoint, params=params)
            data = response.json()
            if data.get('code') == '200000' and data['data']:
                df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df = df.iloc[::-1].reset_index(drop=True)
                return df.astype(float)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return None
    
    def calculate_ema(self, prices, period):
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_adx(self, high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
        return dx.rolling(period).mean()
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def find_support_resistance(self, df):
        pivot_range = 5
        highs = df['high']
        lows = df['low']
        
        resistance = highs.rolling(window=pivot_range*2+1, center=True).max()
        support = lows.rolling(window=pivot_range*2+1, center=True).min()
        
        current_price = df['close'].iloc[-1]
        closest_res = resistance[resistance > current_price].min()
        closest_sup = support[support < current_price].max()
        
        return closest_sup, closest_res
    
    def analyze_symbol(self, symbol, interval):
        df = self.fetch_data(symbol, interval)
        if df is None or len(df) < 50:
            return None, None
        
        # Calcular indicadores
        df['ema9'] = self.calculate_ema(df['close'], 9)
        df['ema20'] = self.calculate_ema(df['close'], 20)
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'])
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Últimos valores
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Clasificación de volumen
        avg_volume = df['volume'].mean()
        vol_ratio = last['volume'] / avg_volume
        if vol_ratio > 3: vol_class = "Muy Alto"
        elif vol_ratio > 2: vol_class = "Alto"
        elif vol_ratio > 1: vol_class = "Medio"
        else: vol_class = "Bajo"
        
        # Detectar divergencias
        bullish_div = (last['close'] < prev['close'] and last['rsi'] > prev['rsi'] and last['rsi'] < 30)
        bearish_div = (last['close'] > prev['close'] and last['rsi'] < prev['rsi'] and last['rsi'] > 70)
        
        # Encontrar soportes y resistencias
        support, resistance = self.find_support_resistance(df)
        
        # Señal LONG
        long_signal = (
            last['close'] > last['ema20'] and
            last['ema9'] > last['ema20'] and
            last['adx'] > 25 and
            bullish_div and
            support is not None and
            last['close'] <= support * 1.01
        )
        
        # Señal SHORT
        short_signal = (
            last['close'] < last['ema20'] and
            last['ema9'] < last['ema20'] and
            last['adx'] > 25 and
            bearish_div and
            resistance is not None and
            last['close'] >= resistance * 0.99
        )
        
        return {
            'symbol': symbol,
            'price': last['close'],
            'support': support,
            'resistance': resistance,
            'volume_class': vol_class,
            'long_signal': long_signal,
            'short_signal': short_signal,
            'rsi': last['rsi'],
            'adx': last['adx']
        }, df
    
    def calculate_risk_levels(self, signal_type, analysis):
        if signal_type == 'LONG':
            entry = analysis['support']
            sl = entry * 0.97  # -3%
            tp1 = entry * 1.03  # +3%
            tp2 = entry * 1.06  # +6%
            tp3 = entry * 1.09  # +9%
        else:  # SHORT
            entry = analysis['resistance']
            sl = entry * 1.03  # +3%
            tp1 = entry * 0.97  # -3%
            tp2 = entry * 0.94  # -6%
            tp3 = entry * 0.91  # -9%
        
        return {
            'entry': round(entry, 4),
            'sl': round(sl, 4),
            'tp1': round(tp1, 4),
            'tp2': round(tp2, 4),
            'tp3': round(tp3, 4)
        }
    
    def update_signals(self):
        while True:
            start_time = time.time()
            long_signals = []
            short_signals = []
            
            for crypto in self.cryptos:
                try:
                    analysis, _ = self.analyze_symbol(crypto, '4h')
                    if analysis:
                        if analysis['long_signal']:
                            risk_levels = self.calculate_risk_levels('LONG', analysis)
                            long_signals.append({
                                'symbol': crypto,
                                **risk_levels,
                                'rsi': round(analysis['rsi'], 2),
                                'adx': round(analysis['adx'], 2),
                                'volume': analysis['volume_class']
                            })
                        
                        if analysis['short_signal']:
                            risk_levels = self.calculate_risk_levels('SHORT', analysis)
                            short_signals.append({
                                'symbol': crypto,
                                **risk_levels,
                                'rsi': round(analysis['rsi'], 2),
                                'adx': round(analysis['adx'], 2),
                                'volume': analysis['volume_class']
                            })
                    
                    time.sleep(0.1)  # Evitar rate limiting
                except Exception as e:
                    print(f"Error processing {crypto}: {e}")
            
            with self.lock:
                self.signals['LONG'] = long_signals
                self.signals['SHORT'] = short_signals
                self.last_update = datetime.now()
            
            elapsed = time.time() - start_time
            sleep_time = max(DATA_REFRESH - elapsed, 0)
            time.sleep(sleep_time)

analyzer = CryptoAnalyzer()

@app.route('/')
def index():
    with analyzer.lock:
        long_signals = analyzer.signals['LONG']
        short_signals = analyzer.signals['SHORT']
        last_update = analyzer.last_update.strftime("%Y-%m-%d %H:%M:%S") if analyzer.last_update else "Nunca"
    
    return render_template(
        'index.html',
        long_signals=enumerate(long_signals, 1),
        short_signals=enumerate(short_signals, 1),
        last_update=last_update,
        cryptos_count=len(analyzer.cryptos)
    )

def run_scheduler():
    analyzer.update_signals()

if __name__ == '__main__':
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
