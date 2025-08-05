import os
import time
import pandas as pd
from flask import Flask, render_template, request
from utils.data_loader import DataLoader
from utils.indicators import TradingSignals

app = Flask(__name__)
data_loader = DataLoader()
signals_generator = TradingSignals()

def get_signals(timeframe):
    cryptos = data_loader.load_cryptos()
    all_signals = []
    
    for crypto in cryptos:
        try:
            df = data_loader.get_data(crypto, timeframe)
            if df is None or df.empty:
                continue
                
            signals = signals_generator.generate(df, crypto, timeframe)
            if signals:
                all_signals.append(signals)
        except Exception as e:
            print(f"Error processing {crypto}: {str(e)}")
    
    return all_signals

@app.route('/')
def index():
    timeframe = request.args.get('timeframe', '4h')
    
    # Parámetros configurables
    params = {
        'ema_fast': int(request.args.get('ema_fast', 9)),
        'ema_slow': int(request.args.get('ema_slow', 20)),
        'adx_period': int(request.args.get('adx_period', 14)),
        'adx_level': int(request.args.get('adx_level', 25)),
        'rsi_period': int(request.args.get('rsi_period', 14)),
        's_r_period': int(request.args.get('s_r_period', 20))
    }
    
    signals_generator.update_params(params)
    signals = get_signals(timeframe)
    
    long_signals = [s for s in signals if s['signal'] == 'LONG']
    short_signals = [s for s in signals if s['signal'] == 'SHORT']
    
    return render_template(
        'index.html',
        long_signals=long_signals,
        short_signals=short_signals,
        params=params,
        timeframe=timeframe,
        last_update=time.strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == '__main__':
    app.run(debug=True)
