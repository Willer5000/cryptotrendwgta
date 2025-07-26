import requests
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta

TIMEFRAMES = {
    '30m': 1800,
    '1h': 3600,
    '2h': 7200,
    '4h': 14400,
    '1d': 86400,
    '1w': 604800
}

def fetch_ohlcv(symbol, timeframe, limit=200):
    try:
        end = int(time.time())
        start = end - (TIMEFRAMES[timeframe] * limit)
        
        url = f"https://api.kucoin.com/api/v1/market/candles?type={timeframe}&symbol={symbol}&startAt={start}&endAt={end}"
        response = requests.get(url)
        data = response.json()
        
        if data['code'] == '200000' and data['data']:
            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
            df = df.iloc[::-1]  # Invertir orden
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            return df
        return None
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        return None

def update_all_data():
    # Leer lista de criptos
    with open('cryptos.txt', 'r') as f:
        cryptos = [line.strip() for line in f if line.strip()]
    
    conn = sqlite3.connect('ohlcv.db')
    
    for crypto in cryptos:
        for tf in TIMEFRAMES.keys():
            df = fetch_ohlcv(crypto, tf)
            if df is not None:
                df.to_sql('ohlcv_data', conn, if_exists='append', index=False)
            time.sleep(0.2)  # Evitar rate limiting
    
    conn.close()
