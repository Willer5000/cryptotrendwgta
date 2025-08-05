import os
import time
import pandas as pd
import requests
import json
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        self.cryptos = []
        self.data_cache = {}
        self.last_fetch_time = {}
        self.cache_duration = 900  # 15 minutos en segundos
        
    def load_cryptos(self):
        if not self.cryptos:
            try:
                with open('cryptos.txt', 'r') as f:
                    self.cryptos = [line.strip() for line in f.readlines() if line.strip()]
            except:
                # Lista predeterminada si el archivo no existe
                self.cryptos = [
                    'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT', 'XRP-USDT',
                    'ADA-USDT', 'DOGE-USDT', 'AVAX-USDT', 'DOT-USDT', 'LINK-USDT'
                ]
        return self.cryptos
    
    def get_data(self, crypto, timeframe):
        cache_key = f"{crypto}_{timeframe}"
        current_time = time.time()
        
        # Verificar si los datos están en caché y son recientes
        if cache_key in self.data_cache:
            if current_time - self.last_fetch_time.get(cache_key, 0) < self.cache_duration:
                return self.data_cache[cache_key]
        
        # Mapear timeframe a parámetros de Kucoin
        tf_mapping = {
            '30min': '30min',
            '1h': '1hour',
            '2h': '2hour',
            '4h': '4hour',
            '1d': '1day',
            '1w': '1week'
        }
        
        kucoin_tf = tf_mapping.get(timeframe, '4hour')
        url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_tf}&symbol={crypto}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data['code'] == '200000' and data['data']:
                df = pd.DataFrame(data['data'], columns=[
                    'timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'
                ])
                
                # Convertir y ordenar datos
                df = df.iloc[::-1].reset_index(drop=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df = df.apply(pd.to_numeric)
                
                # Almacenar en caché
                self.data_cache[cache_key] = df
                self.last_fetch_time[cache_key] = current_time
                
                return df
        except Exception as e:
            print(f"Error fetching data for {crypto}: {str(e)}")
        
        return None
