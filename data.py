#récupère les données (yfinance)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

class DataFetcher:
    def __init__(self, api_key=None):
        """
        Initialise le DataFetcher avec une clé API Alpha Vantage
        """
        # ⚠️ INTERVENTION REQUISE : Remplacez par votre clé API Alpha Vantage
        self.api_key = api_key or "VOTRE_CLE_API_ICI"  # Clé gratuite sur https://www.alphavantage.co/support/#api-key
        self.data = None
    
    def fetch_data(self, symbol, period="2y", interval="daily"):
    
     try:
        print(f"🔍 Recherche des données Alpha Vantage pour {symbol}...")
        
        # Mapping des périodes
        period_mapping = {
            "1mo": "compact",
            "3mo": "compact", 
            "6mo": "compact",
            "1y": "compact",
            "2y": "full",
            "5y": "full"
        }
        
        output_size = period_mapping.get(period, "compact")
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={self.api_key}"
        
        print(f"   URL: {url.split('apikey')[0]}...")
        
        response = requests.get(url)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convertir les types de données
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Trier par date
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Renommer les colonnes pour correspondre à yfinance
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # CORRECTION : Filtrer pour n'avoir que les dates passées
            current_date = datetime.now()
            df = df[df.index <= current_date]
            
            # Filtrer selon la période demandée
            if period in ["1y", "2y", "5y"]:
                if period == "1y":
                    start_date = current_date - timedelta(days=365)
                elif period == "2y":
                    start_date = current_date - timedelta(days=730)
                else:  # 5y
                    start_date = current_date - timedelta(days=1825)
                
                df = df[df.index >= start_date]
            
            self.data = df
            print(f"✅ Données Alpha Vantage récupérées: {len(df)} jours")
            if len(df) > 0:
                print(f"   Période: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                print(f"   Dernier prix: ${df['Close'].iloc[-1]:.2f}")
            return df
        else:
            error_msg = data.get("Error Message", "Unknown error")
            note_msg = data.get("Note", "")
            print(f"❌ Erreur Alpha Vantage: {error_msg}")
            if note_msg:
                print(f"   Note: {note_msg}")
            return None
                
     except Exception as e:
        print(f"❌ Erreur lors de la récupération des données: {e}")
        return None
    
    def fetch_data_with_dates(self, symbol, start_date="2023-01-01", end_date=None):
        """
        Récupère les données avec des dates spécifiques
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            print(f"🔍 Recherche des données pour {symbol} du {start_date} au {end_date}...")
            
            # Récupérer les données complètes
            df = self.fetch_data(symbol, period="5y")
            if df is None:
                return None
            
            # Filtrer par dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            filtered_df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            if filtered_df.empty:
                print("❌ Aucune donnée dans la plage de dates spécifiée")
                return None
            
            print(f"✅ Données filtrées: {len(filtered_df)} jours")
            return filtered_df
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def get_multiple_symbols(self, symbols, period="1y"):
        """
        Récupère les données pour plusieurs symboles
        """
        data_dict = {}
        for symbol in symbols:
            print(f"\n📊 Récupération des données pour {symbol}...")
            data_dict[symbol] = self.fetch_data(symbol, period)
            time.sleep(1)  # Respecter les limites de l'API
        return data_dict
    
    def get_features(self):
        """
        Retourne les données brutes pour le calcul des features
        """
        return self.data
    
    def fetch_intraday_data(self, symbol, interval="5min", output_size="compact"):
  
     try:
        print(f"🔍 Récupération des données intraday pour {symbol} ({interval})...")
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={output_size}&apikey={self.api_key}"
        
        print(f"   URL: {url.split('apikey')[0]}...")
        
        response = requests.get(url)
        data = response.json()
        
        # La clé varie selon l'intervalle (ex: "Time Series (5min)")
        time_series_key = f"Time Series ({interval})"
        
        if time_series_key in data:
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convertir les types de données
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Trier par date
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Renommer les colonnes pour correspondre au format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            self.data = df
            print(f"✅ Données intraday récupérées: {len(df)} périodes de {interval}")
            if len(df) > 0:
                print(f"   Période: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
                print(f"   Dernier prix: ${df['Close'].iloc[-1]:.2f}")
            return df
        else:
            error_msg = data.get("Error Message", "Unknown error")
            note_msg = data.get("Note", "")
            print(f"❌ Erreur Alpha Vantage: {error_msg}")
            if note_msg:
                print(f"   Note: {note_msg}")
            return None
            
     except Exception as e:
        print(f"❌ Erreur lors de la récupération des données intraday: {e}")
        return None

def fetch_current_price(self, symbol):
    """
    Récupère le prix actuel et les données récentes
    """
    try:
        print(f"🔍 Récupération du prix actuel pour {symbol}...")
        
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            current_data = {
                'symbol': symbol,
                'price': float(quote['05. price']),
                'change': float(quote['09. change']),
                'change_percent': quote['10. change percent'],
                'timestamp': datetime.now()
            }
            
            print(f"✅ Prix actuel: ${current_data['price']:.2f} ({current_data['change_percent']})")
            return current_data
        else:
            print("❌ Impossible de récupérer le prix actuel")
            return None
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None
