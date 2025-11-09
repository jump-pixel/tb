#calcule des indicateurs (RSI, MA…)

import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self):
        self.features = None
    
    def calculate_technical_indicators(self, data):
        """
        Calcule les indicateurs techniques
        """
        if data is None or data.empty:
            return None
            
        df = data.copy()
        
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            # Moyennes mobiles
            df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
            
            # CCI
            df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
            
            # ADX
            df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            
            # Retours
            df['returns_1d'] = df['Close'].pct_change()
            df['returns_5d'] = df['Close'].pct_change(5)
            
            # Volatilité
            df['volatility_10d'] = df['returns_1d'].rolling(10).std()
            
            # Volume
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            self.features = df
            print(f"✅ {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} indicateurs calculés")
            return df
            
        except Exception as e:
            print(f"❌ Erreur calcul indicateurs: {e}")
            return data
    
    def get_feature_columns(self):
        """
        Retourne la liste des colonnes de features
        """
        return ['rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal', 
                'macd_hist', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'adx', 
                'returns_1d', 'returns_5d', 'volatility_10d', 'volume_ratio']