# src/strategy.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

class TradingStrategy:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, features_df):
        """
        Prépare les features pour l'entraînement
        """
        if features_df is None:
            return None, None, None
            
        feature_columns = ['rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal', 
                          'macd_hist', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'adx', 
                          'returns_1d', 'returns_5d', 'volatility_10d', 'volume_ratio']
        
        # Vérifier que les colonnes existent
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        if len(available_features) < 5:
            print("❌ Pas assez de features disponibles")
            return None, None, None
        
        # Supprimer les lignes avec des valeurs manquantes
        df_clean = features_df.dropna().copy()
        
        if len(df_clean) < 30:
            print("❌ Pas assez de données après nettoyage")
            return None, None, None
        
        # Créer la variable cible (1 si le prix monte le jour suivant, 0 sinon)
        df_clean['target'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)
        
        # Features et target
        X = df_clean[available_features]
        y = df_clean['target']
        
        return X, y, df_clean
    
    def train_model(self, features_df):
        """
        Entraîne le modèle de machine learning
        """
        X, y, _ = self.prepare_features(features_df)
        
        if X is None or len(X) < 50:
            print(f"❌ Pas assez de données pour l'entraînement ({len(X) if X is not None else 0} échantillons)")
            return False
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        if len(X_train) == 0 or len(X_test) == 0:
            print("❌ Erreur dans le split train/test")
            return False
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraînement du modèle
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Accuracy du modèle: {accuracy:.2%}")
        print("📈 Rapport de classification:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return True
    
    def generate_signals(self, features_df):
        """
        Génère les signaux d'achat/vente
        """
        if not self.is_trained or self.model is None:
            print("❌ Le modèle n'est pas entraîné")
            return None
        
        feature_columns = ['rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal', 
                          'macd_hist', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'adx', 
                          'returns_1d', 'returns_5d', 'volatility_10d', 'volume_ratio']
        
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        if len(available_features) == 0:
            print("❌ Aucune feature disponible")
            return None
        
        # Dernières données
        latest_data = features_df[available_features].dropna().tail(1)
        
        if latest_data.empty:
            print("❌ Aucune donnée récente")
            return None
        
        try:
            # Prédiction
            latest_scaled = self.scaler.transform(latest_data)
            prediction = self.model.predict(latest_scaled)[0]
            probability = self.model.predict_proba(latest_scaled)[0]
            
            # Génération du signal
            signal = "ACHAT" if prediction == 1 else "VENTE"
            confidence = probability.max()
            
            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': prediction,
                'probabilities': probability
            }
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return None
    
    def save_model(self, filename='trading_model.pkl'):
        """
        Sauvegarde le modèle entraîné
        """
        if self.is_trained and self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, filename)
            print(f"💾 Modèle sauvegardé sous {filename}")
        else:
            print("❌ Aucun modèle à sauvegarder")
    
    def load_model(self, filename='trading_model.pkl'):
        """
        Charge un modèle pré-entraîné
        """
        try:
            loaded = joblib.load(filename)
            self.model = loaded['model']
            self.scaler = loaded['scaler']
            self.is_trained = True
            print(f"📂 Modèle chargé depuis {filename}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False