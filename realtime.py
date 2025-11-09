import time
import pandas as pd
from datetime import datetime, timedelta

class RealTimeTester:
    def __init__(self, api_key=None):
        from data import DataFetcher
        from features import FeatureEngineer
        from strategy import TradingStrategy
        
        self.data_fetcher = DataFetcher(api_key=api_key)
        self.feature_engineer = FeatureEngineer()
        self.strategy = TradingStrategy()
        self.prediction_history = []
    
    def test_realtime_predictions(self, symbol, interval="5min", duration_minutes=30, update_interval=5):
        """
        Teste le bot en temps réel
        """
        print(f"🚀 TEST TEMPS RÉEL - {symbol}")
        print("="*50)
        
        # Charger ou entraîner le modèle
        model_file = f'trading_model_{symbol}.pkl'
        if not self.strategy.load_model(model_file):
            print("🤖 Entraînement du modèle...")
            historical_data = self.data_fetcher.fetch_data(symbol, "1y")
            
            if historical_data is not None and len(historical_data) >= 50:
                features_df = self.feature_engineer.calculate_technical_indicators(historical_data)
                if self.strategy.train_model(features_df):
                    self.strategy.save_model(model_file)
                else:
                    print("❌ Échec de l'entraînement")
                    return
            else:
                print("❌ Données historiques insuffisantes")
                return
        
        # Monitoring en temps réel
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        print(f"⏰ Début: {start_time.strftime('%H:%M:%S')}")
        print(f"⏰ Fin: {end_time.strftime('%H:%M:%S')}")
        print("="*50)
        
        iteration = 0
        while datetime.now() < end_time:
            iteration += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\n🔄 Itération {iteration} - {current_time}")
            
            try:
                # Récupérer données
                data = self.data_fetcher.fetch_intraday_data(symbol, interval)
                
                if data is not None and len(data) > 20:
                    # Calculer indicateurs
                    features = self.feature_engineer.calculate_technical_indicators(data)
                    
                    # Générer signal
                    signal = self.strategy.generate_signals(features)
                    
                    if signal:
                        # Prix actuel
                        current_price = self.data_fetcher.fetch_current_price(symbol)
                        
                        if current_price:
                            # Enregistrer
                            prediction = {
                                'timestamp': datetime.now(),
                                'symbol': symbol,
                                'signal': signal['signal'],
                                'confidence': signal['confidence'],
                                'price': current_price['price'],
                                'change': current_price['change_percent']
                            }
                            
                            self.prediction_history.append(prediction)
                            self.display_prediction(prediction)
                            
                            # Analyser tendance
                            if len(self.prediction_history) > 1:
                                self.analyze_trend()
                
                else:
                    print("❌ Données insuffisantes")
                    
            except Exception as e:
                print(f"❌ Erreur: {e}")
            
            # Attendre
            if datetime.now() < end_time:
                print(f"⏳ Attente de {update_interval} minutes...")
                time.sleep(update_interval * 60)
        
        # Résumé final
        self.display_summary()
    
    def display_prediction(self, prediction):
        """Affiche une prédiction"""
        print("\n" + "🎯" * 20)
        print(f"PRÉDICTION TEMPS RÉEL")
        print(f"🕒 {prediction['timestamp'].strftime('%H:%M:%S')}")
        print(f"📊 {prediction['symbol']}")
        print(f"💰 ${prediction['price']:.2f} {prediction['change']}")
        print(f"🎯 {prediction['signal']} ({prediction['confidence']:.1%})")
        
        if prediction['signal'] == 'ACHAT':
            print("💚 ACTION: ACHETER")
        else:
            print("💔 ACTION: VENDRE")
        print("🎯" * 20)
    
    def analyze_trend(self):
        """Analyse la tendance"""
        if len(self.prediction_history) < 2:
            return
        
        recent = self.prediction_history[-5:]
        buy_count = sum(1 for p in recent if p['signal'] == 'ACHAT')
        
        print(f"📈 Tendance: {buy_count}A/{len(recent)-buy_count}V")
    
    def display_summary(self):
        """Affiche le résumé"""
        print("\n" + "="*50)
        print("📊 RÉSUMÉ FINAL")
        print("="*50)
        
        if not self.prediction_history:
            print("❌ Aucune prédiction")
            return
        
        total = len(self.prediction_history)
        buy_signals = [p for p in self.prediction_history if p['signal'] == 'ACHAT']
        sell_signals = [p for p in self.prediction_history if p['signal'] == 'VENTE']
        
        print(f"📈 Prédictions: {total}")
        print(f"🛒 ACHAT: {len(buy_signals)} ({len(buy_signals)/total:.1%})")
        print(f"🏪 VENTE: {len(sell_signals)} ({len(sell_signals)/total:.1%})")
        
        # Dernier signal
        last = self.prediction_history[-1]
        print(f"\n🎯 Dernier signal: {last['signal']} ({last['confidence']:.1%})")
        print(f"💰 Prix: ${last['price']:.2f}")