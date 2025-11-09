# src/main.py
import argparse
import sys
import os

# Configuration des imports
sys.path.append(os.path.dirname(__file__))

from data import DataFetcher
from features import FeatureEngineer
from strategy import TradingStrategy
from backtest import Backtester
from realtime import RealTimeTester

def main():
    parser = argparse.ArgumentParser(description='Bot de Trading IA')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbole boursier')
    parser.add_argument('--period', type=str, default='1y', help='Période des données')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle')
    parser.add_argument('--backtest', action='store_true', help='Lancer le backtest')
    parser.add_argument('--predict', action='store_true', help='Faire une prédiction')
    parser.add_argument('--realtime', action='store_true', help='Test temps réel')
    parser.add_argument('--api-key', type=str, help='Clé API Alpha Vantage')
    
    args = parser.parse_args()
    
    print(f"🤖 BOT DE TRADING IA - {args.symbol}")
    print("="*50)
    
    # Initialisation
    data_fetcher = DataFetcher(api_key=args.api_key)
    feature_engineer = FeatureEngineer()
    strategy = TradingStrategy()
    backtester = Backtester()
    
    if args.realtime:
        # Test temps réel
        tester = RealTimeTester(api_key=args.api_key)
        tester.test_realtime_predictions(args.symbol, duration_minutes=15, update_interval=2)
        return
    
    # Récupération des données
    data = data_fetcher.fetch_data(args.symbol, args.period)
    
    if data is None or data.empty:
        print("❌ Impossible de récupérer les données")
        return
    
    # Calcul des indicateurs
    features_df = feature_engineer.calculate_technical_indicators(data)
    
    if features_df is None:
        print("❌ Erreur calcul des indicateurs")
        return
    
    # Entraînement
    if args.train:
        print("🤖 Entraînement du modèle...")
        if strategy.train_model(features_df):
            strategy.save_model(f'trading_model_{args.symbol}.pkl')
    
    # Backtest
    if args.backtest:
        print("📈 Backtest en cours...")
        results = backtester.run_backtest(features_df, strategy)
        if results:
            backtester.print_results()
            backtester.plot_results(features_df)
    
    # Prédiction
    if args.predict:
        print("🔮 Génération du signal...")
        if not strategy.is_trained:
            # Charger modèle existant
            if not strategy.load_model(f'trading_model_{args.symbol}.pkl'):
                print("❌ Modèle non entraîné")
                return
        
        signal = strategy.generate_signals(features_df)
        if signal:
            print(f"\n🎯 SIGNAL: {signal['signal']}")
            print(f"📊 Confiance: {signal['confidence']:.2%}")
            print(f"💰 Prix: ${data['Close'].iloc[-1]:.2f}")
            
            if signal['signal'] == 'ACHAT':
                print("💚 RECOMMANDATION: ACHETER")
            else:
                print("💔 RECOMMANDATION: VENDRE")

if __name__ == "__main__":
    main()