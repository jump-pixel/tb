# test.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data import DataFetcher
    from features import FeatureEngineer
    print("✅ Toutes les bibliothèques sont importées avec succès!")
    
    # Test rapide
    fetcher = DataFetcher()
    data = fetcher.fetch_data('AAPL', period='1mo')
    print(f"✅ Données récupérées: {len(data)} lignes")
    
    engineer = FeatureEngineer()
    features = engineer.calculate_technical_indicators(data)
    print(f"✅ Features calculées: {len(features.columns)} indicateurs")
    
except Exception as e:
    print(f"❌ Erreur: {e}")