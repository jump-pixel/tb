#teste la stratégie sur le passé

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.results = None
    
    def run_backtest(self, features_df, strategy):
        """
        Exécute un backtest de la stratégie
        """
        if features_df is None or len(features_df) < 50:
            print("❌ Données insuffisantes pour le backtest")
            return None
        
        df = features_df.copy().dropna()
        
        # Préparer les features
        X, y, df_clean = strategy.prepare_features(df)
        
        if X is None or len(X) < 30:
            print("❌ Pas assez de données après préparation")
            return None
        
        # Backtest simple
        capital = self.initial_capital
        position = 0  # 0: pas de position, 1: long
        trades = []
        portfolio_value = []
        
        # Utiliser les prédictions du modèle
        try:
            X_scaled = strategy.scaler.transform(X)
            predictions = strategy.model.predict(X_scaled)
            probabilities = strategy.model.predict_proba(X_scaled)
        except Exception as e:
            print(f"❌ Erreur lors des prédictions: {e}")
            return None
        
        entry_price = 0
        shares = 0
        
        for i in range(len(df_clean)):
            current_price = df_clean['Close'].iloc[i]
            prediction = predictions[i]
            confidence = probabilities[i].max()
            
            # Règles de trading
            if position == 0 and prediction == 1 and confidence > 0.6:
                # Achat
                position = 1
                entry_price = current_price
                shares = capital / current_price
                trades.append({
                    'date': df_clean.index[i],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'confidence': confidence
                })
            
            elif position == 1 and prediction == 0 and confidence > 0.6:
                # Vente
                position = 0
                exit_price = current_price
                pnl = (exit_price - entry_price) * shares
                capital = shares * exit_price
                trades.append({
                    'date': df_clean.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'pnl': pnl,
                    'confidence': confidence
                })
            
            # Valeur du portefeuille
            if position == 1:
                portfolio_value.append(shares * current_price)
            else:
                portfolio_value.append(capital)
        
        # Calcul des métriques
        final_value = portfolio_value[-1] if portfolio_value else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Métriques de performance
        returns = pd.Series(portfolio_value).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calcul du maximum drawdown
        portfolio_series = pd.Series(portfolio_value)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        self.results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'portfolio_value': portfolio_value,
            'returns': returns
        }
        
        return self.results
    
    def print_results(self):
        """
        Affiche les résultats du backtest
        """
        if self.results is None:
            print("❌ Aucun résultat à afficher")
            return
        
        print("\n" + "="*60)
        print("📊 RÉSULTATS DU BACKTEST")
        print("="*60)
        print(f"💰 Capital initial: ${self.results['initial_capital']:,.2f}")
        print(f"💰 Valeur finale: ${self.results['final_value']:,.2f}")
        print(f"📈 Rendement total: {self.results['total_return']:+.2%}")
        print(f"🎯 Ratio de Sharpe: {self.results['sharpe_ratio']:.2f}")
        print(f"📉 Maximum Drawdown: {self.results['max_drawdown']:+.2%}")
        
        buy_trades = [t for t in self.results['trades'] if t['action'] == 'BUY']
        sell_trades = [t for t in self.results['trades'] if t['action'] == 'SELL']
        
        print(f"🛒 Trades d'achat: {len(buy_trades)}")
        print(f"🏪 Trades de vente: {len(sell_trades)}")
        
        if sell_trades:
            total_pnl = sum(t['pnl'] for t in sell_trades)
            avg_pnl = total_pnl / len(sell_trades)
            win_rate = len([t for t in sell_trades if t['pnl'] > 0]) / len(sell_trades)
            print(f"📊 PNL total: ${total_pnl:+.2f}")
            print(f"📊 PNL moyen: ${avg_pnl:+.2f}")
            print(f"🎯 Taux de réussite: {win_rate:.2%}")
        
        print("="*60)
    
    def plot_results(self, features_df):
        """
        Affiche un graphique des performances
        """
        if self.results is None:
            print("❌ Aucun résultat à afficher")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Prix et valeur du portefeuille
        df_clean = features_df.dropna()
        
        plt.subplot(2, 1, 1)
        if len(self.results['portfolio_value']) > 0:
            dates = df_clean.index[:len(self.results['portfolio_value'])]
            plt.plot(dates, self.results['portfolio_value'], label='Valeur du portefeuille', linewidth=2)
            plt.title('Performance du Portefeuille', fontsize=14, fontweight='bold')
            plt.ylabel('Valeur ($)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(df_clean.index, df_clean['Close'], label='Prix de clôture', linewidth=2)
        
        # Marquer les points d'achat/vente
        buy_dates = [t['date'] for t in self.results['trades'] if t['action'] == 'BUY']
        sell_dates = [t['date'] for t in self.results['trades'] if t['action'] == 'SELL']
        buy_prices = [t['price'] for t in self.results['trades'] if t['action'] == 'BUY']
        sell_prices = [t['price'] for t in self.results['trades'] if t['action'] == 'SELL']
        
        if buy_dates:
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Achat', zorder=5)
        if sell_dates:
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Vente', zorder=5)
        
        plt.title('Prix et Signaux de Trading', fontsize=14, fontweight='bold')
        plt.ylabel('Prix ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()