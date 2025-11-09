# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def train_model(df):
    """Entraîne un modèle IA pour prédire la direction du marché"""
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    features = ["ma10", "ma50", "rsi"]
    X = df[features]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"✅ Modèle entraîné - Train acc: {train_acc:.2f}, Test acc: {test_acc:.2f}")

    df["prob_up"] = model.predict_proba(X_scaled)[:, 1]
    return df, model
