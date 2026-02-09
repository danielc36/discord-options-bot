import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from sklearn.ensemble import RandomForestClassifier

print("📥 Downloading historical SPY data...")

df1m = yf.download("SPY", period="7d", interval="1m", auto_adjust=True)
df15m = yf.download("SPY", period="30d", interval="15m", auto_adjust=True)

df1m.dropna(inplace=True)
df15m.dropna(inplace=True)

# Force 1D series (fix for your error)
close1m = df1m["Close"].squeeze()
high1m = df1m["High"].squeeze()
low1m = df1m["Low"].squeeze()

close15m = df15m["Close"].squeeze()
high15m = df15m["High"].squeeze()
low15m = df15m["Low"].squeeze()

# ---------------- INDICATORS ---------------- #

bb = BollingerBands(close1m)
df1m["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()

stoch = StochasticOscillator(high1m, low1m, close1m)
df1m["stoch"] = stoch.stoch()

atr = AverageTrueRange(high1m, low1m, close1m)
df1m["atr"] = atr.average_true_range()

df1m["std"] = close1m.rolling(20).std()

adx = ADXIndicator(high15m, low15m, close15m)
df15m["adx"] = adx.adx()

df1m.dropna(inplace=True)
df15m.dropna(inplace=True)

# Align sizes
min_len = min(len(df1m), len(df15m))
df1m = df1m.iloc[-min_len:]
df15m = df15m.iloc[-min_len:]

# ---------------- FEATURES ---------------- #

X = pd.DataFrame({
    "stoch": df1m["stoch"],
    "bb_width": df1m["bb_width"],
    "atr": df1m["atr"],
    "adx": df15m["adx"],
    "std": df1m["std"]
})

# Label: next candle direction
y = (close1m.shift(-1).iloc[-min_len:] > close1m.iloc[-min_len:]).astype(int)

X.dropna(inplace=True)
y = y.loc[X.index]

# ---------------- TRAIN MODEL ---------------- #

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
print("Features used:", list(X.columns))
