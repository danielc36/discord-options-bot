import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice
from sklearn.ensemble import RandomForestClassifier

print("📥 Downloading historical SPY data...")

df1m = yf.download("SPY", period="7d", interval="1m", auto_adjust=True)
df15m = yf.download("SPY", period="30d", interval="15m", auto_adjust=True)

df1m.dropna(inplace=True)
df15m.dropna(inplace=True)

# Force 1D series
close1m = df1m["Close"].squeeze()
high1m = df1m["High"].squeeze()
low1m = df1m["Low"].squeeze()
vol1m = df1m["Volume"].squeeze()

close15m = df15m["Close"].squeeze()
high15m = df15m["High"].squeeze()
low15m = df15m["Low"].squeeze()

# ---------- Indicators (1m) ---------- #

bb1 = BollingerBands(close1m)
df1m["bb_width_1m"] = bb1.bollinger_hband() - bb1.bollinger_lband()

stoch1 = StochasticOscillator(high1m, low1m, close1m)
df1m["stoch_1m"] = stoch1.stoch()

atr = AverageTrueRange(high1m, low1m, close1m)
df1m["atr_1m"] = atr.average_true_range()

df1m["std_1m"] = close1m.rolling(20).std()

vwap = VolumeWeightedAveragePrice(high1m, low1m, close1m, vol1m)
df1m["vwap_1m"] = vwap.vwap

# ---------- Indicators (15m) ---------- #

bb15 = BollingerBands(close15m)
df15m["bb_width_15m"] = bb15.bollinger_hband() - bb15.bollinger_lband()

stoch15 = StochasticOscillator(high15m, low15m, close15m)
df15m["stoch_15m"] = stoch15.stoch()

adx = ADXIndicator(high15m, low15m, close15m)
df15m["adx_15m"] = adx.adx()

df15m["std_15m"] = close15m.rolling(20).std()

df1m.dropna(inplace=True)
df15m.dropna(inplace=True)

# Align rows
min_len = min(len(df1m), len(df15m))
df1m = df1m.iloc[-min_len:]
df15m = df15m.iloc[-min_len:]

# ---------- Features ---------- #

X = pd.DataFrame({
    "stoch_1m": df1m["stoch_1m"],
    "bb_width_1m": df1m["bb_width_1m"],
    "atr_1m": df1m["atr_1m"],
    "std_1m": df1m["std_1m"],
    "vwap_1m": df1m["vwap_1m"],
    "adx_15m": df15m["adx_15m"],
    "stoch_15m": df15m["stoch_15m"],
    "bb_width_15m": df15m["bb_width_15m"],
    "std_15m": df15m["std_15m"]
})

y = (close1m.shift(-1).iloc[-min_len:] > close1m.iloc[-min_len:]).astype(int)

X.dropna(inplace=True)
y = y.loc[X.index]

# ---------- Train ---------- #

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
print("Features:", list(X.columns))
