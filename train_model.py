import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice
from market import get_stock_df

symbol = "SPY"

df = get_stock_df(symbol, interval="1m")

bb = BollingerBands(df["Close"])
df["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()

adx = ADXIndicator(df["High"], df["Low"], df["Close"])
df["adx"] = adx.adx()

stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
df["stoch"] = stoch.stoch()

vwap = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"])
df["vwap"] = vwap.vwap

df["std"] = df["Close"].rolling(20).std()
df["future"] = df["Close"].shift(-3)

df.dropna(inplace=True)

df["label"] = (df["future"] > df["Close"]).astype(int)

features = ["stoch", "bb_width", "adx", "vwap", "std"]
X = df[features]
y = df["label"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully")
