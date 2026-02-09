import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("📥 Downloading historical SPY data...")

df1m = yf.download("SPY", period="5d", interval="1m")
df15m = yf.download("SPY", period="60d", interval="15m")

# Fix multi-index columns
df1m.columns = df1m.columns.get_level_values(0)
df15m.columns = df15m.columns.get_level_values(0)

df1m.dropna(inplace=True)
df15m.dropna(inplace=True)

# -------- 1 MIN INDICATORS -------- #

bb1 = BollingerBands(close=df1m["Close"])
df1m["bb_width_1m"] = bb1.bollinger_hband() - bb1.bollinger_lband()

atr1 = AverageTrueRange(df1m["High"], df1m["Low"], df1m["Close"])
df1m["atr_1m"] = atr1.average_true_range()

stoch1 = StochasticOscillator(df1m["High"], df1m["Low"], df1m["Close"])
df1m["stoch_1m"] = stoch1.stoch()

vwap1 = VolumeWeightedAveragePrice(
    high=df1m["High"],
    low=df1m["Low"],
    close=df1m["Close"],
    volume=df1m["Volume"]
)
df1m["vwap_1m"] = vwap1.vwap   # FIXED

df1m["std_1m"] = df1m["Close"].rolling(20).std()

# -------- 15 MIN INDICATORS -------- #

bb15 = BollingerBands(close=df15m["Close"])
df15m["bb_width_15m"] = bb15.bollinger_hband() - bb15.bollinger_lband()

stoch15 = StochasticOscillator(df15m["High"], df15m["Low"], df15m["Close"])
df15m["stoch_15m"] = stoch15.stoch()

adx15 = ADXIndicator(df15m["High"], df15m["Low"], df15m["Close"])
df15m["adx_15m"] = adx15.adx()

df15m["std_15m"] = df15m["Close"].rolling(20).std()

# -------- ALIGN TIMEFRAMES -------- #

df1m = df1m.reset_index()
df15m = df15m.reset_index()

df1m["time"] = pd.to_datetime(df1m["Datetime"]).dt.floor("15min")
df15m["time"] = pd.to_datetime(df15m["Datetime"])

merged = pd.merge(df1m, df15m, on="time", suffixes=("_1m", "_15m"))

# -------- LABELS -------- #

merged["future_close"] = merged["Close_1m"].shift(-5)
merged["target"] = np.where(merged["future_close"] > merged["Close_1m"], 1, 0)

# -------- FEATURES -------- #

feature_cols = [
    "stoch_1m",
    "bb_width_1m",
    "atr_1m",
    "std_1m",
    "vwap_1m",
    "adx_15m",
    "stoch_15m",
    "bb_width_15m",
    "std_15m"
]

merged.dropna(inplace=True)

X = merged[feature_cols]
y = merged["target"]

# -------- TRAIN -------- #

print("🧠 Training ML model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(model, "model.pkl")
print("✅ model.pkl saved successfully")
