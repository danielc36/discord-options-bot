import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

SYMBOL = "SPY"
INTERVAL = "1m"
PERIOD = "7d"  # yfinance max for 1m is 7 days

# ---------------- DOWNLOAD DATA ---------------- #

df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD)
df.dropna(inplace=True)

# ---------------- INDICATORS ---------------- #

bb = BollingerBands(df["Close"])
df["bb_high"] = bb.bollinger_hband()
df["bb_low"] = bb.bollinger_lband()

stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
df["stoch"] = stoch.stoch()

adx = ADXIndicator(df["High"], df["Low"], df["Close"])
df["adx"] = adx.adx()

ichi = IchimokuIndicator(df["High"], df["Low"])
df["ichimoku"] = ichi.ichimoku_base_line()

vwap = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"])
df["vwap"] = vwap.vwap

atr = AverageTrueRange(df["High"], df["Low"], df["Close"])
df["atr"] = atr.average_true_range()

df.dropna(inplace=True)

# ---------------- PATTERN FEATURES ---------------- #

def detect_fvg(i):
    if i < 2:
        return 0
    if df["Low"].iloc[i] > df["High"].iloc[i-2]:
        return 1
    if df["High"].iloc[i] < df["Low"].iloc[i-2]:
        return -1
    return 0

def detect_wedge(i):
    if i < 20:
        return 0
    closes = df["Close"].iloc[i-20:i].values
    x = np.arange(len(closes))
    slope, _, _, _, _ = linregress(x, closes)
    return slope

def detect_trendline_break(i):
    if i < 20:
        return 0
    closes = df["Close"].iloc[i-20:i].values
    x = np.arange(len(closes))
    slope, intercept, _, _, _ = linregress(x, closes)
    trendline = slope * x + intercept
    return 1 if closes[-1] < trendline[-1] else 0

df["fvg"] = [detect_fvg(i) for i in range(len(df))]
df["wedge"] = [detect_wedge(i) for i in range(len(df))]
df["trend_break"] = [detect_trendline_break(i) for i in range(len(df))]

# ---------------- LABEL DATA ---------------- #

LOOKAHEAD = 10  # minutes
TARGET_MULT = 1.5
STOP_MULT = 1

labels = []

for i in range(len(df) - LOOKAHEAD):
    entry = df["Close"].iloc[i]
    atr_val = df["atr"].iloc[i]

    target = entry + atr_val * TARGET_MULT
    stop = entry - atr_val * STOP_MULT

    future = df.iloc[i+1:i+LOOKAHEAD]

    hit_target = future["High"].max() >= target
    hit_stop = future["Low"].min() <= stop

    if hit_target and not hit_stop:
        labels.append(1)
    else:
        labels.append(0)

df = df.iloc[:-LOOKAHEAD]
df["label"] = labels

# ---------------- FEATURES ---------------- #

df["price_vwap"] = df["Close"] - df["vwap"]
df["price_ichimoku"] = df["Close"] - df["ichimoku"]
df["price_bb_low"] = df["Close"] - df["bb_low"]

features = df[[
    "price_vwap",
    "stoch",
    "adx",
    "price_ichimoku",
    "price_bb_low",
    "fvg",
    "wedge",
    "trend_break",
    "atr",
    "Volume"
]]

X = features.values
y = df["label"].values

# ---------------- TRAIN MODEL ---------------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

# ---------------- SAVE MODEL ---------------- #

joblib.dump(model, "model.pkl")
print("✅ model.pkl saved successfully")
