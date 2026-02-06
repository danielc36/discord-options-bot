import ta

def analyze_trend(df):
    df["ema9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)

    last = df.iloc[-1]

    if last["ema9"] > last["ema21"] and last["rsi"] > 55:
        return "CALL", df
    elif last["ema9"] < last["ema21"] and last["rsi"] < 45:
        return "PUT", df
    else:
        return "NO TRADE", df
