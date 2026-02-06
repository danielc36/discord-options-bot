def confidence_score(df, direction):
    score = 0
    last = df.iloc[-1]

    # Trend
    if direction == "CALL" and last["ema9"] > last["ema21"]:
        score += 30
    if direction == "PUT" and last["ema9"] < last["ema21"]:
        score += 30

    # RSI strength
    if 55 < last["rsi"] < 70 or 30 < last["rsi"] < 45:
        score += 25

    # Volatility proxy
    if df["Close"].pct_change().std() > 0.002:
        score += 25

    # Structure
    if abs(last["Close"] - df["Close"].mean()) / last["Close"] < 0.01:
        score += 20

    return min(score, 100)
