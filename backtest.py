import pandas as pd
import numpy as np
from market import get_stock_df
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

SYMBOL = "SPY"

def add_indicators(df):
    bb = BollingerBands(df["Close"])
    df["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()

    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["adx"] = adx.adx()

    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["stoch"] = stoch.stoch()

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"])
    df["atr"] = atr.average_true_range()

    vwap = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"])
    df["vwap"] = vwap.vwap

    df.dropna(inplace=True)
    return df

def determine_direction(df1m, df15m):
    price = df1m["Close"]
    vwap = df1m["vwap"]
    adx = df15m["adx"].iloc[-1]

    if price.iloc[-1] > vwap.iloc[-1] and adx > 20:
        return "CALL"
    elif price.iloc[-1] < vwap.iloc[-1] and adx > 20:
        return "PUT"
    else:
        return "NO TRADE"

def backtest():
    df1m = get_stock_df(SYMBOL, interval="1m")
    df15m = get_stock_df(SYMBOL, interval="15m")

    df1m = add_indicators(df1m)
    df15m = add_indicators(df15m)

    trade_active = False
    entry_price = 0
    direction = None

    wins = 0
    losses = 0
    pnl = []

    for i in range(30, len(df1m)):
        slice1m = df1m.iloc[:i]
        slice15m = df15m.iloc[:max(1, i//15)]

        signal = determine_direction(slice1m, slice15m)

        price = slice1m["Close"].iloc[-1]
        atr = slice1m["atr"].iloc[-1]

        if not trade_active and signal in ["CALL", "PUT"]:
            trade_active = True
            entry_price = price
            direction = signal

        elif trade_active and signal == "NO TRADE":
            if direction == "CALL":
                result = price - entry_price
            else:
                result = entry_price - price

            pnl.append(result)

            if result > 0:
                wins += 1
            else:
                losses += 1

            trade_active = False
            direction = None

    total_trades = wins + losses
    winrate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = round(sum(pnl), 2)
    avg_trade = round(np.mean(pnl), 2) if pnl else 0

    print("====== BACKTEST RESULTS ======")
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {round(winrate,2)}%")
    print(f"Total P/L: ${total_pnl}")
    print(f"Avg Trade: ${avg_trade}")

if __name__ == "__main__":
    backtest()
