import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
import pytz
import joblib
import pandas as pd
import numpy as np

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

from market import get_stock_df

# ---------------- CONFIG ---------------- #

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

SYMBOL = "SPY"
TRADE_ACTIVE = False
LAST_DIRECTION = None

# Load ML model
try:
    model = joblib.load("model.pkl")
    print("✅ ML model loaded")
except:
    model = None
    print("⚠️ model.pkl not found — running without ML filter")

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------- MARKET HOURS ---------------- #

def market_is_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    if now.weekday() >= 5:
        return False

    return datetime.strptime("09:30", "%H:%M").time() <= now.time() <= datetime.strptime("16:00", "%H:%M").time()

# ---------------- INDICATORS ---------------- #

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

    df["std"] = df["Close"].rolling(20).std()

    ichi = IchimokuIndicator(df["High"], df["Low"])
    df["tenkan"] = ichi.ichimoku_conversion_line()
    df["kijun"] = ichi.ichimoku_base_line()

    df.dropna(inplace=True)
    return df

# ---------------- PATTERN DETECTION ---------------- #

def detect_patterns(df):
    patterns = []
    prices = df["Close"].values[-50:]

    if len(prices) < 20:
        return patterns

    # Peaks & troughs
    highs = prices[np.r_[True, prices[1:] > prices[:-1]] & np.r_[prices[:-1] > prices[1:], True]]
    lows = prices[np.r_[True, prices[1:] < prices[:-1]] & np.r_[prices[:-1] < prices[1:], True]]

    # Double Top
    if len(highs) >= 2 and abs(highs[-1] - highs[-2]) / highs[-1] < 0.002:
        patterns.append("Double Top")

    # Double Bottom
    if len(lows) >= 2 and abs(lows[-1] - lows[-2]) / lows[-1] < 0.002:
        patterns.append("Double Bottom")

    # Head and Shoulders
    if len(highs) >= 3:
        if highs[-2] > highs[-1] and highs[-2] > highs[-3]:
            patterns.append("Head & Shoulders")

    # Rounding Bottom
    if np.polyfit(range(len(prices)), prices, 2)[0] > 0:
        patterns.append("Rounding Bottom")

    # Cup and Handle (simplified)
    if prices[-1] > prices.mean() and np.min(prices) == prices[len(prices)//2]:
        patterns.append("Cup & Handle")

    # Wedges / Triangles via trend slope
    slope = np.polyfit(range(len(prices)), prices, 1)[0]

    if abs(slope) < 0.001:
        patterns.append("Symmetrical Triangle")
    elif slope > 0.002:
        patterns.append("Ascending Triangle")
    elif slope < -0.002:
        patterns.append("Descending Triangle")

    # Flags / Pennants
    if df["atr"].iloc[-1] < df["atr"].rolling(20).mean().iloc[-1]:
        patterns.append("Flag / Pennant")

    return patterns

# ---------------- TREND LOGIC ---------------- #

def determine_direction(df1m, df15m):
    price = df1m["Close"].iloc[-1]
    vwap = df1m["vwap"].iloc[-1]
    adx = df15m["adx"].iloc[-1]

    if price > vwap and adx > 20:
        return "CALL"
    elif price < vwap and adx > 20:
        return "PUT"
    else:
        return "NO TRADE"

# ---------------- ML FEATURES ---------------- #

def build_features(df1m, df15m):
    return [
        df1m["stoch"].iloc[-1],
        df1m["bb_width"].iloc[-1],
        df1m["atr"].iloc[-1],
        df1m["std"].iloc[-1],
        df1m["vwap"].iloc[-1],
        df15m["adx"].iloc[-1],
        df15m["stoch"].iloc[-1],
        df15m["bb_width"].iloc[-1],
        df15m["std"].iloc[-1]
    ]

# ---------------- CONFLUENCES ---------------- #

def confluences(df1m, df15m):
    conf = []

    if df1m["Close"].iloc[-1] > df1m["vwap"].iloc[-1]:
        conf.append("Above VWAP")
    else:
        conf.append("Below VWAP")

    if df15m["adx"].iloc[-1] > 25:
        conf.append("Strong Trend (ADX)")

    if df1m["stoch"].iloc[-1] < 20:
        conf.append("Oversold (Stoch)")

    if df1m["stoch"].iloc[-1] > 80:
        conf.append("Overbought (Stoch)")

    patterns = detect_patterns(df1m)
    for p in patterns:
        conf.append(f"Pattern: {p}")

    return conf

# ---------------- EMBED ---------------- #

def build_embed(direction, df1m, df15m):
    price = round(df1m["Close"].iloc[-1], 2)
    atr = df1m["atr"].iloc[-1]
    time_now = datetime.now(timezone.utc)

    if direction == "CALL":
        target = round(price + atr * 1.5, 2)
        stop = round(price - atr, 2)
        emoji = "🟢📈"
        color = discord.Color.green()
    else:
        target = round(price - atr * 1.5, 2)
        stop = round(price + atr, 2)
        emoji = "🔴📉"
        color = discord.Color.red()

    conf = confluences(df1m, df15m)

    embed = discord.Embed(
        title=f"{emoji} SPY {direction}",
        color=color,
        timestamp=time_now
    )

    embed.add_field(name="Price", value=f"${price}", inline=True)
    embed.add_field(name="Target", value=f"${target}", inline=True)
    embed.add_field(name="Stop", value=f"${stop}", inline=True)

    embed.add_field(
        name="Confluences",
        value="• " + "\n• ".join(conf[:12]),
        inline=False
    )

    embed.set_footer(text="Educational use only")
    return embed

# ---------------- CORE LOGIC ---------------- #

async def check_trade():
    global TRADE_ACTIVE, LAST_DIRECTION

    df1m = get_stock_df(SYMBOL, interval="1m")
    df15m = get_stock_df(SYMBOL, interval="15m")

    df1m = add_indicators(df1m)
    df15m = add_indicators(df15m)

    direction = determine_direction(df1m, df15m)

    if model:
        features = build_features(df1m, df15m)
        prob = model.predict_proba([features])[0][1]
        if prob < 0.65:
            direction = "NO TRADE"

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    if not TRADE_ACTIVE and direction in ["CALL", "PUT"]:
        embed = build_embed(direction, df1m, df15m)
        await channel.send(embed=embed)
        TRADE_ACTIVE = True
        LAST_DIRECTION = direction

    elif TRADE_ACTIVE and direction == "NO TRADE":
        await channel.send("⚠️ Trade exit signal — pattern or trend invalidated.")
        TRADE_ACTIVE = False
        LAST_DIRECTION = None

# ---------------- TASK LOOP ---------------- #

@tasks.loop(minutes=1)
async def spy_loop():
    if market_is_open():
        try:
            await check_trade()
        except Exception as e:
            print("Loop error:", e)

# ---------------- EVENTS ---------------- #

@bot.event
async def on_ready():
    print("🚀 SPY PATTERN BOT ONLINE")
    spy_loop.start()

bot.run(TOKEN)
