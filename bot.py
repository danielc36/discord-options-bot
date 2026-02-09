import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
import joblib

import pandas as pd
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

from market import get_stock_df

# ---------- CONFIG ---------- #

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

SYMBOL = "SPY"
TRADE_ACTIVE = False

model = joblib.load("model.pkl")
print("✅ ML model loaded")

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# ---------- MARKET HOURS ---------- #

def market_is_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    return now.hour >= 9 and (now.hour < 16 or (now.hour == 16 and now.minute == 0))

# ---------- INDICATORS ---------- #

def add_indicators(df):
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    vol = df["Volume"].squeeze()

    bb = BollingerBands(close)
    df["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()

    stoch = StochasticOscillator(high, low, close)
    df["stoch"] = stoch.stoch()

    atr = AverageTrueRange(high, low, close)
    df["atr"] = atr.average_true_range()

    df["std"] = close.rolling(20).std()

    vwap = VolumeWeightedAveragePrice(high, low, close, vol)
    df["vwap"] = vwap.vwap

    df.dropna(inplace=True)
    return df

# ---------- FEATURES ---------- #

def build_features(df1m, df15m):
    return [[
        df1m["stoch"].iloc[-1],
        df1m["bb_width"].iloc[-1],
        df1m["atr"].iloc[-1],
        df1m["std"].iloc[-1],
        df1m["vwap"].iloc[-1],
        df15m["adx"].iloc[-1],
        df15m["stoch"].iloc[-1],
        df15m["bb_width"].iloc[-1],
        df15m["std"].iloc[-1]
    ]]

# ---------- DIRECTION ---------- #

def determine_direction(df1m, df15m):
    price = df1m["Close"].iloc[-1]
    vwap = df1m["vwap"].iloc[-1]
    adx = df15m["adx"].iloc[-1]

    if price > vwap and adx > 20:
        return "BUY"
    elif price < vwap and adx > 20:
        return "SELL"
    return "HOLD"

# ---------- EMBED ---------- #

def build_embed(direction, price, prob):
    color = discord.Color.green() if direction == "BUY" else discord.Color.red()
    emoji = "🟢📈" if direction == "BUY" else "🔴📉"

    embed = discord.Embed(
        title=f"{emoji} SPY {direction}",
        color=color,
        timestamp=datetime.utcnow()
    )

    embed.add_field(name="Price", value=f"${round(price,2)}", inline=True)
    embed.add_field(name="ML Confidence", value=f"{round(prob*100,1)}%", inline=True)
    embed.set_footer(text="Educational use only")
    return embed

# ---------- CORE ---------- #

async def check_trade():
    global TRADE_ACTIVE

    df1m = get_stock_df("SPY", interval="1m")
    df15m = get_stock_df("SPY", interval="15m")

    df1m = add_indicators(df1m)
    df15m = add_indicators(df15m)

    adx = ADXIndicator(df15m["High"], df15m["Low"], df15m["Close"])
    df15m["adx"] = adx.adx()

    direction = determine_direction(df1m, df15m)

    features = build_features(df1m, df15m)
    prob = model.predict_proba(features)[0][1]

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    price = df1m["Close"].iloc[-1]

    if not TRADE_ACTIVE and direction in ["BUY", "SELL"] and prob > 0.65:
        embed = build_embed(direction, price, prob)
        await channel.send(embed=embed)
        TRADE_ACTIVE = True

    elif TRADE_ACTIVE and direction == "HOLD":
        await channel.send("⚠️ Exit Signal — trend conditions failed.")
        TRADE_ACTIVE = False

# ---------- LOOP ---------- #

@tasks.loop(minutes=1)
async def spy_loop():
    if market_is_open():
        try:
            await check_trade()
        except Exception as e:
            print("Loop error:", e)

@bot.event
async def on_ready():
    print("🚀 SPY ML BOT ONLINE")
    spy_loop.start()

bot.run(TOKEN)
