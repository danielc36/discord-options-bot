import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import os
from datetime import datetime, time, timezone
import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from scipy.stats import linregress
import joblib

from market import get_stock_df

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

# ---------------- LOAD ENV ---------------- #

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

SYMBOL = "SPY"
ACTIVE_TRADE = None
MODEL = joblib.load("model.pkl")  # ML model

eastern = pytz.timezone("US/Eastern")

# ---------------- MARKET HOURS ---------------- #

def market_is_open():
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    return (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and now.hour < 16

# ---------------- INDICATORS ---------------- #

def apply_indicators(df):
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

    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()

    df.dropna(inplace=True)
    return df

# ---------------- PATTERN DETECTION ---------------- #

def detect_fvg(df):
    if len(df) < 3:
        return 0
    if df["Low"].iloc[-1] > df["High"].iloc[-3]:
        return 1
    if df["High"].iloc[-1] < df["Low"].iloc[-3]:
        return -1
    return 0

def detect_wedge(df):
    closes = df["Close"].tail(20).values
    x = np.arange(len(closes))
    slope, _, _, _, _ = linregress(x, closes)
    return slope

def detect_trendline_break(df):
    closes = df["Close"].tail(20).values
    x = np.arange(len(closes))
    slope, intercept, _, _, _ = linregress(x, closes)
    trendline = slope * x + intercept
    return 1 if closes[-1] < trendline[-1] else 0

# ---------------- FEATURE ENGINEERING ---------------- #

def build_features(df1, df15):
    price = df1["Close"].iloc[-1]

    features = [
        price - df1["vwap"].iloc[-1],
        df1["stoch"].iloc[-1],
        df1["adx"].iloc[-1],
        price - df1["ichimoku"].iloc[-1],
        price - df1["bb_low"].iloc[-1],
        detect_fvg(df1),
        detect_wedge(df1),
        detect_trendline_break(df1),
        df15["adx"].iloc[-1],
        df15["Close"].iloc[-1] - df15["vwap"].iloc[-1],
        df1["atr"].iloc[-1],
        df1["Volume"].iloc[-1]
    ]

    return np.array(features).reshape(1, -1)

# ---------------- CONFLUENCE TEXT ---------------- #

def confluences(df):
    conf = []
    price = df["Close"].iloc[-1]

    if price > df["vwap"].iloc[-1]:
        conf.append("Above VWAP")
    if df["stoch"].iloc[-1] < 20:
        conf.append("Stoch Oversold")
    if df["adx"].iloc[-1] > 20:
        conf.append("Strong Trend (ADX)")
    if price > df["ichimoku"].iloc[-1]:
        conf.append("Above Ichimoku")
    if detect_fvg(df) == 1:
        conf.append("Bullish FVG")
    if detect_trendline_break(df):
        conf.append("Trendline Break")

    return conf

# ---------------- ENTRY CHECK ---------------- #

def check_entry():
    df1 = apply_indicators(get_stock_df(SYMBOL, interval="1m"))
    df15 = apply_indicators(get_stock_df(SYMBOL, interval="15m"))

    features = build_features(df1, df15)
    prob = MODEL.predict_proba(features)[0][1]  # win probability

    conf = confluences(df1)

    if prob >= 0.65 and len(conf) >= 3:
        return prob, conf, df1

    return None, [], df1

# ---------------- EXIT CHECK ---------------- #

def check_exit(df):
    price = df["Close"].iloc[-1]
    if price < df["vwap"].iloc[-1]:
        return True, "Lost VWAP"
    if df["adx"].iloc[-1] < 15:
        return True, "Trend Weakening"
    return False, ""

# ---------------- RISK / REWARD ---------------- #

def risk_reward(entry, atr):
    stop = round(entry - atr, 2)
    target = round(entry + atr * 2, 2)
    rr = round((target - entry) / (entry - stop), 2)
    return stop, target, rr

# ---------------- MAIN LOOP ---------------- #

@tasks.loop(minutes=1)
async def spy_loop():
    global ACTIVE_TRADE

    if not market_is_open():
        return

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    prob, conf, df = check_entry()
    price = round(df["Close"].iloc[-1], 2)
    atr = round(df["atr"].iloc[-1], 2)

    # ENTRY
    if ACTIVE_TRADE is None and prob:
        stop, target, rr = risk_reward(price, atr)

        ACTIVE_TRADE = "CALL"

        embed = discord.Embed(
            title="📊 SPY ENTRY",
            color=discord.Color.green(),
            timestamp=datetime.now(pytz.utc)
        )

        embed.add_field(name="Entry", value=f"${price}")
        embed.add_field(name="Stop", value=f"${stop}")
        embed.add_field(name="Target", value=f"${target}")
        embed.add_field(name="R:R", value=str(rr))
        embed.add_field(name="Win Probability", value=f"{round(prob*100,1)}%")

        embed.add_field(name="Confluences", value="\n".join(conf), inline=False)

        embed.set_footer(text="ML Confirmed Trade")

        await channel.send(embed=embed)

    # EXIT
    elif ACTIVE_TRADE:
        exit_trade, reason = check_exit(df)
        if exit_trade:
            embed = discord.Embed(
                title="🚪 SPY EXIT",
                description=reason,
                color=discord.Color.orange(),
                timestamp=datetime.now(pytz.utc)
            )
            await channel.send(embed=embed)
            ACTIVE_TRADE = None

# ---------------- COMMAND ---------------- #

@bot.command()
async def status(ctx):
    await ctx.send("🤖 ML SPY Bot is running")

# ---------------- READY ---------------- #

@bot.event
async def on_ready():
    print("🔥 ML SPY BOT ONLINE")
    spy_loop.start()

bot.run(TOKEN)
