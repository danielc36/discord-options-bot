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
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded")
except Exception as e:
    model = None
    print("⚠️ model.pkl not found — running without ML filter")
    print("Error:", e)

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

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice


def add_indicators_1m(df):
    bb = BollingerBands(close=df["Close"])
    df["bb_width_1m"] = bb.bollinger_hband() - bb.bollinger_lband()

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"])
    df["atr_1m"] = atr.average_true_range()

    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["stoch_1m"] = stoch.stoch()

    vwap = VolumeWeightedAveragePrice(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"]
    )
    df["vwap_1m"] = vwap.vwap

    df["std_1m"] = df["Close"].rolling(20).std()

    df.dropna(inplace=True)
    return df


def add_indicators_15m(df):
    bb = BollingerBands(close=df["Close"])
    df["bb_width_15m"] = bb.bollinger_hband() - bb.bollinger_lband()

    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["stoch_15m"] = stoch.stoch()

    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["adx_15m"] = adx.adx()

    df["std_15m"] = df["Close"].rolling(20).std()

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
    vwap = df1m["vwap_1m"].iloc[-1]
    adx = df15m["adx_15m"].iloc[-1]

    if price > vwap and adx > 20:
        return "BUY"
    elif price < vwap and adx > 20:
        return "SELL"
    else:
        return "NO TRADE"

# ---------------- ML FEATURES ---------------- #

def build_features(df1m, df15m):
    return [[
        df1m["stoch_1m"].iloc[-1],
        df1m["bb_width_1m"].iloc[-1],
        df1m["atr_1m"].iloc[-1],
        df1m["std_1m"].iloc[-1],
        df1m["vwap_1m"].iloc[-1],
        df15m["adx_15m"].iloc[-1],
        df15m["stoch_15m"].iloc[-1],
        df15m["bb_width_15m"].iloc[-1],
        df15m["std_15m"].iloc[-1]
    ]]


# ---------------- CONFLUENCES ---------------- #

def confluences(df1m, df15m):
    conflu = []

    if df1m["Close"].iloc[-1] > df1m["vwap_1m"].iloc[-1]:
        conflu.append("Above VWAP (1m)")

    if df15m["adx_15m"].iloc[-1] > 25:
        conflu.append("Strong Trend (15m ADX)")

    if df1m["stoch_1m"].iloc[-1] < 20:
        conflu.append("Oversold (1m Stoch)")

    if df1m["stoch_1m"].iloc[-1] > 80:
        conflu.append("Overbought (1m Stoch)")

    return conflu

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

    df1m = add_indicators_1m(df1m)
    df15m = add_indicators_15m(df15m)

    features = build_features(df1m, df15m)
    prob = model.predict_proba(features)[0][1]

    direction = determine_direction(df1m, df15m)

    exit_reasons = []

    # ---------- ML FILTER ----------
    if model:
        features = build_features(df1m, df15m)
        prob = model.predict_proba([features])[0][1]
        if prob < 0.65:
            direction = "NO TRADE"
            exit_reasons.append("ML confidence dropped")

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    price = round(df1m["Close"].iloc[-1], 2)

    # ---------- ENTRY ----------
    if not TRADE_ACTIVE and direction in ["CALL", "PUT"]:
        embed = build_embed(direction, df1m, df15m)
        await channel.send(content="🟢 **BUY SIGNAL**", embed=embed)

        TRADE_ACTIVE = True
        LAST_DIRECTION = direction
        return

    # ---------- HOLD ----------
    if TRADE_ACTIVE and direction == LAST_DIRECTION:
        # Do nothing (still valid trade)
        return

    # ---------- EXIT ----------
    if TRADE_ACTIVE and (direction == "NO TRADE" or direction != LAST_DIRECTION):

        # Build exit reason
        if direction == "NO TRADE":
            exit_reasons.append("Indicators lost confluence")
        else:
            exit_reasons.append("Trend reversal detected")

        exit_text = (
            f"🔴 **SELL SIGNAL – {LAST_DIRECTION}**\n"
            f"Price: ${price}\n"
            f"Reason:\n• " + "\n• ".join(exit_reasons)
        )

        await channel.send(exit_text)

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
