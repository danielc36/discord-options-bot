import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
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
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

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

# ---------- MARKET HOURS ---------- #

def market_is_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    return datetime.strptime("09:30", "%H:%M").time() <= now.time() <= datetime.strptime("16:00", "%H:%M").time()

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

    adx = ADXIndicator(high, low, close)
    df["adx"] = adx.adx()

    df["std"] = close.rolling(20).std()

    vwap = VolumeWeightedAveragePrice(high, low, close, vol)
    df["vwap"] = vwap.vwap

    df.dropna(inplace=True)
    return df

# ---------- FEATURES ---------- #

def build_features(df1m, df15m):
    data = {
        "stoch_1m": df1m["stoch"].iloc[-1],
        "bb_width_1m": df1m["bb_width"].iloc[-1],
        "atr_1m": df1m["atr"].iloc[-1],
        "std_1m": df1m["std"].iloc[-1],
        "vwap_1m": df1m["vwap"].iloc[-1],
        "adx_15m": df15m["adx"].iloc[-1],
        "stoch_15m": df15m["stoch"].iloc[-1],
        "bb_width_15m": df15m["bb_width"].iloc[-1],
        "std_15m": df15m["std"].iloc[-1]
    }

    return pd.DataFrame([data])

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

# ---------- EMBEDS ---------- #

def build_entry_embed(direction, price, prob, atr):
    color = discord.Color.green() if direction == "BUY" else discord.Color.red()
    emoji = "🟢📈" if direction == "BUY" else "🔴📉"

    if direction == "BUY":
        target = round(price + atr * 1.5, 2)
        stop = round(price - atr, 2)
    else:
        target = round(price - atr * 1.5, 2)
        stop = round(price + atr, 2)

    embed = discord.Embed(
        title=f"{emoji} SPY {direction}",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )

    embed.add_field(name="Price", value=f"${round(price,2)}", inline=True)
    embed.add_field(name="Target", value=f"🟢 ${target}", inline=True)
    embed.add_field(name="Stop Loss", value=f"🔴 ${stop}", inline=True)
    embed.add_field(name="ML Confidence", value=f"{round(prob*100,1)}%", inline=True)

    embed.set_footer(text="Educational use only")
    return embed


def build_exit_embed(reason, price):
    embed = discord.Embed(
        title="⚠️ SPY EXIT SIGNAL",
        color=discord.Color.orange(),
        timestamp=datetime.now(timezone.utc)
    )

    embed.add_field(name="Exit Price", value=f"${round(price,2)}", inline=True)
    embed.add_field(name="Reason", value=reason, inline=False)
    embed.set_footer(text="Educational use only")
    return embed

# ---------- CORE LOGIC ---------- #

async def check_trade():
    global TRADE_ACTIVE, LAST_DIRECTION

    df1m = get_stock_df(SYMBOL, interval="1m")
    df15m = get_stock_df(SYMBOL, interval="15m")

    df1m = add_indicators(df1m)
    df15m = add_indicators(df15m)

    price = df1m["Close"].iloc[-1]
    atr = df1m["atr"].iloc[-1]

    direction = determine_direction(df1m, df15m)

    prob = 0
    if model:
        features = build_features(df1m, df15m)
        prob = model.predict_proba(features)[0][1]

        if prob < 0.65:
            direction = "HOLD"

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    # ---------- ENTRY ---------- #
    if not TRADE_ACTIVE and direction in ["BUY", "SELL"]:
        embed = build_entry_embed(direction, price, prob, atr)
        await channel.send(embed=embed)
        TRADE_ACTIVE = True
        LAST_DIRECTION = direction
        return

    # ---------- EXIT ---------- #
    if TRADE_ACTIVE:
        exit_reason = None

        if direction == "HOLD":
            exit_reason = "Indicators lost confirmation"

        elif direction != LAST_DIRECTION:
            exit_reason = "Trend reversal detected"

        if model and prob < 0.55:
            exit_reason = "ML confidence dropped"

        if exit_reason:
            exit_embed = build_exit_embed(exit_reason, price)
            await channel.send(embed=exit_embed)
            TRADE_ACTIVE = False
            LAST_DIRECTION = None

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
