import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import os
from datetime import datetime, time, timezone
import pytz
import pandas as pd

from market import get_stock_df, get_option_chain
from strategy import analyze_trend
from options import filter_options
from confidence import confidence_score

from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from scipy.stats import linregress

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

FAST_WATCHLIST = ["SPY", "QQQ", "IWM"]
SLOW_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------- MARKET HOURS ---------------- #

def market_is_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    if now.weekday() >= 5:
        return False

    open_time = time(9, 30)
    close_time = time(16, 0)

    return open_time <= now.time() <= close_time

# ---------------- ADVANCED INDICATORS ---------------- #

def calculate_support_resistance(df):
    support = round(df["Low"].tail(20).min(), 2)
    resistance = round(df["High"].tail(20).max(), 2)
    return support, resistance

def detect_fvg(df):
    for i in range(2, len(df)):
        high1 = df["High"].iloc[i-2]
        low3 = df["Low"].iloc[i]
        if high1 < low3:
            return "Bullish FVG"
        low1 = df["Low"].iloc[i-2]
        high3 = df["High"].iloc[i]
        if low1 > high3:
            return "Bearish FVG"
    return None

def trend_strength(df):
    highs = df["High"].tail(20).values
    lows = df["Low"].tail(20).values

    high_slope = linregress(range(len(highs)), highs).slope
    low_slope = linregress(range(len(lows)), lows).slope

    if high_slope > 0 and low_slope > 0:
        return "Strong Bullish"
    elif high_slope < 0 and low_slope < 0:
        return "Strong Bearish"
    else:
        return "Neutral"

# ---------------- MAIN EMBED ---------------- #

async def generate_trade_embed(symbol):
    df = get_stock_df(symbol)
    if df.empty:
        return None

    direction, df = analyze_trend(df)
    if direction == "NO TRADE":
        return None

    calls, puts, expiry = get_option_chain(symbol)
    price = round(df["Close"].iloc[-1], 2)

    option = filter_options(
        calls if direction == "CALL" else puts,
        direction,
        price
    )

    if option is None:
        return None

    confidence = confidence_score(df, direction)
    if confidence < 70:
        return None

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range().iloc[-1]

    vwap_indicator = VolumeWeightedAveragePrice(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    )
    vwap = round(vwap_indicator.vwap.iloc[-1], 2)

    support, resistance = calculate_support_resistance(df)
    fvg = detect_fvg(df)
    trend = trend_strength(df)

    # Trade levels
    if direction == "CALL":
        entry = price
        target = round(price + atr * 1.5, 2)
        stop = round(price - atr, 2)
        emoji = "🟢📈"
        option_label = f"${symbol} ${int(option['strike'])}C"
    else:
        entry = price
        target = round(price - atr * 1.5, 2)
        stop = round(price + atr, 2)
        emoji = "🔴📉"
        option_label = f"${symbol} ${int(option['strike'])}P"

    # Build confluence list
    confluences = []

    if price > vwap and direction == "CALL":
        confluences.append("Above VWAP")
    if price < vwap and direction == "PUT":
        confluences.append("Below VWAP")

    if fvg:
        confluences.append(fvg)

    confluences.append(f"Trend: {trend}")

    if direction == "CALL":
        confluences.append(f"Demand zone near ${support}")
    else:
        confluences.append(f"Supply zone near ${resistance}")

    confluence_text = "\n".join([f"• {c}" for c in confluences[:4]])

    embed = discord.Embed(
        title=f"{emoji} {option_label}",
        color=discord.Color.green() if direction == "CALL" else discord.Color.red()
    )

    embed.add_field(name="Entry", value=f"${entry}", inline=True)
    embed.add_field(name="Target", value=f"${target}", inline=True)
    embed.add_field(name="Stop", value=f"${stop}", inline=True)
    embed.add_field(name="Confidence", value=f"{confidence}%", inline=True)

    embed.add_field(name="Confluence", value=confluence_text, inline=False)

    embed.add_field(
        name="AI Confirmation",
        value=f"High probability {direction} setup based on multi-indicator confluence.",
        inline=False
    )

    embed.set_footer(text="⚠️ Educational use only")
    embed.timestamp = datetime.now(timezone.utc)

    return embed

# ---------------- EVENTS ---------------- #

@bot.event
async def on_ready():
    print("🔥 BOT ONLINE 🔥")
    fast_alerts.start()
    slow_alerts.start()

# ---------------- COMMAND ---------------- #

@bot.command()
async def play(ctx, symbol: str):
    if not market_is_open():
        await ctx.send("⏰ Market is closed (9:30am–4:00pm ET).")
        return

    embed = await generate_trade_embed(symbol.upper())
    if embed:
        await ctx.send(embed=embed)
    else:
        await ctx.send(f"⚠️ No high-quality setup for {symbol.upper()}")

# ---------------- AUTO TASKS ---------------- #

@tasks.loop(minutes=15)
async def fast_alerts():
    if not market_is_open():
        return

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    for symbol in FAST_WATCHLIST:
        embed = await generate_trade_embed(symbol)
        if embed:
            await channel.send(embed=embed)

@tasks.loop(minutes=30)
async def slow_alerts():
    if not market_is_open():
        return

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    for symbol in SLOW_WATCHLIST:
        embed = await generate_trade_embed(symbol)
        if embed:
            await channel.send(embed=embed)

bot.run(TOKEN)
