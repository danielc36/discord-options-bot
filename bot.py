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
    return time(9,30) <= now.time() <= time(16,0)

# ---------------- INDICATORS ---------------- #

def calculate_support_resistance(df):
    return round(df["Low"].tail(20).min(),2), round(df["High"].tail(20).max(),2)

def supply_demand_zones(df):
    recent = df.tail(50)
    return round(recent["Low"].min(),2), round(recent["High"].max(),2)

def detect_fvg(df):
    for i in range(len(df)-2, 2, -1):
        if df["High"].iloc[i-2] < df["Low"].iloc[i]:
            return "Bullish FVG"
        if df["Low"].iloc[i-2] > df["High"].iloc[i]:
            return "Bearish FVG"
    return "None"

def detect_liquidity_sweep(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if last["Low"] < prev["Low"] and last["Close"] > prev["Low"]:
        return "Bullish Liquidity Sweep"
    if last["High"] > prev["High"] and last["Close"] < prev["High"]:
        return "Bearish Liquidity Sweep"
    return "None"

def detect_break_retest(df):
    support, resistance = calculate_support_resistance(df)
    last_close = df["Close"].iloc[-1]

    if last_close > resistance:
        return "Breakout"
    elif last_close < support:
        return "Breakdown"
    else:
        return "Retest Zone"

def trend_strength(df):
    highs = df["High"].tail(10)
    lows = df["Low"].tail(10)
    if highs.is_monotonic_increasing:
        return "Strong Bullish 📈"
    elif lows.is_monotonic_decreasing:
        return "Strong Bearish 📉"
    return "Neutral ⚖️"

def confluence_score(confidence, fvg, liquidity, trend):
    score = confidence
    if fvg != "None":
        score += 5
    if liquidity != "None":
        score += 5
    if "Strong" in trend:
        score += 5
    return min(score, 100)

def grade_trade(score):
    if score >= 90:
        return "A+ 🔥"
    elif score >= 80:
        return "A ✅"
    elif score >= 70:
        return "B ⚠️"
    else:
        return "C ❌"

def ai_summary(symbol, direction, trend, fvg, liquidity, break_retest, score):
    return (
        f"🤖 **AI Confirmation**\n"
        f"Trend: {trend}\n"
        f"Liquidity: {liquidity}\n"
        f"FVG: {fvg}\n"
        f"Structure: {break_retest}\n"
        f"Confluence Score: {score}/100\n"
        f"Bias favors {direction} setup."
    )

# ---------------- TRADE EMBED ---------------- #

async def generate_trade_embed(symbol):
    df = get_stock_df(symbol)
    if df.empty:
        return None

    direction, df = analyze_trend(df)
    if direction == "NO TRADE":
        return None

    calls, puts, expiry = get_option_chain(symbol)
    price = round(df["Close"].iloc[-1], 2)

    option = filter_options(calls if direction=="CALL" else puts, direction, price)
    if option is None:
        return None

    confidence = confidence_score(df, direction)
    if confidence < 70:
        return None

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range().iloc[-1]

    vwap = round(
        VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).vwap.iloc[-1],2
    )

    support, resistance = calculate_support_resistance(df)
    demand, supply = supply_demand_zones(df)
    fvg = detect_fvg(df)
    liquidity = detect_liquidity_sweep(df)
    break_retest = detect_break_retest(df)
    trend = trend_strength(df)

    score = confluence_score(confidence, fvg, liquidity, trend)
    grade = grade_trade(score)

    summary = ai_summary(symbol, direction, trend, fvg, liquidity, break_retest, score)

    if direction == "CALL":
        entry = price
        target = round(price + atr*1.5,2)
        stop = round(price - atr,2)
        emoji="🟢📈"
        label=f"${symbol} ${int(option['strike'])}C"
        color=discord.Color.green()
    else:
        entry = price
        target = round(price - atr*1.5,2)
        stop = round(price + atr,2)
        emoji="🔴📉"
        label=f"${symbol} ${int(option['strike'])}P"
        color=discord.Color.red()

    embed = discord.Embed(title=f"{emoji} {label}", color=color)

    embed.add_field(name="Trade Grade", value=grade, inline=True)
    embed.add_field(name="Confidence", value=f"{confidence}%", inline=True)
    embed.add_field(name="Confluence Score", value=f"{score}/100", inline=True)

    embed.add_field(name="VWAP", value=str(vwap), inline=True)
    embed.add_field(name="Entry", value=f"${entry}", inline=True)
    embed.add_field(name="Target", value=f"${target}", inline=True)
    embed.add_field(name="Stop Loss", value=f"${stop}", inline=True)

    embed.add_field(name="Support", value=f"${support}", inline=True)
    embed.add_field(name="Resistance", value=f"${resistance}", inline=True)
    embed.add_field(name="Demand Zone", value=f"${demand}", inline=True)
    embed.add_field(name="Supply Zone", value=f"${supply}", inline=True)

    embed.add_field(name="FVG", value=fvg, inline=True)
    embed.add_field(name="Liquidity Sweep", value=liquidity, inline=True)
    embed.add_field(name="Structure", value=break_retest, inline=True)
    embed.add_field(name="Trend Strength", value=trend, inline=False)
    embed.add_field(name="AI Confirmation", value=summary, inline=False)

    embed.timestamp = datetime.now(pytz.utc)
    embed.set_footer(text="⚠️ Educational use only")

    return embed

# ---------------- EVENTS ---------------- #

@bot.event
async def on_ready():
    print("🔥 PRO VERSION DEPLOYED 🔥")
    fast_alerts.start()
    slow_alerts.start()

# ---------------- COMMAND ---------------- #

@bot.command()
async def play(ctx, symbol: str):
    if not market_is_open():
        await ctx.send("⏰ Market closed.")
        return

    embed = await generate_trade_embed(symbol.upper())
    if embed:
        await ctx.send(embed=embed)
    else:
        await ctx.send("⚠️ No high-quality setup found.")

# ---------------- AUTO TASKS ---------------- #

@tasks.loop(minutes=15)
async def fast_alerts():
    if not market_is_open(): return
    channel = bot.get_channel(CHANNEL_ID)
    for symbol in FAST_WATCHLIST:
        embed = await generate_trade_embed(symbol)
        if embed:
            await channel.send(embed=embed)

@tasks.loop(minutes=30)
async def slow_alerts():
    if not market_is_open(): return
    channel = bot.get_channel(CHANNEL_ID)
    for symbol in SLOW_WATCHLIST:
        embed = await generate_trade_embed(symbol)
        if embed:
            await channel.send(embed=embed)

bot.run(TOKEN)
