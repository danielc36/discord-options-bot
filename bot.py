import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import pytz
import ta

from market import get_stock_df, get_option_chain
from strategy import analyze_trend
from options import filter_options
from confidence import confidence_score

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

ALERT_CHANNEL_ID = 1464049148792799390

FAST_SYMBOLS = ["SPY", "QQQ", "IWM"]
SLOW_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

last_alert_time = {}

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ================= MARKET HOURS =================
def market_is_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    if now.weekday() >= 5:
        return False

    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return open_time <= now <= close_time

# ================= COOLDOWN =================
def can_send(symbol, minutes):
    now = datetime.utcnow()
    if symbol not in last_alert_time:
        last_alert_time[symbol] = now
        return True

    if now - last_alert_time[symbol] >= timedelta(minutes=minutes):
        last_alert_time[symbol] = now
        return True

    return False

# ================= TRADE GENERATOR =================
async def generate_trade_embed(symbol):
    df = get_stock_df(symbol)
    if df.empty:
        return None

    direction, df = analyze_trend(df)
    if direction == "NO TRADE":
        return None

    calls, puts, expiry = get_option_chain(symbol)
    price = df["Close"].iloc[-1]

    option = filter_options(calls if direction=="CALL" else puts, direction, price)
    if option is None:
        return None

    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        df["High"], df["Low"], df["Close"], df["Volume"]
    ).vwap()

    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"]
    ).average_true_range()

    vwap = round(df["vwap"].iloc[-1], 2)
    atr = round(df["atr"].iloc[-1], 2)

    support = round(df["Low"].rolling(20).min().iloc[-1], 2)
    resistance = round(df["High"].rolling(20).max().iloc[-1], 2)

    if direction == "CALL":
        entry = vwap
        target = round(price + atr*2, 2)
        stop = round(price - atr, 2)
        emoji = "🟢📈"
        color = discord.Color.green()
        contract_type = "C"
    else:
        entry = vwap
        target = round(price - atr*2, 2)
        stop = round(price + atr, 2)
        emoji = "🔴📉"
        color = discord.Color.red()
        contract_type = "P"

    strike = option["strike"]
    contract_display = f"${symbol} ${strike}{contract_type}"

    confidence = confidence_score(df, direction)

    if confidence < 70:
        return None

    embed = discord.Embed(
        title=f"{emoji} {symbol} {direction} Setup (AUTO)",
        color=color,
        timestamp=datetime.utcnow()
    )

    embed.add_field(name="💰 Contract", value=contract_display, inline=False)
    embed.add_field(name="🎯 Entry", value=f"${entry}", inline=True)
    embed.add_field(name="✅ Target", value=f"${target}", inline=True)
    embed.add_field(name="🛑 Stop", value=f"${stop}", inline=True)

    embed.add_field(name="📊 Support", value=f"${support}", inline=True)
    embed.add_field(name="📈 Resistance", value=f"${resistance}", inline=True)

    embed.add_field(name="📐 ATR", value=str(atr), inline=True)
    embed.add_field(name="📊 VWAP", value=str(vwap), inline=True)
    embed.add_field(name="🔥 Confidence", value=f"{confidence}%", inline=True)

    embed.set_footer(text="Educational use only")

    return embed

# ================= COMMAND =================
@bot.command()
async def play(ctx, symbol: str):
    if not market_is_open():
        await ctx.send("⏰ Market is closed.")
        return

    embed = await generate_trade_embed(symbol.upper())
    if embed is None:
        await ctx.send("⚠️ No trade setup found.")
        return

    await ctx.send(embed=embed)

# ================= FAST LOOP (15 MIN) =================
@tasks.loop(minutes=15)
async def fast_alerts():
    if not market_is_open():
        return

    channel = bot.get_channel(ALERT_CHANNEL_ID)
    if not channel:
        return

    for symbol in FAST_SYMBOLS:
        if not can_send(symbol, 15):
            continue

        embed = await generate_trade_embed(symbol)
        if embed:
            await channel.send(embed=embed)

# ================= SLOW LOOP (30 MIN) =================
@tasks.loop(minutes=30)
async def slow_alerts():
    if not market_is_open():
        return

    channel = bot.get_channel(ALERT_CHANNEL_ID)
    if not channel:
        return

    for symbol in SLOW_SYMBOLS:
        if not can_send(symbol, 30):
            continue

        embed = await generate_trade_embed(symbol)
        if embed:
            await channel.send(embed=embed)

# ================= EVENTS =================
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    fast_alerts.start()
    slow_alerts.start()

bot.run(DISCORD_TOKEN)
