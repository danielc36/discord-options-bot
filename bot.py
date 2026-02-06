import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import os
from datetime import datetime, time, timezone
import pytz
import matplotlib.pyplot as plt

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

    open_time = time(9, 30)
    close_time = time(16, 0)

    return open_time <= now.time() <= close_time

# ---------------- INDICATORS ---------------- #

def calculate_support_resistance(df):
    support = round(df["Low"].tail(20).min(), 2)
    resistance = round(df["High"].tail(20).max(), 2)
    return support, resistance


def ai_summary(symbol, direction, price, vwap, support, resistance, confidence):
    if direction == "CALL":
        return (
            f"🤖 {symbol} above VWAP with bullish momentum. "
            f"Support at ${support} holding, targeting resistance ${resistance}. "
            f"Confidence {confidence}% confirms CALL setup."
        )
    else:
        return (
            f"🤖 {symbol} below VWAP with bearish momentum. "
            f"Resistance at ${resistance} holding, targeting support ${support}. "
            f"Confidence {confidence}% confirms PUT setup."
        )


def trend_strength_score(price, vwap, atr, confidence):
    distance = abs(price - vwap)
    strength = (distance / atr) * 50 + confidence * 0.5
    return min(round(strength, 1), 100)


def generate_chart(symbol, df, support, resistance, vwap):
    plt.figure(figsize=(6,4))
    prices = df["Close"].tail(50)

    plt.plot(prices, label="Price", linewidth=2)
    plt.axhline(support, linestyle="--", label="Support")
    plt.axhline(resistance, linestyle="--", label="Resistance")
    plt.axhline(vwap, linestyle="--", label="VWAP")

    plt.title(symbol)
    plt.legend()
    plt.tight_layout()

    filename = f"{symbol}_chart.png"
    plt.savefig(filename)
    plt.close()

    return filename

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

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ).average_true_range().iloc[-1]

    vwap_indicator = VolumeWeightedAveragePrice(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"]
    )
    vwap = round(vwap_indicator.vwap.iloc[-1], 2)

    support, resistance = calculate_support_resistance(df)
    summary = ai_summary(symbol, direction, price, vwap, support, resistance, confidence)
    strength = trend_strength_score(price, vwap, atr, confidence)

    chart_file = generate_chart(symbol, df, support, resistance, vwap)
    file = discord.File(chart_file, filename="chart.png")

    if direction == "CALL":
        entry = round(price, 2)
        target = round(price + atr * 1.5, 2)
        stop = round(price - atr, 2)
        emoji = "🟢📈"
        option_label = f"${symbol} ${int(option['strike'])}C"
        color = discord.Color.green()
    else:
        entry = round(price, 2)
        target = round(price - atr * 1.5, 2)
        stop = round(price + atr, 2)
        emoji = "🔴📉"
        option_label = f"${symbol} ${int(option['strike'])}P"
        color = discord.Color.red()

    embed = discord.Embed(title=f"{emoji} {option_label}", color=color)
    embed.timestamp = datetime.now(timezone.utc)

    embed.add_field(name="Expiration", value=str(expiry), inline=True)
    embed.add_field(name="Confidence", value=f"{confidence}%", inline=True)
    embed.add_field(name="Trend Strength", value=f"{strength}/100", inline=True)

    embed.add_field(name="VWAP", value=f"${vwap}", inline=True)
    embed.add_field(name="Best Entry", value=f"${entry}", inline=True)
    embed.add_field(name="Target", value=f"${target}", inline=True)
    embed.add_field(name="Stop Loss", value=f"${stop}", inline=True)

    embed.add_field(name="Support", value=f"${support}", inline=True)
    embed.add_field(name="Resistance", value=f"${resistance}", inline=True)
    embed.add_field(name="ATR", value=round(atr, 2), inline=True)

    embed.add_field(name="AI Confirmation", value=summary, inline=False)

    embed.set_image(url="attachment://chart.png")
    embed.set_footer(text="⚠️ Educational use only")

    return embed, file

# ---------------- EVENTS ---------------- #

@bot.event
async def on_ready():
    print("🔥 BOT ONLINE 🔥")
    print(f"Logged in as {bot.user}")
    fast_alerts.start()
    slow_alerts.start()

# ---------------- COMMAND ---------------- #

@bot.command()
async def play(ctx, symbol: str):
    if not market_is_open():
        await ctx.send("⏰ Market is closed (9:30am–4:00pm ET).")
        return

    result = await generate_trade_embed(symbol.upper())
    if result:
        embed, file = result
        await ctx.send(embed=embed, file=file)
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
        result = await generate_trade_embed(symbol)
        if result:
            embed, file = result
            await channel.send(embed=embed, file=file)


@tasks.loop(minutes=30)
async def slow_alerts():
    if not market_is_open():
        return

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    for symbol in SLOW_WATCHLIST:
        result = await generate_trade_embed(symbol)
        if result:
            embed, file = result
            await channel.send(embed=embed, file=file)

bot.run(TOKEN)
