import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
from datetime import datetime
import pytz

from market import get_stock_df, get_option_chain
from strategy import analyze_trend
from options import filter_options
from confidence import confidence_score

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def market_is_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    if now.weekday() >= 5:  # Saturday/Sunday
        return False

    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return open_time <= now <= close_time


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.command()
async def play(ctx, symbol: str):

    if not market_is_open():
        await ctx.send("⏰ Market is closed. Try again during market hours (9:30am–4:00pm ET).")
        return

    symbol = symbol.upper()

    df = get_stock_df(symbol)
    if df.empty:
        await ctx.send("❌ No market data found.")
        return

    direction, df = analyze_trend(df)
    if direction == "NO TRADE":
        await ctx.send(f"⚠️ {symbol}: No high-quality setup.")
        return

    calls, puts, expiry = get_option_chain(symbol)
    price = df["Close"].iloc[-1]

    option = filter_options(
        calls if direction == "CALL" else puts,
        direction,
        price
    )

    if option is None:
        await ctx.send("⚠️ No liquid option contracts found.")
        return

    confidence = confidence_score(df, direction)

    await ctx.send(
        f"""
📈 **{symbol} {direction} Setup**
Expiration: {expiry}
Contract: `{option['contractSymbol']}`
Strike: {option['strike']}
Last Price: ${option['lastPrice']}
Volume / OI: {option['volume']} / {option['openInterest']}
Confidence: **{confidence}%**

⚠️ Educational use only
"""
    )


bot.run(os.getenv("DISCORD_TOKEN"))
