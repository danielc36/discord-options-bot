import discord
from discord.ext import commands
from dotenv import load_dotenv
import os

from market import get_stock_df, get_option_chain
from strategy import analyze_trend
from options import filter_options
from confidence import confidence_score

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.command()
async def play(ctx, symbol: str):
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
