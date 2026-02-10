import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import pytz
import joblib
import pandas as pd

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

from market import get_stock_df

# ================= CONFIG ================= #

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

SYMBOL = "SPY"

# ================= IMPROVED STATE TRACKING ================= #
TRADE_ACTIVE = False
LAST_DIRECTION = None
ENTRY_PRICE = None
ENTRY_TIME = None
LAST_EXIT_TIME = None
HOLD_COUNTER = 0  # Count consecutive HOLD signals before exiting

# Thresholds
MIN_ENTRY_CONFIDENCE = 0.65
MIN_HOLD_CONFIDENCE = 0.50  # Lower threshold to stay in trade
COOLDOWN_MINUTES = 3  # Wait after exit before new entry
HOLD_SIGNALS_TO_EXIT = 3  # Need 3 consecutive HOLDs to exit

model = joblib.load("model.pkl")
print("✅ ML model loaded")

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# ================= MARKET HOURS ================= #

def market_is_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    return datetime.strptime("09:30", "%H:%M").time() <= now.time() <= datetime.strptime("16:00", "%H:%M").time()

# ================= INDICATORS ================= #

def add_indicators(df):
    if len(df) < 30:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

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

    if len(df) < 5:
        return None

    return df

# ================= FEATURES ================= #

def build_features(df1m, df15m):
    return pd.DataFrame([{
        "stoch_1m": df1m["stoch"].iloc[-1],
        "bb_width_1m": df1m["bb_width"].iloc[-1],
        "atr_1m": df1m["atr"].iloc[-1],
        "std_1m": df1m["std"].iloc[-1],
        "vwap_1m": df1m["vwap"].iloc[-1],
        "adx_15m": df15m["adx"].iloc[-1],
        "stoch_15m": df15m["stoch"].iloc[-1],
        "bb_width_15m": df15m["bb_width"].iloc[-1],
        "std_15m": df15m["std"].iloc[-1]
    }])

# ================= DIRECTION ================= #

def determine_direction(df1m, df15m):
    price = df1m["Close"].iloc[-1]
    vwap = df1m["vwap"].iloc[-1]
    adx = df15m["adx"].iloc[-1]
    stoch_1m = df1m["stoch"].iloc[-1]
    
    # Stronger confirmation required
    if price > vwap and adx > 25 and stoch_1m < 80:  # Not overbought
        return "BUY"
    elif price < vwap and adx > 25 and stoch_1m > 20:  # Not oversold
        return "SELL"
    return "HOLD"

# ================= EMBEDS ================= #

def build_entry_embed(direction, price, prob, atr):
    color = discord.Color.green() if direction == "BUY" else discord.Color.red()
    emoji = "🟢📈" if direction == "BUY" else "🔴📉"

    target = round(price + atr * 1.5, 2) if direction == "BUY" else round(price - atr * 1.5, 2)
    stop = round(price - atr, 2) if direction == "BUY" else round(price + atr, 2)

    embed = discord.Embed(
        title=f"{emoji} SPY {direction}",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )

    embed.add_field(name="Entry Price", value=f"${price:.2f}", inline=True)
    embed.add_field(name="Target", value=f"${target}", inline=True)
    embed.add_field(name="Stop Loss", value=f"${stop}", inline=True)
    embed.add_field(name="ML Confidence", value=f"{prob*100:.1f}%", inline=True)

    embed.set_footer(text="Educational use only • Position OPENED")
    return embed


def build_exit_embed(reason, price, pnl=None):
    embed = discord.Embed(
        title="🔔 SPY EXIT SIGNAL",
        color=discord.Color.orange(),
        timestamp=datetime.now(timezone.utc)
    )
    embed.add_field(name="Exit Price", value=f"${price:.2f}", inline=True)
    if pnl is not None:
        pnl_emoji = "📈" if pnl > 0 else "📉"
        embed.add_field(name="P/L", value=f"{pnl_emoji} ${pnl:.2f}", inline=True)
    embed.add_field(name="Reason", value=reason, inline=False)
    embed.set_footer(text="Position CLOSED • Wait for next signal")
    return embed

# ================= CORE ================= #

async def check_trade():
    global TRADE_ACTIVE, LAST_DIRECTION, ENTRY_PRICE, ENTRY_TIME, LAST_EXIT_TIME, HOLD_COUNTER

    df1m = get_stock_df(SYMBOL, interval="1m")
    df15m = get_stock_df(SYMBOL, interval="15m")

    df1m = add_indicators(df1m)
    df15m = add_indicators(df15m)

    if df1m is None or df15m is None:
        print("⚠️ Not enough data yet")
        return

    price = df1m["Close"].iloc[-1]
    atr = df1m["atr"].iloc[-1]

    # Calculate ML confidence first
    prob = 0
    if model:
        features = build_features(df1m, df15m)
        prob = model.predict_proba(features)[0][1]

    direction = determine_direction(df1m, df15m)

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        return

    # ==================== ENTRY LOGIC ==================== #
    if not TRADE_ACTIVE:
        # Check cooldown period
        if LAST_EXIT_TIME:
            time_since_exit = (datetime.now(timezone.utc) - LAST_EXIT_TIME).total_seconds() / 60
            if time_since_exit < COOLDOWN_MINUTES:
                print(f"⏳ Cooldown active ({COOLDOWN_MINUTES - time_since_exit:.1f}min remaining)")
                return

        # Only enter if direction is clear AND ML is confident
        if direction in ["BUY", "SELL"] and prob >= MIN_ENTRY_CONFIDENCE:
            embed = build_entry_embed(direction, price, prob, atr)
            await channel.send(embed=embed)
            
            TRADE_ACTIVE = True
            LAST_DIRECTION = direction
            ENTRY_PRICE = price
            ENTRY_TIME = datetime.now(timezone.utc)
            HOLD_COUNTER = 0
            
            print(f"✅ ENTRY: {direction} @ ${price:.2f} (confidence: {prob*100:.1f}%)")
        else:
            print(f"⏸️ No entry: direction={direction}, prob={prob*100:.1f}%")
        return

    # ==================== EXIT LOGIC (when in trade) ==================== #
    if TRADE_ACTIVE:
        exit_reason = None
        
        # 1. HARD STOP: ML confidence dropped significantly
        if prob < MIN_HOLD_CONFIDENCE:
            exit_reason = f"ML confidence dropped to {prob*100:.1f}%"
        
        # 2. HARD STOP: Direction reversed completely
        elif (LAST_DIRECTION == "BUY" and direction == "SELL") or \
             (LAST_DIRECTION == "SELL" and direction == "BUY"):
            exit_reason = "Strong reversal detected"
        
        # 3. SOFT EXIT: Multiple consecutive HOLD signals
        elif direction == "HOLD":
            HOLD_COUNTER += 1
            print(f"⚠️ HOLD signal {HOLD_COUNTER}/{HOLD_SIGNALS_TO_EXIT}")
            
            if HOLD_COUNTER >= HOLD_SIGNALS_TO_EXIT:
                exit_reason = f"Trend weakened ({HOLD_SIGNALS_TO_EXIT} HOLD signals)"
        else:
            # Direction still matches, reset HOLD counter
            HOLD_COUNTER = 0
            print(f"✅ Staying in {LAST_DIRECTION} @ ${price:.2f} (confidence: {prob*100:.1f}%)")

        # Execute exit if triggered
        if exit_reason:
            pnl = None
            if ENTRY_PRICE:
                if LAST_DIRECTION == "BUY":
                    pnl = price - ENTRY_PRICE
                else:  # SELL/PUT
                    pnl = ENTRY_PRICE - price
            
            exit_embed = build_exit_embed(exit_reason, price, pnl)
            await channel.send(embed=exit_embed)
            
            print(f"🚪 EXIT: {exit_reason} @ ${price:.2f}")
            
            TRADE_ACTIVE = False
            LAST_DIRECTION = None
            ENTRY_PRICE = None
            ENTRY_TIME = None
            LAST_EXIT_TIME = datetime.now(timezone.utc)
            HOLD_COUNTER = 0

# ================= LOOP ================= #

@tasks.loop(minutes=1)
async def spy_loop():
    if market_is_open():
        try:
            await check_trade()
        except Exception as e:
            print("Loop error:", e)

@bot.event
async def on_ready():
    print("🚀 SPY ML BOT v2.0 ONLINE")
    print(f"📊 Entry confidence: {MIN_ENTRY_CONFIDENCE*100:.0f}%")
    print(f"🛡️ Hold confidence: {MIN_HOLD_CONFIDENCE*100:.0f}%")
    print(f"⏱️ Cooldown: {COOLDOWN_MINUTES}min")
    spy_loop.start()

bot.run(TOKEN)
