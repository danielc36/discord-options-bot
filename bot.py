"""
bot_manual_signals.py - Manual Trading Signals Bot
Optimized for iPad alerts → Robinhood phone execution
"""

import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import pytz
import joblib
import logging
from typing import Optional

from market_pro import market_data
from indicators_pro import indicator_suite
from strategy_pro import strategy, TradingSignal, SignalStrength
from position_manager import PositionManager, ExitReason

# ================= LOGGING SETUP ================= #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signals_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIG ================= #

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))
SYMBOL = os.getenv("SYMBOL", "SPY")

# Manual Trading Settings
MIN_SIGNAL_CONFIDENCE = 0.70  # Only send high-confidence signals
MIN_SIGNAL_STRENGTH = 4  # Strong or Very Strong only
MAX_SIGNALS_PER_DAY = 10
PREFERRED_EXPIRATION_DAYS = 0  # 0DTE (same day)

# ================= INITIALIZE ================= #

try:
    model = joblib.load("model.pkl")
    logger.info("✅ ML model loaded")
except:
    logger.warning("⚠️ ML model not found, using indicators only")
    model = None

# Position tracker for manual trades
position_manager = PositionManager(
    max_hold_time_minutes=360,  # 6 hours (more lenient for manual)
    cooldown_minutes=5,  # Longer cooldown between signals
    hold_signals_to_exit=4,  # More patient exits
    trailing_stop_enabled=True,
    trailing_stop_activation_pct=1.5,
    trailing_stop_distance_pct=0.75
)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Track signals sent today
signals_sent_today = 0
last_reset_date = None

# ================= MARKET HOURS ================= #

def market_is_open() -> bool:
    """Check if market is open"""
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    
    if now.weekday() >= 5:
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def get_options_expiration() -> str:
    """Get the next options expiration date"""
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    
    if PREFERRED_EXPIRATION_DAYS == 0:
        # Same day expiration
        return now.strftime("%m/%d")
    else:
        # Calculate days ahead
        exp_date = now + timedelta(days=PREFERRED_EXPIRATION_DAYS)
        # Adjust to Friday if not already
        days_until_friday = (4 - exp_date.weekday()) % 7
        if days_until_friday > 0:
            exp_date = exp_date + timedelta(days=days_until_friday)
        return exp_date.strftime("%m/%d")

def calculate_strike_price(current_price: float, direction: str, otm_amount: float = 0.5) -> float:
    """
    Calculate ATM or slightly OTM strike
    
    Args:
        current_price: Current stock price
        direction: BUY (calls) or SELL (puts)
        otm_amount: How far OTM (0.5 = $0.50 OTM)
    """
    if direction == "BUY":
        # Calls - round to nearest strike above current price
        strike = round(current_price + otm_amount, 0)
    else:  # SELL = Puts
        # Puts - round to nearest strike below current price
        strike = round(current_price - otm_amount, 0)
    
    return strike

def suggest_position_size(signal: TradingSignal) -> tuple:
    """
    Suggest position size based on signal strength
    
    Returns:
        (min_contracts, max_contracts)
    """
    if signal.strength == SignalStrength.VERY_STRONG:
        return (3, 5)
    elif signal.strength == SignalStrength.STRONG:
        return (2, 3)
    else:
        return (1, 2)

# ================= DISCORD EMBEDS ================= #

def build_signal_embed(signal: TradingSignal, current_price: float) -> discord.Embed:
    """Create actionable trading signal for manual execution"""
    
    # Determine colors and emojis
    if signal.direction == "BUY":
        color = discord.Color.green()
        emoji = "🟢📈"
        option_type = "CALL"
    else:
        color = discord.Color.red()
        emoji = "🔴📉"
        option_type = "PUT"
    
    # Calculate options details
    strike = calculate_strike_price(current_price, signal.direction)
    expiration = get_options_expiration()
    min_size, max_size = suggest_position_size(signal)
    
    # Create embed
    embed = discord.Embed(
        title=f"{emoji} {SYMBOL} {option_type} SIGNAL",
        description=f"**{signal.strength.name}** setup detected",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )
    
    # PRIMARY ACTION
    embed.add_field(
        name="📱 ROBINHOOD ACTION",
        value=f"Buy **{SYMBOL} ${strike:.0f}{option_type[0]}** exp {expiration}\n"
              f"Position size: **{min_size}-{max_size} contracts**",
        inline=False
    )
    
    # PRICE LEVELS
    embed.add_field(
        name="💰 Current Price",
        value=f"${current_price:.2f}",
        inline=True
    )
    embed.add_field(
        name="🎯 Price Target",
        value=f"${signal.target_price:.2f}",
        inline=True
    )
    embed.add_field(
        name="🛡️ Stop Loss",
        value=f"${signal.stop_loss:.2f}",
        inline=True
    )
    
    # CONFIDENCE METRICS
    embed.add_field(
        name="📊 ML Confidence",
        value=f"**{signal.confidence*100:.0f}%**",
        inline=True
    )
    embed.add_field(
        name="⚖️ Risk:Reward",
        value=f"**1:{signal.risk_reward_ratio:.1f}**",
        inline=True
    )
    embed.add_field(
        name="🌡️ Market Regime",
        value=signal.regime.value.replace('_', ' ').title(),
        inline=True
    )
    
    # EXPECTED MOVE
    expected_profit_pct = abs((signal.target_price - current_price) / current_price) * 100
    expected_loss_pct = abs((signal.stop_loss - current_price) / current_price) * 100
    
    embed.add_field(
        name="📈 Expected Move",
        value=f"Target: **+{expected_profit_pct:.1f}%** | Stop: **-{expected_loss_pct:.1f}%**",
        inline=False
    )
    
    # INDICATOR AGREEMENT
    bullish = sum(1 for v in signal.contributing_factors.values() if v > 0)
    bearish = sum(1 for v in signal.contributing_factors.values() if v < 0)
    total = len(signal.contributing_factors)
    
    if signal.direction == "BUY":
        agreement = f"✅ {bullish}/{total} indicators bullish"
    else:
        agreement = f"✅ {bearish}/{total} indicators bearish"
    
    embed.add_field(
        name="📊 Confirmation",
        value=agreement,
        inline=False
    )
    
    # FOOTER WITH INSTRUCTIONS
    embed.set_footer(
        text="✅ React with ✅ when you ENTER the trade | "
             "⚠️ Manual execution only - not financial advice"
    )
    
    return embed

def build_exit_embed(
    current_price: float,
    position_direction: str,
    entry_price: float,
    exit_reason: str,
    unrealized_pnl: float,
    unrealized_pnl_pct: float
) -> discord.Embed:
    """Create exit signal alert"""
    
    color = discord.Color.gold() if unrealized_pnl > 0 else discord.Color.orange()
    
    if position_direction == "BUY":
        option_type = "CALL"
    else:
        option_type = "PUT"
    
    embed = discord.Embed(
        title=f"🔔 {SYMBOL} EXIT SIGNAL",
        description=f"Close your {option_type} position",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )
    
    # ACTION
    pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
    action = "SELL TO CLOSE" if position_direction == "BUY" else "BUY TO CLOSE"
    
    embed.add_field(
        name="📱 ROBINHOOD ACTION",
        value=f"**{action}** your {SYMBOL} {option_type} position",
        inline=False
    )
    
    # PRICES
    embed.add_field(
        name="Entry → Current",
        value=f"${entry_price:.2f} → ${current_price:.2f}",
        inline=True
    )
    embed.add_field(
        name=f"{pnl_emoji} Expected P/L",
        value=f"${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.1f}%)",
        inline=True
    )
    
    # REASON
    embed.add_field(
        name="🎯 Exit Reason",
        value=exit_reason,
        inline=False
    )
    
    embed.set_footer(
        text="✅ React with ✅ when you EXIT the trade"
    )
    
    return embed

def build_position_update_embed(
    current_price: float,
    signal: TradingSignal,
    position_entry: float,
    unrealized_pnl: float,
    unrealized_pnl_pct: float,
    hold_time_minutes: int
) -> discord.Embed:
    """Periodic position update"""
    
    color = discord.Color.green() if unrealized_pnl > 0 else discord.Color.red()
    pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
    
    embed = discord.Embed(
        title=f"📊 {SYMBOL} Position Update",
        description="Your trade is still active",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(
        name="Current Price",
        value=f"${current_price:.2f}",
        inline=True
    )
    embed.add_field(
        name=f"{pnl_emoji} Unrealized P/L",
        value=f"${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.1f}%)",
        inline=True
    )
    embed.add_field(
        name="⏱️ Hold Time",
        value=f"{hold_time_minutes}min",
        inline=True
    )
    
    # Progress to target
    pos = position_manager.current_position
    if pos.direction == "BUY":
        progress = (current_price - position_entry) / (pos.target_price - position_entry)
    else:
        progress = (position_entry - current_price) / (position_entry - pos.target_price)
    
    progress_pct = max(0, min(100, progress * 100))
    
    embed.add_field(
        name="Target Progress",
        value=f"{'█' * int(progress_pct/10)}{'░' * (10-int(progress_pct/10))} {progress_pct:.0f}%",
        inline=False
    )
    
    embed.add_field(
        name="Current Confidence",
        value=f"{signal.confidence*100:.0f}%",
        inline=True
    )
    
    embed.set_footer(text="Holding position - watching for exit conditions")
    
    return embed

# ================= CORE LOGIC ================= #

async def check_for_signals():
    """Main signal generation logic"""
    global signals_sent_today, last_reset_date
    
    try:
        # Reset daily counter
        eastern = pytz.timezone("US/Eastern")
        today = datetime.now(eastern).date()
        if last_reset_date != today:
            signals_sent_today = 0
            last_reset_date = today
            logger.info(f"📅 New trading day - signal counter reset")
        
        # Check if we've hit daily limit
        if signals_sent_today >= MAX_SIGNALS_PER_DAY:
            logger.debug(f"Daily signal limit reached ({MAX_SIGNALS_PER_DAY})")
            return

        # Fetch data (1m = today, 15m = 5 days for enough history)
        df1m = market_data.get_stock_data(SYMBOL, interval="1m", period="1d")
        df15m = market_data.get_stock_data(SYMBOL, interval="15m", period="5d")

        if df1m is None or df15m is None:
            logger.warning("Failed to fetch data")
            return
        
        # Add indicators
        df1m = indicator_suite.add_all_indicators(df1m)
        df15m = indicator_suite.add_all_indicators(df15m)
        
        if df1m is None or df15m is None:
            logger.warning("Indicator calculation failed")
            return
        
        current_price = df1m["Close"].iloc[-1]
        
        # Generate signal
        signal = strategy.analyze(df1m, df15m, ml_model=model)
        
        logger.info(f"Analysis: {signal.direction} | {signal.strength.name} | {signal.confidence:.1%}")
        
        channel = bot.get_channel(CHANNEL_ID)
        if not channel:
            logger.error("Channel not found")
            return
        
        # ==================== ENTRY SIGNALS ==================== #
        if position_manager.state.value == "flat":
            can_enter, reason = position_manager.can_enter_position()
            
            if not can_enter:
                logger.debug(f"Entry blocked: {reason}")
                return
            
            # HIGH QUALITY SIGNALS ONLY
            if (signal.direction in ["BUY", "SELL"] and 
                signal.confidence >= MIN_SIGNAL_CONFIDENCE and
                signal.strength.value >= MIN_SIGNAL_STRENGTH):
                
                # Send signal
                embed = build_signal_embed(signal, current_price)
                message = await channel.send(embed=embed)
                
                # Add reaction for user confirmation
                await message.add_reaction("✅")
                
                # Enter position in tracker
                position_manager.enter_position(
                    direction=signal.direction,
                    entry_price=current_price,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    confidence=signal.confidence,
                    regime=signal.regime.value
                )
                
                signals_sent_today += 1
                
                logger.info(f"🚀 SIGNAL SENT ({signals_sent_today}/{MAX_SIGNALS_PER_DAY}): "
                          f"{signal.direction} @ ${current_price:.2f}")
        
        # ==================== EXIT SIGNALS ==================== #
        elif position_manager.state.value == "holding":
            should_exit, exit_reason = position_manager.check_exit_conditions(
                current_price=current_price,
                current_signal_direction=signal.direction,
                current_confidence=signal.confidence,
                min_hold_confidence=0.55  # Slightly more lenient for manual trading
            )
            
            if should_exit:
                pos = position_manager.current_position
                unrealized_pnl = pos.get_unrealized_pnl(current_price)
                unrealized_pnl_pct = pos.get_unrealized_pnl_pct(current_price)
                
                # Format exit reason
                exit_reason_text = exit_reason.value.replace('_', ' ').title()
                
                # Send exit alert
                embed = build_exit_embed(
                    current_price=current_price,
                    position_direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_reason=exit_reason_text,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct
                )
                
                message = await channel.send(embed=embed)
                await message.add_reaction("✅")
                
                # Close position
                position_manager.exit_position(
                    exit_price=current_price,
                    exit_reason=exit_reason
                )
                
                logger.info(f"🚪 EXIT SIGNAL: {exit_reason_text} @ ${current_price:.2f}")
                
                # Save trade history
                position_manager.save_trade_history()
            
            else:
                # Send periodic update every 30 minutes
                if datetime.now(timezone.utc).minute % 30 == 0:
                    pos = position_manager.current_position
                    hold_time = int((datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60)
                    unrealized_pnl = pos.get_unrealized_pnl(current_price)
                    unrealized_pnl_pct = pos.get_unrealized_pnl_pct(current_price)
                    
                    update_embed = build_position_update_embed(
                        current_price=current_price,
                        signal=signal,
                        position_entry=pos.entry_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        hold_time_minutes=hold_time
                    )
                    
                    await channel.send(embed=update_embed)
    
    except Exception as e:
        logger.error(f"Error in signal check: {e}", exc_info=True)

# ================= COMMANDS ================= #

@bot.command(name="status")
async def status(ctx):
    """Check bot status"""
    embed = discord.Embed(
        title=f"🤖 {SYMBOL} Signals Bot Status",
        color=discord.Color.blue(),
        timestamp=datetime.now(timezone.utc)
    )
    
    market_status = "🟢 OPEN" if market_is_open() else "🔴 CLOSED"
    embed.add_field(name="Market", value=market_status, inline=True)
    
    embed.add_field(
        name="Signals Today",
        value=f"{signals_sent_today}/{MAX_SIGNALS_PER_DAY}",
        inline=True
    )
    
    if position_manager.current_position:
        pos = position_manager.current_position
        embed.add_field(
            name="Your Position",
            value=f"{"CALL" if pos.direction == "BUY" else "PUT"} @ ${pos.entry_price:.2f}",
            inline=True
        )
    else:
        embed.add_field(name="Your Position", value="None (watching for signals)", inline=True)
    
    stats = position_manager.get_performance_stats()
    if stats.get('total_trades', 0) > 0:
        embed.add_field(
            name="Performance",
            value=f"Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.0%}",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name="stats")
async def stats_command(ctx):
    """Show detailed statistics"""
    stats = position_manager.get_performance_stats()
    
    if stats.get('total_trades', 0) == 0:
        await ctx.send("No completed trades yet!")
        return
    
    embed = discord.Embed(
        title="📊 Trading Performance",
        color=discord.Color.gold(),
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(name="Total Signals", value=stats['total_trades'], inline=True)
    embed.add_field(name="Winners", value=stats['winning_trades'], inline=True)
    embed.add_field(name="Losers", value=stats['losing_trades'], inline=True)
    
    embed.add_field(name="Win Rate", value=f"{stats['win_rate']:.0%}", inline=True)
    embed.add_field(name="Profit Factor", value=f"{stats['profit_factor']:.2f}", inline=True)
    embed.add_field(name="Avg Win", value=f"${stats['avg_win']:.2f}", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="close")
async def manual_close(ctx):
    """Manually mark position as closed"""
    if position_manager.current_position is None:
        await ctx.send("No active position to close!")
        return
    
    df = market_data.get_stock_data(SYMBOL, interval="1m")
    if df is None:
        await ctx.send("Failed to fetch current price")
        return
    
    current_price = df["Close"].iloc[-1]
    
    position_manager.exit_position(
        exit_price=current_price,
        exit_reason=ExitReason.MANUAL
    )
    
    await ctx.send("✅ Position marked as closed. Ready for next signal!")

# ================= MAIN LOOP ================= #

@tasks.loop(minutes=1)
async def signal_loop():
    """Check for signals every minute during market hours"""
    if market_is_open():
        try:
            await check_for_signals()
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)

@bot.event
async def on_ready():
    """Bot startup"""
    logger.info("=" * 60)
    logger.info("🚀 MANUAL SIGNALS BOT ONLINE")
    logger.info(f"📊 Symbol: {SYMBOL}")
    logger.info(f"🎯 Min Confidence: {MIN_SIGNAL_CONFIDENCE:.0%}")
    logger.info(f"📈 Max Signals/Day: {MAX_SIGNALS_PER_DAY}")
    logger.info(f"📅 Expiration: {get_options_expiration()}")
    logger.info("=" * 60)
    
    signal_loop.start()

if __name__ == "__main__":
    bot.run(TOKEN)
