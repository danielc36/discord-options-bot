"""
bot_ultimate.py - Production-Grade Trading Bot
Enterprise-level implementation with all advanced features
"""

import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import pytz
import joblib
import logging
from typing import Optional

# Import our advanced modules
from market_pro import market_data
from indicators_pro import indicator_suite
from strategy_pro import strategy, TradingSignal
from position_manager import PositionManager, ExitReason

# ================= LOGGING SETUP ================= #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIG ================= #

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))
SYMBOL = os.getenv("SYMBOL", "SPY")

# ================= INITIALIZE COMPONENTS ================= #

# Load ML model
try:
    model = joblib.load("model.pkl")
    logger.info("✅ ML model loaded successfully")
except Exception as e:
    logger.warning(f"ML model not found: {e}")
    model = None

# Initialize position manager
position_manager = PositionManager(
    max_hold_time_minutes=240,  # 4 hours max
    cooldown_minutes=3,
    hold_signals_to_exit=3,
    trailing_stop_enabled=True,
    trailing_stop_activation_pct=1.0,
    trailing_stop_distance_pct=0.5
)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ================= MARKET HOURS ================= #

def market_is_open() -> bool:
    """Check if market is currently open"""
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    
    # Weekend check
    if now.weekday() >= 5:
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

# ================= DISCORD EMBEDS ================= #

def build_entry_embed(signal: TradingSignal) -> discord.Embed:
    """Create rich entry signal embed"""
    color = discord.Color.green() if signal.direction == "BUY" else discord.Color.red()
    emoji = "🟢📈" if signal.direction == "BUY" else "🔴📉"
    
    embed = discord.Embed(
        title=f"{emoji} {SYMBOL} {signal.direction} SIGNAL",
        description=f"**{signal.strength.name}** signal detected",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Price levels
    embed.add_field(
        name="💰 Entry",
        value=f"${signal.entry_price:.2f}",
        inline=True
    )
    embed.add_field(
        name="🎯 Target",
        value=f"${signal.target_price:.2f}",
        inline=True
    )
    embed.add_field(
        name="🛡️ Stop",
        value=f"${signal.stop_loss:.2f}",
        inline=True
    )
    
    # Risk metrics
    embed.add_field(
        name="📊 ML Confidence",
        value=f"{signal.confidence*100:.1f}%",
        inline=True
    )
    embed.add_field(
        name="⚖️ Risk:Reward",
        value=f"1:{signal.risk_reward_ratio:.2f}",
        inline=True
    )
    embed.add_field(
        name="🌡️ Regime",
        value=signal.regime.value.replace('_', ' ').title(),
        inline=True
    )
    
    # Supporting factors
    bullish_factors = sum(1 for v in signal.contributing_factors.values() if v > 0)
    bearish_factors = sum(1 for v in signal.contributing_factors.values() if v < 0)
    total_factors = len(signal.contributing_factors)
    
    if signal.direction == "BUY":
        agreement = f"{bullish_factors}/{total_factors} indicators bullish"
    else:
        agreement = f"{bearish_factors}/{total_factors} indicators bearish"
    
    embed.add_field(
        name="📈 Indicator Agreement",
        value=agreement,
        inline=False
    )
    
    embed.set_footer(text="⚠️ Educational only • Not financial advice • Position OPENED")
    
    return embed

def build_status_embed(
    current_price: float,
    signal: TradingSignal
) -> discord.Embed:
    """Create position status update embed"""
    pos = position_manager.current_position
    if pos is None:
        return None
    
    unrealized_pnl = pos.get_unrealized_pnl(current_price)
    unrealized_pnl_pct = pos.get_unrealized_pnl_pct(current_price)
    
    color = discord.Color.green() if unrealized_pnl > 0 else discord.Color.red()
    pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
    
    embed = discord.Embed(
        title=f"📊 {SYMBOL} Position Update",
        description=f"Holding **{pos.direction}** position",
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
        value=f"${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.2f}%)",
        inline=True
    )
    embed.add_field(
        name="⏱️ Hold Time",
        value=f"{int((datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60)}min",
        inline=True
    )
    
    embed.add_field(
        name="Entry → Target",
        value=f"${pos.entry_price:.2f} → ${pos.target_price:.2f}",
        inline=True
    )
    embed.add_field(
        name="Current Confidence",
        value=f"{signal.confidence*100:.1f}%",
        inline=True
    )
    embed.add_field(
        name="HOLD Signals",
        value=f"{pos.hold_signals_count}/{position_manager.hold_signals_to_exit}",
        inline=True
    )
    
    # Progress bar toward target
    if pos.direction == "BUY":
        progress = (current_price - pos.entry_price) / (pos.target_price - pos.entry_price)
    else:
        progress = (pos.entry_price - current_price) / (pos.entry_price - pos.target_price)
    
    progress_pct = max(0, min(100, progress * 100))
    filled_bars = int(progress_pct / 10)
    bar = "█" * filled_bars + "░" * (10 - filled_bars)
    
    embed.add_field(
        name="Progress to Target",
        value=f"{bar} {progress_pct:.1f}%",
        inline=False
    )
    
    embed.set_footer(text="Position HOLDING • Monitoring exit conditions")
    
    return embed

def build_exit_embed(
    exit_price: float,
    exit_reason: ExitReason,
    pnl: float,
    pnl_pct: float
) -> discord.Embed:
    """Create exit signal embed"""
    color = discord.Color.gold() if pnl > 0 else discord.Color.orange()
    pnl_emoji = "🎉" if pnl > 0 else "😔"
    
    embed = discord.Embed(
        title=f"🔔 {SYMBOL} Position CLOSED",
        description=f"{pnl_emoji} Trade completed",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(
        name="Exit Price",
        value=f"${exit_price:.2f}",
        inline=True
    )
    embed.add_field(
        name="Realized P/L",
        value=f"${pnl:.2f} ({pnl_pct:+.2f}%)",
        inline=True
    )
    embed.add_field(
        name="Exit Reason",
        value=exit_reason.value.replace('_', ' ').title(),
        inline=True
    )
    
    # Performance stats
    stats = position_manager.get_performance_stats()
    
    embed.add_field(
        name="📊 Session Stats",
        value=f"Trades: {stats.get('total_trades', 0)} | Win Rate: {stats.get('win_rate', 0):.1%}",
        inline=False
    )
    
    if stats.get('total_trades', 0) > 0:
        embed.add_field(
            name="Total P/L",
            value=f"${stats.get('total_pnl', 0):.2f}",
            inline=True
        )
        embed.add_field(
            name="Profit Factor",
            value=f"{stats.get('profit_factor', 0):.2f}",
            inline=True
        )
    
    embed.set_footer(text="Position CLOSED • Cooldown period active")
    
    return embed

# ================= CORE TRADING LOGIC ================= #

async def check_trade():
    """Main trading logic - runs every minute"""
    try:
        # Fetch multi-timeframe data
        data = market_data.get_multi_timeframe_data(SYMBOL, ["1m", "15m"])
        
        if "1m" not in data or "15m" not in data:
            logger.warning("Failed to fetch required timeframe data")
            return
        
        df1m = data["1m"]
        df15m = data["15m"]
        
        # Add all technical indicators
        df1m = indicator_suite.add_all_indicators(df1m)
        df15m = indicator_suite.add_all_indicators(df15m)
        
        if df1m is None or df15m is None:
            logger.warning("Indicator calculation failed")
            return
        
        # Get current price
        current_price = df1m["Close"].iloc[-1]
        
        # Generate trading signal using advanced strategy
        signal = strategy.analyze(df1m, df15m, ml_model=model)
        
        logger.info(f"Signal: {signal.direction} | Strength: {signal.strength.name} | Confidence: {signal.confidence:.1%}")
        
        # Get Discord channel
        channel = bot.get_channel(CHANNEL_ID)
        if not channel:
            logger.error("Discord channel not found")
            return
        
        # ==================== ENTRY LOGIC ==================== #
        if position_manager.state.value == "flat":
            can_enter, reason = position_manager.can_enter_position()
            
            if not can_enter:
                logger.debug(f"Entry blocked: {reason}")
                return
            
            # Only enter if we have a clear BUY or SELL signal
            if signal.direction in ["BUY", "SELL"]:
                success = position_manager.enter_position(
                    direction=signal.direction,
                    entry_price=current_price,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    confidence=signal.confidence,
                    regime=signal.regime.value
                )
                
                if success:
                    embed = build_entry_embed(signal)
                    await channel.send(embed=embed)
            else:
                logger.debug(f"No entry: signal is {signal.direction}")
        
        # ==================== POSITION MANAGEMENT ==================== #
        elif position_manager.state.value == "holding":
            # Check exit conditions
            should_exit, exit_reason = position_manager.check_exit_conditions(
                current_price=current_price,
                current_signal_direction=signal.direction,
                current_confidence=signal.confidence,
                min_hold_confidence=0.50
            )
            
            if should_exit:
                # Execute exit
                trade = position_manager.exit_position(
                    exit_price=current_price,
                    exit_reason=exit_reason
                )
                
                if trade:
                    embed = build_exit_embed(
                        current_price,
                        exit_reason,
                        trade.pnl,
                        trade.pnl_pct
                    )
                    await channel.send(embed=embed)
                    
                    # Save trade history
                    position_manager.save_trade_history()
            else:
                # Send periodic status update (every 15 minutes when holding)
                if datetime.now(timezone.utc).minute % 15 == 0:
                    status_embed = build_status_embed(current_price, signal)
                    if status_embed:
                        await channel.send(embed=status_embed)
        
    except Exception as e:
        logger.error(f"Error in check_trade: {e}", exc_info=True)

# ================= DISCORD COMMANDS ================= #

@bot.command(name="status")
async def status(ctx):
    """Check current bot and position status"""
    embed = discord.Embed(
        title=f"🤖 {SYMBOL} Trading Bot Status",
        color=discord.Color.blue(),
        timestamp=datetime.now(timezone.utc)
    )
    
    # Market status
    market_status = "🟢 OPEN" if market_is_open() else "🔴 CLOSED"
    embed.add_field(name="Market", value=market_status, inline=True)
    
    # Position status
    if position_manager.current_position:
        pos = position_manager.current_position
        embed.add_field(
            name="Position",
            value=f"{pos.direction} @ ${pos.entry_price:.2f}",
            inline=True
        )
    else:
        embed.add_field(name="Position", value="FLAT", inline=True)
    
    # Performance stats
    stats = position_manager.get_performance_stats()
    if stats.get('total_trades', 0) > 0:
        embed.add_field(
            name="Session Performance",
            value=f"Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.1%} | P/L: ${stats['total_pnl']:.2f}",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name="stats")
async def stats(ctx):
    """Show detailed performance statistics"""
    stats = position_manager.get_performance_stats()
    
    if stats.get('total_trades', 0) == 0:
        await ctx.send("No trades yet!")
        return
    
    embed = discord.Embed(
        title="📈 Performance Statistics",
        color=discord.Color.gold(),
        timestamp=datetime.now(timezone.utc)
    )
    
    embed.add_field(name="Total Trades", value=stats['total_trades'], inline=True)
    embed.add_field(name="Winners", value=stats['winning_trades'], inline=True)
    embed.add_field(name="Losers", value=stats['losing_trades'], inline=True)
    
    embed.add_field(name="Win Rate", value=f"{stats['win_rate']:.1%}", inline=True)
    embed.add_field(name="Profit Factor", value=f"{stats['profit_factor']:.2f}", inline=True)
    embed.add_field(name="Total P/L", value=f"${stats['total_pnl']:.2f}", inline=True)
    
    embed.add_field(name="Avg Win", value=f"${stats['avg_win']:.2f}", inline=True)
    embed.add_field(name="Avg Loss", value=f"${stats['avg_loss']:.2f}", inline=True)
    embed.add_field(name="Largest Win", value=f"${stats['largest_win']:.2f}", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="close")
async def force_close(ctx):
    """Manually close current position"""
    if position_manager.current_position is None:
        await ctx.send("No position to close!")
        return
    
    # Fetch current price
    df = market_data.get_stock_data(SYMBOL, interval="1m")
    if df is None:
        await ctx.send("Failed to fetch current price")
        return
    
    current_price = df["Close"].iloc[-1]
    
    trade = position_manager.exit_position(
        exit_price=current_price,
        exit_reason=ExitReason.MANUAL
    )
    
    if trade:
        embed = build_exit_embed(
            current_price,
            ExitReason.MANUAL,
            trade.pnl,
            trade.pnl_pct
        )
        await ctx.send(embed=embed)

# ================= MAIN LOOP ================= #

@tasks.loop(minutes=1)
async def trading_loop():
    """Main trading loop - runs every minute during market hours"""
    if market_is_open():
        try:
            await check_trade()
        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)

@bot.event
async def on_ready():
    """Bot startup"""
    logger.info("=" * 60)
    logger.info("🚀 ULTIMATE TRADING BOT ONLINE")
    logger.info(f"📊 Symbol: {SYMBOL}")
    logger.info(f"🤖 Bot: {bot.user.name}")
    logger.info(f"💬 Channel ID: {CHANNEL_ID}")
    logger.info(f"📈 ML Model: {'Loaded' if model else 'Not available'}")
    logger.info("=" * 60)
    
    trading_loop.start()

# ================= RUN BOT ================= #

if __name__ == "__main__":
    bot.run(TOKEN)
