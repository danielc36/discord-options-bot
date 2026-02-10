"""
position_manager.py - Advanced Position & Risk Management
Implements state machine, risk controls, and trade tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

class PositionState(Enum):
    """Position lifecycle states"""
    FLAT = "flat"
    ENTERING = "entering"
    HOLDING = "holding"
    EXITING = "exiting"
    COOLDOWN = "cooldown"

class ExitReason(Enum):
    """Why a position was closed"""
    TARGET_HIT = "target_hit"
    STOP_LOSS = "stop_loss"
    SIGNAL_REVERSAL = "signal_reversal"
    CONFIDENCE_DROP = "confidence_drop"
    TIME_BASED = "time_based"
    TREND_WEAKENED = "trend_weakened"
    MANUAL = "manual"

@dataclass
class Trade:
    """Record of a completed trade"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    target_price: float
    stop_loss: float
    pnl: float
    pnl_pct: float
    exit_reason: ExitReason
    hold_duration: timedelta
    max_favorable_excursion: float  # Best price achieved
    max_adverse_excursion: float    # Worst price hit
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "exit_reason": self.exit_reason.value,
            "hold_minutes": int(self.hold_duration.total_seconds() / 60),
            "mfe": round(self.max_favorable_excursion, 2),
            "mae": round(self.max_adverse_excursion, 2)
        }

@dataclass
class Position:
    """Current open position"""
    direction: str
    entry_price: float
    entry_time: datetime
    target_price: float
    stop_loss: float
    entry_confidence: float
    regime: str
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    hold_signals_count: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_price_extremes(self, current_price: float):
        """Track best/worst prices seen"""
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        self.last_update = datetime.now(timezone.utc)
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P/L"""
        if self.direction == "BUY":
            return current_price - self.entry_price
        else:  # SELL
            return self.entry_price - current_price
    
    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate current unrealized P/L as percentage"""
        pnl = self.get_unrealized_pnl(current_price)
        return (pnl / self.entry_price) * 100


class PositionManager:
    """
    Advanced position and risk management system
    Implements state machine with strict entry/exit rules
    """
    
    def __init__(
        self,
        max_hold_time_minutes: int = 240,  # 4 hours max
        cooldown_minutes: int = 3,
        hold_signals_to_exit: int = 3,
        trailing_stop_enabled: bool = True,
        trailing_stop_activation_pct: float = 1.0,  # Activate after 1% profit
        trailing_stop_distance_pct: float = 0.5,    # Trail by 0.5%
    ):
        self.state = PositionState.FLAT
        self.current_position: Optional[Position] = None
        self.last_exit_time: Optional[datetime] = None
        self.trade_history: List[Trade] = []
        
        # Configuration
        self.max_hold_time = timedelta(minutes=max_hold_time_minutes)
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
        self.hold_signals_to_exit = hold_signals_to_exit
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        
        logger.info("PositionManager initialized")
        logger.info(f"Max hold time: {max_hold_time_minutes}min")
        logger.info(f"Cooldown: {cooldown_minutes}min")
        logger.info(f"Hold signals to exit: {hold_signals_to_exit}")
    
    def can_enter_position(self) -> tuple[bool, Optional[str]]:
        """
        Check if new position entry is allowed
        
        Returns:
            (allowed, reason) tuple
        """
        # Check if already in position
        if self.state != PositionState.FLAT:
            return False, f"Already in state: {self.state.value}"
        
        # Check cooldown period
        if self.last_exit_time is not None:
            time_since_exit = datetime.now(timezone.utc) - self.last_exit_time
            if time_since_exit < self.cooldown_period:
                remaining = (self.cooldown_period - time_since_exit).total_seconds() / 60
                return False, f"Cooldown active ({remaining:.1f}min remaining)"
        
        return True, None
    
    def enter_position(
        self,
        direction: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        confidence: float,
        regime: str
    ) -> bool:
        """
        Enter a new position
        
        Returns:
            True if position entered successfully
        """
        can_enter, reason = self.can_enter_position()
        if not can_enter:
            logger.warning(f"Position entry blocked: {reason}")
            return False
        
        self.current_position = Position(
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            target_price=target_price,
            stop_loss=stop_loss,
            entry_confidence=confidence,
            regime=regime,
            highest_price=entry_price,
            lowest_price=entry_price
        )
        
        self.state = PositionState.HOLDING
        
        logger.info(f"✅ ENTERED {direction} @ ${entry_price:.2f}")
        logger.info(f"   Target: ${target_price:.2f} | Stop: ${stop_loss:.2f}")
        logger.info(f"   Confidence: {confidence:.1%} | Regime: {regime}")
        
        return True
    
    def check_exit_conditions(
        self,
        current_price: float,
        current_signal_direction: str,
        current_confidence: float,
        min_hold_confidence: float = 0.50
    ) -> tuple[bool, Optional[ExitReason]]:
        """
        Evaluate all exit conditions
        
        Returns:
            (should_exit, exit_reason) tuple
        """
        if self.current_position is None:
            return False, None
        
        # Update position tracking
        self.current_position.update_price_extremes(current_price)
        
        pos = self.current_position
        
        # 1. HARD STOP: Stop loss hit
        if pos.direction == "BUY" and current_price <= pos.stop_loss:
            return True, ExitReason.STOP_LOSS
        elif pos.direction == "SELL" and current_price >= pos.stop_loss:
            return True, ExitReason.STOP_LOSS
        
        # 2. PROFIT TARGET: Target hit
        if pos.direction == "BUY" and current_price >= pos.target_price:
            return True, ExitReason.TARGET_HIT
        elif pos.direction == "SELL" and current_price <= pos.target_price:
            return True, ExitReason.TARGET_HIT
        
        # 3. TRAILING STOP (if enabled and in profit)
        if self.trailing_stop_enabled:
            unrealized_pnl_pct = pos.get_unrealized_pnl_pct(current_price)
            
            if unrealized_pnl_pct > self.trailing_stop_activation_pct:
                # Trailing stop is active
                if pos.direction == "BUY":
                    trailing_stop = pos.highest_price * (1 - self.trailing_stop_distance_pct / 100)
                    if current_price <= trailing_stop:
                        logger.info(f"Trailing stop hit: ${current_price:.2f} <= ${trailing_stop:.2f}")
                        return True, ExitReason.TARGET_HIT  # Profitable exit
                else:  # SELL
                    trailing_stop = pos.lowest_price * (1 + self.trailing_stop_distance_pct / 100)
                    if current_price >= trailing_stop:
                        logger.info(f"Trailing stop hit: ${current_price:.2f} >= ${trailing_stop:.2f}")
                        return True, ExitReason.TARGET_HIT
        
        # 4. HARD EXIT: Full signal reversal
        if (pos.direction == "BUY" and current_signal_direction == "SELL") or \
           (pos.direction == "SELL" and current_signal_direction == "BUY"):
            return True, ExitReason.SIGNAL_REVERSAL
        
        # 5. HARD EXIT: Confidence collapsed
        if current_confidence < min_hold_confidence:
            return True, ExitReason.CONFIDENCE_DROP
        
        # 6. SOFT EXIT: Multiple HOLD signals
        if current_signal_direction == "HOLD":
            pos.hold_signals_count += 1
            logger.debug(f"HOLD signal {pos.hold_signals_count}/{self.hold_signals_to_exit}")
            
            if pos.hold_signals_count >= self.hold_signals_to_exit:
                return True, ExitReason.TREND_WEAKENED
        else:
            # Reset counter if direction matches position
            if current_signal_direction == pos.direction:
                pos.hold_signals_count = 0
        
        # 7. TIME-BASED EXIT: Max hold time exceeded
        hold_duration = datetime.now(timezone.utc) - pos.entry_time
        if hold_duration > self.max_hold_time:
            return True, ExitReason.TIME_BASED
        
        return False, None
    
    def exit_position(
        self,
        exit_price: float,
        exit_reason: ExitReason
    ) -> Optional[Trade]:
        """
        Close current position and record trade
        
        Returns:
            Trade record
        """
        if self.current_position is None:
            logger.warning("No position to exit")
            return None
        
        pos = self.current_position
        exit_time = datetime.now(timezone.utc)
        
        # Calculate P/L
        if pos.direction == "BUY":
            pnl = exit_price - pos.entry_price
            mfe = pos.highest_price - pos.entry_price
            mae = pos.lowest_price - pos.entry_price
        else:  # SELL
            pnl = pos.entry_price - exit_price
            mfe = pos.entry_price - pos.lowest_price
            mae = pos.entry_price - pos.highest_price
        
        pnl_pct = (pnl / pos.entry_price) * 100
        
        # Create trade record
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            target_price=pos.target_price,
            stop_loss=pos.stop_loss,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            hold_duration=exit_time - pos.entry_time,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae
        )
        
        self.trade_history.append(trade)
        
        # Log exit
        pnl_emoji = "📈" if pnl > 0 else "📉"
        logger.info(f"🚪 EXITED {pos.direction} @ ${exit_price:.2f}")
        logger.info(f"   {pnl_emoji} P/L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"   Reason: {exit_reason.value}")
        logger.info(f"   Hold time: {trade.hold_duration}")
        
        # Update state
        self.current_position = None
        self.last_exit_time = exit_time
        self.state = PositionState.FLAT
        
        return trade
    
    def get_performance_stats(self) -> Dict:
        """Calculate overall performance statistics"""
        if not self.trade_history:
            return {"total_trades": 0}
        
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) \
                       if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')
        
        total_pnl = sum(t.pnl for t in self.trade_history)
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "largest_win": round(max(t.pnl for t in self.trade_history), 2),
            "largest_loss": round(min(t.pnl for t in self.trade_history), 2)
        }
    
    def save_trade_history(self, filepath: str = "trade_history.json"):
        """Save trade history to file"""
        data = [trade.to_dict() for trade in self.trade_history]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} trades to {filepath}")


import numpy as np  # For stats calculations
