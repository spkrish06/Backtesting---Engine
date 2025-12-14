"""
Felix Backtester - Momentum Breakout Strategy
"""
import sys
import os
from datetime import datetime
from collections import deque

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "python"))

# Add compiled engine path relative to project root (which is in sys.path[1] or similar, but better explicit)
sys.path.append(os.path.join(project_root, "Release"))

import felix_engine as fe
from felix.strategy.base import Strategy
from felix.analytics.metrics import BacktestResults


class MomentumBreakoutStrategy(Strategy):
    """
    Momentum Breakout Strategy.
    Entry: Price breaks above 20-period high (momentum breakout)
    Exit: Price drops below 10-period low or 5% target or 3% stop-loss
    """
    
    def __init__(self, engine: fe.MatchingEngine, portfolio: fe.Portfolio, risk_engine: fe.RiskEngine):
        self.engine = engine
        self.portfolio = portfolio
        self.risk_engine = risk_engine
        
        # Momentum parameters
        self.breakout_period = 20
        self.exit_period = 10
        self.highs = deque(maxlen=self.breakout_period)
        self.lows = deque(maxlen=self.exit_period)
        
        # Position tracking
        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        
        # Analytics
        self.equity_curve = []
        self.trades = []
        self.initial_cash = portfolio.cash()
        self.peak_equity = portfolio.cash()
        self.risk_halted = False
        
    def on_start(self):
        print("="*60)
        print("Momentum Breakout Strategy Started")
        print(f"Breakout Period: {self.breakout_period}")
        print(f"Exit Period: {self.exit_period}")
        print("Stop Loss: 3.0%, Target: 5.0%")
        print("="*60)
        self.equity_curve.append(self.portfolio.cash())
    
    def on_tick(self, tick):
        # Update portfolio mark-to-market
        self.portfolio.update_prices(tick.symbol_id, tick.price)
        
        # Record equity
        equity = self.portfolio.cash()
        if self.position != 0:
            equity += self.position * tick.price
        self.equity_curve.append(equity)
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        price = tick.price
        
        # Update price history
        self.highs.append(price)
        self.lows.append(price)
        
        # Entry: Price breaks above 20-period high
        if self.position == 0 and len(self.highs) >= self.breakout_period:
            highest_high = max(list(self.highs)[:-1])  # Exclude current price
            
            if price > highest_high:
                # Risk checks
                if self.risk_halted:
                    return
                
                if not self.risk_engine.check_drawdown(self.portfolio, self.peak_equity):
                    print(f"  [RISK] Halted - Max drawdown exceeded!")
                    self.risk_halted = True
                    return
                
                shares = 100
                cost = shares * price
                
                 # Check if we have enough cash
                if self.portfolio.cash() < cost:
                    return

                self.position = shares
                self.entry_price = price
                self.stop_price = price * 0.97  # 3% stop
                self.target_price = price * 1.05  # 5% target
                
                # Create Fill object to update portfolio
                fill = fe.Fill()
                fill.order_id = 999 
                fill.symbol_id = tick.symbol_id
                fill.price = price
                fill.volume = float(shares)
                fill.side = fe.Side.BUY
                self.portfolio.on_fill(fill)
                
                print(f"  [ENTRY] BUY {shares} @ ${price:.2f} | Breakout High=${highest_high:.2f} | Stop=${self.stop_price:.2f} | Target=${self.target_price:.2f}")
                
                self.trades.append({
                    'datetime': datetime.fromtimestamp(tick.timestamp / 1e9).strftime('%Y-%m-%d %H:%M:%S'),
                    'side': 'BUY',
                    'price': round(price, 2),
                    'shares': shares,
                    'pnl': 0.0,
                })
        
        # Exit: Price drops below 10-period low OR stop-loss OR target
        if self.position > 0 and len(self.lows) >= self.exit_period:
            lowest_low = min(self.lows)
            
            exit_reason = None
            exit_price = None
            
            if price < lowest_low:
                exit_reason = "BREAKDOWN"
                exit_price = lowest_low
            elif price <= self.stop_price:
                exit_reason = "STOP_LOSS"
                exit_price = self.stop_price
            elif price >= self.target_price:
                exit_reason = "TARGET"
                exit_price = self.target_price
            
            if exit_reason:
                pnl = (exit_price - self.entry_price) * self.position
                
                # Create Fill object to update portfolio
                fill = fe.Fill()
                fill.order_id = 999 
                fill.symbol_id = tick.symbol_id
                fill.price = exit_price
                fill.volume = float(self.position)
                fill.side = fe.Side.SELL
                self.portfolio.on_fill(fill)

                print(f"  [EXIT] SELL {self.position} @ ${exit_price:.2f} | {exit_reason} | P&L: ${pnl:+.2f}")
                
                self.trades.append({
                    'datetime': datetime.fromtimestamp(tick.timestamp / 1e9).strftime('%Y-%m-%d %H:%M:%S'),
                    'side': 'SELL',
                    'price': round(exit_price, 2),
                    'shares': self.position,
                    'pnl': round(pnl, 2),
                })
                
                self.position = 0
    
    def on_fill(self, fill):
        pass
    
    def on_bar(self, bar):
        pass
    
    def on_end(self):
        print("\n" + "="*60)
        print("Momentum Breakout Strategy Complete")
        print("="*60)
        
        results = BacktestResults(
            equity_curve=self.equity_curve,
            trades=self.trades,
            initial_capital=self.initial_cash
        )
        results.print_summary()
        
        results.export_json("results/momentum_backtest_results.json")
        results.export_trades_csv("results/momentum_trades.csv")


def main():
    data_file = "data/processed/reliance.bin"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        print("Run: python scripts/load_datasets.py --dataset reliance --output data/processed/reliance.bin")
        return
    
    os.makedirs("results", exist_ok=True)
    
    stream = fe.DataStream()
    stream.load(data_file)
    print(f"Loaded {data_file}")
    
    slippage = fe.SlippageConfig()
    slippage.fixed_bps = 5.0
    
    engine = fe.MatchingEngine(slippage)
    portfolio = fe.Portfolio(100000.0)
    
    risk_limits = fe.RiskLimits()
    risk_limits.max_drawdown = 0.20
    risk_limits.max_position_size = 1000
    risk_limits.max_order_size = 500
    risk_engine = fe.RiskEngine(risk_limits)
    
    print(f"Initial Capital: $100,000")
    print(f"Risk Limits: Max DD={risk_limits.max_drawdown*100}%, Max Position={risk_limits.max_position_size}")
    
    strategy = MomentumBreakoutStrategy(engine, portfolio, risk_engine)
    
    strategy.on_start()
    
    loop = fe.EventLoop()
    loop.run(stream, strategy)
    
    strategy.on_end()


if __name__ == "__main__":
    main()