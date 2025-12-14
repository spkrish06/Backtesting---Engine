# Felix Backtester

Production-grade HFT backtesting engine with C++20 core and Python strategy interface.

## Features

- **C++ Event Loop** - Deterministic, microsecond-level tick processing
- **Python Strategy API** - `on_start`, `on_tick`, `on_bar`, `on_fill`, `on_end`
- **Slippage & Latency Models** - Realistic execution simulation
- **Portfolio & Risk Engine** - Position tracking, P&L, max drawdown limits
- **Analytics** - Sharpe, Sortino, Max Drawdown, CAGR, Win Rate
- **Export** - JSON/CSV trade logs and equity curves

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build C++ engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..
copy build\Release\*.pyd .

# 3. Load local market data
python scripts/load_datasets.py --dataset reliance

# 4. Run backtest (Winning Strategy: Momentum)
python scripts/run_backtest_momentum.py
```

## Output

```text
Momentum Breakout Strategy Complete
============================================================

==================================================
BACKTEST RESULTS (Reliance Industries)
==================================================
Duration:           3235 days
Initial Capital:    $100,000.00
Final Equity:       $1,603,302.00
Total P&L:          $+1,503,302.00 (+1503.30%)
--------------------------------------------------
Sharpe Ratio:       8.27
Sortino Ratio:      8.06
Max Drawdown:       1.52%
CAGR:               36.79%
--------------------------------------------------
Total Trades:       35739
Win Rate:           19.9%
Avg Win:            $244.98
Avg Loss:           $-96.83
Profit Factor:      2.53
==================================================
```

Results exported to:
- `results/backtest_results.json`
- `results/trades.csv`

## Project Structure

```
Felix-Backtesting-Engine-New/
├── engine/           # C++ core (event loop, matching, portfolio)
├── python/felix/     # Python strategy API and analytics
├── scripts/          # Data preprocessing and backtest runner
├── data/             # Binary market data
├── results/          # Output JSON/CSV
└── tests/            # C++ and Python tests
```

## Writing Strategies

```python
from felix.strategy.base import Strategy
import felix_engine as fe

class MyStrategy(Strategy):
    def on_tick(self, tick):
        # Example: Simple Buy Logic
        if tick.price < 500:
            shares = 100
            cost = shares * tick.price
            
            # 1. Check Cash
            if self.portfolio.cash() >= cost:
                # 2. Execute Trade (Update Portfolio)
                fill = fe.Fill()
                fill.symbol_id = tick.symbol_id
                fill.price = tick.price
                fill.volume = float(shares)
                fill.side = fe.Side.BUY
                
                self.portfolio.on_fill(fill)
                print(f"Bought {shares} @ {tick.price}")
```

## License

By MIT
