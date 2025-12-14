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

Backtest started.
  [ENTRY] BUY 100 @ $522.90 | Breakout High=$522.10 | Stop=$507.21 | Target=$549.04
  [EXIT] SELL 100 @ $549.04 | TARGET | P&L: $+2614.50
  ...
  [RISK] Halted - Max drawdown exceeded!

==================================================
BACKTEST RESULTS
==================================================
Initial Capital:    $100,000.00
Final Equity:       $152,149.70
Total P&L:          $+52,149.70 (+52.15%)
--------------------------------------------------
Sharpe Ratio:       0.04
Max Drawdown:       21.48% (Halted)
CAGR:               0.01%
Win Rate:           22.9%
--------------------------------------------------

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
