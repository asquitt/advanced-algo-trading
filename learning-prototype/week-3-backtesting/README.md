# Week 3: Backtesting Engine

Build a production-grade vectorized backtesting framework from scratch!

## Learning Objectives

By the end of this week, you will:

✅ Understand backtesting fundamentals and common pitfalls
✅ Implement comprehensive performance metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
✅ Model realistic transaction costs (slippage, commission, spread)
✅ Build vectorized backtesting using NumPy/Pandas (no loops!)
✅ Implement walk-forward analysis for out-of-sample validation
✅ Create parameter optimization with grid search
✅ Avoid look-ahead bias and other backtesting errors

## Why Backtesting Matters

> "In God we trust. All others must bring data." - W. Edwards Deming

Backtesting is how we:
- Validate trading ideas before risking real money
- Understand strategy performance characteristics
- Identify potential issues and edge cases
- Optimize parameters systematically
- Build confidence in our approach

**Common mistake**: Over-optimizing to past data (curve fitting)
**Solution**: Walk-forward analysis and out-of-sample testing

## Prerequisites

- Python basics (Week 1)
- Pandas and NumPy fundamentals
- Basic statistics (mean, standard deviation)
- Understanding of financial metrics (returns, volatility)

## Folder Structure

```
week-3-backtesting/
├── README.md (you are here)
├── CONCEPTS.md (key concepts explained)
├── starter-code/
│   ├── performance_metrics.py  (30 TODOs)
│   ├── transaction_costs.py    (20 TODOs)
│   └── backtesting_engine.py   (40 TODOs)
├── exercises/
│   ├── exercise_1_metrics.py
│   ├── exercise_2_costs.py
│   ├── exercise_3_engine.py
│   └── exercise_4_walkforward.py
├── solutions/
│   ├── performance_metrics_complete.py
│   ├── transaction_costs_complete.py
│   └── backtesting_engine_complete.py
└── tests/
    ├── test_metrics.py
    ├── test_costs.py
    └── test_integration.py
```

## Learning Path

### Day 1: Performance Metrics (4 hours)

**Morning**: Understand metrics
- Read `CONCEPTS.md` - Performance Metrics section
- Study Sharpe, Sortino, Calmar ratios
- Understand VaR and CVaR

**Afternoon**: Implement metrics
- Complete `starter-code/performance_metrics.py`
- Fill in 30 TODOs
- Run tests: `pytest tests/test_metrics.py`

**Resources**:
- [Sharpe Ratio Explained](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Understanding VaR](https://www.investopedia.com/terms/v/var.asp)

### Day 2: Transaction Costs (3 hours)

**Morning**: Learn cost modeling
- Read `CONCEPTS.md` - Transaction Costs section
- Understand bid-ask spread, slippage, market impact

**Afternoon**: Implement cost model
- Complete `starter-code/transaction_costs.py`
- Fill in 20 TODOs
- Run tests: `pytest tests/test_costs.py`

**Key insight**: Ignoring transaction costs leads to unrealistic backtest results!

### Day 3-4: Backtesting Engine (6 hours)

**Day 3 Morning**: Understand vectorization
- Why vectorization? (10-100x faster than loops)
- NumPy and Pandas operations
- Avoiding look-ahead bias

**Day 3 Afternoon**: Build core engine
- Complete `starter-code/backtesting_engine.py` (Part 1)
- Implement basic backtest logic
- TODOs 1-20

**Day 4**: Advanced features
- Complete remaining TODOs 21-40
- Implement walk-forward analysis
- Add parameter optimization
- Run integration tests

### Day 5: Exercises & Integration (2 hours)

Complete all exercises:
```bash
# Exercise 1: Calculate metrics for sample strategy
python exercises/exercise_1_metrics.py

# Exercise 2: Compare costs across strategies
python exercises/exercise_2_costs.py

# Exercise 3: Backtest a simple MA crossover
python exercises/exercise_3_engine.py

# Exercise 4: Walk-forward analysis
python exercises/exercise_4_walkforward.py
```

## Key Concepts

### 1. Look-Ahead Bias

**WRONG** ❌:
```python
# Using today's signal with today's return
signals = calculate_signals(data)
returns = signals * data['returns']  # BIAS!
```

**CORRECT** ✅:
```python
# Shift signal by 1 day (trade tomorrow)
signals = calculate_signals(data)
positions = signals.shift(1)  # Trade next day
returns = positions * data['returns']  # No bias!
```

### 2. Compound vs Simple Returns

**Simple Returns**:
```python
total_return = returns.sum()  # Just add them up
```

**Compound Returns** (more realistic):
```python
total_return = (1 + returns).prod() - 1  # Multiply
```

### 3. Drawdown

Maximum peak-to-trough decline:
```python
running_max = equity.expanding().max()
drawdown = (equity - running_max) / running_max
max_drawdown = drawdown.min()  # Most negative
```

### 4. Sharpe Ratio

Risk-adjusted return:
```
Sharpe = (Return - Risk_Free_Rate) / Volatility

Interpretation:
< 1.0: Not good
1.0-2.0: Good
> 2.0: Excellent
```

### 5. Walk-Forward Analysis

Prevents over-fitting:
```
1. Train on data: Jan-Jun 2023
2. Test on data: Jul 2023
3. Move window forward
4. Repeat
5. Average all out-of-sample results
```

## Common Pitfalls

### Pitfall #1: Overfitting
**Problem**: Perfect backtest, terrible live performance
**Solution**: Use walk-forward, limit parameters, demand large sample

### Pitfall #2: Transaction Costs
**Problem**: Ignoring costs shows unrealistic returns
**Solution**: Model all costs (commission, slippage, spread)

### Pitfall #3: Survivorship Bias
**Problem**: Only testing stocks that survived
**Solution**: Include delisted stocks in dataset

### Pitfall #4: Look-Ahead Bias
**Problem**: Using future information in past decisions
**Solution**: Always shift signals before calculating returns

### Pitfall #5: Data Snooping
**Problem**: Testing too many ideas on same data
**Solution**: Reserve out-of-sample data, use proper statistics

## Success Criteria

You've mastered Week 3 when you can:

✅ Implement all performance metrics from scratch
✅ Explain why we shift signals to avoid look-ahead bias
✅ Calculate realistic transaction costs
✅ Run a vectorized backtest 100x faster than loop-based
✅ Implement walk-forward analysis
✅ Optimize parameters without over-fitting
✅ Pass all integration tests

## Testing Your Knowledge

```bash
# Run all tests
pytest week-3-backtesting/tests/ -v

# Expected output:
# ✓ test_sharpe_ratio_calculation
# ✓ test_sortino_ratio_calculation
# ✓ test_var_calculation
# ✓ test_cvar_calculation
# ✓ test_transaction_costs
# ✓ test_vectorized_backtest
# ✓ test_walk_forward_analysis
# ✓ test_parameter_optimization
#
# 15 passed in 2.34s
```

## Next Steps

After completing Week 3:
- **Week 4**: Use your backtesting engine to test real strategies!
- **Build a strategy**: Create your own and backtest it
- **Read paper**: ["The Deflated Sharpe Ratio"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) by Lopez de Prado

## Resources

### Articles
- [Quantopian Lectures on Backtesting](https://www.quantopian.com/lectures)
- [Common Backtesting Mistakes](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-I/)

### Books
- "Advances in Financial Machine Learning" by Marcos Lopez de Prado (Chapter 11: Backtesting)
- "Quantitative Trading" by Ernest Chan (Chapter 4: Backtesting)

### Papers
- ["Pseudo-Mathematics and Financial Charlatanism"](https://arxiv.org/abs/1105.3822) (warns about backtesting pitfalls)

## Questions?

Common questions and answers:

**Q: Why not just use backtrader or zipline?**
A: Building from scratch teaches you the fundamentals. You'll understand what's happening under the hood.

**Q: How much historical data do I need?**
A: Minimum 2-3 years daily data. More is better. For intraday: months to a year.

**Q: What's a good Sharpe ratio?**
A:
- < 1: Not good
- 1-2: Good
- > 2: Excellent
- > 3: Suspicious (check for errors!)

**Q: Can I skip transaction costs for now?**
A: NO! This is the #1 mistake. Always model costs.

**Happy backtesting! 📊**
