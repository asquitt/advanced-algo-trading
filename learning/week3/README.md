# Week 3: Risk Management 🛡️

**Goal:** Implement institutional-grade risk management to protect your capital.

**Time Estimate:** 10-12 hours

## 📚 What You'll Learn

- Kelly Criterion for optimal position sizing
- CVaR (Conditional Value at Risk) calculation
- Portfolio-level risk limits
- Drawdown management
- Position concentration limits

## 🎯 Key Topics

### Day 1-2: Kelly Criterion
- Calculate optimal position size
- Fractional Kelly for safety
- Parameter estimation
- **Implement:** `kelly_calculator.py`

### Day 3-4: CVaR Risk Management
- Calculate tail risk (CVaR)
- Set position limits (≤2% CVaR)
- Portfolio limits (≤5% CVaR)
- **Implement:** `cvar_manager.py`

### Day 5: Integration
- Combine Kelly + CVaR
- Dynamic position sizing
- Risk-adjusted returns
- **Implement:** `risk_manager.py`

## 📊 Key Formula

**Kelly Criterion:**
```
f* = (p*b - q) / b

Where:
- p = win probability
- q = loss probability (1-p)
- b = win/loss ratio
- f* = optimal position fraction
```

**CVaR:**
```
CVaR = E[Loss | Loss > VaR]

Expected loss in worst 5% of cases
```

## ⏭️ Next Week

**Week 4: Advanced Features** - HFT techniques and statistical validation.
