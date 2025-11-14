# 🚀 Getting Started - Quick Guide

Welcome! This guide will get you up and running in 15 minutes.

## ⚡ 15-Minute Quickstart

### Step 1: Clone and Setup (5 min)

```bash
# Clone the repository
git clone https://github.com/yourusername/reimagined-winner.git
cd reimagined-winner

# Run setup script
cd learning/scripts
chmod +x setup_environment.sh
./setup_environment.sh

# Activate virtual environment
source ../../venv/bin/activate
```

### Step 2: Configure API Keys (5 min)

```bash
# Copy environment template
cp ../.env.example ../.env

# Edit .env file
nano ../.env  # Or use your favorite editor
```

Add your API keys:
```bash
# Required for Week 2+
ANTHROPIC_API_KEY=your_claude_key_here
GROQ_API_KEY=your_groq_key_here  # Optional

# Required for trading
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
```

**Get Free API Keys:**
- **Claude API**: https://console.anthropic.com/
- **Alpaca (Paper Trading)**: https://alpaca.markets/
- **Groq** (Optional, fast/cheap): https://console.groq.com/

### Step 3: Run Your First Code (5 min)

```bash
# Go to Week 1
cd ../week1

# Run starter code
python starter.py

# You should see:
# === Testing TradingSignal ===
# ✓ Created signal: BUY AAPL at $150.0
# ...
# ✓ All manual tests passed!
```

## 🎯 What to Do Next

### Day-by-Day Path

**Today (Day 1):**
1. Read `learning/README.md` (main overview)
2. Read `week1/README.md` (week overview)
3. Open `week1/starter.py`
4. Fill in the TODO sections
5. Run tests: `pytest tests/test_day1.py -v`

**Tomorrow (Day 2):**
1. Open `week1/indicators.py`
2. Implement SMA, RSI, MACD
3. Run visualization: `python indicators.py --visualize`
4. Run tests: `pytest tests/test_day2.py -v`

**Continue this pattern for 5 weeks!**

## 📖 Learning Resources in This Repo

```
learning/
├── README.md                    # Main learning guide
├── GETTING_STARTED.md          # This file!
├── week1/                      # Week 1: Foundations
│   ├── README.md              # Week overview
│   ├── starter.py             # Fill in TODOs here
│   ├── indicators.py          # Technical indicators
│   └── tests/                 # Run to verify your code
├── week2/                      # Week 2: LLM Integration
├── week3/                      # Week 3: Risk Management
├── week4/                      # Week 4: Advanced Features
├── week5/                      # Week 5: Production
├── notes/                      # Conceptual explanations
├── exercises/                  # Extra practice problems
├── solutions/                  # Complete solutions (peek only when stuck!)
└── scripts/                    # Helper scripts
```

## 🎓 How to Use This Course

### The Learning Loop

1. **Read** the weekly README
2. **Code** by filling in TODOs in starter files
3. **Test** with `pytest tests/`
4. **Compare** with solutions (only after trying!)
5. **Repeat** daily for 5 weeks

### Tips for Success

✅ **DO:**
- Write code yourself (don't copy-paste)
- Run tests frequently
- Take notes on confusing concepts
- Ask questions in GitHub issues
- Experiment and break things!

❌ **DON'T:**
- Skip weeks (each builds on previous)
- Just read code (implement it!)
- Use real money (paper trade only!)
- Give up when stuck (check solutions, ask for help)

## 🆘 Troubleshooting

### "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "API key invalid"
```bash
# Check your .env file
cat .env

# Make sure keys are not wrapped in quotes
# GOOD: ANTHROPIC_API_KEY=sk-ant-123456
# BAD:  ANTHROPIC_API_KEY="sk-ant-123456"
```

### "Tests are failing"
```bash
# Run with verbose output to see what's wrong
pytest tests/test_day1.py -v

# Run a specific test
pytest tests/test_day1.py::test_trading_signal -v

# Check the solution
cat solutions/week1_solution.py
```

### "I'm stuck on a TODO"
1. Read the hints in the code comments
2. Check the docstring for the function
3. Run the tests to see what's expected
4. Look at similar code in the main `src/` folder
5. Check the solution (but try yourself first!)

## 📊 Progress Tracking

Create a progress file to track your journey:

```bash
touch learning/my_progress.md
```

Example format:
```markdown
# My Learning Progress

## Week 1
- [x] Day 1: Data models - 2 hours
- [x] Day 2: Indicators - 3 hours
- [ ] Day 3: Strategy
- [ ] Day 4: Backtesting
- [ ] Day 5: Review

**Challenges:** Understanding RSI calculation
**Learnings:** Now I understand momentum indicators!
```

## 🎁 Bonus: Quick Reference

### Common Commands

```bash
# Activate environment
source venv/bin/activate

# Run starter code
python starter.py

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_day1.py::test_trading_signal -v

# Check code coverage
pytest tests/ --cov=. --cov-report=html

# Visualize indicators
python indicators.py --visualize
```

### File Structure

- `starter.py` = Your code goes here (fill in TODOs)
- `tests/` = Run these to verify your code
- `solutions/` = Complete working versions
- `README.md` = Explanation and learning objectives

## 🚀 Ready to Start?

```bash
cd learning/week1
cat README.md
python starter.py
```

**Good luck, and enjoy building your AI trading system! 🎉**

---

**Need help?** Open a GitHub issue or check the solutions folder.

**Want to contribute?** Found a bug or have a suggestion? Pull requests welcome!
