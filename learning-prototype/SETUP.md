# 🛠️ Learning Prototype - Complete Setup Guide

**Get your development environment ready for the 8-week journey**

---

## Prerequisites

### Required
- **Python 3.11+** (Check with `python --version`)
- **pip** package manager
- **Git** for version control
- **Text Editor** (VS Code recommended)
- **Terminal** (bash/zsh)
- **Internet Connection** for API access

### Recommended
- **Docker Desktop** (for Week 7)
- **Postman or Thunder Client** (API testing)
- **Jupyter Notebook** (for exploration)
- **PostgreSQL Client** (for Week 3+)

### Free API Keys Needed
You'll need these accounts (all have free tiers):
- **Alpaca** - Paper trading (100% free)
- **Groq** - Fast LLM inference (free tier: 30 req/min)
- **Anthropic** - Claude API (free tier: $5 credit)

---

## Step-by-Step Setup

### 1. Clone or Download

```bash
# If you have git
git clone <repository-url>
cd learning-prototype

# Or download ZIP and extract
```

### 2. Create Virtual Environment

**Why?** Isolates dependencies, prevents conflicts.

```bash
# Create venv
python -m venv venv

# Activate venv
# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# You should see (venv) in your prompt
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# This installs:
# - fastapi (web framework)
# - uvicorn (ASGI server)
# - pydantic (data validation)
# - alpaca-py (trading API)
# - groq (LLM API)
# - anthropic (Claude API)
# - pytest (testing)
# - pandas (data analysis)
# - numpy (numerical computing)
# - And more...
```

### 4. Get API Keys

#### Alpaca (Paper Trading) - FREE FOREVER

1. Go to https://alpaca.markets
2. Click "Sign Up" → "Paper Trading" (NOT live trading)
3. Verify email
4. Go to "Dashboard" → "Your API Keys"
5. Copy:
   - API Key ID: `PKXXXXXXXXXXXXX`
   - Secret Key: `XXXXXXXXXXXXXXXXX`
6. Save these securely!

**Cost**: $0 (paper trading is free forever)

#### Groq (Fast LLM) - FREE TIER

1. Go to https://console.groq.com
2. Sign up with Google/GitHub
3. Go to "API Keys"
4. Click "Create API Key"
5. Copy key: `gsk_XXXXXXXXXXXXXXXXXX`

**Cost**:
- Free Tier: 30 requests/min, 14,400/day
- Paid (if needed): ~$0.0001 per 1K tokens

#### Anthropic (Claude) - FREE CREDIT

1. Go to https://console.anthropic.com
2. Sign up
3. Go to "API Keys"
4. Click "Create Key"
5. Copy key: `sk-ant-XXXXXXXXXXXXXXXXXX`

**Cost**:
- Free: $5 credit (lasts weeks)
- Paid (if needed): ~$3 per 1M tokens

### 5. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env
nano .env  # or use your favorite editor
```

**Add your keys to .env**:
```bash
# Alpaca
ALPACA_API_KEY=PKXXXXXXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXX
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Groq
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXX

# Anthropic
ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXXXX

# Paper Trading (KEEP TRUE FOR SAFETY!)
PAPER_TRADING=true

# Risk Parameters
MAX_POSITION_SIZE=10000.0
RISK_PER_TRADE=0.02
MAX_OPEN_POSITIONS=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log
```

**⚠️ IMPORTANT**: Never commit .env to git! It's in .gitignore.

### 6. Test Your Setup

```bash
# Test Python
python --version
# Should show 3.11+

# Test venv is active
which python
# Should show path to venv/bin/python

# Test imports
python -c "import fastapi, alpaca, groq, anthropic; print('✅ All imports work!')"

# Test API keys
python test_api_keys.py
# Should show all keys valid
```

### 7. Verify Week 1 Setup

```bash
cd week-1-foundations

# Check structure
ls
# Should show: starter-code/ exercises/ solutions/ notes/ scripts/

# Try running starter code (will have TODOs)
cd starter-code
python main.py

# Should show errors about TODOs - this is expected!
```

---

## Troubleshooting

### Issue: "python: command not found"

**Solution**:
```bash
# Try python3 instead
python3 --version
python3 -m venv venv

# Or install Python from python.org
```

### Issue: "pip: command not found"

**Solution**:
```bash
# Try pip3
pip3 install -r requirements.txt

# Or
python -m pip install -r requirements.txt
```

### Issue: "Permission denied"

**Solution**:
```bash
# On Mac/Linux, use sudo (carefully)
sudo pip install -r requirements.txt

# Better: fix permissions
sudo chown -R $USER venv
```

### Issue: "Module not found" when running code

**Solution**:
```bash
# Make sure venv is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "API key invalid"

**Solution**:
```bash
# Check .env file has no extra spaces/quotes
# Should be:
GROQ_API_KEY=gsk_abcd123...

# NOT:
GROQ_API_KEY="gsk_abcd123..."  # No quotes!
GROQ_API_KEY = gsk_abcd123...  # No spaces!

# Regenerate key if needed
```

### Issue: "Port 8000 already in use"

**Solution**:
```bash
# Find process using port
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill it
kill -9 <PID>  # Mac/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
uvicorn main:app --port 8001
```

---

## Editor Setup (VS Code Recommended)

### Install VS Code Extensions

1. **Python** by Microsoft
2. **Pylance** (Python language server)
3. **Python Docstring Generator**
4. **autoDocstring**
5. **Thunder Client** (API testing)
6. **GitLens** (Git visualization)
7. **Error Lens** (inline errors)

### Configure VS Code

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

---

## Testing Your Setup is Complete

Run this comprehensive test:

```bash
# From learning-prototype/ directory

# 1. Check Python version
python --version

# 2. Check venv active
echo $VIRTUAL_ENV  # Should show path

# 3. Test all imports
python << EOF
import fastapi
import pydantic
import alpaca
import groq
import anthropic
import pandas
import numpy
import pytest
print("✅ All imports successful!")
EOF

# 4. Test API keys
python << EOF
import os
from dotenv import load_dotenv
load_dotenv()

keys_to_check = [
    'ALPACA_API_KEY',
    'ALPACA_SECRET_KEY',
    'GROQ_API_KEY',
    'ANTHROPIC_API_KEY'
]

for key in keys_to_check:
    value = os.getenv(key)
    if value:
        print(f"✅ {key}: {value[:10]}...")
    else:
        print(f"❌ {key}: MISSING")
EOF

# 5. Test FastAPI
cd week-1-foundations/starter-code
python -c "from fastapi import FastAPI; app = FastAPI(); print('✅ FastAPI works!')"

# All checks passed? You're ready! 🎉
```

---

## Cost Optimization Tips

### Keep Costs Near $0

1. **Use Paper Trading Only**
   - Never use real money during learning
   - Alpaca paper trading is 100% free

2. **Minimize LLM API Calls**
   - Use caching (Week 2)
   - Batch requests
   - Use Groq for simple tasks (nearly free)
   - Use Claude only for complex reasoning

3. **Weekly Cost Breakdown**:
   - Week 1: $0 (no LLM calls)
   - Week 2: $2-5 (learning LLM APIs)
   - Week 3-8: $5-10/week (with caching)
   - **Total 8 weeks: $30-60**

### Free Alternatives

If you want 100% free:
- Skip Anthropic, use Groq only
- Use ollama for local LLMs (free but slower)
- Mock LLM responses in tests

---

## Development Workflow

### Recommended Daily Routine

```bash
# 1. Start day
cd learning-prototype/week-X
source ../venv/bin/activate

# 2. Read notes
cat notes/topic.md

# 3. Code with tests
python starter-code/file.py
pytest exercises/

# 4. Test frequently
pytest --cov

# 5. Validate progress
./scripts/test_weekX.sh

# 6. Commit (optional)
git add .
git commit -m "Completed exercise 1"

# 7. End day
deactivate  # Exit venv
```

### Git Best Practices

```bash
# Create your own branch
git checkout -b my-learning

# Commit often
git commit -m "Week 1 Day 2: Completed Pydantic models"

# Don't push API keys!
# .env is in .gitignore - verify:
git status  # .env should NOT appear

# Compare with solutions
git diff solutions/main_solution.py starter-code/main.py
```

---

## Additional Tools Setup

### Postman / Thunder Client

For testing APIs:

1. Install Thunder Client in VS Code
2. Import collection: `week-1-foundations/api-collection.json`
3. Test endpoints as you build them

### Jupyter Notebook (Optional)

For exploration:

```bash
pip install jupyter

# Start notebook
jupyter notebook

# Create new notebook
# Import your code
from starter_code.models import TradingSignal
signal = TradingSignal(...)
```

### PostgreSQL (Week 3+)

```bash
# Mac
brew install postgresql

# Ubuntu
sudo apt install postgresql

# Or use Docker
docker run -p 5432:5432 -e POSTGRES_PASSWORD=trading_pass postgres
```

---

## Verification Checklist

Before starting Week 1, verify:

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (requirements.txt)
- [ ] API keys obtained and added to .env
- [ ] .env file not tracked by git
- [ ] Test script runs successfully
- [ ] VS Code (or editor) configured
- [ ] Week 1 folder structure exists
- [ ] Can import fastapi, pydantic, alpaca

**All checked? You're ready to start Week 1!** 🚀

---

## Getting Help

### Resources
- Python docs: https://docs.python.org/3/
- FastAPI docs: https://fastapi.tiangolo.com/
- VS Code Python: https://code.visualstudio.com/docs/python

### Community
- Reddit: r/algotrading, r/learnprogramming
- Discord: FastAPI, Python communities
- Stack Overflow: tag questions with `fastapi`, `python`, `alpaca`

### Debugging
1. Read error message carefully
2. Google exact error message
3. Check docs
4. Ask in community
5. Review solutions (last resort)

---

## Next Steps

Setup complete? Here's what to do:

1. **Read** `learning-prototype/README.md` for overview
2. **Review** `CONCEPTS.md` for key terms
3. **Start** `week-1-foundations/README.md`
4. **Code** through Week 1 starter code
5. **Test** with exercises and scripts
6. **Proceed** to Week 2 when validated

---

**Setup time: 30-60 minutes**
**Worth it: Absolutely! 💪**

**Let's build something amazing! Happy coding! 🎉**
