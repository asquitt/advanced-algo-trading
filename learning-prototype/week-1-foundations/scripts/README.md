# Week 1 Foundations - Scripts

This directory contains helper scripts to test, validate, and run your Week 1 learning materials.

## Available Scripts

### 1. test_week1.sh

**Comprehensive test script** that validates your Week 1 completion.

**What it does:**
- ✓ Checks Python syntax across all files
- ✓ Identifies remaining TODOs in starter code
- ✓ Validates required imports and dependencies
- ✓ Verifies starter code structure (models, endpoints, classes)
- ✓ Checks exercise files
- ✓ Validates environment configuration
- ✓ Runs pytest if available
- ✓ Provides detailed feedback and success metrics

**Usage:**
```bash
cd scripts
./test_week1.sh
```

**Example Output:**
```
========================================
Test 1: Checking Python Syntax
========================================

✓ Syntax OK: main.py
✓ Syntax OK: models.py
✓ Syntax OK: broker.py
✓ Syntax OK: config.py
```

---

### 2. run_api.sh

**API launcher script** that starts your FastAPI application with all the right settings.

**What it does:**
- ✓ Checks Python and FastAPI are installed
- ✓ Verifies main.py exists
- ✓ Loads environment variables from .env
- ✓ Checks if port is available
- ✓ Starts uvicorn server with auto-reload
- ✓ Provides helpful access information

**Usage:**
```bash
cd scripts
./run_api.sh                    # Run with defaults (port 8000)
./run_api.sh --port 3000        # Run on custom port
./run_api.sh --no-reload        # Disable auto-reload
./run_api.sh --help             # Show help
```

**Options:**
- `--port PORT` - Specify port (default: 8000)
- `--host HOST` - Specify host (default: 0.0.0.0)
- `--reload` - Enable auto-reload on code changes (default: on)
- `--no-reload` - Disable auto-reload
- `--help` - Show help message

**Example Output:**
```
╔════════════════════════════════════════════════════════════╗
║           FastAPI Trading Application Launcher            ║
║                  Week 1 - Foundations                      ║
╚════════════════════════════════════════════════════════════╝

Access Your API:
  🌐 Swagger UI:     http://localhost:8000/docs
  📚 ReDoc:          http://localhost:8000/redoc
  🏠 Root:           http://localhost:8000/
  ❤️  Health Check:  http://localhost:8000/health
```

---

### 3. validate.py

**Python validation script** that performs comprehensive checks on your implementation.

**What it checks:**
- ✓ All TODOs completed
- ✓ All required dependencies installed
- ✓ Python syntax in all files
- ✓ Required classes exist (TradingSignal, TradeRequest, etc.)
- ✓ Required functions exist (endpoints, methods)
- ✓ FastAPI endpoints defined
- ✓ Environment configuration
- ✓ Documentation files present

**Usage:**
```bash
cd scripts
python validate.py              # Normal validation
python validate.py --verbose    # Show detailed output
python validate.py --strict     # Treat warnings as failures
python validate.py --help       # Show help
```

**Options:**
- `--verbose, -v` - Enable verbose output with details
- `--strict, -s` - Treat warnings as failures
- `--dir DIR, -d DIR` - Specify Week 1 directory path

**Example Output:**
```
╔════════════════════════════════════════════════════════════════════╗
║      Week 1 Foundations - Comprehensive Validation Script         ║
╚════════════════════════════════════════════════════════════════════╝

========================================
Validating Starter Code
========================================

✓ Main module: Found main.py
✓ Syntax: main.py: Valid Python syntax
⚠ TODOs in main.py: 5 TODOs remaining
✓ Endpoint GET /signal: Defined
✓ Endpoint POST /trade: Defined
✓ Endpoint GET /portfolio: Defined
```

---

## Recommended Workflow

1. **Start Development**
   ```bash
   # Work on starter code TODOs
   code ../starter-code/main.py
   ```

2. **Test Your Work**
   ```bash
   # Run comprehensive tests
   ./test_week1.sh
   ```

3. **Run Your API**
   ```bash
   # Start the API server
   ./run_api.sh
   # Visit http://localhost:8000/docs
   ```

4. **Final Validation**
   ```bash
   # Validate everything is complete
   python validate.py --verbose
   ```

5. **Complete Exercises**
   ```bash
   # Work through each exercise
   python ../exercises/exercise_1_fastapi_basics.py
   python ../exercises/exercise_2_pydantic_models.py
   # etc.
   ```

---

## Troubleshooting

### Script won't run (Permission denied)
```bash
chmod +x test_week1.sh run_api.sh
```

### Import errors
```bash
pip install fastapi uvicorn pydantic pytest
pip install alpaca-py  # Optional, for Alpaca integration
```

### Port already in use
```bash
./run_api.sh --port 3000  # Use a different port
```

### TERM environment variable warning
This is normal in some environments. The scripts will still work correctly.

---

## Exit Codes

All scripts follow standard exit code conventions:
- `0` - Success
- `1` - Failure or errors found

You can check the exit code with:
```bash
./test_week1.sh
echo $?  # Prints exit code
```

---

## Support

If you encounter issues:

1. Check the error messages - they're designed to be helpful
2. Review the notes in `../notes/`
3. Check solutions in `../solutions/` (after trying yourself!)
4. Make sure all dependencies are installed
5. Verify your .env file is configured correctly

---

**Happy Learning!** 🚀
