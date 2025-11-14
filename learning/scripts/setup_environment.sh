#!/bin/bash
# Setup script for learning environment

echo "🎓 Setting up LLM Trading Platform Learning Environment"
echo "========================================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r ../requirements.txt
echo "✓ Dependencies installed"

# Create .env file
echo ""
echo "Creating .env file..."
if [ ! -f ../.env ]; then
    cp ../.env.example ../.env
    echo "✓ .env file created from template"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - ANTHROPIC_API_KEY"
    echo "   - ALPACA_API_KEY"
    echo "   - ALPACA_SECRET_KEY"
else
    echo "ℹ️  .env file already exists"
fi

# Run quick test
echo ""
echo "Running quick test..."
python3 -c "import pandas, numpy, pydantic; print('✓ Core libraries imported successfully')"

echo ""
echo "========================================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys: nano ../.env"
echo "2. Go to Week 1: cd learning/week1"
echo "3. Read the README: cat README.md"
echo "4. Start coding: python starter.py"
echo ""
echo "Happy learning! 🚀"
