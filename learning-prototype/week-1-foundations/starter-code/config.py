"""
Week 1 - Starter Code: Configuration Management

Your task: Create a settings class to manage API keys and configuration.

This module uses Pydantic to validate environment variables and settings.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional

# ============================================================================
# SETTINGS CLASS
# ============================================================================

# TODO #1: Create Settings class that inherits from BaseSettings
class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Your task: Add all required fields and validators.

    Why use Pydantic Settings?
    - Automatic validation
    - Type safety
    - Environment variable loading
    - Default values
    - Documentation
    """

    # ========================================================================
    # ALPACA API SETTINGS
    # ========================================================================

    # TODO #2: Add Alpaca API key fields
    # HINT:
    # alpaca_api_key: str = Field(
    #     ...,  # Required field
    #     description="Alpaca API key",
    #     env="ALPACA_API_KEY"  # Load from env var
    # )
    # alpaca_secret_key: str = Field(
    #     ...,
    #     description="Alpaca secret key",
    #     env="ALPACA_SECRET_KEY"
    # )
    # alpaca_base_url: str = Field(
    #     "https://paper-api.alpaca.markets",  # Default for paper trading
    #     description="Alpaca API base URL",
    #     env="ALPACA_BASE_URL"
    # )

    # TODO: Add fields here
    pass


    # ========================================================================
    # LLM API SETTINGS (For Week 2+)
    # ========================================================================

    # TODO #3: Add LLM API key fields (optional for Week 1)
    # HINT:
    # groq_api_key: Optional[str] = Field(
    #     None,
    #     description="Groq API key for LLM",
    #     env="GROQ_API_KEY"
    # )
    # anthropic_api_key: Optional[str] = Field(
    #     None,
    #     description="Anthropic Claude API key",
    #     env="ANTHROPIC_API_KEY"
    # )

    # TODO: Add fields here
    pass


    # ========================================================================
    # TRADING SETTINGS
    # ========================================================================

    # TODO #4: Add trading configuration fields
    # HINT:
    # paper_trading: bool = Field(
    #     True,  # ALWAYS True for learning!
    #     description="Use paper trading (not real money)",
    #     env="PAPER_TRADING"
    # )
    # max_position_size: float = Field(
    #     10000.0,
    #     description="Maximum position size in dollars",
    #     env="MAX_POSITION_SIZE"
    # )
    # risk_per_trade: float = Field(
    #     0.02,
    #     description="Risk per trade as fraction of portfolio",
    #     env="RISK_PER_TRADE"
    # )
    # max_open_positions: int = Field(
    #     10,
    #     description="Maximum number of open positions",
    #     env="MAX_OPEN_POSITIONS"
    # )

    # TODO: Add fields here
    pass


    # ========================================================================
    # LOGGING SETTINGS
    # ========================================================================

    # TODO #5: Add logging configuration
    # HINT:
    # log_level: str = Field(
    #     "INFO",
    #     description="Logging level",
    #     env="LOG_LEVEL"
    # )
    # log_file: str = Field(
    #     "logs/trading.log",
    #     description="Log file path",
    #     env="LOG_FILE"
    # )

    # TODO: Add fields here
    pass


    # ========================================================================
    # VALIDATORS
    # ========================================================================

    # TODO #6: Add validator for paper_trading
    # Ensure it's always True during learning!
    # @validator("paper_trading")
    # def ensure_paper_trading(cls, v):
    #     """Force paper trading for safety."""
    #     if not v:
    #         raise ValueError("Paper trading must be enabled for learning!")
    #     return v


    # TODO #7: Add validator for risk_per_trade
    # Ensure it's between 0 and 1
    # @validator("risk_per_trade")
    # def validate_risk(cls, v):
    #     """Ensure risk is reasonable."""
    #     if not 0 < v <= 0.1:  # Max 10% risk per trade
    #         raise ValueError("Risk per trade must be between 0 and 0.1")
    #     return v


    # TODO #8: Add validator for max_position_size
    # Ensure it's positive
    # @validator("max_position_size")
    # def validate_position_size(cls, v):
    #     """Ensure position size is positive."""
    #     if v <= 0:
    #         raise ValueError("Max position size must be positive")
    #     return v


    # ========================================================================
    # PYDANTIC CONFIG
    # ========================================================================

    class Config:
        """Pydantic configuration."""

        # TODO #9: Configure to load from .env file
        # HINT:
        # env_file = ".env"
        # env_file_encoding = "utf-8"
        # case_sensitive = False

        pass


# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================

# TODO #10: Create global settings instance
# This will be imported by other modules
# HINT:
# settings = Settings()

settings = None  # Replace with your code


# ============================================================================
# TESTING YOUR CONFIG
# ============================================================================

"""
Test your config with this code:

# 1. Create .env file with:
ALPACA_API_KEY=pk_test_123
ALPACA_SECRET_KEY=sk_test_456
PAPER_TRADING=true
MAX_POSITION_SIZE=10000.0
RISK_PER_TRADE=0.02
LOG_LEVEL=INFO

# 2. Test loading:
from config import settings

print(f"Alpaca API Key: {settings.alpaca_api_key[:10]}...")
print(f"Paper Trading: {settings.paper_trading}")
print(f"Max Position: ${settings.max_position_size}")
print(f"Risk per Trade: {settings.risk_per_trade * 100}%")

# 3. Test validation (this should fail):
try:
    bad_settings = Settings(
        alpaca_api_key="pk_123",
        alpaca_secret_key="sk_456",
        paper_trading=False  # Should fail validation!
    )
except ValueError as e:
    print(f"Validation error (expected): {e}")
"""


# ============================================================================
# ENV FILE TEMPLATE
# ============================================================================

"""
Create a .env file in your project root with these variables:

# Alpaca API (Get from https://alpaca.markets)
ALPACA_API_KEY=pk_your_key_here
ALPACA_SECRET_KEY=sk_your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# LLM APIs (Optional for Week 1)
GROQ_API_KEY=gsk_your_key_here
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Trading Settings (KEEP SAFE FOR LEARNING!)
PAPER_TRADING=true
MAX_POSITION_SIZE=10000.0
RISK_PER_TRADE=0.02
MAX_OPEN_POSITIONS=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

IMPORTANT: Add .env to .gitignore to prevent committing secrets!
"""


# ============================================================================
# HELPFUL HINTS
# ============================================================================

"""
HINT #1 - Installing Dependencies:
    pip install pydantic-settings python-dotenv

HINT #2 - Environment Variables:
    Pydantic automatically loads from:
    1. Environment variables (os.environ)
    2. .env file (if env_file specified)
    3. Default values (if provided)

HINT #3 - Required vs Optional:
    # Required (... means required)
    api_key: str = Field(...)

    # Optional with default
    log_level: str = Field("INFO")

    # Optional (can be None)
    optional_key: Optional[str] = Field(None)

HINT #4 - Validators:
    Use validators to enforce business rules:

    @validator("field_name")
    def validate_field(cls, v):
        if not valid(v):
            raise ValueError("Error message")
        return v

HINT #5 - Config Class:
    The Config class tells Pydantic how to behave:

    class Config:
        env_file = ".env"  # Load from file
        case_sensitive = False  # Ignore case
        env_prefix = "APP_"  # Prefix for env vars

HINT #6 - Security:
    NEVER commit .env file to git!

    Add to .gitignore:
    .env
    *.env
    .env.local

HINT #7 - Testing:
    Test with different env files:

    # .env.test
    ALPACA_API_KEY=test_key
    PAPER_TRADING=true

    # Load specific file:
    settings = Settings(_env_file=".env.test")

Still stuck? Check:
- ../solutions/config_solution.py
- ../notes/pydantic_explained.md
- Pydantic Settings docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
"""
