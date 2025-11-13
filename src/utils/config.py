"""
Configuration management for the LLM Trading Platform.

This module handles all configuration settings using Pydantic for type safety
and validation. It loads from environment variables and provides sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    """
    Main configuration class for the trading platform.

    All settings are loaded from environment variables (or .env file).
    This ensures sensitive data like API keys never gets committed to git.
    """

    # ===== API Keys =====
    groq_api_key: str = Field(..., description="Groq API key for fast LLM inference")
    anthropic_api_key: str = Field(..., description="Anthropic API key for complex reasoning")
    alpaca_api_key: str = Field(..., description="Alpaca API key for paper trading")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL (paper trading by default)"
    )
    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API key (optional, for backup market data)"
    )

    # ===== Kafka Configuration =====
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka broker addresses"
    )
    kafka_topic_news: str = Field(default="market-news")
    kafka_topic_filings: str = Field(default="sec-filings")
    kafka_topic_signals: str = Field(default="trading-signals")

    # ===== Database Configuration =====
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="trading_db")
    postgres_user: str = Field(default="trading_user")
    postgres_password: str = Field(default="trading_pass")

    # ===== Redis Configuration =====
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    # ===== MLflow Configuration =====
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="llm-trading-strategies")

    # ===== Trading Parameters =====
    paper_trading: bool = Field(
        default=True,
        description="ALWAYS use paper trading unless you know what you're doing!"
    )
    max_position_size: float = Field(
        default=10000.0,
        description="Maximum dollar amount per position"
    )
    risk_per_trade: float = Field(
        default=0.02,
        description="Risk 2% of portfolio per trade"
    )
    trading_hours_only: bool = Field(
        default=True,
        description="Only trade during market hours"
    )
    max_open_positions: int = Field(
        default=10,
        description="Maximum number of concurrent positions"
    )

    # ===== LLM Cost Controls =====
    max_tokens_per_analysis: int = Field(
        default=2000,
        description="Limit tokens to control costs"
    )
    cache_analysis_hours: int = Field(
        default=24,
        description="Cache LLM analysis for 24 hours to avoid redundant API calls"
    )
    use_groq_for_speed: bool = Field(
        default=True,
        description="Use Groq for fast, cheap inference (recommended)"
    )
    use_anthropic_for_complex: bool = Field(
        default=True,
        description="Use Claude only for complex reasoning tasks"
    )

    # ===== Logging Configuration =====
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/trading.log")

    # ===== Watchlist =====
    default_watchlist: List[str] = Field(
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"],
        description="Default stocks to monitor"
    )

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @validator("paper_trading")
    def validate_paper_trading(cls, v):
        """Ensure paper trading is enabled by default for safety."""
        if not v:
            print("⚠️  WARNING: Real trading is enabled! Use with extreme caution.")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
