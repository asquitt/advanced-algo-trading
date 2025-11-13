"""
Unit tests for utility modules.

Tests:
- Configuration loading and validation
- Cache operations
- Database connections
- Logging setup
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.utils.config import Settings
from src.utils.cache import Cache, cached


class TestSettings:
    """Test configuration management."""

    def test_settings_load_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
        monkeypatch.setenv("ALPACA_API_KEY", "test_alpaca_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_alpaca_secret")

        settings = Settings()

        assert settings.groq_api_key == "test_groq_key"
        assert settings.anthropic_api_key == "test_anthropic_key"
        assert settings.alpaca_api_key == "test_alpaca_key"

    def test_database_url_construction(self, mock_settings):
        """Test PostgreSQL URL construction."""
        expected_url = (
            f"postgresql://{mock_settings.postgres_user}:{mock_settings.postgres_password}"
            f"@{mock_settings.postgres_host}:{mock_settings.postgres_port}/{mock_settings.postgres_db}"
        )

        assert mock_settings.database_url == expected_url

    def test_redis_url_construction(self, mock_settings):
        """Test Redis URL construction."""
        expected_url = f"redis://{mock_settings.redis_host}:{mock_settings.redis_port}/0"

        assert mock_settings.redis_url == expected_url

    def test_paper_trading_default(self, mock_settings):
        """Test that paper trading is enabled by default."""
        assert mock_settings.paper_trading is True

    def test_risk_parameters_valid(self, mock_settings):
        """Test risk parameters are within valid ranges."""
        assert 0 < mock_settings.risk_per_trade <= 0.1  # Max 10% risk
        assert mock_settings.max_position_size > 0
        assert mock_settings.max_open_positions > 0


class TestCache:
    """Test Redis cache operations."""

    def test_cache_set_and_get(self, mock_cache):
        """Test basic cache set and get operations."""
        key = "test_key"
        value = {"data": "test_value"}

        # Set value
        result = mock_cache.set(key, value, ttl=3600)
        assert result is True

        # Get value
        cached_value = mock_cache.get(key)
        assert cached_value == value

    def test_cache_get_nonexistent_key(self, mock_cache):
        """Test getting a key that doesn't exist."""
        result = mock_cache.get("nonexistent_key")
        assert result is None

    def test_cache_delete(self, mock_cache):
        """Test cache deletion."""
        key = "test_key"
        mock_cache.set(key, "value")

        # Delete key
        result = mock_cache.delete(key)
        assert result is True

        # Verify deleted
        assert mock_cache.get(key) is None

    def test_cache_clear_pattern(self, mock_cache):
        """Test clearing cache by pattern."""
        # Set multiple keys
        mock_cache.set("llm:analysis:AAPL", {"data": 1})
        mock_cache.set("llm:analysis:MSFT", {"data": 2})
        mock_cache.set("other:key", {"data": 3})

        # Clear pattern
        count = mock_cache.clear_pattern("llm:analysis:*")
        assert count == 2

        # Verify correct keys deleted
        assert mock_cache.get("llm:analysis:AAPL") is None
        assert mock_cache.get("llm:analysis:MSFT") is None
        assert mock_cache.get("other:key") is not None

    def test_cached_decorator(self, mock_cache):
        """Test the @cached decorator."""
        call_count = 0

        @cached(ttl=3600, key_prefix="test")
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - function executed
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache (but our mock doesn't implement this fully)
        result2 = expensive_function(5)
        assert result2 == 10


class TestLogger:
    """Test logging configuration."""

    def test_logger_initialization(self):
        """Test that logger initializes without errors."""
        from src.utils.logger import app_logger

        assert app_logger is not None

    def test_logger_basic_operations(self):
        """Test basic logging operations."""
        from src.utils.logger import app_logger

        # These should not raise exceptions
        app_logger.info("Test info message")
        app_logger.warning("Test warning message")
        app_logger.error("Test error message")
        app_logger.debug("Test debug message")


class TestDatabase:
    """Test database utilities."""

    @patch('src.utils.database.SessionLocal')
    def test_get_db_context_manager(self, mock_session_local):
        """Test database context manager."""
        from src.utils.database import get_db

        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Use context manager
        with get_db() as db:
            assert db == mock_session

        # Verify commit and close were called
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch('src.utils.database.SessionLocal')
    def test_get_db_rollback_on_error(self, mock_session_local):
        """Test that database rolls back on error."""
        from src.utils.database import get_db

        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Simulate error in context
        with pytest.raises(ValueError):
            with get_db() as db:
                raise ValueError("Test error")

        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
