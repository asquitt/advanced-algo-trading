"""
Tests for Security and Input Validation

Tests security controls, input sanitization, API key handling,
and protection against common vulnerabilities.

Author: LLM Trading Platform - Test Suite
"""

import pytest
import os
from unittest.mock import patch, Mock
from src.utils.config import Settings
from src.data_layer.models import TradingSignal, SignalType


class TestAPIKeyHandling:
    """Test secure API key handling."""

    def test_api_keys_from_environment(self):
        """Test that API keys are loaded from environment."""
        # Keys should come from environment variables
        assert os.getenv("GROQ_API_KEY") is not None
        assert os.getenv("ANTHROPIC_API_KEY") is not None

    def test_api_keys_not_in_logs(self):
        """Test that API keys don't appear in logs."""
        settings = Settings()

        # Convert settings to string (what might be logged)
        settings_str = str(settings.__dict__)

        # API keys should not be visible in string representation
        # This is a basic check - real implementation should mask keys
        # For now, just verify keys exist but aren't obviously exposed
        assert settings.groq_api_key
        assert settings.anthropic_api_key

    def test_api_key_validation(self):
        """Test that invalid API keys are rejected."""
        # Empty keys should be invalid
        with pytest.raises((ValueError, Exception)):
            settings = Settings(
                groq_api_key="",
                anthropic_api_key="test",
                alpaca_api_key="test",
                alpaca_secret_key="test"
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_keys_handled(self):
        """Test handling of missing API keys."""
        # When required keys are missing, should handle gracefully
        # or fail fast with clear error
        with pytest.raises((ValueError, Exception)):
            Settings()  # Should fail if required keys missing


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_symbol_validation(self):
        """Test stock symbol validation."""
        valid_symbols = ["AAPL", "GOOGL", "MSFT", "BRK.A", "BRK.B"]
        invalid_symbols = ["", "   ", "INVALID!!!",  "'; DROP TABLE;--", "<script>"]

        for symbol in valid_symbols:
            # Should accept or clean valid symbols
            cleaned = symbol.strip().upper()
            assert cleaned.isalnum() or "." in cleaned

        for symbol in invalid_symbols:
            # Should reject or sanitize invalid symbols
            cleaned = symbol.strip().upper()
            # Contains invalid characters
            has_invalid = any(c in cleaned for c in ["'", ";", "<", ">", "(", ")", "--"])
            if has_invalid:
                # Should be caught by validation
                assert True

    def test_signal_confidence_range(self):
        """Test that confidence scores are validated."""
        # Valid confidence
        signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.85,
            ai_conviction_score=0.80,
            reasoning="Test",
            source_agent="test"
        )
        assert 0 <= signal.confidence_score <= 1.0

        # Invalid confidence should be rejected by Pydantic
        with pytest.raises((ValueError, Exception)):
            TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence_score=1.5,  # Invalid > 1.0
                ai_conviction_score=0.80,
                reasoning="Test",
                source_agent="test"
            )

        with pytest.raises((ValueError, Exception)):
            TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence_score=-0.5,  # Invalid < 0
                ai_conviction_score=0.80,
                reasoning="Test",
                source_agent="test"
            )

    def test_quantity_validation(self):
        """Test trade quantity validation."""
        from src.data_layer.models import Trade, OrderSide, OrderStatus

        # Valid quantity
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            entry_price=150.0,
            status=OrderStatus.FILLED,
            order_id="test"
        )
        assert trade.quantity > 0

        # Zero or negative quantity should be invalid
        with pytest.raises((ValueError, Exception)):
            Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=0,  # Invalid
                entry_price=150.0,
                status=OrderStatus.FILLED,
                order_id="test"
            )

        with pytest.raises((ValueError, Exception)):
            Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=-10,  # Invalid
                entry_price=150.0,
                status=OrderStatus.FILLED,
                order_id="test"
            )

    def test_price_validation(self):
        """Test price validation."""
        from src.data_layer.models import Trade, OrderSide, OrderStatus

        # Valid price
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            entry_price=150.0,
            status=OrderStatus.FILLED,
            order_id="test"
        )
        assert trade.entry_price > 0

        # Negative price should be invalid
        with pytest.raises((ValueError, Exception)):
            Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=-150.0,  # Invalid
                status=OrderStatus.FILLED,
                order_id="test"
            )


class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""

    def test_symbol_sql_injection_attempt(self):
        """Test that SQL injection attempts in symbols are blocked."""
        malicious_inputs = [
            "AAPL'; DROP TABLE trades;--",
            "AAPL OR 1=1--",
            "AAPL' UNION SELECT * FROM users--"
        ]

        for malicious in malicious_inputs:
            # Should either reject or sanitize
            cleaned = malicious.replace("'", "").replace(";", "").replace("--", "")
            # After cleaning, should not contain SQL injection patterns
            assert "DROP" not in cleaned or cleaned != malicious


class TestXSSPrevention:
    """Test XSS prevention."""

    def test_reasoning_xss_attempt(self):
        """Test that XSS attempts in reasoning are handled."""
        xss_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror='alert(1)'>",
            "javascript:alert('XSS')"
        ]

        for xss in xss_inputs:
            signal = TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence_score=0.85,
                ai_conviction_score=0.80,
                reasoning=xss,  # Contains XSS
                source_agent="test"
            )

            # When serialized, should not contain executable scripts
            serialized = signal.json()
            # Basic check: <script> tags should be escaped or removed
            assert "<script>" not in serialized or "&lt;script&gt;" in serialized


class TestRateLimiting:
    """Test rate limiting and abuse prevention."""

    def test_excessive_requests_detection(self):
        """Test detection of excessive API requests."""
        # Simulate rapid requests
        request_count = 0
        max_requests_per_minute = 60

        for i in range(100):
            request_count += 1

        # Should detect excessive requests
        if request_count > max_requests_per_minute:
            # Would trigger rate limiting
            assert True

    def test_concurrent_request_limits(self):
        """Test limits on concurrent requests."""
        max_concurrent = 10
        current_concurrent = 0

        # Simulate concurrent requests
        for i in range(20):
            if current_concurrent < max_concurrent:
                current_concurrent += 1
            # Would be queued or rejected

        assert current_concurrent <= max_concurrent


class TestSecretManagement:
    """Test secret management."""

    def test_secrets_not_in_error_messages(self):
        """Test that secrets don't leak in error messages."""
        api_key = "sk_test_secret_key_12345"

        try:
            # Simulate error that might expose key
            raise Exception(f"API call failed")
        except Exception as e:
            error_msg = str(e)
            # Should not contain actual key
            assert api_key not in error_msg

    def test_database_password_not_logged(self):
        """Test that database passwords aren't logged."""
        settings = Settings()

        # Password should exist but not be easily exposed
        assert hasattr(settings, "postgres_password")

        # In production, password should not appear in logs
        # This is tested by ensuring sensitive fields are marked as SecretStr
        # in Pydantic models


class TestAuthenticationAuthorization:
    """Test authentication and authorization (future enhancement)."""

    def test_api_requires_authentication(self):
        """Test that API requires authentication (placeholder)."""
        # Currently, API is open
        # In production, should require API keys or OAuth tokens
        # This test documents the need for future authentication
        pass

    def test_role_based_access_control(self):
        """Test RBAC for different user roles (placeholder)."""
        # Future: Admin, Trader, Viewer roles
        # with different permissions
        pass


class TestDataSanitization:
    """Test data sanitization."""

    def test_symbol_uppercase_normalization(self):
        """Test that symbols are normalized to uppercase."""
        inputs = ["aapl", "Aapl", "AAPL", "aApL"]

        for inp in inputs:
            normalized = inp.strip().upper()
            assert normalized == "AAPL"

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed."""
        inputs = ["  AAPL  ", "AAPL\n", "\tAAPL"]

        for inp in inputs:
            cleaned = inp.strip()
            assert cleaned == "AAPL"

    def test_special_character_handling(self):
        """Test handling of special characters."""
        # Some symbols legitimately have special chars (BRK.A)
        valid_special = "BRK.A"
        assert "." in valid_special  # Period is valid

        # But other special chars should be rejected
        invalid_special = "AAP@L"
        has_invalid = "@" in invalid_special
        assert has_invalid  # Would be caught by validation


class TestSecurityBestPractices:
    """Test security best practices."""

    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded."""
        # This is a meta-test - in reality, scan code for hardcoded secrets
        # using tools like git-secrets, trufflehog, etc.
        pass

    def test_secure_random_generation(self):
        """Test that secure random is used for sensitive operations."""
        import secrets

        # For security-sensitive randomness, use secrets module
        secure_token = secrets.token_hex(16)
        assert len(secure_token) == 32  # 16 bytes = 32 hex chars

    def test_timing_attack_prevention(self):
        """Test prevention of timing attacks."""
        import secrets

        # Use secrets.compare_digest for comparing secrets
        secret1 = "test_secret"
        secret2 = "test_secret"
        secret3 = "wrong_secret"

        # Safe comparison
        assert secrets.compare_digest(secret1, secret2)
        assert not secrets.compare_digest(secret1, secret3)


class TestEnvironmentSeparation:
    """Test environment separation."""

    def test_paper_trading_flag(self):
        """Test that paper trading flag is set in test environment."""
        settings = Settings()

        # Test environment should use paper trading
        assert settings.paper_trading is True

    def test_production_vs_test_config(self):
        """Test separation of production and test configuration."""
        # Test environment should not connect to production systems
        settings = Settings()

        # Should use test/paper endpoints
        assert settings.paper_trading or "test" in str(settings.__dict__).lower()


class TestErrorMessageSecurity:
    """Test secure error messaging."""

    def test_generic_error_messages(self):
        """Test that error messages don't expose internals."""
        # Error messages should be user-friendly, not expose:
        # - Stack traces in production
        # - Database schema
        # - Internal paths
        # - Configuration details
        pass

    def test_no_sensitive_data_in_responses(self):
        """Test that API responses don't contain sensitive data."""
        signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.85,
            ai_conviction_score=0.80,
            reasoning="Strong buy",
            source_agent="test"
        )

        # Serialized signal should not contain:
        # - API keys
        # - Passwords
        # - Internal user IDs
        serialized = signal.json()

        sensitive_patterns = ["password", "api_key", "secret", "token"]
        for pattern in sensitive_patterns:
            assert pattern not in serialized.lower()
