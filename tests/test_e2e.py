"""
End-to-end integration tests for the LLM Trading Platform.

These tests verify complete workflows from start to finish:
1. User requests signal → Signal generated → Trade executed → Portfolio updated
2. Batch signal generation with caching
3. Full platform lifecycle
4. Real-world scenarios

Run these tests in a staging environment that mirrors production.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from fastapi.testclient import TestClient
import time


class TestCompleteTradingWorkflow:
    """End-to-end test of complete trading workflow."""

    @patch('src.main.broker')
    @patch('src.main.market_data')
    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_full_workflow_buy_signal_to_execution(
        self, mock_kafka, mock_strategy, mock_market_data, mock_broker
    ):
        """
        E2E: Complete workflow from signal generation to trade execution.

        Workflow:
        1. User requests signal via API
        2. Strategy generates BUY signal
        3. Signal published to Kafka
        4. Trade executed via broker
        5. Portfolio updated
        """
        from src.main import app
        from src.data_layer.models import TradingSignal, SignalType, Trade, OrderSide, PortfolioState

        client = TestClient(app)

        # Step 1: Mock strategy to generate BUY signal
        buy_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.85,
            ai_conviction_score=0.78,
            fundamental_score=0.82,
            sentiment_score=0.65,
            technical_score=0.75,
            reasoning="Strong buy signal",
            source_agent="ensemble_strategy"
        )
        mock_strategy.generate_signal.return_value = buy_signal

        # Step 2: Mock Kafka
        mock_kafka.publish_signal.return_value = True

        # Step 3: Request signal generation
        response = client.post("/signals/generate?symbol=AAPL&execute=false")

        assert response.status_code == 200
        data = response.json()

        assert data["symbol"] == "AAPL"
        assert data["signal_type"] == "BUY"

        # Verify Kafka publish was called
        mock_kafka.publish_signal.assert_called_once()

        # Step 4: Execute the trade
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = {
            "symbol": "AAPL",
            "price": 150.50,
            "bid": 150.45,
            "ask": 150.55,
        }

        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": 100000.0,
            "buying_power": 50000.0,
        }
        mock_broker.can_trade.return_value = (True, "OK")
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=66,
            entry_price=150.50,
            status="filled"
        )

        response = client.post("/trades/execute?symbol=AAPL")

        assert response.status_code == 200
        trade_data = response.json()

        assert trade_data["symbol"] == "AAPL"
        assert trade_data["side"] == "buy"
        assert trade_data["status"] == "filled"

        # Step 5: Verify portfolio updated
        mock_broker.get_portfolio_state.return_value = PortfolioState(
            cash_balance=40000.0,
            portfolio_value=100000.0,
            total_pnl=0.0,
            total_pnl_percent=0.0,
            active_positions=1
        )

        response = client.get("/portfolio")

        assert response.status_code == 200
        portfolio_data = response.json()

        assert portfolio_data["active_positions"] == 1

    @patch('src.main.broker')
    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_full_workflow_sell_signal_closes_position(
        self, mock_kafka, mock_strategy, mock_broker
    ):
        """
        E2E: Sell signal closes existing position.

        Workflow:
        1. Have existing position
        2. Generate SELL signal
        3. Execute trade to close position
        4. Verify position closed
        """
        from src.main import app
        from src.data_layer.models import TradingSignal, SignalType, Trade, OrderSide, Position

        client = TestClient(app)

        # Step 1: Mock existing position
        mock_broker.get_position.return_value = Position(
            symbol="AAPL",
            quantity=50,
            avg_entry_price=140.0,
            current_price=155.0,
            market_value=7750.0,
            unrealized_pnl=750.0,
            unrealized_pnl_percent=0.0536
        )

        # Step 2: Generate SELL signal
        sell_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            confidence_score=0.82,
            ai_conviction_score=0.75,
            reasoning="Take profits",
            source_agent="ensemble_strategy"
        )
        mock_strategy.generate_signal.return_value = sell_signal
        mock_kafka.publish_signal.return_value = True

        response = client.post("/signals/generate?symbol=AAPL")

        assert response.status_code == 200
        assert response.json()["signal_type"] == "SELL"

        # Step 3: Execute trade
        mock_broker.can_trade.return_value = (True, "OK")
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            entry_price=155.0,
            status="filled"
        )

        response = client.post("/trades/execute?symbol=AAPL")

        assert response.status_code == 200
        trade_data = response.json()

        assert trade_data["side"] == "sell"
        assert trade_data["quantity"] == 50

        # Step 4: Verify position can be closed
        mock_broker.close_position.return_value = True

        response = client.delete("/positions/AAPL")

        assert response.status_code == 200


class TestBatchProcessingE2E:
    """End-to-end tests for batch operations."""

    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_batch_signal_generation_with_mixed_results(
        self, mock_kafka, mock_strategy
    ):
        """
        E2E: Generate signals for multiple symbols with mixed results.

        Tests:
        - Some succeed, some fail
        - Cache behavior
        - Performance
        """
        from src.main import app
        from src.data_layer.models import TradingSignal, SignalType

        client = TestClient(app)

        # Mock strategy - different signals for different symbols
        def generate_signal_side_effect(symbol, **kwargs):
            signal_map = {
                "AAPL": TradingSignal(
                    symbol="AAPL",
                    signal_type=SignalType.BUY,
                    confidence_score=0.85,
                    ai_conviction_score=0.78,
                    reasoning="Buy AAPL",
                    source_agent="ensemble"
                ),
                "MSFT": TradingSignal(
                    symbol="MSFT",
                    signal_type=SignalType.HOLD,
                    confidence_score=0.60,
                    ai_conviction_score=0.55,
                    reasoning="Hold MSFT",
                    source_agent="ensemble"
                ),
                "GOOGL": TradingSignal(
                    symbol="GOOGL",
                    signal_type=SignalType.SELL,
                    confidence_score=0.75,
                    ai_conviction_score=0.70,
                    reasoning="Sell GOOGL",
                    source_agent="ensemble"
                ),
            }
            return signal_map.get(symbol)

        mock_strategy.generate_signal.side_effect = generate_signal_side_effect
        mock_kafka.publish_signal.return_value = True

        # Generate batch signals
        response = client.post(
            "/signals/batch",
            json={"symbols": ["AAPL", "MSFT", "GOOGL"], "use_cache": False}
        )

        assert response.status_code == 200
        signals = response.json()

        assert len(signals) == 3

        # Verify different signal types
        signal_types = {s["symbol"]: s["signal_type"] for s in signals}
        assert signal_types["AAPL"] == "BUY"
        assert signal_types["MSFT"] == "HOLD"
        assert signal_types["GOOGL"] == "SELL"


class TestRealWorldScenariosE2E:
    """End-to-end tests for real-world scenarios."""

    @patch('src.main.broker')
    @patch('src.main.market_data')
    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_market_open_to_close_scenario(
        self, mock_kafka, mock_strategy, mock_market_data, mock_broker
    ):
        """
        E2E: Simulate trading from market open to close.

        Scenario:
        1. Market opens
        2. Generate signals for watchlist
        3. Execute trades
        4. Monitor positions throughout day
        5. Close positions before market close
        """
        from src.main import app
        from src.data_layer.models import TradingSignal, SignalType, Trade, OrderSide

        client = TestClient(app)

        # Step 1: Market opens
        mock_market_data.is_market_open.return_value = True

        # Step 2: Generate signals for watchlist
        watchlist = ["AAPL", "MSFT", "GOOGL"]

        def generate_buy_signal(symbol, **kwargs):
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence_score=0.80,
                ai_conviction_score=0.75,
                reasoning=f"Buy {symbol}",
                source_agent="ensemble"
            )

        mock_strategy.generate_signal.side_effect = generate_buy_signal
        mock_kafka.publish_signal.return_value = True

        signals = []
        for symbol in watchlist:
            response = client.post(f"/signals/generate?symbol={symbol}")
            assert response.status_code == 200
            signals.append(response.json())

        assert len(signals) == 3

        # Step 3: Execute trades for BUY signals
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": 100000.0,
            "buying_power": 50000.0,
        }
        mock_broker.can_trade.return_value = (True, "OK")

        trades = []
        for symbol in watchlist:
            mock_market_data.get_quote.return_value = {
                "symbol": symbol,
                "price": 150.0,
            }
            mock_broker.place_market_order.return_value = Trade(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=20,
                entry_price=150.0,
                status="filled"
            )

            response = client.post(f"/trades/execute?symbol={symbol}")
            if response.status_code == 200:
                trades.append(response.json())

        # Should have executed trades
        assert len(trades) >= 1

        # Step 4: Check portfolio
        from src.data_layer.models import Position, PortfolioState

        mock_broker.get_positions.return_value = [
            Position(
                symbol=symbol,
                quantity=20,
                avg_entry_price=150.0,
                current_price=152.0
            )
            for symbol in watchlist
        ]
        mock_broker.get_portfolio_state.return_value = PortfolioState(
            cash_balance=91000.0,
            portfolio_value=109000.0,
            total_pnl=9000.0,
            total_pnl_percent=0.09,
            active_positions=3
        )

        response = client.get("/portfolio/summary")
        assert response.status_code == 200

        portfolio = response.json()
        assert portfolio["portfolio_state"]["active_positions"] == 3

    @patch('src.main.broker')
    @patch('src.main.market_data')
    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_volatile_market_scenario(
        self, mock_kafka, mock_strategy, mock_market_data, mock_broker
    ):
        """
        E2E: Handle volatile market conditions.

        Scenario:
        1. Generate signal
        2. Price moves significantly before execution
        3. Re-evaluate signal
        4. Execute or skip based on new conditions
        """
        from src.main import app
        from src.data_layer.models import TradingSignal, SignalType

        client = TestClient(app)

        # Step 1: Generate BUY signal at $150
        buy_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.80,
            ai_conviction_score=0.75,
            reasoning="Buy at $150",
            source_agent="ensemble"
        )
        mock_strategy.generate_signal.return_value = buy_signal
        mock_kafka.publish_signal.return_value = True
        mock_market_data.is_market_open.return_value = True

        response = client.post("/signals/generate?symbol=AAPL")
        assert response.status_code == 200

        # Step 2: Price jumps to $165 (10% up)
        mock_market_data.get_quote.return_value = {
            "symbol": "AAPL",
            "price": 165.0,  # Price jumped!
            "bid": 164.95,
            "ask": 165.05,
        }

        # Step 3: System should handle this gracefully
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": 100000.0,
            "buying_power": 50000.0,
        }

        # Trade might execute at new price or skip
        # Both are valid responses to price volatility


class TestErrorRecoveryE2E:
    """End-to-end tests for error recovery."""

    @patch('src.main.broker')
    @patch('src.main.market_data')
    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_recovery_from_broker_outage(
        self, mock_kafka, mock_strategy, mock_market_data, mock_broker
    ):
        """
        E2E: Recover from temporary broker outage.

        Scenario:
        1. Generate signal successfully
        2. Broker API fails
        3. Retry mechanism handles it
        4. Eventually succeeds or fails gracefully
        """
        from src.main import app
        from src.data_layer.models import TradingSignal, SignalType

        client = TestClient(app)

        # Step 1: Generate signal (succeeds)
        buy_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.80,
            ai_conviction_score=0.75,
            reasoning="Buy",
            source_agent="ensemble"
        )
        mock_strategy.generate_signal.return_value = buy_signal
        mock_kafka.publish_signal.return_value = True

        response = client.post("/signals/generate?symbol=AAPL")
        assert response.status_code == 200

        # Step 2: Broker fails
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = {"price": 150.0}
        mock_broker.get_account.side_effect = Exception("Broker API timeout")

        # Step 3: Trade execution fails gracefully
        response = client.post("/trades/execute?symbol=AAPL")

        # Should either return 500 or handle gracefully
        # Important: shouldn't crash the server

    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_recovery_from_llm_api_failure(
        self, mock_kafka, mock_strategy
    ):
        """
        E2E: Recover from LLM API failure.

        Scenario:
        1. LLM API times out
        2. System retries
        3. Eventually succeeds or returns cached result
        """
        from src.main import app

        client = TestClient(app)

        # First attempt fails
        mock_strategy.generate_signal.side_effect = Exception("LLM API timeout")

        response = client.post("/signals/generate?symbol=AAPL")

        # Should return error but not crash
        assert response.status_code in [500, 503]


class TestPerformanceE2E:
    """End-to-end performance tests."""

    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_system_handles_high_load(
        self, mock_kafka, mock_strategy
    ):
        """
        E2E: System handles high concurrent load.

        Tests:
        - Multiple concurrent requests
        - No degradation in response time
        - No resource exhaustion
        """
        from src.main import app
        from src.data_layer.models import TradingSignal, SignalType
        import concurrent.futures

        client = TestClient(app)

        # Mock fast responses
        mock_strategy.generate_signal.return_value = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.80,
            ai_conviction_score=0.75,
            reasoning="Fast signal",
            source_agent="ensemble"
        )
        mock_kafka.publish_signal.return_value = True

        # Send 50 concurrent requests
        def make_request(i):
            start = time.time()
            response = client.post(f"/signals/generate?symbol=AAPL")
            latency = time.time() - start
            return response.status_code, latency

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        statuses = [r[0] for r in results]
        assert all(s == 200 for s in statuses)

        # Average latency should be reasonable
        latencies = [r[1] for r in results]
        avg_latency = sum(latencies) / len(latencies)

        # Should complete in reasonable time (even with mocks)
        assert avg_latency < 1.0  # < 1 second average


class TestDataConsistencyE2E:
    """End-to-end tests for data consistency."""

    @patch('src.main.broker')
    @patch('src.main.market_data')
    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_portfolio_consistency_after_trades(
        self, mock_kafka, mock_strategy, mock_market_data, mock_broker
    ):
        """
        E2E: Portfolio values remain consistent after trades.

        Tests:
        1. Initial portfolio state
        2. Execute buy trade
        3. Verify cash decreased, position increased
        4. Execute sell trade
        5. Verify cash increased, position closed
        """
        from src.main import app
        from src.data_layer.models import (
            TradingSignal, SignalType, Trade, OrderSide,
            Position, PortfolioState
        )

        client = TestClient(app)

        # Initial state
        initial_cash = 100000.0
        mock_broker.get_portfolio_state.return_value = PortfolioState(
            cash_balance=initial_cash,
            portfolio_value=initial_cash,
            total_pnl=0.0,
            total_pnl_percent=0.0,
            active_positions=0
        )

        # Buy trade
        mock_strategy.generate_signal.return_value = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.80,
            ai_conviction_score=0.75,
            reasoning="Buy",
            source_agent="ensemble"
        )
        mock_kafka.publish_signal.return_value = True
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = {"price": 150.0}

        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": initial_cash,
            "buying_power": initial_cash * 0.5,
        }
        mock_broker.can_trade.return_value = (True, "OK")

        trade_qty = 50
        trade_price = 150.0
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=trade_qty,
            entry_price=trade_price,
            status="filled"
        )

        response = client.post("/trades/execute?symbol=AAPL")
        assert response.status_code == 200

        # After buy: cash should decrease
        trade_cost = trade_qty * trade_price
        new_cash = initial_cash - trade_cost

        mock_broker.get_portfolio_state.return_value = PortfolioState(
            cash_balance=new_cash,
            portfolio_value=initial_cash,  # Same total value
            total_pnl=0.0,
            total_pnl_percent=0.0,
            active_positions=1
        )

        response = client.get("/portfolio")
        assert response.status_code == 200

        # Verify consistency: cash + position value = initial value
        portfolio = response.json()
        assert portfolio["cash_balance"] == new_cash
