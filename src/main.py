"""
Main FastAPI application for the LLM Trading Platform.

This provides a REST API for:
- Generating trading signals
- Executing trades
- Monitoring portfolio
- Viewing analytics

Endpoints are documented via OpenAPI/Swagger UI at /docs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
from typing import List, Optional
from datetime import datetime
import uvicorn

from src.llm_agents import EnsembleStrategy
from src.trading_engine import executor, broker
from src.data_layer.models import TradingSignal, Trade, Position, PortfolioState
from src.utils.config import settings
from src.utils.logger import app_logger
from src.data_layer.kafka_stream import KafkaStreamProducer

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Augmented Trading Platform",
    description="Production-grade algorithmic trading platform using LLMs for fundamental analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
signal_counter = Counter(
    "trading_signals_total",
    "Total number of trading signals generated",
    ["symbol", "signal_type"]
)
trade_counter = Counter(
    "trades_executed_total",
    "Total number of trades executed",
    ["symbol", "side"]
)
api_latency = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    ["endpoint"]
)

# Initialize components
strategy = EnsembleStrategy()
kafka_producer = KafkaStreamProducer()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "LLM Trading Platform",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check.

    Returns:
        System health status
    """
    try:
        # Check broker connection
        account = broker.get_account()
        broker_status = "healthy" if account else "degraded"
    except Exception as e:
        broker_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "broker": broker_status,
        "paper_trading": settings.paper_trading,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics
    """
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/signals/generate", response_model=TradingSignal)
async def generate_signal(
    symbol: str,
    use_cache: bool = True,
    execute: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Generate a trading signal for a symbol.

    Args:
        symbol: Stock ticker (e.g., AAPL, MSFT)
        use_cache: Use cached analysis if available
        execute: Automatically execute the signal

    Returns:
        TradingSignal object with analysis and recommendation
    """
    with api_latency.labels(endpoint="/signals/generate").time():
        try:
            app_logger.info(f"Generating signal for {symbol}")

            # Generate signal using ensemble strategy
            signal = strategy.generate_signal(
                symbol=symbol.upper(),
                use_cache=use_cache,
                track_mlflow=True
            )

            # Update metrics
            signal_counter.labels(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value
            ).inc()

            # Publish to Kafka
            kafka_producer.publish_signal(signal.dict())

            # Execute if requested
            if execute and signal.signal_type != "HOLD":
                if background_tasks:
                    background_tasks.add_task(executor.execute_signal, signal)
                else:
                    executor.execute_signal(signal)

            app_logger.info(
                f"Signal generated for {symbol}: {signal.signal_type.value} "
                f"(conviction={signal.ai_conviction_score:.2f})"
            )

            return signal

        except Exception as e:
            app_logger.error(f"Error generating signal for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/batch", response_model=List[TradingSignal])
async def generate_batch_signals(
    symbols: List[str],
    use_cache: bool = True
):
    """
    Generate signals for multiple symbols.

    Args:
        symbols: List of stock tickers
        use_cache: Use cached analysis if available

    Returns:
        List of TradingSignal objects
    """
    signals = []
    errors = []

    for symbol in symbols:
        try:
            signal = strategy.generate_signal(
                symbol=symbol.upper(),
                use_cache=use_cache
            )
            signals.append(signal)

            # Publish to Kafka
            kafka_producer.publish_signal(signal.dict())

        except Exception as e:
            app_logger.error(f"Error generating signal for {symbol}: {e}")
            errors.append({"symbol": symbol, "error": str(e)})

    if errors:
        app_logger.warning(f"Batch signal generation had {len(errors)} errors")

    return signals


@app.post("/trades/execute")
async def execute_trade(
    symbol: str,
    generate_signal: bool = True
):
    """
    Execute a trade for a symbol.

    Args:
        symbol: Stock ticker
        generate_signal: Generate a new signal before trading

    Returns:
        Trade object if executed
    """
    try:
        # Generate signal if requested
        if generate_signal:
            signal = strategy.generate_signal(symbol.upper())
        else:
            # Create a manual signal (not recommended)
            signal = TradingSignal(
                symbol=symbol.upper(),
                signal_type="BUY",
                confidence_score=0.5,
                ai_conviction_score=0.5,
                reasoning="Manual trade execution",
                source_agent="manual"
            )

        # Execute the signal
        trade = executor.execute_signal(signal)

        if trade:
            trade_counter.labels(
                symbol=trade.symbol,
                side=trade.side
            ).inc()
            return trade
        else:
            raise HTTPException(
                status_code=400,
                detail="Trade not executed (check logs for details)"
            )

    except Exception as e:
        app_logger.error(f"Error executing trade for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio", response_model=PortfolioState)
async def get_portfolio():
    """
    Get current portfolio state.

    Returns:
        PortfolioState with performance metrics
    """
    try:
        return broker.get_portfolio_state()
    except Exception as e:
        app_logger.error(f"Error getting portfolio state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/summary")
async def get_portfolio_summary():
    """
    Get detailed portfolio summary.

    Returns:
        Comprehensive portfolio information
    """
    try:
        return executor.get_portfolio_summary()
    except Exception as e:
        app_logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions", response_model=List[Position])
async def get_positions():
    """
    Get all open positions.

    Returns:
        List of Position objects
    """
    try:
        return broker.get_positions()
    except Exception as e:
        app_logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions/{symbol}", response_model=Position)
async def get_position(symbol: str):
    """
    Get position for a specific symbol.

    Args:
        symbol: Stock ticker

    Returns:
        Position object
    """
    try:
        position = broker.get_position(symbol.upper())
        if not position:
            raise HTTPException(
                status_code=404,
                detail=f"No position found for {symbol}"
            )
        return position
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/positions/{symbol}")
async def close_position(symbol: str):
    """
    Close a position.

    Args:
        symbol: Stock ticker

    Returns:
        Success message
    """
    try:
        success = broker.close_position(symbol.upper())
        if success:
            return {"message": f"Position closed for {symbol}"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to close position for {symbol}"
            )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error closing position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/account")
async def get_account():
    """
    Get account information.

    Returns:
        Account details
    """
    try:
        return broker.get_account()
    except Exception as e:
        app_logger.error(f"Error getting account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    app_logger.info("=" * 60)
    app_logger.info("🚀 LLM Trading Platform Starting Up")
    app_logger.info("=" * 60)
    app_logger.info(f"Paper Trading: {settings.paper_trading}")
    app_logger.info(f"Max Position Size: ${settings.max_position_size:,.2f}")
    app_logger.info(f"Risk Per Trade: {settings.risk_per_trade*100}%")
    app_logger.info(f"MLflow Tracking: {settings.mlflow_tracking_uri}")
    app_logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    app_logger.info("Shutting down LLM Trading Platform")
    if kafka_producer:
        kafka_producer.close()


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
