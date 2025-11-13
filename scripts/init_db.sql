-- Initialize trading database schema
-- This script runs automatically when PostgreSQL container starts

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for storing trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence_score DECIMAL(5,4) NOT NULL,
    ai_conviction_score DECIMAL(5,4),
    fundamental_score DECIMAL(5,4),
    sentiment_score DECIMAL(5,4),
    technical_score DECIMAL(5,4),
    reasoning TEXT,
    source_agent VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_created (symbol, created_at),
    INDEX idx_signal_type (signal_type)
);

-- Table for storing executed trades
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES trading_signals(id),
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(12,4),
    exit_price DECIMAL(12,4),
    pnl DECIMAL(12,4),
    pnl_percent DECIMAL(8,4),
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'filled', 'partial', 'cancelled'
    order_id VARCHAR(100),
    filled_at TIMESTAMP,
    closed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_status (symbol, status),
    INDEX idx_created_at (created_at)
);

-- Table for storing portfolio state
CREATE TABLE IF NOT EXISTS portfolio_state (
    id SERIAL PRIMARY KEY,
    cash_balance DECIMAL(15,4) NOT NULL,
    portfolio_value DECIMAL(15,4) NOT NULL,
    total_pnl DECIMAL(15,4),
    total_pnl_percent DECIMAL(8,4),
    active_positions INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing position information
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    quantity INTEGER NOT NULL,
    avg_entry_price DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4),
    market_value DECIMAL(15,4),
    unrealized_pnl DECIMAL(12,4),
    unrealized_pnl_percent DECIMAL(8,4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol (symbol)
);

-- Table for storing market data cache
CREATE TABLE IF NOT EXISTS market_data_cache (
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL, -- 'quote', 'news', 'filing', 'earnings'
    data JSONB NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, data_type),
    INDEX idx_expires (expires_at)
);

-- Table for MLflow experiment metadata
CREATE TABLE IF NOT EXISTS ml_experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(200) NOT NULL,
    strategy_version VARCHAR(50),
    parameters JSONB,
    metrics JSONB,
    tags JSONB,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP
);

-- Table for storing LLM agent analysis results
CREATE TABLE IF NOT EXISTS llm_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    agent_type VARCHAR(50) NOT NULL, -- 'financial', 'sentiment', 'earnings'
    analysis_result JSONB NOT NULL,
    tokens_used INTEGER,
    api_cost DECIMAL(10,6),
    processing_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_agent (symbol, agent_type, created_at)
);

-- Create views for analytics
CREATE OR REPLACE VIEW daily_performance AS
SELECT
    DATE(created_at) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(pnl) as daily_pnl,
    AVG(pnl_percent) as avg_return_pct,
    MAX(pnl) as best_trade,
    MIN(pnl) as worst_trade
FROM trades
WHERE status = 'filled' AND closed_at IS NOT NULL
GROUP BY DATE(created_at)
ORDER BY trade_date DESC;

CREATE OR REPLACE VIEW agent_performance AS
SELECT
    s.source_agent,
    COUNT(t.id) as total_trades,
    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(t.id), 0) as win_rate,
    AVG(t.pnl) as avg_pnl,
    AVG(s.confidence_score) as avg_confidence
FROM trading_signals s
LEFT JOIN trades t ON s.id = t.signal_id
WHERE t.status = 'filled'
GROUP BY s.source_agent;

-- Function to update portfolio state
CREATE OR REPLACE FUNCTION update_portfolio_state()
RETURNS TRIGGER AS $$
BEGIN
    -- This would be called by application logic to update portfolio
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;

-- Insert initial portfolio state
INSERT INTO portfolio_state (cash_balance, portfolio_value, total_pnl, total_pnl_percent)
VALUES (100000.00, 100000.00, 0.00, 0.00)
ON CONFLICT DO NOTHING;

COMMENT ON TABLE trading_signals IS 'Stores AI-generated trading signals with multi-factor scores';
COMMENT ON TABLE trades IS 'Records all executed trades with P&L tracking';
COMMENT ON TABLE llm_analyses IS 'Caches LLM analysis results to reduce API costs';
