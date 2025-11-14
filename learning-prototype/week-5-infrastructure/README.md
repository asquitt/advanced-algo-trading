# Week 5: Infrastructure & Monitoring

Deploy your trading system with production-grade infrastructure!

## Learning Objectives

By the end of this week, you will:

✅ Understand containerization with Docker
✅ Deploy 12-service stack with Docker Compose
✅ Configure Prometheus for metrics collection
✅ Build custom Grafana dashboards
✅ Set up ELK stack for log aggregation
✅ Monitor trading system health in real-time
✅ Configure alerts for critical events

## Why Infrastructure Matters

> "Hope is not a strategy. Monitor everything." - Production Engineering Wisdom

Infrastructure enables:
- **Reliability**: 99.9% uptime
- **Observability**: Know what's happening
- **Scalability**: Handle growth
- **Debugging**: Find issues quickly

## Prerequisites

- Docker basics
- Linux command line
- Networking fundamentals
- Completed Week 3-4

## Folder Structure

```
week-5-infrastructure/
├── README.md (you are here)
├── CONCEPTS.md (infrastructure concepts)
├── exercises/
│   ├── exercise_1_docker.md
│   ├── exercise_2_prometheus.md
│   ├── exercise_3_grafana.md
│   └── exercise_4_elk.md
├── configs/
│   ├── docker-compose-basic.yml
│   ├── prometheus-config.yml
│   ├── grafana-dashboard-example.json
│   └── logstash-pipeline.conf
└── solutions/
    └── docker-compose-complete.yml
```

## Learning Path

### Day 1: Docker & Containers (3-4 hours)

**What you'll learn**:
- Containerization concepts
- Writing Dockerfiles
- Docker Compose basics
- Multi-service orchestration

**Exercises**:
```bash
# Build trading API container
docker build -t trading-api .

# Run with dependencies
docker-compose up -d

# View logs
docker-compose logs -f trading-api
```

### Day 2: Prometheus Metrics (3-4 hours)

**What you'll learn**:
- Metrics collection
- Counter, Gauge, Histogram
- Prometheus query language (PromQL)
- Custom trading metrics

**Key metrics**:
- Portfolio value
- Trade count
- Order latency
- Slippage
- Sharpe ratio

### Day 3: Grafana Dashboards (3-4 hours)

**What you'll learn**:
- Dashboard design
- Visualization types
- Alerting rules
- Dashboard templates

**Dashboards**:
1. Trading Overview (P&L, positions)
2. Risk Management (drawdown, VaR)
3. Execution Quality (fill rate, slippage)
4. System Health (CPU, memory)

### Day 4: Logging (ELK Stack) (3-4 hours)

**What you'll learn**:
- Centralized logging
- Elasticsearch queries
- Kibana visualizations
- Log parsing with Logstash

**Use cases**:
- Trade audit trail
- Error tracking
- Performance debugging
- Compliance

### Day 5: Integration & Testing (2-3 hours)

**Complete system**:
```bash
# Start full stack
docker-compose up -d

# Access services
# API: http://localhost:8000/docs
# Grafana: http://localhost:3000
# Kibana: http://localhost:5601
# Prometheus: http://localhost:9090

# Run health checks
python scripts/health_check.py
```

## Key Concepts

### 1. Observability

**Three pillars**:
- **Metrics**: Numbers (latency, count, gauge)
- **Logs**: Events (trade executed, error occurred)
- **Traces**: Request flow (order → execution)

### 2. Service Architecture

```
┌─────────────┐
│  Trading    │
│   API       │
└──────┬──────┘
       │
   ┌───┴───┬────────┬────────┬─────────┐
   │       │        │        │         │
   ▼       ▼        ▼        ▼         ▼
┌──────┐ ┌────┐  ┌─────┐  ┌─────┐  ┌──────┐
│ DB   │ │Cache│ │Kafka│  │Prom.│  │ ELK  │
└──────┘ └────┘  └─────┘  └─────┘  └──────┘
```

### 3. Docker Compose

```yaml
version: '3.8'
services:
  trading-api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://postgres:5432/trading
```

### 4. Prometheus Metrics

```python
from prometheus_client import Counter, Gauge, Histogram

# Count trades
trade_counter = Counter('trades_total', 'Total trades', ['symbol', 'side'])
trade_counter.labels(symbol='AAPL', side='buy').inc()

# Track portfolio value
portfolio_gauge = Gauge('portfolio_value_dollars', 'Portfolio value')
portfolio_gauge.set(125000.50)

# Measure latency
latency_histogram = Histogram('order_latency_seconds', 'Order latency')
with latency_histogram.time():
    execute_order()
```

## Success Criteria

You've mastered Week 5 when you can:

✅ Deploy full 12-service stack locally
✅ Create custom Prometheus metrics
✅ Build Grafana dashboards
✅ Query logs in Kibana
✅ Set up alerts for critical events
✅ Monitor system health
✅ Debug issues using logs and metrics

## Performance Targets

- **API latency**: < 100ms p99
- **Database queries**: < 50ms p95
- **Cache hit rate**: > 90%
- **Uptime**: > 99.9%
- **Log ingestion**: < 5s delay

## Resources

### Documentation
- [Docker Docs](https://docs.docker.com/)
- [Prometheus](https://prometheus.io/docs/)
- [Grafana](https://grafana.com/docs/)
- [Elastic Stack](https://www.elastic.co/guide/)

### Courses
- Docker Mastery (Udemy)
- Prometheus & Grafana (Udemy)

### Books
- "The Art of Monitoring" by James Turnbull
- "Site Reliability Engineering" by Google

**Next**: Week 6 - Alternative Data & News Integration 📰
