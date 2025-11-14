# Operations Runbook

## Advanced Algo Trading Platform

**Version:** 1.0
**Last Updated:** 2025-11-14
**Maintained by:** Platform Team

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Access and Credentials](#access-and-credentials)
4. [Daily Operations](#daily-operations)
5. [Monitoring](#monitoring)
6. [Common Operations](#common-operations)
7. [Troubleshooting](#troubleshooting)
8. [Incident Response](#incident-response)
9. [Emergency Procedures](#emergency-procedures)
10. [Maintenance](#maintenance)

---

## System Overview

### Purpose
The Advanced Algo Trading Platform executes automated trading strategies across multiple markets with real-time risk management and performance monitoring.

### Key Components
- **Trading Engine**: Core strategy execution
- **Risk Manager**: Real-time risk monitoring and circuit breakers
- **Portfolio Optimizer**: Multi-strategy capital allocation
- **Model Monitor**: ML model drift detection and retraining
- **Data Pipeline**: Market data ingestion and processing
- **API Gateway**: External integrations and webhooks

### Service Dependencies
- PostgreSQL (primary database)
- Redis (caching and pub/sub)
- AWS S3 (data storage)
- Broker API (trade execution)
- Market Data Feed (real-time prices)

### SLA Targets
- Uptime: 99.9% during market hours
- Order execution latency: <100ms (p95)
- Data feed latency: <50ms
- Alert response time: <5 minutes

---

## Architecture

### Production Environment

```
Internet
    |
    v
Load Balancer (ALB)
    |
    v
Application Servers (ECS/K8s)
    |
    +-- Trading Engine
    +-- Risk Manager
    +-- Portfolio Optimizer
    +-- API Gateway
    |
    v
Database Layer
    |
    +-- PostgreSQL (RDS Multi-AZ)
    +-- Redis (ElastiCache)
    |
    v
External Services
    |
    +-- Broker API
    +-- Market Data Feed
    +-- Cloud Storage (S3)
```

### Key URLs
- **Production**: https://trading.example.com
- **API**: https://api.trading.example.com
- **Monitoring**: https://monitoring.trading.example.com
- **Logs**: https://logs.trading.example.com

---

## Access and Credentials

### Access Control

**Production Access:**
- Restricted to on-call engineers
- Requires MFA
- All actions logged

**Getting Access:**
1. Request access via ticket system
2. Manager approval required
3. Security team provisions access
4. Expires after 30 days (renewable)

### Credentials Locations

| System | Location | Access Method |
|--------|----------|---------------|
| AWS Console | SSO | https://mycompany.awsapps.com/start |
| Database | AWS Secrets Manager | `trading/prod/db_credentials` |
| Broker API | AWS Secrets Manager | `trading/prod/broker_api_key` |
| Redis | AWS Secrets Manager | `trading/prod/redis_password` |

### Emergency Contacts

| Role | Primary | Secondary |
|------|---------|-----------|
| On-call Engineer | PagerDuty | PagerDuty |
| Tech Lead | john@example.com | +1-555-0100 |
| Risk Manager | jane@example.com | +1-555-0101 |
| Platform Lead | bob@example.com | +1-555-0102 |

---

## Daily Operations

### Pre-Market Checklist (30 minutes before market open)

```bash
# 1. Check system health
./scripts/health_check.sh

# 2. Verify data feeds
./scripts/check_data_feeds.sh

# 3. Validate broker connection
./scripts/test_broker_connection.sh

# 4. Check overnight reconciliation
./scripts/reconciliation_check.sh

# 5. Review risk limits
./scripts/check_risk_limits.sh

# 6. Check model versions
./scripts/check_model_versions.sh
```

### Market Open Checklist

```bash
# 1. Monitor first trades
tail -f /var/log/trading/trades.log

# 2. Watch error rates
./scripts/monitor_errors.sh

# 3. Check execution latency
./scripts/monitor_latency.sh

# 4. Verify positions
./scripts/check_positions.sh
```

### Intraday Monitoring

- Check dashboards every hour
- Review alerts immediately
- Monitor PnL in real-time
- Validate data feed health
- Check for stuck orders

### End-of-Day Checklist

```bash
# 1. Position reconciliation
./scripts/eod_reconciliation.sh

# 2. PnL calculation
./scripts/calculate_pnl.sh

# 3. Performance metrics
./scripts/calculate_metrics.sh

# 4. Backup critical data
./scripts/backup_data.sh

# 5. Generate daily report
./scripts/generate_daily_report.sh

# 6. Check for model drift
./scripts/check_model_drift.sh
```

---

## Monitoring

### Key Dashboards

**System Health Dashboard**
- URL: https://monitoring.example.com/system
- Metrics: CPU, Memory, Disk, Network
- Refresh: 1 minute

**Trading Dashboard**
- URL: https://monitoring.example.com/trading
- Metrics: PnL, Win Rate, Sharpe Ratio, Drawdown
- Refresh: 30 seconds

**Risk Dashboard**
- URL: https://monitoring.example.com/risk
- Metrics: Position sizes, Leverage, VaR, Exposure
- Refresh: 10 seconds

**Model Performance Dashboard**
- URL: https://monitoring.example.com/models
- Metrics: Prediction accuracy, Drift scores, Latency
- Refresh: 5 minutes

### Critical Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| Trading halted | Immediate | Investigate circuit breaker trigger |
| Data feed down | >30 seconds | Switch to backup feed |
| High error rate | >1% | Check logs, may need to halt |
| Large loss | >5% daily | Review positions, consider closing |
| Database latency | >500ms | Check DB performance, scale if needed |
| API errors | >5% | Check broker API status |
| Model drift | Score >0.2 | Review model, may need retraining |

### Log Locations

```bash
# Application logs
/var/log/trading/app.log

# Trading activity
/var/log/trading/trades.log

# Errors
/var/log/trading/errors.log

# Risk events
/var/log/trading/risk.log

# Model events
/var/log/trading/models.log

# CloudWatch (AWS)
aws logs tail /ecs/trading-app --follow
```

---

## Common Operations

### Deploy New Version

```bash
# 1. Verify new version in staging
./scripts/verify_staging.sh

# 2. Schedule deployment (outside market hours preferred)
# 3. Notify team

# 4. Deploy with zero-downtime
./scripts/deploy_production.sh v1.2.3

# 5. Monitor deployment
kubectl rollout status deployment/trading-app

# 6. Verify health
./scripts/health_check.sh

# 7. Monitor for errors
./scripts/monitor_errors.sh --duration 30m
```

### Scale Resources

```bash
# Scale ECS service
aws ecs update-service \
  --cluster trading-cluster \
  --service trading-service \
  --desired-count 5

# Scale Kubernetes deployment
kubectl scale deployment trading-app --replicas=5

# Verify scaling
kubectl get pods -l app=trading
```

### Restart Service

```bash
# Graceful restart (preferred)
kubectl rollout restart deployment/trading-app

# Force restart (if needed)
kubectl delete pod -l app=trading

# Verify pods are healthy
kubectl get pods
kubectl logs -l app=trading --tail=100
```

### Update Configuration

```bash
# 1. Update ConfigMap
kubectl edit configmap trading-config

# 2. Restart to pick up changes
kubectl rollout restart deployment/trading-app

# 3. Verify configuration
kubectl exec -it POD_NAME -- env | grep CONFIG
```

### Database Operations

```bash
# Connect to database
psql -h RDS_ENDPOINT -U admin -d trading_db

# Run migration
./scripts/db_migrate.sh

# Create backup
./scripts/backup_database.sh

# Restore from backup
./scripts/restore_database.sh BACKUP_ID
```

### Model Deployment

```bash
# Deploy new model version
./scripts/deploy_model.sh momentum_v2.pkl

# Rollback model
./scripts/rollback_model.sh momentum_v1.pkl

# Verify model version
./scripts/check_model_version.sh momentum
```

---

## Troubleshooting

### High Error Rate

**Symptoms:** Error rate >1%, alerts firing

**Diagnosis:**
```bash
# Check error logs
tail -f /var/log/trading/errors.log

# Check error types
./scripts/analyze_errors.sh --last 1h

# Check external services
./scripts/check_dependencies.sh
```

**Common Causes:**
1. Broker API issues
2. Data feed problems
3. Database connection issues
4. Network problems

**Resolution:**
1. Identify error source
2. If broker API: check status page, switch to backup if needed
3. If data feed: switch to backup feed
4. If database: check connections, scale if needed
5. If persistent: halt trading and investigate

### Orders Not Executing

**Symptoms:** Orders stuck in "pending" state

**Diagnosis:**
```bash
# Check pending orders
./scripts/check_pending_orders.sh

# Check broker API status
./scripts/test_broker_api.sh

# Check order logs
grep "order_id=12345" /var/log/trading/trades.log
```

**Resolution:**
```bash
# 1. Verify broker connection
./scripts/test_broker_connection.sh

# 2. Cancel stuck orders if necessary
./scripts/cancel_order.sh ORDER_ID

# 3. Retry failed orders
./scripts/retry_orders.sh

# 4. If broker is down: halt trading
./scripts/halt_trading.sh --reason "broker_down"
```

### Data Feed Interruption

**Symptoms:** No price updates, stale data

**Diagnosis:**
```bash
# Check data feed status
./scripts/check_data_feed_health.sh

# Check last update time
./scripts/check_last_update.sh

# Test connectivity
curl -v https://datafeed.provider.com/health
```

**Resolution:**
```bash
# 1. Switch to backup feed
./scripts/switch_data_feed.sh --feed backup

# 2. Verify backup is working
./scripts/verify_data_feed.sh

# 3. If both feeds down: halt trading
./scripts/halt_trading.sh --reason "data_feed_down"
```

### High Database Latency

**Symptoms:** Slow queries, timeouts

**Diagnosis:**
```bash
# Check database metrics
aws cloudwatch get-metric-statistics --metric-name DatabaseConnections

# Check slow queries
./scripts/check_slow_queries.sh

# Check connection pool
./scripts/check_db_connections.sh
```

**Resolution:**
```bash
# 1. Kill long-running queries if needed
./scripts/kill_slow_queries.sh

# 2. Increase connection pool
./scripts/scale_connection_pool.sh --size 50

# 3. Scale database instance if needed
./scripts/scale_rds.sh --instance-class db.r5.xlarge

# 4. Check and optimize indexes
./scripts/analyze_indexes.sh
```

### Circuit Breaker Triggered

**Symptoms:** Trading halted, alert fired

**Diagnosis:**
```bash
# Check circuit breaker reason
./scripts/check_circuit_breakers.sh

# Review recent trades
./scripts/review_recent_trades.sh --last 1h

# Check PnL
./scripts/calculate_pnl.sh --real-time
```

**Resolution:**
1. Review why circuit breaker triggered
2. Verify it's working correctly (not false alarm)
3. Fix underlying issue if any
4. Get approval from risk manager
5. Reset circuit breaker
```bash
./scripts/reset_circuit_breaker.sh --confirm
```

### Model Drift Detected

**Symptoms:** Model drift alert, degraded performance

**Diagnosis:**
```bash
# Check drift scores
./scripts/check_drift_scores.sh

# Review model performance
./scripts/check_model_performance.sh

# Compare to baseline
./scripts/compare_to_baseline.sh
```

**Resolution:**
```bash
# 1. Retrain model
./scripts/retrain_model.sh --strategy momentum

# 2. Validate new model
./scripts/validate_model.sh --model momentum_v2

# 3. Deploy if validation passes
./scripts/deploy_model.sh momentum_v2

# 4. Monitor new model performance
./scripts/monitor_model.sh --duration 1d
```

---

## Incident Response

### Severity Levels

**P0 - Critical:** System down, trading halted, major financial loss
- Response time: Immediate
- All hands on deck
- Executive notification required

**P1 - High:** Degraded performance, some strategies down
- Response time: <15 minutes
- On-call engineer + backup
- Team lead notification

**P2 - Medium:** Non-critical issues, workarounds available
- Response time: <1 hour
- On-call engineer
- Can wait for business hours if after market close

**P3 - Low:** Minor issues, no immediate impact
- Response time: Next business day
- Create ticket for follow-up

### Incident Response Process

**1. Detection**
- Alert fires or manual detection
- Acknowledge alert in PagerDuty
- Create incident ticket

**2. Assessment**
- Determine severity
- Identify scope of impact
- Estimate time to resolution

**3. Response**
- Form incident response team
- Start incident call/chat
- Begin investigation

**4. Communication**
- Notify stakeholders
- Update status page
- Regular updates every 30 minutes

**5. Mitigation**
- Implement fix or workaround
- Verify mitigation works
- Monitor for recurrence

**6. Recovery**
- Restore full functionality
- Verify system health
- Close incident

**7. Post-Mortem**
- Schedule within 48 hours
- Document timeline
- Identify root cause
- Create action items

---

## Emergency Procedures

### Emergency Shutdown

**When to use:** Critical system failure, runaway trading, severe security breach

```bash
# 1. Halt all trading immediately
./scripts/emergency_halt.sh

# 2. Cancel all pending orders
./scripts/cancel_all_orders.sh

# 3. Close all positions (if necessary)
./scripts/close_all_positions.sh --confirm

# 4. Disable auto-restart
./scripts/disable_auto_restart.sh

# 5. Notify team
./scripts/send_emergency_alert.sh

# 6. Document reason
./scripts/log_emergency_shutdown.sh --reason "REASON"
```

### Rollback Deployment

**When to use:** New deployment causing issues

```bash
# 1. Rollback to previous version
kubectl rollout undo deployment/trading-app

# 2. Verify rollback
kubectl rollout status deployment/trading-app

# 3. Check health
./scripts/health_check.sh

# 4. Monitor for issues
./scripts/monitor_errors.sh --duration 30m
```

### Failover to DR

**When to use:** Primary region failure

```bash
# 1. Verify DR region is ready
./scripts/check_dr_readiness.sh

# 2. Update DNS to point to DR
./scripts/failover_dns.sh --region us-west-2

# 3. Verify traffic routing
./scripts/verify_failover.sh

# 4. Monitor DR region
./scripts/monitor_dr.sh
```

---

## Maintenance

### Weekly Maintenance

```bash
# 1. Review logs for errors
./scripts/analyze_logs.sh --last 1w

# 2. Check disk usage
./scripts/check_disk_usage.sh

# 3. Review performance metrics
./scripts/weekly_metrics.sh

# 4. Update documentation
# (Manual task)
```

### Monthly Maintenance

```bash
# 1. Database maintenance
./scripts/db_maintenance.sh

# 2. Review and rotate credentials
./scripts/rotate_credentials.sh

# 3. Security patches
./scripts/apply_security_patches.sh

# 4. Capacity planning review
./scripts/capacity_report.sh
```

### Quarterly Maintenance

- Disaster recovery drill
- Update runbook
- Review and update alerts
- Security audit
- Compliance review

---

## Appendix

### Quick Reference

```bash
# Get pod logs
kubectl logs -l app=trading --tail=100

# Check system health
./scripts/health_check.sh

# Check PnL
./scripts/check_pnl.sh

# Halt trading
./scripts/halt_trading.sh

# Resume trading
./scripts/resume_trading.sh
```

### Useful Commands

```bash
# AWS ECS
aws ecs describe-services --cluster trading-cluster --services trading-service
aws ecs list-tasks --cluster trading-cluster --service trading-service

# Kubernetes
kubectl get pods -l app=trading
kubectl logs POD_NAME
kubectl exec -it POD_NAME -- /bin/bash
kubectl describe pod POD_NAME

# Database
psql -h RDS_ENDPOINT -U admin -d trading_db
SELECT * FROM positions WHERE timestamp > NOW() - INTERVAL '1 hour';

# Logs
tail -f /var/log/trading/app.log
grep ERROR /var/log/trading/app.log | tail -100
```

---

**Document Control:**
- Review this runbook monthly
- Update after each incident
- Version control in Git
- Training required for all on-call engineers
