# Exercise 3: Grafana Dashboard Creation

## Objective
Create custom Grafana dashboards to visualize trading metrics and system performance in real-time.

## Prerequisites
- Completed Exercise 1 (Docker) and Exercise 2 (Prometheus)
- Prometheus collecting metrics from trading API
- Services running via docker-compose

## Part 1: Access Grafana

1. **Open Grafana in your browser:**
   ```
   http://localhost:3000
   ```

2. **Login with default credentials:**
   - Username: `admin`
   - Password: `admin`
   - You'll be prompted to change the password

## Part 2: Configure Prometheus Data Source

1. **Navigate to Configuration > Data Sources**
2. **Click "Add data source"**
3. **Select "Prometheus"**
4. **Configure settings:**
   - Name: `Prometheus`
   - URL: `http://prometheus:9090`
   - Access: `Server (default)`
   - Scrape interval: `15s`
5. **Click "Save & Test"**
   - You should see "Data source is working"

## Part 3: Create Your First Dashboard

### Dashboard 1: Trading Overview

1. **Click "+" > Create Dashboard**
2. **Add Panel** and configure:

#### Panel 1: Total Orders (Stat)
- Query: `sum(increase(order_count_total[24h]))`
- Visualization: Stat
- Title: "Total Orders (24h)"
- Unit: `short`
- Color mode: Value
- Graph mode: Area

#### Panel 2: Order Rate (Graph)
- Query A: `sum by (symbol) (rate(order_count_total[5m]))`
- Legend: `{{symbol}}`
- Visualization: Time series
- Title: "Order Rate by Symbol"
- Y-axis unit: `ops` (operations per second)

#### Panel 3: Buy vs Sell Orders (Pie Chart)
- Query A: `sum(increase(order_count_total{side="BUY"}[1h]))`
- Query B: `sum(increase(order_count_total{side="SELL"}[1h]))`
- Visualization: Pie chart
- Title: "Buy vs Sell Orders (1h)"
- Legend values: Value

#### Panel 4: Current P&L (Stat)
- Query: `sum(trade_pnl_dollars)`
- Visualization: Stat
- Title: "Total P&L"
- Unit: `currencyUSD`
- Thresholds:
  - Red: `< 0`
  - Green: `>= 0`

3. **Save Dashboard:**
   - Click save icon
   - Name: "Trading Overview"
   - Add tags: `trading`, `overview`

### Dashboard 2: System Performance

1. **Create new dashboard**
2. **Add the following panels:**

#### Panel 1: API Request Rate
```promql
rate(http_requests_total{job="trading_api"}[5m])
```
- Visualization: Time series
- Legend: `{{method}} {{endpoint}}`

#### Panel 2: API Latency Percentiles
```promql
# Add three queries:
# p50
histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))
# p95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
# p99
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```
- Visualization: Time series
- Y-axis: seconds

#### Panel 3: HTTP Status Codes
```promql
sum by (status) (rate(http_requests_total[5m]))
```
- Visualization: Bar gauge
- Legend: `{{status}}`

#### Panel 4: Order Execution Latency (Heatmap)
```promql
sum(increase(order_latency_seconds_bucket[1m])) by (le)
```
- Visualization: Heatmap
- Color scheme: Spectral

3. **Save Dashboard** as "System Performance"

### Dashboard 3: Risk Monitoring

#### Panel 1: Position Values
```promql
position_value_dollars
```
- Visualization: Time series
- Legend: `{{symbol}}`
- Y-axis: currencyUSD

#### Panel 2: Open Positions
```promql
open_positions_count
```
- Visualization: Stat
- Color: Value

#### Panel 3: Market Data Lag
```promql
market_data_lag_seconds
```
- Visualization: Gauge
- Min: 0
- Max: 60
- Thresholds: Green (0-5), Yellow (5-15), Red (15+)

#### Panel 4: Cache Performance
```promql
rate(redis_cache_hits_total[5m]) /
(rate(redis_cache_hits_total[5m]) + rate(redis_cache_misses_total[5m]))
```
- Visualization: Time series
- Y-axis: Percent (0-1)
- Min: 0, Max: 1

## Part 4: Add Variables to Dashboard

Variables make dashboards interactive and reusable.

1. **Open Dashboard Settings > Variables**
2. **Add Variable:**
   - Name: `symbol`
   - Type: Query
   - Data source: Prometheus
   - Query: `label_values(order_count_total, symbol)`
   - Multi-value: Yes
   - Include All option: Yes

3. **Use variable in panels:**
   ```promql
   rate(order_count_total{symbol=~"$symbol"}[5m])
   ```

4. **Add time range variable:**
   - Name: `timerange`
   - Type: Interval
   - Values: `5m,15m,1h,6h,24h`

## Part 5: Configure Alerts in Grafana

1. **Edit a panel (e.g., API Latency)**
2. **Go to Alert tab**
3. **Create alert rule:**
   - Name: High API Latency
   - Evaluate every: 1m
   - For: 5m
   - Condition: WHEN avg() OF query(A, 5m, now) IS ABOVE 1
   - No data: Alerting
   - Error: Alerting

4. **Add notification channel:**
   - Settings > Alerting > Notification channels
   - Type: Email / Slack / Webhook
   - Configure as needed

## Part 6: Organize Dashboards

1. **Create folders:**
   - Dashboards > Manage > New Folder
   - Create folders: "Trading", "System", "Risk"

2. **Move dashboards to folders**

3. **Create Playlist:**
   - Dashboards > Playlists > New Playlist
   - Add your dashboards
   - Set interval (e.g., 10s)
   - Useful for monitoring displays

## Part 7: Dashboard Best Practices

### Panel Organization
```
Row 1: Key Metrics (Stats/Gauges)
Row 2: Trends (Time series)
Row 3: Distributions (Heatmaps/Histograms)
Row 4: Details (Tables)
```

### Color Schemes
- Use consistent colors for similar metrics
- Red/Green for profit/loss
- Traffic light colors for status (Green/Yellow/Red)

### Refresh Rates
- Real-time trading: 5-10s
- Analysis dashboards: 1m
- Historical dashboards: None (manual refresh)

## Part 8: Export and Import Dashboards

### Export Dashboard
1. Dashboard settings > JSON Model
2. Copy JSON or Save to file
3. Share with team or version control

### Import Dashboard
1. Create > Import
2. Upload JSON file or paste JSON
3. Select data source
4. Import

## Part 9: Advanced Panel Types

### Table Panel Example
```promql
sort_desc(
  sum by (symbol) (increase(order_count_total[24h]))
)
```
- Visualization: Table
- Columns: Symbol, Order Count
- Sort by: Order Count (desc)

### Heatmap Panel Example
```promql
sum(increase(order_latency_seconds_bucket[5m])) by (le)
```
- Useful for latency distributions
- Shows patterns over time

## Part 10: Create a Mobile-Friendly Dashboard

1. **Create new dashboard**
2. **Use single-stat and gauge panels**
3. **Limit to 4-6 panels**
4. **Enable kiosk mode:**
   ```
   http://localhost:3000/d/<dashboard-id>?kiosk
   ```

## Tasks to Complete

- [ ] Successfully login to Grafana
- [ ] Configure Prometheus data source
- [ ] Create Trading Overview dashboard with 4+ panels
- [ ] Create System Performance dashboard
- [ ] Add at least one variable to a dashboard
- [ ] Configure an alert rule
- [ ] Export a dashboard as JSON
- [ ] Create a custom color scheme
- [ ] Organize dashboards into folders

## Sample Dashboard JSON Structure

See `configs/grafana-dashboard-example.json` for a complete example.

## Troubleshooting

**Queries return no data:**
- Verify Prometheus is scraping metrics
- Check time range
- Confirm metric names match

**Dashboard not saving:**
- Check Grafana permissions
- Ensure you're logged in as admin

**Alerts not triggering:**
- Verify alert rule configuration
- Check notification channel settings
- Review alert history

## Expected Outcomes

After completing this exercise, you should be able to:
1. Create informative Grafana dashboards
2. Use various visualization types effectively
3. Configure alerts for critical metrics
4. Use variables for interactive dashboards
5. Export/import dashboards for sharing
6. Organize and manage multiple dashboards

## Next Steps

- Explore Grafana plugins for additional visualization types
- Set up alerting for critical trading metrics
- Create dashboards for different stakeholders (traders, ops, management)
- Implement role-based access control (RBAC) in Grafana

## Additional Resources

- Grafana Documentation: https://grafana.com/docs/
- PromQL Guide: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Dashboard Best Practices: https://grafana.com/docs/grafana/latest/best-practices/
