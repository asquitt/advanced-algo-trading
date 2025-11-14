# AWS Deployment Guide

Complete guide to deploy the LLM Trading Platform on AWS using Terraform.

## Architecture

The deployment creates:
- **VPC** with public and private subnets across 3 AZs (High Availability)
- **ECS Fargate** cluster for containerized application
- **Application Load Balancer** for traffic distribution
- **RDS PostgreSQL** (Multi-AZ) for relational data
- **ElastiCache Redis** (Multi-AZ) for caching
- **Amazon MSK** (Kafka) for message streaming
- **CloudWatch** for logging and monitoring
- **ECR** for Docker image storage
- **Secrets Manager** for sensitive data

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured (`aws configure`)
3. **Terraform** (>= 1.0) installed
4. **Docker** for building images
5. **API Keys** (Alpaca, Claude, Groq)

## Cost Estimate

Monthly AWS costs (approximate):
- ECS Fargate (2 tasks): ~$50-70
- RDS Multi-AZ (db.t3.medium): ~$60-80
- ElastiCache (2 nodes): ~$30-40
- MSK (3 brokers): ~$300-400
- Load Balancer: ~$20-30
- Data transfer & storage: ~$50-100
- **Total**: ~$510-720/month

**Savings options**:
- Use Reserved Instances (30-40% savings)
- Use Savings Plans
- Right-size instances after monitoring
- Use Spot instances for non-critical tasks

## Quick Start

### 1. Clone and Configure

```bash
cd deployment/aws

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
aws_region         = "us-east-1"
environment        = "production"
app_name           = "trading-platform"
db_instance_class  = "db.t3.medium"
cache_node_type    = "cache.t3.micro"
ecs_task_cpu       = 1024
ecs_task_memory    = 2048
EOF
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan Deployment

```bash
terraform plan -out=tfplan
```

Review the plan carefully. Expected resources: ~50-60 resources.

### 4. Apply Deployment

```bash
terraform apply tfplan
```

This will take 15-20 minutes.

### 5. Build and Push Docker Image

```bash
# Get ECR repository URL from Terraform output
ECR_URL=$(terraform output -raw ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URL

# Build image
cd ../../  # Back to project root
docker build -t trading-platform .

# Tag and push
docker tag trading-platform:latest $ECR_URL:latest
docker push $ECR_URL:latest
```

### 6. Update ECS Service

```bash
# Force new deployment with updated image
aws ecs update-service \
  --cluster trading-platform-cluster \
  --service trading-platform-service \
  --force-new-deployment
```

### 7. Verify Deployment

```bash
# Get ALB DNS name
ALB_DNS=$(terraform output -raw alb_dns_name)

# Test health endpoint
curl http://$ALB_DNS/health

# Expected response: {"status": "healthy"}
```

## Configuration

### Environment Variables

Add secrets to AWS Secrets Manager:

```bash
# Create secrets
aws secretsmanager create-secret \
  --name trading-platform/alpaca-api-key \
  --secret-string "your_alpaca_key"

aws secretsmanager create-secret \
  --name trading-platform/alpaca-secret-key \
  --secret-string "your_alpaca_secret"

aws secretsmanager create-secret \
  --name trading-platform/claude-api-key \
  --secret-string "your_claude_key"
```

Update ECS task definition to use secrets:

```hcl
secrets = [
  {
    name      = "ALPACA_API_KEY"
    valueFrom = "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:trading-platform/alpaca-api-key"
  }
]
```

### Database Migration

```bash
# Connect to RDS (via bastion or ECS Exec)
aws ecs execute-command \
  --cluster trading-platform-cluster \
  --task TASK_ID \
  --container trading-api \
  --interactive \
  --command "/bin/bash"

# Inside container, run migrations
python scripts/migrate_database.py
```

## Monitoring

### CloudWatch Dashboards

Access CloudWatch dashboard:
```bash
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1
```

Key metrics to monitor:
- ECS CPU/Memory utilization
- RDS connections and CPU
- ElastiCache hit rate
- MSK broker metrics
- ALB request count and latency

### Alarms

Create CloudWatch alarms:

```bash
# High CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name trading-platform-high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold

# Database connections alarm
aws cloudwatch put-metric-alarm \
  --alarm-name trading-platform-db-connections \
  --alarm-description "Alert when DB connections exceed 80%" \
  --metric-name DatabaseConnections \
  --namespace AWS/RDS \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold
```

## Scaling

### Auto Scaling

ECS service auto-scales based on CPU:
- Min instances: 2
- Max instances: 10
- Target CPU: 70%

### Manual Scaling

```bash
# Scale ECS service
aws ecs update-service \
  --cluster trading-platform-cluster \
  --service trading-platform-service \
  --desired-count 4

# Scale RDS
aws rds modify-db-instance \
  --db-instance-identifier trading-platform-postgres \
  --db-instance-class db.t3.large \
  --apply-immediately
```

## Backup and Disaster Recovery

### Automated Backups

- **RDS**: Automated daily backups (7-day retention)
- **ElastiCache**: Daily snapshots (5-day retention)
- **ECS**: Task definitions versioned automatically

### Manual Backups

```bash
# Create RDS snapshot
aws rds create-db-snapshot \
  --db-instance-identifier trading-platform-postgres \
  --db-snapshot-identifier trading-platform-snapshot-$(date +%Y%m%d)

# Create ElastiCache snapshot
aws elasticache create-snapshot \
  --replication-group-id trading-platform-redis \
  --snapshot-name trading-platform-redis-snapshot-$(date +%Y%m%d)
```

### Disaster Recovery

Multi-region DR setup:
1. Enable Cross-Region Replication for RDS
2. Setup CloudFormation StackSets for multi-region
3. Use Route53 for DNS failover
4. Replicate S3 buckets cross-region

## Security

### Network Security

- All services in private subnets
- Only ALB in public subnet
- Security groups restrict traffic flow
- VPC Flow Logs enabled

### Data Encryption

- RDS: Encryption at rest (KMS)
- ElastiCache: Encryption in transit and at rest
- MSK: TLS encryption
- S3: Server-side encryption

### IAM Best Practices

- Least privilege principle
- Separate execution and task roles
- Use IAM roles, not access keys
- Enable MFA for AWS Console

## Troubleshooting

### ECS Task Not Starting

```bash
# Check task events
aws ecs describe-tasks \
  --cluster trading-platform-cluster \
  --tasks TASK_ID

# View logs
aws logs tail /ecs/trading-platform --follow
```

### Database Connection Issues

```bash
# Test connectivity from ECS task
aws ecs execute-command \
  --cluster trading-platform-cluster \
  --task TASK_ID \
  --container trading-api \
  --interactive \
  --command "/bin/bash"

# Inside container
nc -zv $RDS_ENDPOINT 5432
```

### High Latency

1. Check ALB target health
2. Review CloudWatch metrics
3. Enable ECS Container Insights
4. Check RDS slow query logs

## Cost Optimization

### Right-Sizing

```bash
# Analyze ECS task metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-31T23:59:59Z \
  --period 3600 \
  --statistics Average
```

### Reserved Instances

Consider purchasing:
- RDS Reserved Instances (1-3 year)
- ElastiCache Reserved Nodes
- Compute Savings Plans for ECS

### Spot Instances

The deployment uses Fargate Spot for cost savings (configured in capacity providers).

## Cleanup

**WARNING**: This will destroy all resources!

```bash
# Destroy all infrastructure
terraform destroy

# Verify all resources deleted
terraform show
```

## Next Steps

1. Setup CI/CD pipeline (see `deployment/github-actions/`)
2. Configure monitoring dashboards
3. Setup log aggregation
4. Implement blue-green deployments
5. Add WAF for additional security

## Support

- AWS Support: https://aws.amazon.com/support/
- Terraform AWS Provider: https://registry.terraform.io/providers/hashicorp/aws/latest/docs
- Internal docs: `docs/`
