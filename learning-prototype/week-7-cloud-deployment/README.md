# Week 7: Cloud Deployment & Scaling

Deploy your trading system to production on AWS and Kubernetes!

## Learning Objectives

By the end of this week, you will:

✅ Understand cloud architecture patterns
✅ Deploy infrastructure with Terraform
✅ Set up AWS ECS Fargate containers
✅ Configure Multi-AZ high availability
✅ Implement auto-scaling
✅ Deploy to Kubernetes
✅ Set up CI/CD pipelines
✅ Manage costs effectively

## Why Cloud Deployment Matters

> "If you're not prepared to be wrong, you'll never come up with anything original." - Ken Robinson

Cloud enables:
- **Scalability**: Handle growth automatically
- **Reliability**: 99.99% uptime
- **Global reach**: Deploy anywhere
- **Cost efficiency**: Pay for what you use
- **Disaster recovery**: Multi-region backup

## Prerequisites

- AWS account (free tier)
- Terraform basics
- Docker knowledge
- Linux/networking fundamentals
- Completed Week 3-6

## Folder Structure

```
week-7-cloud-deployment/
├── README.md (you are here)
├── CONCEPTS.md (cloud architecture)
├── terraform/
│   ├── aws/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
│   ├── modules/
│   │   ├── vpc/
│   │   ├── ecs/
│   │   ├── rds/
│   │   └── msk/
│   └── environments/
│       ├── dev.tfvars
│       ├── staging.tfvars
│       └── prod.tfvars
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   └── README.md
├── ci-cd/
│   ├── github-actions.yml
│   ├── deploy.sh
│   └── rollback.sh
└── exercises/
    ├── exercise_1_terraform.md
    ├── exercise_2_ecs.md
    ├── exercise_3_kubernetes.md
    └── exercise_4_cicd.md
```

## Learning Path

### Day 1: Cloud Architecture (3-4 hours)

**What you'll learn**:
- Infrastructure as Code (IaC)
- Terraform basics
- AWS services overview
- Multi-AZ architecture
- Security best practices

**Key services**:
- **ECS Fargate**: Serverless containers
- **RDS**: Managed PostgreSQL
- **ElastiCache**: Redis cache
- **MSK**: Managed Kafka
- **ALB**: Load balancer

### Day 2: AWS Deployment (4-5 hours)

**What you'll learn**:
- VPC and networking
- ECS cluster setup
- RDS Multi-AZ
- Secrets management
- CloudWatch monitoring

**Deploy with Terraform**:
```bash
cd deployment/aws

# Initialize
terraform init

# Plan (review changes)
terraform plan -out=tfplan

# Apply (deploy!)
terraform apply tfplan

# Get outputs
terraform output alb_dns_name
terraform output rds_endpoint
```

**Cost**: ~$510-720/month (can optimize to $300-400)

### Day 3: Kubernetes Deployment (4-5 hours)

**What you'll learn**:
- Kubernetes concepts (Pods, Services, Deployments)
- High availability patterns
- HorizontalPodAutoscaler
- PodDisruptionBudget
- Zero-downtime deployments

**Key resources**:
```yaml
# Deployment with 3 replicas
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero downtime!

---
# Auto-scaling based on CPU
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

**Deploy**:
```bash
# Apply all manifests
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods
kubectl get hpa
kubectl get pdb

# View logs
kubectl logs -f deployment/trading-api
```

### Day 4: CI/CD Pipeline (3-4 hours)

**What you'll learn**:
- GitHub Actions
- Automated testing
- Build and push Docker images
- Automated deployment
- Rollback strategies

**Pipeline stages**:
1. **Test**: Run unit tests
2. **Build**: Build Docker image
3. **Push**: Push to ECR/Docker Hub
4. **Deploy**: Deploy to ECS/K8s
5. **Verify**: Health checks

**GitHub Actions**:
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run tests
        run: pytest tests/

      - name: Build image
        run: docker build -t trading-api .

      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login ...
          docker push $ECR_URL:latest

      - name: Deploy to ECS
        run: aws ecs update-service --force-new-deployment
```

### Day 5: Scaling & Cost Optimization (2-3 hours)

**What you'll learn**:
- Auto-scaling policies
- Cost monitoring
- Reserved Instances
- Spot instances
- Right-sizing

**Cost optimization strategies**:
- Use Reserved Instances (30-40% savings)
- Spot instances for non-critical (60-70% savings)
- Right-size after monitoring
- Use multi-cloud for best pricing

## Key Concepts

### 1. High Availability (HA)

**Multi-AZ deployment**:
```
┌─────────────────────────────────────┐
│         Load Balancer               │
└─────────┬──────────┬────────────────┘
          │          │
    ┌─────▼────┐  ┌──▼──────┐
    │   AZ-A   │  │  AZ-B   │
    │          │  │         │
    │ ECS Task │  │ECS Task │
    │   RDS    │  │  RDS    │
    │ Primary  │  │Standby  │
    └──────────┘  └─────────┘
```

**Benefits**:
- Survives AZ failures
- Zero downtime deployments
- Automatic failover

### 2. Auto-Scaling

**Metrics-based**:
```python
# Scale up if:
- CPU > 70% for 5 minutes
- Memory > 80%
- Request latency > 500ms p95

# Scale down if:
- CPU < 30% for 10 minutes
- Off-peak hours
```

**Example**:
```
Normal load: 3 containers
Peak load: 10 containers
Night: 2 containers (save costs!)
```

### 3. Zero-Downtime Deployment

**Rolling update**:
```
1. Start new version (v2)
2. Wait for health check
3. Route traffic to v2
4. Stop old version (v1)
5. Repeat for all instances
```

**At any time**: Mix of v1 and v2 running

### 4. Infrastructure as Code

**Benefits**:
- **Version control**: Track changes
- **Reproducible**: Same infra every time
- **Testable**: Test in staging first
- **Documented**: Code IS documentation

**Example (Terraform)**:
```hcl
resource "aws_ecs_service" "api" {
  name            = "trading-api"
  cluster         = aws_ecs_cluster.main.id
  desired_count   = 3  # High availability

  deployment_configuration {
    minimum_healthy_percent = 100  # Zero downtime
    maximum_percent         = 200  # Allow surge
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "trading-api"
    container_port   = 8000
  }
}
```

## Deployment Options Comparison

| Feature | AWS ECS | Kubernetes | Serverless |
|---------|---------|------------|------------|
| **Complexity** | Medium | High | Low |
| **Cost** | $500-700/mo | $115-150/mo | $100-300/mo |
| **Scalability** | Good | Excellent | Excellent |
| **Learning curve** | Moderate | Steep | Easy |
| **Best for** | AWS-only | Multi-cloud | Event-driven |

## Cost Breakdown

### AWS ECS (Production)
- ECS Fargate (2 tasks): $50-70
- RDS Multi-AZ (db.t3.medium): $60-80
- ElastiCache (2 nodes): $30-40
- MSK (3 brokers): $300-400
- Load Balancer: $20-30
- Data transfer: $50-100
- **Total**: $510-720/month

**Optimized** (Reserved Instances):
- **Total**: $300-400/month

### Kubernetes (Self-Managed)
- 3 nodes (t3.medium): $75-90
- EBS storage: $20-30
- Load balancer: $20-30
- **Total**: $115-150/month

### Development (Free/Cheap)
- Local Docker: **$0**
- AWS Free Tier: **$0** (first year)
- DigitalOcean: **$5-10/month**

## Security Best Practices

### 1. Network Security
- All services in private subnets
- Only load balancer in public subnet
- Security groups restrict traffic
- VPC Flow Logs enabled

### 2. Secrets Management
```python
# DON'T store in code
API_KEY = "sk_live_12345..."  # ❌

# DO use AWS Secrets Manager
import boto3
secrets = boto3.client('secretsmanager')
api_key = secrets.get_secret_value(SecretId='alpaca-api-key')['SecretString']  # ✅
```

### 3. IAM Best Practices
- Least privilege principle
- Use roles, not access keys
- Enable MFA
- Rotate credentials regularly

## Success Criteria

You've mastered Week 7 when you can:

✅ Deploy infrastructure with Terraform
✅ Set up ECS Fargate with Auto-Scaling
✅ Configure Multi-AZ high availability
✅ Deploy to Kubernetes
✅ Set up CI/CD pipeline
✅ Implement zero-downtime deployments
✅ Monitor and optimize costs
✅ Secure production environment

## Troubleshooting

### Issue: ECS task won't start
```bash
# Check task logs
aws ecs describe-tasks --tasks TASK_ID
aws logs tail /ecs/trading-api --follow
```

### Issue: High costs
```bash
# Check cost breakdown
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31

# Right-size instances
# Use Reserved Instances
# Enable auto-scaling
```

### Issue: Deployment failed
```bash
# Rollback
kubectl rollout undo deployment/trading-api

# Or with Terraform
terraform apply -auto-approve -var="app_version=v1.2.3"
```

## Resources

### Documentation
- [AWS ECS](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Kubernetes](https://kubernetes.io/docs/)
- [Helm Charts](https://helm.sh/docs/)

### Courses
- AWS Certified Solutions Architect
- Kubernetes (CKA/CKAD)
- Terraform Associate

### Books
- "Kubernetes in Action" by Marko Lukša
- "Terraform: Up & Running" by Yevgeniy Brikman
- "AWS Certified Solutions Architect Study Guide"

**Next**: Week 8 - Advanced Features & Production 🚀
