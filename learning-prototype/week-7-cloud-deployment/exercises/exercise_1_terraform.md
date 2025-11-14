# Exercise 1: Deploy Infrastructure with Terraform

## Objective
Deploy a complete AWS infrastructure for the algo trading platform using Terraform, including VPC, ECS, RDS, and load balancers.

## Prerequisites
- AWS account with appropriate permissions
- Terraform installed (v1.0+)
- AWS CLI configured
- Docker image built and pushed to ECR

## Part 1: Setup and Configuration (30 minutes)

### Task 1.1: Configure AWS Credentials
```bash
# Configure AWS CLI
aws configure

# Verify access
aws sts get-caller-identity
```

### Task 1.2: Create Terraform Backend
Create an S3 bucket for Terraform state:
```bash
aws s3 mb s3://algo-trading-terraform-state
aws s3api put-bucket-versioning \
  --bucket algo-trading-terraform-state \
  --versioning-configuration Status=Enabled
```

### Task 1.3: Create backend.tf
Create a file `terraform/aws/backend.tf`:
```hcl
terraform {
  backend "s3" {
    bucket         = "algo-trading-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}
```

### Task 1.4: Create terraform.tfvars
Create your variables file with actual values:
```hcl
project_name        = "algo-trading"
environment         = "dev"
aws_region          = "us-east-1"
db_password         = "YOUR_SECURE_PASSWORD_HERE"
container_image     = "YOUR_ECR_IMAGE_URI"
ecs_desired_count   = 2
ecs_min_capacity    = 2
ecs_max_capacity    = 10
```

**Important**: Never commit `terraform.tfvars` to version control!

## Part 2: Deploy Infrastructure (45 minutes)

### Task 2.1: Initialize Terraform
```bash
cd terraform/aws
terraform init
```

**Questions to answer:**
- What providers were downloaded?
- Where is the state stored?

### Task 2.2: Validate Configuration
```bash
terraform validate
terraform fmt -check
```

**Fix any errors before proceeding.**

### Task 2.3: Review Execution Plan
```bash
terraform plan -out=tfplan
```

**Questions to answer:**
- How many resources will be created?
- What are the most critical resources?
- What are the estimated costs?

### Task 2.4: Deploy Infrastructure
```bash
terraform apply tfplan
```

This will take 10-15 minutes. Monitor the output carefully.

**Questions to answer:**
- Did all resources create successfully?
- What are the outputs (VPC ID, ALB DNS, RDS endpoint)?

### Task 2.5: Verify Deployment
```bash
# Check VPC
aws ec2 describe-vpcs --filters "Name=tag:Name,Values=algo-trading-vpc"

# Check ECS cluster
aws ecs describe-clusters --clusters algo-trading-cluster

# Check RDS instance
aws rds describe-db-instances --db-instance-identifier algo-trading-db

# Check load balancer
aws elbv2 describe-load-balancers --names algo-trading-alb
```

## Part 3: Test the Deployment (30 minutes)

### Task 3.1: Check ECS Service
```bash
# Get service status
aws ecs describe-services \
  --cluster algo-trading-cluster \
  --services algo-trading-service

# Get running tasks
aws ecs list-tasks \
  --cluster algo-trading-cluster \
  --service-name algo-trading-service
```

### Task 3.2: Test Load Balancer
```bash
# Get ALB DNS name
ALB_DNS=$(terraform output -raw alb_dns_name)

# Test health endpoint
curl http://$ALB_DNS/health

# Expected response: {"status": "healthy", "timestamp": "..."}
```

### Task 3.3: Test Database Connection
```bash
# Get RDS endpoint
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)

# Test connection (requires psql)
psql -h $RDS_ENDPOINT -U admin -d trading_db

# Run a test query
SELECT version();
```

## Part 4: Modify and Update (30 minutes)

### Task 4.1: Scale ECS Service
Modify `terraform.tfvars`:
```hcl
ecs_desired_count = 3
ecs_max_capacity  = 15
```

Apply changes:
```bash
terraform plan
terraform apply
```

**Questions to answer:**
- How long did scaling take?
- Were there any service interruptions?

### Task 4.2: Add Tags
Update `terraform.tfvars`:
```hcl
tags = {
  Project     = "AlgoTrading"
  Team        = "Trading-Platform"
  CostCenter  = "Engineering"
  Compliance  = "SOC2"
}
```

Apply changes:
```bash
terraform apply
```

### Task 4.3: Review State
```bash
# List all resources in state
terraform state list

# Show specific resource
terraform state show aws_ecs_service.trading_service

# Show outputs
terraform output
```

## Part 5: Monitoring and Logging (30 minutes)

### Task 5.1: Check CloudWatch Logs
```bash
# List log streams
aws logs describe-log-streams \
  --log-group-name /ecs/algo-trading

# Tail logs
aws logs tail /ecs/algo-trading --follow
```

### Task 5.2: Review Metrics
```bash
# Get ECS service metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=algo-trading-service \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

### Task 5.3: Set Up Alarms
Create a CloudWatch alarm for high CPU:
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name algo-trading-high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## Part 6: Cleanup (15 minutes)

### Task 6.1: Destroy Infrastructure
**WARNING**: This will delete all resources!

```bash
terraform destroy
```

Review the plan carefully before confirming.

### Task 6.2: Verify Cleanup
```bash
# Check for remaining resources
aws ec2 describe-vpcs --filters "Name=tag:Name,Values=algo-trading-vpc"
aws ecs list-clusters
aws rds describe-db-instances
```

## Challenges and Troubleshooting

### Challenge 1: Cost Optimization
Modify the infrastructure to reduce costs while maintaining high availability:
- Use smaller instance types
- Implement reserved instances
- Use S3 for static assets
- Optimize RDS instance size

### Challenge 2: Multi-Region Deployment
Extend the Terraform configuration to deploy to multiple regions:
- Create a modules structure
- Deploy to us-east-1 and us-west-2
- Set up cross-region replication for RDS

### Challenge 3: Blue-Green Deployment
Implement blue-green deployment:
- Create two identical environments
- Use Route53 for traffic shifting
- Automate rollback on failure

## Common Issues

### Issue 1: Insufficient IAM Permissions
**Error**: `UnauthorizedOperation: You are not authorized to perform this operation`

**Solution**: Ensure your IAM user/role has the following policies:
- AmazonVPCFullAccess
- AmazonECS_FullAccess
- AmazonRDSFullAccess
- ElasticLoadBalancingFullAccess

### Issue 2: Resource Limits
**Error**: `LimitExceededException: You have exceeded the limit`

**Solution**: Request limit increases or clean up unused resources:
```bash
aws service-quotas list-service-quotas --service-code ec2
```

### Issue 3: State Lock
**Error**: `Error locking state: state is already locked`

**Solution**: Force unlock (use carefully):
```bash
terraform force-unlock LOCK_ID
```

## Learning Objectives Checklist

- [ ] Understand Terraform state management
- [ ] Deploy multi-tier AWS infrastructure
- [ ] Configure networking (VPC, subnets, security groups)
- [ ] Set up container orchestration with ECS
- [ ] Deploy and manage RDS databases
- [ ] Configure load balancing and auto-scaling
- [ ] Implement monitoring and logging
- [ ] Manage infrastructure as code
- [ ] Handle Terraform state and locking
- [ ] Perform updates and rollbacks

## Additional Resources

- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

## Submission

Document your deployment with:
1. Screenshots of AWS console showing resources
2. Terraform output values
3. Answers to all questions
4. CloudWatch metrics screenshots
5. Cost estimate from AWS Cost Explorer
