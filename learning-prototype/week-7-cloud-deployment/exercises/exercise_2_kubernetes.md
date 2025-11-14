# Exercise 2: Deploy to Kubernetes

## Objective
Deploy the algo trading platform to a Kubernetes cluster with proper configuration, scaling, and monitoring.

## Prerequisites
- Kubernetes cluster (EKS, GKE, or local minikube)
- kubectl installed and configured
- Docker image pushed to container registry
- Helm installed (optional)

## Part 1: Cluster Setup (30 minutes)

### Task 1.1: Verify Cluster Access
```bash
# Check cluster connection
kubectl cluster-info

# List nodes
kubectl get nodes

# Check current context
kubectl config current-context
```

### Task 1.2: Create Namespace
```bash
# Create trading namespace
kubectl create namespace trading

# Set as default namespace
kubectl config set-context --current --namespace=trading

# Verify
kubectl get namespaces
```

### Task 1.3: Create Docker Registry Secret
```bash
# For Docker Hub
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_PASSWORD \
  --docker-email=YOUR_EMAIL \
  --namespace=trading

# For AWS ECR
kubectl create secret docker-registry ecr-secret \
  --docker-server=ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com \
  --docker-username=AWS \
  --docker-password=$(aws ecr get-login-password --region REGION) \
  --namespace=trading
```

### Task 1.4: Update ConfigMap and Secrets
Edit `kubernetes/deployment.yaml` and update:
- ConfigMap with your database name
- Secret with your credentials (base64 encoded)

```bash
# Encode password
echo -n 'your_password' | base64

# Update the Secret in deployment.yaml
```

## Part 2: Deploy Core Application (45 minutes)

### Task 2.1: Review Deployment Configuration
Open `kubernetes/deployment.yaml` and understand:
- Resource requests and limits
- Health probes (liveness, readiness, startup)
- Environment variables
- Volume mounts
- Anti-affinity rules
- HPA configuration

**Questions to answer:**
- Why do we need both liveness and readiness probes?
- What happens if a pod exceeds its memory limit?
- How does pod anti-affinity improve reliability?

### Task 2.2: Deploy the Application
```bash
# Apply namespace
kubectl apply -f kubernetes/deployment.yaml

# Watch pods come up
kubectl get pods -w

# Check deployment status
kubectl rollout status deployment/algo-trading-deployment
```

### Task 2.3: Verify Deployment
```bash
# Check all resources
kubectl get all -n trading

# Describe deployment
kubectl describe deployment algo-trading-deployment

# Check pod logs
kubectl logs -l app=algo-trading --tail=100
```

## Part 3: Configure Services and Networking (30 minutes)

### Task 3.1: Deploy Services
```bash
# Apply service configuration
kubectl apply -f kubernetes/service.yaml

# Get service details
kubectl get services -n trading

# Describe load balancer service
kubectl describe service algo-trading-service
```

### Task 3.2: Test Internal Communication
```bash
# Create a test pod
kubectl run test-pod --image=busybox -it --rm --restart=Never -- sh

# Inside the pod, test internal service
wget -O- http://algo-trading-internal:8000/health

# Exit test pod
exit
```

### Task 3.3: Configure Ingress (if applicable)
Update the Ingress in `service.yaml` with your domain:
```yaml
spec:
  tls:
    - hosts:
        - your-domain.com
      secretName: trading-tls-cert
  rules:
    - host: your-domain.com
```

Apply changes:
```bash
kubectl apply -f kubernetes/service.yaml

# Check ingress
kubectl get ingress
kubectl describe ingress algo-trading-ingress
```

### Task 3.4: Test External Access
```bash
# Get external IP/hostname
kubectl get service algo-trading-service

# Test health endpoint
EXTERNAL_IP=$(kubectl get service algo-trading-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$EXTERNAL_IP/health
```

## Part 4: Scaling and Auto-scaling (30 minutes)

### Task 4.1: Manual Scaling
```bash
# Scale to 5 replicas
kubectl scale deployment algo-trading-deployment --replicas=5

# Watch pods scale
kubectl get pods -w

# Check pod distribution across nodes
kubectl get pods -o wide
```

### Task 4.2: Configure Horizontal Pod Autoscaler
```bash
# The HPA is already defined in deployment.yaml
# Check HPA status
kubectl get hpa

# Describe HPA
kubectl describe hpa algo-trading-hpa

# Watch HPA in action
kubectl get hpa -w
```

### Task 4.3: Load Test Auto-scaling
Generate load to trigger auto-scaling:
```bash
# Install hey (HTTP load testing tool)
# On Mac: brew install hey
# On Linux: wget https://hey-release.s3.us-east-2.amazonaws.com/hey_linux_amd64

# Run load test
hey -z 5m -c 50 http://$EXTERNAL_IP/api/strategy/backtest

# Watch HPA scale up
kubectl get hpa -w

# Monitor pod count
kubectl get pods -l app=algo-trading
```

**Questions to answer:**
- How long did it take for HPA to scale up?
- What was the maximum number of pods?
- How long did it take to scale down after load stopped?

### Task 4.4: Configure Pod Disruption Budget
The PDB is already in deployment.yaml. Verify:
```bash
kubectl get pdb
kubectl describe pdb algo-trading-pdb
```

## Part 5: Rolling Updates and Rollbacks (30 minutes)

### Task 5.1: Perform Rolling Update
Update the image tag in deployment.yaml:
```yaml
containers:
  - name: trading-app
    image: your-registry/algo-trading:v2.0
```

Apply update:
```bash
kubectl apply -f kubernetes/deployment.yaml

# Watch rolling update
kubectl rollout status deployment/algo-trading-deployment

# Check rollout history
kubectl rollout history deployment/algo-trading-deployment
```

### Task 5.2: Monitor Update Progress
```bash
# Watch pods during update
kubectl get pods -w

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

**Questions to answer:**
- How many pods were unavailable during the update?
- Did the update cause any downtime?
- How long did the complete update take?

### Task 5.3: Rollback Deployment
```bash
# Rollback to previous version
kubectl rollout undo deployment/algo-trading-deployment

# Watch rollback
kubectl rollout status deployment/algo-trading-deployment

# Verify rollback
kubectl get pods
kubectl describe deployment algo-trading-deployment | grep Image
```

### Task 5.4: Rollback to Specific Revision
```bash
# View history
kubectl rollout history deployment/algo-trading-deployment

# Rollback to revision 2
kubectl rollout undo deployment/algo-trading-deployment --to-revision=2
```

## Part 6: Monitoring and Debugging (45 minutes)

### Task 6.1: View Logs
```bash
# Logs from all pods
kubectl logs -l app=algo-trading --tail=100

# Follow logs
kubectl logs -l app=algo-trading -f

# Logs from specific pod
kubectl logs POD_NAME

# Previous pod logs (after crash)
kubectl logs POD_NAME --previous
```

### Task 6.2: Execute Commands in Pods
```bash
# Get a shell in a pod
kubectl exec -it POD_NAME -- /bin/bash

# Run a one-off command
kubectl exec POD_NAME -- env

# Check database connectivity
kubectl exec POD_NAME -- nc -zv postgres-service 5432
```

### Task 6.3: Debug Network Issues
```bash
# Check DNS resolution
kubectl run test-dns --image=busybox -it --rm -- nslookup algo-trading-internal

# Test service connectivity
kubectl run test-curl --image=curlimages/curl -it --rm -- \
  curl http://algo-trading-internal:8000/health

# Check network policies
kubectl get networkpolicies
kubectl describe networkpolicy algo-trading-netpol
```

### Task 6.4: Resource Usage
```bash
# Check resource usage
kubectl top nodes
kubectl top pods

# Describe pod resource usage
kubectl describe pod POD_NAME | grep -A 5 "Limits\|Requests"
```

### Task 6.5: Events and Troubleshooting
```bash
# Get recent events
kubectl get events --sort-by='.lastTimestamp' | tail -20

# Describe pod for issues
kubectl describe pod POD_NAME

# Check pod status
kubectl get pods -o wide
```

## Part 7: Advanced Configuration (30 minutes)

### Task 7.1: Configure Resource Quotas
Create resource quota:
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: trading-quota
  namespace: trading
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "5"
```

Apply and verify:
```bash
kubectl apply -f resource-quota.yaml
kubectl describe resourcequota trading-quota
```

### Task 7.2: Configure Network Policies
The network policy is in service.yaml. Verify it:
```bash
kubectl get networkpolicies
kubectl describe networkpolicy algo-trading-netpol
```

Test network isolation:
```bash
# Try to access from different namespace (should fail)
kubectl run test-outside --image=curlimages/curl -it --rm --namespace=default -- \
  curl http://algo-trading-service.trading:8000/health
```

### Task 7.3: Persistent Storage
```bash
# Check PVC status
kubectl get pvc

# Describe PVC
kubectl describe pvc trading-pvc

# Check mounted volumes in pods
kubectl describe pod POD_NAME | grep -A 5 "Volumes"
```

## Part 8: Cleanup (15 minutes)

### Task 8.1: Delete Application
```bash
# Delete deployments and services
kubectl delete -f kubernetes/deployment.yaml
kubectl delete -f kubernetes/service.yaml

# Verify deletion
kubectl get all -n trading
```

### Task 8.2: Delete Namespace
```bash
# Delete namespace (removes all resources)
kubectl delete namespace trading

# Verify
kubectl get namespaces
```

## Challenges

### Challenge 1: Multi-Environment Setup
Create separate deployments for dev, staging, and production:
- Use Kustomize or Helm
- Different resource limits per environment
- Separate namespaces
- Environment-specific configurations

### Challenge 2: StatefulSet Deployment
Convert the deployment to use StatefulSet:
- Maintain pod identity
- Use persistent volumes per pod
- Implement ordered deployment
- Configure headless service

### Challenge 3: Service Mesh
Implement Istio service mesh:
- Install Istio
- Configure traffic management
- Implement circuit breakers
- Set up distributed tracing

### Challenge 4: GitOps with ArgoCD
Set up GitOps deployment:
- Install ArgoCD
- Configure application sync
- Implement automated deployments
- Set up rollback procedures

## Common Issues

### Issue 1: ImagePullBackOff
**Symptom**: Pods stuck in ImagePullBackOff

**Solution**:
```bash
# Check image name
kubectl describe pod POD_NAME

# Verify registry secret
kubectl get secret regcred -o yaml

# Recreate secret if needed
```

### Issue 2: CrashLoopBackOff
**Symptom**: Pods continuously crashing

**Solution**:
```bash
# Check logs
kubectl logs POD_NAME --previous

# Check events
kubectl describe pod POD_NAME

# Common causes: wrong DB credentials, missing dependencies
```

### Issue 3: Pods Not Scheduling
**Symptom**: Pods stuck in Pending state

**Solution**:
```bash
# Check events
kubectl describe pod POD_NAME

# Check node resources
kubectl top nodes

# Check resource quotas
kubectl describe resourcequota
```

## Learning Objectives Checklist

- [ ] Deploy applications to Kubernetes
- [ ] Configure services and ingress
- [ ] Implement auto-scaling (HPA)
- [ ] Perform rolling updates
- [ ] Handle rollbacks
- [ ] Monitor and debug applications
- [ ] Configure resource limits and quotas
- [ ] Implement network policies
- [ ] Manage persistent storage
- [ ] Handle secrets and config maps
- [ ] Understand pod lifecycle
- [ ] Configure health checks

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kubernetes Patterns](https://www.redhat.com/en/resources/cloud-native-container-design-whitepaper)
- [Production Best Practices](https://learnk8s.io/production-best-practices)

## Submission

Document your deployment with:
1. Screenshots of running pods
2. Service endpoints and test results
3. HPA scaling screenshots
4. Rolling update process
5. Monitoring dashboards
6. Answers to all questions
