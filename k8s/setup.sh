#!/bin/bash
# StreamLens Local Kubernetes Demo
# Proves: containerized deployment, health checks, rolling restart, HPA

set -e

echo "=== StreamLens Local K8s Setup ==="

# 1. Install kind if needed
if ! command -v kind &> /dev/null; then
  echo "Installing kind..."
  brew install kind kubectl
fi

# 2. Create cluster
echo "Creating kind cluster..."
kind create cluster --name streamlens --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
- role: worker
EOF

# 3. Load local Docker image into kind
echo "Loading API image into kind..."
docker build -t streamlens-api:latest -f docker/Dockerfile.api .
kind load docker-image streamlens-api:latest --name streamlens

# 4. Deploy
echo "Deploying to Kubernetes..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/hpa.yaml

# 5. Wait for pods
echo "Waiting for pods..."
kubectl wait --for=condition=ready pod -l app=streamlens-api -n streamlens --timeout=120s

# 6. Port forward
echo "Port forwarding..."
kubectl port-forward svc/streamlens-api 8001:8000 -n streamlens &

echo ""
echo "=== StreamLens running on Kubernetes ==="
echo "API:     http://localhost:8001/health"
echo "Pods:    kubectl get pods -n streamlens"
echo "Logs:    kubectl logs -l app=streamlens-api -n streamlens"
echo "Scale:   kubectl scale deployment streamlens-api --replicas=3 -n streamlens"
echo "Rolling: kubectl rollout restart deployment/streamlens-api -n streamlens"
