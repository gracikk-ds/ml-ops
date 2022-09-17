To start cluster run:
---
```bash
minikube start --driver=docker --ports=30081:30081

kubectl apply -f ./application/inference-config.yaml 
kubectl apply -f ./application/inference-secret.yaml
kubectl apply -f ./application/inference-deployment.yaml
```