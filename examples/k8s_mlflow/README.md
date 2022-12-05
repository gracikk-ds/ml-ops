To start cluster run:
---
```bash
minikube start --driver=docker \
    --ports=30080:30080 \
    --ports=30090:30090 \
    --ports=30091:30091 \
    --ports=30432:30432 \
    --ports=30501:30501
    
kubectl apply -f ./db/db-config.yaml 
kubectl apply -f ./db/db-secret.yaml
kubectl apply -f ./db/db-deployment.yaml
kubectl apply -f ./db/pgadmin-deployment.yaml

kubectl apply -f ./minio/minio-config.yaml 
kubectl apply -f ./minio/minio-secret.yaml
kubectl apply -f ./minio/minio-deployment.yaml

kubectl apply -f ./mlflow/mlflow-config.yaml 
kubectl apply -f ./mlflow/mlflow-secret.yaml
kubectl apply -f ./mlflow/mlflow-deployment.yaml
```