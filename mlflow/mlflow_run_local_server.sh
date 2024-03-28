#!/usr/bin/env sh

echo "Run MLflow tracking server ..."
# An MLflow tracking server has two components for storage:
# (1) backend store
# (2) artifact store
mlflow server --backend-store-uri sqlite:///mlregistry.db --default-artifact-root smb://d:/fishtail/projects/ft-vessel-utilization/mlflow/mlruns
