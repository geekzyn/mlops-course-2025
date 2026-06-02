# MLOps End-to-End Pipeline 

This guide demonstrates how to implement a complete MLOps pipeline featuring automated CI/CD workflows and cloud deployment using AWS ECR and App Runner. You will learn how to implement MLflow for experiment tracking and model versioning.You'll learn to build end-to-end automation that handles data changes, model retraining, containerization, and deployment.

## Prerequisites

* [Python](https://www.python.org)
* [Docker](https://docs.docker.com/get-docker/)
* [AWS CLI](https://aws.amazon.com/cli/)
* [GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github)

## Problem Statement

As MLOps projects reach production maturity, several advanced challenges emerge:
- **Experiment Tracking**: Need to track model experiments, parameters, and metrics systematically
- **Model Registry**: Centralized model versioning and lifecycle management
- **Automated Retraining**: Trigger model updates when data or code changes
- **Deployment**: Seamless container orchestration and serving infrastructure

## Solution Architecture

This course implements a mature MLOps pipeline with:
- **MLflow for Experiment Tracking and Model Registry**: Centralized experiment management
- **AWS ECR for Container Registry**: Secure Docker image storage and versioning  
- **AWS App Runner for Model Serving**: Serverless container deployment platform
- **GitHub Actions CI/CD**: Automated workflows with manual approval gates

## 1. Project Structure

```bash
mlops-course-04/
├── src/
│   ├── data/
│   ├── models/
│   ├── pipelines/
│   │   ├── clean.py          # Data cleaning pipeline
│   │   ├── ingest.py         # Data ingestion pipeline
│   │   ├── predict.py        # Model prediction pipeline
│   │   └── train.py          # Model training pipeline
│   ├── .dvc/                 # DVC configuration
│   ├── .gitignore            # Files to ignore in Git
│   ├── app.py                # FastAPI model serving application
│   ├── config.yml            # ML pipeline configuration
│   ├── data.dvc              # DVC data tracking metadata
│   ├── Dockerfile            # Container definition
│   ├── main.py               # MLflow-integrated pipeline orchestrator
│   └── requirements.txt      # Python dependencies
├── terraform/                # Infrastructure as Code
│   ├── modules/
│   │   ├── apprunner-service/   # App Runner service module
│   │   ├── ecr-repository/      # ECR repository module  
│   │   └── s3-bucket/           # S3 bucket module
│   ├── backends/
│   │   └── dev.conf             # Terraform backend configuration
│   ├── environments/
│   │   └── dev.tfvars           # Environment-specific variables
│   ├── apprunner_services.tf    # App Runner infrastructure
│   ├── ecr_repositories.tf      # ECR infrastructure
│   ├── provider.tf              # Terraform provider configuration
│   ├── s3_buckets.tf            # S3 infrastructure
│   └── variables.tf             # Variable definitions
├── .github/
│   └── workflows/
│       ├── app-cicd-dev.yml     # Application CI/CD pipeline
│       └── infra-cicd-dev.yml   # Infrastructure CI/CD pipeline
└── README.md
```

## 2. Add ECR and App Runner Services to Infrastructure

### ECR Repository Module
The ECR (Elastic Container Registry) module provides secure Docker image storage:

```hcl
# modules/ecr-repository/main.tf
resource "aws_ecr_repository" "ecr" {
  name                 = local.name
  image_tag_mutability = var.image_tag_mutability
  dynamic "image_scanning_configuration" {
    for_each = var.image_scanning_configuration
    content {
      scan_on_push = var.image_scanning_configuration.scan_on_push
    }
  }
  tags = var.tags
}
```

### App Runner Service Module  
The App Runner module provides serverless container deployment:

```hcl
# modules/apprunner-service/main.tf
resource "aws_apprunner_service" "ars" {
  service_name = local.service_name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.iamr.arn
    }
    image_repository {
      image_identifier      = var.source_configuration.image_repository.image_identifier
      image_repository_type = var.source_configuration.image_repository.image_repository_type
      image_configuration {
        port = var.source_configuration.image_repository.image_configuration.port
      }
    }
    auto_deployments_enabled = var.source_configuration.auto_deployments_enabled
  }
  tags = var.tags
}
```

### Environment Configuration
```hcl
# environments/dev.tfvars
ecr_repositories = [
  {
    key                  = "mlops-course-ehb-repository"
    image_tag_mutability = "MUTABLE"
    image_scanning_configuration = {
      scan_on_push = true
    }
    tags = {}
  }
]

apprunner_services = [
  {
    key = "mlops-course-ehb-app"
    source_configuration = {
      image_repository = {
        image_identifier      = "926022988101.dkr.ecr.eu-west-1.amazonaws.com/ecr-mlops-course-ehb-repository-dev:latest"
        image_repository_type = "ECR"
        image_configuration = {
          port = 80
        }
      }
      auto_deployments_enabled = true
    }
    tags = {}
  }
]
```

## 3. MLflow Integration for Experiment Tracking

### Enhanced Main Pipeline with MLflow
The `main.py` file now integrates MLflow for comprehensive experiment tracking:

```python
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

def mlflow_main():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Model Training Experiment")

    with mlflow.start_run() as run:
        # Execute ML pipeline steps...
        
        # Log model parameters
        model_params = config['model']['params']
        mlflow.log_params(model_params)
        
        # Log performance metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc", roc_auc_score)
        mlflow.log_metric('precision', report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])
        
        # Log model with signature
        signature = mlflow.models.infer_signature(
            model_input=X_train, 
            model_output=trainer.pipeline.predict(X_test)
        )
        mlflow.sklearn.log_model(trainer.pipeline, "model", signature=signature)

        # Register model in MLflow Model Registry
        model_name = "insurance_model" 
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
```

mlflow ui
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

### MLflow Benefits
- **Experiment Comparison**: Track multiple model runs and compare performance
- **Parameter Tracking**: Automatically log hyperparameters and model configurations
- **Model Registry**: Centralized model versioning with lifecycle management
- **Artifact Storage**: Store models, plots, and other experiment artifacts
- **Reproducibility**: Complete experiment reproduction from logged metadata

## 4. CI/CD Workflows

### Infrastructure CI/CD with Manual Approval
```yaml
# .github/workflows/infra-cicd-dev.yml
name: Infrastructure CI/CD

on:
  pull_request:
    branches: [ "main" ]
    paths: 
      - 'mlops-course-04/terraform/**'
  workflow_dispatch:

jobs:
  terraform-plan-apply:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: mlops-course-04/terraform
    permissions:
      issues: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: eu-west-1

      - name: Terraform Format
        run: terraform fmt -check
        continue-on-error: true

      - name: Terraform Init
        run: terraform init --backend-config='backends/dev.conf'

      - name: Terraform Validate
        run: terraform validate -no-color

      - name: Terraform Plan
        run: terraform plan -no-color --var-file='environments/dev.tfvars' -out=plan.tfout
      
      - name: Approval
        uses: trstringer/manual-approval@v1
        timeout-minutes: 60
        with:
          secret: ${{ github.token }}
          approvers: geekzyn
          issue-title: "Deploy Terraform Plan to dev"
          issue-body: "Please review the Terraform Plan"
          exclude-workflow-initiator-as-approver: false

      - name: Terraform Apply
        run: terraform apply -auto-approve plan.tfout
```

### Application CI/CD with Automated Retraining
```yaml
# .github/workflows/app-cicd-dev.yml
name: Application CI/CD 

on:
  pull_request:
    branches: [ "main" ]
    paths: 
      - 'mlops-course-04/src/**'
  workflow_dispatch:

jobs:
  retrain-build-push:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: mlops-course-04/src

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: eu-west-1
      
      - name: Pull latest data with DVC
        run: dvc pull

      - name: Retrain model
        run: python main.py

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build, tag, and push docker image to Amazon ECR 
        env:
          REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          REPOSITORY: ${{ secrets.ECR_REPOSITORY }}
          IMAGE_TAG: latest
        run: |
          docker build -t $REGISTRY/$REPOSITORY:$IMAGE_TAG .
          docker push $REGISTRY/$REPOSITORY:$IMAGE_TAG
```

## 5. GitHub Actions Workflow

### GitHub Secrets Configuration
Set up the following secrets in your GitHub repository:
- `AWS_ACCESS_KEY_ID`: AWS access key for automation
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for automation  
- `ECR_REPOSITORY`: ECR repository name (e.g., `ecr-mlops-course-ehb-repository-dev`)

### Automated Trigger Points
The CI/CD pipeline automatically triggers on:
- **Code Changes**: Modifications to `mlops-course-04/src/**`
- **Data Changes**: When new data is pushed via DVC `mlops-course-04/src/data.dvc`
- **Configuration Updates**: Changes to model parameters or pipeline configuration

### Complete Automation Flow
1. **Data Update**: New data pushed to DVC remote storage
2. **Pull Request**: Developer creates PR with code/config changes
3. **Automated Pipeline**: GitHub Actions pulls latest data and retrains model
4. **Model Registry**: New model version registered in MLflow
5. **Containerization**: Updated model packaged in Docker container
6. **ECR Push**: Container image pushed to AWS ECR
7. **Auto-deployment**: App Runner automatically deploys new container version
8. **Production Serving**: Updated model serves predictions via REST API

### Model Serving Endpoints
Once deployed, your model will be available at:
- **Health Check**: `https://your-app-runner-url.eu-west-1.awsapprunner.com/`
- **API Documentation**: `https://your-app-runner-url.eu-west-1.awsapprunner.com/docs`
- **Predictions**: `https://your-app-runner-url.eu-west-1.awsapprunner.com/predict`

### Sample Prediction Request
```bash
curl -X 'POST' \
  'https://your-app-runner-url.eu-west-1.awsapprunner.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Gender": "Male",
  "Age": 49,
  "HasDrivingLicense": 1,
  "RegionID": 28,
  "Switch": 0,
  "PastAccident": "1-2 Year",
  "AnnualPremium": 1885.05
}'
```

## 6. Key Achievements

This MLOps pipeline achieves:
- **Automated Retraining**: Models update automatically when data changes
- **Experiment Tracking**: Complete visibility into model development lifecycle
- **Model Registry**: Centralized model versioning and artifact management
- **Infrastructure as Code**: Reproducible cloud resources via Terraform
- **Manual Approval Gates**: Production safety with human oversight
- **Container Orchestration**: Scalable, serverless model deployment
- **End-to-End Automation**: From data changes to production deployment

This represents a mature MLOps implementation ready for enterprise production workloads, combining the best practices of DevOps automation with machine learning lifecycle management.
