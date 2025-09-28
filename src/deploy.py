# deploy.py
import mlflow.sagemaker as mfs
import os

# Configuration
app_name = 'insurance-prediction-endpoint'
aws_region = 'us-east-1' # Change to your preferred AWS region
aws_account_id = '<your-aws-account-id>' # CHANGE THIS
# Ensure this IAM role exists and has SageMaker & S3 permissions
execution_role_arn = 'arn:aws:iam::<your-aws-account-id>:role/SageMakerExecutionRole' # CHANGE THIS

# Set the MLflow tracking URI. This points to your local tracking server.
mlflow.set_tracking_uri("./mlruns")

# Define the model URI from the MLflow Model Registry
# This will deploy Version 1 of the registered Random Forest model.
# Change the version number as needed.
model_uri = "models:/InsuranceCostModelRF/1"

# Deploy the model to SageMaker
print(f"Deploying model '{model_uri}' to SageMaker endpoint '{app_name}'...")
mfs.deploy(
    app_name=app_name,
    model_uri=model_uri,
    region_name=aws_region,
    mode=mfs.DEPLOYMENT_MODE_CREATE, # Use 'REPLACE' to update an existing endpoint
    execution_role_arn=execution_role_arn,
    instance_type='ml.t2.medium', # Choose an appropriate instance type for your needs
    instance_count=1,
)

print(f"Endpoint '{app_name}' deployed successfully.")