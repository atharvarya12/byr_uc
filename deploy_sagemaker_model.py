import boto3
import sagemaker
from sagemaker import image_uris

# Configs
region = "us-east-2"
sagemaker_client = boto3.client("sagemaker", region_name=region)
model_name = "random-forest-model"
endpoint_config_name = "random-forest-endpoint-config"
endpoint_name = "random-forest-endpoint"

role_arn = "arn:aws:iam::611610376622:role/ExecutionRole"
model_artifact = "s3://byr-bucket/models/sklearn/model.tar.gz"

# ✅ Get the correct pre-built image from SageMaker registry
image_uri = image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="0.23-1",
    py_version="py3",
    instance_type="ml.m5.large"
)

# 1. Create the model
def create_model():
    response = sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_artifact
        }
    )
    print("✅ Model created:", response["ModelArn"])

# 2. Create endpoint configuration
def create_endpoint_config():
    response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.m5.large",
                "InitialInstanceCount": 1
            }
        ]
    )
    print("✅ Endpoint config created:", response["EndpointConfigArn"])

# 3. Create the endpoint
def create_endpoint():
    response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print("⏳ Creating endpoint... This may take 5–10 minutes.")
    waiter = sagemaker_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print("✅ Endpoint is live:", endpoint_name)

# Run everything
if __name__ == "__main__":
    create_model()
    create_endpoint_config()
    create_endpoint()
