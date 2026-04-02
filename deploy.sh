#!/bin/bash

# Build and push Docker images to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build images
docker build -t ml-api -f src/Dockerfile.api .
docker build -t ml-streamlit -f src/Dockerfile.streamlit .

# Tag and push
docker tag ml-api:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest
docker tag ml-streamlit:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ml-streamlit:latest

docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ml-streamlit:latest

# Deploy to EC2
scp -i key.pem docker-compose.yml ec2-user@$EC2_IP:~
ssh -i key.pem ec2-user@$EC2_IP "docker-compose up -d"