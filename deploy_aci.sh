#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "🚀 Deploy to Azure Container Instances (ACI)"
echo "=========================================="

RESOURCE_GROUP="rg-medical-api"
LOCATION="westeurope"
CONTAINER_NAME="medical-api"
IMAGE_TAG="latest"
TARGET_PORT=8000
CONTAINER_IMAGE="mcr.microsoft.com/azuredocs/aci-helloworld"  # Placeholder - build your own

#################################
# 1) Providers
#################################
echo ""
echo "📦 Registering providers..."
az provider register --namespace Microsoft.ContainerInstance --wait
echo "✅ Provider registered"

#################################
# 2) Resource Group
#################################
echo ""
echo "📁 Setting up resource group..."
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" >/dev/null 2>&1 || true
echo "✅ Resource group: $RESOURCE_GROUP (region: $LOCATION)"

#################################
# 3) Build Docker image locally
#################################
echo ""
echo "🐳 Building Docker image..."
if docker build -t "$CONTAINER_NAME:$IMAGE_TAG" .; then
    echo "✅ Image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

#################################
# 4) Push to Docker Hub (or use local)
#################################
echo ""
echo "📤 Image options:"
echo "   Option A: Push to Docker Hub and deploy from there"
echo "   Option B: Use ACR alternative (below)"
echo ""
echo "For now, using local Docker image..."

# Convert to base64 for ACI if needed
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"

if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
    echo "🔐 Logging into Docker Hub..."
    echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
    
    DOCKER_IMAGE="${DOCKER_USERNAME}/${CONTAINER_NAME}:${IMAGE_TAG}"
    docker tag "$CONTAINER_NAME:$IMAGE_TAG" "$DOCKER_IMAGE"
    docker push "$DOCKER_IMAGE"
    echo "✅ Image pushed to: $DOCKER_IMAGE"
else
    echo "⚠️  No Docker Hub credentials provided"
    echo "   Set DOCKER_USERNAME and DOCKER_PASSWORD to push to Docker Hub"
    DOCKER_IMAGE="$CONTAINER_NAME:$IMAGE_TAG"
fi

#################################
# 5) Deploy to ACI
#################################
echo ""
echo "🚀 Deploying to Azure Container Instances..."

ACI_NAME="aci-medical-api-$(date +%s)"
CONTAINER_PORT=8000
CPU="1.0"
MEMORY="1"

az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACI_NAME" \
    --image "$DOCKER_IMAGE" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --ports "$CONTAINER_PORT" \
    --protocol TCP \
    --environment-variables \
        ENVIRONMENT=production \
    --restart-policy OnFailure \
    --output none

sleep 10

# Get the IP address
APP_IP=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACI_NAME" \
    --query ipAddress.ip \
    -o tsv 2>/dev/null || echo "Pending...")

echo "✅ Container deployed: $ACI_NAME"

#################################
# 6) Get container info
#################################
echo ""
echo "=========================================="
echo "✅ DEPLOYMENT SUCCESSFUL"
echo "=========================================="
echo ""
echo "Container Details:"
echo "  Name        : $ACI_NAME"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Region      : $LOCATION"
echo "  CPU         : $CPU cores"
echo "  Memory      : $MEMORY GB"
echo ""
echo "Access URLs:"
echo "  HTTP        : http://$APP_IP:$CONTAINER_PORT"
echo "  Health      : http://$APP_IP:$CONTAINER_PORT/health"
echo "  API Docs    : http://$APP_IP:$CONTAINER_PORT/docs"
echo ""
echo "View logs:"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $ACI_NAME"
echo ""
echo "Delete container:"
echo "  az container delete --resource-group $RESOURCE_GROUP --name $ACI_NAME --yes"
echo ""
echo "=========================================="

# Show logs
echo ""
echo "📋 Container logs:"
az container logs --resource-group "$RESOURCE_GROUP" --name "$ACI_NAME" || echo "Logs not yet available"