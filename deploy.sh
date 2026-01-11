#!/usr/bin/env bash
set -euo pipefail
#################################
# VARIABLES DÉFINITIVES
#################################
RESOURCE_GROUP="rg-medical-api"  
ACR_NAME="mlops$(whoami | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')"
CONTAINER_APP_NAME="deploy_mlops" 
CONTAINERAPPS_ENV="env-medical_app"
IMAGE_NAME="medical-api"
IMAGE_TAG="v1"
TARGET_PORT=8000

#################################
# 0) Contexte Azure + Vérification Extensions
#################################
echo "Vérification du contexte Azure..."
az account show --query "{name:name, cloudName:cloudName}" -o json >/dev/null

echo "Vérification/installation des extensions Azure CLI..."

if ! az extension show --name containerapp >/dev/null 2>&1; then
    echo "📦 Installation de l'extension containerapp..."
    az extension add --name containerapp --upgrade -y --only-show-errors
    echo "✅ Extension containerapp installée"
else
    echo "✅ Extension containerapp déjà installée"
    az extension update --name containerapp -y --only-show-errors 2>/dev/null || true
fi

#################################
# 1) Providers nécessaires
#################################
echo "Register providers..."
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.Web --wait
az provider register --namespace Microsoft.OperationalInsights --wait

#################################
# 2) Déterminer la LOCATION autorisée
#################################
echo ""
echo "🔍 Recherche d'une région autorisée pour ACR..."

# Régions à tester (par ordre de préférence)
REGIONS_TO_TEST=("eastus" "westus" "northeurope" "southeastasia" "canadacentral" "japaneast" "uksouth" "australiaeast")
ALLOWED_LOCATION=""

for REGION in "${REGIONS_TO_TEST[@]}"; do
    echo -n "  ✓ Test $REGION... "
    
    # Créer un ACR de test
    TEST_ACR_NAME="test${REGION}$RANDOM"
    
    if az acr create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$TEST_ACR_NAME" \
        --sku Basic \
        --location "$REGION" \
        --output none 2>/dev/null; then
        
        echo "✅ AUTORISÉE"
        ALLOWED_LOCATION="$REGION"
        
        # Nettoyer le test ACR
        echo "    → Nettoyage du test ACR..."
        az acr delete --name "$TEST_ACR_NAME" --resource-group "$RESOURCE_GROUP" --yes --output none 2>/dev/null || true
        
        break
    else
        echo "❌ bloquée"
    fi
done

if [ -z "$ALLOWED_LOCATION" ]; then
    echo ""
    echo "❌ ERREUR: Aucune région trouvée dans les régions testées!"
    echo "   Contactez le support Azure pour élargir les régions autorisées."
    exit 1
fi

LOCATION="$ALLOWED_LOCATION"
echo ""
echo "✅ Région sélectionnée: $LOCATION"

#################################
# 3) Resource Group (créer si nécessaire)
#################################
echo ""
echo "Validation du groupe de ressources..."
RG_LOCATION=$(az group show -n "$RESOURCE_GROUP" --query location -o tsv 2>/dev/null || echo "")

if [ -z "$RG_LOCATION" ]; then
    echo "Création du groupe de ressources en $LOCATION..."
    az group create -n "$RESOURCE_GROUP" -l "$LOCATION" >/dev/null
    echo "✅ RG créé : $RESOURCE_GROUP (location: $LOCATION)"
else
    echo "✅ RG existant trouvé : $RESOURCE_GROUP (location: $RG_LOCATION)"
    if [ "$RG_LOCATION" != "$LOCATION" ]; then
        echo "   ⚠️  RG est en $RG_LOCATION mais ACR sera en $LOCATION (régions différentes OK)"
    fi
fi

#################################
# 4) Création ACR
#################################
echo ""
echo "Création du Container Registry (ACR) en $LOCATION..."

if [[ ! "$ACR_NAME" =~ ^[a-z0-9]{5,50}$ ]]; then
    echo "❌ ERREUR: Nom ACR invalide: $ACR_NAME"
    exit 1
fi

echo "Nom ACR validé: $ACR_NAME"

if az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
    echo "✅ ACR déjà existant : $ACR_NAME"
else
    az acr create \
      --resource-group "$RESOURCE_GROUP" \
      --name "$ACR_NAME" \
      --sku Basic \
      --admin-enabled true \
      --location "$LOCATION" >/dev/null
    sleep 5
    echo "✅ ACR créé : $ACR_NAME"
fi

#################################
# 5) Login ACR + Push image
#################################
echo ""
echo "Connexion au registry..."
az acr login --name "$ACR_NAME" >/dev/null

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv | tr -d '\r')
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')
IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "Build + Tag + Push..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
echo "✅ Image pushée dans ACR"

#################################
# 6) Log Analytics
#################################
echo ""
LAW_NAME="law-mlops-$(whoami)-$RANDOM"
echo "Création Log Analytics: $LAW_NAME"
az monitor log-analytics workspace create -g "$RESOURCE_GROUP" -n "$LAW_NAME" -l "$LOCATION" >/dev/null
sleep 10

LAW_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LAW_NAME" \
    --query customerId -o tsv | tr -d '\r')

LAW_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LAW_NAME" \
    --query primarySharedKey -o tsv | tr -d '\r')
echo "✅ Log Analytics OK"

#################################
# 7) Container Apps Environment
#################################
echo ""
echo "Création/validation Container Apps Environment: $CONTAINERAPPS_ENV"
if ! az containerapp env show -n "$CONTAINERAPPS_ENV" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create \
    -n "$CONTAINERAPPS_ENV" \
    -g "$RESOURCE_GROUP" \
    -l "$LOCATION" \
    --logs-workspace-id "$LAW_ID" \
    --logs-workspace-key "$LAW_KEY" >/dev/null
fi
echo "✅ Environment OK"

#################################
# 8) Déploiement Container App
#################################
echo ""
echo "Déploiement Container App: $CONTAINER_APP_NAME"
if az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp update \
    -n "$CONTAINER_APP_NAME" \
    -g "$RESOURCE_GROUP" \
    --image "$IMAGE" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" >/dev/null
else
  az containerapp create \
    -n "$CONTAINER_APP_NAME" \
    -g "$RESOURCE_GROUP" \
    --environment "$CONTAINERAPPS_ENV" \
    --image "$IMAGE" \
    --ingress external \
    --target-port "$TARGET_PORT" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --min-replicas 1 \
    --max-replicas 1 >/dev/null
fi
echo "✅ Container App OK"

#################################
# 9) URL API
#################################
APP_URL=$(az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r')

echo ""
echo "=========================================="
echo "✅ DÉPLOIEMENT RÉUSSI"
echo "=========================================="
echo "ACR      : $ACR_NAME"
echo "Region   : $LOCATION"
echo "Resource Group: $RESOURCE_GROUP"
echo ""
echo "URLs de l'application :"
echo "  API      : https://$APP_URL"
echo "  Health   : https://$APP_URL/health"
echo "  Docs     : https://$APP_URL/docs"
echo ""
echo "Pour supprimer toutes les ressources :"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo "=========================================="