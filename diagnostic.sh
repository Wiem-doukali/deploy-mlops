#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "🔍 DIAGNOSTIC: Régions & Policies Azure"
echo "=========================================="

# 1. Afficher subscription info
echo ""
echo "📋 Subscription Info:"
az account show --query "{name:name, subscriptionId:id, cloudName:cloudName}" -o json

# 2. Lister TOUTES les régions disponibles
echo ""
echo "🌍 Toutes les régions Azure disponibles:"
az account list-locations --query "[].{Name:displayName, Code:name}" -o table

# 3. Vérifier les policies au niveau subscription
echo ""
echo "🔐 Policy Assignments au niveau Subscription:"
az policy assignment list \
  --scope "/subscriptions/$(az account show --query id -o tsv)" \
  --query "[].{Name:displayName, Id:id, Scope:scope}" -o table 2>/dev/null || echo "Aucune policy trouvée"

# 4. Vérifier les policies au niveau resource group
RESOURCE_GROUP="rg-medical-api"
echo ""
echo "🔐 Policy Assignments au niveau Resource Group ($RESOURCE_GROUP):"
az policy assignment list \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP" \
  --query "[].{Name:displayName, Id:id}" -o table 2>/dev/null || echo "Aucune policy trouvée"

# 5. Test TOUTES les régions pour ACR
echo ""
echo "🧪 Test ACR dans TOUTES les régions (peut prendre du temps)..."
echo ""

WORKING_REGIONS=()
FAILED_REGIONS=()

# Récupérer TOUTES les régions
ALL_REGIONS=$(az account list-locations --query "[].name" -o tsv)

for REGION in $ALL_REGIONS; do
    echo -n "  Test $REGION... "
    
    TEST_ACR_NAME="diag${REGION}$RANDOM"
    
    if az acr create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$TEST_ACR_NAME" \
        --sku Basic \
        --location "$REGION" \
        --output none 2>/dev/null; then
        
        echo "✅ FONCTIONNE!"
        WORKING_REGIONS+=("$REGION")
        
        # Nettoyer
        az acr delete --name "$TEST_ACR_NAME" --resource-group "$RESOURCE_GROUP" --yes --output none 2>/dev/null || true
    else
        echo "❌"
        FAILED_REGIONS+=("$REGION")
    fi
done

# 6. Résumé
echo ""
echo "=========================================="
echo "📊 RÉSUMÉ"
echo "=========================================="

if [ ${#WORKING_REGIONS[@]} -gt 0 ]; then
    echo ""
    echo "✅ Régions AUTORISÉES pour ACR:"
    printf '   %s\n' "${WORKING_REGIONS[@]}"
    echo ""
    echo "💡 À utiliser dans deploy.sh:"
    echo "   LOCATION=\"${WORKING_REGIONS[0]}\""
else
    echo ""
    echo "❌ AUCUNE région n'est autorisée pour ACR!"
    echo ""
    echo "Actions recommandées:"
    echo "1. Contactez le support Azure"
    echo "2. Demandez à étendre les régions autorisées"
    echo "3. Vérifiez les policies appliquées:"
    echo "   - Au niveau subscription"
    echo "   - Au niveau resource group"
    echo "   - Au niveau management group"
fi

echo ""
echo "❌ Régions BLOQUÉES (${#FAILED_REGIONS[@]}):"
printf '   %s\n' "${FAILED_REGIONS[@]}" | head -10
if [ ${#FAILED_REGIONS[@]} -gt 10 ]; then
    echo "   ... et $((${#FAILED_REGIONS[@]} - 10)) autres"
fi

echo ""
echo "=========================================="