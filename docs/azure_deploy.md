# Guia de despliegue en Azure

Esta guia mantiene el despliegue simple: probar local, crear Azure SQL barato, subir una imagen a ACR y ejecutar el contenedor en App Service o Container Apps. Antes de crear recursos que generen costo, confirma el plan, region y presupuesto.

Referencias oficiales:

- ODBC Driver 18 para SQL Server en Linux: https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server
- Azure SQL Database modelos de compra: https://learn.microsoft.com/en-us/azure/azure-sql/database/purchasing-models
- Firewall de Azure SQL Database: https://learn.microsoft.com/en-us/azure/azure-sql/database/firewall-create-server-level-portal-quickstart
- Azure Container Registry con Docker CLI: https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-portal
- App Service con contenedor custom: https://learn.microsoft.com/en-us/azure/app-service/tutorial-custom-container
- App settings de App Service: https://learn.microsoft.com/en-us/azure/app-service/configure-common
- Ingress de Azure Container Apps: https://learn.microsoft.com/en-us/azure/container-apps/ingress-overview

## 1. Probar local primero

Desde la raiz del repo:

```bash
uv sync --no-dev
uv run gunicorn --bind 0.0.0.0:${PORT:-8000} model_final.app:app
```

Abrir `http://127.0.0.1:8000` y verificar:

- El dashboard carga.
- `GET /api/transactions?filter=todas&page=1&page_size=5` responde JSON paginado.
- `POST /api/transactions` con `{"scenario":"anomala"}` inserta una transaccion sintetica y devuelve su etiqueta.
- `GET /api/state` responde JSON.

Luego probar la imagen:

```bash
docker build -t finanom:local .
docker run --rm -p 8000:8000 -e PORT=8000 finanom:local
```

## 2. Crear Azure SQL barato

Antes de ejecutar estos comandos, confirma que quieres crear recursos con costo.

Opcion simple para pruebas: Azure SQL Database DTU Basic o Standard bajo. Opcion flexible para uso intermitente: vCore serverless con auto-pause. Revisa precio por region antes de confirmar.

Variables sugeridas:

```bash
RG=finanom-rg
LOCATION=eastus
SQL_SERVER=finanom-sql-$USER
SQL_DB=finanom
SQL_ADMIN=finanom_admin
```

Crear recursos:

```bash
az group create --name "$RG" --location "$LOCATION"

az sql server create \
  --resource-group "$RG" \
  --name "$SQL_SERVER" \
  --location "$LOCATION" \
  --admin-user "$SQL_ADMIN" \
  --admin-password '<PASSWORD_SEGURO>'

az sql db create \
  --resource-group "$RG" \
  --server "$SQL_SERVER" \
  --name "$SQL_DB" \
  --service-objective Basic
```

Para una base serverless, cambia el comando de base de datos segun el plan que confirmes en Azure Pricing Calculator.

## 3. Firewall de Azure SQL

Permitir tu IP local para migracion:

```bash
MY_IP=$(curl -s https://ifconfig.me)

az sql server firewall-rule create \
  --resource-group "$RG" \
  --server "$SQL_SERVER" \
  --name allow-local-ip \
  --start-ip-address "$MY_IP" \
  --end-ip-address "$MY_IP"
```

Si el contenedor corre en App Service o Container Apps y conecta a Azure SQL por red publica, habilita acceso desde servicios Azure o crea reglas especificas segun la red elegida. Para produccion, preferir Private Endpoint/VNet.

## 4. Migracion inicial

El Dockerfile conserva `model_final/data/transacciones_limpio.parquet` para migracion o fallback local. Define una URL de SQLAlchemy con ODBC Driver 18:

```bash
export DATABASE_URL='mssql+pyodbc://finanom_admin:<PASSWORD>@finanom-sql.database.windows.net:1433/finanom?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no'
```

Ejecuta la migracion inicial. Esta migracion usa `model_final/migrate_to_sql.py`: toma el ultimo año, cruza el parquet limpio con `reporte_revision.parquet`, aplica el estado de feedback y crea la tabla `transacciones` con columnas de scoring.

```bash
DATABASE_URL="$DATABASE_URL" scripts/migrate_azure_sql.sh --force
```

No ejecutes migraciones contra Azure sin confirmar que el destino, credenciales y costo son correctos.

## 5. Crear ACR y subir imagen

Antes de crear ACR, confirma que quieres crear recursos con costo.

```bash
ACR=finanomacr$USER
IMAGE=finanom
TAG=$(date +%Y%m%d%H%M)

az acr create \
  --resource-group "$RG" \
  --name "$ACR" \
  --sku Basic

az acr login --name "$ACR"

docker build -t "$ACR.azurecr.io/$IMAGE:$TAG" .
docker push "$ACR.azurecr.io/$IMAGE:$TAG"
```

Alternativa sin build local:

```bash
az acr build \
  --resource-group "$RG" \
  --registry "$ACR" \
  --image "$IMAGE:$TAG" .
```

## 6. Opcion A: App Service para contenedor

Antes de crear App Service Plan/Web App, confirma que quieres crear recursos con costo.

```bash
PLAN=finanom-plan
APP=finanom-app-$USER

az appservice plan create \
  --resource-group "$RG" \
  --name "$PLAN" \
  --is-linux \
  --sku B1

az webapp create \
  --resource-group "$RG" \
  --plan "$PLAN" \
  --name "$APP" \
  --deployment-container-image-name "$ACR.azurecr.io/$IMAGE:$TAG"
```

Permitir que App Service haga pull desde ACR con identidad administrada:

```bash
PRINCIPAL_ID=$(az webapp identity assign \
  --resource-group "$RG" \
  --name "$APP" \
  --query principalId \
  --output tsv)

ACR_ID=$(az acr show \
  --resource-group "$RG" \
  --name "$ACR" \
  --query id \
  --output tsv)

az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --scope "$ACR_ID" \
  --role AcrPull

az webapp config set \
  --resource-group "$RG" \
  --name "$APP" \
  --generic-configurations '{"acrUseManagedIdentityCreds": true}'
```

Configurar variables:

```bash
az webapp config appsettings set \
  --resource-group "$RG" \
  --name "$APP" \
  --settings \
    PORT=8000 \
    WEBSITES_PORT=8000 \
    DATABASE_URL="$DATABASE_URL"
```

`PORT` lo usa gunicorn dentro del contenedor. `WEBSITES_PORT` le indica a App Service a que puerto enrutar trafico HTTP.

## 7. Opcion B: Container Apps

Antes de crear Container Apps Environment/Container App, confirma que quieres crear recursos con costo.

```bash
ENV_NAME=finanom-env
APP=finanom-container-$USER

az containerapp env create \
  --resource-group "$RG" \
  --name "$ENV_NAME" \
  --location "$LOCATION"

az containerapp create \
  --resource-group "$RG" \
  --environment "$ENV_NAME" \
  --name "$APP" \
  --image "$ACR.azurecr.io/$IMAGE:$TAG" \
  --target-port 8000 \
  --ingress external \
  --env-vars PORT=8000 DATABASE_URL="$DATABASE_URL"
```

Si el ACR es privado, configura identidad administrada o credenciales de registry antes de crear la app.

## 8. Verificar HTTPS y salud

App Service:

```bash
az webapp show --resource-group "$RG" --name "$APP" --query defaultHostName -o tsv
curl -I "https://<defaultHostName>/"
curl "https://<defaultHostName>/api/transactions?filter=todas&page=1&page_size=5" | head
```

Container Apps:

```bash
az containerapp show --resource-group "$RG" --name "$APP" --query properties.configuration.ingress.fqdn -o tsv
curl -I "https://<fqdn>/"
curl "https://<fqdn>/api/transactions?filter=todas&page=1&page_size=5" | head
```

Verificar en navegador:

- La URL usa `https://`.
- El dashboard carga sin errores visibles.
- Los endpoints `/api/transactions` y `/api/state` responden.
- El boton `Simular transaccion nueva` muestra la fila resaltada y etiquetada.
- Logs no muestran errores de ODBC, `DATABASE_URL` o puerto.

## 9. Limpieza de recursos

Para detener costos de prueba:

```bash
az group delete --name "$RG"
```

Confirma antes de ejecutar este comando porque elimina todos los recursos del grupo.
