# Architecture Requirements API - Simplified

A simplified FastAPI application for storing and managing architecture requirements in Cosmos DB.

## Project Structure (2 files)

```
architectureapi/
├── app.py              # Main application (FastAPI routes, models, auth)
├── cosmos_service.py   # Cosmos DB service layer
└── README.md          # This file
```

## Setup

1. **Install dependencies:**
```bash
pip install fastapi uvicorn azure-cosmos pydantic python-jose
```

2. **Set environment variables:**
```bash
# Cosmos DB Configuration
$env:COSMOS_ENDPOINT = "your-cosmos-endpoint"
$env:COSMOS_KEY = "your-cosmos-key"
$env:COSMOS_DATABASE_NAME = "architecturedb"
$env:COSMOS_REQUIREMENTS_CONTAINER = "requirements"
$env:COSMOS_USERS_CONTAINER = "users"
$env:ALLOWED_ORIGINS = "*"

# Auth0 Configuration
$env:AUTH0_DOMAIN = "your-tenant.auth0.com"
$env:AUTH0_AUDIENCE = "https://your-api-identifier"
```

3. **Run the application:**
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health check with DB status
- `POST /api/requirements/create` - Create a new requirement
- `GET /api/requirements/{id}` - Get a specific requirement
- `GET /api/requirements` - List all requirements for tenant

## Authentication

The API uses **Auth0 JWT tokens** for authentication.

### Setup Auth0:
1. Create an Auth0 account and application
2. Create an API in Auth0 and note the **Audience** identifier
3. Add a custom claim for `tenant_id` in your Auth0 Action/Rule:
```javascript
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://yourapp.com';
  api.idToken.setCustomClaim(`${namespace}/tenant_id`, event.user.app_metadata.tenant_id);
  api.accessToken.setCustomClaim(`${namespace}/tenant_id`, event.user.app_metadata.tenant_id);
};
```

### Making Authenticated Requests:
```bash
# Get token from Auth0
TOKEN="your-auth0-jwt-token"

curl -X POST "http://localhost:8000/api/requirements/create" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

### Development Mode:
If `AUTH0_DOMAIN` and `AUTH0_AUDIENCE` are not set, the API falls back to header-based auth:
- `Authorization: Bearer <any-token>`
- `X-Tenant-Id: <your-tenant-id>`
- `X-Username: <your-username>`

**Note:** Disable header fallback in production.

## Example Request

```bash
curl -X POST "http://localhost:8000/api/requirements/create" \
  -H "Authorization: Bearer dev-token" \
  -H "X-Tenant-Id: tenant1" \
  -H "X-Username: user1" \
  -H "Content-Type: application/json" \
  -d '{
    "customerid": "customer_123",
    "sessionId": "session_456",
    "timestamp": "2026-01-05T10:00:00.000Z",
    "requirements": {
      "domain": "Healthcare",
      "coreUseCases": "EHR, Patient Portal",
      "features": "User authentication",
      "nfrs": {
        "users": "10,000 - 100,000",
        "requests": "1,000 - 10,000 requests/second",
        "latency": "< 100ms",
        "concurrent": "1,000 - 10,000"
      },
      "userRoles": "End users, Admins",
      "dataProfile": "User profiles",
      "security": "MFA, RBAC"
    },
    "metadata": {
      "completionTime": 420,
      "stepTransitions": []
    }
  }'
```
