"""
Architecture Requirements API - Simplified
FastAPI application for storing architecture requirements in Cosmos DB
"""
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import logging
import os
import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from cosmos_service import CosmosDBService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://architecture.documents.azure.com:443/")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "architecture")
COSMOS_REQUIREMENTS_CONTAINER = os.getenv("COSMOS_REQUIREMENTS_CONTAINER", "requirements")
COSMOS_USERS_CONTAINER = os.getenv("COSMOS_USERS_CONTAINER", "users")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")  # e.g., "your-tenant.auth0.com"
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")  # e.g., "https://your-api-identifier"
AUTH0_ALGORITHMS = ["RS256"]

# Global cosmos service instance
cosmos_service: Optional[CosmosDBService] = None
security = HTTPBearer()


# ==================== DATA MODELS ====================

class NFRs(BaseModel):
    """Non-functional requirements"""
    users: Optional[str] = ""
    requests: Optional[str] = ""
    latency: Optional[str] = ""
    concurrent: Optional[str] = ""


class Requirements(BaseModel):
    """Application requirements"""
    domain: Any  # Can be string or array
    coreUseCases: Any  # Can be string or array
    features: Any  # Can be string or array
    nfrs: NFRs
    userRoles: Any  # Can be string or array
    dataProfile: Any  # Can be string or array
    security: Any  # Can be string or array
    performance: Optional[Any] = None
    reliability: Optional[Any] = None
    integrations: Optional[Any] = None


class StepTransition(BaseModel):
    """Step transition metadata"""
    step: str
    timestamp: str
    timeSpent: int


class Metadata(BaseModel):
    """Requirement metadata"""
    completionTime: int
    stepTransitions: List[StepTransition]


class RequirementRequest(BaseModel):
    """Request model for creating a requirement"""
    tenantId: str = Field(..., description="Tenant ID")
    sessionId: str = Field(..., description="Session ID")
    applicationName: str = Field(..., description="Application Name")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    requirements: Requirements
    metadata: Metadata


class RequirementResponse(BaseModel):
    """Response model for requirement operations"""
    success: bool
    message: str
    requirementId: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ListRequirementsRequest(BaseModel):
    """Request model for listing requirements"""
    tenantId: str = Field(..., description="Tenant ID")
    limit: Optional[int] = Field(100, description="Maximum number of requirements to return")


class AuthenticatedUser:
    """Represents an authenticated user with tenant context"""
    def __init__(self, tenant_id: str, user_id: str, username: str, email: str):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.username = username
        self.email = email


# ==================== DEPENDENCIES & AUTH ====================

def get_cosmos_service() -> CosmosDBService:
    """Dependency injection for Cosmos DB service"""
    if cosmos_service is None:
        raise RuntimeError("Cosmos DB service not initialized")
    return cosmos_service


async def get_current_user(
    # credentials: HTTPAuthorizationCredentials = Depends(security),  # Commented out for development
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-Id"),
    x_username: Optional[str] = Header(None, alias="X-Username")
) -> AuthenticatedUser:
    """
    Simple header-based authentication for development
    JWT validation is commented out - use X-Tenant-Id and X-Username headers
    """
    try:
        # Simple header-based auth (for development)
        # Use default values if headers are not provided
        tenant_id = x_tenant_id or "default_tenant"
        username = x_username or "anonymous"
        
        logger.info(f"Using header-based auth - tenant: {tenant_id}, user: {username}")
        return AuthenticatedUser(
            tenant_id=tenant_id,
            user_id=username,
            username=username,
            email=f"{username}@example.com"
        )
        
        # ===== JWT VALIDATION (COMMENTED OUT FOR DEVELOPMENT) =====
        # token = credentials.credentials
        # 
        # # Development fallback (disable in production)
        # if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
        #     raise HTTPException(
        #         status_code=500,
        #         detail="Auth0 not configured. Set AUTH0_DOMAIN and AUTH0_AUDIENCE environment variables."
        #     )
        # 
        # # Get Auth0 public key
        # jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
        # jwks_client = PyJWKClient(jwks_url)
        # signing_key = jwks_client.get_signing_key_from_jwt(token)
        # 
        # # Verify and decode JWT
        # payload = jwt.decode(
        #     token,
        #     signing_key.key,
        #     algorithms=AUTH0_ALGORITHMS,
        #     audience=AUTH0_AUDIENCE,
        #     issuer=f"https://{AUTH0_DOMAIN}/"
        # )
        # 
        # # Extract user information from JWT claims
        # user_id = payload.get("sub")
        # email = payload.get("email")
        # username = payload.get("preferred_username") or payload.get("nickname") or email
        # 
        # # Extract tenant_id from custom claim
        # tenant_id = (
        #     payload.get("https://yourapp.com/tenant_id") or 
        #     payload.get("tenant_id") or
        #     payload.get("org_id") or
        #     x_tenant_id
        # )
        # 
        # if not user_id:
        #     raise HTTPException(
        #         status_code=401,
        #         detail="Invalid token: missing user ID (sub claim)"
        #     )
        # 
        # if not tenant_id:
        #     raise HTTPException(
        #         status_code=401,
        #         detail="Invalid token: missing tenant_id claim."
        #     )
        # 
        # return AuthenticatedUser(
        #     tenant_id=tenant_id,
        #     user_id=user_id,
        #     username=username or "unknown",
        #     email=email or ""
        # )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )


# ==================== APPLICATION SETUP ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global cosmos_service
    
    # Startup
    logger.info("Starting up Architecture Requirements API...")
    cosmos_service = CosmosDBService(
        endpoint=COSMOS_ENDPOINT,
        key=COSMOS_KEY,
        database_name=COSMOS_DATABASE_NAME,
        requirements_container=COSMOS_REQUIREMENTS_CONTAINER,
        users_container=COSMOS_USERS_CONTAINER
    )
    await cosmos_service.validate_connection()
    logger.info("Cosmos DB connection established and validated")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Architecture Requirements API...")
    if cosmos_service:
        await cosmos_service.close()
    logger.info("Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Architecture Requirements API",
    description="API for storing and managing architecture requirements in Cosmos DB",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ROUTES ====================

@app.post("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Architecture Requirements API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/health")
async def health_check(cosmos: CosmosDBService = Depends(get_cosmos_service)):
    """Detailed health check"""
    cosmos_health = await cosmos.health_check()
    return {
        "status": "healthy" if cosmos_health else "unhealthy",
        "cosmos_db": "connected" if cosmos_health else "disconnected"
    }


@app.post("/api/requirements/create", response_model=RequirementResponse)
async def create_requirement(
    requirement: RequirementRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    cosmos: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Create a new architecture requirement in Cosmos DB
    
    Security: Requires valid JWT token with tenant and user claims
    """
    try:
        logger.info(
            f"Received requirement creation request for tenant: {requirement.tenantId}, "
            f"user: {current_user.username}"
        )
        
        # Add user info to requirement
        requirement_data = requirement.dict()
        requirement_data["username"] = current_user.username
        requirement_data["userId"] = current_user.user_id
        requirement_data["userEmail"] = current_user.email
        
        # Save requirement to Cosmos DB
        saved_requirement = await cosmos.create_requirement(requirement_data)
        
        logger.info(f"Requirement saved successfully: {saved_requirement['id']}")
        
        return RequirementResponse(
            success=True,
            message="Requirement created successfully",
            requirementId=saved_requirement["id"],
            data=saved_requirement
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating requirement: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create requirement: {str(e)}"
        )


@app.post("/api/requirements/{requirement_id}")
async def get_requirement(
    requirement_id: str,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-Id"),
    username: Optional[str] = Header(None, alias="X-Username"),
    cosmos: CosmosDBService = Depends(get_cosmos_service)
):
    """Get a specific requirement by ID"""
    try:
        if not tenant_id or not username:
            raise HTTPException(
                status_code=400,
                detail="X-Tenant-Id and X-Username headers are required"
            )
        
        requirement = await cosmos.get_requirement(requirement_id, tenant_id)
        
        if not requirement:
            raise HTTPException(
                status_code=404,
                detail=f"Requirement {requirement_id} not found"
            )
        
        return RequirementResponse(
            success=True,
            message="Requirement retrieved successfully",
            requirementId=requirement_id,
            data=requirement
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving requirement: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve requirement: {str(e)}"
        )


@app.post("/api/requirements")
async def list_requirements(
    request: ListRequirementsRequest,
    cosmos: CosmosDBService = Depends(get_cosmos_service)
):
    """List all requirements for the specified tenant"""
    try:
        requirements = await cosmos.list_requirements(
            tenant_id=request.tenantId,
            limit=request.limit
        )
        
        return requirements
        
    except Exception as e:
        logger.error(f"Error listing requirements: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list requirements: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
