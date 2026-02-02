"""
Architecture Requirements API - Simplified
FastAPI application for storing architecture requirements in Cosmos DB
"""
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import logging
import os
import json
from openai import OpenAI
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage
import hmac
import hashlib
import base64
import requests

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

# Azure Service Bus Configuration
SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
SERVICE_BUS_QUEUE_NAME = os.getenv("AZURE_SERVICE_BUS_QUEUE", "architecture-recommendations")

# Azure SignalR Service Configuration
SIGNALR_CONNECTION_STRING = os.getenv("AZURE_SIGNALR_NAME")
SIGNALR_HUB_NAME = "architectureHub"

# Azure OpenAI Configuration
OPENAI_ENDPOINT =os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT = "gpt-5-mini"
MAX_OVERVIEW_LENGTH = 4000  # Maximum characters for overview input
MAX_OUTPUT_TOKENS = 16000  # Maximum tokens for OpenAI response (increased for diagram JSON)


# Global cosmos service instance
cosmos_service: Optional[CosmosDBService] = None

# Global Service Bus client for architecture recommendation requests
service_bus_client: Optional[ServiceBusClient] = None

# SignalR service endpoint and key (parsed from connection string)
signalr_endpoint: Optional[str] = None
signalr_access_key: Optional[str] = None


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
    overview: Optional[str] = Field(None, description="Comprehensive application overview for architecture analysis")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary object containing overview and other data")
    requirements: Requirements
    metadata: Metadata
    username: Optional[str] = Field(None, description="Username")


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


class GetRequirementRequest(BaseModel):
    """Request model for getting a specific requirement"""
    requirementId: str = Field(..., description="Requirement ID")
    tenantId: str = Field(..., description="Tenant ID")


class ArchitectureRecommendationRequest(BaseModel):
    """Request model for getting architecture recommendations from OpenAI"""
    tenantId: str = Field(..., description="Tenant ID")
    sessionId: str = Field(..., description="Session ID")
    applicationName: str = Field(..., description="Application Name")
    overview: str = Field(..., description="Comprehensive application overview for architecture analysis", max_length=4000)


# Architecture recommendation response models
class Metrics(BaseModel):
    """All architecture metrics in one place"""
    latency: List[int] = Field(default=[], description="[min, max] in ms")
    throughput: List[int] = Field(default=[], description="[min, max] requests/sec")
    availability: float = Field(default=0, description="Availability percentage")
    autoscaling: str = Field(default="", description="Yes/No/Limited")
    cost: List[int] = Field(default=[], description="[min, max] monthly USD")
    scalability: int = Field(default=5, ge=1, le=10)
    reliability: int = Field(default=5, ge=1, le=10)
    maintainability: int = Field(default=5, ge=1, le=10)
    complexity: int = Field(default=5, ge=1, le=10)


class Infrastructure(BaseModel):
    """Infrastructure components"""
    compute: str = Field(default="", description="Compute services")
    database: str = Field(default="", description="Database services")
    cache: str = Field(default="", description="Caching services")
    messaging: str = Field(default="", description="Messaging/Queue services")
    storage: str = Field(default="", description="Storage services")
    apiGateway: str = Field(default="", description="API Gateway services")
    authentication: str = Field(default="", description="Authentication services")
    security: str = Field(default="", description="Security services and practices")
    networking: str = Field(default="", description="Networking and load balancing")
    monitoring: str = Field(default="", description="Monitoring tools")
    logging: str = Field(default="", description="Logging services")


class CICD(BaseModel):
    """CI/CD and DevOps components"""
    pipeline: str = Field(default="", description="CI/CD pipeline tools")
    containerization: str = Field(default="", description="Container and orchestration tools")
    testing: str = Field(default="", description="Testing frameworks")
    iac: str = Field(default="", description="Infrastructure as Code tools")


class DiagramShape(BaseModel):
    """Diagram shape - either a cloud service or an arrow"""
    tool: str = Field(..., description="Tool type: aws, azure, gcp, security, arrow")
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")
    id: str = Field(..., description="Unique identifier for the shape")
    service: Optional[str] = Field(None, description="Service name for cloud resources")
    arrowType: Optional[str] = Field(None, description="Arrow type: single, double, dashed")
    width: Optional[int] = Field(None, description="Width of the shape")
    height: Optional[int] = Field(None, description="Height of the shape")
    rotation: Optional[int] = Field(0, description="Rotation in degrees")
    points: Optional[List[int]] = Field(None, description="Points for arrows [x1, y1, x2, y2]")
    stroke: Optional[str] = Field(None, description="Stroke color for arrows")
    strokeWidth: Optional[int] = Field(None, description="Stroke width for arrows")
    dash: Optional[List[int]] = Field(None, description="Dash pattern for dashed arrows")


class ArchitectureDiagram(BaseModel):
    """Visual architecture diagram"""
    name: str = Field(..., description="Diagram name")
    description: str = Field(..., description="Diagram description")
    shapes: List[DiagramShape] = Field(default=[], description="List of shapes in the diagram")


class TechnologyStack(BaseModel):
    """Technology stack components"""
    languages: str = Field(default="", description="Programming languages")
    frameworks: str = Field(default="", description="Frameworks")
    runtime: str = Field(default="", description="Runtime environment")
    cloudProvider: str = Field(default="", description="Cloud provider (AWS, Azure, GCP)")
    infra: Infrastructure = Field(default_factory=Infrastructure)
    cicd: CICD = Field(default_factory=CICD)


class Architecture(BaseModel):
    """Architecture recommendation"""
    id: str
    icon: str = ""
    name: str
    description: str
    ranking: int = Field(default=1, ge=1, le=10)
    shortPros: str = ""
    shortCons: str = ""
    recommendationReason: str = ""
    whyChoose: str = ""
    best: bool = False
    metrics: Metrics = Field(default_factory=Metrics)
    technologyStack: TechnologyStack = Field(default_factory=TechnologyStack)
    bestFor: List[str] = []
    avoidWhen: List[str] = []


class ArchitectureRecommendationResponse(BaseModel):
    """Response model for architecture recommendations"""
    success: bool
    message: str
    tenantId: str
    sessionId: str
    architectures: Optional[List[Architecture]] = None
    error: Optional[str] = None


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
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-Id"),
    x_username: Optional[str] = Header(None, alias="X-Username")
) -> AuthenticatedUser:
    """
    Simple header-based authentication using X-Tenant-Id and X-Username headers
    """
    try:
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
    global cosmos_service, service_bus_client, signalr_endpoint, signalr_access_key
    
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
    
    # Initialize SignalR configuration
    if SIGNALR_CONNECTION_STRING:
        try:
            # Parse SignalR connection string
            parts = SIGNALR_CONNECTION_STRING.split(';')
            for part in parts:
                if part.startswith('Endpoint='):
                    signalr_endpoint = part.split('=', 1)[1]
                elif part.startswith('AccessKey='):
                    signalr_access_key = part.split('=', 1)[1]
            
            if signalr_endpoint and signalr_access_key:
                logger.info(f"SignalR Service initialized: {signalr_endpoint}")
            else:
                logger.warning("SignalR connection string is incomplete")
        except Exception as e:
            logger.error(f"Failed to parse SignalR connection string: {str(e)}")
            signalr_endpoint = None
            signalr_access_key = None
    else:
        logger.warning("AZURE_SIGNALR_CONNECTION_STRING not configured - real-time notifications will not work")
    
    # Initialize Service Bus client
    if SERVICE_BUS_CONNECTION_STRING:
        try:
            service_bus_client = ServiceBusClient.from_connection_string(
                conn_str=SERVICE_BUS_CONNECTION_STRING,
                logging_enable=True
            )
            logger.info(f"Service Bus client initialized for queue: {SERVICE_BUS_QUEUE_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Service Bus client: {str(e)}", exc_info=True)
            logger.warning("Service Bus is not available - recommendation queueing will not work")
            service_bus_client = None
    else:
        logger.warning("SERVICE_BUS_CONNECTION_STRING not configured - recommendation queueing will not work")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Architecture Requirements API...")
    if cosmos_service:
        await cosmos_service.close()
    if service_bus_client:
        await service_bus_client.close()
        logger.info("Service Bus client closed")
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
    
    Security: Uses header-based authentication with X-Tenant-Id and X-Username
    """
    try:
        logger.info(
            f"Received requirement creation request for tenant: {requirement.tenantId}, "
            f"session: {requirement.sessionId}, user: {current_user.username}"
        )
        
        # Check if a requirement already exists for this session
        existing_requirement = await cosmos.get_requirement_by_session(
            session_id=requirement.sessionId,
            tenant_id=requirement.tenantId
        )
        
        # Check if application name already exists for this tenant
        existing_app = await cosmos.get_requirement_by_application_name(
            application_name=requirement.applicationName,
            tenant_id=requirement.tenantId
        )
        
        # Prevent duplicate app creation if incoming request hasn't progressed beyond Step 1
        if existing_app and not existing_requirement:
            # Get step transitions from incoming request
            incoming_step_transitions = requirement.metadata.stepTransitions if requirement.metadata else []
            completed_steps = {transition.step for transition in incoming_step_transitions}
            
            # Check if user has gone beyond Step 1 (applicationName)
            # Allow overwrite only if they have at least one step beyond applicationName
            steps_beyond_app_name = completed_steps - {"applicationName"}
            
            if not steps_beyond_app_name:
                # User is still at Step 1 - don't allow duplicate
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "DUPLICATE_APPLICATION_NAME",
                        "message": f"An application with the name '{requirement.applicationName}' already exists for this tenant. Duplicate applications are not allowed. Please use a different application name."
                    }
                )
        
        # Determine which requirement to update (prioritize existing_app for overwrite behavior)
        requirement_to_update = existing_app or existing_requirement
        
        # Add user info to requirement
        requirement_data = requirement.dict()
        requirement_data["username"] = current_user.username
        requirement_data["userId"] = current_user.user_id
        requirement_data["userEmail"] = current_user.email
        
        # Extract overview from summary if not provided directly
        if not requirement_data.get("overview"):
            if requirement_data.get("summary") and requirement_data["summary"].get("overview"):
                requirement_data["overview"] = requirement_data["summary"]["overview"]
            else:
                requirement_data["overview"] = generate_overview_from_requirements(requirement_data)
        
        if requirement_to_update:
            # Update existing requirement (overwrite if same app name)
            logger.info(f"Updating existing requirement: {requirement_to_update['id']}")
            saved_requirement = await cosmos.update_requirement(
                requirement_id=requirement_to_update["id"],
                tenant_id=requirement.tenantId,
                updates=requirement_data
            )
            message = "Requirement updated successfully"
        else:
            # Create new requirement
            logger.info("Creating new requirement")
            saved_requirement = await cosmos.create_requirement(requirement_data)
            message = "Requirement created successfully"
        
        # Add status field
        saved_requirement["status"] = calculate_requirement_status(saved_requirement)
        
        logger.info(f"Requirement saved successfully: {saved_requirement['id']}")
        
        return RequirementResponse(
            success=True,
            message=message,
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


@app.post("/api/requirements/get")
async def get_requirement(
    request: GetRequirementRequest,
    username: Optional[str] = Header(None, alias="X-Username"),
    cosmos: CosmosDBService = Depends(get_cosmos_service)
):
    """Get a specific requirement by ID"""
    try:
        if not username:
            raise HTTPException(
                status_code=400,
                detail="X-Username header is required"
            )
        
        requirement = await cosmos.get_requirement(request.requirementId, request.tenantId)
        
        if not requirement:
            raise HTTPException(
                status_code=404,
                detail=f"Requirement {request.requirementId} not found"
            )
        
        # Add overview if not present
        if not requirement.get("overview"):
            requirement["overview"] = generate_overview_from_requirements(requirement)
        
        # Add status field
        requirement["status"] = calculate_requirement_status(requirement)
        
        return RequirementResponse(
            success=True,
            message="Requirement retrieved successfully",
            requirementId=request.requirementId,
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


def calculate_requirement_status(req: Dict[str, Any]) -> str:
    """Calculate requirement status based on stepTransitions"""
    # Define all 8 required steps (matching actual step names in transitions)
    # Note: applicationName is not included as it's a setup step, not part of the workflow
    required_steps = {
        "domain",
        "coreUseCases",
        "features",
        "data",
        "nfrs",
        "userRoles",
        "security",
        "review"
    }
    
    # Get step transitions from metadata
    metadata = req.get("metadata", {})
    step_transitions = metadata.get("stepTransitions", [])
    
    # Extract completed steps
    completed_steps = {transition.get("step") for transition in step_transitions if transition.get("step")}
    
    # Check if all 8 steps are completed
    if required_steps.issubset(completed_steps):
        return "Complete"
    else:
        return "Edit"


def generate_overview_from_requirements(req: Dict[str, Any]) -> str:
    """Generate an overview string from requirements data"""
    parts = []
    
    app_name = req.get("applicationName", "Application")
    if app_name:
        parts.append(f"{app_name} is")
    
    # Domain
    domain = req.get("requirements", {}).get("domain", [])
    if domain:
        domain_str = domain[0] if isinstance(domain, list) and domain else domain
        parts.append(f"a {domain_str} application")
    
    # Core use cases
    use_cases = req.get("requirements", {}).get("coreUseCases", [])
    if use_cases:
        use_case_str = use_cases[0] if isinstance(use_cases, list) and use_cases else use_cases
        parts.append(f"designed for {use_case_str}")
    
    # Features
    features = req.get("requirements", {}).get("features", "")
    if features:
        parts.append(f"with features including {features}")
    
    # NFRs
    nfrs = req.get("requirements", {}).get("nfrs", {})
    nfr_parts = []
    if nfrs.get("users"):
        nfr_parts.append(f"{nfrs['users']} users")
    if nfrs.get("requests"):
        nfr_parts.append(f"{nfrs['requests']} requests")
    if nfrs.get("concurrent"):
        nfr_parts.append(f"{nfrs['concurrent']} concurrent users")
    if nfr_parts:
        parts.append(f"supporting {', '.join(nfr_parts)}")
    
    # User roles
    roles = req.get("requirements", {}).get("userRoles", [])
    if roles:
        role_str = roles[0] if isinstance(roles, list) and roles else roles
        parts.append(f"for {role_str}")
    
    # Security
    security = req.get("requirements", {}).get("security", [])
    if security and isinstance(security, list) and security:
        # Extract just the first few security items
        sec_items = [s.strip().replace("  • ", "") for s in security[:3]]
        parts.append(f"requiring {', '.join(sec_items)}")
    
    return " ".join(parts) if parts else "No overview available"


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
        
        # Add overview and status to each requirement if not present
        for req in requirements:
            if not req.get("overview"):
                req["overview"] = generate_overview_from_requirements(req)
            req["status"] = calculate_requirement_status(req)
        
        return requirements
        
    except Exception as e:
        logger.error(f"Error listing requirements: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list requirements: {str(e)}"
        )


@app.post("/api/architecture/recommendations", response_model=ArchitectureRecommendationResponse)
async def get_architecture_recommendations(
    request: ArchitectureRecommendationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Queue architecture recommendation request for asynchronous processing
    
    This endpoint accepts the application overview and queues it for processing.
    Returns immediately with a success response.
    """
    try:
        logger.info(
            f"Received architecture recommendation request for tenant: {request.tenantId}, "
            f"session: {request.sessionId}, user: {current_user.username}"
        )
        logger.info(f"Overview length: {len(request.overview)} characters")
        
        # Validate Service Bus client is available
        if not service_bus_client:
            logger.error("Service Bus client not initialized")
            raise HTTPException(
                status_code=500,
                detail="Service Bus is not configured"
            )
        
        # Send message to Service Bus with proper error handling
        try:
            sender = service_bus_client.get_queue_sender(queue_name=SERVICE_BUS_QUEUE_NAME)
            async with sender:
                message = ServiceBusMessage(
                    body=json.dumps(request.model_dump()),
                    content_type="application/json"
                )
                await sender.send_messages(message)
                logger.info(
                    f"Message sent to Service Bus queue '{SERVICE_BUS_QUEUE_NAME}' for tenant: {request.tenantId}, "
                    f"session: {request.sessionId}"
                )
        except Exception as sb_error:
            logger.error(f"Service Bus error: {str(sb_error)}", exc_info=True)
            # Return error response with details
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "SERVICE_BUS_UNAVAILABLE",
                    "message": f"Failed to queue request to Service Bus. The queue '{SERVICE_BUS_QUEUE_NAME}' may not exist or connection is misconfigured.",
                    "details": str(sb_error)
                }
            )
        
        # Return success response immediately
        return ArchitectureRecommendationResponse(
            success=True,
            message="Architecture recommendation request queued successfully",
            tenantId=request.tenantId,
            sessionId=request.sessionId,
            architectures=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queueing architecture recommendation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue architecture recommendation request: {str(e)}"
        )


# ==================== SIGNALR ENDPOINTS ====================

class SignalRConnectionInfo(BaseModel):
    """SignalR connection info response"""
    url: str
    accessToken: str


class SignalRNegotiateRequest(BaseModel):
    """Request model for SignalR negotiate"""
    userId: str = Field(..., description="User ID for SignalR connection")


def generate_signalr_token(hub_name: str, user_id: str) -> str:
    """Generate SignalR access token using HMAC"""
    if not signalr_endpoint or not signalr_access_key:
        raise ValueError("SignalR not configured")
    
    # Remove https:// and trailing slash
    endpoint = signalr_endpoint.replace('https://', '').replace('http://', '').rstrip('/')
    
    # Construct audience
    audience = f"{signalr_endpoint}/client/?hub={hub_name}"
    
    # Token expiration (1 hour from now)
    import time
    exp = int(time.time()) + 3600
    
    # Construct payload
    payload = f"{audience}\n{exp}"
    
    # Sign with HMAC-SHA256
    key_bytes = base64.b64decode(signalr_access_key)
    signature = hmac.new(key_bytes, payload.encode('utf-8'), hashlib.sha256).digest()
    encoded_signature = base64.b64encode(signature).decode('utf-8')
    
    # Construct token
    token = f"{audience}\n{exp}\n{encoded_signature}"
    return base64.b64encode(token.encode('utf-8')).decode('utf-8')


@app.post("/api/signalr/negotiate", response_model=SignalRConnectionInfo)
async def signalr_negotiate(
    request: SignalRNegotiateRequest
):
    """SignalR negotiate endpoint for client connections"""
    try:
        if not signalr_endpoint or not signalr_access_key:
            raise HTTPException(
                status_code=503,
                detail="SignalR Service is not configured"
            )
        
        # Generate access token for the user
        access_token = generate_signalr_token(SIGNALR_HUB_NAME, request.userId)
        
        # Construct client connection URL
        connection_url = f"{signalr_endpoint}/client/?hub={SIGNALR_HUB_NAME}"
        
        logger.info(f"SignalR connection negotiated for user: {request.userId}")
        
        return SignalRConnectionInfo(
            url=connection_url,
            accessToken=access_token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error negotiating SignalR connection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to negotiate SignalR connection: {str(e)}"
        )


class SignalRMessage(BaseModel):
    """SignalR message to send"""
    target: str = Field(..., description="Target method name")
    arguments: List[Any] = Field(..., description="Arguments to pass to the method")


class SignalRSendRequest(BaseModel):
    """Request to send a message via SignalR"""
    userId: Optional[str] = Field(None, description="Send to specific user")
    groupName: Optional[str] = Field(None, description="Send to specific group")
    message: SignalRMessage


async def send_signalr_message(
    hub_name: str,
    target: str,
    arguments: List[Any],
    user_id: Optional[str] = None,
    group_name: Optional[str] = None
) -> bool:
    """Send a message via SignalR to users or groups"""
    try:
        if not signalr_endpoint or not signalr_access_key:
            logger.warning("SignalR not configured, skipping message send")
            return False
        
        # Construct API URL
        if user_id:
            url = f"{signalr_endpoint}/api/v1/hubs/{hub_name}/users/{user_id}"
        elif group_name:
            url = f"{signalr_endpoint}/api/v1/hubs/{hub_name}/groups/{group_name}"
        else:
            url = f"{signalr_endpoint}/api/v1/hubs/{hub_name}"
        
        # Generate server token
        import time
        exp = int(time.time()) + 3600
        payload = f"{signalr_endpoint}/api/v1/hubs/{hub_name}\n{exp}"
        key_bytes = base64.b64decode(signalr_access_key)
        signature = hmac.new(key_bytes, payload.encode('utf-8'), hashlib.sha256).digest()
        encoded_signature = base64.b64encode(signature).decode('utf-8')
        token_payload = f"{signalr_endpoint}/api/v1/hubs/{hub_name}\n{exp}\n{encoded_signature}"
        bearer_token = base64.b64encode(token_payload.encode('utf-8')).decode('utf-8')
        
        # Send message
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "target": target,
            "arguments": arguments
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        
        if response.status_code == 202:
            logger.info(f"SignalR message sent successfully to {user_id or group_name or 'all'}")
            return True
        else:
            logger.error(f"Failed to send SignalR message: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error sending SignalR message: {str(e)}", exc_info=True)
        return False


@app.post("/api/signalr/send")
async def signalr_send(
    request: SignalRSendRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Send a message via SignalR (for testing or manual triggers)"""
    try:
        success = await send_signalr_message(
            hub_name=SIGNALR_HUB_NAME,
            target=request.message.target,
            arguments=request.message.arguments,
            user_id=request.userId,
            group_name=request.groupName
        )
        
        if success:
            return {"success": True, "message": "Message sent successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to send SignalR message"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in SignalR send endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
