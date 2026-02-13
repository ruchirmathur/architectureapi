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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from openai import OpenAI
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage

from cosmos_service import CosmosDBService

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Suppress Azure SDK verbose logging
logging.getLogger('azure').setLevel(logging.ERROR)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)
logging.getLogger('cosmos_service').setLevel(logging.ERROR)

# Configuration from environment variables
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://architecture.documents.azure.com:443/")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "architecture")
COSMOS_REQUIREMENTS_CONTAINER = os.getenv("COSMOS_REQUIREMENTS_CONTAINER", "requirements")
COSMOS_USERS_CONTAINER = os.getenv("COSMOS_USERS_CONTAINER", "users")
COSMOS_RECOMMENDATIONS_CONTAINER = os.getenv("COSMOS_RECOMMENDATIONS_CONTAINER", "recommendations")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Azure Service Bus Configuration
SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
SERVICE_BUS_QUEUE_NAME = os.getenv("AZURE_SERVICE_BUS_QUEUE", "architecture-recommendations")

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
    userId: Optional[str] = Field(None, description="User ID")
    applicationType: Optional[str] = Field(None, description="Application type (e.g., PWA, SPA, Mobile)")
    domain: Optional[List[str]] = Field(default=[], description="Domain areas (e.g., Healthcare)")
    industry: Optional[List[str]] = Field(default=[], description="Industry sectors")
    coreUseCases: Optional[List[str]] = Field(default=[], description="Core use cases")
    useCases: Optional[List[str]] = Field(default=[], description="Use cases")
    features: Optional[str] = Field(default="", description="Comma-separated features string")
    featuresList: Optional[List[str]] = Field(default=[], description="Features as a list")
    nfrs: Optional[NFRs] = Field(default_factory=NFRs, description="Non-functional requirements")
    performance: Optional[str] = Field(default="", description="Performance requirements")
    reliability: Optional[str] = Field(default="", description="Reliability requirements")
    dataProfile: Optional[str] = Field(default="", description="Data profile description")
    dataTypes: Optional[List[str]] = Field(default=[], description="Types of data managed")
    userRoles: Optional[List[str]] = Field(default=[], description="User roles")
    userTypes: Optional[List[str]] = Field(default=[], description="User types")
    security: Optional[List[str]] = Field(default=[], description="Security requirements")
    securityRequirements: Optional[List[str]] = Field(default=[], description="Security requirements (clean)")
    integrations: Optional[str] = Field(default="", description="Integrations description")
    integrationsList: Optional[List[str]] = Field(default=[], description="Integrations as a list")


# Architecture recommendation response models
class Metrics(BaseModel):
    """All architecture metrics in one place"""
    latency: List[float] = Field(default=[], description="[min, max] in ms")
    throughput: List[float] = Field(default=[], description="[min, max] requests/sec")
    availability: float = Field(default=0, description="Availability percentage")
    autoscaling: str = Field(default="", description="Yes/No/Limited")
    cost: List[float] = Field(default=[], description="[min, max] monthly USD")
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
    features: Optional[List[str]] = None
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
    global cosmos_service, service_bus_client
    
    # Startup
    # Validate required environment variables
    if not COSMOS_KEY:
        logger.error("COSMOS_KEY environment variable is not set")
        raise ValueError("COSMOS_KEY is required but not configured")
    
    cosmos_service = CosmosDBService(
        endpoint=COSMOS_ENDPOINT,
        key=COSMOS_KEY,
        database_name=COSMOS_DATABASE_NAME,
        requirements_container=COSMOS_REQUIREMENTS_CONTAINER,
        users_container=COSMOS_USERS_CONTAINER,
        recommendations_container=COSMOS_RECOMMENDATIONS_CONTAINER
    )
    
    try:
        await cosmos_service.validate_connection()
    except Exception as e:
        logger.error(f"Failed to connect to Cosmos DB: {str(e)}", exc_info=True)
        raise
    
    # Initialize Service Bus client
    if SERVICE_BUS_CONNECTION_STRING:
        try:
            service_bus_client = ServiceBusClient.from_connection_string(
                conn_str=SERVICE_BUS_CONNECTION_STRING,
                logging_enable=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize Service Bus client: {str(e)}")
            service_bus_client = None
    
    yield
    
    # Shutdown
    if cosmos_service:
        await cosmos_service.close()
    if service_bus_client:
        await service_bus_client.close()


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
            saved_requirement = await cosmos.update_requirement(
                requirement_id=requirement_to_update["id"],
                tenant_id=requirement.tenantId,
                updates=requirement_data
            )
            message = "Requirement updated successfully"
        else:
            # Create new requirement
            saved_requirement = await cosmos.create_requirement(requirement_data)
            message = "Requirement created successfully"
        
        # Add status field
        saved_requirement["status"] = calculate_requirement_status(saved_requirement)
        
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
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Get architecture recommendations - checks for existing recommendations first,
    otherwise queues request for asynchronous processing
    
    This endpoint first checks if a recommendation already exists for the user and application.
    If found, returns it immediately. Otherwise, queues the request for processing.
    """
    try:
        # Use userId from request if provided, otherwise use authenticated user
        user_id = request.userId or current_user.user_id
        
        # Check if a recommendation already exists for this application
        existing_recommendation = await db.get_recommendation(
            application_name=request.applicationName,
            tenant_id=request.tenantId
        )
        
        if existing_recommendation:
            # Recommendation found - return it directly
            logger.info(f"Found existing recommendation for user {user_id}, app {request.applicationName}")
            
            # Extract architectures from the stored recommendation
            # The document stores architectures in 'architectureRecommendations' field
            architectures_data = existing_recommendation.get("architectureRecommendations", [])
            architectures = [Architecture(**arch) for arch in architectures_data] if architectures_data else None
            
            # Extract features from requestMetadata
            request_metadata = existing_recommendation.get("requestMetadata", {})
            features = request_metadata.get("features", [])
            # Ensure features is a list
            if isinstance(features, str):
                features = [f.strip() for f in features.split(',') if f.strip()]
            
            return ArchitectureRecommendationResponse(
                success=True,
                message="Retrieved existing architecture recommendation",
                tenantId=request.tenantId,
                sessionId=request.sessionId,
                architectures=architectures,
                features=features
            )
        
        # No existing recommendation found - queue for processing
        logger.info(f"No existing recommendation found for user {user_id}, app {request.applicationName}. Queuing request.")
        
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
                # Create message payload with all fields + resolved userId
                message_data = request.model_dump()
                message_data["userId"] = user_id
                
                # Convert NFRs from nested object to dict if present
                if message_data.get("nfrs") and hasattr(message_data["nfrs"], "__dict__"):
                    message_data["nfrs"] = dict(message_data["nfrs"])
                
                logger.info(f"Service Bus message payload keys: {list(message_data.keys())}")
                logger.info(f"Service Bus message payload: {json.dumps(message_data, indent=2)}")
                
                message = ServiceBusMessage(
                    body=json.dumps(message_data),
                    content_type="application/json"
                )
                await sender.send_messages(message)
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
        logger.error(f"Error processing architecture recommendation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process architecture recommendation request: {str(e)}"
        )


@app.post("/api/architecture/recommendations/direct", response_model=ArchitectureRecommendationResponse)
async def get_architecture_recommendations_direct(
    request: ArchitectureRecommendationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Get architecture recommendations directly from OpenAI without queuing.
    
    This endpoint constructs a detailed prompt for OpenAI to act as a professional architect
    and returns architecture recommendations synchronously.
    """
    try:
        # Use userId from request if provided, otherwise use authenticated user
        user_id = request.userId or current_user.user_id
        
        logger.info(f"Processing direct architecture recommendation for user {user_id}, app {request.applicationName}")
        
        # Check if OpenAI is configured
        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API is not configured"
            )
        
        # Construct detailed architecture prompt
        prompt = construct_architecture_prompt(request)
        
        # Call OpenAI API
        try:
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_ENDPOINT
            )
            
            logger.info("Calling OpenAI API for architecture recommendations...")
            
            response = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert software architect with deep knowledge of cloud-native architectures, microservices, 
serverless patterns, and modern application design. Your role is to analyze application requirements and provide detailed, 
practical architecture recommendations tailored to the specific needs, constraints, and scale of each project."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"}
            )
            
            # Parse OpenAI response
            architectures_json = json.loads(response.choices[0].message.content)
            logger.info(f"Received OpenAI response: {json.dumps(architectures_json, indent=2)[:500]}...")
            
            # Transform response into Architecture objects
            architectures = parse_openai_architectures(architectures_json)
            
            # Store recommendation in database for future retrieval
            recommendation_data = {
                "tenantId": request.tenantId,
                "sessionId": request.sessionId,
                "applicationName": request.applicationName,
                "userId": user_id,
                "overview": request.overview,
                "architectureRecommendations": [arch.dict() for arch in architectures],
                "requestMetadata": {
                    "applicationType": request.applicationType,
                    "domain": request.domain,
                    "industry": request.industry,
                    "coreUseCases": request.coreUseCases,
                    "features": request.featuresList or request.features,
                    "nfrs": request.nfrs.dict() if request.nfrs else {},
                    "userRoles": request.userRoles,
                    "security": request.securityRequirements or request.security,
                    "integrations": request.integrationsList or request.integrations
                }
            }
            
            await db.create_recommendation(recommendation_data)
            logger.info(f"Stored recommendation for app {request.applicationName}")
            
            # Extract features list for response
            features = request.featuresList if request.featuresList else (
                [f.strip() for f in request.features.split(',') if f.strip()] if request.features else []
            )
            
            return ArchitectureRecommendationResponse(
                success=True,
                message="Architecture recommendations generated successfully",
                tenantId=request.tenantId,
                sessionId=request.sessionId,
                architectures=architectures,
                features=features
            )
            
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse OpenAI response as JSON: {str(je)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse OpenAI response"
            )
        except Exception as openai_error:
            logger.error(f"OpenAI API error: {str(openai_error)}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API error: {str(openai_error)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing direct architecture recommendation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process architecture recommendation: {str(e)}"
        )


def construct_architecture_prompt(request: ArchitectureRecommendationRequest) -> str:
    """
    Construct a detailed prompt for OpenAI to generate architecture recommendations.
    
    This function creates a comprehensive prompt that includes all application requirements,
    constraints, and expectations for the architecture recommendation.
    """
    # Extract and format requirements
    features = request.featuresList if request.featuresList else (request.features.split(',') if request.features else [])
    features_str = ", ".join([f.strip() for f in features if f.strip()]) if features else "Not specified"
    
    # Format features as bullet points for better emphasis
    features_bullets = "\n".join([f"  • {f.strip()}" for f in features if f.strip()]) if features else "  • Not specified"
    
    integrations = request.integrationsList if request.integrationsList else (request.integrations.split(',') if request.integrations else [])
    integrations_str = ", ".join([i.strip() for i in integrations if i.strip()]) if integrations else "None"
    
    user_roles_str = ", ".join(request.userRoles) if request.userRoles else "Not specified"
    domain_str = ", ".join(request.domain) if request.domain else "General"
    industry_str = ", ".join(request.industry) if request.industry else "General"
    core_use_cases_str = ", ".join(request.coreUseCases) if request.coreUseCases else "Not specified"
    
    # Format security requirements as bullet points
    security_list = request.securityRequirements or request.security or []
    security_bullets = "\n".join([f"  • {s.strip()}" for s in security_list if s.strip()]) if security_list else "  • Standard security practices"
    
    data_types_str = ", ".join(request.dataTypes) if request.dataTypes else "Not specified"
    
    # NFRs formatting
    nfrs = request.nfrs or NFRs()
    nfrs_section = f"""
    - Expected Users: {nfrs.users or 'Not specified'}
    - Expected Requests per Day: {nfrs.requests or 'Not specified'}
    - Latency Requirements: {nfrs.latency or 'Standard latency requirements'}
    - Concurrent Users: {nfrs.concurrent or 'Not specified'}"""
    
    prompt = f"""As an expert software architect, analyze the following application requirements and provide 3-5 distinct architecture recommendations. Each recommendation MUST be specifically tailored to these exact requirements and features.

CRITICAL REQUIREMENTS:
- Your recommendations MUST be specific to the {industry_str} industry
- Each architecture MUST explicitly explain how it implements EVERY feature listed below
- The technology stack MUST be chosen specifically for the {core_use_cases_str} use case
- Do NOT provide generic architectures - they must be customized for this exact application

APPLICATION OVERVIEW:
{request.overview}

APPLICATION DETAILS:
- Application Name: {request.applicationName}
- Application Type: {request.applicationType or 'Not specified'}
- Industry: {industry_str} (architectures MUST be tailored to this industry's specific needs)
- Domain: {domain_str}
- Core Use Cases: {core_use_cases_str} (architectures MUST directly support these use cases)

MUST-HAVE FEATURES (Each architecture MUST explain HOW it implements EACH of these):
{features_bullets}

FUNCTIONAL REQUIREMENTS:
- User Roles: {user_roles_str}
- Data Types: {data_types_str}
- Data Profile: {request.dataProfile or 'Not specified'}

NON-FUNCTIONAL REQUIREMENTS:{nfrs_section}
- Performance: {request.performance or 'Must handle the expected user load efficiently'}
- Reliability: {request.reliability or 'High availability and fault tolerance required'}

SECURITY REQUIREMENTS (MUST be addressed):
{security_bullets}

INTEGRATION REQUIREMENTS:
{integrations_str if integrations_str != "None" else "No external integrations specified"}

INSTRUCTIONS:
You MUST provide 3-5 architecture recommendations that are HIGHLY SPECIFIC to the requirements above. Generic or template architectures are NOT acceptable.

CRITICAL: Only recommend services and technologies that are DIRECTLY REQUIRED by the features and requirements listed above. Do NOT recommend services for features that are not mentioned. Do NOT add "nice to have" features. Be minimal and targeted.

MANDATORY REQUIREMENTS FOR EACH ARCHITECTURE:

1. INDUSTRY-SPECIFIC: The architecture must address {industry_str} industry requirements, compliance needs, and best practices
2. FEATURE-SPECIFIC: Explicitly describe HOW each of the {len(features) if features else 0} features listed above is implemented with specific technologies. DO NOT recommend technologies for features not in the list.
3. USE-CASE-SPECIFIC: Explain why this architecture is optimal for "{core_use_cases_str}" specifically
4. SCALE-SPECIFIC: Design must handle {nfrs.users or 'the specified number of'} users efficiently
5. REQUIREMENTS-ONLY: Every service/technology you recommend MUST map to a specific requirement above. Do not recommend anything else.

For EACH architecture recommendation, provide:

1. A unique identifier (id) in the format "arch-X" where X is a number
2. An icon name (e.g., "layers", "hub", "cloud", "microchip")
3. A clear, specific name that reflects the pattern AND the use case (not just generic pattern names)
4. A comprehensive description (3-4 DETAILED paragraphs) that MUST include:
   
   PARAGRAPH 1: Introduction explaining why this architecture is specifically suited for {industry_str} {core_use_cases_str}
   
   PARAGRAPH 2-3: For EACH feature listed above, explain:
   - The SPECIFIC technology/service used to implement it
   - WHY that technology is chosen for this feature
   - HOW it integrates with the overall architecture
   - DO NOT mention services for features that are not in the requirements
   
   PARAGRAPH 4: Explain how this architecture handles ONLY:
   - The {nfrs.users or 'expected'} user scale (if specified in NFRs)
   - The {industry_str} industry compliance and security requirements (ONLY those listed above)
   - The performance and reliability needs (ONLY those specified in requirements)
   
5. A ranking score (1-10) based on how well it matches the SPECIFIC requirements (not generic quality)
6. Short pros (3-4 bullet points) - MUST be specific to {industry_str} {core_use_cases_str}, not generic pros
7. Short cons (3-4 bullet points) - MUST be specific to this application's context
8. Recommendation reason - explain SPECIFICALLY why this pattern fits {industry_str} + {core_use_cases_str} + these exact features
9. Why choose - when to select this over others FOR THIS SPECIFIC USE CASE (mention the use case explicitly)
10. Mark best option (best: true for the single best match for THESE SPECIFIC requirements)
11. Detailed metrics (realistic for {industry_str} applications at this scale):
    - Latency range [min, max] in milliseconds (appropriate for {request.applicationType or 'the application type'})
    - Throughput range [min, max] in requests/second (realistic for {nfrs.users or 'the user count'})
    - Availability percentage (meeting {industry_str} industry standards)
    - Autoscaling: "Yes", "No", or "Limited" (based on actual architecture capabilities)
    - Cost range [min, max] in monthly USD (realistic for this scale and feature set)
    - Scalability score (1-10): How well it scales for THIS use case
    - Reliability score (1-10): Meeting {industry_str} reliability requirements
    - Maintainability score (1-10): For THIS specific technology stack
    - Complexity score (1-10): For teams building {core_use_cases_str} applications
    . ONLY include services needed for the listed requirements:
    - Languages: Chosen specifically for the features (explain why in your description)
    - Frameworks: Must match {request.applicationType or 'the application type'} and listed features
    - Runtime: Optimized for the workload type
    - Cloud provider: AWS, Azure, or GCP - choose based on {industry_str} needs and feature requirements
    - Infrastructure components (ONLY include if required by the features above):
      * compute: List actual services ONLY if needed for the features (e.g., "ECS Fargate for containerized workloads")
      * database: Specify exact databases ONLY for the data types mentioned ({data_types_str}). Do not add databases not needed.
      * cache: ONLY specify if features explicitly require caching or performance needs demand it
      * messaging: ONLY if features require async/real-time communication. List actual services (SQS, SNS, EventBridge, etc.)
      * storage: Specify storage ONLY for the data profile mentioned ({request.dataProfile or 'the data needs'}). Do not add S3/blob storage unless needed.
      * apiGateway: ONLY if the architecture requires API management
      * authentication: ONLY if user roles ({user_roles_str}) require it. Specify solution.
      * security: List ONLY services addressing the security requirements listed above. Do not add extra security services.
      * networking: ONLY include CDN/load balancers if scale or architecture demands it
      * monitoring: Include ONLY if it's essential for the architecture
      * logging: Include ONLY if it's essential for the architecture
    - CI/CD: Specific tools appropriate for this stack (standard DevOps needs)
    
    IMPORTANT: If a component is not needed for the listed features, DO NOT include it or leave it empty/blank.LK, etc.)
    - CI/CD: Specific tools appropriate for this stack
    
13. bestFor: 3-5 scenarios that specifically match {industry_str} {core_use_cases_str} with these features
14. avoidWhen: 3-5 scenarios where this would NOT work (be specific to the industry and use case)

Return the response as a valid JSON object with the following structure:
{{
  "architectures": [
    {{
      "id": "arch-1",
      "icon": "icon-name",
      "name": "Architecture Pattern Name",
      "description": "Detailed description...",
      "ranking": 9,
      "shortPros": "Pro 1. Pro 2. Pro 3.",
      "shortCons": "Con 1. Con 2. Con 3.",
      "recommendationReason": "Why this architecture fits...",
      "whyChoose": "Choose this when...",
      "best": true,
      "metrics": {{
        "latency": [50, 200],
        "throughput": [100, 1000],
        "availability": 99.9,
        "autoscaling": "Yes",
        "cost": [500, 2000],
        "scalability": 9,
        "reliability": 9,
        "maintainability": 8,
        "complexity": 6
      }},
      "technologyStack": {{
        "languages": "Python, JavaScript",
        "frameworks": "FastAPI, React",
        "runtime": "Node.js, Python 3.11",
        "cloudProvider": "AWS",
        "infra": {{
          "compute": "AWS Lambda, ECS",
          "database": "DynamoDB, RDS PostgreSQL",
          "cache": "Redis (ElastiCache)",
          "messaging": "SQS, SNS",
          "storage": "S3",
          "apiGateway": "API Gateway",
          "authentication": "Cognito",
          "security": "WAF, Security Groups, IAM",
          "networking": "CloudFront, ALB",
          "monitoring": "CloudWatch",
          "logging": "CloudWatch Logs"
        }},
        "cicd": {{
          "pipeline": "GitHub Actions, AWS CodePipeline",
          "containerization": "Docker, ECS",
          "testing": "Jest, Pytest",
          "iac": "Terraform, CloudFormation"
        }}
      }},
      "bestFor": ["Scenario 1", "Scenario 2", "Scenario 3"],
      "avoidWhen": ["Scenario 1", "Scenario 2", "Scenario 3"]
    }}
  ]
}}

FINAL REMINDERS:
1. DO NOT provide generic architectures - every recommendation must be customized for {industry_str} {core_use_cases_str}
2. EVERY feature must be explicitly addressed with specific implementation details
3. Technology choices must be justified by the requirements, not just listed
4. Consider {industry_str} industry compliance, security standards, and best practices
5. Scale recommendations for {nfrs.users or 'the specified number of'} users
6. Ensure descriptions explain HOW each feature is implemented, not just THAT it's supported
7. CRITICAL: Only recommend services that are REQUIRED by the listed features. If a feature is not mentioned, do not include related services.
8. Be minimal and targeted - avoid over-engineering or suggesting unnecessary components.
9. Each service in your technology stack must directly map to a specific requirement or feature listed above."""

    return prompt


def parse_openai_architectures(response_json: Dict[str, Any]) -> List[Architecture]:
    """
    Parse OpenAI response JSON into Architecture objects.
    
    Handles various response formats and ensures all required fields are present.
    """
    architectures = []
    
    # Extract architectures array from response
    architectures_data = response_json.get("architectures", [])
    
    for idx, arch_data in enumerate(architectures_data):
        try:
            # Ensure required fields have defaults
            arch_data.setdefault("id", f"arch-{idx + 1}")
            arch_data.setdefault("icon", "architecture")
            arch_data.setdefault("name", f"Architecture {idx + 1}")
            arch_data.setdefault("description", "")
            arch_data.setdefault("ranking", 5)
            arch_data.setdefault("shortPros", "")
            arch_data.setdefault("shortCons", "")
            arch_data.setdefault("recommendationReason", "")
            arch_data.setdefault("whyChoose", "")
            arch_data.setdefault("best", idx == 0)
            arch_data.setdefault("bestFor", [])
            arch_data.setdefault("avoidWhen", [])
            
            # Ensure nested objects exist
            arch_data.setdefault("metrics", {})
            arch_data.setdefault("technologyStack", {})
            arch_data["technologyStack"].setdefault("infra", {})
            arch_data["technologyStack"].setdefault("cicd", {})
            
            # Create Architecture object
            architecture = Architecture(**arch_data)
            architectures.append(architecture)
            
        except Exception as e:
            logger.error(f"Error parsing architecture {idx}: {str(e)}", exc_info=True)
            continue
    
    return architectures


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
