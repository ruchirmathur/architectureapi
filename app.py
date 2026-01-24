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

# Azure OpenAI Configuration
OPENAI_ENDPOINT =os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT = "gpt-5-mini"
MAX_OVERVIEW_LENGTH = 4000  # Maximum characters for overview input
MAX_OUTPUT_TOKENS = 16000  # Maximum tokens for OpenAI response (increased for diagram JSON)

# Global cosmos service instance
cosmos_service: Optional[CosmosDBService] = None


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
    diagram: Optional[ArchitectureDiagram] = None
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
        
        if existing_requirement:
            # Update existing requirement
            logger.info(f"Updating existing requirement: {existing_requirement['id']}")
            saved_requirement = await cosmos.update_requirement(
                requirement_id=existing_requirement["id"],
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
    # Define all 9 required steps
    required_steps = {
        "domain",
        "coreUseCases",
        "features",
        "nfrs",
        "userRoles",
        "dataProfile",
        "security",
        "performance",
        "reliability"
    }
    
    # Get step transitions from metadata
    metadata = req.get("metadata", {})
    step_transitions = metadata.get("stepTransitions", [])
    
    # Extract completed steps
    completed_steps = {transition.get("step") for transition in step_transitions if transition.get("step")}
    
    # Check if all 9 steps are completed
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
    Get architecture recommendations from OpenAI based on the application overview
    
    This endpoint analyzes the provided application overview and returns
    a list of architecture recommendations including cloud services,
    design patterns, and best practices.
    """
    try:
        logger.info(
            f"Received architecture recommendation request for tenant: {request.tenantId}, "
            f"session: {request.sessionId}, user: {current_user.username}"
        )
        logger.info(f"Overview length: {len(request.overview)} characters")
        
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured")
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key is not configured"
            )
        
        logger.info(f"OpenAI endpoint: {OPENAI_ENDPOINT}")
        logger.info(f"OpenAI deployment: {OPENAI_DEPLOYMENT}")
        
        # Construct the prompt for OpenAI using the overview from the request
        system_prompt = """You are an expert software and cloud architect with deep knowledge of production systems, cloud platforms, and proven architectural patterns. 

Analyze the provided application requirements and recommend 2-4 DISTINCT architecture patterns that are:
- PRODUCTION-READY: Use proven, battle-tested technologies and patterns
- COHERENT: All technology choices must work together seamlessly
- COMPATIBLE: Ensure all components integrate well with each other
- SPECIFIC: Use actual service names (e.g., "AWS Lambda" not "serverless compute")
- REALISTIC: Base metrics and costs on real-world production data
- APPROPRIATE: Match technology choices to the actual requirements (scale, traffic, complexity, team size)

CRITICAL RULES:
1. Each recommendation MUST use a DIFFERENT architectural pattern (e.g., Microservices, Serverless, Monolith, Event-Driven)
2. Each recommendation should target a DIFFERENT cloud provider (AWS, Azure, or Google Cloud)
3. Pick ONE primary technology stack per architecture - DO NOT mix incompatible technologies:
   - Choose ONE language (e.g., "Java" OR "C#" OR "Python" OR "Node.js/TypeScript", NOT "Java, .NET")
   - Framework must match the chosen language (Spring for Java, ASP.NET for C#, FastAPI for Python, Express for Node.js)
   - Runtime must align with language (JVM for Java, .NET Runtime for C#, Python runtime, Node.js)
   - All choices must form a SINGLE coherent tech stack
4. Technology ecosystem coherence:
   - Database must match data requirements and scale needs
   - CI/CD tools should align with cloud provider and language
   - Monitoring/logging should integrate with chosen platform
   - Containerization approach should match deployment strategy
5. Metrics must be realistic for the chosen architecture and scale
6. Consider actual costs - don't randomly select expensive services for simple apps
7. Security and networking choices should match compliance and architecture needs

Return a JSON object with this structure (populate all fields with thoughtful, integrated recommendations):
{
  "architectures": [
    {
      "id": "<unique-kebab-case-id>",
      "name": "<architecture pattern name>",
      "description": "<brief description>",
      "ranking": <1-4>,
      "best": <true/false>,
      "shortPros": "<one line advantages>",
      "shortCons": "<one line drawbacks>",
      "recommendationReason": "<why this fits>",
      "whyChoose": "<why choose over alternatives>",
      "metrics": {
        "latency": [<min ms>, <max ms>],
        "throughput": [<min req/s>, <max req/s>],
        "availability": <percentage like 99.95>,
        "autoscaling": "<Yes/No/Limited>",
        "cost": [<min monthly USD>, <max monthly USD>],
        "scalability": <1-10>,
        "reliability": <1-10>,
        "maintainability": <1-10>,
        "complexity": <1-10>
      },
      "technologyStack": {
        "languages": "<programming languages>",
        "frameworks": "<frameworks>",
        "runtime": "<runtime environment>",
        "cloudProvider": "<AWS/Azure/GCP>",
        "infra": {
          "compute": "<compute services>",
          "database": "<database services>",
          "cache": "<caching services>",
          "messaging": "<messaging services>",
          "storage": "<storage services>",
          "apiGateway": "<API gateway>",
          "authentication": "<auth services>",
          "security": "<security services>",
          "networking": "<networking and load balancing>",
          "monitoring": "<monitoring tools>",
          "logging": "<logging services>"
        },
        "cicd": {
          "pipeline": "<CI/CD pipeline tools>",
          "containerization": "<container tools>",
          "testing": "<testing frameworks>",
          "iac": "<Infrastructure as Code tools>"
        }
      },
      "bestFor": ["<scenario1>", "<scenario2>"],
      "avoidWhen": ["<scenario1>", "<scenario2>"],
      "diagram": {
        "name": "<diagram title>",
        "description": "<diagram description>",
        "shapes": [
          {
            "tool": "<aws|azure|gcp|security>",
            "x": <x-coordinate>,
            "y": <y-coordinate>,
            "id": "<unique-id>",
            "service": "<service-name>",
            "width": 64,
            "height": 64,
            "rotation": 0
          },
          {
            "tool": "arrow",
            "arrowType": "<single|double|dashed>",
            "id": "<arrow-id>",
            "points": [<x1>, <y1>, <x2>, <y2>],
            "stroke": "<color-hex>",
            "strokeWidth": <width>,
            "dash": [<dash>, <gap>]
          }
        ]
      }
    }
  ]
}

DIAGRAM REQUIREMENTS - CRITICAL:
1. **Canvas Dimensions**: Keep all shapes within 0-1000px width and 0-600px height
2. **Tool Selection**: Use "aws" for AWS, "azure" for Azure, "gcp" for GCP based on cloudProvider in technologyStack
3. **Service Mapping** - Map EXACTLY from technologyStack to diagram services:
   - Compute: aws-lambda (serverless), aws-ecs/azure-container-instances (containers), aws-ec2/azure-vm (VMs), aws-eks/azure-aks (k8s)
   - Database: aws-rds, aws-dynamodb, aws-aurora, azure-sql, azure-cosmos, gcp-cloud-sql
   - Cache: aws-elasticache, azure-redis, gcp-memorystore
   - Storage: aws-s3, azure-storage, gcp-cloud-storage
   - Messaging: aws-sqs, aws-sns, azure-service-bus, gcp-pub-sub
   - API Gateway: aws-api-gateway, azure-api-management, gcp-api-gateway
   - Auth: security-auth0, security-azuread, security-cognito
   - Monitoring: aws-cloudwatch, azure-monitor, gcp-operations
   - Networking: aws-elb, aws-cloudfront, azure-front-door, azure-cdn, gcp-load-balancer

4. **Layout Strategy** (FOLLOW EXACTLY):
   - Row 1 (y: 50): DNS/CDN - center at x: 500
   - Row 2 (y: 150): Load Balancer/API Gateway - center at x: 500
   - Row 3 (y: 280): Authentication service - right side at x: 700
   - Row 4 (y: 350): Application Services/Compute - spread evenly x: 200, 400, 600, 800 (use only as many as needed)
   - Row 5 (y: 500): Data Layer (databases, cache, storage, queues) - spread evenly x: 200, 400, 600, 800

5. **Arrow Positioning** - Calculate arrow points correctly:
   - Shape center is at (x + 32, y + 32) since width/height is 64
   - For vertical connections: same x-coordinate, different y values
   - For horizontal: different x values, same y-coordinate
   - Example: Shape at (400, 150) to shape at (400, 350): points: [432, 214, 432, 350]
   - Always connect center-to-center or edge-to-edge properly

6. **Connection Logic** - Only connect services that actually integrate:
   - User traffic: DNS → CDN → Load Balancer → API Gateway
   - API Gateway ↔ Authentication (bidirectional)
   - API Gateway → Compute Services (one-way)
   - Compute Services ↔ Databases (bidirectional for their specific DB)
   - Compute Services ↔ Cache (bidirectional)
   - Compute Services → Storage (one-way writes, or bidirectional for reads)
   - Compute Services → Message Queues (one-way publish, or subscribe)
   - DO NOT create random connections between unrelated services

7. **Arrow Types & Colors**:
   - "single": One-way data flow, use for: API calls, writes, publishes
   - "double": Bidirectional, use for: database reads/writes, cache operations, auth validation
   - "dashed": Optional/async, use for: monitoring, logging, background jobs
   - Colors: #3498db (user traffic), #2ecc71 (service calls), #e74c3c (auth), #2c3e50 (database), #f39c12 (storage), #9b59b6 (events/queues)

8. **Service Count**: Include only services mentioned in technologyStack - do not add extras. If compute is "Lambda", show 1-3 Lambda functions. If database is "RDS + DynamoDB", show both.

9. **Validation**: Before finalizing, verify:
   - All x coordinates are 50-950 (leaving 50px margin)
   - All y coordinates are 50-550 (leaving 50px margin)
   - Arrow points match actual shape center positions (x+32, y+32)
   - No services are duplicated
   - Every service shown is mentioned in technologyStack

Be concise. Rank architectures 1-4 based on fit to requirements (1 is best). Set "best": true only for rank 1."""

        # Truncate overview if too long to save tokens
        overview_text = request.overview[:MAX_OVERVIEW_LENGTH] if len(request.overview) > MAX_OVERVIEW_LENGTH else request.overview
        logger.info(f"Truncated overview to {len(overview_text)} characters")
        
        user_prompt = f"""Please analyze the following application requirements overview and provide architecture recommendations:

APPLICATION REQUIREMENTS:
{overview_text}

Based on these requirements, provide comprehensive architecture recommendations as a JSON object."""

        logger.info("Creating OpenAI client...")
        # Call Azure OpenAI API
        try:
            client = OpenAI(
                base_url=OPENAI_ENDPOINT,
                api_key=OPENAI_API_KEY
            )
            logger.info("OpenAI client created successfully")
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {str(e)}", exc_info=True)
            raise
        
        logger.info("Calling OpenAI API...")
        try:
            completion = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_OUTPUT_TOKENS
            )
            logger.info("OpenAI API call completed successfully")
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}", exc_info=True)
            raise
        
        # Parse the OpenAI response
        logger.info(f"OpenAI finish_reason: {completion.choices[0].finish_reason}")
        logger.info(f"OpenAI usage: {completion.usage}")
        
        content = completion.choices[0].message.content
        
        # Handle None or empty content
        if not content:
            finish_reason = completion.choices[0].finish_reason
            logger.error(f"OpenAI returned empty content. Finish reason: {finish_reason}")
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI returned empty response. Finish reason: {finish_reason}"
            )
        
        logger.info(f"OpenAI response length: {len(content)} characters")
        logger.debug(f"OpenAI raw response: {content[:500]}...")  # Log first 500 chars
        
        logger.info("Parsing JSON response...")
        try:
            recommendations_data = json.loads(content)
            logger.info("JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}. Content preview: {content[:200]}", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse OpenAI response as JSON: {str(e)}"
            )
        
        # Validate response structure
        if "architectures" not in recommendations_data:
            logger.error(f"Missing 'architectures' key in response. Keys found: {list(recommendations_data.keys())}")
            raise HTTPException(
                status_code=502,
                detail="OpenAI response missing 'architectures' key"
            )
        
        logger.info(f"Found {len(recommendations_data.get('architectures', []))} architectures in response")
        
        # Convert OpenAI response to full response model
        architectures = []
        for idx, arch in enumerate(recommendations_data.get("architectures", [])):
            logger.info(f"Processing architecture {idx + 1}...")
            try:
                arch_id = arch.get("id", "unknown")
                logger.debug(f"Architecture ID: {arch_id}")
            
                # Extract technology stack
                tech_stack_data = arch.get("technologyStack", {})
                infra_data = tech_stack_data.get("infra", {})
                cicd_data = tech_stack_data.get("cicd", {})
            
                infrastructure = Infrastructure(
                    compute=infra_data.get("compute", ""),
                    database=infra_data.get("database", ""),
                    cache=infra_data.get("cache", ""),
                    messaging=infra_data.get("messaging", ""),
                    storage=infra_data.get("storage", ""),
                    apiGateway=infra_data.get("apiGateway", ""),
                    authentication=infra_data.get("authentication", ""),
                    security=infra_data.get("security", ""),
                    networking=infra_data.get("networking", ""),
                    monitoring=infra_data.get("monitoring", ""),
                    logging=infra_data.get("logging", "")
                )
                logger.debug(f"Infrastructure created for {arch_id}")
            
                cicd = CICD(
                    pipeline=cicd_data.get("pipeline", ""),
                    containerization=cicd_data.get("containerization", ""),
                    testing=cicd_data.get("testing", ""),
                    iac=cicd_data.get("iac", "")
                )
                logger.debug(f"CICD created for {arch_id}")
            
                technology_stack = TechnologyStack(
                    languages=tech_stack_data.get("languages", ""),
                    frameworks=tech_stack_data.get("frameworks", ""),
                    runtime=tech_stack_data.get("runtime", ""),
                    cloudProvider=tech_stack_data.get("cloudProvider", ""),
                    infra=infrastructure,
                    cicd=cicd
                )
                logger.debug(f"Technology stack created for {arch_id}")
            
                # Extract metrics
                metrics_data = arch.get("metrics", {})
                metrics = Metrics(
                    latency=metrics_data.get("latency", []),
                    throughput=metrics_data.get("throughput", []),
                    availability=metrics_data.get("availability", 0),
                    autoscaling=metrics_data.get("autoscaling", ""),
                    cost=metrics_data.get("cost", []),
                    scalability=metrics_data.get("scalability", 5),
                    reliability=metrics_data.get("reliability", 5),
                    maintainability=metrics_data.get("maintainability", 5),
                    complexity=metrics_data.get("complexity", 5)
                )
                logger.debug(f"Metrics created for {arch_id}")
            
                # Extract diagram
                diagram = None
                diagram_data = arch.get("diagram")
                if diagram_data:
                    logger.debug(f"Processing diagram for {arch_id}...")
                    shapes = []
                    for shape_idx, shape_data in enumerate(diagram_data.get("shapes", [])):
                        try:
                            shape = DiagramShape(
                                tool=shape_data.get("tool", ""),
                                x=shape_data.get("x", 0),
                                y=shape_data.get("y", 0),
                                id=shape_data.get("id", ""),
                                service=shape_data.get("service"),
                                arrowType=shape_data.get("arrowType"),
                                width=shape_data.get("width"),
                                height=shape_data.get("height"),
                                rotation=shape_data.get("rotation", 0),
                                points=shape_data.get("points"),
                                stroke=shape_data.get("stroke"),
                                strokeWidth=shape_data.get("strokeWidth"),
                                dash=shape_data.get("dash")
                            )
                            shapes.append(shape)
                        except Exception as shape_error:
                            logger.error(f"Error creating shape {shape_idx} for {arch_id}: {str(shape_error)}", exc_info=True)
                            logger.error(f"Shape data: {shape_data}")
                            # Continue processing other shapes
                    
                    diagram = ArchitectureDiagram(
                        name=diagram_data.get("name", ""),
                        description=diagram_data.get("description", ""),
                        shapes=shapes
                    )
                    logger.debug(f"Diagram created with {len(shapes)} shapes for {arch_id}")
            
                # Build the full architecture object from OpenAI response
                architecture = Architecture(
                    id=arch_id,
                    icon=f"/icons/{arch_id}.png",
                    name=arch.get("name", ""),
                    description=arch.get("description", ""),
                    ranking=arch.get("ranking", 1),
                    shortPros=arch.get("shortPros", ""),
                    shortCons=arch.get("shortCons", ""),
                    recommendationReason=arch.get("recommendationReason", ""),
                    whyChoose=arch.get("whyChoose", ""),
                    best=arch.get("best", False),
                    metrics=metrics,
                    technologyStack=technology_stack,
                    diagram=diagram,
                    bestFor=arch.get("bestFor", []),
                    avoidWhen=arch.get("avoidWhen", [])
                )
                architectures.append(architecture)
                logger.info(f"Architecture {idx + 1} ({arch_id}) processed successfully")
                
            except Exception as arch_error:
                logger.error(f"Error processing architecture {idx + 1}: {str(arch_error)}", exc_info=True)
                logger.error(f"Architecture data: {arch}")
                # Continue processing other architectures instead of failing completely
                continue
        
        logger.info(
            f"Successfully generated {len(architectures)} architecture recommendations for "
            f"tenant: {request.tenantId}, session: {request.sessionId}"
        )
        
        logger.info("Creating response object...")
        try:
            response = ArchitectureRecommendationResponse(
                success=True,
                message="Architecture recommendations generated successfully",
                tenantId=request.tenantId,
                sessionId=request.sessionId,
                architectures=architectures
            )
            logger.info("Response object created successfully")
            
            # Try to serialize to dict to check for serialization issues
            logger.info("Attempting to serialize response to dict...")
            response_dict = response.model_dump()
            logger.info(f"Response serialized successfully. Size: {len(str(response_dict))} characters")
            
            logger.info("Returning response...")
            return response
            
        except Exception as serialize_error:
            logger.error(f"Error during response creation or serialization: {str(serialize_error)}", exc_info=True)
            logger.error(f"Number of architectures: {len(architectures)}")
            for idx, arch in enumerate(architectures):
                logger.error(f"Architecture {idx + 1} ID: {arch.id}, has diagram: {arch.diagram is not None}")
                if arch.diagram:
                    logger.error(f"Architecture {idx + 1} diagram has {len(arch.diagram.shapes)} shapes")
            raise
        
    except HTTPException as http_ex:
        logger.error(f"HTTP exception in get_architecture_recommendations: {http_ex.detail}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_architecture_recommendations: {str(e)}", exc_info=True)
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate architecture recommendations: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
