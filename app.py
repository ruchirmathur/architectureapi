"""
Architecture Requirements API - Simplified
FastAPI application for storing architecture requirements in Cosmos DB
"""
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
import logging
import os
import json
import httpx
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv

# Load root .env (shared config) then python_api/.env (secrets) — local wins
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'), override=False)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)
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
AZURE_SERVICE_BUS_DESIGN_QUEUE = os.getenv("AZURE_SERVICE_BUS_DESIGN_QUEUE", "architecture-design")
AZURE_SERVICE_BUS_CODE_QUEUE = os.getenv("AZURE_SERVICE_BUS_CODE_QUEUE", "architecture-code")

# Azure OpenAI Configuration
OPENAI_ENDPOINT =os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT = "gpt-5-mini"
MAX_OVERVIEW_LENGTH = 4000  # Maximum characters for overview input
MAX_OUTPUT_TOKENS = 16000  # Maximum tokens for OpenAI response (increased for diagram JSON)
MAX_LLD_OUTPUT_TOKENS = 32000  # Maximum tokens for LLD generation (increased for complex architectures)
MAX_CODE_GEN_OUTPUT_TOKENS = 64000  # Maximum tokens for code generation (needs to process full LLD and generate all files)

# Cosmos DB Design Container
COSMOS_DESIGNS_CONTAINER = os.getenv("COSMOS_DESIGNS_CONTAINER", "design")

# GitHub OAuth Configuration
GITHUB_CLIENT_ID = os.getenv("REACT_APP_GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")


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
    domain: Optional[Union[str, List[str]]] = Field(default=[], description="Domain areas (e.g., Healthcare)")
    industry: Optional[Union[str, List[str]]] = Field(default=[], description="Industry sectors")
    coreUseCases: Optional[Union[str, List[str]]] = Field(default=[], description="Core use cases")
    useCases: Optional[Union[str, List[str]]] = Field(default=[], description="Use cases")
    features: Optional[str] = Field(default="", description="Comma-separated features string")
    featuresList: Optional[List[str]] = Field(default=[], description="Features as a list")
    nfrs: Optional[NFRs] = Field(default_factory=NFRs, description="Non-functional requirements")
    performance: Optional[str] = Field(default="", description="Performance requirements")
    reliability: Optional[str] = Field(default="", description="Reliability requirements")
    dataProfile: Optional[str] = Field(default="", description="Data profile description")
    dataTypes: Optional[Union[str, List[str]]] = Field(default=[], description="Types of data managed")
    userRoles: Optional[Union[str, List[str]]] = Field(default=[], description="User roles")
    userTypes: Optional[Union[str, List[str]]] = Field(default=[], description="User types")
    security: Optional[Union[str, List[str]]] = Field(default=[], description="Security requirements")
    securityRequirements: Optional[Union[str, List[str]]] = Field(default=[], description="Security requirements (clean)")
    integrations: Optional[str] = Field(default="", description="Integrations description")
    integrationsList: Optional[Union[str, List[str]]] = Field(default=[], description="Integrations as a list")
    cloudPlatform: Optional[str] = Field(default="", description="Preferred cloud platform (AWS, Azure, GCP, etc.)")
    sourceControl: Optional[str] = Field(default="", description="Source control system (GitHub, GitLab, Bitbucket, Azure DevOps)")
    
    @field_validator('domain', 'industry', 'coreUseCases', 'useCases', 'dataTypes', 'userRoles', 'userTypes', 'security', 'securityRequirements', 'integrationsList', mode='before')
    @classmethod
    def convert_string_to_list(cls, v):
        """Convert string values to lists"""
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v if v is not None else []


# Architecture recommendation response models
class ServiceCost(BaseModel):
    """Detailed cost for a specific service"""
    service: str = Field(..., description="Service name")
    pricingModel: str = Field(default="", description="Pay-as-you-go, Reserved, Committed Use, etc.")
    monthlyCost: List[float] = Field(default=[], description="[min, max] monthly cost in USD")
    details: str = Field(default="", description="Cost calculation details and assumptions")


class CapacityPlanning(BaseModel):
    """Capacity planning based on user load"""
    expectedUsers: str = Field(default="", description="Expected user count")
    peakConcurrentUsers: str = Field(default="", description="Peak concurrent users estimate")
    requestsPerSecond: str = Field(default="", description="Estimated requests per second")
    dataGrowthPerMonth: str = Field(default="", description="Estimated data growth per month")
    computeUnits: str = Field(default="", description="Required compute capacity (cores, memory, instances)")
    storageCapacity: str = Field(default="", description="Required storage capacity (GB, TB)")
    bandwidthRequirements: str = Field(default="", description="Network bandwidth requirements")
    scalingStrategy: str = Field(default="", description="How the system scales with user growth")


class CostBreakdown(BaseModel):
    """Detailed cost breakdown"""
    serviceCosts: List[ServiceCost] = Field(default_factory=list, description="Cost per service")
    totalMonthly: List[float] = Field(default=[], description="[min, max] total monthly cost in USD")
    assumptions: str = Field(default="", description="Cost calculation assumptions and user load basis")
    optimizationTips: str = Field(default="", description="Cost optimization recommendations")
    capacityPlanning: Optional[CapacityPlanning] = Field(default=None, description="Capacity planning details")


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
    
    @field_validator('latency', 'throughput', 'cost', mode='before')
    @classmethod
    def clean_metric_arrays(cls, v):
        """Clean metric arrays by removing units from values"""
        if not isinstance(v, list):
            return []
        
        def clean_value(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                # Remove common units and convert to float
                cleaned = val.lower().replace('ms', '').replace('req/s', '').replace('requests/sec', '').replace('rps', '').replace('$', '').replace(',', '').strip()
                try:
                    return float(cleaned)
                except:
                    return 0.0
            return 0.0
            
        return [clean_value(val) for val in v]
    
    @field_validator('availability', mode='before')
    @classmethod  
    def clean_availability(cls, v):
        """Clean availability value"""
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.replace('%', '').strip()
            try:
                return float(cleaned)
            except:
                return 0.0
        return 0.0


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
    """Diagram shape used by the canvas (basic, cloud, or arrow)."""
    tool: str = Field(
        ...,
        description=(
            "Tool type: basic shapes (rect, ellipse, circle, triangle, diamond, "
            "parallelogram, star, polygon, line), cloud & services (aws, azure, gcp, "
            "security, cloud-infra, generic, uploaded-image, component), or arrow."
        ),
    )
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
    costBreakdown: Optional[CostBreakdown] = Field(default=None, description="Detailed cost breakdown and capacity planning")
    bestFor: List[str] = []
    avoidWhen: List[str] = []


class ArchitectureDiagramRequest(BaseModel):
    """Request model for generating an architecture diagram"""
    tenantId: str = Field(default="default", description="Tenant ID")
    applicationName: str = Field(..., description="Application name")
    sessionId: str = Field(default="default", description="Session ID")
    overview: str = Field(..., max_length=MAX_OVERVIEW_LENGTH, description="Architecture summary / overview")
    architecture: Optional[Dict[str, Any]] = Field(default=None, description="Selected architecture recommendation object")
    features: Optional[List[str]] = Field(default_factory=list, description="Key features to highlight in the diagram")
    notes: Optional[str] = Field(default=None, description="Any additional notes or assumptions")


class ArchitectureDiagramResponse(BaseModel):
    """Response model for architecture diagram generation"""
    success: bool
    diagramId: Optional[str] = None
    diagram: Optional[ArchitectureDiagram] = None
    error: Optional[str] = None


class LLDDiagram(BaseModel):
    """Low level design diagram"""
    name: str = Field(..., description="Diagram name")
    description: str = Field(..., description="Diagram description")
    shapes: List[DiagramShape] = Field(default=[], description="List of shapes in the diagram")


class LLDDiagramRequest(BaseModel):
    """Request model for generating an LLD diagram"""
    tenantId: str = Field(default="default", description="Tenant ID")
    applicationName: str = Field(..., description="Application name")
    sessionId: str = Field(default="default", description="Session ID")
    overview: Optional[str] = Field(default=None, description="High level summary / requirements")
    architecture: Optional[Dict[str, Any]] = Field(default=None, description="Selected architecture recommendation object")
    lld: Optional[Dict[str, Any]] = Field(default=None, description="Existing LLD JSON to convert into a diagram")
    features: Optional[List[str]] = Field(default_factory=list, description="Key features or flows to emphasize")
    notes: Optional[str] = Field(default=None, description="Any additional notes or assumptions")


class LLDDiagramResponse(BaseModel):
    """Response model for LLD diagram generation"""
    success: bool
    diagramId: Optional[str] = None
    diagram: Optional[LLDDiagram] = None
    error: Optional[str] = None


class ArchitectureRecommendationResponse(BaseModel):
    """Response model for architecture recommendations"""
    success: bool
    message: str
    tenantId: str
    sessionId: str
    architectures: Optional[List[Architecture]] = None
    features: Optional[List[str]] = None
    error: Optional[str] = None


class LLDRequest(BaseModel):
    """Request model for generating Low Level Design"""
    tenantId: str = Field(default="default", description="Tenant ID")
    applicationName: str = Field(default="Application", description="Application name")
    sessionId: str = Field(default="default", description="Session ID")
    userId: str = Field(default="default", description="User ID")
    requirements: Optional[str] = Field(default="", description="Requirements or overview")
    overview: Optional[str] = Field(default="", description="Application overview")
    architecture: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Architecture object")
    features: Optional[List[Any]] = Field(default_factory=list, description="Features list")


class LLDResponse(BaseModel):
    """Response model for LLD generation"""
    success: bool
    designId: Optional[str] = None
    featureCount: Optional[int] = None
    lld: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== CODE GENERATION MODELS ====================

class CodeGenerationRequest(BaseModel):
    """Request model for generating code from LLD"""
    tenantId: str = Field(default="default", description="Tenant ID")
    applicationName: str = Field(default="Application", description="Application name")
    sessionId: str = Field(default="default", description="Session ID")
    userId: Optional[str] = Field(default=None, description="User ID")
    designId: Optional[str] = Field(default=None, description="Design identifier")
    featureCount: Optional[int] = Field(default=None, description="Number of features to generate code for")
    lld: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Low Level Design specification")
    architecture: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Architecture object")
    error: Optional[str] = None


class GeneratedFile(BaseModel):
    """Represents a generated code file"""
    path: str = Field(..., description="File path relative to project root")
    content: str = Field(..., description="Generated file content")


class FeatureFiles(BaseModel):
    """Files grouped by feature"""
    featureName: str = Field(..., description="Feature name")
    frontend: List[GeneratedFile] = Field(default_factory=list, description="Frontend files for this feature")
    backend: List[GeneratedFile] = Field(default_factory=list, description="Backend files for this feature")
    dataModels: List[GeneratedFile] = Field(default_factory=list, description="Data model files for this feature")
    tests: List[GeneratedFile] = Field(default_factory=list, description="Test files for this feature")


class CategorizedFiles(BaseModel):
    """Files organized by category"""
    frontend: List[GeneratedFile] = Field(default_factory=list, description="Frontend components, pages, styles")
    backend: List[GeneratedFile] = Field(default_factory=list, description="Backend APIs, handlers, services")
    dataModels: List[GeneratedFile] = Field(default_factory=list, description="Database schemas, models, migrations")
    infrastructure: List[GeneratedFile] = Field(default_factory=list, description="IaC templates, deployment configs")
    config: List[GeneratedFile] = Field(default_factory=list, description="Configuration files (package.json, tsconfig, etc.)")
    documentation: List[GeneratedFile] = Field(default_factory=list, description="README, API docs, architecture docs")
    tests: List[GeneratedFile] = Field(default_factory=list, description="Test files")


class ProjectStructure(BaseModel):
    """Represents the generated project structure"""
    projectName: str = Field(default="", description="Project/application name")
    frontend: Dict[str, Any] = Field(default_factory=dict, description="Frontend folder structure")
    backend: Dict[str, Any] = Field(default_factory=dict, description="Backend folder structure")
    infrastructure: Dict[str, Any] = Field(default_factory=dict, description="Infrastructure folder structure")
    dataModels: Dict[str, Any] = Field(default_factory=dict, description="Data models folder structure")


class CodeGenerationResponse(BaseModel):
    """Response model for code generation"""
    success: bool
    message: str = Field(..., description="Response message")
    projectName: str = Field(default="", description="Project/application name")
    projectStructure: Optional[ProjectStructure] = None
    featureBasedFiles: Optional[List[FeatureFiles]] = Field(default_factory=list, description="Files organized by feature")
    categorizedFiles: Optional[CategorizedFiles] = None
    generatedFiles: Optional[List[GeneratedFile]] = None
    fileCount: Optional[int] = None
    technologies: Optional[List[str]] = None
    instructions: Optional[str] = None
    error: Optional[str] = None


# ==================== INFRASTRUCTURE GENERATION MODELS ====================

class InfrastructureRequest(BaseModel):
    """Request model for generating infrastructure scripts"""
    tenantId: str = Field(..., description="Tenant ID")
    sessionId: Optional[str] = Field(default="", description="Session ID")
    applicationName: str = Field(..., description="Application name")
    description: str = Field(..., description="Architecture description")
    features: Optional[List[str]] = Field(default_factory=list, description="Application features")
    technologyStack: Dict[str, Any] = Field(..., description="Technology stack with cloudProvider, runtime, infra details")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class InfrastructureScript(BaseModel):
    """Represents a generated infrastructure script"""
    fileName: str = Field(..., description="Script file name")
    format: str = Field(..., description="IaC format (terraform, cloudformation, sam, etc.)")
    content: str = Field(..., description="Script content")
    description: str = Field(..., description="What this script provisions")
    services: List[str] = Field(default_factory=list, description="Cloud services provisioned by this script")


class InfrastructureResponse(BaseModel):
    """Response model for infrastructure generation"""
    success: bool
    message: str = Field(..., description="Response message")
    applicationName: str = Field(default="", description="Application name")
    cloudProvider: str = Field(default="", description="Cloud provider")
    iacFormat: str = Field(default="", description="Infrastructure as Code format used")
    scripts: List[InfrastructureScript] = Field(default_factory=list, description="Generated infrastructure scripts")
    services: List[str] = Field(default_factory=list, description="All cloud services required")
    deploymentInstructions: str = Field(default="", description="Step-by-step deployment instructions")
    estimatedCost: Optional[str] = Field(default="", description="Estimated monthly cost range")
    error: Optional[str] = None


class AuthenticatedUser:
    """Represents an authenticated user with tenant context"""
    def __init__(self, tenant_id: str, user_id: str, username: str, email: str):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.username = username
        self.email = email


def _build_fallback_architecture_diagram(
    application_name: str,
    architecture_name: str,
    overview: str,
    tech_summary: str,
    features: Optional[List[str]]
) -> Dict[str, Any]:
    """Build a simple default architecture diagram if AI returns no content.

    This ensures /api/architecture/diagram can still return a valid diagram JSON
    even when the AI service does not provide a response.
    """
    title = architecture_name or application_name or "Architecture"
    desc_parts = [
        f"Fallback diagram for {title}.",
    ]
    if overview:
        desc_parts.append("Overview: " + overview[:200])
    if tech_summary:
        desc_parts.append("Tech: " + tech_summary.replace("\n", "; ")[:200])
    if features:
        desc_parts.append("Features: " + ", ".join(features)[:200])

    description = " ".join(desc_parts)

    shapes: List[Dict[str, Any]] = []

    # Client
    shapes.append({
        "tool": "generic",
        "x": 80,
        "y": 140,
        "id": "node-client",
        "service": "Client",
        "width": 140,
        "height": 70,
        "rotation": 0,
    })

    # API / App
    shapes.append({
        "tool": "generic",
        "x": 320,
        "y": 120,
        "id": "node-api",
        "service": "API / App",
        "width": 180,
        "height": 80,
        "rotation": 0,
    })

    # Database
    shapes.append({
        "tool": "generic",
        "x": 620,
        "y": 120,
        "id": "node-db",
        "service": "Database",
        "width": 180,
        "height": 80,
        "rotation": 0,
    })

    # Optional background service
    shapes.append({
        "tool": "generic",
        "x": 320,
        "y": 260,
        "id": "node-bg",
        "service": "Background Jobs",
        "width": 200,
        "height": 80,
        "rotation": 0,
    })

    # Arrows
    shapes.append({
        "tool": "arrow",
        "x": 0,
        "y": 0,
        "id": "arrow-client-api",
        "arrowType": "single",
        "points": [220, 175, 320, 160],
        "stroke": "#333333",
        "strokeWidth": 2,
        "dash": [],
    })

    shapes.append({
        "tool": "arrow",
        "x": 0,
        "y": 0,
        "id": "arrow-api-db",
        "arrowType": "single",
        "points": [500, 160, 620, 160],
        "stroke": "#333333",
        "strokeWidth": 2,
        "dash": [],
    })

    shapes.append({
        "tool": "arrow",
        "x": 0,
        "y": 0,
        "id": "arrow-api-bg",
        "arrowType": "single",
        "points": [410, 200, 410, 260],
        "stroke": "#999999",
        "strokeWidth": 2,
        "dash": [6, 4],
    })

    return {
        "name": f"{title} - Fallback Diagram",
        "description": description,
        "shapes": shapes,
    }


def _normalize_shape_tool_and_service(shape: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure each shape has a consistent tool and service.

    - If tool is missing for non-arrow nodes, infer from service name or default to "generic".
    - For arrow shapes, ensure tool="arrow" and allow service to be null/omitted.
    - For known AWS/Azure/GCP icon ids (e.g., "aws-dynamodb"), set the provider tool automatically.
    """
    tool = shape.get("tool")
    service = (shape.get("service") or "").strip()

    # Normalize arrows first
    if tool == "arrow" or shape.get("arrowType") or shape.get("points"):
        shape["tool"] = "arrow"
        return shape

    # Basic geometric tools and uploaded images are passed through unchanged
    basic_shapes = {"rect", "ellipse", "circle", "triangle", "diamond", "parallelogram", "star", "polygon", "line"}
    if tool in basic_shapes or tool == "uploaded-image":
        return shape

    # If service is empty but id looks like an icon key, propagate it
    normalized_service = service
    if not normalized_service and isinstance(shape.get("id"), str):
        sid = shape["id"]
        if "-" in sid:
            normalized_service = sid

    lower_service = normalized_service.lower()

    # Explicit overrides for cloud-infra and security service ids
    service_tool_overrides: Dict[str, str] = {
        # Cloud infrastructure containers
        "aws-account": "cloud-infra",
        "aws-region": "cloud-infra",
        "aws-vpc": "cloud-infra",
        "aws-availability-zone": "cloud-infra",
        "aws-subnet": "cloud-infra",
        "azure-subscription": "cloud-infra",
        "azure-management-group": "cloud-infra",
        "azure-resource-group": "cloud-infra",
        "azure-region": "cloud-infra",
        "azure-vnet": "cloud-infra",
        "gcp-organization": "cloud-infra",
        "gcp-folder": "cloud-infra",
        "gcp-project": "cloud-infra",
        "gcp-region": "cloud-infra",
        "gcp-zone": "cloud-infra",
        "gcp-vpc": "cloud-infra",
        # Security services
        "security-auth0": "security",
        "security-ping": "security",
        "security-okta": "security",
        "security-azuread": "security",
    }

    if lower_service in service_tool_overrides and (not tool or tool == "generic"):
        shape["tool"] = service_tool_overrides[lower_service]
        shape["service"] = normalized_service or shape.get("service") or lower_service
        return shape

    # Infer cloud provider from service/icon id if tool is missing or generic
    inferred_tool = None
    if lower_service.startswith("aws-"):
        inferred_tool = "aws"
    elif lower_service.startswith("azure-"):
        inferred_tool = "azure"
    elif lower_service.startswith("gcp-") or lower_service.startswith("google-"):
        inferred_tool = "gcp"
    elif lower_service.startswith("security-") or "security" in lower_service:
        inferred_tool = "security"

    if not tool or tool == "generic":
        if inferred_tool:
            shape["tool"] = inferred_tool
        else:
            shape["tool"] = "generic"

    # Ensure service is set to the canonical id when we inferred from it
    if normalized_service and not service:
        shape["service"] = normalized_service

    return shape


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
        recommendations_container=COSMOS_RECOMMENDATIONS_CONTAINER,
        designs_container=COSMOS_DESIGNS_CONTAINER
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
        "technology",
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
            logger.info(f"Recommendation keys: {list(existing_recommendation.keys())}")
            
            # Extract architectures from the stored recommendation
            # The document stores architectures in 'architectureRecommendations' field
            architectures_data = existing_recommendation.get("architectureRecommendations", [])
            architectures = [Architecture(**arch) for arch in architectures_data] if architectures_data else None
            
            # Extract features - try multiple locations
            features = []
            
            # Try requestMetadata.features first
            request_metadata = existing_recommendation.get("requestMetadata", {})
            if request_metadata and request_metadata.get("features"):
                features = request_metadata.get("features", [])
                logger.info(f"Found features in requestMetadata: {features}")
            
            # Fallback: try featuresList at root level
            if not features and existing_recommendation.get("featuresList"):
                features = existing_recommendation.get("featuresList", [])
                logger.info(f"Found features in featuresList: {features}")
            
            # Fallback: try features at root level
            if not features and existing_recommendation.get("features"):
                features = existing_recommendation.get("features", [])
                logger.info(f"Found features at root: {features}")
            
            # Final fallback: use features from the incoming request
            if not features:
                features = request.featuresList if request.featuresList else (
                    [f.strip() for f in request.features.split(',') if f.strip()] if request.features else []
                )
                logger.info(f"Using features from request: {features}")
            
            # Ensure features is a list
            if isinstance(features, str):
                features = [f.strip() for f in features.split(',') if f.strip()]
            
            logger.info(f"Final features list: {features}")
            
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
    """Get architecture recommendations directly from OpenAI without queuing."""
    try:
        # Use userId from request if provided, otherwise use authenticated user
        user_id = request.userId or current_user.user_id

        logger.info(
            f"Processing direct architecture recommendation for user {user_id}, app {request.applicationName}"
        )

        # Check if OpenAI is configured
        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API is not configured",
            )

        # Construct detailed architecture prompt
        prompt = construct_architecture_prompt(request)

        # Call OpenAI API
        try:
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_ENDPOINT,
            )

            logger.info("Calling OpenAI API for architecture recommendations...")

            response = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert software architect with deep knowledge of cloud-native architectures, microservices,
serverless patterns, and modern application design. Your role is to analyze application requirements and provide detailed,
practical architecture recommendations tailored to the specific needs, constraints, and scale of each project.""",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )

            # Parse OpenAI response (use parsed when available for json_object)
            message = response.choices[0].message
            if hasattr(message, "parsed") and message.parsed is not None:
                architectures_json = message.parsed
            else:
                if not message.content:
                    raise HTTPException(
                        status_code=502,
                        detail="AI service returned empty response for architecture recommendations.",
                    )
                architectures_json = json.loads(message.content)
            logger.info(
                f"Received OpenAI response: {json.dumps(architectures_json, indent=2)[:500]}..."
            )

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
                    "integrations": request.integrationsList or request.integrations,
                },
            }

            await db.create_recommendation(recommendation_data)
            logger.info(f"Stored recommendation for app {request.applicationName}")

            # Extract features list for response
            features = request.featuresList if request.featuresList else (
                [f.strip() for f in request.features.split(",") if f.strip()]
                if request.features
                else []
            )

            return ArchitectureRecommendationResponse(
                success=True,
                message="Architecture recommendations generated successfully",
                tenantId=request.tenantId,
                sessionId=request.sessionId,
                architectures=architectures,
                features=features,
            )

        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse OpenAI response as JSON: {str(je)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse OpenAI response",
            )
        except Exception as openai_error:
            logger.error(f"OpenAI API error: {str(openai_error)}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API error: {str(openai_error)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing direct architecture recommendation: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process architecture recommendation: {str(e)}",
        )


@app.post("/api/lld/recommendations", response_model=LLDResponse)
async def get_lld_recommendations(
    request: LLDRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Get LLD recommendations - checks for existing LLD first,
    otherwise queues request for asynchronous processing
    
    This endpoint first checks if an LLD already exists for the application.
    If found, returns it immediately. Otherwise, queues the request for processing.
    """
    try:
        # Use userId from request if provided, otherwise use authenticated user
        user_id = request.userId or current_user.user_id
        
        # Extract architecture ID for lookup if provided
        arch_obj = request.architecture if isinstance(request.architecture, dict) else {}
        architecture_id = arch_obj.get('id', '')
        
        # Check if LLD already exists for this application/architecture
        if architecture_id:
            try:
                logger.info(f"Looking for existing LLD with architectureId={architecture_id}, tenantId={request.tenantId}, applicationName={request.applicationName}")
                
                # Debug: List all designs for this tenant to see what's available
                debug_designs = await db.list_designs_debug(tenant_id=request.tenantId, limit=10)
                logger.info(f"Debug: Found {len(debug_designs)} total designs for tenant {request.tenantId}")
                
                existing_lld = await db.get_design_by_architecture(
                    architecture_id=architecture_id,
                    tenant_id=request.tenantId,
                    application_name=request.applicationName
                )
                
                if existing_lld:
                    logger.info(f"Found existing LLD: designId={existing_lld.get('designId')}, type={existing_lld.get('type')}")
                    
                    # Verify it's a lowLevelDesign type
                    if existing_lld.get('type') == 'lowLevelDesign':
                        logger.info(f"Returning existing LLD for user {user_id}, architecture {architecture_id}")
                        
                        # Extract LLD response data
                        lld_data = existing_lld.get('lldResponse', {})
                        if not lld_data:
                            logger.warning(f"LLD document found but lldResponse is empty or missing")
                            lld_data = {"error": "LLD data is empty"}
                        
                        # Extract features from multiple possible locations
                        features = []
                        if lld_data.get('features'):
                            features = [f.get('name', 'Unknown') for f in lld_data.get('features', [])]
                        elif existing_lld.get('generatedFeatures'):
                            features = existing_lld.get('generatedFeatures', [])
                        elif existing_lld.get('features'):
                            features = existing_lld.get('features', [])
                        
                        logger.info(f"Extracted {len(features)} features from LLD: {features}")
                        
                        return LLDResponse(
                            success=True,
                            designId=existing_lld.get('designId'),
                            featureCount=len(features),
                            lld=lld_data,
                            error=None
                        )
                    else:
                        logger.warning(f"Found design but type is '{existing_lld.get('type')}', not 'lowLevelDesign'")
                else:
                    logger.info(f"No LLD found for architectureId={architecture_id}")
                    
                    # Try flexible fallback query without applicationName requirement
                    logger.info(f"Trying flexible fallback query without applicationName requirement")
                    existing_lld = await db.get_design_by_architecture_flexible(
                        architecture_id=architecture_id,
                        tenant_id=request.tenantId
                    )
                    
                    if existing_lld and existing_lld.get('type') == 'lowLevelDesign':
                        logger.info(f"Found LLD via flexible query: designId={existing_lld.get('designId')}")
                        
                        # Extract LLD response data
                        lld_data = existing_lld.get('lldResponse', {})
                        if not lld_data:
                            logger.warning(f"LLD document found via flexible query but lldResponse is empty")
                            lld_data = {"error": "LLD data is empty"}
                        
                        # Extract features from multiple possible locations
                        features = []
                        if lld_data.get('features'):
                            features = [f.get('name', 'Unknown') for f in lld_data.get('features', [])]
                        elif existing_lld.get('generatedFeatures'):
                            features = existing_lld.get('generatedFeatures', [])
                        elif existing_lld.get('features'):
                            features = existing_lld.get('features', [])
                        
                        logger.info(f"Extracted {len(features)} features from flexible LLD query: {features}")
                        
                        return LLDResponse(
                            success=True,
                            designId=existing_lld.get('designId'),
                            featureCount=len(features),
                            lld=lld_data,
                            error=None
                        )
                    else:
                        logger.info(f"No LLD found via flexible query either")
                    
            except Exception as e:
                logger.error(f"Error checking for existing LLD: {str(e)}", exc_info=True)
        
        # No existing LLD found - queue for processing
        logger.info(f"No existing LLD found for user {user_id}, app {request.applicationName}. Queuing request.")
        
        # Validate Service Bus client is available
        if not service_bus_client:
            logger.error("Service Bus client not initialized")
            raise HTTPException(
                status_code=500,
                detail="Service Bus is not configured"
            )
        
        # Send message to Service Bus with proper error handling
        try:
            sender = service_bus_client.get_queue_sender(queue_name=AZURE_SERVICE_BUS_DESIGN_QUEUE)
            async with sender:
                # Create message payload with all fields + resolved userId
                message_data = request.model_dump()
                message_data["userId"] = user_id
                message_data["requestType"] = "lld"  # Add request type for queue processing
                
                logger.info(f"LLD Service Bus message payload keys: {list(message_data.keys())}")
                
                message = ServiceBusMessage(
                    body=json.dumps(message_data),
                    content_type="application/json"
                )
                await sender.send_messages(message)
        except Exception as sb_error:
            logger.error(f"Service Bus error for LLD: {str(sb_error)}", exc_info=True)
            # Return error response with details
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "SERVICE_BUS_UNAVAILABLE",
                    "message": f"Failed to queue LLD request to Service Bus. The queue '{AZURE_SERVICE_BUS_DESIGN_QUEUE}' may not exist or connection is misconfigured.",
                    "details": str(sb_error)
                }
            )
        
        # Return success response immediately
        return LLDResponse(
            success=True,
            designId=None,
            featureCount=None,
            lld=None,
            error="Request queued successfully for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing LLD recommendation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process LLD recommendation request: {str(e)}"
        )


@app.post("/api/code/recommendations", response_model=CodeGenerationResponse)
async def get_code_recommendations(
    request: CodeGenerationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Get code generation recommendations - checks for existing generated code first,
    otherwise queues request for asynchronous processing
    
    This endpoint first checks if generated code already exists for the design.
    If found, returns it immediately. Otherwise, queues the request for processing.
    """
    try:
        # Use userId from request if provided, otherwise use authenticated user
        user_id = request.userId or current_user.user_id
        
        # Check if generated code already exists for this design
        if request.designId:
            try:
                logger.info(f"Looking for existing code with designId={request.designId}, tenantId={request.tenantId}")
                
                existing_code = await db.get_generated_code_by_design_id(
                    design_id=request.designId,
                    tenant_id=request.tenantId
                )
                
                if existing_code:
                    logger.info(f"Found existing code for user {user_id}, design {request.designId}")
                    
                    # Extract generated files and metadata from stored code
                    generated_files = existing_code.get('generatedFiles', [])
                    categorized_files = existing_code.get('categorizedFiles', {})
                    project_structure = existing_code.get('projectStructure', {})
                    technologies = existing_code.get('technologies', [])
                    instructions = existing_code.get('instructions', '')
                    
                    # Convert to proper models
                    files = [GeneratedFile(path=f.get('path', ''), content=f.get('content', '')) for f in generated_files]
                    
                    logger.info(f"Returning existing generated code with {len(files)} files for user {user_id}")
                    
                    return CodeGenerationResponse(
                        success=True,
                        message=f"Retrieved existing generated code with {len(files)} files",
                        projectName=project_structure.get('projectName', ''),
                        projectStructure=ProjectStructure(**project_structure) if project_structure else None,
                        categorizedFiles=CategorizedFiles(**categorized_files) if categorized_files else None,
                        generatedFiles=files,
                        fileCount=len(files),
                        technologies=technologies,
                        instructions=instructions,
                        error=None
                    )
                else:
                    logger.info(f"No generated code found for designId={request.designId}, tenantId={request.tenantId}")
            except Exception as e:
                logger.error(f"Error checking for existing generated code: {str(e)}", exc_info=True)
        
        # No existing generated code found - queue for processing
        logger.info(f"No existing code found for user {user_id}, design {request.designId}. Queuing request.")
        
        # Validate Service Bus client is available
        if not service_bus_client:
            logger.error("Service Bus client not initialized")
            raise HTTPException(
                status_code=500,
                detail="Service Bus is not configured"
            )
        
        # Send message to Service Bus with proper error handling
        try:
            sender = service_bus_client.get_queue_sender(queue_name=AZURE_SERVICE_BUS_CODE_QUEUE)
            async with sender:
                # Create message payload with all fields + resolved userId
                message_data = request.model_dump()
                message_data["userId"] = user_id
                message_data["requestType"] = "code"  # Add request type for queue processing
                
                logger.info(f"Code generation Service Bus message payload keys: {list(message_data.keys())}")
                
                message = ServiceBusMessage(
                    body=json.dumps(message_data),
                    content_type="application/json"
                )
                await sender.send_messages(message)
        except Exception as sb_error:
            logger.error(f"Service Bus error for code generation: {str(sb_error)}", exc_info=True)
            # Return error response with details
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "SERVICE_BUS_UNAVAILABLE",
                    "message": f"Failed to queue code generation request to Service Bus. The queue '{AZURE_SERVICE_BUS_CODE_QUEUE}' may not exist or connection is misconfigured.",
                    "details": str(sb_error)
                }
            )
        
        # Return success response immediately
        return CodeGenerationResponse(
            success=True,
            message="Code generation request queued successfully for processing",
            projectName="",
            projectStructure=None,
            categorizedFiles=None,
            generatedFiles=None,
            fileCount=None,
            technologies=None,
            instructions=None,
            error="Request queued successfully for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing code generation recommendation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process code generation recommendation request: {str(e)}"
        )


@app.post("/api/architecture/diagram", response_model=ArchitectureDiagramResponse)
async def generate_architecture_diagram(
    request: ArchitectureDiagramRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """Generate an architecture diagram JSON from architecture summary and details."""
    try:
        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API is not configured",
            )

        tenant_id = request.tenantId or "default"
        application_name = request.applicationName
        session_id = request.sessionId or "default"

        arch_obj = request.architecture or {}
        arch_name = arch_obj.get("name", application_name)
        arch_description = arch_obj.get("description", "")
        tech_stack = arch_obj.get("technologyStack", {}) if isinstance(arch_obj, dict) else {}
        infra = tech_stack.get("infra", {}) if isinstance(tech_stack, dict) else {}

        # Build a compact technology summary for the diagram prompt
        tech_summary_parts: List[str] = []
        if tech_stack.get("cloudProvider"):
            tech_summary_parts.append(f"Cloud: {tech_stack.get('cloudProvider')}")
        if infra.get("compute"):
            tech_summary_parts.append(f"Compute: {infra.get('compute')}")
        if infra.get("database"):
            tech_summary_parts.append(f"Database: {infra.get('database')}")
        if infra.get("messaging"):
            tech_summary_parts.append(f"Messaging: {infra.get('messaging')}")
        if infra.get("storage"):
            tech_summary_parts.append(f"Storage: {infra.get('storage')}")
        if infra.get("apiGateway"):
            tech_summary_parts.append(f"API Gateway: {infra.get('apiGateway')}")
        if infra.get("authentication"):
            tech_summary_parts.append(f"Auth: {infra.get('authentication')}")
        if infra.get("security"):
            tech_summary_parts.append(f"Security: {infra.get('security')}")

        tech_summary = (
            "\n".join(tech_summary_parts)
            if tech_summary_parts
            else "No detailed technology stack provided. Use generic components only."
        )
        features_text = (
            "\n".join(f"- {f}" for f in (request.features or []))
            if request.features
            else "Not explicitly listed. Infer 3-7 main components from the overview."
        )

        system_prompt = (
            "You are an expert cloud solution architect. "
            "Given an application overview and optional architecture details, design a clear, logical cloud architecture diagram. "
            "Return ONLY JSON matching the requested schema and the allowed tool/service values used by the canvas."
        )

        user_prompt = f"""Generate a JSON representation of an architecture diagram for this application.

Application Name: {application_name}
Architecture Name: {arch_name}

Overview / Summary:
{request.overview}

Additional Architecture Description:
{arch_description}

Technology Summary:
{tech_summary}

Key features / responsibilities to represent as components:
{features_text}

Requirements for the diagram:
- Use a small number of high-level components (typically 6-20 nodes).
- Use the following shape schema for each node or arrow:
        - tool: one of [
                "rect", "ellipse", "circle", "triangle", "diamond", "parallelogram", "star", "polygon", "line",
                "aws", "azure", "gcp", "security", "cloud-infra", "generic", "uploaded-image", "component", "arrow"
            ].
        - x, y: integers for positioning (grid 0-1200 for x, 0-800 for y).
        - id: stable unique id like "node-1", "db-1", "arrow-1".
        - service: for cloud & service tools (aws, azure, gcp, security, cloud-infra) use a short canonical id in the
            form "aws-...", "azure-...", "gcp-...", "security-..." or cloud-infra ids like "aws-vpc", "azure-vnet",
            "gcp-project". For generic boxes, use a short human-readable label like "Web App" or "API".
        - For non-arrow components you MAY set width and height (defaults: 160 x 80) and rotation.
    - For arrows (tool = "arrow"), set points as [x1, y1, x2, y2], stroke color, strokeWidth, and optional dash pattern.

Layout guidelines:
- Place entrypoints (clients, front-end) on the left or top.
- Place core services and APIs in the center.
- Place databases, storage, and analytics on the right or bottom.
- Use arrows to indicate main request/response and data flows.

Return JSON with EXACTLY this top-level structure:
{{
  "name": "short diagram name",
  "description": "1-2 sentence description of what the diagram shows",
  "shapes": [
    {{
      "tool": "azure",
      "x": 100,
      "y": 100,
      "id": "node-1",
      "service": "Web App",
      "width": 160,
      "height": 80,
      "rotation": 0
    }},
    {{
      "tool": "arrow",
      "x": 0,
      "y": 0,
      "id": "arrow-1",
      "arrowType": "single",
      "points": [180, 140, 380, 140],
      "stroke": "#333333",
      "strokeWidth": 2,
      "dash": [5, 5]
    }}
  ]
}}

IMPORTANT:
- Only use fields defined in the schema above.
- Ensure all ids are unique.
- Ensure coordinates place components in a readable left-to-right or top-to-bottom flow.
- If cloud provider is unknown, use tool = "generic" for nodes.
"""

        # Call OpenAI; if it fails, return an explicit error instead of a hardcoded fallback
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
            completion = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                # Use the higher MAX_OUTPUT_TOKENS budget so reasoning models have room for both reasoning and output
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            )

            message = completion.choices[0].message
            # Prefer parsed JSON when using response_format=json_object
            if hasattr(message, "parsed") and message.parsed is not None:
                diagram_json = message.parsed
            else:
                content = (message.content or "").strip()
                if not content:
                    logger.error(
                        "OpenAI returned empty content for architecture diagram. Completion object: %s",
                        completion,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="AI service returned empty response for architecture diagram.",
                    )
                try:
                    diagram_json = json.loads(content)
                except json.JSONDecodeError as je:
                    logger.error(
                        "Failed to parse architecture diagram JSON from AI (%s). Content: %s",
                        str(je),
                        content,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="AI service returned invalid JSON for architecture diagram.",
                    )
        except HTTPException:
            # Let HTTPExceptions bubble up directly
            raise
        except Exception as e:
            logger.error(
                "Error generating architecture diagram via OpenAI (%s).",
                str(e),
                exc_info=True,
            )
            raise HTTPException(
                status_code=502,
                detail="AI service failed while generating architecture diagram.",
            )

        # Normalize tool/service pairs on all shapes before validation
        shapes = diagram_json.get("shapes") or []
        normalized_shapes: List[Dict[str, Any]] = []
        for raw_shape in shapes:
            if isinstance(raw_shape, dict):
                normalized_shapes.append(_normalize_shape_tool_and_service(raw_shape))
            else:
                normalized_shapes.append(raw_shape)
        diagram_json["shapes"] = normalized_shapes

        # Validate and normalize diagram structure
        try:
            diagram = ArchitectureDiagram(**diagram_json)
        except Exception as e:
            logger.error(
                "Diagram JSON did not match expected schema: %s | JSON: %s",
                str(e),
                diagram_json,
            )
            raise HTTPException(
                status_code=502,
                detail="Diagram JSON has invalid structure",
            )

        diagram_id = str(uuid4())

        # Optionally store in Cosmos designs container
        try:
            document = {
                "id": diagram_id,
                "tenantId": tenant_id,
                "applicationName": application_name,
                "sessionId": session_id,
                "userId": current_user.user_id,
                "type": "architectureDiagram",
                "architecture": arch_obj,
                "overview": request.overview[:4000],
                "features": request.features or [],
                "diagram": diagram_json,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await db.create_design(document)
        except Exception as e:
            logger.error(f"Failed to store architecture diagram in Cosmos DB: {str(e)}")

        return ArchitectureDiagramResponse(
            success=True,
            diagramId=diagram_id,
            diagram=diagram,
        )

    except HTTPException:
        # Re-raise HTTP errors so FastAPI returns proper status codes
        raise
    except Exception as e:
        logger.error(f"Error generating architecture diagram: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate architecture diagram: {str(e)}",
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
- Preferred Cloud Platform: {request.cloudPlatform or 'No preference - recommend best fit'}
- Source Control Preference: {request.sourceControl or 'No preference'}

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

1. A unique identifier (id) in the format "arch-{{uuid}}" using a random 8-character UUID
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

12. DETAILED COST BREAKDOWN & CAPACITY PLANNING (CRITICAL - Be extremely thorough):
    
    A. CAPACITY PLANNING (Based on {nfrs.users or 'expected'} users):
       - Expected Users: {nfrs.users or 'analyze requirements to estimate'}
       - Peak Concurrent Users: Calculate based on typical usage patterns (e.g., 5-15% of total users)
       - Requests Per Second: Estimate based on user behavior and feature usage
       - Data Growth Per Month: Project based on user count, data types, and retention policies
       - Compute Units Required: Specify cores, memory, number of instances/functions
       - Storage Capacity: Calculate based on data profile and user count (include DB + object storage + backups)
       - Bandwidth Requirements: Estimate ingress/egress based on data transfer patterns
       - Scaling Strategy: How capacity grows as users increase (linear, logarithmic, step-function)
    
    B. SERVICE-LEVEL COST BREAKDOWN:
       For EACH infrastructure service you recommend, provide:
       - Service name (e.g., "Compute - AWS Lambda", "Database - DynamoDB")
       - Pricing model: Pay-as-you-go, Reserved Instances, Committed Use Discounts, Spot Instances
       - Monthly cost range [min, max] with clear assumptions:
         * Min: Low usage scenario (expected normal load)
         * Max: High usage scenario (peak load, growth buffer)
       - Cost calculation details showing:
         * Units consumed (e.g., "10M Lambda invocations", "500GB DynamoDB storage", "50GB data transfer")
         * Unit pricing (e.g., "$0.20 per 1M requests")
         * Regional pricing considerations
         * Free tier applicability
    
    C. TOTAL COST ANALYSIS:
       - Total monthly cost range [min, max] (sum of all services)
       - Cost per user (total / user count) for scaling projections
       - Cost assumptions clearly stated:
         * Based on {nfrs.users or 'X'} users
         * Request patterns (read/write ratio, API calls per user)
         * Data retention period
         * High availability requirements
         * Regional deployment (single region vs multi-region)
       
    D. COST OPTIMIZATION TIPS:
       - Reserved capacity savings (e.g., "30-50% savings with 1-year reserved instances")
       - Committed use discounts implications
       - Spot instances for non-critical workloads
       - Auto-scaling optimization to match demand curves
       - Data lifecycle policies (archive cold data)
       - CDN caching to reduce origin requests
       - Query optimization to reduce database costs
       - Right-sizing recommendations
    
    E. COST SCALING PROJECTIONS:
       - At 2x users: Estimated cost increase
       - At 5x users: How costs scale
       - At 10x users: Architecture changes needed
       - Cost efficiency metrics (cost per 1000 users, cost per transaction)
    
    IMPORTANT: Do NOT provide generic costs. Calculate actual costs based on:
    - The SPECIFIC services you're recommending
    - The ACTUAL user count ({nfrs.users or 'provided in requirements'})
    - The REAL feature set and workload patterns
    - Current cloud provider pricing (2026 pricing)

13. Technology Stack: ONLY include services needed for the listed requirements:
    - Languages: Chosen specifically for the features (explain why in your description)
    - Frameworks: Must match {request.applicationType or 'the application type'} and listed features
    - Runtime: Optimized for the workload type
    - Cloud provider: {'Prefer ' + request.cloudPlatform + ' if suitable, but ' if request.cloudPlatform else ''}AWS, Azure, or GCP - choose based on {industry_str} needs and feature requirements
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
    - CI/CD: {'Consider ' + request.sourceControl + ' for source control if compatible. ' if request.sourceControl else ''}Specific tools appropriate for this stack and cloud platform (standard DevOps needs)
    
    IMPORTANT: If a component is not needed for the listed features, DO NOT include it or leave it empty/blank.
    
14. bestFor: 3-5 scenarios that specifically match {industry_str} {core_use_cases_str} with these features
15. avoidWhen: 3-5 scenarios where this would NOT work (be specific to the industry and use case)

Return the response as a valid JSON object with the following structure:
{{
  "architectures": [
    {{
      "id": "arch-{{8_char_uuid}}",
      "icon": "icon-name",
      "name": "Architecture Pattern Name tailored to {industry_str} {core_use_cases_str}",
      "description": "Detailed description explaining HOW each feature is implemented...",
      "ranking": 9,
      "shortPros": "{industry_str}-specific Pro 1. {core_use_cases_str}-specific Pro 2. Pro 3.",
      "shortCons": "Context-specific Con 1. Con 2. Con 3.",
      "recommendationReason": "Why this fits {industry_str} + {core_use_cases_str} + these features...",
      "whyChoose": "Choose this when working with {core_use_cases_str} in {industry_str}...",
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
        "languages": "Languages chosen for {request.applicationType or 'the application'}",
        "frameworks": "Frameworks matching {request.applicationType or 'the application type'}",
        "runtime": "Runtime optimized for the workload",
        "cloudProvider": "Chosen cloud provider based on requirements",
        "infra": {{
          "compute": "ONLY if needed: specific compute services",
          "database": "ONLY if needed: specific database services for {data_types_str}",
          "cache": "ONLY if needed: caching solution",
          "messaging": "ONLY if needed: messaging services",
          "storage": "ONLY if needed: storage for {request.dataProfile or 'data needs'}",
          "apiGateway": "ONLY if needed: API management",
          "authentication": "ONLY if {user_roles_str} require it",
          "security": "ONLY services addressing listed security requirements",
          "networking": "ONLY if scale demands it",
          "monitoring": "ONLY if essential",
          "logging": "ONLY if essential"
        }},
        "cicd": {{
          "pipeline": "CI/CD tools appropriate for this stack",
          "containerization": "Container tools if architecture uses containers",
          "testing": "Testing frameworks for chosen languages",
          "iac": "IaC tools matching cloud provider"
        }}
      }},
      "costBreakdown": {{
        "serviceCosts": [
          {{
            "service": "Compute Service Name",
            "pricingModel": "Pay-as-you-go with option for Reserved/Committed",
            "monthlyCost": [150, 400],
            "details": "Based on {nfrs.users or 'X'} users: Estimated Y compute units @ $Z per unit. Min assumes typical load, Max includes 2x peak capacity buffer."
          }},
          {{
            "service": "Database Service Name",
            "pricingModel": "Pay-as-you-go with reserved capacity discounts",
            "monthlyCost": [200, 600],
            "details": "Storage: XGB @ $Y/GB + operations: Z requests/sec @ $A per million. Includes replication and backups."
          }},
          {{
            "service": "Additional Service",
            "pricingModel": "Pricing model",
            "monthlyCost": [50, 150],
            "details": "Specific calculation based on usage patterns"
          }}
        ],
        "totalMonthly": [500, 2000],
        "assumptions": "Cost calculated for {nfrs.users or 'X'} users with Y daily active users, Z requests/user/day, A GB storage growth/month. Single region deployment. Standard support tier. Assumes 70% steady load, 30% peak coverage.",
        "optimizationTips": "1) Reserved instances can reduce compute costs by 30-50% for predictable workloads. 2) Implement aggressive caching to reduce database queries by 40-60%. 3) Use lifecycle policies to archive old data reducing storage costs by 20-30%. 4) Right-size compute after monitoring actual usage patterns.",
        "capacityPlanning": {{
          "expectedUsers": "{nfrs.users or 'X users based on requirements'}",
          "peakConcurrentUsers": "Y concurrent users (Z% of total during peak hours)",
          "requestsPerSecond": "A-B req/s (avg-peak) based on feature usage patterns",
          "dataGrowthPerMonth": "C GB/month (D MB per active user)",
          "computeUnits": "E instances/functions with F cores, G GB RAM each",
          "storageCapacity": "H GB database + I GB object storage + J GB backups",
          "bandwidthRequirements": "K GB/month ingress, L GB/month egress",
          "scalingStrategy": "Horizontal auto-scaling: add 1 compute unit per M additional concurrent users. Database scales via read replicas and sharding when N GB reached."
        }}
      }},
      "bestFor": [
        "{industry_str} organizations requiring {core_use_cases_str}",
        "Scenarios matching feature set: [specific feature combinations]",
        "Teams with expertise in [specific technologies recommended]",
        "When [specific requirement or constraint] is priority",
        "Organizations with [size/scale/budget] characteristics"
      ],
      "avoidWhen": [
        "Feature set doesn't include [missing capability]",
        "Team lacks expertise in [technology]",
        "Budget constraints under [threshold]",
        "{industry_str} specific reason",
        "{core_use_cases_str} specific limitation"
      ]
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
9. Each service in your technology stack must directly map to a specific requirement or feature listed above.
10. COST ANALYSIS IS MANDATORY: Every architecture MUST include detailed costBreakdown with:
    - Service-level costs with pricing models and calculation details
    - Capacity planning based on actual user count ({nfrs.users or 'analyze requirements'})
    - Total cost range with clear assumptions
    - Cost optimization recommendations
    - Scaling projections (2x, 5x, 10x users)
11. DO NOT use hardcoded examples or generic costs - calculate based on ACTUAL requirements
12. For capacity planning, consider: concurrent users (typically 5-15% of total), requests per user, data growth, compute needs, storage requirements, bandwidth
13. Pricing models should specify: Pay-as-you-go baseline, Reserved/Committed savings potential, Spot/preemptible options where applicable
14. Cost assumptions must be explicit: user count basis, usage patterns, regional deployment, support tier, retention policies"""

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
            arch_data.setdefault("id", f"arch-{str(uuid4())[:8]}")
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
            
            # Create Architecture object - field validators will clean metrics automatically
            architecture = Architecture(**arch_data)
            architectures.append(architecture)
            
        except Exception as e:
            logger.error(f"Error parsing architecture {idx}: {str(e)}", exc_info=True)
            continue
    
    return architectures


@app.post("/api/generate-lld", response_model=LLDResponse)
async def generate_lld(
    request: LLDRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """
    HTTP endpoint to generate a Low Level Design (LLD) using OpenAI.
    Simply passes request data to OpenAI and returns the LLD JSON.
    """
    logger.info('Generate LLD API triggered')

    try:
        # Extract fields with fallbacks
        tenant_id = request.tenantId or 'default'
        application_name = request.applicationName or 'Application'
        session_id = request.sessionId or 'default'
        user_id = request.userId or current_user.user_id or 'default'
        requirements = request.requirements or request.overview or ''
        architecture = request.architecture or {}
        features = request.features or []

        # Extract architectureId for lookup
        arch_obj = architecture if isinstance(architecture, dict) else {}
        architecture_id = arch_obj.get('id', '')

        logger.info(f'Generating LLD - App: {application_name}, ArchitectureId: {architecture_id}, Features: {len(features)}')

        # Check if LLD already exists for this architectureId and tenantId
        if architecture_id:
            existing_lld = await db.get_design_by_architecture(
                architecture_id=architecture_id,
                tenant_id=tenant_id,
                application_name=application_name
            )
            
            if existing_lld:
                logger.info(f'Found existing LLD for architectureId: {architecture_id}, tenantId: {tenant_id}')
                lld_response = existing_lld.get('lldResponse', {})
                
                # Log retrieved data structure for debugging
                logger.info(f'Retrieved LLD has {len(lld_response.get("features", []))} features')
                logger.info(f'LLD has baseFramework: {"baseFramework" in lld_response}')
                logger.info(f'LLD has header: {"header" in lld_response}')
                logger.info(f'LLD document keys: {list(existing_lld.keys())}')
                logger.info(f'Full lldResponse size: {len(json.dumps(lld_response))} bytes')
                
                # Validate LLD structure
                if not lld_response:
                    logger.warning('Retrieved lldResponse is empty, regenerating...')
                elif 'features' not in lld_response:
                    logger.warning('Retrieved lldResponse missing features field, may be incomplete')
                elif 'baseFramework' not in lld_response:
                    logger.warning('Retrieved lldResponse missing baseFramework field, may be incomplete')
                else:
                    # Return existing LLD only if it has the required structure
                    return LLDResponse(
                        success=True,
                        designId=existing_lld.get('id'),
                        featureCount=len(lld_response.get('features', [])),
                        lld=lld_response,
                        error=None
                    )

        logger.info(f'No existing LLD found, generating new LLD for architectureId: {architecture_id}')

        # Validate OpenAI configuration
        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="OpenAI is not configured"
            )

        # Extract details from architecture object
        arch_name = arch_obj.get('name', application_name)
        arch_description = arch_obj.get('description', '')
        tech_stack = arch_obj.get('technologyStack', {})
        infra = tech_stack.get('infra', {})
        
        # Extract tech stack details to pass to LLM
        cloud_provider = tech_stack.get('cloudProvider', '')
        compute = infra.get('compute', '')
        database = infra.get('database', '')
        cache = infra.get('cache', '')
        messaging = infra.get('messaging', '')
        storage = infra.get('storage', '')
        authentication = infra.get('authentication', '')
        languages = tech_stack.get('languages', '')
        frameworks = tech_stack.get('frameworks', '')
        
        # Build tech stack section for LLM context
        tech_stack_lines = []
        if cloud_provider:
            tech_stack_lines.append(f"- Cloud: {cloud_provider}")
        if compute:
            tech_stack_lines.append(f"- Compute: {compute}")
        if database:
            tech_stack_lines.append(f"- Database: {database}")
        if cache:
            tech_stack_lines.append(f"- Cache: {cache}")
        if messaging:
            tech_stack_lines.append(f"- Messaging: {messaging}")
        if storage:
            tech_stack_lines.append(f"- Storage: {storage}")
        if authentication:
            tech_stack_lines.append(f"- Auth: {authentication}")
        if languages:
            tech_stack_lines.append(f"- Languages: {languages}")
        if frameworks:
            tech_stack_lines.append(f"- Frameworks: {frameworks}")
        
        tech_stack_text = '\n'.join(tech_stack_lines) if tech_stack_lines else "No specific technology stack provided"
        
        # Format features if provided
        features_list = []
        if features and isinstance(features, list):
            for f in features:
                if isinstance(f, dict) and f.get('name'):
                    features_list.append(f.get('name'))
                elif f:
                    features_list.append(str(f))
        
        features_text = '\n'.join(f'{i+1}. {name}' for i, name in enumerate(features_list)) if features_list else "Features not specified - analyze the architecture description and extract/design appropriate features"
        
        # Single unified prompt - let LLM do all the work
        system_prompt = """Expert software architect. Generate production-ready low-level design with OWASP security, proper indexes, input validation, error handling, and performance optimization. Return ONLY valid JSON matching the exact structure specified."""
        
        user_prompt = f"""Generate a complete low-level design for the following architecture:

Architecture Name: {arch_name}

Architecture Description: {arch_description}

Application Name: {application_name}

Technology Stack:
{tech_stack_text}

Features:
{features_text}

CRITICAL REQUIREMENTS (Production-Ready):

1. SECURITY: Validate all inputs, JWT/OAuth auth, parameterized queries, encrypt PII/PHI, rate limiting, secure headers (HSTS, CSP)

2. DATABASE: Define indexes (name, columns, type, purpose) on PKs, FKs, queried columns; connection pooling; encrypt sensitive data

3. ERROR HANDLING: Error format with code/message/correlationId (no stack traces); handle nulls/boundaries/concurrency/network failures; retry logic, circuit breakers

4. PERFORMANCE: Multi-level caching, async I/O, connection pooling, load balancing, pagination

5. BEST PRACTICES: Layered architecture, SOLID principles, testing (unit/integration/E2E), API docs

Instructions: Use exact tech stack. Design features with validation, security, error handling, edge cases. Define indexes and optimizations. Create featureFlow.

Return valid JSON with this EXACT structure:

{{
  "baseFramework": {{
    "pattern": "architecture pattern name",
    "frontend": {{
      "framework": "frontend framework",
      "language": "programming language",
      "techStack": ["array of frontend technologies"],
      "components": {{
        "routing": "routing approach",
        "stateManagement": "state management",
        "lazyLoading": "lazy loading",
        "bundling": "bundling",
        "styling": "styling",
        "authentication": "auth handling",
        "errorHandling": "error handling",
        "caching": "caching",
        "performance": "optimization",
        "accessibility": "accessibility",
        "formValidation": "validation",
        "apiIntegration": "API client",
        "testingStrategy": "testing",
        "errorRecovery": "error recovery"
      }}
    }},
    "backend": {{
      "framework": "backend framework",
      "database": "database technology",
      "techStack": ["array of backend technologies"],
      "components": {{
        "apiPattern": "API pattern",
        "middleware": "middleware stack",
        "errorHandling": "error handling approach",
        "dataValidation": "validation approach",
        "authentication": "auth strategy",
        "authorization": "authorization model",
        "logging": "logging approach",
        "monitoring": "monitoring tools",
        "caching": "caching strategy",
        "dataAccess": "data access pattern",
        "securityHeaders": "security headers",
        "rateLimiting": "rate limiting",
        "inputValidation": "input validation",
        "errorRecovery": "error recovery",
        "concurrencyControl": "concurrency handling",
        "performanceOptimization": "performance approach",
        "testingStrategy": "testing approach",
        "codeOrganization": "code organization"
      }}
    }}
  }},
  "features": [
    {{
      "name": "feature name",
      "techStack": ["technologies for this feature"],
      "screens": [
        {{
          "name": "screen name",
          "route": "route path",
          "file": "file path",
          "apis": [
            {{
              "endpoint": "API endpoint",
              "method": "HTTP method",
              "description": "description",
              "authentication": "required/optional/none",
              "authorization": "required roles",
              "inputValidation": {{
                "requestBody": "validation rules",
                "pathParams": "path param validation",
                "headers": "required headers",
                "maxRequestSize": "max size"
              }},
              "security": {{
                "inputSanitization": "sanitization approach",
                "rateLimiting": "rate limit",
                "csrfProtection": "CSRF protection",
                "encryption": "encryption requirements"
              }},
              "tables": [
                {{
                  "name": "table name",
                  "type": "database type",
                  "operations": ["operations"],
                  "structure": {{
                    "columns": [
                      {{
                        "name": "column name",
                        "type": "data type",
                        "constraints": "constraints",
                        "encrypted": "true if PII"
                      }}
                    ],
                    "indexes": [
                      {{
                        "name": "index name",
                        "columns": ["indexed columns"],
                        "type": "PRIMARY/UNIQUE/COMPOSITE/BTREE",
                        "purpose": "purpose"
                      }}
                    ],
                    "partitionKey": "partition key or null",
                    "sortKey": "sort key or null",
                    "relationships": ["foreign keys"],
                    "accessControl": "access control"
                  }}
                }}
              ],
              "responseHandling": {{
                "successCodes": "200/201/204",
                "errorCodes": "400/401/403/404/429/500",
                "errorMessages": "user-friendly messages",
                "dataFiltering": "filter sensitive fields"
              }},
              "edgeCases": {{
                "nullHandling": "null/empty handling",
                "boundaryConditions": "min/max/empty",
                "concurrency": "race condition handling",
                "idempotency": "duplicate request handling",
                "timeouts": "timeout handling",
                "partialFailures": "partial failure handling"
              }},
              "errorHandling": {{
                "errorFormat": "error response structure",
                "retryStrategy": "retry logic",
                "fallbackBehavior": "fallback approach",
                "logging": "error logging",
                "monitoring": "error monitoring"
              }},
              "performance": {{
                "caching": "cache strategy",
                "pagination": "pagination approach",
                "queryOptimization": "query optimization",
                "responseTime": "target response time",
                "rateLimiting": "rate limit config"
              }},
              "handlerFile": "handler file path"
            }}
          ]
        }}
      ],
      "apis": [
        {{
          "endpoint": "endpoint",
          "method": "method",
          "authentication": "required/optional/none",
          "inputValidation": "validation",
          "security": "security",
          "tables": [{{"name": "table", "operations": ["operations"]}}]
        }}
      ],
      "dataModels": [
        {{
          "tableName": "table name",
          "type": "database type",
          "purpose": "purpose",
          "encryptionRequired": "true if PII/PHI",
          "structure": {{
            "columns": [
              {{
                "name": "column name",
                "type": "data type",
                "constraints": "constraints",
                "encrypted": "true/false",
                "pii": "true/false",
                "defaultValue": "default value",
                "nullable": "true/false"
              }}
            ],
            "indexes": [
              {{
                "name": "index name",
                "columns": ["columns"],
                "type": "index type",
                "purpose": "purpose",
                "cardinality": "high/medium/low"
              }}
            ],
            "relationships": ["foreign keys"],
            "partitionKey": "partition key or null",
            "sortKey": "sort key or null",
            "backupPolicy": "backup policy",
            "accessControl": "access control"
          }},
          "queryOptimization": "query optimization",
          "dataSecurity": "data security",
          "edgeCases": {{
            "concurrency": "concurrency handling",
            "dataIntegrity": "integrity constraints",
            "orphanRecords": "orphan handling",
            "duplicates": "duplicate handling",
            "nullHandling": "null handling"
          }},
          "performanceConsiderations": {{
            "scalability": "scalability approach",
            "caching": "caching approach",
            "archiving": "archiving policy",
            "monitoring": "monitoring approach"
          }}
        }}
      ],
      "fileConventions": ["file conventions"],
      "security": {{
        "authentication": "authentication",
        "authorization": "authorization",
        "dataProtection": "data protection",
        "inputValidation": "input validation",
        "outputEncoding": "output encoding",
        "auditLogging": "audit logging"
      }},
      "performance": {{
        "caching": "caching strategy",
        "indexing": "indexing approach",
        "queryOptimization": "query optimization",
        "asynchronousProcessing": "async processing",
        "loadTesting": "load testing"
      }},
      "errorHandling": {{
        "errorScenarios": "error scenarios",
        "userFeedback": "user feedback",
        "recoveryMechanisms": "recovery mechanisms",
        "logging": "error logging"
      }},
      "edgeCases": {{
        "boundaryConditions": "boundary handling",
        "concurrentUsers": "concurrency handling",
        "networkFailures": "network failure handling",
        "dataInconsistencies": "inconsistency handling",
        "unusualInputs": "unusual input handling"
      }},
      "bestPractices": {{
        "codeOrganization": "code organization",
        "testingStrategy": "testing strategy",
        "documentation": "documentation",
        "versionControl": "version control",
        "codeReview": "code review"
      }},
      "caching": "caching strategy",
      "messaging": "messaging approach"
    }}
  ],
  "featureFlow": {{
    "description": "feature flow description",
    "sequence": [
      {{
        "step": 1,
        "feature": "feature name",
        "action": "action",
        "triggeredBy": "trigger",
        "triggers": "next trigger",
        "dataFlow": "data flow"
      }}
    ],
    "interactions": [
      {{
        "from": "feature name",
        "to": "feature name",
        "type": "interaction type",
        "description": "interaction"
      }}
    ]
  }}
}}

IMPORTANT: Use ONLY the fields shown above. Do NOT add additional fields.

CHECKLIST: APIs have inputValidation/security/edgeCases/errorHandling. DBs have proper indexes. Features have errorHandling/edgeCases/performance. Include featureFlow."""

        # Call OpenAI
        try:
            client = OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)
            
            logger.info('Calling OpenAI for LLD...')
            logger.info(f'Prompt length: system={len(system_prompt)}, user={len(user_prompt)}')
            
            completion = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_LLD_OUTPUT_TOKENS
            )
            
            # Check completion status
            finish_reason = completion.choices[0].finish_reason if completion.choices else None
            logger.info(f'OpenAI finish_reason: {finish_reason}')
            
            # Parse response (use parsed when available)
            message = completion.choices[0].message
            if hasattr(message, "parsed") and message.parsed is not None:
                ai_response = message.parsed
            else:
                content = message.content or ""
                if not content.strip():
                    logger.error(f'OpenAI returned empty response. Finish reason: {finish_reason}')
                    if finish_reason == "length":
                        raise HTTPException(
                            status_code=502,
                            detail="Response exceeded token limit. Please simplify the architecture or reduce the number of features."
                        )
                    elif finish_reason == "content_filter":
                        raise HTTPException(
                            status_code=502,
                            detail="Response was filtered. Please review the architecture description for inappropriate content."
                        )
                    else:
                        raise HTTPException(
                            status_code=502,
                            detail=f"AI service returned empty response (reason: {finish_reason}). Please try again."
                        )
                try:
                    ai_response = json.loads(content)
                except json.JSONDecodeError as json_err:
                    logger.error(f'JSON parse error: {str(json_err)}')
                    logger.error(f'Content preview: {content[:500]}...')
                    raise HTTPException(
                        status_code=502,
                        detail=f"Failed to parse AI response: {str(json_err)}"
                    )
        except HTTPException:
            # Re-raise HTTPExceptions as-is
            raise
        except Exception as e:
            logger.error(f'OpenAI call failed: {str(e)}')
            raise HTTPException(
                status_code=502,
                detail=f"AI generation failed: {str(e)}"
            )

        # Get features from LLM response
        returned_features = ai_response.get('features', [])
        returned_feature_names = [f.get('name', '') for f in returned_features if isinstance(f, dict)]
        
        logger.info(f'LLM generated {len(returned_features)} features: {returned_feature_names}')
        
        # Build LLD response directly from LLM output
        lld_data = {
            "header": {
                "title": "Low Level Design",
                "subtitle": f"Technical design for {application_name}"
            },
            "baseFramework": ai_response.get('baseFramework', {}),
            "features": returned_features,
            "featureFlow": ai_response.get('featureFlow', {})
        }

        # Store in Cosmos DB Design Container
        design_id = str(uuid4())
        # Use architecture_id from request, or generate a new one
        final_architecture_id = architecture_id if architecture_id else str(uuid4())
        
        try:
            document = {
                'id': design_id,
                'tenantId': tenant_id,
                'architectureId': final_architecture_id,
                'designId': design_id,
                'applicationName': application_name,
                'sessionId': session_id,
                'userId': user_id,
                'type': 'lowLevelDesign',
                'requirements': requirements[:4000] if isinstance(requirements, str) else str(requirements)[:4000],
                'features': features,  # Original features from request
                'generatedFeatures': returned_feature_names,  # Features generated by LLM
                'architecture': arch_obj,
                'lldResponse': lld_data,
                'openAIUsage': {
                    'promptTokens': completion.usage.prompt_tokens if completion.usage else 0,
                    'completionTokens': completion.usage.completion_tokens if completion.usage else 0,
                    'totalTokens': completion.usage.total_tokens if completion.usage else 0
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Check document size (Cosmos DB has 2MB limit)
            doc_size = len(json.dumps(document))
            logger.info(f'LLD document size: {doc_size} bytes (~{doc_size/1024:.2f} KB)')
            if doc_size > 1.8 * 1024 * 1024:  # Warn at 1.8MB (90% of 2MB limit)
                logger.warning(f'LLD document size ({doc_size/1024/1024:.2f} MB) is approaching Cosmos DB 2MB limit!')
            
            await db.create_design(document)
            logger.info(f'LLD stored in Design Container - designId: {design_id}, architectureId: {final_architecture_id}')
        except Exception as cosmos_err:
            logger.error(f'Cosmos DB error: {str(cosmos_err)}')
            # Continue even if storage fails - still return the LLD

        # Return LLD
        return LLDResponse(
            success=True,
            designId=design_id,
            featureCount=len(lld_data.get("features", [])),
            lld=lld_data,
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating LLD: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate LLD: {str(e)}"
        )


@app.post("/api/generate-code", response_model=CodeGenerationResponse)
async def generate_code_from_lld(
    request: CodeGenerationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    cosmos: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Generate code files from Low Level Design (LLD) specification using OpenAI.

    Takes a comprehensive LLD structure and generates actual source code for every
    file referenced in the LLD, using only the technologies, frameworks, and
    patterns described in the LLD itself — no hardcoded assumptions.
    """
    try:
        logger.info(f"Generating code for design: {request.designId}")

        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(status_code=500, detail="OpenAI API is not configured")

        if not request.lld:
            raise HTTPException(status_code=400, detail="LLD specification is required")

        lld_data = request.lld
        base_framework = lld_data.get("baseFramework", {})
        features = lld_data.get("features", [])
        header = lld_data.get("header", {})
        app_title = header.get("subtitle", header.get("title", "Application"))
        
        # Extract application name for proper project naming
        tenant_id = request.lld.get("tenantId", "default")
        application_name = request.lld.get("applicationName", app_title)
        # Clean up common prefixes if present
        for prefix in ["Technical design for ", "Architecture for ", "Design for "]:
            if application_name.startswith(prefix):
                application_name = application_name[len(prefix):]
                break
        # Create a clean project name (lowercase, no spaces)
        project_name = application_name.lower().replace(" ", "-").replace("_", "-")

        # Extract all framework/language/tech details directly from the LLD
        frontend_info = base_framework.get("frontend", {})
        backend_info = base_framework.get("backend", {})
        frontend_framework = frontend_info.get("framework", "")
        frontend_language = frontend_info.get("language", "")
        frontend_tech_stack: List[str] = frontend_info.get("techStack", [])
        frontend_components: Dict[str, Any] = frontend_info.get("components", {})

        backend_framework = backend_info.get("framework", "")
        backend_database = backend_info.get("database", "")
        backend_tech_stack: List[str] = backend_info.get("techStack", [])
        backend_components: Dict[str, Any] = backend_info.get("components", {})

        # Use the first backend tech stack item or framework as runtime
        backend_runtime = backend_tech_stack[0] if backend_tech_stack else backend_framework

        # Collect all unique technologies across the entire LLD
        all_technologies: set = set(frontend_tech_stack + backend_tech_stack)
        for feature in features:
            all_technologies.update(feature.get("techStack", []))

        # Build the explicit file list from every screen and every API referenced in the LLD
        files_to_generate: List[Dict[str, Any]] = []
        seen_paths: set = set()

        for feature in features:
            feature_name = feature.get("name", "")
            for screen in feature.get("screens", []):
                file_path = screen.get("file", "")
                if file_path and file_path not in seen_paths:
                    seen_paths.add(file_path)
                    files_to_generate.append({
                        "path": file_path,
                        "type": "component",
                        "framework": frontend_framework,
                        "language": frontend_language,
                        "context": {
                            "feature": feature_name,
                            "screen": screen.get("name"),
                            "route": screen.get("route"),
                            "stateManagement": frontend_components.get("stateManagement", ""),
                            "authentication": frontend_components.get("authentication", ""),
                            "errorHandling": frontend_components.get("errorHandling", ""),
                            "caching": frontend_components.get("caching", ""),
                            "apis": screen.get("apis", [])
                        }
                    })
                # Backend handler/API files from screen-level APIs
                for api in screen.get("apis", []):
                    # Try multiple possible field names for backend handler files
                    handler_file = api.get("lambdaFile") or api.get("handlerFile") or api.get("apiFile") or api.get("functionFile") or ""
                    if handler_file and handler_file not in seen_paths:
                        seen_paths.add(handler_file)
                        files_to_generate.append({
                            "path": handler_file,
                            "type": "api",
                            "framework": backend_framework,
                            "language": backend_runtime,
                            "context": {
                                "feature": feature_name,
                                "endpoint": api.get("endpoint"),
                                "method": api.get("method"),
                                "description": api.get("description", ""),
                                "tables": api.get("tables", []),
                                "apiPattern": backend_components.get("apiPattern", ""),
                                "dataValidation": backend_components.get("dataValidation", ""),
                                "errorHandling": backend_components.get("errorHandling", ""),
                                "authentication": backend_components.get("authentication", ""),
                                "logging": backend_components.get("logging", ""),
                                "dataAccess": backend_components.get("dataAccess", "")
                            }
                        })
            # Feature-level API handler files
            for api in feature.get("apis", []):
                # Try multiple possible field names for backend handler files
                handler_file = api.get("lambdaFile") or api.get("handlerFile") or api.get("apiFile") or api.get("functionFile") or ""
                if handler_file and handler_file not in seen_paths:
                    seen_paths.add(handler_file)
                    files_to_generate.append({
                        "path": handler_file,
                        "type": "api",
                        "framework": backend_framework,
                        "language": backend_runtime,
                        "context": {
                            "feature": feature_name,
                            "endpoint": api.get("endpoint"),
                            "method": api.get("method"),
                            "description": api.get("description", ""),
                            "tables": api.get("tables", []),
                            "apiPattern": backend_components.get("apiPattern", ""),
                            "dataValidation": backend_components.get("dataValidation", ""),
                            "errorHandling": backend_components.get("errorHandling", ""),
                            "authentication": backend_components.get("authentication", ""),
                            "logging": backend_components.get("logging", ""),
                            "dataAccess": backend_components.get("dataAccess", "")
                        }
                    })

        # Collect all data models across all features
        all_data_models: List[Dict[str, Any]] = []
        for feature in features:
            for model in feature.get("dataModels", []):
                if model.get("tableName") and model not in all_data_models:
                    all_data_models.append(model)

        # Collect all fileConventions from all features
        all_file_conventions: List[str] = []
        feature_names_list: List[str] = []
        for feature in features:
            all_file_conventions.extend(feature.get("fileConventions", []))
            feature_names_list.append(feature.get("name", ""))

        # ── SCAFFOLDING PROMPT SECTION ─────────────────────────────────────────
        # Build a natural-language description of every project-wide file the model
        # must produce, derived entirely from the LLD tech-stack variables already
        # extracted above.  No paths are hardcoded in Python — the model resolves
        # them from its own knowledge of the frameworks/tools listed here.
        fe_tech_str  = ', '.join(frontend_tech_stack)  if frontend_tech_stack  else frontend_framework
        be_tech_str  = ', '.join(backend_tech_stack)   if backend_tech_stack   else backend_framework
        all_tech_str = ', '.join(sorted(all_technologies))

        # ── PROMPTS ────────────────────────────────────────────────────────────
        features_list_str = '\n'.join([f'  - {name}' for name in feature_names_list]) if feature_names_list else '  - No features specified'
        frontend_styling = frontend_components.get("styling", "")
        frontend_routing = frontend_components.get("routing", "")
        frontend_state   = frontend_components.get("stateManagement", "")
        frontend_auth    = frontend_components.get("authentication", "")
        backend_api_pat  = backend_components.get("apiPattern", "")
        backend_data_acc = backend_components.get("dataAccess", "")
        backend_auth     = backend_components.get("authentication", "")
        backend_logging  = backend_components.get("logging", "")
        backend_mw       = backend_components.get("middleware", "")

        system_prompt = (
            "You are a senior full-stack software engineer who writes production-grade code. "
            "Your job: read an LLD spec and return a single valid JSON object containing every source file "
            "needed to implement the application features described in the LLD. "
            "Rules:\n"
            "1. Use ONLY the technologies listed in the LLD — no additions.\n"
            "2. Every file must have complete, working code. No TODO, no '...', no placeholders.\n"
            "3. Generate ONLY application source code: screen components, API handlers, data models, "
            "shared utilities, routing, state management, and the minimal config files required by the "
            "tech stack.\n"
            "4. DO NOT generate any CI/CD, Docker, Kubernetes, Terraform, nginx, monitoring, "
            "backup scripts, or any other infrastructure/deployment files.\n"
            "5. Every import path inside a generated file must resolve to another file in generatedFiles. "
            "Scan imports and add missing shared files before responding.\n"
            "6. All dependency versions must be the latest stable semver ranges — never '1.0.0', '*', or 'latest'.\n"
            "7. Tag every file with the exact feature name from the LLD, or 'shared' for project-wide files.\n"
            "8. Return ONLY the JSON — no prose, no markdown fences.\n\n"
            "BEST PRACTICES — apply these universally across every generated file regardless of tech stack:\n\n"
            "SECURITY:\n"
            "- Validate and sanitise all user inputs before processing or persisting.\n"
            "- Never hardcode secrets, tokens, or credentials — read from environment variables.\n"
            "- Apply authentication and authorisation checks on every protected route/endpoint as described in the LLD.\n"
            "- Use parameterised queries / ORM methods — never string-concatenated queries.\n"
            "- Set security headers (CSP, X-Frame-Options, etc.) in the application layer where the stack supports it.\n"
            "- Sanitise data before rendering to prevent XSS; never use dangerouslySetInnerHTML or equivalent without explicit sanitisation.\n\n"
            "LOGGING:\n"
            "- Use a structured logging approach (JSON-formatted log entries) with consistent fields: timestamp, level, message, correlationId, and relevant context.\n"
            "- Apply appropriate log levels (debug, info, warn, error) — never log sensitive data (passwords, tokens, PII).\n"
            "- Generate or propagate a correlationId per request and include it in every log entry and outbound call.\n"
            "- Log the start and completion of significant operations and all unhandled errors with a stack trace.\n\n"
            "ERROR HANDLING:\n"
            "- Wrap async operations in try/catch; never swallow errors silently.\n"
            "- Return consistent, structured error responses from APIs (e.g. {error, message, code}).\n"
            "- Add UI-level error boundaries or equivalent so one failing component cannot crash the whole page.\n"
            "- Distinguish between client errors (4xx) and server errors (5xx) and respond accordingly.\n\n"
            "STYLING & UI:\n"
            "- Use a single, centrally defined theme/design-token file for colours, spacing, typography, and breakpoints.\n"
            "- All components must be fully responsive and mobile-first.\n"
            "- Follow accessibility basics: semantic HTML elements, ARIA labels on interactive controls, sufficient colour contrast, keyboard navigability.\n"
            "- No magic numbers in style files — use variables or tokens.\n\n"
            "CODE QUALITY:\n"
            "- Follow single-responsibility principle; keep functions/methods short and focused.\n"
            "- Avoid duplication — extract reusable utilities, hooks, or services.\n"
            "- Use meaningful, descriptive names for variables, functions, and files.\n"
            "- Add concise JSDoc/docstring comments on public functions and complex logic.\n"
            "- Use strict typing wherever the chosen language supports it."
        )

        user_prompt = f"""Generate the complete application source code for this project based solely on the LLD.

PROJECT: {app_title}  |  NAME: {project_name}

TECH STACK (derive all file names/paths from this — no other technologies):
  Frontend : {frontend_framework} · {frontend_language} · {fe_tech_str}
             styling={frontend_styling}  routing={frontend_routing}  state={frontend_state}  auth={frontend_auth}
  Backend  : {backend_framework} · {backend_runtime} · {be_tech_str}
             db={backend_database}  api={backend_api_pat}  dataAccess={backend_data_acc}  auth={backend_auth}
             logging={backend_logging}  middleware={backend_mw}

FEATURES (use these exact names as the "feature" tag, or "shared" for project-wide files):
{features_list_str}

LLD (source of truth — generate code only for what is described here):
{json.dumps(lld_data, indent=2)}

SCREEN & API FILES TO GENERATE (every file in this list must appear in generatedFiles):
{json.dumps(files_to_generate, indent=2)}

DATA MODELS (use exact table names, column names, types, partition keys, and sort keys as specified):
{json.dumps(all_data_models, indent=2)}

INSTRUCTIONS:
1. Generate every screen component and API handler listed above.
2. Generate the minimal scaffold files the tech stack requires (dependency manifests, config files, entry points, routing setup, state management setup, global styles/theme files, environment variable templates).
3. Generate data access / repository files for every data model listed above.
4. Resolve all imports — if a component imports a service or utility, include that file too.
5. Use latest stable package versions.
6. DO NOT generate Dockerfiles, docker-compose, CI/CD pipelines, Kubernetes manifests, Terraform, nginx, IaC, or any deployment/infra files.
7. IMPLEMENT EXACT LLD SPECIFICATIONS:
   - For each API, implement the EXACT input validation rules specified in api.inputValidation (required fields, data types, regex patterns, range checks, etc.)
   - Apply the EXACT security measures from api.security (authentication, authorization, sanitization, rate limiting, encryption)
   - Handle ALL edge cases listed in api.edgeCases (null checks, boundary conditions, concurrent operations, network failures, etc.)
   - Implement the EXACT error handling patterns from api.errorHandling (error codes, messages, retry logic, circuit breakers, correlationId propagation)
   - For each table, create the EXACT indexes specified in table.indexes with correct columns, types (btree/hash/unique), and include conditions
   - Apply performance optimizations from feature.performance (caching strategies, async patterns, connection pooling, pagination)
   - Follow the feature interaction flows from featureFlow (understand data sharing, event triggers, navigation sequences)
8. SECURITY: validate all inputs per LLD specs, read secrets from env vars, apply auth checks on protected routes, use parameterised queries, sanitise rendered output.
9. LOGGING: add structured logging (JSON, with timestamp/level/correlationId) to every API handler; log errors with stack traces; never log sensitive data.
10. ERROR HANDLING: wrap all async calls in try/catch, return error shapes matching api.errorHandling from LLD, add UI error boundaries.
11. STYLING: define a single theme/token file for colours, spacing, and typography; make all UI components responsive and accessible (semantic HTML, ARIA, keyboard nav).
12. CODE QUALITY: single-responsibility functions, no magic numbers, descriptive names, JSDoc/docstrings on public APIs, strict typing throughout.

Return this exact JSON (nothing else):
{{
  "projectName": "{project_name}",
  "technologies": ["<every tech from the LLD>"],
  "instructions": "<numbered local dev setup and run steps for this exact stack>",
  "projectStructure": {{
    "projectName": "{project_name}",
    "frontend": {{}},
    "backend": {{}},
    "infrastructure": {{}},
    "dataModels": {{}}
  }},
  "generatedFiles": [
    {{
      "path": "<relative path>",
      "category": "<frontend|backend|dataModel|config|documentation|tests>",
      "feature": "<feature name from LLD or 'shared'>",
      "content": "<complete file content>"
    }}
  ]
}}"""

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
        try:
            completion = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_CODE_GEN_OUTPUT_TOKENS
            )

            # Enhanced error handling with finish_reason diagnostics
            message = completion.choices[0].message
            finish_reason = completion.choices[0].finish_reason
            
            # Log prompt size for diagnostics
            prompt_length = len(system_prompt) + len(user_prompt)
            logger.info(f"Code generation prompt length: {prompt_length} characters, finish_reason: {finish_reason}")
            
            # Check for specific finish reasons
            if finish_reason == "length":
                logger.error(f"Code generation exceeded token limit. Prompt: {prompt_length} chars, Max tokens: {MAX_CODE_GEN_OUTPUT_TOKENS}")
                raise HTTPException(
                    status_code=502, 
                    detail=f"Response exceeded token limit. Try reducing the number of features or simplifying the LLD structure."
                )
            elif finish_reason == "content_filter":
                logger.error("Code generation blocked by content filter")
                raise HTTPException(status_code=502, detail="Content was filtered by AI safety systems. Please review your LLD for potentially problematic content.")
            
            if hasattr(message, "parsed") and message.parsed is not None:
                result = message.parsed
            else:
                content = message.content or ""
                if not content.strip():
                    if finish_reason != "stop":
                        logger.error(f"Empty code generation response with finish_reason={finish_reason}")
                        raise HTTPException(status_code=502, detail=f"AI service returned empty response (finish_reason: {finish_reason}). Please try again.")
                    raise HTTPException(status_code=502, detail="AI service returned an empty response. Please try again.")
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse code generation JSON: {str(e)}, Content preview: {content[:500]}")
                    raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {str(e)}")

        except HTTPException:
            raise
        except Exception as openai_err:
            logger.error(f"OpenAI error during code generation: {str(openai_err)}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI code generation failed: {str(openai_err)}")

        # Parse and validate generated files
        raw_files = result.get("generatedFiles", [])
        generated_files = []
        
        # Categorize files by both category and feature
        categorized = CategorizedFiles()
        feature_based = {}  # Dict to organize by feature name
        
        for f in raw_files:
            if isinstance(f, dict) and f.get("path") and f.get("content"):
                category = f.get("category", "other")
                feature_name = f.get("feature", "shared")
                
                file_obj = GeneratedFile(
                    path=f.get("path", ""),
                    content=f.get("content", "")
                )
                generated_files.append(file_obj)
                
                # Add to category-based organization
                if category == "frontend":
                    categorized.frontend.append(file_obj)
                elif category == "backend":
                    categorized.backend.append(file_obj)
                elif category == "dataModel":
                    categorized.dataModels.append(file_obj)
                elif category in ["infrastructure", "deployment", "monitoring", "security"]:
                    categorized.infrastructure.append(file_obj)
                elif category == "config":
                    categorized.config.append(file_obj)
                elif category == "documentation":
                    categorized.documentation.append(file_obj)
                elif category == "tests":
                    categorized.tests.append(file_obj)
                
                # Add to feature-based organization
                if feature_name not in feature_based:
                    feature_based[feature_name] = FeatureFiles(
                        featureName=feature_name,
                        frontend=[],
                        backend=[],
                        dataModels=[],
                        tests=[]
                    )
                
                if category == "frontend":
                    feature_based[feature_name].frontend.append(file_obj)
                elif category == "backend":
                    feature_based[feature_name].backend.append(file_obj)
                elif category == "dataModel":
                    feature_based[feature_name].dataModels.append(file_obj)
                elif category == "tests":
                    feature_based[feature_name].tests.append(file_obj)

        # Convert feature_based dict to list
        feature_based_list = list(feature_based.values())

        # Build project structure from the AI response
        raw_structure = result.get("projectStructure", {})
        response_project_name = result.get("projectName", project_name)

        def _sanitise_structure_section(val: Any) -> Dict[str, Any]:
            """Ensure the section is always a plain dict regardless of what the AI returned."""
            if isinstance(val, dict):
                return val
            if isinstance(val, list):
                # e.g. [{"path": "frontend", ...}, ...] — index by path
                return {item.get("path", str(i)): item for i, item in enumerate(val) if isinstance(item, dict)}
            # scalar (str, None, etc.) — wrap in a dict
            return {"value": val} if val else {}

        project_structure = ProjectStructure(
            projectName=response_project_name,
            frontend=_sanitise_structure_section(raw_structure.get("frontend", {})),
            backend=_sanitise_structure_section(raw_structure.get("backend", {})),
            infrastructure=_sanitise_structure_section(raw_structure.get("infrastructure", {})),
            dataModels=_sanitise_structure_section(raw_structure.get("dataModels", {}))
        )

        technologies = result.get("technologies", list(all_technologies))
        instructions = result.get("instructions", "")
        
        # Log categorization summary
        logger.info(f"Generated code categorization - Frontend: {len(categorized.frontend)}, Backend: {len(categorized.backend)}, DataModels: {len(categorized.dataModels)}, Infrastructure/Ops: {len(categorized.infrastructure)}, Config: {len(categorized.config)}, Docs: {len(categorized.documentation)}, Tests: {len(categorized.tests)}")
        logger.info(f"Generated code by feature - {len(feature_based_list)} features: {', '.join(feature_based.keys())}")

        return CodeGenerationResponse(
            success=True,
            message=f"Successfully generated {len(generated_files)} files for {request.featureCount} features using {', '.join([frontend_framework, backend_framework]) if frontend_framework or backend_framework else 'specified'} tech stack",
            projectName=response_project_name,
            projectStructure=project_structure,
            featureBasedFiles=feature_based_list,
            categorizedFiles=categorized,
            generatedFiles=generated_files,
            fileCount=len(generated_files),
            technologies=technologies,
            instructions=instructions,
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate code: {str(e)}"
        )


@app.post("/api/generate-infrastructure", response_model=InfrastructureResponse)
async def generate_infrastructure_scripts(
    request: InfrastructureRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Generate infrastructure-as-code scripts based on architecture and technology stack.
    
    Analyzes the application requirements, cloud provider, and specific services needed
    to generate production-ready IaC scripts (Terraform, CloudFormation, AWS SAM, etc.)
    """
    try:
        logger.info(f"Generating infrastructure for: {request.applicationName}")

        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(status_code=500, detail="OpenAI API is not configured")

        # Extract technology stack details
        tech_stack = request.technologyStack
        cloud_provider = tech_stack.get("cloudProvider", "")
        runtime = tech_stack.get("runtime", "")
        languages = tech_stack.get("languages", "")
        frameworks = tech_stack.get("frameworks", "")
        infra = tech_stack.get("infra", {})

        # Extract all infrastructure services
        compute = infra.get("compute", "")
        database = infra.get("database", "")
        cache = infra.get("cache", "")
        messaging = infra.get("messaging", "")
        storage = infra.get("storage", "")
        api_gateway = infra.get("apiGateway", "")
        authentication = infra.get("authentication", "")

        # Collect all mentioned services
        all_services = []
        for service_desc in [compute, database, cache, messaging, storage, api_gateway, authentication]:
            if service_desc:
                all_services.append(service_desc)

        # Build features string
        features_str = "\\n".join([f"  - {f}" for f in request.features]) if request.features else "  - No specific features listed"

        system_prompt = (
            "You are a senior DevOps and cloud infrastructure architect specializing in Infrastructure-as-Code and CI/CD automation. "
            "Generate production-ready IaC templates, CI/CD pipelines, and parameter files based on exact requirements. "
            "Ensure ALL services are properly integrated with cross-resource references, IAM permissions, and network connectivity. "
            "Use ONLY the cloud provider and services specified - do NOT add unrequested services. "
            "Every service must be connected and able to communicate with dependent services. "
            "Analyze the cloud provider and services to determine the BEST IaC format (e.g., CloudFormation/SAM for AWS, ARM/Bicep for Azure, Deployment Manager for GCP, Terraform for multi-cloud). "
            "Generate appropriate CI/CD pipeline scripts (GitHub Actions, Azure DevOps, Bitbucket Pipelines, GitLab CI) for automated deployment. "
            "Create comprehensive parameter/variable files to eliminate hardcoding and enable multi-environment deployment. "
            "Return ONLY valid JSON matching the schema."
        )

        user_prompt = f"""Generate complete Infrastructure-as-Code templates, CI/CD pipelines, and configuration for the following application:

Application: {request.applicationName}

Architecture Description: {request.description}

Cloud Provider: {cloud_provider}
Runtime: {runtime}
Languages: {languages}
Frameworks: {frameworks}

Infrastructure Services Required:
- Compute: {compute}
- Database: {database}
- Cache: {cache if cache else "None"}
- Messaging: {messaging if messaging else "None"}
- Storage: {storage}
- API Gateway: {api_gateway if api_gateway else "None"}
- Authentication: {authentication if authentication else "None"}

Application Features:
{features_str}

CRITICAL INFRASTRUCTURE-AS-CODE REQUIREMENTS:

0. IAC FORMAT SELECTION:
   - Analyze the cloud provider and services to determine the BEST Infrastructure-as-Code format
   - AWS: CloudFormation (standard), AWS SAM (serverless/Lambda), or Terraform
   - Azure: ARM Templates, Bicep (modern ARM), or Terraform
   - GCP: Deployment Manager, Terraform
   - Multi-cloud: Terraform or Pulumi
   - Choose the format that best fits the services and deployment model
   - Specify your chosen format in the 'iacFormat' field

1. CI/CD PIPELINE GENERATION (MANDATORY - MUST BE FULLY FUNCTIONAL):
   - Generate COMPLETE, WORKING CI/CD pipeline scripts that can actually execute deployments
   - Platform selection: Choose the best platform for the cloud provider (GitHub Actions for AWS/multi-cloud, Azure DevOps for Azure, GitLab CI for GitLab/GCP, Bitbucket Pipelines for Bitbucket)
   - CRITICAL: Generate the ACTUAL pipeline files with correct paths:
     * GitHub Actions: .github/workflows/deploy.yml (or multiple workflow files)
     * Azure DevOps: azure-pipelines.yml  
     * GitLab CI: .gitlab-ci.yml
     * Bitbucket: bitbucket-pipelines.yml
   - Pipeline MUST include REAL, EXECUTABLE commands:
     * Infrastructure validation commands (terraform plan, cloudformation validate, etc.)
     * Infrastructure deployment commands (terraform apply, sam deploy, az deployment group create, etc.)
     * Application build commands specific to the runtime/language
     * Application deployment commands to the provisioned infrastructure
     * Environment-specific deployment jobs (dev, staging, production)
   - Include proper secrets/environment variable references using the platform's syntax
   - Include approval gates and manual triggers for production deployments
   - Use proper CI/CD platform syntax and features (jobs, steps, dependencies, conditions)
   - MUST BE READY-TO-USE: someone should be able to commit these files and have working deployments

2. PARAMETER/CONFIGURATION FILES (MANDATORY):
   - Generate comprehensive parameter files for each environment (dev, staging, prod)
   - Include variables for: resource names, regions, sizes, connection strings, feature flags
   - NO HARDCODED VALUES in main templates - everything must be parameterized
   - Create .env templates, parameter.json files, terraform.tfvars examples, or equivalent
   - Include secure parameter handling for secrets and credentials

3. PROPER IAC FORMAT:
   - Use valid syntax for your chosen IaC format with proper parameterization
   - Include all required template sections (version headers, resource definitions, outputs, parameters)
   - Use proper resource types for the target cloud provider
   - Follow the format's best practices and conventions
   - Use parameters/variables instead of hardcoded values

4. SERVICE INTEGRATION - ALL services MUST be connected:
   - Compute → Database: Include IAM policies/roles for database access, connection strings via parameters
   - Compute → Storage: IAM policies for bucket/blob access, parameterized resource names
   - Compute → Cache: Connection endpoints in parameters, security group/network rules
   - API Gateway → Compute: Proper integration type (Lambda proxy, HTTP proxy), invoke permissions
   - Authentication → API Gateway: Authorizer configuration (Cognito, AAD), JWT validation
   - Messaging → Compute: Event source mappings, trigger configurations, IAM permissions
   - Use proper cross-reference syntax for your chosen IaC format with parameter-driven values

5. NETWORKING & SECURITY:
   - Private networking configuration with parameterized CIDR blocks and subnets
   - Network security rules with configurable ingress/egress for service-to-service communication
   - Subnets (public/private) with parameterized configurations
   - Service endpoints for private connectivity
   - Use cloud-appropriate networking resources with environment-specific parameters

6. IAM & PERMISSIONS:
   - Execution roles/identities with parameterized policies
   - Service-to-service permissions based on actual service integrations
   - Least privilege principle with configurable permission sets
   - Use managed policies where available, parameterized custom policies when needed
   - Resource-based policies with parameter-driven resource ARNs

7. ENVIRONMENT CONFIGURATION:
   - Environment variables with parameterized resource ARNs/names/endpoints
   - Connection strings derived from parameters
   - API endpoints for service discovery using parameter references
   - Region-specific configurations via parameters
   - Use parameter references throughout, eliminate hardcoded values

8. MONITORING & OBSERVABILITY:
   - Log collection and aggregation resources with configurable retention periods
   - Metric-based alarms with parameterized thresholds
   - Application monitoring with environment-specific configurations
   - Distributed tracing configuration if applicable

9. SECURITY CONFIGURATIONS:
   - Encryption at rest with parameterized key management
   - Encryption in transit (HTTPS, TLS) with configurable certificate management
   - Secret management for credentials with parameter-driven access policies
   - Web application firewall with configurable rules if API is exposed

10. DEPLOYMENT DEPENDENCIES:
    - Use DependsOn with parameterized resource references
    - Outputs section with parameterized values (API endpoints, resource IDs)
    - Parameters section with comprehensive environment configurations

11. GENERATE ORGANIZED SCRIPTS WITH PARAMETERIZATION:
    - Main template: Core application resources with full parameterization
    - Database template: Database, cache resources with environment-specific sizing
    - Network template: Network infrastructure with configurable CIDR and security rules
    - Monitoring template: Logging, alarms, dashboards with parameterized thresholds
    - Security template: Encryption keys, secret storage, firewalls with configurable policies
    - Parameter files: Separate parameter sets for dev, staging, production environments
    - CI/CD pipeline: Complete automation scripts for infrastructure and application deployment

12. PRODUCTION-READY WITH PARAMETERS:
    - Auto-scaling configurations with parameterized thresholds
    - High availability (multi-AZ, zone redundancy) with configurable regions
    - Backup and disaster recovery with parameterized retention policies
    - Cost optimization tags with environment-specific values

Return JSON with EXACTLY this structure:
{{
  "applicationName": "{request.applicationName}",
  "cloudProvider": "{cloud_provider}",
  "iacFormat": "Your chosen IaC format (CloudFormation, SAM, ARM Templates, Bicep, Deployment Manager, Terraform, etc.)",
  "cicdPlatform": "Chosen CI/CD platform (GitHub Actions, Azure DevOps, GitLab CI, Bitbucket Pipelines, etc.)",
  "scripts": [
    {{
      "fileName": "Appropriate file name with correct extension for your IaC format",
      "format": "Same as iacFormat",
      "content": "Complete valid IaC template with parameters, no hardcoded values, proper resource types, cross-resource references, IAM roles/policies",
      "description": "What this template provisions",
      "services": ["List of cloud services provisioned by this specific script"]
    }}
  ],
  "parameterFiles": [
    {{
      "fileName": "Parameter file name (e.g., dev.parameters.json, staging.tfvars)",
      "environment": "Environment name (dev, staging, prod)",
      "format": "Parameter format (json, yaml, tfvars, etc.)",
      "content": "Complete parameter file with environment-specific values, no sensitive data",
      "description": "Environment-specific parameters for this deployment"
    }}
  ],
  "cicdPipelines": [
    {{
      "fileName": "EXACT pipeline file path (e.g., .github/workflows/deploy.yml, azure-pipelines.yml, .gitlab-ci.yml, bitbucket-pipelines.yml)",
      "platform": "Same as cicdPlatform",
      "content": "COMPLETE, FUNCTIONAL CI/CD pipeline with REAL executable commands for infrastructure deployment, application build/deploy, environment promotion, approval workflow",
      "description": "Production-ready automated deployment pipeline"
    }}
  ],
  "configFiles": [
    {{
      "fileName": "Configuration file name (e.g., .env.example, config.yaml)",
      "content": "Application configuration template with environment variables, connection templates",
      "description": "Application configuration template"
    }}
  ],
  "services": ["Complete list of all cloud services used across all templates"],
  "estimatedCost": "Realistic monthly cost estimate per environment (dev/staging/prod) based on actual services and expected usage"
}}

MANDATORY: Generate COMPLETE, PARAMETERIZED IaC templates and FUNCTIONAL CI/CD pipelines with:
- Proper syntax and parameterized resource types for the target cloud
- ALL services integrated with parameter-driven cross-references 
- IAM/RBAC roles and policies using parameterized resource references
- Environment variables/configurations using parameter files
- Network configurations with parameterized values enabling service connectivity
- FUNCTIONAL CI/CD pipelines with REAL deployment commands that actually work
- CI/CD pipelines must include ACTUAL commands (terraform apply, sam deploy, kubectl apply, etc.)
- No hardcoded values - everything must be parameterized and environment-configurable
- Comprehensive parameter files for multiple environments
- Working pipeline triggers and approval workflows using platform-specific syntax"""

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
        
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

            message = completion.choices[0].message
            if hasattr(message, "parsed") and message.parsed is not None:
                result = message.parsed
            else:
                content = message.content or ""
                if not content.strip():
                    raise HTTPException(status_code=502, detail="AI service returned empty response")
                result = json.loads(content)

        except HTTPException:
            raise
        except Exception as openai_err:
            logger.error(f"OpenAI error during infrastructure generation: {str(openai_err)}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI infrastructure generation failed: {str(openai_err)}")

        # Parse results - LLM determines IaC format based on cloud provider and services
        iac_format = result.get("iacFormat", "")
        cicd_platform = result.get("cicdPlatform", "")
        
        raw_scripts = result.get("scripts", [])
        scripts = []
        for s in raw_scripts:
            if isinstance(s, dict) and s.get("fileName") and s.get("content"):
                scripts.append(InfrastructureScript(
                    fileName=s.get("fileName", ""),
                    format=s.get("format", iac_format),
                    content=s.get("content", ""),
                    description=s.get("description", ""),
                    services=s.get("services", [])
                ))

        # Parse parameter files
        raw_parameter_files = result.get("parameterFiles", [])
        parameter_files = []
        for p in raw_parameter_files:
            if isinstance(p, dict) and p.get("fileName") and p.get("content"):
                parameter_files.append(InfrastructureScript(
                    fileName=p.get("fileName", ""),
                    format=p.get("format", "parameters"),
                    content=p.get("content", ""),
                    description=p.get("description", ""),
                    services=[f"{p.get('environment', 'unknown')} environment parameters"]
                ))

        # Parse CI/CD pipelines
        raw_cicd_pipelines = result.get("cicdPipelines", [])
        cicd_pipelines = []
        for c in raw_cicd_pipelines:
            if isinstance(c, dict) and c.get("fileName") and c.get("content"):
                cicd_pipelines.append(InfrastructureScript(
                    fileName=c.get("fileName", ""),
                    format=c.get("platform", cicd_platform),
                    content=c.get("content", ""),
                    description=c.get("description", ""),
                    services=["CI/CD automation"]
                ))

        # Parse config files
        raw_config_files = result.get("configFiles", [])
        config_files = []
        for cfg in raw_config_files:
            if isinstance(cfg, dict) and cfg.get("fileName") and cfg.get("content"):
                config_files.append(InfrastructureScript(
                    fileName=cfg.get("fileName", ""),
                    format="config",
                    content=cfg.get("content", ""),
                    description=cfg.get("description", ""),
                    services=["Application configuration"]
                ))

        # Combine all files into scripts for response
        all_scripts = scripts + parameter_files + cicd_pipelines + config_files

        services = result.get("services", all_services)
        estimated_cost = result.get("estimatedCost", "")

        logger.info(f"Generated {len(scripts)} IaC scripts, {len(parameter_files)} parameter files, {len(cicd_pipelines)} CI/CD pipelines, {len(config_files)} config files for {request.applicationName} using {iac_format}")

        return InfrastructureResponse(
            success=True,
            message=f"Successfully generated {len(all_scripts)} infrastructure and automation files using {iac_format} and {cicd_platform}",
            applicationName=request.applicationName,
            cloudProvider=cloud_provider,
            iacFormat=iac_format,
            scripts=all_scripts,  # Combined all types for backward compatibility
            services=services,
            estimatedCost=estimated_cost,
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating infrastructure: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate infrastructure: {str(e)}"
        )


@app.post("/api/lld/diagram", response_model=LLDDiagramResponse)
async def generate_lld_diagram(
    request: LLDDiagramRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """Generate a low-level design diagram JSON from LLD or requirements details."""
    try:
        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API is not configured"
            )

        tenant_id = request.tenantId or "default"
        application_name = request.applicationName
        session_id = request.sessionId or "default"

        arch_obj = request.architecture or {}
        arch_name = arch_obj.get("name", application_name)

        # Prefer explicit LLD JSON if provided, otherwise use overview + architecture summary
        lld_obj = request.lld or {}
        overview_text = request.overview or ""

        # Extract basic lists for the prompt to help layout
        features_list: List[str] = []
        if isinstance(lld_obj, dict):
            for f in lld_obj.get("features", []):
                if isinstance(f, dict) and f.get("name"):
                    features_list.append(str(f.get("name")))
        if not features_list and request.features:
            features_list = request.features

        features_text = "\n".join(f"- {f}" for f in features_list) if features_list else "Infer key features from the LLD and overview."

        system_prompt = (
            "You are an expert software architect and solution designer. "
            "Convert a textual low-level design into a clean, node-and-arrow style diagram. "
            "Return ONLY JSON matching the requested schema and the allowed canvas tool/service values."
        )

        user_prompt = f"""Generate a JSON representation of a low-level design diagram.

Application Name: {application_name}
Architecture Name: {arch_name}

Overview / Requirements (if any):
{overview_text}

Existing LLD (if provided, summarize into components and flows):
{json.dumps(lld_obj)[:4000] if lld_obj else "No explicit LLD JSON provided."}

Key features or flows to represent:
{features_text}

Requirements for the diagram:
- Focus on services, modules, and data stores involved in main user flows.
- Show main request paths, background processing, and data storage.
- Use the following shape schema for each node or arrow:
    - tool: one of [
            "rect", "ellipse", "circle", "triangle", "diamond", "parallelogram", "star", "polygon", "line",
            "aws", "azure", "gcp", "security", "cloud-infra", "generic", "uploaded-image", "component", "arrow"
        ].
    - x, y: integers for positioning (grid 0-1400 for x, 0-900 for y).
    - id: stable unique id like "svc-1", "db-1", "queue-1", "arrow-1".
    - service: for cloud & service tools (aws, azure, gcp, security, cloud-infra) use a canonical id such as:
        - AWS: aws-ec2, aws-lambda, aws-elastic-beanstalk, aws-batch, aws-s3, aws-ebs, aws-efs, aws-s3-glacier,
            aws-rds, aws-dynamodb, aws-aurora, aws-elasticache, aws-vpc, aws-cloudfront, aws-route53, aws-elb,
            aws-ecs, aws-eks, aws-fargate, aws-ecr, aws-api-gateway, aws-sqs, aws-sns.
        - Azure: azure-vm, azure-functions, azure-app-service, azure-batch, azure-vmss, azure-storage, azure-files,
            azure-netapp, azure-backup, azure-databox, azure-sql, azure-cosmos, azure-mysql, azure-postgresql,
            azure-redis, azure-vnet, azure-load-balancer, azure-application-gateway, azure-cdn, azure-dns,
            azure-firewall, azure-container-instances, azure-aks, azure-container-registry, azure-service-fabric.
        - GCP: gcp-compute-engine, gcp-cloud-functions, gcp-cloud-storage, gcp-cloud-sql, gcp-vpc, gcp-kubernetes-engine.
        - Security: security-auth0, security-ping, security-okta, security-azuread.
        - Cloud infra: aws-account, aws-region, aws-vpc, aws-availability-zone, aws-subnet,
            azure-subscription, azure-management-group, azure-resource-group, azure-region, azure-vnet,
            gcp-organization, gcp-folder, gcp-project, gcp-region, gcp-zone, gcp-vpc.
        - For generic/component boxes, use a short human-readable label like "Auth Service" or "Order API".
    - For components you MAY set width, height, rotation.
    - For arrows (tool = "arrow"), set points as [x1, y1, x2, y2], stroke color, strokeWidth, and optional dash.

Layout guidelines:
- Organize by layers (UI, API/services, data) from top to bottom or left to right.
- Group related components (e.g., all auth-related pieces near each other).
- Use arrows to clearly show sequence and data flow for typical user actions.

Return JSON with EXACTLY this top-level structure:
{{
  "name": "short diagram name",
  "description": "1-2 sentence description of what the diagram shows",
  "shapes": [
    {{
      "tool": "component",
      "x": 120,
      "y": 80,
      "id": "svc-1",
      "service": "Auth Service",
      "width": 160,
      "height": 80,
      "rotation": 0
    }},
    {{
      "tool": "arrow",
      "x": 0,
      "y": 0,
      "id": "arrow-1",
      "arrowType": "single",
      "points": [200, 120, 450, 120],
      "stroke": "#555555",
      "strokeWidth": 2,
      "dash": []
    }}
  ]
}}

IMPORTANT:
- Only use fields defined in the schema above.
- Ensure all ids are unique.
- Ensure the layout makes the main flows easy to follow.
"""

        try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
            completion = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=4000
            )

            message = completion.choices[0].message
            if hasattr(message, "parsed") and message.parsed is not None:
                diagram_json = message.parsed
            else:
                content = message.content or ""
                if not content.strip():
                    raise HTTPException(
                        status_code=502,
                        detail="AI service returned empty response. Please try again."
                    )

                diagram_json = json.loads(content)

            # Normalize tool/service pairs on all shapes before validation
            shapes = diagram_json.get("shapes") or []
            normalized_shapes: List[Dict[str, Any]] = []
            for raw_shape in shapes:
                if isinstance(raw_shape, dict):
                    normalized_shapes.append(_normalize_shape_tool_and_service(raw_shape))
                else:
                    normalized_shapes.append(raw_shape)
            diagram_json["shapes"] = normalized_shapes
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating LLD diagram via OpenAI: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to generate LLD diagram: {str(e)}"
            )

        try:
            diagram = LLDDiagram(**diagram_json)
        except Exception as e:
            logger.error(f"LLD diagram JSON did not match expected schema: {str(e)} | JSON: {diagram_json}")
            raise HTTPException(
                status_code=502,
                detail="AI LLD diagram response has invalid structure"
            )

        diagram_id = str(uuid4())

        # Optionally store in Cosmos designs container
        try:
            document = {
                "id": diagram_id,
                "tenantId": tenant_id,
                "applicationName": application_name,
                "sessionId": session_id,
                "userId": current_user.user_id,
                "type": "lldDiagram",
                "architecture": arch_obj,
                "overview": overview_text[:4000] if overview_text else None,
                "lld": lld_obj,
                "features": features_list,
                "diagram": diagram_json,
                "timestamp": datetime.utcnow().isoformat()
            }
            await db.create_design(document)
        except Exception as e:
            logger.error(f"Failed to store LLD diagram in Cosmos DB: {str(e)}")

        return LLDDiagramResponse(
            success=True,
            diagramId=diagram_id,
            diagram=diagram
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating LLD diagram: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate LLD diagram: {str(e)}"
        )



# ==================== GITHUB OAUTH ====================

class GitHubExchangeRequest(BaseModel):
    code: str = Field(..., description="OAuth code from GitHub callback")
    redirect_uri: str = Field(..., description="Redirect URI used in the OAuth flow")


@app.post("/api/github/exchange")
async def github_exchange(request: GitHubExchangeRequest):
    """
    Exchange a GitHub OAuth code for an access token.
    The client secret is kept server-side and never exposed to the browser.
    """
    if not GITHUB_CLIENT_SECRET or not GITHUB_CLIENT_ID:
        print(f"[GitHub] ERROR: CLIENT_ID={GITHUB_CLIENT_ID!r}, SECRET={'SET' if GITHUB_CLIENT_SECRET else 'MISSING'}")
        raise HTTPException(
            status_code=500,
            detail="GitHub OAuth is not configured (missing REACT_APP_GITHUB_CLIENT_ID or GITHUB_CLIENT_SECRET)"
        )

    print(f"[GitHub] Exchange: client_id={GITHUB_CLIENT_ID}, code={request.code[:8]}..., redirect_uri={request.redirect_uri}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": request.code,
                    "redirect_uri": request.redirect_uri,
                },
                headers={"Accept": "application/json"},
                timeout=15.0
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"GitHub token exchange failed: {str(e)}")
            raise HTTPException(status_code=502, detail=f"GitHub API error: {str(e)}")

    token_data = response.json()
    print(f"[GitHub] Exchange response: {token_data}")
    if "error" in token_data:
        raise HTTPException(
            status_code=400,
            detail=token_data.get("error_description", token_data["error"])
        )

    return token_data


class CreateRepoRequest(BaseModel):
    token: str = Field(..., description="GitHub access token")
    name: str = Field(..., description="Repository name")
    description: str = Field("", description="Repository description")
    private: bool = Field(False, description="Whether the repo is private")
    auto_init: bool = Field(False, description="Whether to auto-initialize with README")


@app.post("/api/github/create-repo")
async def github_create_repo(request: CreateRepoRequest):
    """
    Create a GitHub repository on behalf of the user.
    The token is passed from the client but GitHub REST calls stay server-side.
    """
    print(f"[GitHub] create-repo: token={request.token[:12]}..., name={request.name}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.github.com/user/repos",
                json={
                    "name": request.name,
                    "description": request.description,
                    "private": request.private,
                    "auto_init": request.auto_init,
                },
                headers={
                    "Authorization": f"Bearer {request.token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=15.0,
            )
        except httpx.HTTPError as e:
            logger.error(f"GitHub create-repo failed: {str(e)}")
            raise HTTPException(status_code=502, detail=f"GitHub API error: {str(e)}")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token expired or invalid")
    if not response.is_success:
        error = response.json()
        raise HTTPException(
            status_code=response.status_code,
            detail=error.get("message", "Failed to create repository"),
        )

    return response.json()


class UploadFileRequest(BaseModel):
    token: str = Field(..., description="GitHub access token")
    repo_full_name: str = Field(..., description="owner/repo")
    file_path: str = Field(..., description="Path inside the repo")
    content: str = Field(..., description="Base64-encoded file content")
    message: str = Field("", description="Commit message")


@app.post("/api/github/upload-file")
async def github_upload_file(request: UploadFileRequest):
    """
    Upload / create a single file in a GitHub repository.
    """
    commit_message = request.message or f"Add {request.file_path}"
    url = f"https://api.github.com/repos/{request.repo_full_name}/contents/{request.file_path}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(
                url,
                json={"message": commit_message, "content": request.content},
                headers={
                    "Authorization": f"Bearer {request.token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=15.0,
            )
        except httpx.HTTPError as e:
            logger.error(f"GitHub upload-file failed: {str(e)}")
            raise HTTPException(status_code=502, detail=f"GitHub API error: {str(e)}")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token expired or invalid")
    if not response.is_success:
        error = response.json()
        raise HTTPException(
            status_code=response.status_code,
            detail=error.get("message", f"Failed to upload {request.file_path}"),
        )

    return response.json()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)