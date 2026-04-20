"""
Architecture Requirements API - Simplified
FastAPI application for storing architecture requirements in Cosmos DB
"""
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
import asyncio
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
from openai import OpenAI, AsyncOpenAI
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage

from cosmos_service import CosmosDBService

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Per-repo lock to serialize file uploads (prevents 409 conflicts from parallel PUTs)
_repo_upload_locks: Dict[str, asyncio.Lock] = {}


class _WorkflowScopeMissing(Exception):
    """Raised when a GitHub token lacks the 'workflow' OAuth scope for .github/workflows/ paths."""
    def __init__(self, reauth_url: str, current_scopes: str, file_path: str = ""):
        self.reauth_url = reauth_url
        self.current_scopes = current_scopes
        self.file_path = file_path
        super().__init__(f"workflow_scope_missing: {current_scopes}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reauth_required": True,
            "reauth_reason": "workflow_scope_missing",
            "reauth_url": self.reauth_url,
            "current_scopes": self.current_scopes,
            "required_scopes": "read:user repo workflow",
            "file_path": self.file_path,
            "skipped": True,
            "message": (
                "GitHub token is missing the 'workflow' OAuth scope. "
                "Redirect the user to reauth_url to re-authorize, then retry the upload."
            ),
        }

# Suppress Azure SDK verbose logging
logging.getLogger('azure').setLevel(logging.ERROR)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)
logging.getLogger('cosmos_service').setLevel(logging.INFO)

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
AZURE_SERVICE_BUS_CODE_QUEUE = os.getenv("AZURE_SERVICE_BUS_CODE_QUEUE", "codequeue")

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
COSMOS_CODE_CONTAINER = os.getenv("COSMOS_CODE_CONTAINER", "code")

# GitHub OAuth Configuration
GITHUB_CLIENT_ID = os.getenv("REACT_APP_GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Bitbucket OAuth Configuration
BITBUCKET_CLIENT_ID = os.getenv("BITBUCKET_CLIENT_ID")
BITBUCKET_CLIENT_SECRET = os.getenv("BITBUCKET_CLIENT_SECRET")

# Azure DevOps OAuth Configuration
AZURE_DEVOPS_CLIENT_ID = os.getenv("AZURE_DEVOPS_CLIENT_ID")
AZURE_DEVOPS_CLIENT_SECRET = os.getenv("AZURE_DEVOPS_CLIENT_SECRET")

# GitLab OAuth Configuration
GITLAB_CLIENT_ID = os.getenv("GITLAB_CLIENT_ID")
GITLAB_CLIENT_SECRET = os.getenv("GITLAB_CLIENT_SECRET")


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
    typeOfData: Optional[Any] = None  # Can be string or array
    typeOfUsers: Optional[Any] = None  # Can be string or array
    numberOfScreens: Optional[Any] = None


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
    tenantId: str = Field(..., description="Tenant ID")
    applicationName: str = Field(..., description="Application name")
    sessionId: str = Field(..., description="Session ID")
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


# ==================== DEVOPS SCRIPT GENERATION MODELS ====================

class DevOpsScriptRequest(BaseModel):
    """Request model for generating DevOps scripts from a single architecture recommendation"""
    tenantId: str = Field(..., description="Tenant ID")
    sessionId: str = Field(..., description="Session ID")
    applicationName: Optional[str] = Field(default="", description="Application name")
    architecture: Dict[str, Any] = Field(..., description="Single architecture recommendation object")
    features: Optional[List[str]] = Field(default_factory=list, description="Application features")
    overview: Optional[str] = Field(default="", description="Application overview")


class DevOpsScript(BaseModel):
    """Represents a single generated DevOps script"""
    fileName: str = Field(..., description="Script file name with extension (e.g., main.tf, deploy.sh)")
    category: str = Field(..., description="Category: iac, cicd, deployment, configuration, monitoring, security, networking, database, helper")
    language: str = Field(..., description="Script language/format: terraform, bash, powershell, yaml, json, hcl, dockerfile, etc.")
    content: str = Field(..., description="Complete script content")
    description: str = Field(..., description="What this script does")
    executionOrder: Optional[int] = Field(default=None, description="Suggested execution order (1 = first)")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Other script fileNames this depends on")


class DevOpsServiceCost(BaseModel):
    """Cost estimate for a single cloud service"""
    service: str = Field(..., description="Cloud service name (e.g., Cloud Run, Cloud SQL, Lambda, RDS)")
    description: str = Field(default="", description="What this service is used for in the architecture")
    monthlyCostMin: float = Field(default=0, description="Minimum estimated monthly cost in USD")
    monthlyCostMax: float = Field(default=0, description="Maximum estimated monthly cost in USD")
    pricingModel: str = Field(default="", description="Pricing model (pay-per-use, reserved, committed, flat-rate)")
    pricingDetails: str = Field(default="", description="Detailed pricing calculation assumptions (units, rates, expected usage)")
    costOptimizationTips: Optional[str] = Field(default="", description="Tips to reduce cost for this specific service")


class DevOpsEnvironmentCost(BaseModel):
    """Cost estimate for a specific environment"""
    environment: str = Field(..., description="Environment name (dev, staging, production)")
    monthlyCostMin: float = Field(default=0, description="Minimum estimated total monthly cost in USD")
    monthlyCostMax: float = Field(default=0, description="Maximum estimated total monthly cost in USD")
    notes: str = Field(default="", description="Environment-specific cost notes (e.g., dev uses smaller instances)")


class DevOpsCostBreakdown(BaseModel):
    """Comprehensive cost breakdown for the architecture"""
    totalMonthlyCostMin: float = Field(default=0, description="Total minimum monthly cost in USD (production)")
    totalMonthlyCostMax: float = Field(default=0, description="Total maximum monthly cost in USD (production)")
    currency: str = Field(default="USD", description="Currency")
    serviceCosts: List[DevOpsServiceCost] = Field(default_factory=list, description="Per-service cost breakdown")
    environmentCosts: List[DevOpsEnvironmentCost] = Field(default_factory=list, description="Per-environment cost estimates")
    assumptions: str = Field(default="", description="Key pricing assumptions (region, usage patterns, reserved vs on-demand)")
    costOptimizationSummary: str = Field(default="", description="Overall cost optimization recommendations")
    freeTrialNotes: Optional[str] = Field(default="", description="Free tier or trial credits applicable")


class RequiredSecret(BaseModel):
    """A single CI/CD secret required by the generated scripts — value NOT included, to be filled in by the user"""
    name: str = Field(..., description="Secret key name as referenced in the pipeline (e.g. GCP_SA_KEY, AWS_ACCESS_KEY_ID)")
    description: str = Field(..., description="Human-readable explanation of what this secret is and where to obtain it")
    required: bool = Field(default=True, description="Whether the secret is mandatory for the pipeline to run")
    example: str = Field(default="", description="Non-sensitive format hint or placeholder (e.g. 'my-project-123', '{\"type\":\"service_account\",...}'). Never a real value.")


class DevOpsScriptResponse(BaseModel):
    """Response model for DevOps script generation"""
    success: bool
    message: str = Field(..., description="Response message")
    applicationName: str = Field(default="", description="Application name")
    architectureName: str = Field(default="", description="Architecture name")
    cloudProvider: str = Field(default="", description="Cloud provider")
    iacTool: str = Field(default="", description="Primary IaC tool used (Terraform, CloudFormation, Bicep, etc.)")
    cicdPlatform: str = Field(default="", description="CI/CD platform used (GitHub Actions, GitLab CI, etc.)")
    scripts: List[DevOpsScript] = Field(default_factory=list, description="All generated scripts")
    executionGuide: str = Field(default="", description="Step-by-step guide to execute all scripts in order")
    prerequisites: List[str] = Field(default_factory=list, description="Required tools and access before running scripts")
    services: List[str] = Field(default_factory=list, description="All cloud services provisioned")
    secrets: List[RequiredSecret] = Field(default_factory=list, description="All CI/CD secrets required by the generated scripts — names only, values to be supplied by the user")
    costBreakdown: Optional[DevOpsCostBreakdown] = Field(default=None, description="Detailed cost breakdown per service and environment")
    estimatedCost: Optional[str] = Field(default="", description="Estimated monthly cost summary")
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
        designs_container=COSMOS_DESIGNS_CONTAINER,
        code_container=COSMOS_CODE_CONTAINER
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
        
        logger.info(f"[CODE API] Incoming request: designId={request.designId}, tenantId={request.tenantId}, userId={user_id}")
        
        # Check if generated code already exists for this design
        if request.designId:
            try:
                logger.info(f"[CODE API] Looking for existing code with designId={request.designId}, tenantId={request.tenantId}")
                
                existing_code = await db.get_generated_code_by_design_id(
                    design_id=request.designId,
                    tenant_id=request.tenantId
                )
                
                if existing_code:
                    logger.info(f"[CODE API] Found existing code for designId={request.designId}")
                    logger.info(f"[CODE API] Document id={existing_code.get('id')}, type={existing_code.get('type')}, requestType={existing_code.get('requestType')}")
                    logger.info(f"[CODE API] Keys in response: {list(existing_code.keys())}")
                    logger.info(f"[CODE API] generatedFiles count: {len(existing_code.get('generatedFiles', []))}")
                    logger.info(f"[CODE API] technologies: {existing_code.get('technologies', [])}")
                    logger.info(f"[CODE API] Document size: ~{len(str(existing_code))} chars")
                    
                    # Extract generated files and metadata from stored code
                    # Data may be nested under 'codeGeneration' key
                    code_data = existing_code.get('codeGeneration', existing_code)
                    generated_files = code_data.get('generatedFiles', [])
                    categorized_files = code_data.get('categorizedFiles', {})
                    project_structure = code_data.get('projectStructure', {})
                    technologies = code_data.get('technologies', [])
                    instructions = code_data.get('instructions', '')
                    
                    # Convert to proper models
                    files = [GeneratedFile(path=f.get('path', ''), content=f.get('content', '')) for f in generated_files]
                    
                    logger.info(f"[CODE API] Returning {len(files)} files to client for designId={request.designId}")
                    
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
                    logger.warning(f"[CODE API] No document returned from Cosmos for designId={request.designId}, tenantId={request.tenantId}")
            except Exception as e:
                logger.error(f"[CODE API] Error checking for existing generated code: {str(e)}", exc_info=True)
        
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


# ==================== DEVOPS SCRIPT GENERATION ====================

@app.post("/api/generate-devops-scripts", response_model=DevOpsScriptResponse)
async def generate_devops_scripts(
    request: DevOpsScriptRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: CosmosDBService = Depends(get_cosmos_service)
):
    """
    Generate comprehensive DevOps scripts from a single architecture recommendation.
    
    Analyzes the architecture's infrastructure, technology stack, CI/CD requirements,
    and generates production-ready scripts for:
    - Infrastructure provisioning (Terraform/CloudFormation/Bicep/etc.)
    - CI/CD pipelines (GitHub Actions/GitLab CI/Azure DevOps/etc.)
    - Application deployment scripts (Docker, Kubernetes, serverless deploy)
    - Configuration and environment setup
    - Monitoring and observability setup
    - Security and networking configuration
    - Database migration and setup scripts
    - Helper/utility scripts
    """
    try:
        arch = request.architecture
        arch_name = arch.get("name", "")
        arch_description = arch.get("description", "")
        tech_stack = arch.get("technologyStack", {})
        infra = tech_stack.get("infra", {})
        cicd = tech_stack.get("cicd", {})
        metrics = arch.get("metrics", {})

        cloud_provider = tech_stack.get("cloudProvider", "")
        languages = tech_stack.get("languages", "")
        frameworks = tech_stack.get("frameworks", "")
        runtime = tech_stack.get("runtime", "")

        # Extract infra details
        compute = infra.get("compute", "")
        database = infra.get("database", "")
        cache = infra.get("cache", "")
        messaging = infra.get("messaging", "")
        storage = infra.get("storage", "")
        api_gateway = infra.get("apiGateway", "")
        authentication = infra.get("authentication", "")
        security = infra.get("security", "")
        networking = infra.get("networking", "")
        monitoring = infra.get("monitoring", "")
        logging_infra = infra.get("logging", "")

        # Extract CI/CD details
        pipeline = cicd.get("pipeline", "")
        containerization = cicd.get("containerization", "")
        testing = cicd.get("testing", "")
        iac = cicd.get("iac", "")

        app_name = request.applicationName or arch_name or "application"
        features_str = "\n".join([f"  - {f}" for f in request.features]) if request.features else "  - No specific features listed"
        overview = request.overview or arch_description

        logger.info(f"Generating DevOps scripts for: {app_name} ({arch_name}) on {cloud_provider}")

        if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
            raise HTTPException(status_code=500, detail="OpenAI API is not configured")

        # Build compact infra summary — only include services that exist
        infra_lines = []
        for label, val in [("Compute", compute), ("Database", database), ("Cache", cache),
                           ("Messaging", messaging), ("Storage", storage), ("API Gateway", api_gateway),
                           ("Auth", authentication), ("Security", security), ("Networking", networking),
                           ("Monitoring", monitoring), ("Logging", logging_infra)]:
            if val:
                infra_lines.append(f"- {label}: {val}")
        infra_block = "\n".join(infra_lines) if infra_lines else "- See architecture description"

        cicd_lines = []
        for label, val in [("Pipeline", pipeline), ("Containers", containerization),
                           ("Testing", testing), ("IaC", iac)]:
            if val:
                cicd_lines.append(f"- {label}: {val}")
        cicd_block = "\n".join(cicd_lines) if cicd_lines else "- Choose best for cloud provider"

        system_prompt = (
            "You are a senior DevOps engineer. Every script you generate will be executed verbatim "
            "by a real CI/CD runner with no human fixups. If a command fails, the pipeline fails. "
            "Apply every rule below with zero exceptions. Return ONLY valid JSON.\n\n"

            "RULE 1 — CI/CD ENV VARS MUST NEVER BE BLANK:\n"
            "The env: block of every workflow job or step MUST source each variable from the platform "
            "secret store. An empty value (e.g. 'GCP_PROJECT: ' with nothing after the colon) is "
            "FORBIDDEN — it causes fatal failures such as 'gcr.io//app:sha' (invalid Docker reference) "
            "and 'set -u: GCP_PROJECT: unbound variable'. \n"
            "GitHub Actions: ${{ secrets.VAR }}. Azure Pipelines: $(VAR). "
            "GitLab CI / Bitbucket Pipelines: $VAR (from masked CI/CD variables).\n\n"

            "RULE 2 — TEST STEPS MUST BE GUARDED BY FILE-EXISTENCE CHECKS:\n"
            "NEVER run a test command unconditionally. With set -euo pipefail active, running "
            "'go test ./...' in a directory that has no go.mod, or 'npm test' where no package.json "
            "exists, will immediately fail and abort the pipeline. "
            "Guard every test step with the correct existence check for that language:\n"
            "  Go:            if [ -f go.mod ]; then go test ./... -v; else echo 'No go.mod, skipping'; fi\n"
            "  Node (npm):    if [ -f package.json ]; then npm ci && npm test; else echo 'Skipping'; fi\n"
            "  Node (yarn):   if [ -f yarn.lock ]; then yarn --frozen-lockfile && yarn test; else echo 'Skipping'; fi\n"
            "  Python:        if [ -f pytest.ini ] || [ -f setup.cfg ] || [ -d tests ]; then pytest -v; else echo 'Skipping'; fi\n"
            "  Java (Maven):  if [ -f pom.xml ]; then mvn test -q; else echo 'Skipping'; fi\n"
            "  Java (Gradle): if [ -f build.gradle ] || [ -f build.gradle.kts ]; then ./gradlew test; else echo 'Skipping'; fi\n"
            "  Ruby:          if [ -f Gemfile ]; then bundle install && bundle exec rspec; else echo 'Skipping'; fi\n"
            "  .NET:          find . -maxdepth 2 \\( -name '*.sln' -o -name '*.csproj' \\) | grep -q . && dotnet test || echo 'Skipping'\n"
            "  Rust:          if [ -f Cargo.toml ]; then cargo test; else echo 'Skipping'; fi\n\n"

            "RULE 3 — SHELL SCRIPT STRUCTURE AND VAR VALIDATION:\n"
            "Every .sh file must open with:\n"
            "  #!/usr/bin/env bash\n"
            "  set -euo pipefail\n"
            "Then immediately validate every required env var with the Bash null-check expansion "
            "(this is safe under set -u and gives a clear error message):\n"
            "  : ${GCP_PROJECT:?'GCP_PROJECT is required. Set it in .env or CI secrets.'}\n"
            "Build compound values (image tags, URLs) ONLY after all component vars are validated.\n\n"

            "RULE 4 — DOCKER IMAGE REFERENCES:\n"
            "Construct derived values only after validating every component:\n"
            "  : ${GCP_PROJECT:?'required'}\n"
            "  : ${APP_NAME:?'required'}\n"
            "  : ${IMAGE_TAG:?'required'}\n"
            "  IMAGE=\"gcr.io/${GCP_PROJECT}/${APP_NAME}:${IMAGE_TAG}\"   # safe — all vars validated\n"
            "Never inline the construction before the guards.\n\n"

            "RULE 5 — COMMANDS MUST MATCH THE ACTUAL LANGUAGE AND RUNTIME:\n"
            "Read the runtime, languages, and frameworks fields. Use ONLY the matching toolchain commands. "
            "Do not emit 'go test' for a Node project or 'npm install' for a Python project. "
            "The runtime setup step in CI/CD MUST use the correct official action pinned to a major version:\n"
            "  Go:      actions/setup-go@v5     (go-version from go.mod or specify e.g. '1.22')\n"
            "  Node:    actions/setup-node@v4   (node-version, e.g. '20')\n"
            "  Python:  actions/setup-python@v5 (python-version, e.g. '3.12')\n"
            "  Java:    actions/setup-java@v4   (distribution: 'temurin', java-version)\n"
            "  .NET:    actions/setup-dotnet@v4\n"
            "  Ruby:    ruby/setup-ruby@v1\n"
            "Cloud auth MUST use the official action:\n"
            "  GCP:   google-github-actions/auth@v2 — credentials_json MUST go under 'with:', NOT 'env:'.\n"
            "    The action requires EXACTLY ONE of workload_identity_provider OR credentials_json as a 'with:' input.\n"
            "    Passing it via 'env:' is silently ignored and causes this fatal error:\n"
            "      'must specify exactly one of workload_identity_provider or credentials_json'\n"
            "    CORRECT:\n"
            "      - uses: google-github-actions/auth@v2\n"
            "        with:\n"
            "          credentials_json: '${{ secrets.GCP_SA_KEY }}'\n"
            "    WRONG (triggers the error above):\n"
            "      - uses: google-github-actions/auth@v2\n"
            "        env:\n"
            "          credentials_json: ${{ secrets.GCP_SA_KEY }}   # env: is NOT read by this action\n"
            "    Also WRONG: specifying both workload_identity_provider AND credentials_json in the same step.\n"
            "  AWS:   aws-actions/configure-aws-credentials@v4\n"
            "  Azure: azure/login@v2\n\n"

            "RULE 6 — GITHUB ACTIONS WORKFLOW REQUIRED STRUCTURE:\n"
            "on: MUST include push (branches: [main]) AND workflow_dispatch with an environment input. "
            "Every job MUST declare runs-on. Repo checkout MUST use actions/checkout@v4. "
            "No job may reference a secret that is not listed in the env: or with: block of that job/step.\n\n"

            "RULE 7 — SYNTAX CORRECTNESS — HCL ESCAPING IS CRITICAL:\n"
            "Every file must be valid for its format with no modifications.\n"
            "YAML: correct indentation (2 spaces), no tabs, booleans unquoted, strings quoted when containing ':'.\n"
            "Bash: all variable expansions double-quoted, no undefined vars under set -u.\n"
            "Dockerfile: FROM first, valid instruction order, no shell variable expansion in COPY paths.\n"
            "HCL / Terraform — the following rules are MANDATORY:\n"
            "  1. A backslash (\\) inside a double-quoted HCL string is an ESCAPE CHARACTER. "
            "'\"\\\"' is a FATAL syntax error: it consumes the closing quote and turns EVERY subsequent "
            "line in the file into a continuation of that broken string, producing hundreds of "
            "'Invalid multi-line string' errors. A literal backslash in HCL requires '\\\\'.\n"
            "  2. GCP project IDs contain ONLY lowercase letters, digits, and hyphens — they NEVER "
            "contain backslashes, forward slashes, or any character that needs replace(). "
            "NEVER call replace(var.project, \"\\\\\\\\\", \"-\") or replace(var.project, \"/\", \"-\"). "
            "Use the project ID directly in interpolation: \"${var.app_name}-${var.project}\".\n"
            "  3. Bucket names and resource names must be constructed with simple string interpolation "
            "only: \"${var.app_name}-suffix-${var.project}\". No replace(), no regex, no backslash literals.\n"
            "  4. All HCL blocks must be closed. Attribute syntax is: name = value (with = not :).\n"
            "  5. String interpolation uses \"${...}\" (HCL) — never \"${{...}}\" or \"$(...)\" inside .tf files.\n"
            "  6. Never mix YAML, shell, or Python syntax into .tf files.\n\n"

            "RULE 8 — COST: ALWAYS FREE TIER / LOWEST COST:\n"
            "Use free-tier eligible SKUs (f1-micro, t2.micro, B1s, Cloud Run free tier, etc.). "
            "Prefer serverless / scale-to-zero over always-on VMs. Set min-instances=0. "
            "Use free-tier databases and single-region deployments for dev/staging."
        )

        user_prompt = f"""Generate DevOps scripts for this architecture.

App: {app_name}
Architecture: {arch_name}
Cloud: {cloud_provider}
Runtime: {runtime} | Languages: {languages} | Frameworks: {frameworks}
Features: {', '.join(request.features) if request.features else 'N/A'}
Availability: {metrics.get('availability', 'N/A')} | Cost range: {metrics.get('cost', 'N/A')}

Infrastructure:
{infra_block}

CI/CD:
{cicd_block}

Generate these scripts. Each MUST be complete, immediately runnable, and pass without modification:
1. IaC (category:"iac") — main infra + variables + outputs + networking + IAM. Use the best tool for the cloud provider.
2. CI/CD pipeline (category:"cicd") — full build, test, and deploy pipeline. No echo-stub steps.
3. Deployment (category:"deployment") — deploy.sh, rollback.sh, health-check.sh
4. Config (category:"configuration") — .env.example, Dockerfile (if containerised), env-specific var files
5. Helpers (category:"helper") — cleanup.sh, status.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. CI/CD FILE NAMING AND TRIGGERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• The primary CI/CD file MUST be named EXACTLY ".github/workflows/deploy.yml".
• If Azure DevOps is used, ALSO generate "azure-pipelines.yml".
• If GitLab is the source control, ALSO generate ".gitlab-ci.yml".
• If Bitbucket is the source control, ALSO generate "bitbucket-pipelines.yml".
• GitHub Actions on: MUST include BOTH:
    push:
      branches: [main]
    workflow_dispatch:
      inputs:
        environment:
          description: 'Target environment'
          required: false
          default: 'dev'
          type: choice
          options: [dev, staging, production]
• For Azure Pipelines: include trigger: and a parameters: block for manual environment selection.
• For GitLab CI: add when: manual to every deploy job.
• For Bitbucket Pipelines: add a custom: section for manual pipeline triggers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. CI/CD ENV VARS — SECRETS ONLY, NEVER BLANK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORBIDDEN — these patterns WILL cause failures at runtime:
  env:
    GCP_PROJECT:          # blank  → gcr.io//app:sha  → invalid Docker reference
    GCP_REGION:           # blank  → missing region argument → command not found
    GCP_SA_KEY:           # blank  → auth failure
    IMAGE_REGISTRY:       # blank  → malformed image URL

REQUIRED — every deployment-specific variable MUST reference the platform secret store:
  GitHub Actions:       ${{{{ secrets.GCP_PROJECT }}}}, ${{{{ secrets.AWS_ACCOUNT_ID }}}}, ${{{{ secrets.AZURE_SUBSCRIPTION_ID }}}}
  Azure Pipelines:      $(GCP_PROJECT), $(AWS_ACCOUNT_ID)
  GitLab CI:            $GCP_PROJECT, $AWS_ACCOUNT_ID  (masked CI/CD variable)
  Bitbucket Pipelines:  $GCP_PROJECT, $AWS_ACCOUNT_ID  (repository variable)

This applies to ALL of: cloud project/account IDs, regions, registry hostnames, service account
keys, cluster names, subscription IDs, connection strings, and any deployment identifier.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C. TEST STEPS — ALWAYS GUARDED BY FILE-EXISTENCE CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
With set -euo pipefail, running 'go test ./...' where no go.mod exists produces:
  "pattern ./...: directory prefix . does not contain main module" → exit 1 → pipeline aborted.
The same applies to every other language: 'npm test' without package.json, 'mvn test' without
pom.xml, etc. ALL test commands MUST be wrapped with the appropriate guard:

  Go:            if [ -f go.mod ]; then go test ./... -v; else echo "No go.mod — skipping tests"; fi
  Node (npm):    if [ -f package.json ]; then npm ci && npm test; else echo "No package.json — skipping"; fi
  Node (yarn):   if [ -f yarn.lock ]; then yarn --frozen-lockfile && yarn test; else echo "No yarn.lock — skipping"; fi
  Python:        if [ -f pytest.ini ] || [ -f setup.cfg ] || [ -d tests ]; then pytest -v; else echo "No tests — skipping"; fi
  Java (Maven):  if [ -f pom.xml ]; then mvn test -q; else echo "No pom.xml — skipping"; fi
  Java (Gradle): if [ -f build.gradle ] || [ -f build.gradle.kts ]; then ./gradlew test; else echo "No build.gradle — skipping"; fi
  Ruby:          if [ -f Gemfile ]; then bundle install && bundle exec rspec; else echo "No Gemfile — skipping"; fi
  .NET:          find . -maxdepth 2 \\( -name '*.sln' -o -name '*.csproj' \\) | grep -q . && dotnet test || echo "No project — skipping"
  Rust:          if [ -f Cargo.toml ]; then cargo test; else echo "No Cargo.toml — skipping"; fi

In GitHub Actions, this can be implemented as an inline shell guard (preferred) or as a step
"if:" condition checking a file with hashFiles().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D. SHELL SCRIPT VAR VALIDATION PATTERN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every .sh script MUST open with:
  #!/usr/bin/env bash
  set -euo pipefail

Then validate EVERY required variable using Bash null-check expansion before first use:
  : ${{GCP_PROJECT:?'GCP_PROJECT is required — set it in .env or CI secrets.'}}
  : ${{GCP_REGION:?'GCP_REGION is required.'}}
  : ${{APP_NAME:?'APP_NAME is required.'}}
  : ${{IMAGE_TAG:?'IMAGE_TAG is required.'}}

Build compound values ONLY after all component vars are validated:
  IMAGE="gcr.io/${{GCP_PROJECT}}/${{APP_NAME}}:${{IMAGE_TAG}}"   # safe — every component validated above

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
E. RUNTIME-SPECIFIC SETUP — USE THE CORRECT ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use ONLY the build/test/install commands that match the runtime and language above.
Do NOT emit Go commands for a Node project, npm commands for a Python project, etc.
The setup step MUST use the correct official action, pinned to the latest major version:
  Go:      uses: actions/setup-go@v5      with: {{ go-version: '1.22' }}
  Node:    uses: actions/setup-node@v4    with: {{ node-version: '20' }}
  Python:  uses: actions/setup-python@v5  with: {{ python-version: '3.12' }}
  Java:    uses: actions/setup-java@v4    with: {{ distribution: 'temurin', java-version: '21' }}
  .NET:    uses: actions/setup-dotnet@v4
  Ruby:    uses: ruby/setup-ruby@v1       with: {{ bundler-cache: true }}

Cloud authentication MUST use the official action:
  GCP:   uses: google-github-actions/auth@v2
         with:
           credentials_json: '${{{{ secrets.GCP_SA_KEY }}}}'

  CRITICAL — google-github-actions/auth@v2 RULES (failure to follow = always-failing pipeline):
  • credentials_json MUST be placed under 'with:' — NEVER under 'env:'. The action does NOT read 'env:' inputs.
  • The action requires EXACTLY ONE of 'workload_identity_provider' OR 'credentials_json'. Not both, not neither.
  • If either is missing or put in the wrong block, the action always fails with:
      "must specify exactly one of workload_identity_provider or credentials_json"

  CORRECT:
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v2
      with:
        credentials_json: '${{{{ secrets.GCP_SA_KEY }}}}'

  WRONG — each pattern below causes the error above:
    # Missing with: entirely
    - uses: google-github-actions/auth@v2

    # Key placed in env: instead of with:
    - uses: google-github-actions/auth@v2
      env:
        credentials_json: '${{{{ secrets.GCP_SA_KEY }}}}'

    # Both inputs specified simultaneously
    - uses: google-github-actions/auth@v2
      with:
        workload_identity_provider: 'projects/123/...'
        credentials_json: '${{{{ secrets.GCP_SA_KEY }}}}'

  AWS:   uses: aws-actions/configure-aws-credentials@v4
  Azure: uses: azure/login@v2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
F. SYNTAX CORRECTNESS — READ EVERY BULLET, EACH PREVENTS A REAL FAILURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HCL / TERRAFORM (most dangerous — one wrong character corrupts the entire file):
• BACKSLASH IN HCL STRINGS IS AN ESCAPE CHARACTER. Writing replace(var.x, "\" — with a SINGLE
  backslash before the closing quote — is a fatal parse error. Terraform treats the \" as an
  escaped quote, so the string never closes, and EVERY remaining line in the file becomes part
  of that one broken string, producing dozens of "Invalid multi-line string" errors.
  A literal backslash requires "\\". However:
• GCP project IDs contain ONLY [a-z0-9-]. They NEVER contain backslashes or forward slashes.
  NEVER use replace(var.project, "...", "-") for any backslash or slash. Use the project ID
  directly: "${{var.app_name}}-${{var.project}}". This is always safe and correct.
• Bucket names, resource names, and all name attributes MUST use plain string interpolation
  only: "${{var.app_name}}-<suffix>-${{var.project}}". No replace(), no regex, no escape sequences.
• All HCL blocks MUST be closed (matching braces). Attribute syntax is `name = value`.
• String interpolation in .tf files uses "${{...}}" — NEVER "$(...)" or "${{{{ }}}}", which belong
  to shell and GitHub Actions respectively.
• Comments in HCL use # or // — never /* inside a string value.

YAML:
• 2-space indentation, no tabs. Booleans unquoted (true/false). Strings containing `:` or `#`
  MUST be quoted. No := anywhere — it is invalid YAML. Multiline strings use | or >.

Bash:
• All variable expansions double-quoted: "$VAR", "${{VAR}}". No undefined vars under set -u.
• Use [[ ]] for string comparisons, [ ] for POSIX. Do not mix.

Dockerfile:
• FROM is first (except ARG for build args). No shell variable expansion in COPY paths.

JSON:
• Double-quoted keys only, no trailing commas, no comments.
• Do not borrow syntax from another language into a file of a different type.

List every required secret/variable in executionGuide and prerequisites.

COST PRIORITY: Always use FREE TIER and LOWEST COST options:
- Use free-tier eligible instance sizes/SKUs (e.g., f1-micro, t2.micro, B1s, e2-micro, Cloud Run free tier)
- Use serverless/pay-per-use services over always-on instances wherever possible
- Use free-tier databases (Cloud SQL free trial, RDS free tier, Cosmos DB free tier, Firestore free tier)
- Use free-tier caching (small Redis, or skip cache in dev and use application-level caching)
- Prefer managed serverless (Cloud Functions, Lambda, Azure Functions) over VMs/containers for low traffic
- Set min-instances=0 so services scale to zero when idle
- Use single-region, no HA replicas for dev/staging
- In IaC variables, set defaults to the smallest/cheapest tier available
- Add comments in scripts showing the free-tier limits and when paid pricing kicks in

Also provide a costBreakdown with per-service costs (realistic, based on cloud pricing) and per-environment totals (dev/staging/prod).

Return JSON:
{{
  "applicationName": "{app_name}",
  "architectureName": "{arch_name}",
  "cloudProvider": "",
  "iacTool": "",
  "cicdPlatform": "",
  "scripts": [
    {{
      "fileName": "path/file.ext",
      "category": "iac|cicd|deployment|configuration|helper",
      "language": "terraform|bash|yaml|json|hcl|dockerfile",
      "content": "full script content",
      "description": "what it does",
      "executionOrder": 1,
      "dependencies": []
    }}
  ],
  "executionGuide": "step-by-step execution instructions",
  "prerequisites": ["required tools"],
  "services": ["cloud services used"],
  "secrets": [
    {{
      "name": "SECRET_NAME",
      "description": "What this secret is and where to obtain it (e.g. GCP Service Account Key — download JSON from IAM & Admin > Service Accounts)",
      "required": true,
      "example": "non-sensitive format hint only — e.g. my-project-123 or {{\"type\":\"service_account\",...}}"
    }}
  ],
  "costBreakdown": {{
    "totalMonthlyCostMin": 0.0,
    "totalMonthlyCostMax": 0.0,
    "currency": "USD",
    "serviceCosts": [
      {{
        "service": "name",
        "description": "purpose",
        "monthlyCostMin": 0.0,
        "monthlyCostMax": 0.0,
        "pricingModel": "pay-per-use|reserved",
        "pricingDetails": "calculation details",
        "costOptimizationTips": "tip"
      }}
    ],
    "environmentCosts": [
      {{"environment": "dev", "monthlyCostMin": 0.0, "monthlyCostMax": 0.0, "notes": ""}},
      {{"environment": "staging", "monthlyCostMin": 0.0, "monthlyCostMax": 0.0, "notes": ""}},
      {{"environment": "production", "monthlyCostMin": 0.0, "monthlyCostMax": 0.0, "notes": ""}}
    ],
    "assumptions": "",
    "costOptimizationSummary": "",
    "freeTrialNotes": ""
  }},
  "estimatedCost": "$X-$Y/month production"
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
                max_completion_tokens=MAX_OUTPUT_TOKENS
            )

            message = completion.choices[0].message
            if hasattr(message, "parsed") and message.parsed is not None:
                result = message.parsed
            else:
                content = message.content or ""
                if not content.strip():
                    raise HTTPException(status_code=502, detail="AI service returned empty response for DevOps scripts")
                result = json.loads(content)

        except HTTPException:
            raise
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse OpenAI response as JSON: {str(je)}")
            raise HTTPException(status_code=502, detail="Failed to parse AI response for DevOps scripts")
        except Exception as openai_err:
            logger.error(f"OpenAI error during DevOps script generation: {str(openai_err)}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI DevOps script generation failed: {str(openai_err)}")

        # Parse scripts from result
        raw_scripts = result.get("scripts", [])
        scripts = []
        for s in raw_scripts:
            if isinstance(s, dict) and s.get("fileName") and s.get("content"):
                scripts.append(DevOpsScript(
                    fileName=s.get("fileName", ""),
                    category=s.get("category", "iac"),
                    language=s.get("language", ""),
                    content=s.get("content", ""),
                    description=s.get("description", ""),
                    executionOrder=s.get("executionOrder"),
                    dependencies=s.get("dependencies", [])
                ))

        # Sort scripts by execution order
        scripts.sort(key=lambda x: x.executionOrder if x.executionOrder is not None else 999)

        iac_tool = result.get("iacTool", "")
        cicd_platform = result.get("cicdPlatform", "")
        services = result.get("services", [])
        execution_guide = result.get("executionGuide", "")
        prerequisites = result.get("prerequisites", [])
        estimated_cost = result.get("estimatedCost", "")
        detected_cloud = result.get("cloudProvider", cloud_provider)

        # Parse required secrets
        secrets = [
            RequiredSecret(
                name=s.get("name", ""),
                description=s.get("description", ""),
                required=bool(s.get("required", True)),
                example=s.get("example", ""),
            )
            for s in result.get("secrets", [])
            if isinstance(s, dict) and s.get("name")
        ]

        # Parse cost breakdown
        cost_breakdown_data = result.get("costBreakdown", {})
        cost_breakdown = None
        if cost_breakdown_data and isinstance(cost_breakdown_data, dict):
            service_costs = [
                DevOpsServiceCost(
                    service=sc.get("service", ""),
                    description=sc.get("description", ""),
                    monthlyCostMin=float(sc.get("monthlyCostMin", 0)),
                    monthlyCostMax=float(sc.get("monthlyCostMax", 0)),
                    pricingModel=sc.get("pricingModel", ""),
                    pricingDetails=sc.get("pricingDetails", ""),
                    costOptimizationTips=sc.get("costOptimizationTips", "")
                )
                for sc in cost_breakdown_data.get("serviceCosts", [])
                if isinstance(sc, dict) and sc.get("service")
            ]
            env_costs = [
                DevOpsEnvironmentCost(
                    environment=ec.get("environment", ""),
                    monthlyCostMin=float(ec.get("monthlyCostMin", 0)),
                    monthlyCostMax=float(ec.get("monthlyCostMax", 0)),
                    notes=ec.get("notes", "")
                )
                for ec in cost_breakdown_data.get("environmentCosts", [])
                if isinstance(ec, dict) and ec.get("environment")
            ]
            cost_breakdown = DevOpsCostBreakdown(
                totalMonthlyCostMin=float(cost_breakdown_data.get("totalMonthlyCostMin", 0)),
                totalMonthlyCostMax=float(cost_breakdown_data.get("totalMonthlyCostMax", 0)),
                currency=cost_breakdown_data.get("currency", "USD"),
                serviceCosts=service_costs,
                environmentCosts=env_costs,
                assumptions=cost_breakdown_data.get("assumptions", ""),
                costOptimizationSummary=cost_breakdown_data.get("costOptimizationSummary", ""),
                freeTrialNotes=cost_breakdown_data.get("freeTrialNotes", "")
            )

        logger.info(f"Generated {len(scripts)} DevOps scripts for {app_name} using {iac_tool} + {cicd_platform}")

        return DevOpsScriptResponse(
            success=True,
            message=f"Successfully generated {len(scripts)} DevOps scripts for {app_name}",
            applicationName=app_name,
            architectureName=arch_name,
            cloudProvider=detected_cloud,
            iacTool=iac_tool,
            cicdPlatform=cicd_platform,
            scripts=scripts,
            executionGuide=execution_guide,
            prerequisites=prerequisites,
            services=services,
            secrets=secrets,
            costBreakdown=cost_breakdown,
            estimatedCost=estimated_cost,
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating DevOps scripts: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate DevOps scripts: {str(e)}"
        )


# ==================== PUSH DEVOPS SCRIPTS TO REPO ====================

class PushScriptsFile(BaseModel):
    fileName: str = Field(..., description="File path inside the repo (e.g. .github/workflows/deploy.yml)")
    content: str = Field(..., description="Raw file content")


class PushDevOpsScriptsRequest(BaseModel):
    token: str = Field(..., description="OAuth access token for the target provider")
    provider: str = Field(..., description="Source control provider: github, bitbucket, azure-devops, gitlab")
    repo_full_name: str = Field(..., description="Repository identifier (owner/repo for GitHub/Bitbucket, org/project/repo for Azure DevOps, project_id for GitLab)")
    branch: str = Field("main", description="Target branch")
    scripts: List[PushScriptsFile] = Field(..., description="List of files to push (from generate-devops-scripts response)")
    commit_message: str = Field("Add DevOps pipeline and infrastructure scripts", description="Commit message")
    # GitLab-specific
    gitlab_url: Optional[str] = Field("https://gitlab.com", description="GitLab instance URL (GitLab only)")
    # Azure DevOps-specific
    organization: Optional[str] = Field(None, description="Azure DevOps organization (Azure DevOps only)")
    project: Optional[str] = Field(None, description="Azure DevOps project (Azure DevOps only)")


@app.post("/api/devops-scripts/push")
async def push_devops_scripts(request: PushDevOpsScriptsRequest):
    """
    Push generated DevOps scripts to a repository.
    Supports GitHub, Bitbucket, GitLab, and Azure DevOps.
    Call this after /api/generate-devops-scripts to commit the generated files.
    """
    import base64 as b64
    import re as _re

    if not request.scripts:
        raise HTTPException(status_code=422, detail="No scripts provided to push")

    provider = request.provider.lower().strip()
    results = []
    errors = []

    if provider == "github":
        _validate_repo_full_name(request.repo_full_name)
        gh_headers = {
            "Authorization": f"Bearer {request.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        import base64 as _b64_push

        # Split into two buckets: regular files first, workflow files second.
        # This guarantees all non-workflow files are committed even when the token
        # lacks the 'workflow' OAuth scope.
        regular_files = []
        workflow_files = []
        for script in request.scripts:
            content_b64 = _b64_push.b64encode(script.content.encode()).decode()
            entry = {"file_path": script.fileName, "content": content_b64}
            if ".github/workflows/" in script.fileName.replace("\\", "/"):
                workflow_files.append(entry)
            else:
                regular_files.append(entry)

        async with httpx.AsyncClient() as client:
            # ── Pass 1: push all regular files ──────────────────────────────
            if regular_files:
                try:
                    push_result = await _github_git_api_push(
                        client=client,
                        gh_headers=gh_headers,
                        repo_full_name=request.repo_full_name,
                        branch=request.branch,
                        files=regular_files,
                        commit_message=request.commit_message,
                    )
                    pushed_set = set(push_result.get("files_pushed", []))
                    results += [
                        {"fileName": s.fileName, "status": "pushed"}
                        for s in request.scripts
                        if s.fileName in pushed_set
                    ]
                except HTTPException as e:
                    errors += [{"fileName": s.fileName, "error": e.detail}
                                for s in request.scripts if s.fileName in {f["file_path"] for f in regular_files}]

            # ── Pass 2: push workflow files separately, AFTER regular files ─
            if workflow_files:
                try:
                    wf_result = await _github_git_api_push(
                        client=client,
                        gh_headers=gh_headers,
                        repo_full_name=request.repo_full_name,
                        branch=request.branch,
                        files=workflow_files,
                        commit_message=f"{request.commit_message} [CI/CD workflows]",
                    )
                    pushed_set_wf = set(wf_result.get("files_pushed", []))
                    results += [
                        {"fileName": s.fileName, "status": "pushed"}
                        for s in request.scripts
                        if s.fileName in pushed_set_wf
                    ]
                    # Workflow files skipped inside _github_git_api_push (scope missing)
                    for skipped in wf_result.get("skipped_workflow_files", []):
                        errors.append({"fileName": skipped["file_path"], "error": skipped})
                        logger.warning(
                            f"[push-devops] '{skipped['file_path']}' skipped — "
                            f"workflow scope missing. reauth_url provided."
                        )
                except _WorkflowScopeMissing as wse:
                    logger.warning(
                        f"[push-devops] workflow scope missing for '{wse.file_path}'"
                    )
                    errors += [{"fileName": f["file_path"], "error": wse.to_dict()}
                                for f in workflow_files]
                except HTTPException as e:
                    errors += [{"fileName": f["file_path"], "error": e.detail}
                                for f in workflow_files]

    elif provider == "bitbucket":
        if not _re.match(r'^[a-zA-Z0-9._\-]+/[a-zA-Z0-9._\-]+$', request.repo_full_name):
            raise HTTPException(status_code=422, detail="repo_full_name must be 'workspace/repo_slug'")
        parts = request.repo_full_name.split("/")
        workspace, repo_slug = parts[0], parts[1]
        url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/src"
        async with httpx.AsyncClient() as client:
            for script in request.scripts:
                try:
                    resp = await client.post(
                        url,
                        data={
                            script.fileName: script.content,
                            "message": f"{request.commit_message}: {script.fileName}",
                            "branch": request.branch,
                        },
                        headers={"Authorization": f"Bearer {request.token}"},
                        timeout=15.0,
                    )
                except httpx.HTTPError:
                    errors.append({"fileName": script.fileName, "error": "Bitbucket API unreachable"})
                    continue

                if resp.is_success or resp.status_code == 201:
                    results.append({"fileName": script.fileName, "status": "pushed"})
                else:
                    err = resp.json() if resp.content else {}
                    msg = err.get("error", {}).get("message", f"HTTP {resp.status_code}") if isinstance(err.get("error"), dict) else str(err)
                    errors.append({"fileName": script.fileName, "error": msg})

    elif provider in ("azure-devops", "azuredevops"):
        if not request.organization or not request.project:
            raise HTTPException(status_code=422, detail="organization and project are required for Azure DevOps")
        if not _re.match(r'^[a-zA-Z0-9._\-]+$', request.organization):
            raise HTTPException(status_code=422, detail="Invalid characters in organization")
        if not _re.match(r'^[a-zA-Z0-9._\- ]+$', request.project):
            raise HTTPException(status_code=422, detail="Invalid characters in project")

        repo_name = request.repo_full_name
        headers = {
            "Authorization": f"Bearer {request.token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            # Get latest ref
            refs_url = f"https://dev.azure.com/{request.organization}/{request.project}/_apis/git/repositories/{repo_name}/refs?filter=heads/{request.branch}&api-version=7.1"
            try:
                refs_resp = await client.get(refs_url, headers=headers, timeout=10.0)
            except httpx.HTTPError:
                raise HTTPException(status_code=502, detail="Azure DevOps API unreachable")

            refs_data = refs_resp.json()
            ref_list = refs_data.get("value", [])
            old_object_id = ref_list[0]["objectId"] if ref_list else "0000000000000000000000000000000000000000"
            change_type = "add" if old_object_id == "0000000000000000000000000000000000000000" else "edit"

            # Push all files in a single commit
            changes = []
            for script in request.scripts:
                changes.append({
                    "changeType": change_type,
                    "item": {"path": f"/{script.fileName}"},
                    "newContent": {"content": script.content, "contentType": "rawtext"},
                })

            push_url = f"https://dev.azure.com/{request.organization}/{request.project}/_apis/git/repositories/{repo_name}/pushes?api-version=7.1"
            push_payload = {
                "refUpdates": [{"name": f"refs/heads/{request.branch}", "oldObjectId": old_object_id}],
                "commits": [{"comment": request.commit_message, "changes": changes}],
            }

            try:
                resp = await client.post(push_url, json=push_payload, headers=headers, timeout=30.0)
            except httpx.HTTPError:
                raise HTTPException(status_code=502, detail="Azure DevOps API unreachable")

            if resp.is_success:
                results = [{"fileName": s.fileName, "status": "pushed"} for s in request.scripts]
            else:
                err = resp.json() if resp.content else {}
                raise HTTPException(status_code=resp.status_code, detail=err.get("message", "Failed to push files"))

    elif provider == "gitlab":
        gitlab_url = (request.gitlab_url or "https://gitlab.com").rstrip("/")
        if not _re.match(r'^https://[a-zA-Z0-9._\-]+(:[0-9]+)?$', gitlab_url):
            raise HTTPException(status_code=422, detail="gitlab_url must be a valid HTTPS URL")

        project_id = request.repo_full_name
        url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/commits"
        headers = {
            "Authorization": f"Bearer {request.token}",
            "Content-Type": "application/json",
        }

        # GitLab supports multi-file commits natively
        actions = []
        for script in request.scripts:
            actions.append({
                "action": "create",
                "file_path": script.fileName,
                "content": script.content,
            })

        commit_payload = {
            "branch": request.branch,
            "commit_message": request.commit_message,
            "actions": actions,
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, json=commit_payload, headers=headers, timeout=30.0)
            except httpx.HTTPError:
                raise HTTPException(status_code=502, detail="GitLab API unreachable")

            if resp.status_code == 400 and "already exists" in (resp.text or "").lower():
                # Retry with "update" action for existing files
                for action in actions:
                    action["action"] = "update"
                try:
                    resp = await client.post(url, json=commit_payload, headers=headers, timeout=30.0)
                except httpx.HTTPError:
                    raise HTTPException(status_code=502, detail="GitLab API unreachable")

            if resp.is_success:
                results = [{"fileName": s.fileName, "status": "pushed"} for s in request.scripts]
            else:
                err = resp.json() if resp.content else {}
                msg = err.get("message", "Failed to push files")
                raise HTTPException(status_code=resp.status_code, detail=msg if isinstance(msg, str) else str(msg))

    else:
        raise HTTPException(status_code=422, detail=f"Unsupported provider '{provider}'. Use: github, bitbucket, azure-devops, gitlab")

    return {
        "success": len(errors) == 0,
        "message": f"Pushed {len(results)}/{len(request.scripts)} files to {request.repo_full_name}",
        "pushed": results,
        "errors": errors,
    }


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
                max_completion_tokens=MAX_OUTPUT_TOKENS
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


@app.get("/api/github/auth-url")
async def github_auth_url(
    redirect_uri: str,
    state: Optional[str] = None,
):
    """
    Returns the GitHub OAuth authorization URL with the correct scopes.
    Includes 'workflow' scope so the token can write .github/workflows/ files.
    Call this when the user needs to (re-)authorize GitHub access.
    """
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GITHUB_CLIENT_ID is not configured")

    import urllib.parse
    params: Dict[str, str] = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "read:user repo workflow",
    }
    if state:
        params["state"] = state
    auth_url = "https://github.com/login/oauth/authorize?" + urllib.parse.urlencode(params)
    return {"auth_url": auth_url, "scopes": "read:user repo workflow"}


@app.post("/api/github/exchange")
async def github_exchange(request: GitHubExchangeRequest):
    """
    Exchange a GitHub OAuth code for an access token.
    The client secret is kept server-side and never exposed to the browser.
    If the resulting token is missing the 'workflow' scope, returns reauth_url
    so the frontend can redirect the user to re-authorize with correct scopes.
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

    # --- Check scopes on the returned token ---
    access_token = token_data.get("access_token", "")
    token_scope = token_data.get("scope", "")
    logger.info(f"[GitHub] Exchange: granted scopes='{token_scope}'")

    if access_token and "workflow" not in token_scope:
        import urllib.parse as _ul
        # Fetch the GitHub username so we can add &login=<user> to the reauth URL.
        # This forces GitHub to show the authorization page (with the new 'workflow'
        # permission listed) instead of silently restoring the old cached token.
        github_login: str = ""
        try:
            async with httpx.AsyncClient() as _uc:
                _ur = await _uc.get(
                    "https://api.github.com/user",
                    headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github.v3+json"},
                    timeout=8.0,
                )
                if _ur.is_success:
                    github_login = _ur.json().get("login", "")
        except Exception:
            pass

        params: Dict[str, str] = {
            "client_id": GITHUB_CLIENT_ID,
            "redirect_uri": request.redirect_uri,
            "scope": "read:user repo workflow",
        }
        if github_login:
            params["login"] = github_login  # forces GitHub to prompt for the extra scope
        reauth_url = "https://github.com/login/oauth/authorize?" + _ul.urlencode(params)
        logger.warning(
            f"[GitHub] Token missing 'workflow' scope (got: '{token_scope}', "
            f"login='{github_login}'). Returning token with reauth_url."
        )
        return {
            **token_data,
            "reauth_required": True,
            "reauth_reason": "workflow_scope_missing",
            "reauth_url": reauth_url,
            "github_login": github_login,
            "current_scopes": token_scope,
            "required_scopes": "read:user repo workflow",
            "message": (
                f"Token granted (scopes: '{token_scope}'). "
                f"The 'workflow' scope is missing — .github/workflows/ files will be skipped "
                f"during upload until the user re-authorizes via reauth_url."
            ),
        }

    return {**token_data, "reauth_required": False}


@app.post("/api/github/check-scopes")
async def github_check_scopes(body: Dict[str, str]):
    """
    Check whether a GitHub token has all required scopes.
    Body: {"token": "<access_token>", "redirect_uri": "<optional>"}
    Returns {"has_workflow_scope": bool, "scopes": str, "reauth_url": str|null}
    """
    token = body.get("token", "")
    redirect_uri = body.get("redirect_uri", "")
    if not token:
        raise HTTPException(status_code=422, detail="token is required")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"},
            timeout=10.0,
        )
    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token is invalid or expired")

    token_scopes = resp.headers.get("X-OAuth-Scopes", "")
    has_workflow = "workflow" in token_scopes
    reauth_url = None
    if not has_workflow and GITHUB_CLIENT_ID:
        import urllib.parse as _ul
        p: Dict[str, str] = {"client_id": GITHUB_CLIENT_ID, "scope": "read:user repo workflow"}
        if redirect_uri:
            p["redirect_uri"] = redirect_uri
        reauth_url = "https://github.com/login/oauth/authorize?" + _ul.urlencode(p)

    return {
        "has_workflow_scope": has_workflow,
        "scopes": token_scopes,
        "reauth_required": not has_workflow,
        "reauth_url": reauth_url,
    }


class CreateRepoRequest(BaseModel):
    token: str = Field(..., description="GitHub access token")
    name: str = Field(..., description="Repository name")
    description: str = Field("", description="Repository description")
    private: bool = Field(False, description="Whether the repo is private")
    auto_init: bool = Field(True, description="Whether to auto-initialize with README")


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
                    "auto_init": True,
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


# ---- GitHub secret encryption (libsodium sealed box via PyNaCl) ----
def _encrypt_github_secret(public_key_b64: str, secret_value: str) -> str:
    """
    Encrypt a plaintext secret value using the repo's GitHub public key.
    GitHub requires libsodium crypto_box_seal (PyNaCl SealedBox).
    Returns base64-encoded ciphertext ready for the GitHub API.
    """
    import base64 as _b64
    try:
        from nacl.public import PublicKey, SealedBox
        from nacl.encoding import RawEncoder
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyNaCl is required for GitHub secret encryption. Run: pip install PyNaCl>=1.5.0"
        )
    public_key_bytes = _b64.b64decode(public_key_b64)
    public_key = PublicKey(public_key_bytes, encoder=RawEncoder)
    sealed_box = SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return _b64.b64encode(encrypted).decode("utf-8")


class SetSecretItem(BaseModel):
    name: str = Field(..., description="Secret name — GitHub allows uppercase letters, digits, underscores; must not start with GITHUB_")
    value: str = Field(..., description="Plaintext secret value (encrypted server-side before sending to GitHub; never logged or stored)")


class SetSecretsRequest(BaseModel):
    token: str = Field(..., description="GitHub OAuth access token (requires repo scope)")
    repo_full_name: str = Field(..., description="owner/repo")
    secrets: List[SetSecretItem] = Field(..., min_length=1, description="List of secrets to set")


class SetSecretResult(BaseModel):
    name: str
    success: bool
    error: Optional[str] = None


@app.post("/api/github/set-secrets")
async def github_set_secrets(request: SetSecretsRequest) -> Dict[str, Any]:
    """
    Encrypt and store one or more secrets in a GitHub repository's Actions secrets store.
    Secret values are encrypted on the server using the repo's libsodium public key
    before being transmitted to GitHub — they are never logged or persisted.
    Requires a token with the 'repo' scope.
    """
    _validate_repo_full_name(request.repo_full_name)

    # Validate secret names: GitHub allows [A-Za-z0-9_], must not start with GITHUB_
    import re as _re_sec
    invalid_names = [
        s.name for s in request.secrets
        if not _re_sec.match(r'^[A-Za-z_][A-Za-z0-9_]*$', s.name) or s.name.upper().startswith("GITHUB_")
    ]
    if invalid_names:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid secret name(s): {invalid_names}. Names must match [A-Za-z_][A-Za-z0-9_]* and must not start with GITHUB_."
        )

    gh_headers = {
        "Authorization": f"Bearer {request.token}",
        "Accept": "application/vnd.github.v3+json",
    }
    api = f"https://api.github.com/repos/{request.repo_full_name}"

    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        # Fetch the repo's public key (one call per request — same key used for all secrets)
        try:
            key_resp = await client.get(
                f"{api}/actions/secrets/public-key",
                headers=gh_headers,
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"GitHub API error fetching public key: {str(e)}")

        if key_resp.status_code == 401:
            raise HTTPException(status_code=401, detail="GitHub token expired or invalid")
        if key_resp.status_code == 403:
            raise HTTPException(status_code=403, detail="Token lacks permission to manage secrets for this repository (repo scope required)")
        if key_resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Repository '{request.repo_full_name}' not found or token has no access")
        if not key_resp.is_success:
            raise HTTPException(status_code=key_resp.status_code, detail=f"Failed to fetch repo public key: {key_resp.text[:300]}")

        key_data = key_resp.json()
        repo_key_id: str = key_data["key_id"]
        repo_public_key: str = key_data["key"]

        # Encrypt and PUT each secret
        for secret in request.secrets:
            try:
                encrypted_value = _encrypt_github_secret(repo_public_key, secret.value)
                put_resp = await client.put(
                    f"{api}/actions/secrets/{secret.name}",
                    json={"encrypted_value": encrypted_value, "key_id": repo_key_id},
                    headers=gh_headers,
                    timeout=10.0,
                )
                if put_resp.status_code in (201, 204):
                    results.append({"name": secret.name, "success": True, "error": None})
                else:
                    err_msg = put_resp.json().get("message", put_resp.text[:200]) if put_resp.content else "Unknown error"
                    results.append({"name": secret.name, "success": False, "error": err_msg})
            except HTTPException:
                raise
            except Exception as e:
                results.append({"name": secret.name, "success": False, "error": str(e)})

    succeeded = [r["name"] for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info(f"[GitHub] set-secrets: repo={request.repo_full_name} set={succeeded} failed={[f['name'] for f in failed]}")

    return {
        "success": len(failed) == 0,
        "message": f"Set {len(succeeded)}/{len(results)} secret(s) on {request.repo_full_name}",
        "set": succeeded,
        "failed": failed,
    }



    token: str = Field(..., description="GitHub access token")
    repo_full_name: str = Field(..., description="owner/repo")
    file_path: str = Field(..., description="Path inside the repo")
    content: str = Field(..., description="Base64-encoded file content")
    message: str = Field("", description="Commit message")
    branch: str = Field("main", description="Target branch")


class UploadFilesRequest(BaseModel):
    token: str = Field(..., description="GitHub access token")
    repo_full_name: str = Field(..., description="owner/repo")
    files: List[Dict[str, str]] = Field(..., description="List of {file_path, content} dicts. content is base64-encoded.")
    message: str = Field("Add generated files", description="Commit message")
    branch: str = Field("main", description="Target branch")


class UploadFileRequest(BaseModel):
    token: str = Field(..., description="GitHub access token")
    repo_full_name: str = Field(..., description="owner/repo")
    file_path: str = Field(..., description="Path of the file in the repo")
    content: str = Field(..., description="Base64-encoded file content")
    message: str = Field("", description="Commit message")
    branch: str = Field("main", description="Target branch")


async def _gh_ensure_repo_ready(
    client: httpx.AsyncClient,
    gh_headers: Dict[str, str],
    api: str,
    repo_full_name: str,
    branch: str,
) -> tuple:
    """
    Ensures the GitHub repo has at least one commit and the target branch exists.
    Returns (parent_sha, base_tree_sha). Retries until the git database is ready.
    """
    import base64 as _b64_git

    # ---- Step A: probe the target branch ----
    for probe_attempt in range(8):
        ref_resp = await client.get(
            f"{api}/git/ref/heads/{branch}", headers=gh_headers, timeout=10.0
        )
        logger.info(
            f"[GH-PUSH] {repo_full_name} branch={branch} "
            f"probe #{probe_attempt+1}: status={ref_resp.status_code} body={ref_resp.text[:200]}"
        )

        if ref_resp.is_success:
            parent_sha = ref_resp.json()["object"]["sha"]
            logger.info(f"[GH-PUSH] Branch '{branch}' found, parent_sha={parent_sha}")

            commit_resp = await client.get(
                f"{api}/git/commits/{parent_sha}", headers=gh_headers, timeout=10.0
            )
            logger.info(
                f"[GH-PUSH] GET commit/{parent_sha}: status={commit_resp.status_code} "
                f"body={commit_resp.text[:200]}"
            )
            base_tree_sha = commit_resp.json().get("tree", {}).get("sha") if commit_resp.is_success else None
            logger.info(f"[GH-PUSH] base_tree_sha={base_tree_sha}")
            return parent_sha, base_tree_sha

        # Branch/repo not ready yet — check if repo just has a different default branch
        repo_info_resp = await client.get(api, headers=gh_headers, timeout=10.0)
        if not repo_info_resp.is_success:
            logger.warning(f"[GH-PUSH] Repo info fetch failed: {repo_info_resp.status_code}")
            await asyncio.sleep(2 * (probe_attempt + 1))
            continue

        repo_info = repo_info_resp.json()
        default_branch = repo_info.get("default_branch", "main")
        logger.info(
            f"[GH-PUSH] Repo default_branch={default_branch}, "
            f"empty={repo_info.get('size', -1) == 0}, size={repo_info.get('size')}"
        )

        # ---- Step B: if repo is empty, create initial commit via Contents API ----
        if repo_info.get("size", 1) == 0 or probe_attempt == 0:
            repo_name = repo_full_name.split('/')[-1]
            readme_b64 = _b64_git.b64encode(
                f"# {repo_name}\n\nAuto-generated repository.\n".encode()
            ).decode()
            init_resp = await client.put(
                f"{api}/contents/README.md",
                json={"message": "Initial commit", "content": readme_b64},
                headers=gh_headers,
                timeout=15.0,
            )
            logger.info(
                f"[GH-PUSH] README init: status={init_resp.status_code} "
                f"body={init_resp.text[:300]}"
            )
            # 409/422 = already initialized by another parallel request — that's OK
            if init_resp.status_code in (200, 201, 409, 422):
                await asyncio.sleep(3)
            else:
                await asyncio.sleep(2 * (probe_attempt + 1))
                continue

        # ---- Step C: if target branch != default, create it ----
        if default_branch != branch:
            def_ref = await client.get(
                f"{api}/git/ref/heads/{default_branch}", headers=gh_headers, timeout=10.0
            )
            logger.info(
                f"[GH-PUSH] Default branch ref: status={def_ref.status_code} "
                f"body={def_ref.text[:200]}"
            )
            if def_ref.is_success:
                def_sha = def_ref.json()["object"]["sha"]
                create_br = await client.post(
                    f"{api}/git/refs",
                    json={"ref": f"refs/heads/{branch}", "sha": def_sha},
                    headers=gh_headers,
                    timeout=10.0,
                )
                logger.info(
                    f"[GH-PUSH] Create branch '{branch}': status={create_br.status_code} "
                    f"body={create_br.text[:200]}"
                )
                await asyncio.sleep(2)
                continue  # re-probe

        await asyncio.sleep(2 * (probe_attempt + 1))

    raise HTTPException(
        status_code=500,
        detail=(
            f"Repo '{repo_full_name}' git database is not ready after multiple retries. "
            f"Check server logs for [GH-PUSH] entries for details."
        ),
    )


async def _github_git_api_push(
    client: httpx.AsyncClient,
    gh_headers: Dict[str, str],
    repo_full_name: str,
    branch: str,
    files: List[Dict[str, str]],
    commit_message: str,
) -> Dict[str, Any]:
    """
    Push one or more files to a GitHub repo using the low-level Git Data API.
    Works reliably for ANY file path including .github/workflows/deploy.yml.
    Handles empty repos (no commits/branches) automatically.
    files: list of {"file_path": "...", "content": "<base64>"}.
    """
    api = f"https://api.github.com/repos/{repo_full_name}"
    logger.info(
        f"[GH-PUSH] Starting push to {repo_full_name} branch={branch} "
        f"files={[f['file_path'] for f in files]}"
    )

    # ------------------------------------------------------------------
    # 0. Log scopes when workflow files are present (diagnostic only).
    #    We no longer preemptively skip — just attempt the push and let
    #    GitHub's actual API response tell us if the scope is missing.
    # ------------------------------------------------------------------
    _skipped_wf: List[Dict[str, Any]] = []
    _reauth_url_wf: str = ""
    _token_scopes_wf: str = ""

    has_workflow_files = any(
        ".github/workflows/" in f["file_path"].replace("\\", "/") for f in files
    )
    if has_workflow_files:
        scope_resp = await client.get(
            "https://api.github.com/user", headers=gh_headers, timeout=10.0
        )
        _token_scopes_wf = scope_resp.headers.get("X-OAuth-Scopes", "")
        logger.info(
            f"[GH-PUSH] Workflow file(s) detected. Token scopes: '{_token_scopes_wf}' — attempting push anyway."
        )

    # ------------------------------------------------------------------
    # 1. Ensure repo is ready and get parent commit + base tree
    # ------------------------------------------------------------------
    parent_sha, base_tree_sha = await _gh_ensure_repo_ready(
        client, gh_headers, api, repo_full_name, branch
    )
    logger.info(
        f"[GH-PUSH] Repo ready. parent_sha={parent_sha} base_tree_sha={base_tree_sha}"
    )

    # ------------------------------------------------------------------
    # 2. Create blobs for every file
    # ------------------------------------------------------------------
    tree_items = []
    for f in files:
        logger.info(f"[GH-PUSH] Creating blob for '{f['file_path']}' ...")
        blob_resp = await client.post(
            f"{api}/git/blobs",
            json={"content": f["content"], "encoding": "base64"},
            headers=gh_headers,
            timeout=15.0,
        )
        logger.info(
            f"[GH-PUSH] Blob '{f['file_path']}': status={blob_resp.status_code} "
            f"body={blob_resp.text[:200]}"
        )
        if not blob_resp.is_success:
            err = blob_resp.json() if blob_resp.content else {}
            raise HTTPException(
                status_code=blob_resp.status_code,
                detail=f"Failed to create blob for '{f['file_path']}': {err.get('message', blob_resp.text[:200])}",
            )
        blob_sha = blob_resp.json()["sha"]
        tree_items.append({
            "path": f["file_path"],
            "mode": "100644",
            "type": "blob",
            "sha": blob_sha,
        })

    # ------------------------------------------------------------------
    # 3. Create a new tree (retry on 404 — GitHub eventual consistency)
    # ------------------------------------------------------------------
    tree_payload: Dict[str, Any] = {"tree": tree_items}
    if base_tree_sha:
        tree_payload["base_tree"] = base_tree_sha

    tree_resp = None
    for tree_attempt in range(5):
        logger.info(
            f"[GH-PUSH] Creating tree attempt #{tree_attempt+1}, "
            f"base_tree={tree_payload.get('base_tree')}, "
            f"items={[t['path'] for t in tree_items]}"
        )
        tree_resp = await client.post(
            f"{api}/git/trees",
            json=tree_payload,
            headers=gh_headers,
            timeout=15.0,
        )
        logger.info(
            f"[GH-PUSH] Tree response: status={tree_resp.status_code} "
            f"body={tree_resp.text[:400]}"
        )
        if tree_resp.is_success:
            break
        # On 404: base_tree SHA not yet visible — drop it and retry without base_tree
        if tree_resp.status_code == 404:
            scope_resp = await client.get("https://api.github.com/user", headers=gh_headers, timeout=10.0)
            token_scopes = scope_resp.headers.get("X-OAuth-Scopes", "unknown")
            logger.warning(
                f"[GH-PUSH] Tree 404 on attempt #{tree_attempt+1}. "
                f"Token scopes='{token_scopes}'. "
                f"base_tree={'present' if 'base_tree' in tree_payload else 'absent'}"
            )
            if "base_tree" in tree_payload:
                logger.warning(
                    f"[GH-PUSH] Retrying tree without base_tree={base_tree_sha}..."
                )
                tree_payload.pop("base_tree")
                base_tree_sha = None
                await asyncio.sleep(2 * (tree_attempt + 1))
                continue
        # Other error — bail
        err = tree_resp.json() if tree_resp.content else {}
        raise HTTPException(
            status_code=tree_resp.status_code,
            detail=f"Failed to create tree: {err.get('message', tree_resp.text[:300])}",
        )

    if tree_resp is None or not tree_resp.is_success:
        err = tree_resp.json() if (tree_resp and tree_resp.content) else {}
        raise HTTPException(
            status_code=tree_resp.status_code if tree_resp else 500,
            detail=f"Failed to create tree after retries: {err.get('message', 'Unknown')}",
        )
    new_tree_sha = tree_resp.json()["sha"]
    logger.info(f"[GH-PUSH] Tree created: sha={new_tree_sha}")

    # ------------------------------------------------------------------
    # 4. Create a commit
    # ------------------------------------------------------------------
    commit_payload: Dict[str, Any] = {
        "message": commit_message,
        "tree": new_tree_sha,
        "parents": [parent_sha] if parent_sha else [],
    }
    logger.info(f"[GH-PUSH] Creating commit with tree={new_tree_sha} parents={commit_payload['parents']}")
    commit_resp = await client.post(
        f"{api}/git/commits",
        json=commit_payload,
        headers=gh_headers,
        timeout=15.0,
    )
    logger.info(
        f"[GH-PUSH] Commit response: status={commit_resp.status_code} "
        f"body={commit_resp.text[:300]}"
    )
    if not commit_resp.is_success:
        err = commit_resp.json() if commit_resp.content else {}
        raise HTTPException(
            status_code=commit_resp.status_code,
            detail=f"Failed to create commit: {err.get('message', commit_resp.text[:300])}",
        )
    new_commit_sha = commit_resp.json()["sha"]
    logger.info(f"[GH-PUSH] Commit created: sha={new_commit_sha}")

    # ------------------------------------------------------------------
    # 5. Update (or create) the branch ref
    # ------------------------------------------------------------------
    if parent_sha:
        ref_update = await client.patch(
            f"{api}/git/refs/heads/{branch}",
            json={"sha": new_commit_sha, "force": True},
            headers=gh_headers,
            timeout=10.0,
        )
    else:
        ref_update = await client.post(
            f"{api}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": new_commit_sha},
            headers=gh_headers,
            timeout=10.0,
        )
    logger.info(
        f"[GH-PUSH] Ref update ({branch}): status={ref_update.status_code} "
        f"body={ref_update.text[:200]}"
    )
    if not ref_update.is_success:
        err = ref_update.json() if ref_update.content else {}
        raise HTTPException(
            status_code=ref_update.status_code,
            detail=f"Failed to update branch ref: {err.get('message', ref_update.text[:300])}",
        )

    logger.info(
        f"[GH-PUSH] SUCCESS: {len(files)} file(s) pushed to {repo_full_name}/{branch} "
        f"commit={new_commit_sha}"
    )
    return {
        "commit_sha": new_commit_sha,
        "tree_sha": new_tree_sha,
        "files_pushed": [f["file_path"] for f in files],
        "skipped_workflow_files": [
            {
                "file_path": f["file_path"],
                "reauth_required": True,
                "reauth_reason": "workflow_scope_missing",
                "reauth_url": _reauth_url_wf,
                "current_scopes": _token_scopes_wf,
                "required_scopes": "read:user repo workflow",
                "message": "Re-authenticate with the reauth_url to gain the 'workflow' scope, then retry.",
            }
            for f in _skipped_wf
        ],
    }


@app.post("/api/github/upload-file")
async def github_upload_file(request: UploadFileRequest):
    """
    Upload / create a single file in a GitHub repository.
    Uses the Git Data API (blobs→trees→commits→refs) instead of the Contents API
    to reliably handle nested paths like .github/workflows/deploy.yml.
    Serialized per-repo to prevent conflicts from parallel uploads.
    """
    commit_message = request.message or f"Add {request.file_path}"
    gh_headers = {
        "Authorization": f"Bearer {request.token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Verify repo exists
    async with httpx.AsyncClient() as client:
        repo_resp = await client.get(
            f"https://api.github.com/repos/{request.repo_full_name}",
            headers=gh_headers,
            timeout=10.0,
        )
    if repo_resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Repository '{request.repo_full_name}' not found. Create it first via /api/github/create-repo.",
        )
    if repo_resp.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token expired or invalid")

    # Serialize per-repo so parallel uploads don't conflict
    if request.repo_full_name not in _repo_upload_locks:
        _repo_upload_locks[request.repo_full_name] = asyncio.Lock()
    lock = _repo_upload_locks[request.repo_full_name]

    async with lock:
        async with httpx.AsyncClient() as client:
            try:
                result = await _github_git_api_push(
                    client=client,
                    gh_headers=gh_headers,
                    repo_full_name=request.repo_full_name,
                    branch=request.branch,
                    files=[{"file_path": request.file_path, "content": request.content}],
                    commit_message=commit_message,
                )
            except _WorkflowScopeMissing as wse:
                from fastapi.responses import JSONResponse
                logger.warning(
                    f"[upload-file] workflow scope missing for '{request.file_path}', "
                    f"returning 200 with reauth_required=true"
                )
                return JSONResponse(status_code=200, content=wse.to_dict())

    return {
        "content": {"path": request.file_path, "sha": result["commit_sha"]},
        "commit": {"sha": result["commit_sha"], "message": commit_message},
    }


@app.post("/api/github/upload-files")
async def github_upload_files(request: UploadFilesRequest):
    """
    Upload multiple files to a GitHub repository in a SINGLE commit.
    Uses the Git Data API (blobs→trees→commits→refs).
    This is faster and more reliable than calling upload-file per file.
    """
    if not request.files:
        raise HTTPException(status_code=422, detail="No files provided")

    gh_headers = {
        "Authorization": f"Bearer {request.token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Verify repo exists
    async with httpx.AsyncClient() as client:
        repo_resp = await client.get(
            f"https://api.github.com/repos/{request.repo_full_name}",
            headers=gh_headers,
            timeout=10.0,
        )
    if repo_resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Repository '{request.repo_full_name}' not found. Create it first via /api/github/create-repo.",
        )
    if repo_resp.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token expired or invalid")

    # Serialize per-repo
    if request.repo_full_name not in _repo_upload_locks:
        _repo_upload_locks[request.repo_full_name] = asyncio.Lock()
    lock = _repo_upload_locks[request.repo_full_name]

    async with lock:
        async with httpx.AsyncClient() as client:
            try:
                result = await _github_git_api_push(
                    client=client,
                    gh_headers=gh_headers,
                    repo_full_name=request.repo_full_name,
                    branch=request.branch,
                    files=request.files,
                    commit_message=request.message,
                )
            except _WorkflowScopeMissing as wse:
                from fastapi.responses import JSONResponse
                logger.warning(
                    f"[upload-files] workflow scope missing for '{wse.file_path}', "
                    f"returning 200 with reauth_required=true"
                )
                return JSONResponse(status_code=200, content=wse.to_dict())

    skipped = result.get("skipped_workflow_files", [])
    return {
        "success": True,
        "commit_sha": result["commit_sha"],
        "files_pushed": result["files_pushed"],
        "total_files": len(request.files),
        "skipped_workflow_files": skipped,
    }


# ==================== BITBUCKET OAUTH & REPO ====================

class BitbucketExchangeRequest(BaseModel):
    code: str = Field(..., description="OAuth code from Bitbucket callback")
    redirect_uri: str = Field(..., description="Redirect URI used in the OAuth flow")


@app.post("/api/bitbucket/exchange")
async def bitbucket_exchange(request: BitbucketExchangeRequest):
    """
    Exchange a Bitbucket OAuth code for an access token.
    Uses OAuth2 authorization code grant — client secret stays server-side.
    """
    if not BITBUCKET_CLIENT_ID or not BITBUCKET_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Bitbucket OAuth is not configured (missing BITBUCKET_CLIENT_ID or BITBUCKET_CLIENT_SECRET)"
        )

    logger.info(f"Bitbucket OAuth exchange: code={request.code[:8]}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://bitbucket.org/site/oauth2/access_token",
                data={
                    "grant_type": "authorization_code",
                    "code": request.code,
                    "redirect_uri": request.redirect_uri,
                },
                auth=(BITBUCKET_CLIENT_ID, BITBUCKET_CLIENT_SECRET),
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Bitbucket token exchange failed: network error")
            raise HTTPException(status_code=502, detail="Bitbucket OAuth API unreachable")

    token_data = response.json()
    if "error" in token_data:
        raise HTTPException(
            status_code=400,
            detail=token_data.get("error_description", token_data["error"])
        )

    return token_data


class BitbucketRefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Bitbucket refresh token")


@app.post("/api/bitbucket/refresh")
async def bitbucket_refresh(request: BitbucketRefreshRequest):
    """
    Refresh a Bitbucket OAuth access token using the refresh token.
    Bitbucket tokens expire after 2 hours, so this is needed for long sessions.
    """
    if not BITBUCKET_CLIENT_ID or not BITBUCKET_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Bitbucket OAuth is not configured (missing BITBUCKET_CLIENT_ID or BITBUCKET_CLIENT_SECRET)"
        )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://bitbucket.org/site/oauth2/access_token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": request.refresh_token,
                },
                auth=(BITBUCKET_CLIENT_ID, BITBUCKET_CLIENT_SECRET),
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Bitbucket token refresh failed: network error")
            raise HTTPException(status_code=502, detail="Bitbucket OAuth API unreachable")

    token_data = response.json()
    if "error" in token_data:
        raise HTTPException(
            status_code=400,
            detail=token_data.get("error_description", token_data["error"])
        )

    return token_data


@app.get("/api/bitbucket/user")
async def bitbucket_get_user(authorization: str = Header(..., description="Bearer <bitbucket_access_token>")):
    """
    Get the authenticated Bitbucket user profile.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'")
    token = authorization[7:]

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.bitbucket.org/2.0/user",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Bitbucket API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Bitbucket token expired or invalid")
    if not response.is_success:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch Bitbucket user")

    return response.json()


class BitbucketCreateRepoRequest(BaseModel):
    token: str = Field(..., description="Bitbucket OAuth access token (from /api/bitbucket/exchange)")
    workspace: str = Field(..., description="Bitbucket workspace slug")
    name: str = Field(..., description="Repository name")
    description: str = Field("", description="Repository description")
    is_private: bool = Field(True, description="Whether the repo is private")
    project_key: Optional[str] = Field(None, description="Project key to create the repo under")


@app.post("/api/bitbucket/create-repo")
async def bitbucket_create_repo(request: BitbucketCreateRepoRequest):
    """
    Create a Bitbucket repository on behalf of the authenticated user.
    """
    if not re.match(r'^[a-zA-Z0-9._\-]+$', request.workspace):
        raise HTTPException(status_code=422, detail="Invalid characters in workspace")
    if not re.match(r'^[a-zA-Z0-9._\-]+$', request.name):
        raise HTTPException(status_code=422, detail="Invalid characters in repo name")

    repo_slug = request.name.lower().replace(" ", "-")
    url = f"https://api.bitbucket.org/2.0/repositories/{request.workspace}/{repo_slug}"

    payload: Dict[str, Any] = {
        "scm": "git",
        "name": request.name,
        "description": request.description,
        "is_private": request.is_private,
    }
    if request.project_key:
        payload["project"] = {"key": request.project_key}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {request.token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Bitbucket API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Bitbucket token expired or invalid")
    if not response.is_success:
        error = response.json() if response.content else {}
        msg = error.get("error", {}).get("message", "Failed to create repository") if isinstance(error.get("error"), dict) else str(error)
        raise HTTPException(status_code=response.status_code, detail=msg)

    return response.json()


class BitbucketUploadFileRequest(BaseModel):
    token: str = Field(..., description="Bitbucket OAuth access token (from /api/bitbucket/exchange)")
    workspace: str = Field(..., description="Bitbucket workspace slug")
    repo_slug: str = Field(..., description="Bitbucket repository slug")
    file_path: str = Field(..., description="Path inside the repo (e.g. src/main.py)")
    content: str = Field(..., description="Raw file content (plain text)")
    message: str = Field("", description="Commit message")
    branch: str = Field("main", description="Target branch")


@app.post("/api/bitbucket/upload-file")
async def bitbucket_upload_file(request: BitbucketUploadFileRequest):
    """
    Upload / commit a single file to a Bitbucket repository.
    Uses the Bitbucket src endpoint (form-encoded file upload).
    """
    if not re.match(r'^[a-zA-Z0-9._\-]+$', request.workspace):
        raise HTTPException(status_code=422, detail="Invalid characters in workspace")
    if not re.match(r'^[a-zA-Z0-9._\-]+$', request.repo_slug):
        raise HTTPException(status_code=422, detail="Invalid characters in repo_slug")

    commit_message = request.message or f"Add {request.file_path}"
    url = f"https://api.bitbucket.org/2.0/repositories/{request.workspace}/{request.repo_slug}/src"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                data={
                    request.file_path: request.content,
                    "message": commit_message,
                    "branch": request.branch,
                },
                headers={
                    "Authorization": f"Bearer {request.token}",
                },
                timeout=15.0,
            )
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Bitbucket API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Bitbucket token expired or invalid")
    if not response.is_success:
        error = response.json() if response.content else {}
        msg = error.get("error", {}).get("message", f"Failed to upload {request.file_path}") if isinstance(error.get("error"), dict) else str(error)
        raise HTTPException(status_code=response.status_code, detail=msg)

    return {"success": True, "message": f"File '{request.file_path}' committed to {request.workspace}/{request.repo_slug} on branch '{request.branch}'"}


# ==================== AZURE DEVOPS OAUTH & REPO ====================

class AzureDevOpsExchangeRequest(BaseModel):
    code: str = Field(..., description="OAuth code from Azure DevOps callback")
    redirect_uri: str = Field(..., description="Redirect URI used in the OAuth flow")


@app.post("/api/azure-devops/exchange")
async def azure_devops_exchange(request: AzureDevOpsExchangeRequest):
    """
    Exchange an Azure DevOps OAuth code for an access token.
    Client secret stays server-side.
    """
    if not AZURE_DEVOPS_CLIENT_ID or not AZURE_DEVOPS_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Azure DevOps OAuth is not configured (missing AZURE_DEVOPS_CLIENT_ID or AZURE_DEVOPS_CLIENT_SECRET)"
        )

    logger.info(f"Azure DevOps OAuth exchange: code={request.code[:8]}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://app.vssps.visualstudio.com/oauth2/token",
                data={
                    "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                    "client_assertion": AZURE_DEVOPS_CLIENT_SECRET,
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": request.code,
                    "redirect_uri": request.redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.error("Azure DevOps token exchange failed: network error")
            raise HTTPException(status_code=502, detail="Azure DevOps OAuth API unreachable")

    token_data = response.json()
    if "error" in token_data:
        raise HTTPException(
            status_code=400,
            detail=token_data.get("error_description", token_data["error"])
        )

    return token_data


class AzureDevOpsRefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Azure DevOps refresh token")
    redirect_uri: str = Field(..., description="Redirect URI used in the OAuth flow")


@app.post("/api/azure-devops/refresh")
async def azure_devops_refresh(request: AzureDevOpsRefreshRequest):
    """
    Refresh an Azure DevOps OAuth access token.
    """
    if not AZURE_DEVOPS_CLIENT_ID or not AZURE_DEVOPS_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Azure DevOps OAuth is not configured"
        )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://app.vssps.visualstudio.com/oauth2/token",
                data={
                    "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                    "client_assertion": AZURE_DEVOPS_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "assertion": request.refresh_token,
                    "redirect_uri": request.redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.error("Azure DevOps token refresh failed: network error")
            raise HTTPException(status_code=502, detail="Azure DevOps OAuth API unreachable")

    token_data = response.json()
    if "error" in token_data:
        raise HTTPException(
            status_code=400,
            detail=token_data.get("error_description", token_data["error"])
        )

    return token_data


@app.get("/api/azure-devops/user")
async def azure_devops_get_user(authorization: str = Header(..., description="Bearer <azure_devops_access_token>")):
    """
    Get the authenticated Azure DevOps user profile.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'")
    token = authorization[7:]

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://app.vssps.visualstudio.com/_apis/profile/profiles/me?api-version=7.1",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Azure DevOps API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Azure DevOps token expired or invalid")
    if not response.is_success:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch Azure DevOps user")

    return response.json()


class AzureDevOpsCreateRepoRequest(BaseModel):
    token: str = Field(..., description="Azure DevOps OAuth access token (from /api/azure-devops/exchange)")
    organization: str = Field(..., description="Azure DevOps organization name")
    project: str = Field(..., description="Azure DevOps project name")
    name: str = Field(..., description="Repository name")


@app.post("/api/azure-devops/create-repo")
async def azure_devops_create_repo(request: AzureDevOpsCreateRepoRequest):
    """
    Create an Azure DevOps Git repository in a project.
    """
    import re as _re
    if not _re.match(r'^[a-zA-Z0-9._\-]+$', request.organization):
        raise HTTPException(status_code=422, detail="Invalid characters in organization")
    if not _re.match(r'^[a-zA-Z0-9._\- ]+$', request.project):
        raise HTTPException(status_code=422, detail="Invalid characters in project")

    url = f"https://dev.azure.com/{request.organization}/{request.project}/_apis/git/repositories?api-version=7.1"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json={"name": request.name},
                headers={
                    "Authorization": f"Bearer {request.token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Azure DevOps API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Azure DevOps token expired or invalid")
    if not response.is_success:
        error = response.json() if response.content else {}
        raise HTTPException(
            status_code=response.status_code,
            detail=error.get("message", "Failed to create repository"),
        )

    return response.json()


class AzureDevOpsUploadFileRequest(BaseModel):
    token: str = Field(..., description="Azure DevOps OAuth access token (from /api/azure-devops/exchange)")
    organization: str = Field(..., description="Azure DevOps organization name")
    project: str = Field(..., description="Azure DevOps project name")
    repo_name: str = Field(..., description="Repository name")
    file_path: str = Field(..., description="Path inside the repo (e.g. src/main.py)")
    content: str = Field(..., description="Raw file content (plain text)")
    message: str = Field("", description="Commit message")
    branch: str = Field("main", description="Target branch")


@app.post("/api/azure-devops/upload-file")
async def azure_devops_upload_file(request: AzureDevOpsUploadFileRequest):
    """
    Push a single file to an Azure DevOps Git repository via the Pushes API.
    """
    import re as _re
    import base64
    if not _re.match(r'^[a-zA-Z0-9._\-]+$', request.organization):
        raise HTTPException(status_code=422, detail="Invalid characters in organization")
    if not _re.match(r'^[a-zA-Z0-9._\- ]+$', request.project):
        raise HTTPException(status_code=422, detail="Invalid characters in project")

    commit_message = request.message or f"Add {request.file_path}"

    # First, get the latest commit on the branch to use as oldObjectId
    refs_url = f"https://dev.azure.com/{request.organization}/{request.project}/_apis/git/repositories/{request.repo_name}/refs?filter=heads/{request.branch}&api-version=7.1"
    headers = {
        "Authorization": f"Bearer {request.token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            refs_resp = await client.get(refs_url, headers=headers, timeout=10.0)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Azure DevOps API unreachable")

        if refs_resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Azure DevOps token expired or invalid")

        refs_data = refs_resp.json()
        ref_list = refs_data.get("value", [])
        old_object_id = ref_list[0]["objectId"] if ref_list else "0000000000000000000000000000000000000000"

        # Push the file
        push_url = f"https://dev.azure.com/{request.organization}/{request.project}/_apis/git/repositories/{request.repo_name}/pushes?api-version=7.1"
        content_b64 = base64.b64encode(request.content.encode()).decode()
        change_type = "add" if old_object_id == "0000000000000000000000000000000000000000" else "edit"

        push_payload = {
            "refUpdates": [{"name": f"refs/heads/{request.branch}", "oldObjectId": old_object_id}],
            "commits": [{
                "comment": commit_message,
                "changes": [{
                    "changeType": change_type,
                    "item": {"path": f"/{request.file_path}"},
                    "newContent": {"content": request.content, "contentType": "rawtext"},
                }]
            }]
        }

        try:
            response = await client.post(push_url, json=push_payload, headers=headers, timeout=15.0)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Azure DevOps API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Azure DevOps token expired or invalid")
    if not response.is_success:
        error = response.json() if response.content else {}
        raise HTTPException(
            status_code=response.status_code,
            detail=error.get("message", f"Failed to upload {request.file_path}"),
        )

    return response.json()


# ==================== GITLAB OAUTH & REPO ====================

class GitLabExchangeRequest(BaseModel):
    code: str = Field(..., description="OAuth code from GitLab callback")
    redirect_uri: str = Field(..., description="Redirect URI used in the OAuth flow")
    gitlab_url: str = Field("https://gitlab.com", description="GitLab instance URL")


@app.post("/api/gitlab/exchange")
async def gitlab_exchange(request: GitLabExchangeRequest):
    """
    Exchange a GitLab OAuth code for an access token.
    Client secret stays server-side.
    """
    if not GITLAB_CLIENT_ID or not GITLAB_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="GitLab OAuth is not configured (missing GITLAB_CLIENT_ID or GITLAB_CLIENT_SECRET)"
        )

    import re as _re
    if not _re.match(r'^https://[a-zA-Z0-9._\-]+(:[0-9]+)?$', request.gitlab_url.rstrip("/")):
        raise HTTPException(status_code=422, detail="gitlab_url must be a valid HTTPS URL")
    base_url = request.gitlab_url.rstrip("/")

    logger.info(f"GitLab OAuth exchange: code={request.code[:8]}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/oauth/token",
                data={
                    "client_id": GITLAB_CLIENT_ID,
                    "client_secret": GITLAB_CLIENT_SECRET,
                    "code": request.code,
                    "grant_type": "authorization_code",
                    "redirect_uri": request.redirect_uri,
                },
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.error("GitLab token exchange failed: network error")
            raise HTTPException(status_code=502, detail="GitLab OAuth API unreachable")

    token_data = response.json()
    if "error" in token_data:
        raise HTTPException(
            status_code=400,
            detail=token_data.get("error_description", token_data["error"])
        )

    return token_data


class GitLabRefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="GitLab refresh token")
    gitlab_url: str = Field("https://gitlab.com", description="GitLab instance URL")


@app.post("/api/gitlab/refresh")
async def gitlab_refresh(request: GitLabRefreshRequest):
    """
    Refresh a GitLab OAuth access token.
    """
    if not GITLAB_CLIENT_ID or not GITLAB_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="GitLab OAuth is not configured")

    import re as _re
    if not _re.match(r'^https://[a-zA-Z0-9._\-]+(:[0-9]+)?$', request.gitlab_url.rstrip("/")):
        raise HTTPException(status_code=422, detail="gitlab_url must be a valid HTTPS URL")
    base_url = request.gitlab_url.rstrip("/")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/oauth/token",
                data={
                    "client_id": GITLAB_CLIENT_ID,
                    "client_secret": GITLAB_CLIENT_SECRET,
                    "refresh_token": request.refresh_token,
                    "grant_type": "refresh_token",
                    "redirect_uri": "",
                },
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.error("GitLab token refresh failed: network error")
            raise HTTPException(status_code=502, detail="GitLab OAuth API unreachable")

    token_data = response.json()
    if "error" in token_data:
        raise HTTPException(
            status_code=400,
            detail=token_data.get("error_description", token_data["error"])
        )

    return token_data


@app.get("/api/gitlab/user")
async def gitlab_get_user(
    authorization: str = Header(..., description="Bearer <gitlab_access_token>"),
    gitlab_url: str = "https://gitlab.com",
):
    """
    Get the authenticated GitLab user profile.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'")
    token = authorization[7:]

    import re as _re
    if not _re.match(r'^https://[a-zA-Z0-9._\-]+(:[0-9]+)?$', gitlab_url.rstrip("/")):
        raise HTTPException(status_code=422, detail="gitlab_url must be a valid HTTPS URL")
    base_url = gitlab_url.rstrip("/")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{base_url}/api/v4/user",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="GitLab API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="GitLab token expired or invalid")
    if not response.is_success:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch GitLab user")

    return response.json()


class GitLabCreateRepoRequest(BaseModel):
    token: str = Field(..., description="GitLab OAuth access token (from /api/gitlab/exchange)")
    name: str = Field(..., description="Repository (project) name")
    description: str = Field("", description="Repository description")
    visibility: str = Field("private", description="Visibility: private, internal, or public")
    namespace_id: Optional[int] = Field(None, description="Namespace/group ID to create under (defaults to user namespace)")
    initialize_with_readme: bool = Field(False, description="Whether to initialize with a README")
    gitlab_url: str = Field("https://gitlab.com", description="GitLab instance URL")


@app.post("/api/gitlab/create-repo")
async def gitlab_create_repo(request: GitLabCreateRepoRequest):
    """
    Create a GitLab project (repository).
    """
    import re as _re
    if not _re.match(r'^https://[a-zA-Z0-9._\-]+(:[0-9]+)?$', request.gitlab_url.rstrip("/")):
        raise HTTPException(status_code=422, detail="gitlab_url must be a valid HTTPS URL")
    base_url = request.gitlab_url.rstrip("/")

    if request.visibility not in ("private", "internal", "public"):
        raise HTTPException(status_code=422, detail="visibility must be 'private', 'internal', or 'public'")

    payload: Dict[str, Any] = {
        "name": request.name,
        "description": request.description,
        "visibility": request.visibility,
        "initialize_with_readme": request.initialize_with_readme,
    }
    if request.namespace_id:
        payload["namespace_id"] = request.namespace_id

    url = f"{base_url}/api/v4/projects"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {request.token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="GitLab API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="GitLab token expired or invalid")
    if not response.is_success:
        error = response.json() if response.content else {}
        msg = error.get("message", "Failed to create project")
        raise HTTPException(status_code=response.status_code, detail=msg if isinstance(msg, str) else str(msg))

    return response.json()


class GitLabUploadFileRequest(BaseModel):
    token: str = Field(..., description="GitLab OAuth access token (from /api/gitlab/exchange)")
    project_id: int = Field(..., description="GitLab project ID")
    file_path: str = Field(..., description="Path inside the repo (e.g. src/main.py)")
    content: str = Field(..., description="Raw file content (plain text)")
    message: str = Field("", description="Commit message")
    branch: str = Field("main", description="Target branch")
    gitlab_url: str = Field("https://gitlab.com", description="GitLab instance URL")


@app.post("/api/gitlab/upload-file")
async def gitlab_upload_file(request: GitLabUploadFileRequest):
    """
    Create or update a single file in a GitLab repository via the Repository Files API.
    """
    import re as _re
    import urllib.parse
    if not _re.match(r'^https://[a-zA-Z0-9._\-]+(:[0-9]+)?$', request.gitlab_url.rstrip("/")):
        raise HTTPException(status_code=422, detail="gitlab_url must be a valid HTTPS URL")
    base_url = request.gitlab_url.rstrip("/")

    commit_message = request.message or f"Add {request.file_path}"
    encoded_path = urllib.parse.quote(request.file_path, safe="")
    url = f"{base_url}/api/v4/projects/{request.project_id}/repository/files/{encoded_path}"
    headers = {
        "Authorization": f"Bearer {request.token}",
        "Content-Type": "application/json",
    }
    payload = {
        "branch": request.branch,
        "content": request.content,
        "commit_message": commit_message,
    }

    async with httpx.AsyncClient() as client:
        # Try create first; if file exists, update instead
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=15.0)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="GitLab API unreachable")

        if response.status_code == 400 and "already exists" in (response.text or "").lower():
            try:
                response = await client.put(url, json=payload, headers=headers, timeout=15.0)
            except httpx.HTTPError:
                raise HTTPException(status_code=502, detail="GitLab API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="GitLab token expired or invalid")
    if not response.is_success:
        error = response.json() if response.content else {}
        msg = error.get("message", f"Failed to upload {request.file_path}")
        raise HTTPException(status_code=response.status_code, detail=msg if isinstance(msg, str) else str(msg))

    return response.json()


# ==================== CI/CD PIPELINE TRIGGERS ====================

import re

def _validate_path_segment(value: str, field_name: str) -> str:
    """Validate that a value is a safe path segment (no traversal or injection)."""
    if not re.match(r'^[a-zA-Z0-9._\-]+$', value):
        raise HTTPException(status_code=422, detail=f"Invalid characters in {field_name}")
    return value


def _validate_repo_full_name(value: str) -> str:
    """Validate owner/repo format."""
    if not re.match(r'^[a-zA-Z0-9._\-]+/[a-zA-Z0-9._\-]+$', value):
        raise HTTPException(status_code=422, detail="repo_full_name must be in 'owner/repo' format with safe characters")
    return value


async def _get_github_user(token: str) -> Dict[str, Any]:
    """Fetch the authenticated GitHub user to validate the token."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"},
            timeout=10.0,
        )
    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token expired or invalid. Please re-authenticate via GitHub OAuth.")
    if not resp.is_success:
        raise HTTPException(status_code=resp.status_code, detail="Failed to verify GitHub identity")
    return resp.json()


class GitHubActionsTriggerRequest(BaseModel):
    token: str = Field(..., description="GitHub OAuth access token (from /api/github/exchange)")
    repo_full_name: str = Field(..., description="owner/repo")
    workflow_id: str = Field(..., description="Workflow file name (e.g. deploy.yml) or workflow ID")
    ref: str = Field("main", description="Git branch or tag to run the workflow on")
    inputs: Optional[Dict[str, str]] = Field(default=None, description="Workflow input parameters")


@app.post("/api/pipeline/github-actions/trigger")
async def trigger_github_actions(request: GitHubActionsTriggerRequest):
    """
    Trigger a GitHub Actions workflow dispatch event.
    Uses the same OAuth token obtained from /api/github/exchange.
    If the workflow file doesn't exist in the repo, it auto-creates a default one.
    """
    _validate_repo_full_name(request.repo_full_name)
    _validate_path_segment(request.workflow_id, "workflow_id")

    # Verify the token is valid by fetching the user
    user = await _get_github_user(request.token)
    logger.info(f"Pipeline trigger by GitHub user: {user.get('login')}")

    gh_headers = {
        "Authorization": f"Bearer {request.token}",
        "Accept": "application/vnd.github.v3+json",
    }

    async with httpx.AsyncClient() as client:
        # Step 1: Check if repo exists
        repo_resp = await client.get(
            f"https://api.github.com/repos/{request.repo_full_name}",
            headers=gh_headers,
            timeout=10.0,
        )
        if repo_resp.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Repository '{request.repo_full_name}' not found or you don't have access."
            )

        # Step 2: Check if workflow file exists
        workflow_path = f".github/workflows/{request.workflow_id}"
        file_resp = await client.get(
            f"https://api.github.com/repos/{request.repo_full_name}/contents/{workflow_path}",
            headers=gh_headers,
            params={"ref": request.ref},
            timeout=10.0,
        )

        if file_resp.status_code == 404:
            # Auto-create a default workflow with workflow_dispatch trigger
            import base64
            default_workflow = f"""name: {request.workflow_id.replace('.yml', '').replace('.yaml', '').replace('-', ' ').title()}

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: false
        default: 'dev'
        type: choice
        options:
          - dev
          - staging
          - production

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Display trigger info
        run: |
          echo "Triggered by: ${{{{ github.actor }}}}"
          echo "Environment: ${{{{ github.event.inputs.environment }}}}"
          echo "Branch: ${{{{ github.ref_name }}}}"
          echo "Repository: ${{{{ github.repository }}}}"

      - name: Deploy
        run: |
          echo "Deploying to ${{{{ github.event.inputs.environment }}}} environment..."
          echo "Add your deployment steps here"
"""
            content_b64 = base64.b64encode(default_workflow.encode()).decode()
            create_resp = await client.put(
                f"https://api.github.com/repos/{request.repo_full_name}/contents/{workflow_path}",
                json={
                    "message": f"Add {request.workflow_id} workflow with workflow_dispatch trigger",
                    "content": content_b64,
                    "branch": request.ref,
                },
                headers=gh_headers,
                timeout=15.0,
            )
            if not create_resp.is_success:
                error = create_resp.json() if create_resp.content else {}
                raise HTTPException(
                    status_code=create_resp.status_code,
                    detail=f"Workflow '{request.workflow_id}' not found and auto-create failed: {error.get('message', 'Unknown error')}",
                )
            logger.info(f"Auto-created workflow '{workflow_path}' in {request.repo_full_name}")

            # GitHub needs a moment to index the new workflow — return success with info
            return {
                "success": True,
                "message": f"Workflow '{request.workflow_id}' was created in {request.repo_full_name}. It will be available to trigger in a few seconds. Please retry the trigger.",
                "workflowCreated": True,
                "triggeredBy": user.get("login"),
            }

        # Step 3: Trigger the workflow dispatch
        dispatch_url = f"https://api.github.com/repos/{request.repo_full_name}/actions/workflows/{request.workflow_id}/dispatches"
        dispatch_payload: Dict[str, Any] = {"ref": request.ref}
        if request.inputs:
            dispatch_payload["inputs"] = request.inputs

        try:
            response = await client.post(
                dispatch_url,
                json=dispatch_payload,
                headers=gh_headers,
                timeout=15.0,
            )
        except httpx.HTTPError:
            logger.error("GitHub Actions trigger failed: network error")
            raise HTTPException(status_code=502, detail="GitHub API unreachable")

    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{request.workflow_id}' exists but cannot be triggered. Ensure it has 'on: workflow_dispatch' in the YAML."
        )
    if not response.is_success:
        error = response.json() if response.content else {}
        raise HTTPException(status_code=response.status_code, detail=error.get("message", "Failed to trigger workflow"))

    return {"success": True, "message": f"GitHub Actions workflow '{request.workflow_id}' triggered on '{request.ref}'", "workflowCreated": False, "triggeredBy": user.get("login")}


class AzureDevOpsTriggerRequest(BaseModel):
    token: str = Field(..., description="GitHub OAuth access token (from /api/github/exchange) — used to verify user identity")
    organization: str = Field(..., description="Azure DevOps organization name")
    project: str = Field(..., description="Azure DevOps project name")
    pipeline_id: int = Field(..., description="Pipeline definition ID")
    branch: str = Field("main", description="Source branch to build")
    parameters: Optional[Dict[str, str]] = Field(default=None, description="Pipeline parameters")


@app.post("/api/pipeline/azure-devops/trigger")
async def trigger_azure_devops(request: AzureDevOpsTriggerRequest):
    """
    Trigger an Azure DevOps pipeline run.
    Uses the GitHub OAuth token to verify user identity, and server-side
    env var AZURE_DEVOPS_PAT for Azure DevOps authentication.
    """
    import base64
    # Verify user identity via GitHub OAuth token
    user = await _get_github_user(request.token)
    logger.info(f"Azure DevOps pipeline trigger by GitHub user: {user.get('login')}")

    # Use server-side PAT — never sent by the client
    azure_pat = os.getenv("AZURE_DEVOPS_PAT")
    if not azure_pat:
        raise HTTPException(status_code=500, detail="Azure DevOps integration not configured (AZURE_DEVOPS_PAT missing)")

    _validate_path_segment(request.organization, "organization")
    _validate_path_segment(request.project, "project")

    url = f"https://dev.azure.com/{request.organization}/{request.project}/_apis/pipelines/{request.pipeline_id}/runs?api-version=7.1"
    payload: Dict[str, Any] = {
        "resources": {
            "repositories": {
                "self": {"refName": f"refs/heads/{request.branch}"}
            }
        }
    }
    if request.parameters:
        payload["templateParameters"] = request.parameters

    auth = base64.b64encode(f":{azure_pat}".encode()).decode()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        except httpx.HTTPError as e:
            logger.error("Azure DevOps trigger failed: network error")
            raise HTTPException(status_code=502, detail="Azure DevOps API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Azure DevOps PAT expired or invalid")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Pipeline not found. Check organization, project, and pipeline ID.")
    if not response.is_success:
        error = response.json() if response.content else {}
        raise HTTPException(status_code=response.status_code, detail=error.get("message", "Failed to trigger pipeline"))

    data = response.json()
    return {"success": True, "message": f"Azure DevOps pipeline {request.pipeline_id} triggered", "triggeredBy": user.get("login"), "runId": data.get("id"), "url": data.get("_links", {}).get("web", {}).get("href", "")}


class GitLabCITriggerRequest(BaseModel):
    token: str = Field(..., description="GitHub OAuth access token (from /api/github/exchange) — used to verify user identity")
    gitlab_url: str = Field("https://gitlab.com", description="GitLab instance URL")
    project_id: int = Field(..., description="GitLab project ID")
    ref: str = Field("main", description="Branch or tag name")
    variables: Optional[Dict[str, str]] = Field(default=None, description="Pipeline variables")


@app.post("/api/pipeline/gitlab-ci/trigger")
async def trigger_gitlab_ci(request: GitLabCITriggerRequest):
    """
    Trigger a GitLab CI/CD pipeline.
    Uses GitHub OAuth token to verify user identity, and server-side
    env var GITLAB_TRIGGER_TOKEN for GitLab authentication.
    """
    # Verify user identity via GitHub OAuth token
    user = await _get_github_user(request.token)
    logger.info(f"GitLab CI pipeline trigger by GitHub user: {user.get('login')}")

    # Use server-side trigger token — never sent by the client
    gitlab_token = os.getenv("GITLAB_TRIGGER_TOKEN")
    if not gitlab_token:
        raise HTTPException(status_code=500, detail="GitLab integration not configured (GITLAB_TRIGGER_TOKEN missing)")

    # Validate gitlab_url is a proper HTTPS URL
    if not re.match(r'^https://[a-zA-Z0-9._\-]+(:[0-9]+)?$', request.gitlab_url.rstrip("/")):
        raise HTTPException(status_code=422, detail="gitlab_url must be a valid HTTPS URL")
    base_url = request.gitlab_url.rstrip("/")

    url = f"{base_url}/api/v4/projects/{request.project_id}/trigger/pipeline"
    form_data: Dict[str, str] = {"token": gitlab_token, "ref": request.ref}
    if request.variables:
        for key, val in request.variables.items():
            form_data[f"variables[{key}]"] = val

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=form_data, timeout=15.0)
        except httpx.HTTPError as e:
            logger.error("GitLab CI trigger failed: network error")
            raise HTTPException(status_code=502, detail="GitLab API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="GitLab trigger token expired or invalid")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Project not found. Check project ID and permissions.")
    if not response.is_success:
        error = response.json() if response.content else {}
        raise HTTPException(status_code=response.status_code, detail=error.get("message", {}) if isinstance(error.get("message"), str) else str(error))

    data = response.json()
    return {"success": True, "message": f"GitLab CI pipeline triggered on '{request.ref}'", "triggeredBy": user.get("login"), "pipelineId": data.get("id"), "url": data.get("web_url", "")}


class BitbucketPipelineTriggerRequest(BaseModel):
    token: str = Field(..., description="GitHub OAuth access token (from /api/github/exchange) — used to verify user identity")
    workspace: str = Field(..., description="Bitbucket workspace slug")
    repo_slug: str = Field(..., description="Bitbucket repository slug")
    branch: str = Field("main", description="Branch to run the pipeline on")
    variables: Optional[List[Dict[str, str]]] = Field(default=None, description="Pipeline variables [{key, value, secured}]")


@app.post("/api/pipeline/bitbucket/trigger")
async def trigger_bitbucket_pipeline(request: BitbucketPipelineTriggerRequest):
    """
    Trigger a Bitbucket Pipelines run.
    Uses GitHub OAuth token to verify user identity, and server-side
    env var BITBUCKET_ACCESS_TOKEN (OAuth2 token) for Bitbucket auth.
    """
    # Verify user identity via GitHub OAuth token
    user = await _get_github_user(request.token)
    logger.info(f"Bitbucket pipeline trigger by GitHub user: {user.get('login')}")

    # Use server-side OAuth2 access token — no username/password
    bb_token = os.getenv("BITBUCKET_ACCESS_TOKEN")
    if not bb_token:
        raise HTTPException(status_code=500, detail="Bitbucket integration not configured (BITBUCKET_ACCESS_TOKEN missing)")

    _validate_path_segment(request.workspace, "workspace")
    _validate_path_segment(request.repo_slug, "repo_slug")

    url = f"https://api.bitbucket.org/2.0/repositories/{request.workspace}/{request.repo_slug}/pipelines/"
    payload: Dict[str, Any] = {
        "target": {
            "ref_type": "branch",
            "type": "pipeline_ref_target",
            "ref_name": request.branch,
        }
    }
    if request.variables:
        payload["variables"] = request.variables

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {bb_token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        except httpx.HTTPError as e:
            logger.error("Bitbucket pipeline trigger failed: network error")
            raise HTTPException(status_code=502, detail="Bitbucket API unreachable")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Bitbucket credentials invalid")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Repository not found. Check workspace and repo slug.")
    if not response.is_success:
        error = response.json() if response.content else {}
        msg = error.get("error", {}).get("message", "Failed to trigger pipeline") if isinstance(error.get("error"), dict) else str(error)
        raise HTTPException(status_code=response.status_code, detail=msg)

    data = response.json()
    return {"success": True, "message": f"Bitbucket pipeline triggered on '{request.branch}'", "triggeredBy": user.get("login"), "pipelineUuid": data.get("uuid", ""), "buildNumber": data.get("build_number", "")}


# ==================== ANALYZE REQUIREMENTS (OpenAI) ====================

class AnalyzeRequirementsRequest(BaseModel):
    requirementText: str = Field(..., min_length=1, max_length=10000, description="Raw requirement text to analyze")
    applicationName: str = Field("Untitled", description="Application name")
    tenantId: str = Field("anonymous", description="Tenant ID")
    sessionId: str = Field("", description="Session ID")


@app.post("/api/requirements/analyze")
async def analyze_requirements(request: AnalyzeRequirementsRequest):
    """
    Analyze raw requirement text using OpenAI (single call, streamed field-by-field).
    Returns SSE events for each field as it completes:
      data: {"field": "thinking"}
      data: {"field": "industry", "data": [...]}
      data: {"field": "useCases", "data": [...]}
      data: {"field": "featuresList", "data": [...]}
      data: {"field": "businessRequirements", "data": [...]}
      data: {"field": "userTypes", "data": [...]}
      data: {"field": "security", "data": [...]}
      data: {"field": "error", "detail": "..."} on failure
    """
    import time

    if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
        raise HTTPException(status_code=500, detail="OpenAI not configured")

    session_id = request.sessionId or f"session_{int(time.time() * 1000)}_{uuid4().hex[:8]}"
    now = datetime.utcnow()

    system_prompt = (
        "Extract structured info from requirement text. Return ONLY a JSON object.\n"
        "First determine the projectType and applicationType, then include ONLY the keys that are relevant for that type of application. "
        "You may add any additional keys that provide useful context. Skip keys that don't apply.\n\n"
        "Always include:\n"
        "- projectType (string): 'New Development', 'Enhancement', 'Migration', 'Modernization', 'Re-architecture', 'Integration', or 'Maintenance'.\n"
        "- applicationType (string): 'Web Application', 'Mobile App', 'Desktop Application', 'REST API / Backend Service', "
        "'SPA', 'PWA', 'Microservices', 'CLI Tool', 'Embedded System', 'Data Pipeline', or other. Infer from context.\n"
        "- industry (string[]): relevant industries/domains.\n\n"
        "Then include keys based on what fits the application type:\n"
        "FOR WEB/MOBILE/DESKTOP APPS (apps with UI):\n"
        "- useCases, featuresList, businessRequirements, userTypes\n"
        "- screens: {totalCount, screenList: [{name, type, description}], flow: string[]}\n"
        "- userFlows (string[]): key user journeys e.g. 'User registers -> verifies email -> completes profile -> lands on dashboard'\n"
        "- accessibilityNeeds (string[]): any accessibility requirements (WCAG, screen reader support, etc.)\n\n"
        "FOR APIs / BACKEND SERVICES:\n"
        "- endpoints (array of {method, path, description}): key API endpoints\n"
        "- dataEntities (string[]): key data models/entities\n"
        "- businessRules (string[]): business logic and validation rules\n"
        "- authStrategy (string): authentication approach (OAuth2, API Key, JWT, etc.)\n\n"
        "FOR DATA PIPELINES / ETL:\n"
        "- dataSources (string[]): input data sources\n"
        "- dataDestinations (string[]): output targets\n"
        "- transformations (string[]): key data transformations\n"
        "- scheduleFrequency (string): how often the pipeline runs\n\n"
        "FOR MICROSERVICES:\n"
        "- services (array of {name, responsibility, communicatesWith: string[]}): microservice breakdown\n"
        "- messagingPatterns (string[]): event-driven, queue-based, sync REST, etc.\n\n"
        "FOR MIGRATIONS / MODERNIZATION:\n"
        "- sourceSystem (string): what's being migrated from\n"
        "- targetSystem (string): what's being migrated to\n"
        "- migrationStrategy (string): lift-and-shift, re-platform, re-architect, etc.\n"
        "- risksAndChallenges (string[]): identified risks\n\n"
        "COMMON (include for any type if applicable):\n"
        "- featuresList, businessRequirements, userTypes\n"
        "- integrations (string[]): external systems/APIs to integrate with\n"
        "- dataEntities (string[]): key data models\n"
        "- nonFunctionalRequirements (string[]): performance, scalability, compliance\n"
        "- constraints (string[]): technical/business constraints\n"
        "- assumptions (string[]): assumptions made during analysis\n"
        "\nDo NOT include any technology stack suggestions or recommendations. Focus purely on requirements analysis.\n"
        "Be concise. Derive everything from the requirement text. Do not fabricate. "
        "Omit any key that doesn't apply."
    )

    # Known fields to try extracting during streaming (order matters for progressive emit).
    # Any additional keys OpenAI returns will be emitted after streaming completes.
    KNOWN_FIELDS = [
        "projectType", "applicationType", "industry",
        "useCases", "featuresList", "businessRequirements", "userTypes",
        "screens", "userFlows", "accessibilityNeeds",
        "endpoints", "dataEntities", "businessRules", "authStrategy",
        "dataSources", "dataDestinations", "transformations", "scheduleFrequency",
        "services", "messagingPatterns",
        "sourceSystem", "targetSystem", "migrationStrategy", "risksAndChallenges",
        "integrations", "nonFunctionalRequirements", "constraints",
        "assumptions",
    ]

    def try_extract_value(text, key):
        """Extract a complete JSON value for a top-level key from accumulated tokens."""
        search = f'"{key}"'
        idx = text.find(search)
        if idx == -1:
            return None
        val_start = text.find(":", idx + len(search))
        if val_start == -1:
            return None
        val_start += 1
        while val_start < len(text) and text[val_start] in " \t\n\r":
            val_start += 1
        if val_start >= len(text):
            return None

        ch = text[val_start]
        if ch == '"':
            i = val_start + 1
            while i < len(text):
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == '"':
                    try:
                        return json.loads(text[val_start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        return None
                i += 1
            return None
        elif ch in ('{', '['):
            close = '}' if ch == '{' else ']'
            depth = 0
            in_str = False
            i = val_start
            while i < len(text):
                c = text[i]
                if in_str:
                    if c == '\\':
                        i += 2
                        continue
                    if c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == ch:
                        depth += 1
                    elif c == close:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[val_start:i + 1])
                            except (json.JSONDecodeError, ValueError):
                                return None
                i += 1
            return None
        return None

    async def event_stream():
        start = time.time()
        yield f"data: {json.dumps({'field': 'thinking'})}\n\n"

        accumulated = ""
        emitted = set()

        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
            stream = await client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.requirementText[:4000]},
                ],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort="low",
                response_format={"type": "json_object"},
                stream=True,
            )
            async for event in stream:
                if event.choices and event.choices[0].delta.content:
                    accumulated += event.choices[0].delta.content

                    for key in KNOWN_FIELDS:
                        if key not in emitted:
                            val = try_extract_value(accumulated, key)
                            if val is not None:
                                emitted.add(key)
                                yield f"data: {json.dumps({'field': key, 'data': val})}\n\n"

            if not accumulated:
                yield f"data: {json.dumps({'field': 'error', 'detail': 'OpenAI returned empty content'})}\n\n"
                return

            parsed = json.loads(accumulated)
        except json.JSONDecodeError:
            logger.error(f"Analyze: invalid JSON. Raw: {accumulated[:500]}")
            yield f"data: {json.dumps({'field': 'error', 'detail': 'OpenAI returned invalid JSON'})}\n\n"
            return
        except Exception as e:
            logger.error(f"OpenAI analyze error: {e}")
            yield f"data: {json.dumps({'field': 'error', 'detail': str(e)})}\n\n"
            return

        # Emit any fields missed during streaming — both known and any extra keys OpenAI added
        for key, val in parsed.items():
            if key not in emitted:
                yield f"data: {json.dumps({'field': key, 'data': val})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==================== FRONTEND DESIGN GENERATION (OpenAI) ====================

class FrontendDesignRequest(BaseModel):
    """Request model for generating frontend UI design"""
    tenantId: str = Field(default="anonymous", description="Tenant ID")
    sessionId: str = Field(default="", description="Session ID")
    applicationName: str = Field(..., description="Application name")
    overview: str = Field(..., min_length=1, max_length=10000, description="Application overview / requirements")
    screens: Optional[List[Dict[str, Any]]] = Field(default=None, description="Screen list from analyze endpoint (name, type, description)")
    screenFlow: Optional[List[str]] = Field(default=None, description="Screen flow / navigation paths")
    features: Optional[List[str]] = Field(default=None, description="Application features")
    userTypes: Optional[List[str]] = Field(default=None, description="Types of users")
    industry: Optional[str] = Field(default=None, description="Industry / domain")
    applicationType: Optional[str] = Field(default=None, description="App type e.g. SPA, PWA, Mobile")
    theme: Optional[str] = Field(default="modern", description="Design theme: modern, minimal, corporate, vibrant")


@app.post("/api/design/frontend")
async def generate_frontend_design(request: FrontendDesignRequest):
    """
    Generate a complete frontend UI design for all screens using OpenAI.
    Returns a JSON with pixel-perfect element layouts for every screen,
    streamed as SSE so the client can render screens progressively.
    """
    import time

    if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
        raise HTTPException(status_code=500, detail="OpenAI not configured")

    session_id = request.sessionId or f"session_{int(time.time() * 1000)}_{uuid4().hex[:8]}"

    # Build screen context from input
    screens_info = ""
    if request.screens:
        for i, s in enumerate(request.screens, 1):
            name = s.get("name", f"Screen {i}")
            stype = s.get("type", "page")
            desc = s.get("description", "")
            screens_info += f"  {i}. \"{name}\" (type: {stype}) — {desc}\n"
    else:
        screens_info = "  Infer screens from the overview and features.\n"

    flow_info = ""
    if request.screenFlow:
        flow_info = "SCREEN FLOW / NAVIGATION:\n" + "\n".join(f"  • {f}" for f in request.screenFlow) + "\n"

    features_info = ""
    if request.features:
        features_info = "KEY FEATURES:\n" + "\n".join(f"  • {f}" for f in request.features) + "\n"

    user_types_info = ""
    if request.userTypes:
        user_types_info = "USER TYPES: " + ", ".join(request.userTypes) + "\n"

    system_prompt = (
        "You are an expert UI/UX designer. Generate pixel-precise screen layouts as JSON.\n"
        "Canvas: 1440×900. Elements: nav, rect, text, input, button, card, table, chart, image, list, form, modal.\n"
        "Props: nav:{bg} | rect:{bg,borderColor,borderRadius} | text:{fontSize,fontWeight,color} | "
        "input:{placeholder,inputType,borderRadius} | button:{bg,color,borderRadius} | card/table/chart/image/list/form/modal:{bg,borderRadius}\n\n"

        "PAGE STRUCTURE:\n"
        "App screens (dashboard/list/form/detail/settings/analytics/etc) MUST have:\n"
        " 1) Top nav: nav x:0,y:0,w:1440,h:56 → app name(left), nav links(center, highlight current page), user profile+avatar(right)\n"
        " 2) Sidebar: rect x:0,y:56,w:220,h:844,bg:#F8F9FA → nav items at x:20 starting y:76, h:40 each, current page highlighted\n"
        " 3) Content: x:240,y:72 area. Page title at y:76 fontSize:24. Content starts y:140, width ~1180\n"
        " 4) Footer: rect x:220,y:860,w:1220,h:40 → copyright text\n\n"
        "Auth screens (login/signup): NO sidebar/nav. Background rect full canvas + centered card x:420,y:120,w:600,h:660 with logo, inputs, button. Optional branding panel on left.\n"
        "Landing pages: Top nav with CTA buttons, hero section, feature cards grid, stats row, footer. NO sidebar.\n\n"

        "SCREEN CONTENT (each screen MUST look different based on its type):\n"
        " Dashboard→stat cards row + charts + recent-items table | List→search bar+filters+large data table+pagination\n"
        " Form→labeled input groups in 2 columns+submit/cancel | Detail→entity header+status+tabbed info cards\n"
        " Settings→category tabs left+form right | Analytics→date filters+2x2 chart grid | Calendar→toggle bar+calendar grid\n"
        " Chat→contacts list left+messages center+detail right\n\n"

        "NO-OVERLAP RULES (mandatory):\n"
        " - Vertical: next element y >= prev.y + prev.h + 12\n"
        " - Horizontal: next element x >= prev.x + prev.w + 16\n"
        " - Text h = fontSize + 12 (e.g. fontSize:24→h:36, fontSize:14→h:26, fontSize:12→h:24)\n"
        " - Input/button min h:40. Labels above inputs: label at y, input at y+label.h+4\n"
        " - Children inside card/form: fit within parent bounds with 16px padding\n\n"

        "CONTENT RULES (CRITICAL — violations produce garbage designs):\n"
        " - ONLY design the screens listed in the user prompt. Do NOT invent extra screens.\n"
        " - Screen names in the output MUST exactly match the screen names provided by the user.\n"
        " - ALL text labels, button text, table column headers, input placeholders, card titles, chart titles,\n"
        "   stat labels, and menu items MUST be derived from the application OVERVIEW, FEATURES, and SCREEN DESCRIPTIONS.\n"
        " - NEVER use generic filler like 'Item 1', 'Column A', 'Card Title', 'Lorem ipsum', 'Sample Text', 'Click Here'.\n"
        " - For each screen: read its name, type, and description → then pick the specific data fields, actions, and entities from the overview that belong on that screen.\n"
        " - Nav links and sidebar items must be the actual screen names from the screens list.\n"
        " - Table columns must reflect the real data fields of the entity shown (e.g. for a patient list: 'Patient ID', 'Name', 'Doctor', 'Status', 'Last Visit').\n"
        " - Form inputs must have labels and placeholders matching real fields from the requirements.\n"
        " - Stat card labels must reference real KPIs from the domain (e.g. 'Total Orders', 'Active Users', not 'Stat 1').\n\n"

        "QUALITY:\n"
        " - Colors: 1 primary + 1 accent for industry. Backgrounds #F9FAFB/#F3F4F6, headings #111827, body #374151, secondary #6B7280, borders #E5E7EB\n"
        " - Typography: title 24/700, heading 18/600, body 14/400, label 13/500\n"
        " - Border radius: cards 12, inputs 8, avatars 16. Each screen: 20-40 elements\n"
        " - Tables: include 'columns' and 'rows' (3-5 domain-specific rows). Stats: real numbers. Charts: descriptive titles. Inputs: specific placeholders\n\n"

        "OUTPUT: Return ONLY valid JSON:\n"
        '{"appName":"...","screen":"<first>","screenType":"...","canvasWidth":1440,"canvasHeight":900,'
        '"elements":[...],"allScreens":[{"name":"...","type":"...","elements":[...]},...]}\n'
        'Element: {"type":"...","x":N,"y":N,"w":N,"h":N,"label":"...","props":{...}}'
    )

    # Build a per-screen breakdown so the model knows exactly what content each screen needs
    screen_breakdown = ""
    if request.screens:
        screen_breakdown = "\nPER-SCREEN CONTEXT — use these descriptions to design each screen's content:\n"
        all_features = request.features or []
        for i, s in enumerate(request.screens, 1):
            name = s.get("name", f"Screen {i}")
            stype = s.get("type", "page").lower()
            desc = s.get("description", "")
            # Find features that relate to this screen by matching words
            name_words = [w.lower() for w in name.split() if len(w) > 2]
            desc_words = [w.lower() for w in desc.split() if len(w) > 3] if desc else []
            search_words = name_words + desc_words
            related = [f for f in all_features if any(word in f.lower() for word in search_words)]
            screen_breakdown += f"\n  Screen {i}: '{name}' (type: {stype})\n"
            screen_breakdown += f"    Description: {desc if desc else name}\n"
            if related:
                screen_breakdown += f"    Related features: {'; '.join(related)}\n"

    user_prompt = f"""Design UI for: {request.applicationName}
Industry: {request.industry or 'General'} | Type: {request.applicationType or 'Web App'} | Theme: {request.theme or 'modern'}

APPLICATION OVERVIEW (read carefully — ALL screen content must come from this):
{request.overview}

SCREENS TO DESIGN (design ONLY these screens, use these EXACT names):
{screens_info}
{flow_info if flow_info else ''}{features_info if features_info else ''}{user_types_info if user_types_info else ''}{screen_breakdown}
CRITICAL RULES:
1. Design ONLY the screens listed above — no extra screens, no renamed screens.
2. Every label, column, button, placeholder, stat title, and chart title must come from the OVERVIEW and FEATURES above. Zero generic text.
3. Nav bar links and sidebar items = the screen names listed above.
4. Each screen must have a different layout matching its type. Include nav+sidebar+footer on app screens. No overlapping elements."""

    async def event_stream():
        start = time.time()
        yield f"data: {json.dumps({'field': 'thinking'})}\n\n"

        accumulated = ""
        emitted_screens = 0

        def try_extract_screens(text: str, already_emitted: int):
            """Try to extract completed screen objects from allScreens array as they stream in."""
            screens_found = []
            key = '"allScreens"'
            idx = text.find(key)
            if idx == -1:
                return screens_found
            arr_start = text.find('[', idx + len(key))
            if arr_start == -1:
                return screens_found
            # Walk through the array looking for complete {...} objects
            pos = arr_start + 1
            depth = 0
            in_str = False
            obj_start = -1
            obj_count = 0
            while pos < len(text):
                c = text[pos]
                if in_str:
                    if c == '\\':
                        pos += 2
                        continue
                    if c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == '{':
                        if depth == 0:
                            obj_start = pos
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0 and obj_start != -1:
                            obj_count += 1
                            if obj_count > already_emitted:
                                try:
                                    screen_obj = json.loads(text[obj_start:pos + 1])
                                    screens_found.append(screen_obj)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            obj_start = -1
                    elif c == ']' and depth == 0:
                        break
                pos += 1
            return screens_found

        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
            stream = await client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                stream=True,
            )
            async for event in stream:
                if event.choices and event.choices[0].delta.content:
                    accumulated += event.choices[0].delta.content

                    # Try to emit newly completed screens as they arrive
                    new_screens = try_extract_screens(accumulated, emitted_screens)
                    for screen_obj in new_screens:
                        emitted_screens += 1
                        yield f"data: {json.dumps({'field': 'screen', 'index': emitted_screens - 1, 'data': screen_obj})}\n\n"

            if not accumulated:
                yield f"data: {json.dumps({'field': 'error', 'detail': 'OpenAI returned empty content'})}\n\n"
                return

            parsed = json.loads(accumulated)

            # Emit any screens missed during streaming
            all_screens = parsed.get("allScreens", [])
            for i in range(emitted_screens, len(all_screens)):
                yield f"data: {json.dumps({'field': 'screen', 'index': i, 'data': all_screens[i]})}\n\n"

            # Emit the full design as well for clients that prefer it
            yield f"data: {json.dumps({'field': 'design', 'data': parsed})}\n\n"

            elapsed = round(time.time() - start, 2)
            screen_count = len(all_screens)
            yield f"data: {json.dumps({'field': 'done', 'screenCount': screen_count, 'elapsed': elapsed})}\n\n"

        except json.JSONDecodeError:
            logger.error(f"Frontend design: invalid JSON. Raw: {accumulated[:500]}")
            yield f"data: {json.dumps({'field': 'error', 'detail': 'OpenAI returned invalid JSON'})}\n\n"
        except Exception as e:
            logger.error(f"Frontend design OpenAI error: {e}")
            yield f"data: {json.dumps({'field': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)