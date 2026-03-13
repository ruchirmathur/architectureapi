"""
Cosmos DB Service for managing requirements and users
"""
from azure.cosmos.aio import CosmosClient
from azure.cosmos import exceptions, PartitionKey
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class CosmosDBService:
    """Service for interacting with Cosmos DB"""
    
    def __init__(self, endpoint: str, key: str, database_name: str, 
                 requirements_container: str, users_container: str,
                 recommendations_container: str = "recommendations",
                 designs_container: str = "designs"):
        """Initialize Cosmos DB client and get references to existing containers"""
        try:
            # Create Cosmos client
            self.client = CosmosClient(endpoint, credential=key)
            
            # Get existing database
            self.database = self.client.get_database_client(database=database_name)
            
            # Get existing containers
            self.requirements_container = self.database.get_container_client(
                container=requirements_container
            )
            self.users_container = self.database.get_container_client(
                container=users_container
            )
            self.recommendations_container = self.database.get_container_client(
                container=recommendations_container
            )
            self.designs_container = self.database.get_container_client(
                container=designs_container
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB client: {str(e)}", exc_info=True)
            raise
    
    async def validate_connection(self):
        """Validate Cosmos DB connection by reading database properties"""
        try:
            await self.database.read()
        except Exception as e:
            logger.error(f"Failed to validate Cosmos DB connection: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """Close Cosmos DB client"""
        if self.client:
            await self.client.close()
            logger.info("Cosmos DB connection closed")
    
    async def health_check(self) -> bool:
        """Check if Cosmos DB is healthy"""
        try:
            if not self.database:
                return False
            await self.database.read()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def get_user(self, tenant_id: str, username: str) -> Optional[Dict[str, Any]]:
        """Get user by tenant ID and username"""
        try:
            query = "SELECT * FROM c WHERE c.tenantId = @tenantId AND c.username = @username"
            parameters = [
                {"name": "@tenantId", "value": tenant_id},
                {"name": "@username", "value": username}
            ]
            
            items = self.users_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=tenant_id
            )
            
            async for item in items:
                return item
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}", exc_info=True)
            return None
    
    async def create_requirement(self, requirement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new requirement in Cosmos DB"""
        try:
            # Generate unique ID
            requirement_data["id"] = str(uuid.uuid4())
            
            # Add timestamps
            current_time = datetime.utcnow().isoformat() + "Z"
            requirement_data["createdAt"] = current_time
            requirement_data["updatedAt"] = current_time
            
            # Insert into Cosmos DB
            created_item = await self.requirements_container.create_item(
                body=requirement_data,
                enable_automatic_id_generation=False
            )
            
            logger.info(f"Created requirement: {created_item['id']}")
            return created_item
            
        except Exception as e:
            logger.error(f"Error creating requirement: {str(e)}", exc_info=True)
            raise
    
    async def get_requirement(self, requirement_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific requirement by ID and tenant"""
        try:
            query = "SELECT * FROM c WHERE c.id = @id AND c.tenantId = @tenantId"
            parameters = [
                {"name": "@id", "value": requirement_id},
                {"name": "@tenantId", "value": tenant_id}
            ]
            
            items = self.requirements_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=tenant_id
            )
            
            async for item in items:
                return item
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting requirement: {str(e)}", exc_info=True)
            return None
    
    async def get_requirement_by_session(self, session_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get a requirement by sessionId and tenant"""
        try:
            query = "SELECT * FROM c WHERE c.sessionId = @sessionId AND c.tenantId = @tenantId"
            parameters = [
                {"name": "@sessionId", "value": session_id},
                {"name": "@tenantId", "value": tenant_id}
            ]
            
            items = self.requirements_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=tenant_id
            )
            
            async for item in items:
                return item
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting requirement by session: {str(e)}", exc_info=True)
            return None
    
    async def list_requirements(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """List all requirements for a tenant"""
        try:
            query = "SELECT * FROM c WHERE c.tenantId = @tenantId ORDER BY c.createdAt DESC"
            parameters = [
                {"name": "@tenantId", "value": tenant_id}
            ]
            
            items = self.requirements_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=tenant_id,
                max_item_count=limit
            )
            
            results = []
            async for item in items:
                results.append(item)
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing requirements: {str(e)}", exc_info=True)
            return []
    
    async def get_requirement_by_application_name(self, application_name: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get requirement by application name for a tenant"""
        try:
            query = "SELECT * FROM c WHERE c.tenantId = @tenantId AND c.applicationName = @applicationName"
            parameters = [
                {"name": "@tenantId", "value": tenant_id},
                {"name": "@applicationName", "value": application_name}
            ]
            
            items = self.requirements_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=tenant_id
            )
            
            async for item in items:
                return item  # Return first match
            
            return None
            
        except Exception as e:
            logger.error(f"Error querying requirement by application name: {str(e)}", exc_info=True)
            return None
    
    async def update_requirement(self, requirement_id: str, tenant_id: str, 
                                 updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing requirement"""
        try:
            # Get existing requirement
            existing = await self.get_requirement(requirement_id, tenant_id)
            if not existing:
                return None
            
            # Apply updates
            existing.update(updates)
            existing["updatedAt"] = datetime.utcnow().isoformat() + "Z"
            
            # Replace in Cosmos DB
            updated_item = await self.requirements_container.replace_item(
                item=existing["id"],
                body=existing
            )
            
            logger.info(f"Updated requirement: {requirement_id}")
            return updated_item
            
        except Exception as e:
            logger.error(f"Error updating requirement: {str(e)}", exc_info=True)
            raise
    
    async def delete_requirement(self, requirement_id: str, tenant_id: str) -> bool:
        """Delete a requirement"""
        try:
            await self.requirements_container.delete_item(
                item=requirement_id,
                partition_key=tenant_id
            )
            
            logger.info(f"Deleted requirement: {requirement_id}")
            return True
            
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Requirement not found: {requirement_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting requirement: {str(e)}", exc_info=True)
            raise
    
    async def get_recommendation(self, application_name: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get an existing recommendation by applicationName and tenant"""
        try:
            query = """
            SELECT * FROM c 
            WHERE c.tenantId = @tenantId 
            AND c.applicationName = @applicationName
            """
            parameters = [
                {"name": "@tenantId", "value": tenant_id},
                {"name": "@applicationName", "value": application_name}
            ]
            
            # Cross-partition query (no partition_key specified for composite keys)
            items = self.recommendations_container.query_items(
                query=query,
                parameters=parameters
            )
            
            async for item in items:
                return item  # Return first match
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recommendation: {str(e)}", exc_info=True)
            return None
    
    async def create_recommendation(self, recommendation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new recommendation in Cosmos DB"""
        try:
            # Generate unique ID if not present
            if "id" not in recommendation_data:
                recommendation_data["id"] = str(uuid.uuid4())
            
            # Add timestamps
            current_time = datetime.utcnow().isoformat() + "Z"
            recommendation_data["createdAt"] = current_time
            recommendation_data["updatedAt"] = current_time
            
            # Insert into Cosmos DB
            created_item = await self.recommendations_container.create_item(
                body=recommendation_data,
                enable_automatic_id_generation=False
            )
            
            logger.info(f"Created recommendation: {created_item['id']}")
            return created_item
            
        except Exception as e:
            logger.error(f"Error creating recommendation: {str(e)}", exc_info=True)
            raise

    async def create_design(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new design (LLD) in Cosmos DB"""
        try:
            # Generate unique ID if not present
            if "id" not in design_data:
                design_data["id"] = str(uuid.uuid4())
            
            # Add timestamps
            current_time = datetime.utcnow().isoformat() + "Z"
            design_data["createdAt"] = current_time
            design_data["updatedAt"] = current_time
            
            # Insert into Cosmos DB
            created_item = await self.designs_container.create_item(
                body=design_data,
                enable_automatic_id_generation=False
            )
            
            logger.info(f"Created design: {created_item['id']}")
            return created_item
            
        except Exception as e:
            logger.error(f"Error creating design: {str(e)}", exc_info=True)
            raise

    async def get_design(self, design_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get a design by ID and tenant"""
        try:
            logger.info(f"Querying design by ID: designId={design_id}, tenantId={tenant_id}")
            
            query = "SELECT * FROM c WHERE c.id = @id AND c.tenantId = @tenantId"
            parameters = [
                {"name": "@id", "value": design_id},
                {"name": "@tenantId", "value": tenant_id}
            ]
            
            logger.info(f"Executing query: {query}")
            logger.info(f"Query parameters: {parameters}")
            
            items = self.designs_container.query_items(
                query=query,
                parameters=parameters,
                max_item_count=1
            )
            
            async for item in items:
                logger.info(f"Retrieved design: designId={item.get('designId')}, type={item.get('type')}, size=~{len(str(item))} chars")
                return item
            
            logger.warning(f"No design found for designId={design_id}, tenantId={tenant_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting design: {str(e)}", exc_info=True)
            return None

    async def get_design_by_architecture_id(self, architecture_id: str, tenant_id: str, application_name: str = None) -> Optional[Dict[str, Any]]:
        """Alias for get_design_by_architecture for backwards compatibility"""
        logger.info(f"get_design_by_architecture_id called with: architectureId={architecture_id}, tenantId={tenant_id}, applicationName={application_name}")
        return await self.get_design_by_architecture(architecture_id, tenant_id, application_name)

    async def get_design_by_architecture(self, architecture_id: str, tenant_id: str, application_name: str = None) -> Optional[Dict[str, Any]]:
        """Get a design by architectureId, tenantId, and optionally applicationName"""
        try:
            logger.info(f"Querying LLD: architectureId={architecture_id}, tenantId={tenant_id}, applicationName={application_name}")
            
            # Build query dynamically based on available parameters
            if application_name:
                query = "SELECT * FROM c WHERE c.architectureId = @architectureId AND c.tenantId = @tenantId AND c.applicationName = @applicationName AND c.type = 'lowLevelDesign' ORDER BY c._ts DESC"
                parameters = [
                    {"name": "@architectureId", "value": architecture_id},
                    {"name": "@tenantId", "value": tenant_id},
                    {"name": "@applicationName", "value": application_name}
                ]
            else:
                # Fallback: query without applicationName if not provided
                query = "SELECT * FROM c WHERE c.architectureId = @architectureId AND c.tenantId = @tenantId AND c.type = 'lowLevelDesign' ORDER BY c._ts DESC"
                parameters = [
                    {"name": "@architectureId", "value": architecture_id},
                    {"name": "@tenantId", "value": tenant_id}
                ]
            
            logger.info(f"Executing query: {query}")
            logger.info(f"Query parameters: {parameters}")
            
            items = self.designs_container.query_items(
                query=query,
                parameters=parameters,
                max_item_count=1
            )
            
            async for item in items:
                logger.info(f"Retrieved LLD design: designId={item.get('designId')}, architectureId={item.get('architectureId')}, size=~{len(str(item))} chars")
                return item  # Return the most recent one
            
            logger.warning(f"No LLD design found for architectureId={architecture_id}, tenantId={tenant_id}, applicationName={application_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting design by architecture: {str(e)}", exc_info=True)
            return None

    async def get_design_by_architecture_flexible(self, architecture_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a design by architectureId and tenantId only (no applicationName requirement).
        This is a fallback method when exact applicationName match fails.
        """
        try:
            query = "SELECT * FROM c WHERE c.architectureId = @architectureId AND c.tenantId = @tenantId AND c.type = 'lowLevelDesign' ORDER BY c._ts DESC"
            parameters = [
                {"name": "@architectureId", "value": architecture_id},
                {"name": "@tenantId", "value": tenant_id}
            ]
            
            logger.info(f"Flexible LLD query: {query}")
            logger.info(f"Parameters: {parameters}")
            
            items = self.designs_container.query_items(
                query=query,
                parameters=parameters,
                max_item_count=1
            )
            
            async for item in items:
                logger.info(f"Found LLD via flexible query: designId={item.get('designId')}, applicationName={item.get('applicationName')}")
                return item
            
            logger.info(f"No LLD found via flexible query for architectureId={architecture_id}, tenantId={tenant_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error in flexible design query: {str(e)}", exc_info=True)
            return None

    async def get_generated_code_by_design_id(self, design_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get generated code by designId and tenantId from the designs container"""
        try:
            logger.info(f"Querying generated code: designId={design_id}, tenantId={tenant_id}")

            query = "SELECT * FROM c WHERE c.designId = @designId AND c.tenantId = @tenantId AND c.type = 'generatedCode' ORDER BY c._ts DESC"
            parameters = [
                {"name": "@designId", "value": design_id},
                {"name": "@tenantId", "value": tenant_id}
            ]

            items = self.designs_container.query_items(
                query=query,
                parameters=parameters,
                max_item_count=1
            )

            async for item in items:
                logger.info(f"Found generated code for designId={design_id}, size=~{len(str(item))} chars")
                return item

            logger.info(f"No generated code found for designId={design_id}, tenantId={tenant_id}")
            return None

        except Exception as e:
            logger.error(f"Error getting generated code by design ID: {str(e)}", exc_info=True)
            return None

    async def list_designs_debug(self, tenant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Debug method to list designs for a tenant to help diagnose query issues.
        Returns basic info about designs in the container.
        """
        try:
            query = "SELECT c.id, c.designId, c.architectureId, c.tenantId, c.applicationName, c.type, c._ts FROM c WHERE c.tenantId = @tenantId ORDER BY c._ts DESC"
            parameters = [
                {"name": "@tenantId", "value": tenant_id}
            ]
            
            logger.info(f"Debug query: {query}")
            logger.info(f"Parameters: {parameters}")
            
            items = self.designs_container.query_items(
                query=query,
                parameters=parameters,
                max_item_count=limit
            )
            
            results = []
            async for item in items:
                results.append(item)
            
            logger.info(f"Found {len(results)} designs for tenant {tenant_id}")
            for design in results:
                logger.info(f"Design: id={design.get('id')}, designId={design.get('designId')}, architectureId={design.get('architectureId')}, applicationName={design.get('applicationName')}, type={design.get('type')}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error listing designs for debug: {str(e)}", exc_info=True)
            return []
