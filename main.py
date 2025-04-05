from typing import (
    Any,
    Dict,
    List,
    Optional,
    AsyncIterator,
    Callable,
    TypeVar,
    Awaitable,
    Union,
)
import os
import httpx
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
import inspect
from threading import Thread
import webbrowser
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv
import json
import time

"""
Superset MCP Integration

This module provides a Model Control Protocol (MCP) server for Apache Superset,
enabling AI assistants to interact with and control a Superset instance programmatically.

It includes tools for:
- Authentication and token management
- Dashboard operations (list, get, create, update, delete)
- Chart management (list, get, create, update, delete)
- Database and dataset operations
- SQL execution and query management
- User information and recent activity tracking
- Advanced data type handling
- Tag management

Each tool follows a consistent naming convention: superset_<category>_<action>
"""

# Load environment variables from .env file
load_dotenv()

# Constants
SUPERSET_BASE_URL = os.getenv("SUPERSET_BASE_URL", "http://localhost:8088")
SUPERSET_USERNAME = os.getenv("SUPERSET_USERNAME")
SUPERSET_PASSWORD = os.getenv("SUPERSET_PASSWORD")
ACCESS_TOKEN_STORE_PATH = os.path.join(os.path.dirname(__file__), ".superset_token")

# Initialize FastAPI app for handling additional web endpoints if needed
app = FastAPI(title="Superset MCP Server")


@dataclass
class SupersetContext:
    """Typed context for the Superset MCP server"""

    client: httpx.AsyncClient
    base_url: str
    access_token: Optional[str] = None
    csrf_token: Optional[str] = None
    app: FastAPI = None
    onboarding_completed: bool = False
    instance_name: Optional[str] = None
    instance_metadata: Dict[str, Any] = field(default_factory=dict)
    memories: Dict[str, Any] = field(default_factory=dict)


def load_stored_token() -> Optional[str]:
    """Load stored access token if it exists"""
    try:
        if os.path.exists(ACCESS_TOKEN_STORE_PATH):
            with open(ACCESS_TOKEN_STORE_PATH, "r") as f:
                return f.read().strip()
    except Exception:
        return None
    return None


def save_access_token(token: str):
    """Save access token to file"""
    try:
        with open(ACCESS_TOKEN_STORE_PATH, "w") as f:
            f.write(token)
    except Exception as e:
        print(f"Warning: Could not save access token: {e}")


def get_instance_name_from_url(url: str) -> str:
    """Extract a safe instance name from the URL for memory organization"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    # Use netloc (host and port) for unique instance identification
    instance_name = parsed.netloc.replace(":", "_")
    # Fallback if empty
    if not instance_name:
        instance_name = "default_instance"
    return instance_name


def get_memory_dir(instance_name: str) -> str:
    """Get the directory path for instance-specific memories"""
    memory_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".superset_memories", instance_name)
    # Ensure the directory exists
    os.makedirs(memory_dir, exist_ok=True)
    return memory_dir


def get_memory_path(instance_name: str, memory_name: str) -> str:
    """Get full path for a memory file"""
    memory_dir = get_memory_dir(instance_name)
    # Ensure the memory name is safe for filesystem
    safe_name = memory_name.replace("/", "_").replace("\\", "_")
    return os.path.join(memory_dir, f"{safe_name}.json")


def load_memory(instance_name: str, memory_name: str) -> Optional[Dict[str, Any]]:
    """Load a memory by name for specific instance"""
    try:
        file_path = get_memory_path(instance_name, memory_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.loads(f.read())
        return None
    except Exception as e:
        print(f"Error loading memory {memory_name}: {e}")
        return None


def save_memory(instance_name: str, memory_name: str, data: Dict[str, Any]) -> bool:
    """Save a memory by name for specific instance"""
    try:
        file_path = get_memory_path(instance_name, memory_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Update the memory index
        update_memory_index(instance_name, memory_name, data)
        return True
    except Exception as e:
        print(f"Error saving memory {memory_name}: {e}")
        return False


def list_memories(instance_name: str) -> List[str]:
    """List all available memories for specific instance"""
    try:
        memory_dir = get_memory_dir(instance_name)
        files = os.listdir(memory_dir)
        # Filter for json files and remove extension
        return [os.path.splitext(f)[0] for f in files if f.endswith('.json') and f != 'memory_index.json']
    except Exception as e:
        print(f"Error listing memories: {e}")
        return []


def update_memory_index(instance_name: str, memory_name: str, memory_data: Dict[str, Any]) -> bool:
    """Update the memory index with metadata about this memory"""
    try:
        memory_dir = get_memory_dir(instance_name)
        index_path = os.path.join(memory_dir, "memory_index.json")
        
        # Get or create index
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.loads(f.read())
        else:
            index = {
                "last_updated": int(time.time()),
                "categories": {
                    "datasets": [],
                    "dashboards": [],
                    "databases": [],
                    "queries": [],
                    "analysis_techniques": [],
                    "best_practices": [],
                    "templates": [],
                    "user_preferences": [],
                    "meta": [],
                    "other": []
                },
                "memories": {}
            }
        
        # Extract metadata from memory
        metadata = memory_data.get("metadata", {})
        category = metadata.get("category", "other")
        description = metadata.get("description", "")
        tags = metadata.get("tags", [])
        related_memories = metadata.get("related_memories", [])
        last_updated = metadata.get("last_updated", int(time.time()))
        
        # Update categories list if needed
        if category in index["categories"]:
            if memory_name not in index["categories"][category]:
                index["categories"][category].append(memory_name)
        else:
            # If category doesn't exist, add to "other"
            if memory_name not in index["categories"]["other"]:
                index["categories"]["other"].append(memory_name)
        
        # Update memory metadata in index
        index["memories"][memory_name] = {
            "category": category,
            "description": description,
            "last_updated": last_updated,
            "related_memories": related_memories,
            "tags": tags
        }
        
        # Update last_updated timestamp
        index["last_updated"] = int(time.time())
        
        # Save updated index
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error updating memory index: {e}")
        return False


def search_memories(instance_name: str, query: str) -> List[Dict[str, Any]]:
    """Simple search for memories based on metadata"""
    try:
        memory_dir = get_memory_dir(instance_name)
        index_path = os.path.join(memory_dir, "memory_index.json")
        
        if not os.path.exists(index_path):
            return []
        
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.loads(f.read())
        
        results = []
        query = query.lower()
        
        for memory_name, metadata in index["memories"].items():
            # Search in name, description, and tags
            if (query in memory_name.lower() or 
                query in metadata.get("description", "").lower() or 
                any(query in tag.lower() for tag in metadata.get("tags", []))):
                
                results.append({
                    "name": memory_name,
                    "metadata": metadata
                })
        
        return results
    except Exception as e:
        print(f"Error searching memories: {e}")
        return []


@asynccontextmanager
async def superset_lifespan(server: FastMCP) -> AsyncIterator[SupersetContext]:
    """Manage application lifecycle for Superset integration"""
    print("Initializing Superset context...")

    # Create HTTP client
    client = httpx.AsyncClient(base_url=SUPERSET_BASE_URL, timeout=30.0)

    # Extract instance name from URL
    instance_name = get_instance_name_from_url(SUPERSET_BASE_URL)
    print(f"Using instance name: {instance_name}")

    # Create context
    ctx = SupersetContext(
        client=client, 
        base_url=SUPERSET_BASE_URL, 
        app=app,
        instance_name=instance_name
    )

    # Try to load existing token
    stored_token = load_stored_token()
    if stored_token:
        ctx.access_token = stored_token
        # Set the token in the client headers
        client.headers.update({"Authorization": f"Bearer {stored_token}"})
        print("Using stored access token")

        # Verify token validity
        try:
            response = await client.get("/api/v1/me/")
            if response.status_code != 200:
                print(
                    f"Stored token is invalid (status {response.status_code}). Will need to re-authenticate."
                )
                ctx.access_token = None
                client.headers.pop("Authorization", None)
        except Exception as e:
            print(f"Error verifying stored token: {e}")
            ctx.access_token = None
            client.headers.pop("Authorization", None)
    
    # Load instance metadata if available
    try:
        instance_metadata = load_memory(instance_name, "instance_metadata")
        if instance_metadata:
            ctx.instance_metadata = instance_metadata.get("content", {})
            ctx.onboarding_completed = instance_metadata.get("content", {}).get("onboarding_completed", False)
            print(f"Loaded instance metadata for {instance_name}")
    except Exception as e:
        print(f"Warning: Failed to load instance metadata: {e}")

    try:
        yield ctx
    finally:
        # Cleanup on shutdown
        print("Shutting down Superset context...")
        await client.aclose()


# Initialize FastMCP server with lifespan and dependencies
mcp = FastMCP(
    "superset",
    lifespan=superset_lifespan,
    dependencies=["fastapi", "uvicorn", "python-dotenv", "httpx"],
)

# Type variables for generic function annotations
T = TypeVar("T")
R = TypeVar("R")

# ===== Helper Functions and Decorators =====


def requires_auth(
    func: Callable[..., Awaitable[Dict[str, Any]]],
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Decorator to check authentication before executing a function"""

    @wraps(func)
    async def wrapper(ctx: Context, *args, **kwargs) -> Dict[str, Any]:
        superset_ctx: SupersetContext = ctx.request_context.lifespan_context

        if not superset_ctx.access_token:
            return {"error": "Not authenticated. Please authenticate first."}

        return await func(ctx, *args, **kwargs)

    return wrapper


def handle_api_errors(
    func: Callable[..., Awaitable[Dict[str, Any]]],
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Decorator to handle API errors in a consistent way"""

    @wraps(func)
    async def wrapper(ctx: Context, *args, **kwargs) -> Dict[str, Any]:
        try:
            return await func(ctx, *args, **kwargs)
        except Exception as e:
            # Extract function name for better error context
            function_name = func.__name__
            return {"error": f"Error in {function_name}: {str(e)}"}

    return wrapper


async def with_auto_refresh(
    ctx: Context, api_call: Callable[[], Awaitable[httpx.Response]]
) -> httpx.Response:
    """
    Helper function to handle automatic token refreshing for API calls

    This function will attempt to execute the provided API call. If the call
    fails with a 401 Unauthorized error, it will try to refresh the token
    and retry the API call once.

    Args:
        ctx: The MCP context
        api_call: The API call function to execute (should be a callable that returns a response)
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # First attempt
    try:
        response = await api_call()

        # If not an auth error, return the response
        if response.status_code != 401:
            return response

    except httpx.HTTPStatusError as e:
        if e.response.status_code != 401:
            raise e
        response = e.response
    except Exception as e:
        # For other errors, just raise
        raise e

    # If we got a 401, try to refresh the token
    print("Received 401 Unauthorized. Attempting to refresh token...")
    refresh_result = await superset_auth_refresh_token(ctx)

    if refresh_result.get("error"):
        # If refresh failed, try to re-authenticate
        print(
            f"Token refresh failed: {refresh_result.get('error')}. Attempting re-authentication..."
        )
        auth_result = await superset_auth_authenticate_user(ctx)

        if auth_result.get("error"):
            # If re-authentication failed, raise an exception
            raise HTTPException(status_code=401, detail="Authentication failed")

    # Retry the API call with the new token
    return await api_call()


async def get_csrf_token(ctx: Context) -> Optional[str]:
    """
    Get a CSRF token from Superset

    Makes a request to the /api/v1/security/csrf_token endpoint to get a token

    Args:
        ctx: MCP context
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    client = superset_ctx.client

    try:
        response = await client.get("/api/v1/security/csrf_token/")
        if response.status_code == 200:
            data = response.json()
            csrf_token = data.get("result")
            superset_ctx.csrf_token = csrf_token
            return csrf_token
        else:
            print(f"Failed to get CSRF token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting CSRF token: {str(e)}")
        return None


async def make_api_request(
    ctx: Context,
    method: str,
    endpoint: str,
    data: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    auto_refresh: bool = True,
) -> Dict[str, Any]:
    """
    Helper function to make API requests to Superset

    Args:
        ctx: MCP context
        method: HTTP method (get, post, put, delete)
        endpoint: API endpoint (without base URL)
        data: Optional JSON payload for POST/PUT requests
        params: Optional query parameters
        auto_refresh: Whether to auto-refresh token on 401
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    client = superset_ctx.client

    # For non-GET requests, make sure we have a CSRF token
    if method.lower() != "get" and not superset_ctx.csrf_token:
        await get_csrf_token(ctx)

    async def make_request() -> httpx.Response:
        headers = {}

        # Add CSRF token for non-GET requests
        if method.lower() != "get" and superset_ctx.csrf_token:
            headers["X-CSRFToken"] = superset_ctx.csrf_token

        if method.lower() == "get":
            return await client.get(endpoint, params=params)
        elif method.lower() == "post":
            return await client.post(
                endpoint, json=data, params=params, headers=headers
            )
        elif method.lower() == "put":
            return await client.put(endpoint, json=data, headers=headers)
        elif method.lower() == "delete":
            return await client.delete(endpoint, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    # Use auto_refresh if requested
    response = (
        await with_auto_refresh(ctx, make_request)
        if auto_refresh
        else await make_request()
    )

    if response.status_code not in [200, 201]:
        return {
            "error": f"API request failed: {response.status_code} - {response.text}"
        }

    return response.json()


# ===== Authentication Tools =====


@mcp.tool()
@handle_api_errors
async def superset_auth_check_token_validity(ctx: Context) -> Dict[str, Any]:
    """
    Check if the current access token is still valid

    Makes a request to the /api/v1/me/ endpoint to test if the current token is valid.
    Use this to verify authentication status before making other API calls.

    Returns:
        A dictionary with token validity status and any error information
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"valid": False, "error": "No access token available"}

    try:
        # Make a simple API call to test if token is valid (get user info)
        response = await superset_ctx.client.get("/api/v1/me/")

        if response.status_code == 200:
            return {"valid": True}
        else:
            return {
                "valid": False,
                "status_code": response.status_code,
                "error": response.text,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


@mcp.tool()
@handle_api_errors
async def superset_auth_refresh_token(ctx: Context) -> Dict[str, Any]:
    """
    Refresh the access token using the refresh endpoint

    Makes a request to the /api/v1/security/refresh endpoint to get a new access token
    without requiring re-authentication with username/password.

    Returns:
        A dictionary with the new access token or error information
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "No access token to refresh. Please authenticate first."}

    try:
        # Use the refresh endpoint to get a new token
        response = await superset_ctx.client.post("/api/v1/security/refresh")

        if response.status_code != 200:
            return {
                "error": f"Failed to refresh token: {response.status_code} - {response.text}"
            }

        data = response.json()
        access_token = data.get("access_token")

        if not access_token:
            return {"error": "No access token returned from refresh"}

        # Save and set the new access token
        save_access_token(access_token)
        superset_ctx.access_token = access_token
        superset_ctx.client.headers.update({"Authorization": f"Bearer {access_token}"})

        return {
            "message": "Successfully refreshed access token",
            "access_token": access_token,
        }
    except Exception as e:
        return {"error": f"Error refreshing token: {str(e)}"}


@mcp.tool()
@handle_api_errors
async def superset_auth_authenticate_user(
    ctx: Context,
    username: Optional[str] = None,
    password: Optional[str] = None,
    refresh: bool = True,
) -> Dict[str, Any]:
    """
    Authenticate with Superset and get access token

    Makes a request to the /api/v1/security/login endpoint to authenticate and obtain an access token.
    If there's an existing token, will first try to check its validity.
    If invalid, will attempt to refresh token before falling back to re-authentication.

    Args:
        username: Superset username (falls back to environment variable if not provided)
        password: Superset password (falls back to environment variable if not provided)
        refresh: Whether to refresh the token if invalid (defaults to True)

    Returns:
        A dictionary with authentication status and access token or error information
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    # If we already have a token, check if it's valid
    if superset_ctx.access_token:
        validity = await superset_auth_check_token_validity(ctx)

        if validity.get("valid"):
            return {
                "message": "Already authenticated with valid token",
                "access_token": superset_ctx.access_token,
            }

        # Token invalid, try to refresh if requested
        if refresh:
            refresh_result = await superset_auth_refresh_token(ctx)
            if not refresh_result.get("error"):
                return refresh_result
            # If refresh fails, fall back to re-authentication

    # Use provided credentials or fall back to env vars
    username = username or SUPERSET_USERNAME
    password = password or SUPERSET_PASSWORD

    if not username or not password:
        return {
            "error": "Username and password must be provided either as arguments or set in environment variables"
        }

    try:
        # Get access token directly using the security login API endpoint
        response = await superset_ctx.client.post(
            "/api/v1/security/login",
            json={
                "username": username,
                "password": password,
                "provider": "db",
                "refresh": refresh,
            },
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to get access token: {response.status_code} - {response.text}"
            }

        data = response.json()
        access_token = data.get("access_token")

        if not access_token:
            return {"error": "No access token returned"}

        # Save and set the access token
        save_access_token(access_token)
        superset_ctx.access_token = access_token
        superset_ctx.client.headers.update({"Authorization": f"Bearer {access_token}"})

        # Get CSRF token after successful authentication
        await get_csrf_token(ctx)

        return {
            "message": "Successfully authenticated with Superset",
            "access_token": access_token,
        }

    except Exception as e:
        return {"error": f"Authentication error: {str(e)}"}


# ===== Dashboard Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_list(ctx: Context) -> Dict[str, Any]:
    """
    Get a list of dashboards from Superset

    Makes a request to the /api/v1/dashboard/ endpoint to retrieve all dashboards
    the current user has access to view. Results are paginated.

    Returns:
        A dictionary containing dashboard data including id, title, url, and metadata
    """
    return await make_api_request(ctx, "get", "/api/v1/dashboard/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_get_by_id(
    ctx: Context, dashboard_id: int
) -> Dict[str, Any]:
    """
    Get details for a specific dashboard

    Makes a request to the /api/v1/dashboard/{id} endpoint to retrieve detailed
    information about a specific dashboard.

    Args:
        dashboard_id: ID of the dashboard to retrieve

    Returns:
        A dictionary with complete dashboard information including components and layout
    """
    return await make_api_request(ctx, "get", f"/api/v1/dashboard/{dashboard_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_create(
    ctx: Context, dashboard_title: str, json_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a new dashboard in Superset

    Makes a request to the /api/v1/dashboard/ POST endpoint to create a new dashboard.

    Args:
        dashboard_title: Title of the dashboard
        json_metadata: Optional JSON metadata for dashboard configuration,
                       can include layout, color scheme, and filter configuration

    Returns:
        A dictionary with the created dashboard information including its ID
    """
    payload = {"dashboard_title": dashboard_title}
    if json_metadata:
        payload["json_metadata"] = json_metadata

    return await make_api_request(ctx, "post", "/api/v1/dashboard/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_update(
    ctx: Context, dashboard_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing dashboard

    Makes a request to the /api/v1/dashboard/{id} PUT endpoint to update
    dashboard properties.

    Args:
        dashboard_id: ID of the dashboard to update
        data: Data to update, can include dashboard_title, slug, owners, position, and metadata

    Returns:
        A dictionary with the updated dashboard information
    """
    return await make_api_request(
        ctx, "put", f"/api/v1/dashboard/{dashboard_id}", data=data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_delete(ctx: Context, dashboard_id: int) -> Dict[str, Any]:
    """
    Delete a dashboard

    Makes a request to the /api/v1/dashboard/{id} DELETE endpoint to remove a dashboard.
    This operation is permanent and cannot be undone.

    Args:
        dashboard_id: ID of the dashboard to delete

    Returns:
        A dictionary with deletion confirmation message
    """
    response = await make_api_request(
        ctx, "delete", f"/api/v1/dashboard/{dashboard_id}"
    )

    # For delete endpoints, we may want a custom success message
    if not response.get("error"):
        return {"message": f"Dashboard {dashboard_id} deleted successfully"}

    return response


# ===== Chart Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_list(ctx: Context) -> Dict[str, Any]:
    """
    Get a list of charts from Superset

    Makes a request to the /api/v1/chart/ endpoint to retrieve all charts
    the current user has access to view. Results are paginated.

    Returns:
        A dictionary containing chart data including id, slice_name, viz_type, and datasource info
    """
    return await make_api_request(ctx, "get", "/api/v1/chart/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_get_by_id(ctx: Context, chart_id: int) -> Dict[str, Any]:
    """
    Get details for a specific chart

    Makes a request to the /api/v1/chart/{id} endpoint to retrieve detailed
    information about a specific chart/slice.

    Args:
        chart_id: ID of the chart to retrieve

    Returns:
        A dictionary with complete chart information including visualization configuration
    """
    return await make_api_request(ctx, "get", f"/api/v1/chart/{chart_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_create(
    ctx: Context,
    slice_name: str,
    datasource_id: int,
    datasource_type: str,
    viz_type: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a new chart in Superset

    Makes a request to the /api/v1/chart/ POST endpoint to create a new visualization.

    Args:
        slice_name: Name/title of the chart
        datasource_id: ID of the dataset or SQL table
        datasource_type: Type of datasource ('table' for datasets, 'query' for SQL)
        viz_type: Visualization type (e.g., 'bar', 'line', 'pie', 'big_number', etc.)
        params: Visualization parameters including metrics, groupby, time_range, etc.

    Returns:
        A dictionary with the created chart information including its ID
    """
    payload = {
        "slice_name": slice_name,
        "datasource_id": datasource_id,
        "datasource_type": datasource_type,
        "viz_type": viz_type,
        "params": params,
    }

    return await make_api_request(ctx, "post", "/api/v1/chart/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_update(
    ctx: Context, chart_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing chart

    Makes a request to the /api/v1/chart/{id} PUT endpoint to update
    chart properties and visualization settings.

    Args:
        chart_id: ID of the chart to update
        data: Data to update, can include slice_name, description, viz_type, params, etc.

    Returns:
        A dictionary with the updated chart information
    """
    return await make_api_request(ctx, "put", f"/api/v1/chart/{chart_id}", data=data)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_delete(ctx: Context, chart_id: int) -> Dict[str, Any]:
    """
    Delete a chart

    Makes a request to the /api/v1/chart/{id} DELETE endpoint to remove a chart.
    This operation is permanent and cannot be undone.

    Args:
        chart_id: ID of the chart to delete

    Returns:
        A dictionary with deletion confirmation message
    """
    response = await make_api_request(ctx, "delete", f"/api/v1/chart/{chart_id}")

    if not response.get("error"):
        return {"message": f"Chart {chart_id} deleted successfully"}

    return response


# ===== Database Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_list(ctx: Context) -> Dict[str, Any]:
    """
    Get a list of databases from Superset

    Makes a request to the /api/v1/database/ endpoint to retrieve all database
    connections the current user has access to. Results are paginated.

    Returns:
        A dictionary containing database connection information including id, name, and configuration
    """
    return await make_api_request(ctx, "get", "/api/v1/database/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_by_id(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Get details for a specific database

    Makes a request to the /api/v1/database/{id} endpoint to retrieve detailed
    information about a specific database connection.

    Args:
        database_id: ID of the database to retrieve

    Returns:
        A dictionary with complete database configuration information
    """
    return await make_api_request(ctx, "get", f"/api/v1/database/{database_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_create(
    ctx: Context,
    engine: str,
    configuration_method: str,
    database_name: str,
    sqlalchemy_uri: str,
) -> Dict[str, Any]:
    """
    Create a new database connection in Superset

    IMPORTANT: Don't call this tool, unless user have given connection details. This function will only create database connections with explicit user consent and input.
    No default values or assumptions will be made without user confirmation. All connection parameters,
    including sensitive credentials, must be explicitly provided by the user.

    Makes a POST request to /api/v1/database/ to create a new database connection in Superset.
    The endpoint requires a valid SQLAlchemy URI and database configuration parameters.
    The engine parameter will be automatically determined from the SQLAlchemy URI prefix if not specified:
    - 'postgresql://' -> engine='postgresql'
    - 'mysql://' -> engine='mysql'
    - 'mssql://' -> engine='mssql'
    - 'oracle://' -> engine='oracle'
    - 'sqlite://' -> engine='sqlite'

    The SQLAlchemy URI must follow the format: dialect+driver://username:password@host:port/database
    If the URI is not provided, the function will prompt for individual connection parameters to construct it.

    All required parameters must be provided and validated before creating the connection.
    The configuration_method parameter should typically be set to 'sqlalchemy_form'.

    Args:
        engine: Database engine (e.g., 'postgresql', 'mysql', etc.)
        configuration_method: Method used for configuration (typically 'sqlalchemy_form')
        database_name: Name for the database connection
        sqlalchemy_uri: SQLAlchemy URI for the connection (e.g., 'postgresql://user:pass@host/db')

    Returns:
        A dictionary with the created database connection information including its ID
    """
    payload = {
        "engine": engine,
        "configuration_method": configuration_method,
        "database_name": database_name,
        "sqlalchemy_uri": sqlalchemy_uri,
        "allow_dml": True,
        "allow_cvas": True,
        "allow_ctas": True,
        "expose_in_sqllab": True,
    }

    return await make_api_request(ctx, "post", "/api/v1/database/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_tables(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get a list of tables for a given database

    Makes a request to the /api/v1/database/{id}/tables/ endpoint to retrieve
    all tables available in the database.

    Args:
        database_id: ID of the database

    Returns:
        A dictionary with list of tables including schema and table name information
    """
    return await make_api_request(ctx, "get", f"/api/v1/database/{database_id}/tables/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_schemas(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Get schemas for a specific database

    Makes a request to the /api/v1/database/{id}/schemas/ endpoint to retrieve
    all schemas available in the database.

    Args:
        database_id: ID of the database

    Returns:
        A dictionary with list of schema names
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/schemas/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_test_connection(
    ctx: Context, database_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Test a database connection

    Makes a request to the /api/v1/database/test_connection endpoint to verify if
    the provided connection details can successfully connect to the database.

    Args:
        database_data: Database connection details including sqlalchemy_uri and other parameters

    Returns:
        A dictionary with connection test results
    """
    return await make_api_request(
        ctx, "post", "/api/v1/database/test_connection", data=database_data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_update(
    ctx: Context, database_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing database connection

    Makes a request to the /api/v1/database/{id} PUT endpoint to update
    database connection properties.

    Args:
        database_id: ID of the database to update
        data: Data to update, can include database_name, sqlalchemy_uri, password, and extra configs

    Returns:
        A dictionary with the updated database information
    """
    return await make_api_request(
        ctx, "put", f"/api/v1/database/{database_id}", data=data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_delete(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Delete a database connection

    Makes a request to the /api/v1/database/{id} DELETE endpoint to remove a database connection.
    This operation is permanent and cannot be undone. This will also remove associated datasets.

    Args:
        database_id: ID of the database to delete

    Returns:
        A dictionary with deletion confirmation message
    """
    response = await make_api_request(ctx, "delete", f"/api/v1/database/{database_id}")

    if not response.get("error"):
        return {"message": f"Database {database_id} deleted successfully"}

    return response


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_catalogs(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get all catalogs from a database

    Makes a request to the /api/v1/database/{id}/catalogs/ endpoint to retrieve
    all catalogs available in the database.

    Args:
        database_id: ID of the database

    Returns:
        A dictionary with list of catalog names for databases that support catalogs
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/catalogs/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_connection(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get database connection information

    Makes a request to the /api/v1/database/{id}/connection endpoint to retrieve
    connection details for a specific database.

    Args:
        database_id: ID of the database

    Returns:
        A dictionary with detailed connection information
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/connection"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_function_names(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get function names supported by a database

    Makes a request to the /api/v1/database/{id}/function_names/ endpoint to retrieve
    all SQL functions supported by the database.

    Args:
        database_id: ID of the database

    Returns:
        A dictionary with list of supported function names
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/function_names/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_related_objects(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get charts and dashboards associated with a database

    Makes a request to the /api/v1/database/{id}/related_objects/ endpoint to retrieve
    counts and references of charts and dashboards that depend on this database.

    Args:
        database_id: ID of the database

    Returns:
        A dictionary with counts and lists of related charts and dashboards
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/related_objects/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_validate_sql(
    ctx: Context, database_id: int, sql: str
) -> Dict[str, Any]:
    """
    Validate arbitrary SQL against a database

    Makes a request to the /api/v1/database/{id}/validate_sql/ endpoint to check
    if the provided SQL is valid for the specified database.

    Args:
        database_id: ID of the database
        sql: SQL query to validate

    Returns:
        A dictionary with validation results
    """
    payload = {"sql": sql}
    return await make_api_request(
        ctx, "post", f"/api/v1/database/{database_id}/validate_sql/", data=payload
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_validate_parameters(
    ctx: Context, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate database connection parameters

    Makes a request to the /api/v1/database/validate_parameters/ endpoint to verify
    if the provided connection parameters are valid without creating a connection.

    Args:
        parameters: Connection parameters to validate

    Returns:
        A dictionary with validation results
    """
    return await make_api_request(
        ctx, "post", "/api/v1/database/validate_parameters/", data=parameters
    )


# ===== Dataset Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_list(ctx: Context) -> Dict[str, Any]:
    """
    Get a list of datasets from Superset

    Makes a request to the /api/v1/dataset/ endpoint to retrieve all datasets
    the current user has access to view. Results are paginated.

    Returns:
        A dictionary containing dataset information including id, table_name, and database
    """
    return await make_api_request(ctx, "get", "/api/v1/dataset/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_get_by_id(ctx: Context, dataset_id: int) -> Dict[str, Any]:
    """
    Get details for a specific dataset

    Makes a request to the /api/v1/dataset/{id} endpoint to retrieve detailed
    information about a specific dataset including columns and metrics.

    Args:
        dataset_id: ID of the dataset to retrieve

    Returns:
        A dictionary with complete dataset information
    """
    return await make_api_request(ctx, "get", f"/api/v1/dataset/{dataset_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_create(
    ctx: Context,
    table_name: str,
    database_id: int,
    schema: str = None,
    owners: List[int] = None,
) -> Dict[str, Any]:
    """
    Create a new dataset in Superset

    Makes a request to the /api/v1/dataset/ POST endpoint to create a new dataset
    from an existing database table or view.

    Args:
        table_name: Name of the physical table in the database
        database_id: ID of the database where the table exists
        schema: Optional database schema name where the table is located
        owners: Optional list of user IDs who should own this dataset

    Returns:
        A dictionary with the created dataset information including its ID
    """
    payload = {
        "table_name": table_name,
        "database": database_id,
    }

    if schema:
        payload["schema"] = schema

    if owners:
        payload["owners"] = owners

    return await make_api_request(ctx, "post", "/api/v1/dataset/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_update(
    ctx: Context, dataset_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing dataset in Superset

    Makes a request to the /api/v1/dataset/{id} PUT endpoint to update properties
    of an existing dataset such as column configurations, metrics, and display settings.

    Args:
        dataset_id: ID of the dataset to update
        data: Data to update, can include:
              - table_name: Physical table name
              - columns: Column configurations
              - metrics: Metric definitions
              - owners: List of owner IDs
              - description: Dataset description
              - cache_timeout: Cache timeout in seconds
              - is_managed_externally: Whether managed externally
              - external_url: External URL for managed datasets
              - extra: JSON string with extra configuration

    Returns:
        A dictionary with the updated dataset information
    """
    return await make_api_request(ctx, "put", f"/api/v1/dataset/{dataset_id}", data=data)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_delete(ctx: Context, dataset_id: int) -> Dict[str, Any]:
    """
    Delete a dataset from Superset

    Makes a request to the /api/v1/dataset/{id} DELETE endpoint to remove a dataset.
    This operation is permanent and cannot be undone.

    Args:
        dataset_id: ID of the dataset to delete

    Returns:
        A dictionary with deletion confirmation message
    """
    response = await make_api_request(ctx, "delete", f"/api/v1/dataset/{dataset_id}")

    if not response.get("error"):
        return {"message": f"Dataset {dataset_id} deleted successfully"}

    return response


# ===== SQL Lab Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_execute_query(
    ctx: Context, database_id: int, sql: str
) -> Dict[str, Any]:
    """
    Execute a SQL query in SQL Lab

    Makes a request to the /api/v1/sqllab/execute/ endpoint to run a SQL query
    against the specified database.

    Args:
        database_id: ID of the database to query
        sql: SQL query to execute

    Returns:
        A dictionary with query results or execution status for async queries
    """
    # Ensure we have a CSRF token before executing the query
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    if not superset_ctx.csrf_token:
        await get_csrf_token(ctx)

    payload = {
        "database_id": database_id,
        "sql": sql,
        "schema": "",
        "tab": "MCP Query",
        "runAsync": False,
        "select_as_cta": False,
    }

    return await make_api_request(ctx, "post", "/api/v1/sqllab/execute/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_get_saved_queries(ctx: Context) -> Dict[str, Any]:
    """
    Get a list of saved queries from SQL Lab

    Makes a request to the /api/v1/saved_query/ endpoint to retrieve all saved queries
    the current user has access to. Results are paginated.

    Returns:
        A dictionary containing saved query information including id, label, and database
    """
    return await make_api_request(ctx, "get", "/api/v1/saved_query/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_format_sql(ctx: Context, sql: str) -> Dict[str, Any]:
    """
    Format a SQL query for better readability

    Makes a request to the /api/v1/sqllab/format_sql endpoint to apply standard
    formatting rules to the provided SQL query.

    Args:
        sql: SQL query to format

    Returns:
        A dictionary with the formatted SQL
    """
    payload = {"sql": sql}
    return await make_api_request(
        ctx, "post", "/api/v1/sqllab/format_sql", data=payload
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_get_results(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get results of a previously executed SQL query

    Makes a request to the /api/v1/sqllab/results/ endpoint to retrieve results
    for an asynchronous query using its result key.

    Args:
        key: Result key to retrieve

    Returns:
        A dictionary with query results including column information and data rows
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/sqllab/results/", params={"key": key}
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_estimate_query_cost(
    ctx: Context, database_id: int, sql: str, schema: str = None
) -> Dict[str, Any]:
    """
    Estimate the cost of executing a SQL query

    Makes a request to the /api/v1/sqllab/estimate endpoint to get approximate cost
    information for a query before executing it.

    Args:
        database_id: ID of the database
        sql: SQL query to estimate
        schema: Optional schema name

    Returns:
        A dictionary with estimated query cost metrics
    """
    payload = {
        "database_id": database_id,
        "sql": sql,
    }

    if schema:
        payload["schema"] = schema

    return await make_api_request(ctx, "post", "/api/v1/sqllab/estimate", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_export_query_results(
    ctx: Context, client_id: str
) -> Dict[str, Any]:
    """
    Export the results of a SQL query to CSV

    Makes a request to the /api/v1/sqllab/export/{client_id} endpoint to download
    query results in CSV format.

    Args:
        client_id: Client ID of the query

    Returns:
        A dictionary with the exported data or error information
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    try:
        response = await superset_ctx.client.get(f"/api/v1/sqllab/export/{client_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to export query results: {response.status_code} - {response.text}"
            }

        return {"message": "Query results exported successfully", "data": response.text}

    except Exception as e:
        return {"error": f"Error exporting query results: {str(e)}"}


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_get_bootstrap_data(ctx: Context) -> Dict[str, Any]:
    """
    Get the bootstrap data for SQL Lab

    Makes a request to the /api/v1/sqllab/ endpoint to retrieve configuration data
    needed for the SQL Lab interface.

    Returns:
        A dictionary with SQL Lab configuration including allowed databases and settings
    """
    return await make_api_request(ctx, "get", "/api/v1/sqllab/")


# ===== Saved Query Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_saved_query_get_by_id(ctx: Context, query_id: int) -> Dict[str, Any]:
    """
    Get details for a specific saved query

    Makes a request to the /api/v1/saved_query/{id} endpoint to retrieve information
    about a saved SQL query.

    Args:
        query_id: ID of the saved query to retrieve

    Returns:
        A dictionary with the saved query details including SQL text and database
    """
    return await make_api_request(ctx, "get", f"/api/v1/saved_query/{query_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_saved_query_create(
    ctx: Context, query_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new saved query

    Makes a request to the /api/v1/saved_query/ POST endpoint to save a SQL query
    for later reuse.

    Args:
        query_data: Dictionary containing the query information including:
                   - db_id: Database ID
                   - schema: Schema name (optional)
                   - sql: SQL query text
                   - label: Display name for the saved query
                   - description: Optional description of the query

    Returns:
        A dictionary with the created saved query information including its ID
    """
    return await make_api_request(ctx, "post", "/api/v1/saved_query/", data=query_data)


# ===== Query Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_query_stop(ctx: Context, client_id: str) -> Dict[str, Any]:
    """
    Stop a running query

    Makes a request to the /api/v1/query/stop endpoint to terminate a query that
    is currently running.

    Args:
        client_id: Client ID of the query to stop

    Returns:
        A dictionary with confirmation of query termination
    """
    payload = {"client_id": client_id}
    return await make_api_request(ctx, "post", "/api/v1/query/stop", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_query_list(ctx: Context) -> Dict[str, Any]:
    """
    Get a list of queries from Superset

    Makes a request to the /api/v1/query/ endpoint to retrieve query history.
    Results are paginated and include both finished and running queries.

    Returns:
        A dictionary containing query information including status, duration, and SQL
    """
    return await make_api_request(ctx, "get", "/api/v1/query/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_query_get_by_id(ctx: Context, query_id: int) -> Dict[str, Any]:
    """
    Get details for a specific query

    Makes a request to the /api/v1/query/{id} endpoint to retrieve detailed
    information about a specific query execution.

    Args:
        query_id: ID of the query to retrieve

    Returns:
        A dictionary with complete query execution information
    """
    return await make_api_request(ctx, "get", f"/api/v1/query/{query_id}")


# ===== Activity and User Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_activity_get_recent(ctx: Context) -> Dict[str, Any]:
    """
    Get recent activity data for the current user

    Makes a request to the /api/v1/log/recent_activity/ endpoint to retrieve
    a history of actions performed by the current user.

    Returns:
        A dictionary with recent user activities including viewed charts and dashboards
    """
    return await make_api_request(ctx, "get", "/api/v1/log/recent_activity/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_user_get_current(ctx: Context) -> Dict[str, Any]:
    """
    Get information about the currently authenticated user

    Makes a request to the /api/v1/me/ endpoint to retrieve the user's profile
    information including permissions and preferences.

    Returns:
        A dictionary with user profile data
    """
    return await make_api_request(ctx, "get", "/api/v1/me/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_user_get_roles(ctx: Context) -> Dict[str, Any]:
    """
    Get roles for the current user

    Makes a request to the /api/v1/me/roles/ endpoint to retrieve all roles
    assigned to the current user.

    Returns:
        A dictionary with user role information
    """
    return await make_api_request(ctx, "get", "/api/v1/me/roles/")


# ===== Tag Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_list(ctx: Context) -> Dict[str, Any]:
    """
    Get a list of tags from Superset

    Makes a request to the /api/v1/tag/ endpoint to retrieve all tags
    defined in the Superset instance.

    Returns:
        A dictionary containing tag information including id and name
    """
    return await make_api_request(ctx, "get", "/api/v1/tag/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_create(ctx: Context, name: str) -> Dict[str, Any]:
    """
    Create a new tag in Superset

    Makes a request to the /api/v1/tag/ POST endpoint to create a new tag
    that can be applied to objects like charts and dashboards.

    Args:
        name: Name for the tag

    Returns:
        A dictionary with the created tag information
    """
    payload = {"name": name}
    return await make_api_request(ctx, "post", "/api/v1/tag/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_get_by_id(ctx: Context, tag_id: int) -> Dict[str, Any]:
    """
    Get details for a specific tag

    Makes a request to the /api/v1/tag/{id} endpoint to retrieve information
    about a specific tag.

    Args:
        tag_id: ID of the tag to retrieve

    Returns:
        A dictionary with tag details
    """
    return await make_api_request(ctx, "get", f"/api/v1/tag/{tag_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_objects(ctx: Context) -> Dict[str, Any]:
    """
    Get objects associated with tags

    Makes a request to the /api/v1/tag/get_objects/ endpoint to retrieve
    all objects that have tags assigned to them.

    Returns:
        A dictionary with tagged objects grouped by tag
    """
    return await make_api_request(ctx, "get", "/api/v1/tag/get_objects/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_delete(ctx: Context, tag_id: int) -> Dict[str, Any]:
    """
    Delete a tag

    Makes a request to the /api/v1/tag/{id} DELETE endpoint to remove a tag.
    This operation is permanent and cannot be undone.

    Args:
        tag_id: ID of the tag to delete

    Returns:
        A dictionary with deletion confirmation message
    """
    response = await make_api_request(ctx, "delete", f"/api/v1/tag/{tag_id}")

    if not response.get("error"):
        return {"message": f"Tag {tag_id} deleted successfully"}

    return response


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_object_add(
    ctx: Context, object_type: str, object_id: int, tag_name: str
) -> Dict[str, Any]:
    """
    Add a tag to an object

    Makes a request to tag an object with a specific tag. This creates an association
    between the tag and the specified object (chart, dashboard, etc.)

    Args:
        object_type: Type of the object ('chart', 'dashboard', etc.)
        object_id: ID of the object to tag
        tag_name: Name of the tag to apply

    Returns:
        A dictionary with the tagging confirmation
    """
    payload = {
        "object_type": object_type,
        "object_id": object_id,
        "tag_name": tag_name,
    }

    return await make_api_request(
        ctx, "post", "/api/v1/tag/tagged_objects", data=payload
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_object_remove(
    ctx: Context, object_type: str, object_id: int, tag_name: str
) -> Dict[str, Any]:
    """
    Remove a tag from an object

    Makes a request to remove a tag association from a specific object.

    Args:
        object_type: Type of the object ('chart', 'dashboard', etc.)
        object_id: ID of the object to untag
        tag_name: Name of the tag to remove

    Returns:
        A dictionary with the untagging confirmation message
    """
    response = await make_api_request(
        ctx,
        "delete",
        f"/api/v1/tag/{object_type}/{object_id}",
        params={"tag_name": tag_name},
    )

    if not response.get("error"):
        return {
            "message": f"Tag '{tag_name}' removed from {object_type} {object_id} successfully"
        }

    return response


# ===== Explore Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_form_data_create(
    ctx: Context, form_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create form data for chart exploration

    Makes a request to the /api/v1/explore/form_data POST endpoint to store
    chart configuration data temporarily.

    Args:
        form_data: Chart configuration including datasource, metrics, and visualization settings

    Returns:
        A dictionary with a key that can be used to retrieve the form data
    """
    return await make_api_request(
        ctx, "post", "/api/v1/explore/form_data", data=form_data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_form_data_get(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get form data for chart exploration

    Makes a request to the /api/v1/explore/form_data/{key} endpoint to retrieve
    previously stored chart configuration.

    Args:
        key: Key of the form data to retrieve

    Returns:
        A dictionary with the stored chart configuration
    """
    return await make_api_request(ctx, "get", f"/api/v1/explore/form_data/{key}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_permalink_create(
    ctx: Context, state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a permalink for chart exploration

    Makes a request to the /api/v1/explore/permalink POST endpoint to generate
    a shareable link to a specific chart exploration state.

    Args:
        state: State data for the permalink including form_data

    Returns:
        A dictionary with a key that can be used to access the permalink
    """
    return await make_api_request(ctx, "post", "/api/v1/explore/permalink", data=state)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_permalink_get(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get a permalink for chart exploration

    Makes a request to the /api/v1/explore/permalink/{key} endpoint to retrieve
    a previously saved exploration state.

    Args:
        key: Key of the permalink to retrieve

    Returns:
        A dictionary with the stored exploration state
    """
    return await make_api_request(ctx, "get", f"/api/v1/explore/permalink/{key}")


# ===== Menu Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_menu_get(ctx: Context) -> Dict[str, Any]:
    """
    Get the Superset menu data

    Makes a request to the /api/v1/menu/ endpoint to retrieve the navigation
    menu structure based on user permissions.

    Returns:
        A dictionary with menu items and their configurations
    """
    return await make_api_request(ctx, "get", "/api/v1/menu/")


# ===== Configuration Tools =====


@mcp.tool()
@handle_api_errors
async def superset_config_get_base_url(ctx: Context) -> Dict[str, Any]:
    """
    Get the base URL of the Superset instance

    Returns the configured Superset base URL that this MCP server is connecting to.
    This can be useful for constructing full URLs to Superset resources or for
    displaying information about the connected instance.

    This tool does not require authentication as it only returns configuration information.

    Returns:
        A dictionary with the Superset base URL
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    return {
        "base_url": superset_ctx.base_url,
        "message": f"Connected to Superset instance at: {superset_ctx.base_url}",
    }


# ===== Advanced Data Type Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_advanced_data_type_convert(
    ctx: Context, type_name: str, value: Any
) -> Dict[str, Any]:
    """
    Convert a value to an advanced data type

    Makes a request to the /api/v1/advanced_data_type/convert endpoint to transform
    a value into the specified advanced data type format.

    Args:
        type_name: Name of the advanced data type
        value: Value to convert

    Returns:
        A dictionary with the converted value
    """
    params = {
        "type_name": type_name,
        "value": value,
    }

    return await make_api_request(
        ctx, "get", "/api/v1/advanced_data_type/convert", params=params
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_advanced_data_type_list(ctx: Context) -> Dict[str, Any]:
    """
    List all advanced data types supported by this Superset instance.

    Returns:
        A dictionary with available advanced data types and their configurations
    """
    return await make_api_request(ctx, "get", "/api/v1/advanced_data_type/types")


# ===== Memory Management Tools =====

@mcp.tool()
@handle_api_errors
async def superset_memory_write(ctx: Context, memory_name: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a memory to the memory store for the current Superset instance
    
    Stores a memory for the current instance that can be retrieved later. The memory will
    be associated with this specific Superset instance based on its URL.
    
    Args:
        memory_name: Name of the memory to save (use naming conventions for organization)
        content: Dictionary containing the memory data
    
    Returns:
        Dict with success confirmation
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    # Add timestamp if not present
    if not content.get("metadata", {}).get("created"):
        if not "metadata" in content:
            content["metadata"] = {}
        content["metadata"]["created"] = int(time.time())
    
    # Always update the last_updated timestamp
    if not "metadata" in content:
        content["metadata"] = {}
    content["metadata"]["last_updated"] = int(time.time())
    
    # Save the memory
    success = save_memory(instance_name, memory_name, content)
    
    if success:
        return {"success": True, "message": f"Memory '{memory_name}' saved successfully"}
    else:
        return {"error": f"Failed to save memory '{memory_name}'"}


@mcp.tool()
@handle_api_errors
async def superset_memory_write_standardized(
    ctx: Context, 
    memory_name: str, 
    category: str,
    description: str,
    content: Dict[str, Any], 
    tags: List[str] = None,
    related_memories: List[str] = None
) -> Dict[str, Any]:
    """
    Write a standardized memory to the memory store
    
    Creates a well-structured memory entry with proper metadata.
    This simplifies memory creation with a consistent structure.
    
    Args:
        memory_name: Name of the memory (use consistent naming conventions)
        category: Category for the memory (datasets, dashboards, databases, etc.)
        description: Clear description of the memory's content and purpose
        content: The actual memory content
        tags: List of tags for better searchability
        related_memories: List of related memory names for cross-referencing
    
    Returns:
        Dict with success confirmation
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    
    # Create properly structured memory
    memory_data = {
        "metadata": {
            "name": memory_name,
            "description": description,
            "category": category,
            "tags": tags or [],
            "related_memories": related_memories or [],
            "created": int(time.time()),
            "last_updated": int(time.time())
        },
        "content": content
    }
    
    # Call the base memory write function
    return await superset_memory_write(ctx, memory_name, memory_data)


@mcp.tool()
@handle_api_errors
async def superset_memory_read(ctx: Context, memory_name: str) -> Dict[str, Any]:
    """
    Read a memory from the memory store
    
    Retrieves a specific memory by name from the instance-specific memory store.
    
    Args:
        memory_name: Name of the memory to retrieve
    
    Returns:
        Dict containing the memory data or error
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    memory = load_memory(instance_name, memory_name)
    
    if memory:
        return {"success": True, "memory": memory}
    else:
        return {"error": f"Memory '{memory_name}' not found"}


@mcp.tool()
@handle_api_errors
async def superset_memory_list(ctx: Context) -> Dict[str, Any]:
    """
    List all available memories for the current Superset instance
    
    Returns a list of all memories stored for this specific Superset instance.
    
    Returns:
        Dict containing the list of memory names
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    memories = list_memories(instance_name)
    
    return {"success": True, "memories": memories}


@mcp.tool()
@handle_api_errors
async def superset_memory_search(ctx: Context, query: str) -> Dict[str, Any]:
    """
    Search memories by query
    
    Searches through memory metadata (names, descriptions, tags) for matches.
    
    Args:
        query: Search term to look for in memory metadata
    
    Returns:
        Dict containing search results
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    results = search_memories(instance_name, query)
    
    return {
        "success": True, 
        "query": query, 
        "results_count": len(results),
        "results": results
    }


@mcp.tool()
@handle_api_errors
async def superset_memory_delete(ctx: Context, memory_name: str) -> Dict[str, Any]:
    """
    Delete a memory from the store
    
    Removes a specific memory by name.
    
    Args:
        memory_name: Name of the memory to delete
    
    Returns:
        Dict with success confirmation
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    try:
        memory_path = get_memory_path(instance_name, memory_name)
        if os.path.exists(memory_path):
            os.remove(memory_path)
            
            # Also update the index by removing this memory
            memory_dir = get_memory_dir(instance_name)
            index_path = os.path.join(memory_dir, "memory_index.json")
            
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.loads(f.read())
                
                # Remove from categories
                for category, memories in index["categories"].items():
                    if memory_name in memories:
                        index["categories"][category].remove(memory_name)
                
                # Remove from memories
                if memory_name in index["memories"]:
                    del index["memories"][memory_name]
                
                # Update timestamp
                index["last_updated"] = int(time.time())
                
                # Save updated index
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
            
            return {"success": True, "message": f"Memory '{memory_name}' deleted successfully"}
        else:
            return {"error": f"Memory '{memory_name}' not found"}
    except Exception as e:
        return {"error": f"Error deleting memory '{memory_name}': {str(e)}"}


@mcp.tool()
@handle_api_errors
async def superset_memory_auto_retrieve(ctx: Context, query: str) -> Dict[str, Any]:
    """
    Auto-retrieve relevant memories based on a query
    
    Searches through all memories to find the most relevant ones for the given query.
    This is useful for answering user questions based on previously stored knowledge.
    
    Args:
        query: The user query or topic to find relevant memories for
    
    Returns:
        Dict containing the most relevant memories
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    # Search for relevant memories
    results = search_memories(instance_name, query)
    
    if not results:
        return {
            "success": True,
            "found_relevant_memories": False,
            "message": "No relevant memories found. Consider creating new memories after addressing this query."
        }
    
    # Load full content for the top 3 most relevant memories
    top_memories = results[:3]
    full_memories = []
    
    for memory_info in top_memories:
        memory_name = memory_info["name"]
        memory_data = load_memory(instance_name, memory_name)
        if memory_data:
            full_memories.append(memory_data)
    
    return {
        "success": True,
        "found_relevant_memories": True,
        "query": query,
        "results_count": len(results),
        "top_memories": full_memories
    }


@mcp.tool()
@handle_api_errors
async def superset_instance_onboard(ctx: Context) -> Dict[str, Any]:
    """
    Onboard a new Superset instance
    
    Captures basic information about a Superset instance and stores it in memory.
    This should be run once for each new Superset instance.
    
    Returns:
        Dict with success confirmation
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    # Check if authentication is required
    if not superset_ctx.access_token:
        return {"error": "Authentication required. Please authenticate first."}
    
    try:
        # Get instance information
        user_info = await make_api_request(ctx, "get", "/api/v1/me/")
        databases = await make_api_request(ctx, "get", "/api/v1/database/")
        dashboards = await make_api_request(ctx, "get", "/api/v1/dashboard/")
        datasets = await make_api_request(ctx, "get", "/api/v1/dataset/")
        
        # Extract base info
        instance_info = {
            "base_url": superset_ctx.base_url,
            "user": user_info.get("result", {}).get("username", "unknown"),
            "database_count": len(databases.get("result", [])),
            "dashboard_count": len(dashboards.get("result", [])),
            "dataset_count": len(datasets.get("result", [])),
            "onboarding_completed": True,
            "onboarded_at": int(time.time())
        }
        
        # Save to context
        superset_ctx.instance_metadata = instance_info
        superset_ctx.onboarding_completed = True
        
        # Create memory structure
        memory_data = {
            "metadata": {
                "name": "instance_metadata",
                "description": "Basic information about the Superset instance",
                "category": "meta",
                "tags": ["instance", "metadata", "configuration"],
                "related_memories": [],
                "created": int(time.time()),
                "last_updated": int(time.time())
            },
            "content": instance_info
        }
        
        # Save memory
        success = save_memory(instance_name, "instance_metadata", memory_data)
        
        if success:
            # Also create a memory index guide for this instance
            await superset_memory_write_standardized(
                ctx,
                "meta_memory_index",
                "meta",
                "Guide to the memory system with best practices and organization",
                {
                    "memory_system_overview": {
                        "description": "The memory system allows storing and retrieving structured information for Superset analysis",
                        "use_cases": [
                            "Storing dataset analysis results for future reference",
                            "Creating reusable dashboard design patterns",
                            "Documenting SQL patterns and optimization techniques",
                            "Saving user preferences and commonly used configurations",
                            "Building a knowledge base of best practices for data visualization"
                        ],
                        "benefits": [
                            "Standardized approach to data analysis",
                            "Reusable templates for common tasks",
                            "Persistent storage across conversations",
                            "Searchable and categorized knowledge base"
                        ]
                    },
                    "memory_categories": {
                        "datasets": "Information about specific datasets including column types, metrics, and analysis approaches",
                        "dashboards": "Dashboard designs, layouts, and best practices",
                        "databases": "Database connection details and configuration",
                        "queries": "Useful SQL queries and patterns",
                        "analysis_techniques": "Analytical approaches and methodologies",
                        "best_practices": "General best practices for Superset use",
                        "templates": "Reusable templates for creating other memories",
                        "user_preferences": "User-specific settings and preferences",
                        "meta": "Information about the memory system itself",
                        "other": "Miscellaneous memories that don't fit into other categories"
                    },
                    "naming_conventions": {
                        "prefix_importance": "Always use appropriate prefixes to categorize memories",
                        "prefixes": {
                            "dataset_": "For dataset-specific information",
                            "dashboard_": "For dashboard designs and layouts",
                            "analysis_": "For analytical approaches",
                            "best_practice_": "For best practices",
                            "template_": "For reusable templates",
                            "database_": "For database information",
                            "query_": "For SQL queries and patterns",
                            "meta_": "For memory system information",
                            "user_": "For user preferences"
                        }
                    },
                    "memory_structure_guidelines": {
                        "metadata": {
                            "description": "Always include a clear description",
                            "category": "Assign appropriate category for organization",
                            "tags": "Add relevant tags for improved search",
                            "related_memories": "Link to related memories to create a network of information"
                        }
                    }
                },
                ["meta", "index", "guide", "best_practices"]
            )
            
            return {
                "success": True, 
                "message": "Superset instance onboarded successfully",
                "instance_info": instance_info
            }
        else:
            return {"error": "Failed to save instance metadata"}
    except Exception as e:
        return {"error": f"Error during instance onboarding: {str(e)}"}


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_onboard(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Onboard a specific database from the Superset instance
    
    Collects and stores detailed information about a database, including
    its schemas, tables, and structure.
    
    Args:
        database_id: ID of the database to onboard
    
    Returns:
        Dict with success confirmation
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    try:
        # Get database details
        database_info = await make_api_request(ctx, "get", f"/api/v1/database/{database_id}")
        if database_info.get("error"):
            return {"error": f"Failed to get database info: {database_info.get('error')}"}
        
        database = database_info.get("result", {})
        
        # Get schemas
        schemas_info = await make_api_request(ctx, "get", f"/api/v1/database/{database_id}/schemas/")
        schemas = schemas_info.get("result", [])
        
        # Get tables (first schema only for efficiency)
        tables = []
        if schemas:
            tables_info = await make_api_request(
                ctx, "get", f"/api/v1/database/{database_id}/tables/", 
                params={"schema": schemas[0]}
            )
            tables = tables_info.get("result", [])
        
        # Create memory structure
        memory_data = {
            "metadata": {
                "name": f"database_{database.get('database_name', str(database_id))}",
                "description": f"Details about the {database.get('database_name')} database",
                "category": "databases",
                "tags": [
                    "database", 
                    database.get("backend", "unknown"),
                    database.get("database_name", "").lower()
                ],
                "related_memories": ["instance_metadata"],
                "created": int(time.time()),
                "last_updated": int(time.time())
            },
            "content": {
                "database_id": database_id,
                "name": database.get("database_name"),
                "backend": database.get("backend"),
                "sqlalchemy_uri_placeholder": database.get("sqlalchemy_uri_placeholder"),
                "allows_subquery": database.get("allows_subquery", False),
                "allows_cost_estimate": database.get("allows_cost_estimate", False),
                "schemas": schemas,
                "sample_tables": [t.get("table_name") for t in tables[:10]],
                "schema_count": len(schemas),
                "table_count_in_sample_schema": len(tables)
            }
        }
        
        # Save memory
        success = save_memory(instance_name, f"database_{database.get('database_name', str(database_id))}", memory_data)
        
        if success:
            return {
                "success": True, 
                "message": f"Database {database.get('database_name')} onboarded successfully",
                "database_info": {
                    "id": database_id,
                    "name": database.get("database_name"),
                    "schema_count": len(schemas),
                    "table_count_in_sample_schema": len(tables)
                }
            }
        else:
            return {"error": f"Failed to save database information"}
    except Exception as e:
        return {"error": f"Error during database onboarding: {str(e)}"}


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_onboard(ctx: Context, dataset_id: int) -> Dict[str, Any]:
    """
    Onboard a specific dataset from the Superset instance
    
    Collects and stores detailed information about a dataset, including
    its columns, metrics, and structure.
    
    Args:
        dataset_id: ID of the dataset to onboard
    
    Returns:
        Dict with success confirmation
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    try:
        # Get dataset details
        dataset_info = await make_api_request(ctx, "get", f"/api/v1/dataset/{dataset_id}")
        if dataset_info.get("error"):
            return {"error": f"Failed to get dataset info: {dataset_info.get('error')}"}
        
        dataset = dataset_info.get("result", {})
        
        # Create memory structure
        memory_data = {
            "metadata": {
                "name": f"dataset_{dataset.get('table_name', str(dataset_id))}",
                "description": f"Details about the {dataset.get('table_name')} dataset",
                "category": "datasets",
                "tags": [
                    "dataset", 
                    dataset.get("database", {}).get("backend", "unknown"),
                    dataset.get("table_name", "").lower()
                ],
                "related_memories": ["instance_metadata"],
                "created": int(time.time()),
                "last_updated": int(time.time())
            },
            "content": {
                "dataset_id": dataset_id,
                "name": dataset.get("table_name"),
                "description": dataset.get("description"),
                "schema": dataset.get("schema"),
                "database_id": dataset.get("database", {}).get("id"),
                "database_name": dataset.get("database", {}).get("database_name"),
                "sql": dataset.get("sql"),
                "main_dttm_col": dataset.get("main_dttm_col"),
                "columns": dataset.get("columns", []),
                "metrics": dataset.get("metrics", []),
                "column_count": len(dataset.get("columns", [])),
                "recommended_visualizations": get_recommended_visualizations(dataset)
            }
        }
        
        # Save memory
        success = save_memory(instance_name, f"dataset_{dataset.get('table_name', str(dataset_id))}", memory_data)
        
        if success:
            return {
                "success": True, 
                "message": f"Dataset {dataset.get('table_name')} onboarded successfully",
                "dataset_info": {
                    "id": dataset_id,
                    "name": dataset.get("table_name"),
                    "column_count": len(dataset.get("columns", [])),
                    "metric_count": len(dataset.get("metrics", []))
                }
            }
        else:
            return {"error": f"Failed to save dataset information"}
    except Exception as e:
        return {"error": f"Error during dataset onboarding: {str(e)}"}


def get_recommended_visualizations(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper function to recommend visualizations based on dataset structure"""
    recommendations = []
    
    # Check for time column
    has_time = bool(dataset.get("main_dttm_col"))
    
    # Check for numeric columns
    numeric_columns = [
        col.get("column_name") 
        for col in dataset.get("columns", []) 
        if col.get("type") in ["BIGINT", "FLOAT", "INTEGER", "DECIMAL", "DOUBLE"]
    ]
    
    # Check for categorical columns
    categorical_columns = [
        col.get("column_name") 
        for col in dataset.get("columns", []) 
        if col.get("type") in ["VARCHAR", "STRING", "TEXT", "CHAR"]
    ]
    
    # Time series recommendation
    if has_time and numeric_columns:
        recommendations.append({
            "viz_type": "line",
            "name": "Time Series Analysis",
            "columns": {
                "x": dataset.get("main_dttm_col"),
                "y": numeric_columns[:2]  # First two numeric columns
            }
        })
    
    # Bar chart for categories
    if categorical_columns and numeric_columns:
        recommendations.append({
            "viz_type": "bar",
            "name": "Category Comparison",
            "columns": {
                "dimension": categorical_columns[0],
                "metric": numeric_columns[0]
            }
        })
    
    # Pie chart if we have categories
    if categorical_columns and numeric_columns:
        recommendations.append({
            "viz_type": "pie",
            "name": "Distribution Analysis",
            "columns": {
                "dimension": categorical_columns[0],
                "metric": numeric_columns[0]
            }
        })
    
    # Table for general exploration
    recommendations.append({
        "viz_type": "table",
        "name": "Data Exploration",
        "columns": {
            "columns": (categorical_columns + numeric_columns)[:5]  # First 5 columns
        }
    })
    
    return recommendations


@mcp.tool()
@handle_api_errors
async def superset_memory_prepare_chat_context(ctx: Context) -> Dict[str, Any]:
    """
    Prepare the chat context by loading essential memories
    
    Collects essential information about the Superset instance and 
    recently accessed databases/datasets to prepare for a conversation.
    This reduces the need for repeated API calls during conversation.
    
    Returns:
        Dict with the prepared context
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    # Check if onboarding is completed
    if not superset_ctx.onboarding_completed:
        return {
            "success": False,
            "message": "Instance has not been onboarded yet. Please run superset_instance_onboard first."
        }
    
    # Load essential memories
    key_memories = ["instance_metadata", "meta_memory_index"]
    
    # Add database and dataset memories if available
    all_memories = list_memories(instance_name)
    database_memories = [m for m in all_memories if m.startswith("database_")]
    dataset_memories = [m for m in all_memories if m.startswith("dataset_")]
    
    # Load the memories
    loaded_memories = {}
    
    for memory_name in key_memories + database_memories[:2] + dataset_memories[:3]:
        memory_data = load_memory(instance_name, memory_name)
        if memory_data:
            loaded_memories[memory_name] = memory_data
    
    # Set in context for immediate access
    superset_ctx.memories = loaded_memories
    
    return {
        "success": True,
        "message": "Chat context prepared successfully",
        "loaded_memories": list(loaded_memories.keys()),
        "instance_stats": {
            "total_memories": len(all_memories),
            "database_memories": len(database_memories),
            "dataset_memories": len(dataset_memories)
        }
    }


@mcp.tool()
@handle_api_errors
async def superset_check_instance_onboarding_status(ctx: Context) -> Dict[str, Any]:
    """
    Check if the current Superset instance has been onboarded
    
    Verifies if the current Superset instance has completed onboarding
    and returns its status information.
    
    Returns:
        Dict with onboarding status and instance information
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    instance_name = superset_ctx.instance_name
    
    # Check if instance metadata exists
    instance_metadata = load_memory(instance_name, "instance_metadata")
    
    if not instance_metadata:
        return {
            "onboarded": False,
            "message": "This Superset instance has not been onboarded yet. Run superset_instance_onboard to initialize.",
            "instance_name": instance_name,
            "base_url": superset_ctx.base_url
        }
    
    # Get statistics on memories
    memories = list_memories(instance_name)
    database_memories = [m for m in memories if m.startswith("database_")]
    dataset_memories = [m for m in memories if m.startswith("dataset_")]
    
    return {
        "onboarded": True,
        "instance_name": instance_name,
        "base_url": superset_ctx.base_url,
        "onboarding_data": instance_metadata.get("content", {}),
        "memory_stats": {
            "total_memories": len(memories),
            "database_memories": len(database_memories),
            "dataset_memories": len(dataset_memories)
        }
    }


if __name__ == "__main__":
    print("Starting Superset MCP server...")
    mcp.run()
