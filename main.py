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
from dataclasses import dataclass
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
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

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
        logger.warning(f"Warning: Could not save access token: {e}")


@asynccontextmanager
async def superset_lifespan(server: FastMCP) -> AsyncIterator[SupersetContext]:
    """Manage application lifecycle for Superset integration"""
    logger.info("Initializing Superset context...")

    # Create HTTP client
    client = httpx.AsyncClient(base_url=SUPERSET_BASE_URL, timeout=30.0)

    # Create context
    ctx = SupersetContext(client=client, base_url=SUPERSET_BASE_URL, app=app)

    # Try to load existing token
    stored_token = load_stored_token()
    if stored_token:
        ctx.access_token = stored_token
        # Set the token in the client headers
        client.headers.update({"Authorization": f"Bearer {stored_token}"})
        logger.info("Using stored access token")

        # Verify token validity
        try:
            response = await client.get("/api/v1/me/")
            if response.status_code != 200:
                logger.info(
                    f"Stored token is invalid (status {response.status_code}). Will need to re-authenticate."
                )
                ctx.access_token = None
                client.headers.pop("Authorization", None)
        except Exception as e:
            logger.info(f"Error verifying stored token: {e}")
            ctx.access_token = None
            client.headers.pop("Authorization", None)

    try:
        yield ctx
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Superset context...")
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
    logger.info("Received 401 Unauthorized. Attempting to refresh token...")
    refresh_result = await superset_auth_refresh_token(ctx)

    if refresh_result.get("error"):
        # If refresh failed, try to re-authenticate
        logger.info(
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
            logger.info(
                f"Failed to get CSRF token: {response.status_code} - {response.text}"
            )
            return None
    except Exception as e:
        logger.info(f"Error getting CSRF token: {str(e)}")
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
        "params": json.dumps(params),
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
    Get list of available advanced data types

    Makes a request to the /api/v1/advanced_data_type/types endpoint to retrieve
    all advanced data types supported by this Superset instance.

    Returns:
        A dictionary with available advanced data types and their configurations
    """
    return await make_api_request(ctx, "get", "/api/v1/advanced_data_type/types")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_create_from_sql(
    ctx: Context,
    table_name: str,
    database_id: int,
    sql: str,
    schema: str = None,
    normalize_columns: bool = False,
    always_filter_main_dttm: bool = False,
    owners: List[int] = None,
) -> Dict[str, Any]:
    """
    Create a new virtual dataset in Superset based on SQL query

    Makes a request to the /api/v1/dataset/ POST endpoint to create a new virtual dataset
    from a SQL query rather than an existing table.

    Args:
        table_name: Display name for the virtual dataset
        database_id: ID of the database to query
        sql: SQL query that defines the dataset
        schema: Optional database schema (for table name qualification)
        normalize_columns: Whether to normalize columns (default: False)
        always_filter_main_dttm: Whether to always filter on the main datetime column (default: False)
        owners: Optional list of user IDs who should own this dataset

    Returns:
        A dictionary with the created dataset information including its ID
    """
    payload = {
        "table_name": table_name,
        "database": database_id,
        "sql": sql,
        "normalize_columns": normalize_columns,
        "always_filter_main_dttm": always_filter_main_dttm,
    }

    if schema:
        payload["schema"] = schema

    if owners:
        payload["owners"] = owners

    return await make_api_request(ctx, "post", "/api/v1/dataset/", data=payload)


if __name__ == "__main__":
    logger.info("Starting Superset MCP server...")
    mcp.run()
