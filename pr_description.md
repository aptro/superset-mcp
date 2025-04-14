# Add Dataset Update and Delete API Tools

This PR adds two new API tools to the Superset MCP integration:

1. `superset_dataset_update` - Allows updating existing dataset properties through the `/api/v1/dataset/{id}` PUT endpoint
2. `superset_dataset_delete` - Allows deleting datasets through the `/api/v1/dataset/{id}` DELETE endpoint

These additions complete the CRUD operations for datasets, which previously only had list, get, and create functionality.

## Changes

- Added the `superset_dataset_update` tool with appropriate documentation
- Added the `superset_dataset_delete` tool with appropriate documentation
- Both tools follow the existing patterns in the codebase with proper auth and error handling

## Testing

These tools have been tested with Claude Desktop and work correctly with the Superset API.

## Related Issues

This addresses the missing dataset update functionality in the MCP server. 