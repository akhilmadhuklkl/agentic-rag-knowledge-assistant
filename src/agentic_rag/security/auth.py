"""Request authentication and workspace isolation checks."""

import re
import secrets
from dataclasses import dataclass

from agentic_rag.config import Settings


@dataclass(frozen=True)
class RequestIdentity:
    """Authenticated request metadata."""

    user_id: str
    workspace_id: str


def authenticate_request(headers: dict[str, str], settings: Settings) -> RequestIdentity:
    """Validate API key and workspace headers against configured boundaries."""

    provided_api_key = headers.get("x-api-key", "")
    if settings.api_key and not secrets.compare_digest(provided_api_key, settings.api_key):
        raise PermissionError("Invalid API key")
    user_id = headers.get("x-user-id", "anonymous")
    workspace_id = headers.get("x-workspace-id", settings.workspace_id)
    if settings.require_workspace_header and "x-workspace-id" not in headers:
        raise PermissionError("Workspace header is required")
    if workspace_id != settings.workspace_id:
        raise PermissionError("Workspace access denied")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", workspace_id):
        raise PermissionError("Invalid user or workspace ID")
    return RequestIdentity(user_id=user_id, workspace_id=workspace_id)
