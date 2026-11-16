import pytest

from agentic_rag.config import Settings
from agentic_rag.security.auth import authenticate_request


def test_authentication_accepts_configured_api_key_and_workspace() -> None:
    identity = authenticate_request(
        {"x-api-key": "secret", "x-workspace-id": "team-a", "x-user-id": "user_1"},
        Settings(api_key="secret", workspace_id="team-a", require_workspace_header=True),
    )

    assert identity.user_id == "user_1"
    assert identity.workspace_id == "team-a"


def test_authentication_rejects_wrong_workspace() -> None:
    with pytest.raises(PermissionError, match="Workspace access denied"):
        authenticate_request(
            {"x-api-key": "secret", "x-workspace-id": "team-b"},
            Settings(api_key="secret", workspace_id="team-a"),
        )


def test_authentication_requires_workspace_header_when_configured() -> None:
    with pytest.raises(PermissionError, match="Workspace header is required"):
        authenticate_request(
            {"x-api-key": "secret"},
            Settings(api_key="secret", workspace_id="team-a", require_workspace_header=True),
        )


def test_authentication_rejects_malformed_workspace_id() -> None:
    with pytest.raises(PermissionError, match="Invalid user or workspace ID"):
        authenticate_request(
            {"x-api-key": "secret", "x-workspace-id": "team/a", "x-user-id": "user_1"},
            Settings(api_key="secret", workspace_id="team/a"),
        )
