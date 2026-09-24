import hmac

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from app.container import Container
from app.errors import AppError

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_current_user(
    api_key: str | None = Depends(_api_key_header),
    container: Container = Depends(get_container),
) -> str:
    """Resolve the caller from ``X-API-Key``. In production this would validate an SSO
    (OIDC/JWT) token from the company IdP; API keys keep the assignment self-contained."""
    keys = container.settings.api_key_map
    if not keys:
        return "anonymous"  # auth disabled for local development
    if api_key:
        for key, user in keys.items():
            if hmac.compare_digest(api_key, key):
                return user
    raise UnauthorizedError("Missing or invalid API key (X-API-Key header)")
