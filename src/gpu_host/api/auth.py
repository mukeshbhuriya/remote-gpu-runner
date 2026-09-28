"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from gpu_core.config import get_config
from gpu_core.schemas import TokenResponse, TokenRotateResponse
from gpu_core.security import generate_token
from gpu_host.api.deps import verify_auth

router = APIRouter()


@router.post("/token", response_model=TokenResponse)
async def get_token(_token: str = Depends(verify_auth)) -> TokenResponse:
    """Verify token validity and return it. Primarily for client handshake."""
    return TokenResponse(access_token=_token)


@router.post("/rotate", response_model=TokenRotateResponse)
async def rotate_token(_token: str = Depends(verify_auth)) -> TokenRotateResponse:
    """
    Rotate the API token.

    The old token becomes invalid after rotation.
    """
    config = get_config()
    new_token = generate_token()
    config.security.api_token = new_token
    config.save_generated_secrets()

    return TokenRotateResponse(
        new_token=new_token,
        message="Token rotated. Use the new token for future requests.",
    )
