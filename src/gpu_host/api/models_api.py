"""Model artifacts API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path

from gpu_core.errors import ModelNotFoundError
from gpu_core.models import ModelArtifact
from gpu_core.schemas import ModelArtifactResponse, ModelListResponse
from gpu_host.api.deps import verify_auth, get_session

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    framework: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> ModelListResponse:
    """List all model artifacts."""
    query = db.query(ModelArtifact)
    if framework:
        query = query.filter(ModelArtifact.framework == framework)

    total = query.count()
    models = query.order_by(ModelArtifact.created_at.desc()).offset(offset).limit(limit).all()

    return ModelListResponse(
        models=[ModelArtifactResponse.model_validate(m) for m in models],
        total=total,
    )


@router.get("/models/{model_id}", response_model=ModelArtifactResponse)
async def get_model(
    model_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> ModelArtifactResponse:
    """Get model artifact details."""
    model = db.query(ModelArtifact).filter(ModelArtifact.id == model_id).first()
    if not model:
        raise ModelNotFoundError(model_id)
    return ModelArtifactResponse.model_validate(model)


@router.get("/models/{model_id}/download")
async def download_model(
    model_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
):
    """Download a model artifact file."""
    model = db.query(ModelArtifact).filter(ModelArtifact.id == model_id).first()
    if not model:
        raise ModelNotFoundError(model_id)

    file_path = Path(model.path)
    if not file_path.exists():
        raise ModelNotFoundError(model_id)

    return FileResponse(
        path=str(file_path),
        filename=model.filename,
        media_type="application/octet-stream",
    )
