"""Datasets API endpoints."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from gpu_core.config import get_config
from gpu_core.errors import ConflictError, DatasetNotFoundError
from gpu_core.logging_config import get_logger
from gpu_core.models import Dataset
from gpu_core.schemas import DatasetListResponse, DatasetResponse
from gpu_core.security import sanitize_filename
from gpu_host.api.deps import verify_auth, get_session

logger = get_logger("host.api.datasets")

router = APIRouter()


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> DatasetListResponse:
    """List all registered datasets."""
    total = db.query(Dataset).count()
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).offset(offset).limit(limit).all()

    return DatasetListResponse(
        datasets=[DatasetResponse.model_validate(ds) for ds in datasets],
        total=total,
    )


@router.post("/datasets", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> DatasetResponse:
    """Upload a dataset (directory as zip/tar.gz, or single file)."""
    config = get_config()

    # Check uniqueness
    existing = db.query(Dataset).filter(Dataset.name == name).first()
    if existing:
        raise ConflictError(f"Dataset '{name}' already exists. Use a different name or delete the existing one.")

    # Create storage dir
    dataset_dir = Path(config.storage.datasets_dir) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save file with chunked streaming
    safe_name = sanitize_filename(file.filename or "dataset")
    file_path = dataset_dir / safe_name
    total_size = 0
    sha256 = hashlib.sha256()

    with open(file_path, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
            sha256.update(chunk)
            total_size += len(chunk)

    # Determine format
    fmt = "file"
    if safe_name.endswith(".zip"):
        fmt = "zip"
    elif safe_name.endswith((".tar.gz", ".tgz")):
        fmt = "tar.gz"
    elif safe_name.endswith(".tar"):
        fmt = "tar"

    # Count files if it's a directory-like archive
    file_count = 1

    # Create record
    ds = Dataset(
        name=name,
        description=description or None,
        path=str(file_path),
        size_bytes=total_size,
        checksum=sha256.hexdigest(),
        file_count=file_count,
        format=fmt,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    logger.info(f"Dataset '{name}' uploaded ({total_size} bytes)", extra={"event": "dataset_upload"})
    return DatasetResponse.model_validate(ds)


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> DatasetResponse:
    """Get dataset details by ID."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise DatasetNotFoundError(dataset_id)
    return DatasetResponse.model_validate(ds)


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> None:
    """Delete a dataset."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise DatasetNotFoundError(dataset_id)

    # Remove files
    ds_path = Path(ds.path)
    if ds_path.exists():
        if ds_path.is_dir():
            import shutil
            shutil.rmtree(ds_path)
        else:
            # Remove file and parent dir if empty
            ds_path.unlink()
            parent = ds_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

    db.delete(ds)
    db.commit()

    logger.info(f"Dataset '{ds.name}' deleted.", extra={"event": "dataset_delete"})
