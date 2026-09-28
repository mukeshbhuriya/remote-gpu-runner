"""Experiments API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from gpu_core.models import Experiment
from gpu_core.schemas import ExperimentListResponse, ExperimentResponse
from gpu_host.api.deps import verify_auth, get_session

router = APIRouter()


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> ExperimentListResponse:
    """List all experiments."""
    total = db.query(Experiment).count()
    experiments = (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    exp_responses = []
    for exp in experiments:
        resp = ExperimentResponse.model_validate(exp)
        resp.job_count = len(exp.jobs) if exp.jobs else 0
        exp_responses.append(resp)

    return ExperimentListResponse(
        experiments=exp_responses,
        total=total,
    )
