
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from api.models.models import DrugBatch, DiseaseBatch, RepurposingResult
from api.security.auth import get_current_active_user

router = APIRouter(prefix="/batch", tags=["batch processing"])

@router.post("/analyze", response_model=List[RepurposingResult])
async def batch_analyze_candidates(
    candidates: List[DrugBatch],
    current_user = Depends(get_current_active_user)
):
    """Analyze multiple drug-disease pairs in a single request"""
    results = []
    for candidate in candidates:
        analysis = analyze_repurposing_candidate(
            candidate.drug,
            candidate.disease
        )
        results.append(analysis)
    return results
