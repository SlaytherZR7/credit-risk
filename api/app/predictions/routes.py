from fastapi import APIRouter, Depends, HTTPException, Request
from app.auth.dependencies import get_current_user
from app.users.models import User
import httpx

router = APIRouter(prefix="/predictions", tags=["Predictions"])

MODEL_SERVICE_URL = "http://ml_model:8002"

@router.post("/predict-one")
async def predict_one(
    request: Request,
    _: User = Depends(get_current_user),
):
    payload = await request.json()

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{MODEL_SERVICE_URL}/predict", json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model service not reachable: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

    return response.json()
@router.post("/predict-batch")
async def predict_batch(
    request: Request,
    _: User = Depends(get_current_user),
):
    payload = await request.json()

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{MODEL_SERVICE_URL}/predict-batch", json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model service not reachable: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

    return response.json()

@router.get("/result/{job_id}")
async def get_result(
    job_id: str,
    _: User = Depends(get_current_user),
):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{MODEL_SERVICE_URL}/result/{job_id}")
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model service not reachable: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

    return response.json()
