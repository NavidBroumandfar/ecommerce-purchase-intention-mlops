from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict_purchase

PROJECT_NAME = "ecommerce-purchase-intention-mlops"

app = FastAPI(
    title="E-commerce Purchase Intention API",
    version="1.0.0",
    description="Local-first FastAPI service for purchase intention predictions.",
)


class ShopperSession(BaseModel):
    Administrative: int = Field(..., ge=0)
    Administrative_Duration: float = Field(..., ge=0)
    Informational: int = Field(..., ge=0)
    Informational_Duration: float = Field(..., ge=0)
    ProductRelated: int = Field(..., ge=0)
    ProductRelated_Duration: float = Field(..., ge=0)
    BounceRates: float = Field(..., ge=0)
    ExitRates: float = Field(..., ge=0)
    PageValues: float = Field(..., ge=0)
    SpecialDay: float = Field(..., ge=0)
    Month: str
    OperatingSystems: int = Field(..., ge=0)
    Browser: int = Field(..., ge=0)
    Region: int = Field(..., ge=0)
    TrafficType: int = Field(..., ge=0)
    VisitorType: str
    Weekend: bool


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "project": PROJECT_NAME,
    }


@app.post("/predict")
def predict(session: ShopperSession) -> dict[str, Any]:
    try:
        return predict_purchase(_model_to_dict(session))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
