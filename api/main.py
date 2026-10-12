
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import score

app = FastAPI(
    title="Online Fraud Detection API",
    description="Predicts whether a financial transaction may be fraudulent.",
    version="1.0.0",
)


class Transaction(BaseModel):
    step: int = Field(ge=0)
    type: str
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


@app.on_event("startup")
def load_model():
    score.init()


@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict")
def predict(transaction: Transaction):
    result = score.run(transaction.model_dump())

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
