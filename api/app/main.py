from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.auth.routes import router as auth_router
from app.predictions.routes import router as predictions_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="API to predict credit risk using machine learning models.",
)

# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(predictions_router, prefix=settings.API_V1_STR)

@app.get(f"{settings.API_V1_STR}/")
def root():
    return {
        "message": "Credit Prediction API",
        "version": "1.0.0",
    }

@app.get(f"{settings.API_V1_STR}/health")
def health_check():
    return {"status": "healthy"}