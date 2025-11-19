from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from rq import Queue
from rq.job import Job
from app.model.pipeline import init_model
from app.utils.schema import PredictionRequest, BatchPredictionRequest

app = FastAPI(title="ML Model Service", version="1.0")

ALLOWED_ORIGINS = ["http://api:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST, GET"],
    allow_headers=["*"],
)

redis_conn = Redis(host="redis", port=6379)
queue = Queue("model_queue", connection=redis_conn)
model = None

@app.on_event("startup")
def startup_event():
    global model
    model = init_model()
    print("Model loaded successfully.")


@app.post("/predict")
def enqueue_prediction(request: PredictionRequest):
    try:
        job = queue.enqueue("app.worker.predict_one_task", request.features)
        return {"job_id": job.get_id(), "status": job.get_status()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch")
def enqueue_batch_prediction(request: BatchPredictionRequest):
    try:
        data = [item.dict() for item in request.features]
        job = queue.enqueue("app.worker.predict_batch_task", data)
        return {"job_id": job.get_id(), "status": job.get_status()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/result/{job_id}")
def get_result(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        return {
            "status": job.get_status(),
            "result": job.result,
        }
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

