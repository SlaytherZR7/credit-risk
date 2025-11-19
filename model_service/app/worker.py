from redis import Redis
from rq import Worker, Queue

from app.model.preprocess import EXPECTED_COLS
from app.model.pipeline import init_model, predict_single, predict_batch

redis_conn = Redis(host="redis", port=6379)
queue = Queue("model_queue", connection=redis_conn)

model = init_model()

def predict_one_task(features: dict):
    return predict_single(features, EXPECTED_COLS)

def predict_batch_task(batch: list):
    return predict_batch(batch, EXPECTED_COLS)

if __name__ == "__main__":
    print("Model worker started, waiting for tasks...")
    worker = Worker(queues=[queue], connection=redis_conn)
    worker.work()
