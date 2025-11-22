from redis import Redis
from rq import Worker, Queue

from app.model.pipeline import predict_single, predict_batch, init_model

redis_conn = Redis(host="redis", port=6379)
queue = Queue("model_queue", connection=redis_conn)

# Cargar modelo + preprocessor una vez
init_model()
# Cargar modelo + preprocessor una vez
init_model()

def predict_one_task(features: dict):
    return predict_single(features)
    return predict_single(features)

def predict_batch_task(batch: list):
    print("\n📌 WORKER RECEIVED BATCH:")
    for i, item in enumerate(batch):
        print(f" - item {i}: {type(item)} → {item}")
    return predict_batch(batch)

if __name__ == "__main__":
    print("Model worker started, waiting for tasks...")
    worker = Worker(queues=[queue], connection=redis_conn)
    worker.work()
