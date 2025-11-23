# force-rebuild-01
from redis import Redis
from rq import Worker, Queue
import traceback
import json

from app.model.pipeline import predict_single, predict_batch, init_model

# Conexión a Redis
redis_conn = Redis(host="redis", port=6379)

# Cola donde el API encola las tareas
queue = Queue("model_queue", connection=redis_conn)

# Cargar modelo y preprocessor al iniciar el worker
init_model()

def predict_one_task(features: dict):
    try:
        print("🔍 WORKER DEBUG: predict_one_task INPUT:")
        print(json.dumps(features, indent=2))
        return predict_single(features)
    except Exception as e:
        print("❌ ERROR EN predict_one_task:")
        print(traceback.format_exc())
        raise e

def predict_batch_task(batch: list):
    try:
        print("🔍 WORKER DEBUG: predict_batch_task INPUT (primer elemento):")
        if len(batch) > 0:
            print(json.dumps(batch[0], indent=2))

        print("🔍 WORKER DEBUG: tamaño del batch:", len(batch))

        result = predict_batch(batch)

        print("✅ WORKER DEBUG: predict_batch OUTPUT:")
        print(json.dumps(result, indent=2))

        return result

    except Exception as e:
        print("❌ ERROR EN predict_batch_task:")
        print(traceback.format_exc())
        raise e


if __name__ == "__main__":
    print("Model worker started, waiting for tasks...")
    worker = Worker(
        queues=[queue],
        connection=redis_conn
    )
    worker.work()
