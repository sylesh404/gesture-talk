"""
Real-Time Inference Speed & Latency Benchmark for GestureTalk Competition Edition.
Measures average latency, min/max latency, FPS, input shape, and file sizes.
"""

import time
import os
import numpy as np
import tensorflow as tf
from config import MODEL_PATH_KERAS, MODEL_PATH_TFLITE, SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES

def run_benchmark():
    keras_path = MODEL_PATH_KERAS if os.path.exists(MODEL_PATH_KERAS) else "gesture_model.keras"
    tflite_path = MODEL_PATH_TFLITE if os.path.exists(MODEL_PATH_TFLITE) else "gesture_model.tflite"

    if not os.path.exists(keras_path):
        print("Error: Model file gesture_model.keras not found.")
        return

    keras_size_mb = os.path.getsize(keras_path) / (1024 * 1024)
    tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024) if os.path.exists(tflite_path) else 0.0

    print(f"Loading competition model from '{keras_path}'...")
    model = tf.keras.models.load_model(keras_path)

    # Compiled fast inference function
    @tf.function(reduce_retracing=True)
    def fast_predict_fn(x):
        return model(x, training=False)

    dummy_input = np.random.randn(1, SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES).astype(np.float32)
    tensor_input = tf.constant(dummy_input)

    # Warmup graph execution
    for _ in range(20):
        _ = fast_predict_fn(tensor_input).numpy()

    # Benchmark over 1,000 iterations
    latencies_ms = []
    num_runs = 1000

    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = fast_predict_fn(tensor_input).numpy()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies_ms))
    min_latency = float(np.min(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("\n" + "=" * 40)
    print("REAL-TIME PERFORMANCE")
    print("=" * 40)
    print(f"Average inference latency: {avg_latency:.2f} ms")
    print(f"Minimum inference latency: {min_latency:.2f} ms")
    print(f"Maximum inference latency: {max_latency:.2f} ms")
    print(f"Approx inference FPS:      {fps:.1f} FPS")
    print(f"Model input shape:         (1, {SEQUENCE_LENGTH}, {NUM_LANDMARK_COORDINATES})")
    print(f"Keras model size:          {keras_size_mb:.2f} MB")
    if tflite_size_mb > 0:
        print(f"TFLite model size:         {tflite_size_mb:.2f} MB")
    print("=" * 40 + "\n")

    return {
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "fps": fps,
        "input_shape": f"(1, {SEQUENCE_LENGTH}, {NUM_LANDMARK_COORDINATES})",
        "keras_size_mb": keras_size_mb,
        "tflite_size_mb": tflite_size_mb
    }

if __name__ == "__main__":
    run_benchmark()
