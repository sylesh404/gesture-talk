"""
Inference Speed Benchmark for GestureTalk Competition Edition.
Measures latency (ms/prediction) for Keras predict vs Fast Compiled Tensor graph.
"""

import time
import os
import numpy as np
import tensorflow as tf
from config import MODEL_PATH_KERAS, SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES

def run_benchmark():
    model_path = MODEL_PATH_KERAS if os.path.exists(MODEL_PATH_KERAS) else "gesture_model.keras"
    if not os.path.exists(model_path):
        print("Error: Model file not found. Train gesture_model.keras first.")
        return

    print(f"Loading competition model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    dummy_input = np.random.randn(1, SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES).astype(np.float32)

    # Benchmark 1: Original Keras model.predict()
    t0 = time.time()
    for _ in range(50):
        _ = model.predict(dummy_input, verbose=0)
    t1 = time.time()
    keras_predict_time_ms = ((t1 - t0) / 50.0) * 1000

    # Benchmark 2: Fast @tf.function compiled tensor call
    @tf.function(reduce_retracing=True)
    def fast_predict_fn(x):
        return model(x, training=False)

    tensor_dummy = tf.constant(dummy_input)
    _ = fast_predict_fn(tensor_dummy) # Warmup

    t0 = time.time()
    for _ in range(50):
        _ = fast_predict_fn(tensor_dummy).numpy()
    t1 = time.time()
    fast_predict_time_ms = ((t1 - t0) / 50.0) * 1000

    speedup = keras_predict_time_ms / fast_predict_time_ms if fast_predict_time_ms > 0 else 1.0

    print("\n" + "=" * 50)
    print("GESTURETALK COMPETITION - INFERENCE SPEED BENCHMARK")
    print("=" * 50)
    print(f"Standard Keras predict() latency: {keras_predict_time_ms:.2f} ms / prediction")
    print(f"Fast Compiled Tensor latency:      {fast_predict_time_ms:.2f} ms / prediction")
    print(f"Speedup Factor:                    {speedup:.1f}x faster!")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_benchmark()
