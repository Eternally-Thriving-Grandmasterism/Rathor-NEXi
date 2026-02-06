# scripts/tensorrt_inference_gesture.py – TensorRT Inference for QAT Gesture Transformer v1.0
# Loads .trt engine, dynamic batch support (1–8), valence gating, latency logging
# MIT License – Autonomicity Games Inc. 2026

import argparse
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Dict, List, Tuple

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

GESTURE_NAMES = ['none', 'pinch', 'spiral', 'figure8', 'wave']
SEQUENCE_LENGTH = 45
LANDMARK_DIM = 225
FUTURE_VALENCE_HORIZON = 10
CONFIDENCE_THRESHOLD = 0.75
MERCY_VALENCE_DROP_THRESHOLD = 0.05

class TensorRTGestureInference:
    def __init__(self, engine_path: str):
        self.logger = TRT_LOGGER
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        self.input_shape = None
        self.output_shapes = {}

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            size = trt.volume(shape) * self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append((host_mem, device_mem))
                self.input_shape = shape
            else:
                self.outputs.append((host_mem, device_mem))
                self.output_shapes[binding] = shape

        print(f"[TensorRT] Engine loaded: {engine_path}")
        print(f"[TensorRT] Input shape: {self.input_shape}")
        print(f"[TensorRT] Output shapes: {self.output_shapes}")

    def load_engine(self, engine_path: str) -> trt.ICudaEngine:
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        input_data: (batch, 45, 225) float32 numpy array
        Returns: {'gesture_logits': ..., 'future_valence': ...}
        """
        batch_size = input_data.shape[0]
        if batch_size < 1 or batch_size > 8:
            raise ValueError(f"Batch size must be 1–8, got {batch_size}")

        # Set dynamic shape
        self.context.set_input_shape("input", input_data.shape)

        # Copy input to device
        np.copyto(self.inputs[0][0], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)

        # Execute inference
        start_time = time.perf_counter()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()
        inference_time_ms = (time.time() - start_time) * 1000

        # Copy outputs
        outputs = {}
        for i, (host_mem, _) in enumerate(self.outputs):
            binding_name = self.engine[i] if isinstance(self.engine[i], str) else self.engine.get_binding_name(i)
            out_shape = self.context.get_binding_shape(i)
            out = np.empty(out_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(out, host_mem, self.stream)
            outputs[binding_name] = out

        self.stream.synchronize()

        print(f"[TensorRT] Inference time: {inference_time_ms:.2f} ms (batch={batch_size})")

        return outputs

    def postprocess(self, outputs: Dict[str, np.ndarray], batch_size: int) -> List[dict]:
        gesture_logits = outputs['gesture_logits']  # (B, num_classes)
        future_valence = outputs['future_valence']  # (B, horizon)

        results = []
        for i in range(batch_size):
            logits = gesture_logits[i]
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            confidence = np.max(probs)
            gesture_idx = np.argmax(probs)
            gesture = GESTURE_NAMES[gesture_idx] if confidence > CONFIDENCE_THRESHOLD else 'none'

            traj = future_valence[i]
            projected_valence = np.mean(traj)
            current_valence = currentValence.get()
            is_safe = projected_valence >= current_valence - MERCY_VALENCE_DROP_THRESHOLD

            results.append({
                'gesture': gesture,
                'confidence': confidence,
                'future_valence': traj.tolist(),
                'projected_valence': projected_valence,
                'is_safe': is_safe
            })

            if not is_safe:
                mercyHaptic.playPattern('warningPulse', current_valence * 0.7)

        return results


def main():
    parser = argparse.ArgumentParser(description="TensorRT Inference for QAT Gesture Transformer")
    parser.add_argument("--engine", type=str, required=True, help="Path to .trt engine")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size (1–8)")
    parser.add_argument("--input-shape", type=str, default="4,45,225", help="Input shape B,T,L")
    args = parser.parse_args()

    batch_size = args.batch_size
    input_shape = tuple(int(x) for x in args.input_shape.split(','))
    assert input_shape[0] == batch_size, "Batch size mismatch"

    print(f"[Main] Loading engine: {args.engine}")
    engine = TensorRTGestureInference(args.engine)

    # Dummy input (replace with real landmarks)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    print(f"[Main] Running inference (batch={batch_size})...")
    outputs = engine.infer(input_data)
    results = engine.postprocess(outputs, batch_size)

    for i, res in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Gesture: {res['gesture']} (conf: {res['confidence']:.3f})")
        print(f"  Projected valence: {res['projected_valence']:.3f}")
        print(f"  Safe trajectory: {'Yes' if res['is_safe'] else 'No'}")
        print(f"  Future valence: {[f'{v:.3f}' for v in res['future_valence']]}")

if __name__ == "__main__":
    main()
