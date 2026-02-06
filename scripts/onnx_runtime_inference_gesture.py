# scripts/onnx_runtime_inference_gesture.py – ONNX Runtime Inference v1.0
# Multi-backend (WebNN/WebGPU/WebGL/CPU), dynamic batch, valence gating, latency metrics
# MIT License – Autonomicity Games Inc. 2026

import argparse
import time
import numpy as np
import onnxruntime as ort
from typing import Dict, List

GESTURE_NAMES = ['none', 'pinch', 'spiral', 'figure8', 'wave']
SEQUENCE_LENGTH = 45
LANDMARK_DIM = 225
FUTURE_VALENCE_HORIZON = 10
CONFIDENCE_THRESHOLD = 0.75
MERCY_VALENCE_DROP_THRESHOLD = 0.05

MODEL_PATHS = {
    'high': 'models/gesture_transformer_qat_int4.onnx',    # INT4 – fastest, high valence
    'medium': 'models/gesture_transformer_qat_int8.onnx', # INT8 – balanced
    'low': 'models/gesture_transformer_fp16.onnx',        # FP16 – fallback
}

class ONNXGestureInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = self.load_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"[ONNXRuntime] Loaded model: {model_path}")
        print(f"[ONNXRuntime] Input: {self.session.get_inputs()[0].shape}")
        print(f"[ONNXRuntime] Outputs: {self.output_names}")

    def load_session(self) -> ort.InferenceSession:
        providers = ['WebNNExecutionProvider', 'WebGPUExecutionProvider', 'WebGLExecutionProvider', 'CPUExecutionProvider']
        available_providers = [p for p in providers if p in ort.get_available_providers()]

        if not available_providers:
            raise RuntimeError("No ONNX Runtime execution providers available")

        print(f"[ONNXRuntime] Available providers: {available_providers}")
        print(f"[ONNXRuntime] Using: {available_providers[0]}")

        return ort.InferenceSession(
            self.model_path,
            providers=available_providers,
            provider_options=[{} for _ in available_providers]
        )

    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        input_data: (batch, 45, 225) float32 numpy array
        Returns: {'gesture_logits': ..., 'future_valence': ...}
        """
        batch_size = input_data.shape[0]
        if batch_size < 1 or batch_size > 8:
            raise ValueError(f"Batch size must be 1–8, got {batch_size}")

        input_tensor = ort.OrtValue.ortvalue_from_numpy(input_data)

        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        inference_time_ms = (time.time() - start_time) * 1000

        results = {}
        for name, out in zip(self.output_names, outputs):
            results[name] = out

        print(f"[ONNXRuntime] Inference time: {inference_time_ms:.2f} ms (batch={batch_size})")
        return results

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
    parser = argparse.ArgumentParser(description="ONNX Runtime Inference for Gesture Transformer")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model (qat_int4/int8/fp16)")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size (1–8)")
    parser.add_argument("--input-shape", type=str, default="4,45,225", help="Input shape B,T,L")
    args = parser.parse_args()

    batch_size = args.batch_size
    input_shape = tuple(int(x) for x in args.input_shape.split(','))
    assert input_shape[0] == batch_size, "Batch size mismatch"

    print(f"[Main] Loading model: {args.model}")
    engine = ONNXGestureInference(args.model)

    # Dummy input (replace with real landmarks)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    print(f"[Main] Running inference (batch={batch_size})...")
    outputs = engine.infer(input_data)
    results = engine.postprocess(outputs, batch_size)

    for i, res in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Gesture: {res['gesture']} (conf: {res['confidence']*100:.1f}%)")
        print(f"  Projected valence: {res['projected_valence']:.3f}")
        print(f"  Safe trajectory: {'Yes' if res['is_safe'] else 'No'}")
        print(f"  Future valence: {[f'{v:.3f}' for v in res['future_valence']]}")

if __name__ == "__main__":
    main()
