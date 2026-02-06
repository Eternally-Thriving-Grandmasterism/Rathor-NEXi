# src/ml/gesture_transformer/model_test.py – Unit tests for GestureTransformer v1.0
# Shape checks, forward pass, gradient flow, ONNX export validation
# Runs with: bazelisk test //src/ml/gesture_transformer:model_test
# MIT License – Autonomicity Games Inc. 2026

import unittest
import torch
import torch.nn as nn
from gesture_model import GestureTransformer
import onnxruntime as ort
import numpy as np

class TestGestureTransformer(unittest.TestCase):
    def setUp(self):
        self.seq_len = 45
        self.landmark_dim = 225
        self.num_classes = 5
        self.future_horizon = 10
        self.batch_size = 8

        self.model = GestureTransformer(
            seq_len=self.seq_len,
            landmark_dim=self.landmark_dim,
            num_gesture_classes=self.num_classes,
            future_valence_horizon=self.future_horizon
        )
        self.model.eval()  # inference mode for tests

        # Dummy input
        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.landmark_dim)

    def test_forward_pass_shapes(self):
        """Forward pass should return correct output shapes"""
        gesture_logits, future_valence = self.model(self.dummy_input)

        self.assertEqual(
            gesture_logits.shape,
            (self.batch_size, self.num_classes),
            "Gesture logits shape mismatch"
        )
        self.assertEqual(
            future_valence.shape,
            (self.batch_size, self.future_horizon),
            "Future valence shape mismatch"
        )
        self.assertTrue(
            torch.all(future_valence >= 0) and torch.all(future_valence <= 1),
            "Future valence not in [0,1] after sigmoid"
        )

    def test_gradient_flow(self):
        """Model should allow gradients to flow through both heads"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        gesture_logits, future_valence = self.model(self.dummy_input)
        loss_g = gesture_logits.mean()
        loss_v = future_valence.mean()
        loss = loss_g + loss_v

        optimizer.zero_grad()
        loss.backward()

        # Check that gradients exist on key parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Gradient missing on parameter: {name}"
                )
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"NaN in gradient on parameter: {name}"
                )

    def test_onnx_export_and_roundtrip(self):
        """Export to ONNX and verify inference consistency"""
        dummy_input = torch.randn(1, self.seq_len, self.landmark_dim)
        output_path = "gesture_transformer_test.onnx"

        self.model.export_to_onnx(dummy_input, output_path)

        # Load ONNX model
        ort_session = ort.InferenceSession(output_path)

        # Run PyTorch inference
        with torch.no_grad():
            pt_gesture, pt_valence = self.model(dummy_input)

        # Run ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        ort_gesture = torch.from_numpy(ort_outputs[0])
        ort_valence = torch.from_numpy(ort_outputs[1])

        # Compare outputs (allow small numerical difference due to float32 ops)
        torch.testing.assert_close(pt_gesture, ort_gesture, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(pt_valence, ort_valence, rtol=1e-4, atol=1e-4)

    def test_valence_output_range(self):
        """Future valence should always be in [0,1]"""
        _, future_valence = self.model(self.dummy_input)
        self.assertTrue(
            torch.all(future_valence >= 0) and torch.all(future_valence <= 1),
            "Future valence out of [0,1] bounds"
        )

    def test_model_device_transfer(self):
        """Model should correctly move to GPU and back"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        model_cuda = self.model.to(device)
        input_cuda = self.dummy_input.to(device)

        gesture_logits, future_valence = model_cuda(input_cuda)

        self.assertEqual(gesture_logits.device.type, "cuda")
        self.assertEqual(future_valence.device.type, "cuda")

        # Move back to CPU
        model_cpu = model_cuda.cpu()
        input_cpu = input_cuda.cpu()
        gesture_logits_cpu, _ = model_cpu(input_cpu)

        torch.testing.assert_close(gesture_logits.cpu(), gesture_logits_cpu, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    unittest.main()
