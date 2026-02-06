# scripts/tensorrt_inference.py – TensorRT Native Inference Runner v1
# For Jetson / NVIDIA servers – valence-aware model selection
# MIT License – Autonomicity Games Inc. 2026

import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path: str):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def infer(engine, context, input_data: np.ndarray, output_shapes: list):
    # Allocate buffers
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    # Copy input
    np.copyto(inputs[0][0], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    # Execute
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()

    # Copy outputs
    output_data = []
    for host_mem, device_mem in outputs:
        out = np.empty(trt.volume(engine.get_tensor_shape(binding)), dtype=np.float32)
        cuda.memcpy_dtoh_async(out, device_mem, stream)
        output_data.append(out.reshape(engine.get_tensor_shape(binding)))
    stream.synchronize()

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT inference demo")
    parser.add_argument("--engine", type=str, required=True, help="Path to .trt engine file")
    parser.add_argument("--input_shape", type=str, default="1,45,225", help="Input shape (comma-separated)")
    parser.add_argument("--output_names", type=str, default="gesture,future_valence", help="Output tensor names (comma-separated)")
    args = parser.parse_args()

    input_shape = tuple(int(x) for x in args.input_shape.split(','))
    output_names = args.output_names.split(',')

    print(f"[TensorRT] Loading engine: {args.engine}")
    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    # Dummy input
    input_data = np.random.randn(*input_shape).astype(np.float32)

    print("[TensorRT] Running inference...")
    outputs = infer(engine, context, input_data, output_names)

    for name, out in zip(output_names, outputs):
        print(f"Output '{name}' shape: {out.shape}")
        print(f"Sample: {out.flat[:8]}")
