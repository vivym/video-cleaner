from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch
import tensorrt as trt
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class TRTEngine:
    def __init__(self, engine_path: str | os.PathLike):
        self.engine_path = engine_path

        self.engine = None
        self.context = None

        self.tensors: dict[str, torch.Tensor] = OrderedDict()
        self.cuda_graph_instance = None # cuda graph

    def load(self) -> "TRTEngine":
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            assert rt
            self.engine = rt.deserialize_cuda_engine(f.read())
        assert self.engine

    def activate(self, shared_device_memory_ptr: int | None = None) -> "TRTEngine":
        if shared_device_memory_ptr is not None:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = shared_device_memory_ptr
        else:
            self.context = self.engine.create_execution_context()
        assert self.engine

    def allocate_buffers(
        self,
        shape_dict: dict[str, tuple[int, ...]] | None = None,
        device: str | torch.device = "cuda"
    ) -> None:
        for idx in range(get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor

    def infer(
        self,
        feed_dict: dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
        use_cuda_graph: bool = False,
    ) -> dict[str, torch.Tensor]:
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        cuda_stream = stream.cuda_stream

        stream.synchronize()
        torch.cuda.synchronize()

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, cuda_stream))
                CUASSERT(cudart.cudaStreamSynchronize(cuda_stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(cuda_stream)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(cuda_stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                self.context.execute_async_v3(cuda_stream)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(cuda_stream))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(cuda_stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        stream.synchronize()
        torch.cuda.synchronize()

        return self.tensors


def get_bindings_per_profile(engine: trt.ICudaEngine) -> int:
    return engine.num_bindings // engine.num_optimization_profiles
