import torch
import max.graph.ops as ops
from max.dtype import DType
from max.graph.type import DeviceRef
from max.graph import Graph, TensorType
from max import engine
import torch
import numpy as np
from torch_max_backend import get_accelerators

class MaxDeviceModule:
   
    
    @staticmethod
    def _is_in_bad_fork():
        return False
    
    @staticmethod
    def manual_seed_all(seed):
        np.random.seed(seed)
    
    @staticmethod
    def device_count():
        return 1
    
    @staticmethod
    def get_rng_state(device=None):
        return torch.tensor(np.random.get_state()[1])
    
    @staticmethod
    def set_rng_state(new_state, device=None):
        if isinstance(new_state, torch.Tensor):
            new_state = new_state.cpu().numpy()
        np_state = ('MT19937', new_state, 624, 0, 0.0)
        np.random.set_state(np_state)
    
    @staticmethod
    def is_available():
        return True
    
    @staticmethod
    def current_device():
        return 0
    
    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float16, torch.bfloat16]
    
    def max_device(self):
        print("hello")



class Custom:
    def __init__(self, t):
        self.t = t


def register_max_ops():
    @torch.library.impl("aten::arange", "PrivateUse1")
    def arange_max(end, dtype=None, layout=None, device=None, pin_memory=None):
        print(f"DEBUG: arange called with end={end}, device={device}")
        with Graph(
            "simple_add_graph", input_types=tuple()
        ) as graph:
            out = ops.range(0, end,1, device=DeviceRef.GPU())
            graph.output(out)

        session = engine.InferenceSession(devices=list(get_accelerators()))
        model = session.load(graph)
        output = model.execute()[0]
        
        return torch.tensor(Custom(output), dtype=torch.float32, device="max_device")
    
    @torch.library.impl("aten::add.Tensor", "max_device")
    def custom_cpu_add(self, other, *, alpha=1):
        print(f"Custom CPU add called!")

        input_type = TensorType(
            dtype=DType.float32, shape=("dim1", "dim2"), device=DeviceRef.GPU()
        )
        with Graph(
            "simple_add_graph", input_types=(input_type, input_type)
        ) as graph:
            lhs, rhs = graph.inputs
            out = ops.add(lhs, rhs)
            graph.output(out)

        session = engine.InferenceSession()
        model = session.load(graph)
        output = model.execute(self, other)[0]
        
        return torch.from_dlpack(output)


class MaxDeviceBackend:
    def __init__(self):
        self.device_module = MaxDeviceModule()
    
    def register(self):
        """Register the max backend with PyTorch."""
        # Step 1: Rename privateuse1 backend to "max_device"
        torch.utils.rename_privateuse1_backend("max_device")
        
        # Step 2: Register the device module
        torch._register_device_module("max_device", self.device_module)
        
        # Step 3: Register operations for the PrivateUse1 dispatch key
        register_max_ops()
        
        # Step 4: Generate helper methods for tensors
        torch.utils.generate_methods_for_privateuse1_backend(
            for_tensor=True,
            for_module=True,
            for_packed_sequence=True,
            for_storage=False
        )
