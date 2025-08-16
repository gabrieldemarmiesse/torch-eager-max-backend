"""
Custom max backend implementation for PyTorch using the privateuse1 mechanism.
This demonstrates how to create a custom device similar to XLA's approach.
"""

import torch
import numpy as np


class NumpyDeviceModule:
    """
    Device module that implements the required backend interface for max.
    This follows the same pattern as XLA's device_module.
    """
    
    @staticmethod
    def _is_in_bad_fork():
        """Required for random number generation support."""
        return False
    
    @staticmethod
    def manual_seed_all(seed):
        """Sets the seed for generating random numbers."""
        np.random.seed(seed)
    
    @staticmethod
    def device_count():
        """Returns the number of numpy devices available."""
        return 1
    
    @staticmethod
    def get_rng_state(device=None):
        """Returns the random number generator state."""
        return torch.tensor(np.random.get_state()[1])
    
    @staticmethod
    def set_rng_state(new_state, device=None):
        """Sets the random number generator state."""
        if isinstance(new_state, torch.Tensor):
            new_state = new_state.cpu().numpy()
        # Reconstruct the numpy random state
        np_state = ('MT19937', new_state, 624, 0, 0.0)
        np.random.set_state(np_state)
    
    @staticmethod
    def is_available():
        """Returns whether numpy backend is available."""
        return True
    
    @staticmethod
    def current_device():
        """Returns the current device index."""
        return 0
    
    @staticmethod
    def get_amp_supported_dtype():
        """Returns supported dtypes for automatic mixed precision."""
        return [torch.float16, torch.bfloat16]


# Register operations for the PrivateUse1 dispatch key

def _torch_to_numpy_dtype(torch_dtype):
    """Convert PyTorch dtype to numpy dtype."""
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    return mapping.get(torch_dtype, np.float32)

def _numpy_to_torch_dtype(np_dtype):
    """Convert numpy dtype to PyTorch dtype."""
    mapping = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.int16: torch.int16,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.bool_: torch.bool,
    }
    return mapping.get(np_dtype, torch.float32)

# Register operations using the proper PyTorch dispatch mechanism
def register_numpy_ops():
    """Register numpy device operations using proper dispatch registration."""
    
    @torch.library.impl("aten::empty.memory_format", "PrivateUse1")
    def empty_max(size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
        print(f"DEBUG: empty.memory_format called with size={size}, dtype={dtype}")
        dtype = dtype or torch.float32
        np_dtype = _torch_to_numpy_dtype(dtype)
        array = np.empty(size, dtype=np_dtype)
        
        # Create tensor from numpy array and mark it as max
        tensor = torch.from_numpy(array.copy())
        tensor = tensor.to(dtype=dtype)
        tensor._numpy_array = array
        tensor._device_type = "max_device"
        
        return tensor
    
    @torch.library.impl("aten::empty_strided", "PrivateUse1")
    def empty_strided_max(size, stride, dtype=None, layout=None, device=None, pin_memory=None):
        print(f"DEBUG: empty_strided called with size={size}, stride={stride}, dtype={dtype}")
        dtype = dtype or torch.float32
        np_dtype = _torch_to_numpy_dtype(dtype)
        array = np.empty(size, dtype=np_dtype)
        
        # Create tensor from numpy array
        tensor = torch.from_numpy(array.copy())
        tensor = tensor.to(dtype=dtype)
        tensor._numpy_array = array
        tensor._device_type = "max_device"
        
        return tensor
    
    @torch.library.impl("aten::zeros", "PrivateUse1")
    def zeros_max(size, dtype=None, layout=None, device=None, pin_memory=None):
        dtype = dtype or torch.float32
        np_dtype = _torch_to_numpy_dtype(dtype)
        array = np.zeros(size, dtype=np_dtype)
        
        # Create tensor from numpy array
        tensor = torch.from_numpy(array.copy())
        tensor = tensor.to(dtype=dtype)
        tensor._numpy_array = array
        tensor._device_type = "max_device"
        
        return tensor
    
    @torch.library.impl("aten::arange", "PrivateUse1")
    def arange_max(end, dtype=None, layout=None, device=None, pin_memory=None):
        print(f"DEBUG: arange called with end={end}, device={device}")
        dtype = dtype or torch.float32
        np_dtype = _torch_to_numpy_dtype(dtype)
        array = np.arange(end, dtype=np_dtype)
        
        # Create tensor from numpy array
        tensor = torch.from_numpy(array.copy())
        tensor = tensor.to(dtype=dtype)
        
        # Properly mark as max tensor by creating with device info
        # This is tricky - we need to create it as if it's on the max
        tensor = torch.empty_like(tensor, device=torch.device("max_device:0"))
        tensor.copy_(torch.from_numpy(array.copy()).to(dtype=dtype))
        tensor._numpy_array = array
        
        return tensor
    
    @torch.library.impl("aten::arange.start", "PrivateUse1")
    def arange_start_max(start, end, dtype=None, layout=None, device=None, pin_memory=None):
        dtype = dtype or torch.float32
        np_dtype = _torch_to_numpy_dtype(dtype)
        array = np.arange(start, end, dtype=np_dtype)
        
        # Create tensor from numpy array
        tensor = torch.from_numpy(array.copy())
        tensor = tensor.to(dtype=dtype)
        tensor._numpy_array = array
        tensor._device_type = "max_device"
        
        return tensor
    
    @torch.library.impl("aten::arange.start_step", "PrivateUse1")
    def arange_start_step_max(start, end, step=1, dtype=None, layout=None, device=None, pin_memory=None):
        dtype = dtype or torch.float32
        np_dtype = _torch_to_numpy_dtype(dtype)
        array = np.arange(start, end, step, dtype=np_dtype)
        
        # Create tensor from numpy array
        tensor = torch.from_numpy(array.copy())
        tensor = tensor.to(dtype=dtype)
        tensor._numpy_array = array
        tensor._device_type = "max_device"
        
        return tensor
    
    @torch.library.impl("aten::sqrt", "PrivateUse1")
    def sqrt_max(input):
        if hasattr(input, '_numpy_array'):
            array = input._numpy_array
        else:
            array = input.detach().cpu().numpy()
        
        result_array = np.sqrt(array)
        result_tensor = torch.from_numpy(result_array.copy())
        result_tensor = result_tensor.to(dtype=input.dtype)
        result_tensor._numpy_array = result_array
        result_tensor._device_type = "max_device"
        
        return result_tensor
    
    @torch.library.impl("aten::_to_copy", "PrivateUse1")
    def to_copy_max(input, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
        print(f"DEBUG: _to_copy called with device={device}, dtype={dtype}")
        if hasattr(input, '_numpy_array'):
            array = input._numpy_array
        else:
            array = input.detach().cpu().numpy()
        
        if dtype is not None:
            np_dtype = _torch_to_numpy_dtype(dtype)
            array = array.astype(np_dtype)
        
        result_tensor = torch.from_numpy(array.copy())
        if dtype is not None:
            result_tensor = result_tensor.to(dtype=dtype)
        
        result_tensor._numpy_array = array
        result_tensor._device_type = "max_device"
        
        return result_tensor
    
    # Register copy_default for device conversion
    @torch.library.impl("aten::copy_", "PrivateUse1")
    def copy_max(input, src, non_blocking=False):
        print(f"DEBUG: copy_ called")
        # Convert source to numpy and copy to our tensor
        if hasattr(input, '_numpy_array'):
            input_array = input._numpy_array
        else:
            input_array = input.detach().cpu().numpy()
            
        src_array = src.detach().cpu().numpy()
        np.copyto(input_array, src_array)
        
        return input
    
    print("✓ Registered max operations using proper dispatch mechanism")

class NumpyDeviceBackend:
    """
    Main backend class that handles registration and setup of the max.
    """
    
    def __init__(self):
        self.device_module = NumpyDeviceModule()
    
    def register(self):
        """Register the max backend with PyTorch."""
        # Step 1: Rename privateuse1 backend to "max_device"
        torch.utils.rename_privateuse1_backend("max_device")
        
        # Step 2: Register the device module
        torch._register_device_module("max_device", self.device_module)
        
        # Step 3: Register operations for the PrivateUse1 dispatch key
        register_numpy_ops()
        
        # Step 4: Generate helper methods for tensors
        torch.utils.generate_methods_for_privateuse1_backend(
            for_tensor=True,
            for_module=True,
            for_packed_sequence=True,
            for_storage=False
        )
        
        print("✓ max backend registered successfully!")
        print("  You can now use 'max' as a device string in PyTorch")
        print("  Example: tensor.to('max')")


# Create a global instance for easy access
numpy_backend = NumpyDeviceBackend()

def hello():
    """Legacy function from original max_backend.py"""
    print("Hello from max backend!")