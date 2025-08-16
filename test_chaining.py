import torch
from max_backend import MaxDeviceBackend

def test_gpu_chaining():
    # Register the max backend
    backend = MaxDeviceBackend()
    backend.register()
    
    print("=== Testing GPU Tensor Chaining ===")
    
    # Create tensors on MAX device - these should stay on GPU
    print("\n1. Creating tensors on MAX device:")
    a = torch.arange(200000, device="max_device")
    print(f"a device: {a.device}, has _max_data: {hasattr(a, '_max_data')}")
    
    b = torch.arange(200000, device="max_device") 
    print(f"b device: {b.device}, has _max_data: {hasattr(b, '_max_data')}")
    
    # Chain multiple operations - these should all stay on GPU
    print("\n2. Performing chained operations (should stay on GPU):")
    for _ in range(100):
        a = torch.sqrt(b)
    # First add operation
    print("Computing a + b...")
    c = a + b
    print(f"c device: {c.device}, has _max_data: {hasattr(c, '_max_data')}")
    
    # Second add operation (chaining)
    print("Computing c + a...")
    d = c + a
    print(f"d device: {d.device}, has _max_data: {hasattr(d, '_max_data')}")
    
    # Third add operation (more chaining)
    print("Computing d + b...")
    e = d + b
    print(f"e device: {e.device}, has _max_data: {hasattr(e, '_max_data')}")
    
    # Only now transfer to CPU - this should be the only GPU->CPU transfer
    print("\n3. Transferring final result to CPU (only GPU->CPU transfer):")
    result_cpu = e.to(device="cpu")
    print(f"Final result: {result_cpu}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_gpu_chaining()