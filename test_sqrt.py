import torch
from max_backend import MaxDeviceBackend

def test_sqrt():
    # Register the max backend
    backend = MaxDeviceBackend()
    backend.register()
    
    print("=== Testing Sqrt Function ===")
    
    # Create a tensor with perfect square values by manually specifying them
    print("\n1. Creating test tensor:")
    # We'll create an arange and then manually create squares tensor
    a = torch.arange(4, device="max_device")  # [0, 1, 2, 3]
    print(f"a device: {a.device}, values on CPU: {a.to('cpu')}")
    
    # For testing sqrt, let's use the arange values directly
    # sqrt([0, 1, 2, 3]) ≈ [0, 1, 1.414, 1.732]
    print("\n2. Applying sqrt (should stay on GPU):")
    sqrt_result = torch.sqrt(a)
    print(f"sqrt_result device: {sqrt_result.device}, has _max_data: {hasattr(sqrt_result, '_max_data')}")
    
    # Transfer to CPU to see result
    print("\n3. Converting result to CPU:")
    result_cpu = sqrt_result.to('cpu')
    print(f"sqrt([0, 1, 2, 3]) = {result_cpu}")
    
    # Test chaining sqrt with add operation
    print("\n4. Chaining sqrt with add:")
    b = torch.arange(4, device="max_device")  # [0, 1, 2, 3]
    chained = sqrt_result + b  # sqrt([0,1,2,3]) + [0,1,2,3]
    chained_cpu = chained.to('cpu')
    print(f"sqrt([0,1,2,3]) + [0,1,2,3] = {chained_cpu}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_sqrt()