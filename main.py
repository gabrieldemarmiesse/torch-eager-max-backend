import torch

# Import our custom backend
from max_backend import MaxDeviceBackend

def main():
    
    # Register the max backend
    backend = MaxDeviceBackend()
    backend.register()
    
    arange_tensor = torch.arange(5, device="max_device")
    arange_tensor2 = torch.arange(5, device="max_device")

    add_result = arange_tensor + arange_tensor2
    add_result = add_result.to(device="cpu")
    print(add_result)
  
if __name__ == "__main__":
    main()