import torch

"""
Run this to verify if cuda installation is working
"""

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    
    # Get GPU information
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # Simple tensor test
    try:
        temp = torch.arange(1, 10)
        temp_cuda = temp.cuda()
        print(f"Tensor on CPU: {temp}")
        print(f"Tensor on CUDA: {temp_cuda}")
    except RuntimeError as e:
        print(f"Error moving tensor to CUDA: {e}")
else:
    print("CUDA is not available.")
