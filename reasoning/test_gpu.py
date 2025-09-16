import torch

# This is the main function to check for CUDA
if torch.cuda.is_available():
    print("CUDA is available! 🎉")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch will use the CPU. 😢")
