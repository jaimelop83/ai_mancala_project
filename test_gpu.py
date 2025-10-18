import torch
import subprocess

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. PyTorch is using the CPU.")

    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True)
        print("Detected GPU(s) by OS:")
        print(result.stdout)
    except Exception as e:
        print(f"Could not determine GPU via OS: {e}")


