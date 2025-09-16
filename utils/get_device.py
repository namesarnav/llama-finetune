
import torch

def set_device(): 

    device = "cpu" # defaults to cpu device
    
    if torch.backends.mps.is_available():
        print("Model running on Apple Silicon, MPS is available\nSelecting MPS as default device")
        device = torch.device("mps")
        return device

    elif torch.cuda.is_available(): 
        print("CUDA Device found, Defaulting to CUDA")
        device = torch.device("cuda")
        return device

    else: 
        print("No accelarator device found, Defaulting to CPU")
        return device

