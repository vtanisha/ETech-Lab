import torch

def print_info():
    print("=====================================")
    print("PyTorch version:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device:", device)
        x = torch.randn(3, 3, device=device)
        y = x * 2
        print("MPS computation result:\n", y)
    else:
        print("MPS not available, using CPU")
        x = torch.randn(3, 3)
        y = x * 2
        print("CPU computation result:\n", y)
    print("=====================================")

if __name__ == "__main__":
    print_info()
