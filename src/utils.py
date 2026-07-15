import torch
import random
import numpy as np
import pandas as pd

def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_gpu_info():
    """Print GPU information."""
    print("========== GPU detection results ==========")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Detected GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1024**2:.2f} MiB")
        current_device = torch.cuda.current_device()
        print(f"Using device: cuda:{current_device} ({torch.cuda.get_device_name(current_device)})")
    else:
        print("No available GPU detected; using CPU.")
    print("=================================")

def get_device(gpu_id=None):
    """Return the selected torch device."""
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def read_csv_with_encoding_fallback(path):
    """Read a CSV file with common encoding fallbacks."""
    encodings = ['utf-8-sig', 'gbk', 'gb18030']
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            print(f"Warning: Failed to read with encoding {encoding}: {e}")
    raise ValueError(f"Unable to read file with any supported encoding: {path}")
