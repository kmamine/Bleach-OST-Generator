import pynvml
import time
import torch


def get_gpu_metrics():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you have only one GPU
    gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    pynvml.nvmlShutdown()

    return {
        'GPU_Memory_Used': gpu_info.used,
        'GPU_Memory_Free': gpu_info.free,
        'GPU_Utilization': gpu_utilization,
    }

def get_inference_time(model, data):
    start_time = time.time()
    with torch.no_grad():
        output = model(data)
    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time

def get_model_flops(model, input_size):
    # Function to estimate FLOPs for a given model and input size
    # You can implement this using PyTorch's `torch.autograd.profiler.profile` or external libraries like `thop`
    pass

def get_model_size(model):
    # Function to estimate model size in memory
    # You can implement this using PyTorch's `torchsummary` or calculate the number of parameters and their size manually
    pass


