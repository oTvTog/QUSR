import torch


def has_mps():
    return False


def get_cuda_device_string():
    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()
    if has_mps():
        return "mps"
    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_for_nans(x, where):
    if not torch.all(torch.isnan(x)).item():
        return
    raise RuntimeError(f"A tensor with all NaNs was produced in {where}.")
