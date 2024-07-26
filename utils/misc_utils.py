import os
from typing import List


# import torch
class ErrorMessageHandler:
    def __init__(self):
        self.error_messages = {
            "Tensors of the same index must be on the same device and the same dtype except `step` tensors that can be CPU and float32 notwithstanding": "🔴 NOTE - please try with 'flash_attention2:false 🔴",
            "Caught ValueError in DataLoader worker process 0.": "🔴NOTE - This error might be caused by a missing tokenizer in the given model folder path. Try providing the Huggingface model id to load the correct tokenizer if the tokenizer is missing in the given model folder.🔴",
            "does not support `device_map='auto'`": "🔴NOTE - This model might not support Multi GPU yet. Please try running it on a single GPU.🔴",
            "does not support Flash Attention 2.0": "🔴 NOTE - please try with 'flash_attention2:false 🔴",
            # Add more error string mappings here
        }

    def get_custom_error_message(self, exception):
        """
        Get the custom error message for a given exception,
        or return the original exception message if not found.
        """
        error_string = str(exception)
        for keys in self.error_messages.keys():
            if keys in error_string:
                return self.error_messages[keys]
            else:
                pass
        return "not_found"


class CudaDeviceEnviron:
    _instance = None

    def __new__(cls, cuda_device_ids_str: str) -> "CudaDeviceEnviron":
        if cls._instance is None:
            cls._instance = super(CudaDeviceEnviron, cls).__new__(cls)
            cls._instance._cuda_device_ids = cls._instance._parse_cuda_device_ids(
                cuda_device_ids_str
            )
            cls._instance._set_cuda_visible_devices()
        else:
            print(
                "Warning: CudaDeviceEnviron is a singleton class. Use CudaDeviceEnviron.get_instance() to get the existing instance."
            )
        return cls._instance

    @classmethod
    def get_instance(cls, cuda_device_ids_str: str = None) -> "CudaDeviceEnviron":
        if cls._instance is None:
            if cuda_device_ids_str is None:
                raise ValueError(
                    "cuda_device_ids_str must be provided when creating the first instance of CudaDeviceEnviron."
                )
            cls._instance = CudaDeviceEnviron(cuda_device_ids_str)
        return cls._instance

    @property
    def cuda_device_ids(self) -> List[int]:
        return self._cuda_device_ids

    @property
    def available_cuda_devices(self) -> int:
        import torch

        return torch.cuda.device_count()

    @property
    def num_device(self) -> int:
        return len(self._cuda_device_ids)

    def _parse_cuda_device_ids(self, cuda_device_ids_str: str) -> List[int]:
        cuda_device_ids = [
            int(dev_id.strip()) for dev_id in cuda_device_ids_str.split(",")
        ]
        if not cuda_device_ids:
            raise ValueError(
                "The input string must contain at least one CUDA device ID."
            )
        return cuda_device_ids

    def _set_cuda_visible_devices(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self._cuda_device_ids))
