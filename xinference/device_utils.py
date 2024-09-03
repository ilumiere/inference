# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from typing_extensions import Literal, Union

# 定义设备类型
DeviceType = Literal["cuda", "mps", "xpu", "npu", "cpu"]
# 设备到环境变量名的映射
DEVICE_TO_ENV_NAME = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
}

# 检查XPU是否可用
def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()

# 检查NPU是否可用
def is_npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except ImportError:
        return False

# 获取可用的设备
def get_available_device() -> DeviceType:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"

# 检查指定设备是否可用
def is_device_available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    elif device == "mps":
        return torch.backends.mps.is_available()
    elif device == "xpu":
        return is_xpu_available()
    elif device == "npu":
        return is_npu_available()
    elif device == "cpu":
        return True
    return False

# 将模型移动到可用设备
def move_model_to_available_device(model):
    device = get_available_device()
    if device == "cpu":
        return model
    return model.to(device)

# 获取设备首选的数据类型
def get_device_preferred_dtype(device: str) -> Union[torch.dtype, None]:
    if device == "cpu":
        return torch.float32
    elif device == "cuda" or device == "mps" or device == "npu":
        return torch.float16
    elif device == "xpu":
        return torch.bfloat16
    return None

# 检查是否支持HuggingFace的Accelerate库
def is_hf_accelerate_supported(device: str) -> bool:
    return device == "cuda" or device == "xpu" or device == "npu"

# 清空设备缓存
def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if is_xpu_available():
        torch.xpu.empty_cache()
    if is_npu_available():
        torch.npu.empty_cache()

# 获取可用设备的环境变量名
def get_available_device_env_name():
    return DEVICE_TO_ENV_NAME.get(get_available_device())

# 获取GPU数量
def gpu_count():
    if torch.cuda.is_available():
        cuda_visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices_env is None:
            return torch.cuda.device_count()
        cuda_visible_devices = (
            cuda_visible_devices_env.split(",") if cuda_visible_devices_env else []
        )
        return min(torch.cuda.device_count(), len(cuda_visible_devices))
    elif is_xpu_available():
        return torch.xpu.device_count()
    elif is_npu_available():
        return torch.npu.device_count()
    else:
        return 0


print(f" gpu count9: {gpu_count()}")