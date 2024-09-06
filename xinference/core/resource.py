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

from dataclasses import dataclass
from typing import Dict, Union

import psutil

from .utils import get_nvidia_gpu_info

# 收集和管理系统资源信息，特别是CPU和GPU的使用情况


@dataclass
class ResourceStatus:
    usage: float  # CPU使用率或资源使用百分比
    total: float  # 总资源量（如CPU核心数）
    memory_used: float  # 已使用的内存量
    memory_available: float  # 可用的内存量
    memory_total: float  # 总内存量


@dataclass
class GPUStatus:
    mem_total: float  # GPU总内存
    mem_free: float  # GPU可用内存
    mem_used: float  # GPU已使用内存


def gather_node_info() -> Dict[str, Union[ResourceStatus, GPUStatus]]:
    # 初始化一个字典来存储节点资源信息
    node_resource = dict()
    
    # 获取系统内存信息
    mem_info = psutil.virtual_memory()
    
    # 添加CPU信息到资源字典
    node_resource["cpu"] = ResourceStatus(
        usage=psutil.cpu_percent() / 100.0,  # CPU使用率（转换为0-1之间的小数）
        total=psutil.cpu_count(),  # CPU核心总数
        memory_used=mem_info.used,  # 已使用的内存
        memory_available=mem_info.available,  # 可用内存
        memory_total=mem_info.total,  # 总内存
    )
    
    # 获取并添加GPU信息到资源字典
    for gpu_idx, gpu_info in get_nvidia_gpu_info().items():
        node_resource[gpu_idx] = GPUStatus(  # type: ignore
            mem_total=gpu_info["total"],  # GPU总内存
            mem_used=gpu_info["used"],  # GPU已使用内存
            mem_free=gpu_info["free"],  # GPU可用内存
        )

    # 返回包含CPU和GPU信息的资源字典
    return node_resource  # type: ignore
