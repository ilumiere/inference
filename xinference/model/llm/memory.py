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

# NOTE:
#
#   The algorithum is ported from https://github.com/RahulSChand/gpu_poor
#
#   Improvement:
#
#      The original js code only calculate kv_cache_dtype by float32, instead of most case we run model with float16.
#
#   Known Issue:
#
#       * On vllm, some MHA model use smaller memory than calculation (qwen1.5-7B-chat-gptq-int4,
#       qwen1.5-14B-chat-gptq-int4 with large activation_mem).
#
#       * On vllm, gemma-it-7B pytorch format model use larger gpu mem than calculation

import json
import math
from dataclasses import dataclass
from logging import getLogger
from math import ceil
from typing import Any, Optional, Union

from .llm_family import convert_model_size_to_float

logger = getLogger(__name__)


@dataclass
class ModelLayersInfo:
    """
    模型层信息的数据类
    """
    vocab_size: int  # 词汇表大小
    heads: int  # 注意力头数量（也称为num_attention_heads, num_heads或n_head）
    hidden_dim: int  # 隐藏层维度（也称为hidden_size, d_model或n_embd）
    inter_dim: int  # 中间层维度（也称为intermediate_size, n_inner或d_ff）
    num_layers: int  # 层数（也称为num_layers, num_hidden_layers或n_layer）


@dataclass
class ModelMemInfo:
    """
    模型内存信息的数据类，单位为MB
    """
    model_mem: int  # 模型参数占用的内存
    kv_cache_mem: int  # KV缓存占用的内存
    activation_mem: int  # 激活值占用的内存
    overhead: int  # 额外开销
    total: int  # 总内存占用


# 量化方法的标准化映射
QUANT_NORMALIZE = {"int4": "4-bit", "int8": "8-bit", "4-bit": "4-bit", "8-bit": "8-bit"}

# GGUF格式不同量化方法的内存乘数因子
GGUF_MULTI_FACTOR_DICT = {
    "q4_0": 18,
    "q4_1": 20,
    "q5_0": 22,
    "q5_1": 24,
    "q8_0": 34,
    "q8_1": 40,
}

# GGUF格式64位量化方法的内存乘数因子
GGUF_MULTI_FACTOR_DICT_64 = {
    "q6_K": 54.0,
    "q3": 26.0,
    "q4": 38.0,
    "q5": 46.0,
}

# GGUF格式组合量化方法的内存乘数因子
GGUF_MULTI_FACTOR_DICT_COMBINE = {
    "q3_K_L": [38.0, 26.0],
    "q3_K_M": [46.0, 26.0],
    "q4_K_S": [46.0, 38.0],
    "q4_K_M": [54.0, 38.0],
    "q5_K_M": [54.0, 46.0],
    "q2_K": [26.0, 22.0],
}


# 估算LLM GPU内存占用，返回单位为MB
def estimate_llm_gpu_memory(
    model_size_in_billions: Union[str, int],
    quantization: Optional[str],
    context_length: int,  # 输入+输出的总长度
    model_format: str,
    model_name: Optional[str] = None,
    kv_cache_dtype: int = 16,
) -> Optional[ModelMemInfo]:
    """
    估算LLM GPU内存占用
    
    参数:
    model_size_in_billions: 模型大小，必须是字符串如"1_8"或"46_7"，以匹配llm
    quantization: 量化方法
    context_length: 上下文长度（输入+输出）
    model_format: 模型格式
    model_name: 模型名称（可选）
    kv_cache_dtype: KV缓存的数据类型（默认为16位）

    返回:
    ModelMemInfo对象，包含详细的内存占用信息
    """
    # 获取模型层信息
    info = get_model_layers_info(
        model_size_in_billions,
        model_name,
        model_format,
        quantization,
    )
    
    # 如果无法获取模型层信息，则返回None
    if info is None:
        return None
    
    # 将模型大小转换为浮点数
    size_in_billions = convert_model_size_to_float(model_size_in_billions)
    
    # 调用详细估算函数并返回结果
    return estimate_llm_gpu_memory_details(
        info,
        size_in_billions,
        quantization,
        context_length,
        model_format,
        kv_cache_dtype,
    )


def estimate_llm_gpu_memory_details(
    info: ModelLayersInfo,
    size_in_billions: float,
    quantization: Optional[str],
    context_length: int,  # 输入+输出的总长度
    model_format: str,
    kv_cache_dtype: int = 16,
) -> ModelMemInfo:
    """
    估算LLM GPU内存占用的详细信息
    
    参数:
    info: ModelLayersInfo对象，包含模型层信息
    size_in_billions: 模型大小（单位：十亿参数）
    quantization: 量化方法
    context_length: 上下文长度（输入+输出）
    model_format: 模型格式
    kv_cache_dtype: KV缓存的数据类型（默认为16位）

    返回:
    ModelMemInfo对象，包含模型内存、KV缓存、额外开销和激活内存的详细信息
    """
    # 验证KV缓存数据类型
    if kv_cache_dtype not in [8, 16, 32]:
        raise ValueError(f"无效的kv_cache_dtype {kv_cache_dtype}")
    
    # 根据KV缓存数据类型确定每个元素的字节大小
    if kv_cache_dtype == 8:
        kv_dtype_size = 1
    elif kv_cache_dtype == 16:
        kv_dtype_size = 2
    else:
        kv_dtype_size = 4
    
    # 设置基础开销
    overhead = 650.0
    
    # 根据模型格式进行不同的内存估算
    if model_format == "ggufv2":
        assert quantization is not None and quantization != "none"
        # 计算模型大小
        model_size_in_mb = _compute_model_size_gguf(info, quantization)
        # 计算推理内存
        inference_mem = float(
            context_length * kv_dtype_size * info.hidden_dim * info.num_layers
        )
        inference_mem = inference_mem / 1024.0 / 1024.0
        # 计算激活内存
        activation_mem = _compute_inference_only_activation_memory(context_length, info)
        # 调整开销
        overhead = overhead + context_length * 0.1
    else:
        if quantization is not None:
            assert isinstance(quantization, str)
            quantization = QUANT_NORMALIZE[quantization.lower()]
            assert quantization is not None

        # 计算模型大小
        model_size = size_in_billions * 1000000000.0
        model_size_in_mb = _convert_to_mb_model_size(model_size, quantization)
        # 计算KV缓存
        inference_mem = float(
            context_length * 2 * kv_dtype_size * info.hidden_dim * info.num_layers
        )
        inference_mem = inference_mem / 1024.0 / 1024.0
        # 计算激活内存
        activation_mem = _compute_inference_only_activation_memory(context_length, info)

    # 计算总内存
    total_mem = ceil(inference_mem + model_size_in_mb + overhead + activation_mem)
    
    # 返回ModelMemInfo对象
    return ModelMemInfo(
        model_mem=ceil(model_size_in_mb),
        kv_cache_mem=ceil(inference_mem),
        activation_mem=ceil(activation_mem),
        overhead=ceil(overhead),
        total=total_mem,
    )


def _load_item_from_json(config_data: Any, *keys: str) -> str:
    """
    从JSON配置中加载指定键的值
    
    参数:
    config_data: JSON配置数据
    *keys: 要查找的键名列表

    返回:
    找到的第一个非空值
    """
    assert len(keys) > 0
    for key in keys:
        v = config_data.get(key)
        if v is not None:
            return v
    raise ValueError("加载ModelLayersInfo时缺少 %s" % (keys[0]))


def load_model_config_json(config_path: str) -> ModelLayersInfo:
    """
    从JSON配置文件加载模型层信息
    
    参数:
    config_path: 配置文件路径

    返回:
    ModelLayersInfo对象
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
        return ModelLayersInfo(
            vocab_size=int(_load_item_from_json(config_data, "vocab_size")),
            heads=int(
                _load_item_from_json(
                    config_data, "num_key_value_heads", "num_attention_heads"
                )
            ),
            hidden_dim=int(
                _load_item_from_json(config_data, "hidden_size", "d_model", "n_embd")
            ),
            inter_dim=int(_load_item_from_json(config_data, "intermediate_size")),
            num_layers=int(
                _load_item_from_json(
                    config_data, "num_hidden_layers", "num_layers", "n_layer"
                )
            ),
        )


def get_model_layers_info(
    model_size_in_billions: Union[str, int],
    model_name: Optional[str],
    model_format: Optional[str],
    quantization: Optional[str],
) -> Optional[ModelLayersInfo]:
    """
    获取模型层信息
    
    参数:
    model_size_in_billions: 模型大小
    model_name: 模型名称
    model_format: 模型格式
    quantization: 量化方法

    返回:
    ModelLayersInfo对象或None
    """
    from . import match_llm
    from .llm_family import cache_model_config

    if not model_name:
        logger.debug("通过默认大小获取模型层信息 size=%s", model_size_in_billions)
        size_in_billions = convert_model_size_to_float(model_size_in_billions)
        return _get_default_layers_from_size(size_in_billions)
    match_result = match_llm(
        model_name=model_name,
        model_format=model_format,
        model_size_in_billions=model_size_in_billions,
        quantization=quantization,
    )
    if not match_result:
        return None
    llm_family, llm_spec, _quant = match_result
    config_path = cache_model_config(llm_family, llm_spec)
    return load_model_config_json(config_path)


def _get_default_layers_from_size(size_in_billion: float) -> ModelLayersInfo:
    """
    根据模型大小获取默认的层信息
    
    参数:
    size_in_billion: 模型大小（单位：十亿参数）

    返回:
    ModelLayersInfo对象
    """
    if size_in_billion < 5:
        vocab_size = 32000
        heads = 32
        num_layers = 24
    elif size_in_billion < 10:
        vocab_size = 32000
        heads = 32
        num_layers = 32
    elif size_in_billion < 24:
        vocab_size = 32000
        heads = 40
        num_layers = 40
    elif size_in_billion < 55:
        vocab_size = 32000
        heads = 60
        num_layers = 48
    else:
        vocab_size = 32000
        heads = 64
        num_layers = 80

    model_size = int(size_in_billion * 1000000000)
    A = num_layers * 4 + 3 * 4 * num_layers
    B = 2 * vocab_size
    C = -1 * model_size
    h = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    h = math.ceil(h)
    return ModelLayersInfo(
        vocab_size=vocab_size,
        heads=heads,
        hidden_dim=h,
        inter_dim=4 * h,
        num_layers=num_layers,
    )


def _convert_to_mb_model_size(model_size: float, quantization: Optional[str]) -> float:
    """
    将模型大小转换为MB单位
    
    参数:
    model_size: 模型大小（单位：参数数量）
    quantization: 量化方法

    返回:
    模型大小（单位：MB）
    """
    extra = 0.0
    fB = 2.0
    size = (model_size * fB) / (1024.0 * 1024.0)
    # bnb_q4 == 4-bit ?
    if quantization == "8-bit" or quantization == "4-bit":
        extra = 0.06 * size
    if quantization == "8-bit":
        size = size / 2
    if quantization == "4-bit":
        size = size / 4
    return size + extra


def _compute_inference_only_activation_memory(
    context_length: int, info: ModelLayersInfo
) -> float:
    """
    计算仅推理时的激活内存
    
    参数:
    context_length: 上下文长度
    info: ModelLayersInfo对象

    返回:
    激活内存大小（单位：MB）
    """
    hidden_dim = info.hidden_dim
    heads = info.heads
    ret = (
        (context_length * hidden_dim * 5 * 2 + (context_length**2) * heads * 2)
        / 1024
        / 1024
    )
    return ret


def _compute_model_size_gguf(info: ModelLayersInfo, quantization: str) -> float:
    """
    计算GGUF格式模型的大小
    
    参数:
    info: ModelLayersInfo对象
    quantization: 量化方法

    返回:
    模型大小（单位：MB）
    """
    assert quantization is not None
    vocab_size = info.vocab_size
    num_layers = info.num_layers
    hidden_dim = info.hidden_dim
    inter_dim = info.inter_dim
    total_params = int(
        vocab_size * hidden_dim * 2
        + num_layers * 4 * (hidden_dim**2)
        + num_layers * 3 * inter_dim * hidden_dim
    )
    other_v_down_params = (
        num_layers * (hidden_dim**2) + num_layers * hidden_dim * inter_dim
    )
    other_param_q2k = (
        total_params - (hidden_dim**2) * num_layers * 2 + 2 * vocab_size * hidden_dim
    )

    total = 0.0
    v1 = GGUF_MULTI_FACTOR_DICT.get(quantization)
    if v1 is not None:
        total = (v1 * total_params) / (32 * 1024 * 1024)
    v2 = GGUF_MULTI_FACTOR_DICT_64.get(quantization)
    if v2 is not None:
        total = (v2 * total_params) / (64 * 1024 * 1024)
    v3 = GGUF_MULTI_FACTOR_DICT_COMBINE.get(quantization)
    if v3 is not None:
        factors = v3
        if quantization == "q2_K":
            total = (
                (total_params - other_param_q2k) * factors[1]
                + other_param_q2k * factors[0]
            ) / (64 * 1024 * 1024)
        else:
            total = (
                (total_params - other_v_down_params) * factors[1]
                + other_v_down_params * factors[0]
            ) / (64 * 1024 * 1024)
    return total
