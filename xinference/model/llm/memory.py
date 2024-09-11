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
    从JSON配置文件加载大语言模型的层信息。

    此函数读取指定的JSON配置文件，解析其内容，并创建一个ModelLayersInfo对象，
    该对象包含了模型的关键结构参数。

    参数:
    config_path (str): JSON配置文件的路径。

    返回:
    ModelLayersInfo: 包含模型层信息的对象，具有以下属性：
        - vocab_size: 词汇表大小
        - heads: 注意力头的数量
        - hidden_dim: 隐藏层的维度
        - inter_dim: 中间层的维度
        - num_layers: 模型层数

    异常:
    - IOError: 如果无法打开或读取配置文件。
    - json.JSONDecodeError: 如果配置文件不是有效的JSON格式。
    - ValueError: 如果配置文件中缺少必要的字段。

    注意:
    - 该函数使用_load_item_from_json辅助函数来处理不同模型可能使用的不同键名。
    - 所有从JSON加载的值都被转换为整数类型。
    """
    # try:
    with open(config_path, "r") as f:
        config_data = json.load(f)
        
        # 创建并返回ModelLayersInfo对象，填充所有必要的字段
        return ModelLayersInfo(
            # 获取词汇表大小
            vocab_size=int(_load_item_from_json(config_data, "vocab_size")),
            
            # 获取注意力头数量，可能有不同的键名
            heads=int(
                _load_item_from_json(
                    config_data, "num_key_value_heads", "num_attention_heads"
                )
            ),
            
            # 获取隐藏层维度，可能有不同的键名
            hidden_dim=int(
                _load_item_from_json(config_data, "hidden_size", "d_model", "n_embd")
            ),
            
            # 获取中间层维度
            inter_dim=int(_load_item_from_json(config_data, "intermediate_size")),
            
            # 获取模型层数，可能有不同的键名
            num_layers=int(
                _load_item_from_json(
                    config_data, "num_hidden_layers", "num_layers", "n_layer"
                )
            ),
        )
    # except IOError as e:
    #     logger.error(f"无法打开或读取配置文件: {config_path}")
    #     raise
    # except json.JSONDecodeError as e:
    #     logger.error(f"配置文件不是有效的JSON格式: {config_path}")
    #     raise
    # except ValueError as e:
    #     logger.error(f"配置文件缺少必要字段: {str(e)}")
    #     raise


def get_model_layers_info(
    model_size_in_billions: Union[str, int],
    model_name: Optional[str],
    model_format: Optional[str],
    quantization: Optional[str],
) -> Optional[ModelLayersInfo]:
    """
    获取大语言模型的层信息。

    此函数用于获取指定大语言模型的层级结构信息。它可以处理两种情况：
    1. 当没有提供模型名称时，根据模型大小估算默认的层信息。
    2. 当提供模型名称时，尝试匹配具体的模型并加载其配置信息。

    参数:
    model_size_in_billions (Union[str, int]): 模型大小，以十亿参数为单位。可以是字符串或整数。
    model_name (Optional[str]): 模型名称。如果为None，将使用默认估算方法。
    model_format (Optional[str]): 模型格式，用于模型匹配。
    quantization (Optional[str]): 量化方法，用于模型匹配。

    返回:
    Optional[ModelLayersInfo]: 如果成功获取模型信息，返回ModelLayersInfo对象；否则返回None。

    异常:
    该函数不直接抛出异常，但内部调用的函数可能会抛出异常。

    注意:
    - 此函数依赖于外部导入的match_llm和cache_model_config函数。
    - 使用了日志记录来跟踪函数的执行过程。
    """
    from . import match_llm
    from .llm_family import cache_model_config

    # 处理没有提供模型名称的情况
    if not model_name:
        logger.debug("通过默认大小获取模型层信息 size=%s", model_size_in_billions)
        # 将模型大小转换为浮点数
        size_in_billions = convert_model_size_to_float(model_size_in_billions)
        # 使用默认方法估算模型层信息
        return _get_default_layers_from_size(size_in_billions)

    # 尝试匹配指定的模型
    match_result = match_llm(
        model_name=model_name,
        model_format=model_format,
        model_size_in_billions=model_size_in_billions,
        quantization=quantization,
    )

    # 如果没有匹配到模型，返回None
    if not match_result:
        return None

    # 解包匹配结果
    llm_family, llm_spec, _quant = match_result
    # 缓存模型配置并获取配置文件路径
    config_path = cache_model_config(llm_family, llm_spec)
    # 加载并返回模型配置信息
    return load_model_config_json(config_path)


def _get_default_layers_from_size(size_in_billion: float) -> ModelLayersInfo:
    """
    根据模型大小估算默认的层信息。

    此函数通过给定的模型大小（以十亿参数为单位）来估算模型的各种架构参数，
    如词汇表大小、注意力头数量、层数等。然后，它使用这些估算值计算隐藏维度，
    并返回一个包含所有这些信息的 ModelLayersInfo 对象。

    参数:
    size_in_billion (float): 模型大小，单位为十亿参数。

    返回:
    ModelLayersInfo: 包含估算的模型架构信息的对象。

    算法说明:
    1. 根据模型大小确定基本参数（词汇表大小、注意力头数、层数）。
    2. 使用二次方程求解隐藏维度 (h)。
    3. 使用计算得到的 h 值构建并返回 ModelLayersInfo 对象。
    """
    # 根据模型大小确定基本参数
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

    # 将模型大小转换为参数数量
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
        inter_dim=4 * h,  # 中间层维度通常是隐藏维度的 4 倍
        num_layers=num_layers,
    )

def _convert_to_mb_model_size(model_size: float, quantization: Optional[str]) -> float:
    """
    将模型大小从参数数量转换为MB单位，并考虑量化方法的影响。

    此函数用于估算模型在内存中的实际大小，考虑了不同量化方法对模型大小的影响。
    它首先将参数数量转换为基本的MB大小，然后根据量化方法进行调整。

    参数:
    model_size (float): 模型的参数数量。
    quantization (Optional[str]): 使用的量化方法，可以是 "8-bit", "4-bit" 或 None。

    返回:
    float: 估算的模型大小，单位为MB。

    函数逻辑:
    1. 初始化额外内存 extra 为 0。
    2. 设置基本转换因子 fB 为 2.0（可能是考虑到每个参数占用 2 字节）。
    3. 将参数数量转换为基本的 MB 大小。
    4. 根据量化方法调整大小：
       - 对于 8-bit 和 4-bit 量化，添加额外 6% 的内存占用。
       - 8-bit 量化将大小减半。
       - 4-bit 量化将大小减少到原来的 1/4。
    5. 返回调整后的总大小。

    注意:
    - 此函数假设默认情况下每个参数占用 2 字节。
    - 额外的 6% 内存可能是为了存储量化相关的额外信息。
    """
    extra = 0.0
    fB = 2.0
    # 将参数数量转换为MB
    size = (model_size * fB) / (1024.0 * 1024.0)
    
    # 处理量化情况
    if quantization in ["8-bit", "4-bit"]:
        # 为量化模型添加额外 6% 的内存占用
        extra = 0.06 * size
        
        if quantization == "8-bit":
            # 8-bit 量化将模型大小减半
            size = size / 2
        elif quantization == "4-bit":
            # 4-bit 量化将模型大小减少到原来的 1/4
            size = size / 4
    
    # 返回调整后的总大小
    return size + extra

def _compute_inference_only_activation_memory(
    context_length: int, info: ModelLayersInfo
) -> float:
    """
    计算仅用于推理时的模型激活内存大小。

    此函数估算在仅进行推理（不包括训练）时，模型所需的激活内存大小。
    激活内存是模型在处理输入数据时临时存储中间计算结果所需的内存。

    参数:
    context_length (int): 输入序列的长度，即模型一次处理的token数量。
    info (ModelLayersInfo): 包含模型结构信息的对象，特别是hidden_dim和heads。

    返回:
    float: 估算的激活内存大小，单位为MB（兆字节）。

    计算逻辑:
    1. 提取模型的隐藏层维度(hidden_dim)和注意力头数量(heads)。
    2. 计算激活内存，考虑以下因素：
       - context_length * hidden_dim * 5 * 2: 可能代表各种中间张量，如查询、键、值向量等。
       - (context_length**2) * heads * 2: 可能表示注意力矩阵的内存占用。
    3. 将计算结果转换为MB单位。

    注意:
    - 此计算是一个估算，实际内存使用可能因具体实现而略有不同。
    - 计算假设使用32位浮点数（每个数占4字节，hence * 2 in bytes）。
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
    计算GGUF（GPT-Generated Unified Format）格式模型的大小。

    此函数用于估算使用不同量化方法的GGUF格式模型的内存占用。
    它考虑了模型的各种参数，如词汇表大小、层数、隐藏维度等，
    并根据不同的量化方法应用相应的计算因子。

    参数:
    info (ModelLayersInfo): 包含模型结构信息的对象，如词汇表大小、层数等。
    quantization (str): 使用的量化方法，如 "q2_K", "q4_0" 等。

    返回:
    float: 估算的模型大小，单位为MB。

    函数流程:
    1. 断言确保quantization参数不为None。
    2. 计算模型的总参数量和特定参数组的数量。
    3. 根据不同的量化方法，使用预定义的因子计算模型大小。
    4. 返回计算得到的模型大小。
    """
    # 确保quantization参数已提供
    assert quantization is not None
    
    # 从ModelLayersInfo对象中提取模型结构信息
    vocab_size = info.vocab_size
    num_layers = info.num_layers
    hidden_dim = info.hidden_dim
    inter_dim = info.inter_dim
    
    # 计算模型的总参数量
    total_params = int(
        vocab_size * hidden_dim * 2
        + num_layers * 4 * (hidden_dim**2)
        + num_layers * 3 * inter_dim * hidden_dim
    )
    
    # 计算其他特定参数组的数量
    other_v_down_params = (
        num_layers * (hidden_dim**2) + num_layers * hidden_dim * inter_dim
    )
    other_param_q2k = (
        total_params - (hidden_dim**2) * num_layers * 2 + 2 * vocab_size * hidden_dim
    )

    total = 0.0
    # 使用GGUF_MULTI_FACTOR_DICT字典获取量化因子，并计算模型大小
    v1 = GGUF_MULTI_FACTOR_DICT.get(quantization)
    if v1 is not None:
        total = (v1 * total_params) / (32 * 1024 * 1024)
    
    # 使用GGUF_MULTI_FACTOR_DICT_64字典获取量化因子，并计算模型大小
    v2 = GGUF_MULTI_FACTOR_DICT_64.get(quantization)
    if v2 is not None:
        total = (v2 * total_params) / (64 * 1024 * 1024)
    
    # 使用GGUF_MULTI_FACTOR_DICT_COMBINE字典获取量化因子，并计算模型大小
    v3 = GGUF_MULTI_FACTOR_DICT_COMBINE.get(quantization)
    if v3 is not None:
        factors = v3
        if quantization == "q2_K":
            # 对q2_K量化方法的特殊处理
            total = (
                (total_params - other_param_q2k) * factors[1]
                + other_param_q2k * factors[0]
            ) / (64 * 1024 * 1024)
        else:
            # 对其他量化方法的处理
            total = (
                (total_params - other_v_down_params) * factors[1]
                + other_v_down_params * factors[0]
            ) / (64 * 1024 * 1024)
    
    # 返回计算得到的模型大小（单位：MB）
    return total
