# Copyright 2022-2024 XProbe Inc.
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
import io
import logging
import os
import tempfile
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ....constants import XINFERENCE_TENSORIZER_DIR
from ....device_utils import get_available_device

logger = logging.getLogger(__name__)

__all__ = [
    "get_tensorizer_dir",
    "check_tensorizer_integrity",
    "load_from_tensorizer",
    "save_to_tensorizer",
    "_load_pretrained_from_tensorizer",
    "_load_model_from_tensorizer",
    "_tensorizer_serialize_model",
    "_tensorizer_serialize_pretrained",
    "_file_is_non_empty",
]


def _filter_kwargs(kwargs):
    # 设置 trust_remote_code 的默认值为 True
    kwargs["trust_remote_code"] = kwargs.get("trust_remote_code", True)
    
    # 返回一个新的字典，只包含 "code_revision" 或 "trust_remote_code" 这两个键值对
    # 如果原字典中存在这些键的话
    return {
        k: v for k, v in kwargs.items() if k in ["code_revision", "trust_remote_code"]
    }


def _file_is_non_empty(
    path: str,
) -> bool:
    """
    检查指定路径的文件是否非空

    Args:
        path (str): 要检查的文件路径

    Returns:
        bool: 如果文件存在且非空返回True，否则返回False
    """
    try:
        # 使用os.stat()获取文件状态，并检查文件大小是否大于0
        return os.stat(path).st_size > 0
    except FileNotFoundError:
        # 如果文件不存在，返回False
        return False


def get_tensorizer_dir(model_path: str) -> str:
    """
    路径。让我给你一些具体的输入和输出例子：
    假设 XINFERENCE_TENSORIZER_DIR 的值是 "/home/user/xinference/tensorizer"

    输入:
    model_path = "/models/gpt2"

    输出: 
    "/home/user/xinference/tensorizer/gpt2"

    输入:
    model_path = "/path/to/models/bert-base-uncased/"

    输出:
    "/home/user/xinference/tensorizer/bert-base-uncased"

    Args:
        model_path (str): _description_

    Returns:
        str: _description_
    """
    # 获取模型路径的基本名称，去除末尾的斜杠
    model_dir = os.path.basename(model_path.rstrip("/"))
    # 返回tensorizer目录的完整路径
    # 使用XINFERENCE_TENSORIZER_DIR作为基础目录，并将model_dir添加到路径中
    return f"{XINFERENCE_TENSORIZER_DIR}/{model_dir}"


def check_tensorizer_integrity(
    model_path: str,
    components: Optional[List[str]] = None,
    model_prefix: Optional[str] = "model",
) -> bool:
    """
    
    Tensorizer的作用：
    Tensorizer是一种用于优化深度学习模型存储和加载的技术。
    它将模型参数（张量）序列化为一种高效的格式。
    主要目的是加快模型的加载速度，减少内存使用

    Args:
        model_path (str): _description_
        components (Optional[List[str]], optional): _description_. Defaults to None.
        model_prefix (Optional[str], optional): _description_. Defaults to "model".

    Returns:
        bool: _description_
    """
    # 获取tensorizer目录路径
    tensorizer_dir = get_tensorizer_dir(model_path)
    # 去除路径末尾的斜杠
    dir = tensorizer_dir.rstrip("/")
    # 构造模型张量文件的URI
    tensors_uri: str = f"{dir}/{model_prefix}.tensors"
    # iterate over components and get their paths
    paths = [tensors_uri]
    # 如果提供了组件列表
    # components 的内容：
    # 每个组件通常是一个字符串，代表组件的名称或标识符。
    # 这些可能包括但不限于：
    # 分词器（tokenizer）

    # 配置文件
    # 特定的模型层或子模块
    # 预处理或后处理组件
    # 组件文件的构造：

    # 对于每个组件，函数构造一个 ZIP 文件的 URI：
    # f"{tensorizer_dir.rstrip('/')}/{component}.zip"
    # 这表明每个组件都被期望以 ZIP 文件的形式存储。

    # 示例：

    # 假设 components = ["tokenizer", "config"]，那么函数会检查以下文件：
    # {tensorizer_dir}/model.tensors（主模型文件）
    # {tensorizer_dir}/tokenizer.zip
    # {tensorizer_dir}/config.zip
    
    if components is not None:
        # 遍历组件列表
        for component in components:
            # 构造每个组件的ZIP文件URI
            component_uri: str = f"{tensorizer_dir.rstrip('/')}/{component}.zip"
            # 将组件URI添加到路径列表
            paths.append(component_uri)
    # 检查所有路径对应的文件是否非空，返回结果
    return all(_file_is_non_empty(path) for path in paths)


def load_from_tensorizer(
    model_path: str,
    components: Optional[List[Tuple[str, Any, Dict[str, Any]]]] = None,
    model_class: Any = None,
    config_class: Any = None,
    model_prefix: Optional[str] = "model",
    **kwargs,
):
    # 过滤kwargs参数
    kwargs = _filter_kwargs(kwargs)
    try:
        # 尝试导入transformers中的AutoConfig和AutoModel
        from transformers import AutoConfig, AutoModel
    except ImportError:
        # 如果导入失败，准备错误信息和安装指南
        error_message = "Failed to import module 'transformers'"
        installation_guide = [
            "Please make sure 'transformers' is installed. ",
            "You can install it by `pip install transformers`\n",
        ]
        # 抛出ImportError异常，包含错误信息和安装指南
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    # 如果未指定model_class，则使用AutoModel
    model_class = model_class or AutoModel
    # 如果未指定config_class，则使用AutoConfig
    config_class = config_class or AutoConfig

    # 获取tensorizer目录路径
    tensorizer_dir = get_tensorizer_dir(model_path)
    logger.debug(f"Loading from tensorizer: {tensorizer_dir}")

    # 获取可用的设备（CPU或GPU）
    device = get_available_device()
    # 从tensorizer加载模型，并将其移动到指定设备上，设置为评估模式
    tensorizer_model = (
        _load_model_from_tensorizer(
            model_path,
            tensorizer_dir,
            model_class,
            config_class,
            model_prefix,
            device,
            **kwargs,
        )
        .to(device)
        .eval()
    )

    # 初始化tensorizer组件列表
    tensorizer_components = []

    # 如果指定了组件，则加载每个组件
    if components is not None:
        for component, component_class, kwargs in components:
            # 从tensorizer加载预训练的组件
            deserialized_component = _load_pretrained_from_tensorizer(
                component_class, tensorizer_dir, component, **kwargs
            )
            # 将反序列化的组件添加到列表中
            tensorizer_components.append(deserialized_component)

    # 返回加载的模型和所有组件
    return tensorizer_model, *tensorizer_components


def _load_pretrained_from_tensorizer(
    component_class: Any,
    tensorizer_dir: str,
    prefix: str,
    **kwargs,
):
    # 记录正在加载的组件信息
    logger.debug(f"Loading components from tensorizer: {component_class} {kwargs}")

    try:
        # 尝试导入tensorizer模块中的stream_io
        from tensorizer import stream_io
    except ImportError:
        # 如果导入失败，准备错误信息和安装指南
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        # 抛出ImportError异常，包含错误信息和安装指南
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    # 创建一个偏函数，用于以二进制读取模式打开流
    _read_stream = partial(stream_io.open_stream, mode="rb")

    # 记录正在从tensorizer加载预训练模型的信息
    logger.debug(f"Loading pretrained from tensorizer: {tensorizer_dir}")
    # 构造加载路径
    load_path: str = f"{tensorizer_dir.rstrip('/')}/{prefix}.zip"
    # 记录正在加载的文件路径
    logger.info(f"Loading {load_path}")
    with io.BytesIO() as downloaded:
        # Download to a BytesIO object first, because ZipFile doesn't play nice
        # with streams that don't fully support random access
        with _read_stream(load_path) as stream:
            downloaded.write(stream.read())
        # 将文件指针移到开始位置
        downloaded.seek(0)
        # 使用ZipFile打开下载的内容，并创建一个临时目录
        with zipfile.ZipFile(
            downloaded, mode="r"
        ) as file, tempfile.TemporaryDirectory() as directory:
            # 解压文件到临时目录
            file.extractall(path=directory)
            # 从解压后的目录加载预训练模型，并返回
            return component_class.from_pretrained(
                directory, cache_dir=None, local_files_only=True, **kwargs
            )


def _load_model_from_tensorizer(
    model_path: str,
    tensorizer_dir: str,
    model_class,
    config_class,
    model_prefix: Optional[str] = "model",
    device=None,
    dtype=None,
    **kwargs,
):
    """
    从tensorizer加载模型。

    参数:
    model_path (str): 模型路径
    tensorizer_dir (str): tensorizer目录
    model_class: 模型类
    config_class: 配置类
    model_prefix (Optional[str]): 模型前缀，默认为"model"
    device: 设备，如"cpu"或"cuda"
    dtype: 数据类型
    **kwargs: 其他关键字参数

    返回:
    加载的模型实例
    """
    # 记录日志，显示正在从tensorizer加载模型
    logger.debug(f"Loading model from tensorizer: {tensorizer_dir} {kwargs}")

    # 检查设备是否被指定，如果没有则抛出异常
    if device is None:
        raise ValueError("device must be specified")

    # 导入时间模块，用于性能计时
    import time

    # 尝试导入torch模块，如果失败则提供安装指南
    try:
        import torch
    except ImportError:
        error_message = "Failed to import module 'torch'"
        installation_guide = [
            "Please make sure 'torch' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    # 尝试从transformers导入PretrainedConfig，如果失败则提供安装指南
    try:
        from transformers import PretrainedConfig
    except ImportError:
        error_message = "Failed to import module 'transformers'"
        installation_guide = [
            "Please make sure 'transformers' is installed. ",
            "You can install it by `pip install transformers`\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    # 尝试从tensorizer导入必要的模块，如果失败则提供安装指南
    try:
        from tensorizer import TensorDeserializer, stream_io, utils
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    # 如果未指定模型前缀，则使用默认值"model"
    if model_prefix is None:
        model_prefix = "model"

    # 构建tensorizer文件路径
    dir: str = tensorizer_dir.rstrip("/")
    tensors_uri: str = f"{dir}/{model_prefix}.tensors"

    # 定义读取流的函数
    # partial 函数：
    # partial 来自 Python 的 functools 模块。
    # 它用于创建一个新的函数，这个新函数是原函数的部分应用（partial application）。
    # 部分应用意味着预先设置了某些参数，创建一个新的、更专用的函数。
    # 这个新函数等同于 stream_io.open_stream 但默认 mode 参数设置为 "rb"。
    # 使用时，只需提供其他必要参数（如文件路径或 URL），而不需要每次都指定 mode="rb"
    _read_stream = partial(stream_io.open_stream, mode="rb")

    # 加载配置
    if config_class is None:
        config_loader = model_class.load_config
    else:
        config_loader = config_class.from_pretrained
    try:
        # 尝试加载配置并返回未使用的参数
        config, _ = config_loader(model_path, return_unused_kwargs=True, **kwargs)
        # 如果配置是 PretrainedConfig 的实例，则启用梯度检查
        if isinstance(config, PretrainedConfig):
            config.gradient_checkpointing = True
    except ValueError:
        # 如果上述方法失败，直接加载配置
        # 为什么要这样做：
        # 兼容性：不是所有的配置加载器都支持 return_unused_kwargs 参数。
        # 向后兼容：这种方法允许代码与旧版本的模型或配置类一起工作。
        # 错误恢复：如果因为额外参数导致加载失败，第二次尝试可能会成功。
        # 4. 可能的场景：
        # 新版本的配置加载器支持 return_unused_kwargs，但旧版本不支持。
        # 某些特定模型的配置类可能不完全遵循标准接口。
        config = config_loader(model_path, **kwargs)

    # 创建模型实例，禁用权重初始化
    # 性能优化：
    # 禁用权重初始化可以显著加快模型加载速度。
    # 对于大型模型，初始化权重可能需要相当长的时间。
    # 内存效率：
    # 禁用初始化可以减少内存使用，因为不需要为随机初始化的权重分配额外的内存。
    # 避免不必要的计算：
    # 在这种情况下，模型的权重稍后会从预训练的张量中加载。
    # 初始化权重只是为了立即被覆盖，这是不必要的计算。
    with utils.no_init_or_tensor():
        # 获取模型加载方法，如果 model_class 有 from_config 方法，则使用它，否则使用 model_class 本身
        model_loader = getattr(model_class, "from_config", model_class)
        # 使用配置和参数创建模型实例
        model = model_loader(config, **kwargs)

    # 检查是否使用CUDA
    is_cuda: bool = torch.device(device).type == "cuda"
    # 获取当前内存使用情况
    ram_usage = utils.get_mem_usage()
    # 记录开始加载的日志
    logger.info(f"Loading {tensors_uri}, {ram_usage}")
    # 记录开始加载的时间
    begin_load = time.perf_counter()

    # 加载模型张量
    with _read_stream(tensors_uri) as tensor_stream, TensorDeserializer(
        tensor_stream, device=device, dtype=dtype, plaid_mode=is_cuda
    ) as tensor_deserializer:
        # 将张量加载到模型中
        tensor_deserializer.load_into_module(model)
        # 计算加载时间
        tensor_load_s = time.perf_counter() - begin_load
        # 获取读取的总字节数
        bytes_read: int = tensor_deserializer.total_bytes_read

    # 计算加载速率
    rate_str = utils.convert_bytes(bytes_read / tensor_load_s)
    # 转换读取的字节数为可读格式
    tensors_sz = utils.convert_bytes(bytes_read)
    # 记录加载完成的日志，包括加载时间、大小和速率
    logger.info(
        f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
        f"{tensors_sz} @ {rate_str}/s, {utils.get_mem_usage()}"
    )

    # 返回加载的模型
    return model


def save_to_tensorizer(
    model_path: str,
    model,
    components: Optional[List[Tuple[str, Any]]] = None,
    model_prefix: Optional[str] = "model",
    force: Optional[bool] = False,
    **kwargs,
):
    # 过滤kwargs，移除不需要的参数
    kwargs = _filter_kwargs(kwargs)
    
    # 序列化主模型
    _tensorizer_serialize_model(model_path, model, model_prefix, force, **kwargs)

    # 如果提供了额外的组件，则逐个序列化
    if components is not None:
        for component_prefix, component in components:
            # 序列化每个预训练的组件
            _tensorizer_serialize_pretrained(model_path, component, component_prefix)

def _tensorizer_serialize_model(
    model_path: str,
    model,
    model_prefix: Optional[str] = "model",
    force: Optional[bool] = False,
    **kwargs,
):
    """
    将模型序列化为张量文件。

    参数:
    model_path (str): 模型保存路径
    model: 要序列化的模型
    model_prefix (Optional[str]): 模型前缀，默认为"model"
    force (Optional[bool]): 是否强制序列化，默认为False
    **kwargs: 额外的关键字参数

    功能:
    - 导入必要的tensorizer模块
    - 构建张量文件路径
    - 检查缓存是否存在，如存在则跳过序列化
    - 将模型序列化为张量文件
    """
    try:
        from tensorizer import TensorSerializer, stream_io
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    # 获取tensorizer目录
    tensorizer_dir = get_tensorizer_dir(model_path)
    # 构建张量文件路径
    tensor_path: str = f"{tensorizer_dir}/{model_prefix}.tensors"

    # 定义写入流函数
    _write_stream = partial(stream_io.open_stream, mode="wb+")

    # 检查缓存是否存在
    if os.path.exists(tensor_path):
        logger.info(f"缓存 {tensor_path} 已存在，跳过模型序列化")
        return

    # 开始序列化模型
    logger.info(f"正在将张量写入 {tensor_path}")
    with _write_stream(tensor_path) as f:
        serializer = TensorSerializer(f)
        serializer.write_module(model, include_non_persistent_buffers=False)
        serializer.close()

    logger.info(f"模型序列化完成: {tensor_path}")

def _tensorizer_serialize_pretrained(
    model_path: str, component, prefix: str = "pretrained"
):
    """
    将预训练组件序列化为张量文件。

    参数:
    model_path (str): 模型保存路径
    component: 要序列化的预训练组件
    prefix (str): 保存文件的前缀，默认为"pretrained"

    功能:
    - 导入必要的tensorizer模块
    - 构建保存路径
    - 检查缓存是否存在，如存在则跳过序列化
    - 将预训练组件序列化为zip文件
    """
    try:
        from tensorizer import stream_io
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    # 获取tensorizer目录并构建保存路径
    tensorizer_dir = get_tensorizer_dir(model_path)
    save_path: str = f"{tensorizer_dir.rstrip('/')}/{prefix}.zip"

    # 检查缓存是否存在
    if os.path.exists(save_path):
        logger.info(f"Cache {save_path} exists, skip tensorizer serialize pretrained")
        return

    logger.info(f"正在将组件写入 {save_path}")
    _write_stream = partial(stream_io.open_stream, mode="wb+")

    # 使用ZipFile将预训练组件序列化为zip文件
    with _write_stream(save_path) as stream, zipfile.ZipFile(
        stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=5
    ) as file, tempfile.TemporaryDirectory() as directory:
        # 如果组件有save_pretrained方法，则调用它
        if hasattr(component, "save_pretrained"):
            component.save_pretrained(directory)
        else:
            logger.warning("该组件没有'save_pretrained'方法。")
        # 将临时目录中的所有文件写入zip文件
        for path in Path(directory).iterdir():
            file.write(filename=path, arcname=path.name)

    logger.info(f"Tensorizer serialize pretrained done: {save_path}")
