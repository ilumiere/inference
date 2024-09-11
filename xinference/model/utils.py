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

import functools
import gc
import inspect
import json
import logging
import os
import random
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import huggingface_hub
import numpy as np
import torch

from ..constants import XINFERENCE_CACHE_DIR, XINFERENCE_ENV_MODEL_SRC
from ..device_utils import empty_cache, get_available_device, is_device_available
from .core import CacheableModelSpec

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 3
IS_NEW_HUGGINGFACE_HUB: bool = huggingface_hub.__version__ >= "0.23.0"


def is_locale_chinese_simplified() -> bool:
    import locale

    try:
        lang, _ = locale.getdefaultlocale()
        return lang == "zh_CN"
    except:
        return False


def download_from_modelscope() -> bool:
    if os.environ.get(XINFERENCE_ENV_MODEL_SRC):
        return os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "modelscope"
    elif is_locale_chinese_simplified():
        return True
    else:
        return False


def download_from_csghub() -> bool:
    if os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "csghub":
        return True
    return False


def symlink_local_file(path: str, local_dir: str, relpath: str) -> str:
    from huggingface_hub.file_download import _create_symlink

    # cross-platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*relpath.split("/"))
    if os.name == "nt":
        if relative_filename.startswith("..\\") or "\\..\\" in relative_filename:
            raise ValueError(
                f"Invalid filename: cannot handle filename '{relative_filename}' on Windows. Please ask the repository"
                " owner to rename this file."
            )
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    local_dir_filepath = os.path.join(local_dir, relative_filename)
    if (
        Path(os.path.abspath(local_dir))
        not in Path(os.path.abspath(local_dir_filepath)).parents
    ):
        raise ValueError(
            f"Cannot copy file '{relative_filename}' to local dir '{local_dir}': file would not be in the local"
            " directory."
        )

    os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)
    real_blob_path = os.path.realpath(path)
    _create_symlink(real_blob_path, local_dir_filepath, new_blob=False)
    return local_dir_filepath


def create_symlink(download_dir: str, cache_dir: str):
    """
    为下载目录中的所有文件在缓存目录中创建符号链接。
    
    
    os.walk(download_dir):
    这个函数遍历 download_dir 目录及其所有子目录。
    它返回一个生成器，每次迭代提供三个值：
    
    - subdir: 当前正在遍历的目录的路径
    - dirs: 当前目录中的子目录列表
    - files: 当前目录中的文件列表
    
    os.path.join(subdir, file):
    
    将当前子目录路径 subdir 和文件名 file 连接起来。
    这会生成文件的完整路径。
    
    例如：如果 subdir 是 "/download/models" 且 file 是 "config.json"，
    结果将是 "/download/models/config.json"。
    
    os.path.relpath(os.path.join(subdir, file), download_dir):
    os.path.relpath() 计算一个路径相对于另一个路径的相对路径。
    
    这里计算的是文件路径相对于 download_dir 的相对路径。
    例如：如果文件路径是： "/download/models/subfolder/config.json"
    且 download_dir 是 "/download/models"
    结果将是 "subfolder/config.json"
    
    symlink_local_file(os.path.join(subdir, file), cache_dir, relpath):
    第一个参数是源文件的完整路径。
    第二个参数 cache_dir 是目标缓存目录。
    第三个参数 relpath 是文件相对于下载目录的路径，用于在缓存目录中重建相同的目录结构。
    这种方法的优点：
    1. 保持目录结构：在缓存目录中重建与原始下载目录相同的文件结构。
    2. 相对路径处理：使用相对路径确保在不同环境中的一致性。
    3. 灵活性：可以处理任意深度的目录结构。
    例如，如果原始结构是：

    /download/models/
    ├── model.bin
    └── config/
        └── config.json
    

    在缓存目录中会创建相应的符号链接：
    /cache/dir/
        ├── model.bin -> /download/models/model.bin
        └── config/
            └── config.json -> /download/models/config/config.json
    参数:
    download_dir (str): 下载文件的源目录
    cache_dir (str): 创建符号链接的目标缓存目录

    这个函数会遍历下载目录中的所有文件，并在缓存目录中创建对应的符号链接。
    """
    # 遍历下载目录中的所有子目录和文件
    for subdir, dirs, files in os.walk(download_dir):
        for file in files:
            # 计算文件相对于下载目录的相对路径
            relpath = os.path.relpath(os.path.join(subdir, file), download_dir)
            # 在缓存目录中创建符号链接
            symlink_local_file(os.path.join(subdir, file), cache_dir, relpath)


def retry_download(
    download_func: Callable,
    model_name: str,
    model_info: Optional[Dict],
    *args,
    **kwargs,
):
    """
    尝试多次下载模型，如果失败则抛出异常。

    参数:
    download_func: 下载函数
    model_name: 模型名称
    model_info: 模型信息字典
    *args, **kwargs: 传递给下载函数的参数

    返回:
    下载函数的返回值

    异常:
    RuntimeError: 如果多次尝试后仍然下载失败
    """
    last_ex = None
    for current_attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return download_func(*args, **kwargs)
        except Exception as e:
            remaining_attempts = MAX_ATTEMPTS - current_attempt
            last_ex = e
            logger.debug(
                "下载失败: %s, 下载函数: %s, 下载参数: %s, 关键字参数: %s",
                e,
                download_func,
                args,
                kwargs,
            )
            logger.warning(
                f"第 {current_attempt} 次尝试失败。剩余尝试次数: {remaining_attempts}"
            )

    else:
        # 所有尝试都失败后，准备抛出异常
        model_size = (
            model_info.pop("model_size", None) if model_info is not None else None
        )
        model_format = (
            model_info.pop("model_format", None) if model_info is not None else None
        )
        if model_size is not None or model_format is not None:  # LLM models
            raise RuntimeError(
                f"Failed to download model '{model_name}' "
                f"(size: {model_size}, format: {model_format}) "
                f"after multiple retries"
            ) from last_ex
        else:  # 嵌入模型
            raise RuntimeError(
                f"Failed to download model '{model_name}' after multiple retries"
            ) from last_ex


def valid_model_revision(
    meta_path: str,
    expected_model_revision: Optional[str],
    expected_model_hub: Optional[str] = None,
) -> bool:
    """
    验证模型版本是否有效。

    参数:
    meta_path (str): 元数据文件的路径
    expected_model_revision (Optional[str]): 预期的模型版本
    expected_model_hub (Optional[str]): 预期的模型仓库，默认为None

    返回:
    bool: 如果模型版本有效则返回True，否则返回False
    """
    # 检查元数据文件是否存在
    if not os.path.exists(meta_path):
        return False
    
    with open(meta_path, "r") as f:
        try:
            # 尝试加载JSON格式的元数据
            meta_data = json.load(f)
        except JSONDecodeError:  # legacy meta file for embedding models
            logger.debug("Legacy meta file detected.")
            return True

        # 根据不同类型的模型获取实际的版本信息
        if "model_revision" in meta_data:  # 嵌入模型和图像模型
            real_revision = meta_data["model_revision"]
        elif "revision" in meta_data:  # 大语言模型
            real_revision = meta_data["revision"]
        else:
            # 如果没有找到版本信息，记录警告并返回False
            logger.warning(
                f"在 `__valid_download` 文件中没有找到 `revision` 信息。"
            )
            return False
        
        # 检查模型仓库是否匹配
        # meta_data: 这是一个字典，包含了模型的元数据信息。
        # .get() 方法：
        # 这是字典的一个方法，用于安全地获取字典中的值。
        # 它接受两个参数：
        # 1. 要查找的键（key）
        # 2. 默认值（可选）：如果键不存在，则返回这个默认值

        if expected_model_hub is not None and expected_model_hub != meta_data.get(
            "model_hub", "huggingface"
        ):
            logger.info("使用来自不同仓库的模型缓存。")
            return True
        else:
            # 比较实际版本和预期版本
            return real_revision == expected_model_revision


def get_cache_dir(model_spec: Any) -> str:
    """
    获取模型的缓存目录。

    参数:
    model_spec (Any): 模型规格对象，包含模型名称。

    返回:
    str: 模型缓存目录的绝对路径。

    说明:
    - 使用 os.path.join 将 XINFERENCE_CACHE_DIR 和模型名称组合。
    - 使用 os.path.realpath 获取结果路径的规范化绝对路径。
    """
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def is_model_cached(model_spec: Any, name_to_revisions_mapping: Dict):
    """
    检查模型是否已缓存。

    参数:
    model_spec (Any): 模型规格对象，包含模型名称。
    name_to_revisions_mapping (Dict): 模型名称到版本号的映射。

    返回:
    bool: 如果模型已缓存则返回True，否则返回False

    说明:
    - 获取模型缓存目录
    - 构建元数据文件路径
    - 获取模型名称对应的所有版本
    - 如果当前模型版本不在列表中，添加它（通常用于单元测试）
    - 检查是否有任何一个版本的模型是有效的
    """
    # 获取模型缓存目录
    cache_dir = get_cache_dir(model_spec)
    # 构建元数据文件路径
    meta_path = os.path.join(cache_dir, "__valid_download")
    # 获取模型名称对应的所有版本
    revisions = name_to_revisions_mapping[model_spec.model_name]
    # 如果当前模型版本不在列表中，添加它（通常用于单元测试）
    if model_spec.model_revision not in revisions:  # Usually for UT
        revisions.append(model_spec.model_revision)
    # 检查是否有任何一个版本的模型是有效的
    return any([valid_model_revision(meta_path, revision) for revision in revisions])


def is_valid_model_name(model_name: str) -> bool:
    """
    检查模型名称是否有效。

    参数:
    model_name (str): 要检查的模型名称

    返回:
    bool: 如果模型名称有效则返回True，否则返回False

    说明:
    - 模型名称不能为空
    - 模型名称不能包含以下特殊字符: +, /, ?, %, #, &, =, 以及空白字符
    """
    import re

    # 检查模型名称是否为空
    if len(model_name) == 0:
        return False

    # check if contains +/?%#&=\s
    # 使用正则表达式检查模型名称是否包含非法字符
    # ^[^+\/?%#&=\s]*$ 表示:
    # ^ 开始匹配
    # [^...] 匹配不在方括号内的任何字符
    # * 匹配前面的模式零次或多次
    # $ 结束匹配
    
    # 2. 特殊字符检查：
    # 模型名称不能包含以下特殊字符：
    # + (加号)
    # / (斜杠)
    # ? (问号)
    # % (百分号)
    # # (井号)
    # & (和号)
    # = (等号)
    # 模型名称不能包含任何空白字符（如空格、制表符、换行符等）
    return re.match(r"^[^+\/?%#&=\s]*$", model_name) is not None


def parse_uri(uri: str) -> Tuple[str, str]:
    """
    解析给定的URI，返回其方案（scheme）和路径。

    参数:
    uri (str): 要解析的URI字符串


    本地文件：  
    ("file", "/path/to/local/file.txt")

    网络文件：
    ("https", "example.com/resource/path")
    
    本地文件（Windows）：
    ("file", "C:\\Users\\Username\\Documents\\file.txt")
    
    通配符本地路径模式：
    ("file", "/path/to/files/*.txt")

    返回:
    Tuple[str, str]: 包含方案和路径的元组

    说明:
    - 如果URI是本地文件路径，返回 ("file", 路径)
    - 如果URI是URL，返回 (方案, 路径)
    - 对于Windows系统的本地路径，特殊处理方案
    """
    import glob
    from urllib.parse import urlparse

    # 检查URI是否为本地文件或匹配的文件模式
    # 模式匹配：
    # 它支持使用通配符，如 *（匹配任意数量的字符）和 ?（匹配单个字符）。
    # 在这段代码中的作用：
    # 它用来检查 uri 是否匹配任何实际存在的文件或目录。
    # 如果 uri 包含通配符，glob.glob(uri) 会尝试找到所有匹配的文件。
    # 使用示例：
    # 如果 uri 是 "/path/to/file*.txt"，glob.glob(uri) 会返回所有在 /path/to/ 目录下以 "file" 开头、".txt" 结尾的文件路径列表。

    

    if os.path.exists(uri) or glob.glob(uri):
        return "file", uri
    else:
        # 解析URI
        parsed = urlparse(uri)
        scheme = parsed.scheme
        path = parsed.netloc + parsed.path
        # 处理Windows系统的本地路径
        if parsed.scheme == "" or len(parsed.scheme) == 1:  # len == 1 for windows
            scheme = "file"
        return scheme, path


def is_valid_model_uri(model_uri: Optional[str]) -> bool:
    """
    检查给定的模型URI是否有效。

    参数:
    model_uri (Optional[str]): 要检查的模型URI

    返回:
    bool: 如果URI有效则返回True，否则返回False

    说明:
    - 如果model_uri为None或空字符串，返回False
    - 对于文件URI，检查路径是否为绝对路径且文件存在
    - 对于其他类型的URI，目前默认返回True（待实现其他方案的处理）
    """
    if not model_uri:
        return False

    src_scheme, src_root = parse_uri(model_uri)

    # 检查URI是否为文件URI本地文件需要是绝对路径 
    if src_scheme == "file":
        if not os.path.isabs(src_root):
            raise ValueError(f"Model URI cannot be a relative path: {model_uri}")
        return os.path.exists(src_root)
    else:
        # TODO: 处理其他URI方案。
        return True


def cache_from_uri(model_spec: CacheableModelSpec) -> str:
    """
    从给定的URI缓存模型。

    参数:
    model_spec (CacheableModelSpec): 包含模型规格的对象

    返回:
    str: 缓存目录的路径

    说明:
    - 首先检查缓存目录是否已存在，如果存在则直接返回
    - 解析模型URI，获取scheme和路径
    - 对于文件URI，创建符号链接到缓存目录
    - 目前仅支持文件URI，其他scheme会抛出异常

    符号链接的作用：
    原始位置：/home/user/models/my_model
    缓存位置：/path/to/xinference/cache/my_model -> 指向 /home/user/models/my_model
    结果：
    函数返回新的缓存目录路径。
    系统可以通过这个标准化的路径访问模型，而不需要知道原始模型的实际存储位置。

    统一管理：所有模型都可以通过 XINFERENCE_CACHE_DIR 访问。
    节省空间：不需要复制大型模型文件。
    灵活性：原始模型可以存储在任何位置。
    版本控制：可以轻松切换不同版本的模型

    """
    # 构建缓存目录路径
    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    )
    # 如果缓存目录已存在，直接返回
    if os.path.exists(cache_dir):
        logger.info("cache %s exists", cache_dir)
        return cache_dir

    # 确保model_uri不为None
    assert model_spec.model_uri is not None
    # 解析URI
    src_scheme, src_root = parse_uri(model_spec.model_uri)
    # 移除路径末尾的斜杠
    if src_root.endswith("/"):
        src_root = src_root[:-1]

    # 处理文件URI
    if src_scheme == "file":
        # 确保是绝对路径
        if not os.path.isabs(src_root):
            raise ValueError(
                f"Model URI cannot be a relative path: {model_spec.model_uri}"
            )
        # 创建缓存目录
        os.makedirs(XINFERENCE_CACHE_DIR, exist_ok=True)
        # 创建符号链接
        os.symlink(src_root, cache_dir, target_is_directory=True)
        return cache_dir
    else:
        # 不支持的URI scheme
        raise ValueError(f"Unsupported URL scheme: {src_scheme}")


def cache(model_spec: CacheableModelSpec, model_description_type: type):
    """
    缓存模型并返回缓存目录路径。

    参数:
    model_spec: CacheableModelSpec - 可缓存的模型规格
    model_description_type: type - 模型描述类型

    返回:
    str - 缓存目录路径

    流程:
    1. 如果模型有URI，从URI缓存
    2. 否则，创建或使用现有的缓存目录
    3. 检查模型版本是否有效
    4. 根据模型来源（ModelScope或Hugging Face）下载模型
    5. 创建符号链接并保存元数据
    """
    # 检查模型是否有URI，如果有则从URI缓存
    if (
        hasattr(model_spec, "model_uri")
        and getattr(model_spec, "model_uri", None) is not None
    ):
        logger.info(f"Model caching from URI: {model_spec.model_uri}")
        return cache_from_uri(model_spec=model_spec)

    # 创建或获取缓存目录
    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # 检查模型版本是否有效
    meta_path = os.path.join(cache_dir, "__valid_download")
    if valid_model_revision(meta_path, model_spec.model_revision, model_spec.model_hub):
        return cache_dir

    # 根据模型来源下载模型
    from_modelscope: bool = model_spec.model_hub == "modelscope"
    if from_modelscope:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download

        download_dir = retry_download(
            ms_download,
            model_spec.model_name,
            None,
            model_spec.model_id,
            revision=model_spec.model_revision,
        )
        create_symlink(download_dir, cache_dir)
    else:
        from huggingface_hub import snapshot_download as hf_download

        use_symlinks = {}
        if not IS_NEW_HUGGINGFACE_HUB:
            use_symlinks = {"local_dir_use_symlinks": True, "local_dir": cache_dir}
        download_dir = retry_download(
            hf_download,
            model_spec.model_name,
            None,
            model_spec.model_id,
            revision=model_spec.model_revision,
            **use_symlinks,
        )
        if IS_NEW_HUGGINGFACE_HUB:
            create_symlink(download_dir, cache_dir)
    
    # 保存模型元数据
    with open(meta_path, "w") as f:
        import json

        desc = model_description_type(None, None, model_spec)
        json.dump(desc.to_dict(), f)
    
    return cache_dir


def patch_trust_remote_code():
    """sentence-transformers calls transformers without the trust_remote_code=True, some embedding
    models will fail to load, e.g. jina-embeddings-v2-base-en

    :return:
    """
    try:
        from transformers.dynamic_module_utils import resolve_trust_remote_code
    except ImportError:
        logger.error("Patch transformers trust_remote_code failed.")
    else:

        def _patched_resolve_trust_remote_code(*args, **kwargs):
            logger.info("Patched resolve_trust_remote_code: %s %s", args, kwargs)
            return True

        if (
            resolve_trust_remote_code.__code__
            != _patched_resolve_trust_remote_code.__code__
        ):
            resolve_trust_remote_code.__code__ = (
                _patched_resolve_trust_remote_code.__code__
            )


def select_device(device):
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
        )

    if device == "auto":
        return get_available_device()
    else:
        if not is_device_available(device):
            raise ValueError(f"{device} is unavailable in your environment")

    return device


def convert_float_to_int_or_str(model_size: float) -> Union[int, str]:
    """convert float to int or string

    if float can be presented as int, convert it to int, otherwise convert it to string
    """
    if int(model_size) == model_size:
        return int(model_size)
    else:
        return str(model_size)


def ensure_cache_cleared(func: Callable):
    assert not inspect.iscoroutinefunction(func) and not inspect.isasyncgenfunction(
        func
    )
    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            for obj in func(*args, **kwargs):
                yield obj
            gc.collect()
            empty_cache()

    else:

        @functools.wraps(func)
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                gc.collect()
                empty_cache()

    return inner


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
