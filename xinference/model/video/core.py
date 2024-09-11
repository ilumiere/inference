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
# 导入所需的模块和库
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from ...constants import XINFERENCE_CACHE_DIR
from ..core import CacheableModelSpec, ModelDescription
from ..utils import valid_model_revision
from .diffusers import DiffUsersVideoModel

# 定义最大尝试次数
MAX_ATTEMPTS = 3

# 设置日志记录器
logger = logging.getLogger(__name__)

# 初始化存储模型名称和版本的字典
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
# 初始化存储视频模型描述的字典
VIDEO_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
# 初始化存储内置视频模型的字典
BUILTIN_VIDEO_MODELS: Dict[str, "VideoModelFamilyV1"] = {}
# 初始化存储ModelScope视频模型的字典
MODELSCOPE_VIDEO_MODELS: Dict[str, "VideoModelFamilyV1"] = {}


def get_video_model_descriptions():
    """
    获取视频模型描述的深拷贝。

    返回:
        Dict[str, List[Dict]]: 视频模型描述的深拷贝。
    """
    import copy
    return copy.deepcopy(VIDEO_MODEL_DESCRIPTIONS)


class VideoModelFamilyV1(CacheableModelSpec):
    """
    视频模型家族V1类，继承自CacheableModelSpec。
    
    属性:
        model_family (str): 模型家族名称。
        model_name (str): 模型名称。
        model_id (str): 模型ID。
        model_revision (str): 模型版本。
        model_hub (str): 模型中心，默认为"huggingface"。
        model_ability (Optional[List[str]]): 模型能力列表。
        default_model_config (Optional[Dict[str, Any]]): 默认模型配置。
        default_generate_config (Optional[Dict[str, Any]]): 默认生成配置。
    """
    model_family: str
    model_name: str
    model_id: str
    model_revision: str
    model_hub: str = "huggingface"
    model_ability: Optional[List[str]]
    default_model_config: Optional[Dict[str, Any]]
    default_generate_config: Optional[Dict[str, Any]]


class VideoModelDescription(ModelDescription):
    """
    视频模型描述类，继承自ModelDescription。

    属性:
        _model_spec (VideoModelFamilyV1): 视频模型规格。
    """

    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: VideoModelFamilyV1,
        model_path: Optional[str] = None,
    ):
        """
        初始化VideoModelDescription实例。

        参数:
            address (Optional[str]): 模型地址。
            devices (Optional[List[str]]): 设备列表。
            model_spec (VideoModelFamilyV1): 模型规格。
            model_path (Optional[str]): 模型路径，默认为None。
        """
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        """
        将模型描述转换为字典格式。

        返回:
            Dict: 包含模型信息的字典。
        """
        return {
            "model_type": "video",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "model_family": self._model_spec.model_family,
            "model_revision": self._model_spec.model_revision,
            "model_ability": self._model_spec.model_ability,
        }

    def to_version_info(self):
        """
        生成并返回模型的版本信息。

        此方法用于获取视频模型的详细版本信息，包括模型版本、文件位置和缓存状态。

        方法流程：
        1. 检查模型路径是否已设置。
        2. 根据模型路径的设置情况，确定模型的缓存状态和文件位置。
        3. 构建并返回包含模型版本信息的字典列表。

        返回：
            List[Dict]: 包含以下键值对的字典列表：
                - 'model_version': 模型的名称（字符串）
                - 'model_file_location': 模型文件的位置（字符串）
                - 'cache_status': 模型的缓存状态（布尔值）

        注意：
        - 如果 self._model_path 为 None，表示使用默认路径，需要通过辅助函数获取缓存状态和目录。
        - 如果 self._model_path 不为 None，表示使用自定义路径，模型被视为已缓存。
        """
        if self._model_path is None:
            # 当模型路径未指定时，使用辅助函数获取缓存状态和目录
            is_cached = get_cache_status(self._model_spec)
            file_location = get_cache_dir(self._model_spec)
        else:
            # 当指定了模型路径时，认为模型已缓存，使用指定的路径
            is_cached = True
            file_location = self._model_path

        # 返回包含模型版本信息的字典列表
        return [
            {
                "model_version": self._model_spec.model_name,
                "model_file_location": file_location,
                "cache_status": is_cached,
            }
        ]

def generate_video_description(
    video_model: VideoModelFamilyV1,
) -> Dict[str, List[Dict]]:
    """
    生成视频模型的描述信息。

    此函数用于创建一个包含视频模型详细信息的字典。它主要用于模型管理和信息检索。

    参数:
        video_model (VideoModelFamilyV1): 视频模型实例。
            这个参数包含了视频模型的所有相关信息，如模型名称、版本等。

    函数流程:
    1. 创建一个默认字典 `res`，用于存储模型描述。
    2. 使用模型名称作为键，将模型的版本信息添加到字典中。
    3. 通过 VideoModelDescription 类创建模型描述实例，并调用其 to_version_info 方法获取版本信息。

    返回:
        Dict[str, List[Dict]]: 一个字典，其中：
            - 键是模型名称（字符串）
            - 值是包含模型版本信息的字典列表

    注意:
    - 使用 defaultdict(list) 确保即使键不存在，也可以直接追加值而不会引发 KeyError。
    - VideoModelDescription 初始化时传入 None 作为 address 和 devices，这表示不指定具体的地址和设备。
    - to_version_info() 方法返回的是一个列表，包含模型版本、文件位置和缓存状态等信息。
    """
    res = defaultdict(list)
    res[video_model.model_name].extend(
        VideoModelDescription(None, None, video_model).to_version_info()
    )
    return res


def match_diffusion(
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
) -> VideoModelFamilyV1:
    """
    匹配并返回指定的扩散模型。

    该函数根据提供的模型名称和下载中心，在不同的模型库中查找并返回匹配的视频模型实例。
    它支持从ModelScope和Hugging Face两个主要来源获取模型。

    参数:
        model_name (str): 要查找的模型名称。
        download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 
            指定的下载中心。默认为None，表示会按照预定义的顺序在所有支持的中心中查找。

    返回:
        VideoModelFamilyV1: 匹配的视频模型实例。

    异常:
        ValueError: 如果在指定的下载中心或所有支持的中心中都找不到指定的模型，则抛出此异常。

    函数流程:
    1. 导入必要的模块和常量。
    2. 根据指定的下载中心和模型名称，按照优先级顺序查找模型：
       a. 如果指定了ModelScope且模型存在，则从ModelScope返回模型。
       b. 如果指定了Hugging Face且模型存在，则从Hugging Face返回模型。
       c. 如果未指定下载中心，则先检查是否应该从ModelScope下载，如是则尝试从ModelScope返回模型。
       d. 最后，检查内置的Hugging Face模型列表。
    3. 如果在所有可能的来源中都找不到模型，则抛出ValueError异常。

    注意:
    - 函数使用logger.debug()记录找到模型的位置，有助于调试和跟踪。
    - BUILTIN_VIDEO_MODELS和MODELSCOPE_VIDEO_MODELS是预定义的模型字典，包含了可用的模型列表。
    """
    from ..utils import download_from_modelscope
    from . import BUILTIN_VIDEO_MODELS, MODELSCOPE_VIDEO_MODELS

    # 根据下载中心和模型名称匹配模型
    if download_hub == "modelscope" and model_name in MODELSCOPE_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in ModelScope.")
        return MODELSCOPE_VIDEO_MODELS[model_name]
    elif download_hub == "huggingface" and model_name in BUILTIN_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in Huggingface.")
        return BUILTIN_VIDEO_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in ModelScope.")
        return MODELSCOPE_VIDEO_MODELS[model_name]
    elif model_name in BUILTIN_VIDEO_MODELS:
        logger.debug(f"Video model {model_name} found in Huggingface.")
        return BUILTIN_VIDEO_MODELS[model_name]
    else:
        raise ValueError(
            f"Video model {model_name} not found, available"
            f"model list: {BUILTIN_VIDEO_MODELS.keys()}"
        )


def cache(model_spec: VideoModelFamilyV1):
    """
    缓存模型。

    参数:
        model_spec (VideoModelFamilyV1): 模型规格。

    返回:
        Any: 缓存操作的结果。
    """
    from ..utils import cache
    return cache(model_spec, VideoModelDescription)


def get_cache_dir(model_spec: VideoModelFamilyV1):
    """
    获取缓存目录。

    参数:
        model_spec (VideoModelFamilyV1): 模型规格。

    返回:
        str: 缓存目录的实际路径。
    """
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def get_cache_status(
    model_spec: VideoModelFamilyV1,
) -> bool:
    """
    获取视频模型的缓存状态。

    此函数用于检查指定视频模型的缓存是否有效。它首先确定缓存目录和元数据文件的路径，
    然后根据模型的来源（内置模型或ModelScope模型）检查缓存的有效性。

    参数:
        model_spec (VideoModelFamilyV1): 包含模型规格的对象，用于确定模型名称和其他属性。

    返回:
        bool: 如果缓存有效则返回True，否则返回False。

    函数流程:
    1. 获取模型的缓存目录和元数据文件路径。
    2. 提取模型名称。
    3. 检查模型是否同时存在于内置模型和ModelScope模型中：
       - 如果是，分别获取两个来源的模型规格，并检查任一来源的缓存是否有效。
       - 如果不是，直接检查给定模型规格的缓存有效性（通常用于单元测试）。

    注意:
    - 函数使用 valid_model_revision 辅助函数来验证模型版本的有效性。
    - 对于同时存在于多个来源的模型，只要有一个来源的缓存有效，就认为缓存状态有效。
    """
    # 获取模型的缓存目录
    cache_dir = get_cache_dir(model_spec)
    # 构建元数据文件的完整路径
    meta_path = os.path.join(cache_dir, "__valid_download")

    # 获取模型名称
    model_name = model_spec.model_name
    # 检查模型是否同时存在于内置模型和ModelScope模型中
    if model_name in BUILTIN_VIDEO_MODELS and model_name in MODELSCOPE_VIDEO_MODELS:
        # 获取内置模型和ModelScope模型的规格
        hf_spec = BUILTIN_VIDEO_MODELS[model_name]
        ms_spec = MODELSCOPE_VIDEO_MODELS[model_name]

        # 检查任一来源的缓存是否有效
        return any(
            [
                valid_model_revision(meta_path, hf_spec.model_revision),
                valid_model_revision(meta_path, ms_spec.model_revision),
            ]
        )
    else:  # 通常用于单元测试
        # 直接检查给定模型规格的缓存有效性
        return valid_model_revision(meta_path, model_spec.model_revision)

def create_video_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[DiffUsersVideoModel, VideoModelDescription]:
    """
    创建视频模型实例。

    此函数用于创建和初始化一个视频模型实例，并生成相应的模型描述。它处理模型规格的匹配、
    模型缓存的管理，以及模型实例和描述的创建。

    参数:
        subpool_addr (str): 子池地址，用于标识模型所在的计算资源位置。
        devices (List[str]): 可用于模型运行的设备列表。
        model_uid (str): 模型的唯一标识符，用于区分不同的模型实例。
        model_name (str): 模型的名称，用于匹配和加载正确的模型。
        download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 
            指定模型下载的来源平台。默认为None，表示使用默认来源。
        model_path (Optional[str]): 模型文件的本地路径。如果提供，将直接使用该路径加载模型。
        **kwargs: 其他关键字参数，用于传递给模型初始化函数的额外参数。

    返回:
        Tuple[DiffUsersVideoModel, VideoModelDescription]: 
            返回一个元组，包含：
            - DiffUsersVideoModel实例：初始化后的视频模型对象。
            - VideoModelDescription实例：包含模型详细信息的描述对象。

    函数流程:
    1. 使用match_diffusion函数匹配模型规格。
    2. 如果未提供model_path，则使用cache函数获取或下载模型文件。
    3. 断言确保model_path不为None。
    4. 创建DiffUsersVideoModel实例。
    5. 创建VideoModelDescription实例。
    6. 返回模型实例和描述的元组。

    注意:
    - 此函数依赖于外部函数match_diffusion和cache，这些函数应在同一模块中定义。
    - 函数使用assert语句确保模型路径有效，这可能在生产环境中需要更健壮的错误处理。
    """
    # 匹配模型规格
    model_spec = match_diffusion(model_name, download_hub)
    
    # 如果未提供模型路径，则缓存或下载模型
    if not model_path:
        model_path = cache(model_spec)
    
    # 确保模型路径不为None
    assert model_path is not None

    # 创建DiffUsersVideoModel实例
    model = DiffUsersVideoModel(
        model_uid,
        model_path,
        model_spec,
        **kwargs,
    )
    
    # 创建VideoModelDescription实例
    model_description = VideoModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    
    # 返回模型实例和描述
    return model, model_description
