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
# 导入所需的模块和类型
import collections.abc
import logging
import os
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

from ...constants import XINFERENCE_CACHE_DIR
from ...types import PeftModelConfig
from ..core import CacheableModelSpec, ModelDescription
from ..utils import valid_model_revision
from .stable_diffusion.core import DiffusionModel

# 设置最大尝试次数
MAX_ATTEMPTS = 3

# 配置日志记录器
logger = logging.getLogger(__name__)

# 定义全局变量和数据结构
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
IMAGE_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
BUILTIN_IMAGE_MODELS: Dict[str, "ImageModelFamilyV1"] = {}
MODELSCOPE_IMAGE_MODELS: Dict[str, "ImageModelFamilyV1"] = {}

def get_image_model_descriptions():
    """
    获取图像模型描述的深拷贝。

    返回:
        Dict[str, List[Dict]]: 包含所有图像模型描述的深拷贝字典。

    这个函数通过深拷贝确保返回的数据不会影响原始数据结构，
    保护了全局变量 IMAGE_MODEL_DESCRIPTIONS 的完整性。
    """
    import copy
    return copy.deepcopy(IMAGE_MODEL_DESCRIPTIONS)

class ImageModelFamilyV1(CacheableModelSpec):
    """
    定义图像模型家族的规格。

    属性:
        model_family (str): 模型所属的家族名称。
        model_name (str): 模型的名称。
        model_id (str): 模型的唯一标识符。
        model_revision (str): 模型的版本或修订号。
        model_hub (str): 模型所在的仓库，默认为 "huggingface"。
        model_ability (Optional[List[str]]): 模型支持的能力列表。
        controlnet (Optional[List["ImageModelFamilyV1"]]): 与模型关联的 ControlNet 模型列表。

    这个类继承自 CacheableModelSpec，用于定义可缓存的图像模型规格。
    它包含了描述图像模型的各种属性，如模型家族、名称、ID、版本等。
    """
    model_family: str
    model_name: str
    model_id: str
    model_revision: str
    model_hub: str = "huggingface"
    model_ability: Optional[List[str]]
    controlnet: Optional[List["ImageModelFamilyV1"]]


class ImageModelDescription(ModelDescription):
    """
    图像模型描述类，用于提供模型的详细信息。

    这个类继承自 ModelDescription，专门用于描述图像模型。
    它封装了模型的地址、设备、规格等信息，并提供了将这些信息转换为字典格式的方法。

    属性:
        _model_spec (ImageModelFamilyV1): 模型的规格信息。

    方法:
        __init__: 初始化图像模型描述实例。
        to_dict: 将模型描述转换为字典格式。
        to_version_info: 获取模型版本信息。
    """

    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: ImageModelFamilyV1,
        model_path: Optional[str] = None,
    ):
        """
        初始化 ImageModelDescription 实例。

        参数:
            address (Optional[str]): 模型的地址。
            devices (Optional[List[str]]): 模型可用的设备列表。
            model_spec (ImageModelFamilyV1): 模型的规格信息。
            model_path (Optional[str]): 模型文件的路径，默认为 None。

        这个方法调用父类的初始化方法，并额外存储模型规格信息。
        """
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        """
        将模型描述转换为字典格式。

        返回:
            Dict: 包含模型详细信息的字典。

        这个方法将模型的各种属性整合成一个字典，便于序列化和数据传输。
        它特别处理了 controlnet 属性，确保正确地表示相关的 ControlNet 模型信息。
        """
        if self._model_spec.controlnet is not None:
            controlnet = [cn.dict() for cn in self._model_spec.controlnet]
        else:
            controlnet = self._model_spec.controlnet
        return {
            "model_type": "image",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "model_family": self._model_spec.model_family,
            "model_revision": self._model_spec.model_revision,
            "model_ability": self._model_spec.model_ability,
            "controlnet": controlnet,
        }

    def to_version_info(self):
        """
        获取模型的版本信息。

        返回:
            List[Dict]: 包含模型版本详细信息的字典列表。

        这个方法生成模型的版本信息，包括模型版本、文件位置、缓存状态等。
        如果模型包含 ControlNet，会为每个 ControlNet 模型生成单独的版本信息。
        """
        from .utils import get_model_version

        # 确定模型的缓存状态和文件位置
        if self._model_path is None:
            is_cached = get_cache_status(self._model_spec)
            file_location = get_cache_dir(self._model_spec)
        else:
            is_cached = True
            file_location = self._model_path

        # 处理没有 ControlNet 的情况
        if self._model_spec.controlnet is None:
            return [
                {
                    "model_version": get_model_version(self._model_spec, None),
                    "model_file_location": file_location,
                    "cache_status": is_cached,
                    "controlnet": "zoe-depth",
                }
            ]
        else:
            # 处理有 ControlNet 的情况
            res = []
            for cn in self._model_spec.controlnet:
                res.append(
                    {
                        "model_version": get_model_version(self._model_spec, cn),
                        "model_file_location": file_location,
                        "cache_status": is_cached,
                        "controlnet": cn.model_name,
                    }
                )
            return res


def generate_image_description(
    image_model: ImageModelFamilyV1,
) -> Dict[str, List[Dict]]:
    """
    生成图像模型的描述信息。

    参数:
        image_model (ImageModelFamilyV1): 图像模型的规格对象。

    返回:
        Dict[str, List[Dict]]: 包含模型名称和版本信息的字典。

    功能:
        1. 创建一个默认字典来存储模型描述。
        2. 使用模型名称作为键，将模型的版本信息列表作为值添加到字典中。
        3. 返回生成的描述字典。
    """
    res = defaultdict(list)
    res[image_model.model_name].extend(
        ImageModelDescription(None, None, image_model).to_version_info()
    )
    return res


def match_diffusion(
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
) -> ImageModelFamilyV1:
    """
    匹配指定名称的扩散模型。

    参数:
        model_name (str): 要匹配的模型名称。
        download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 
            指定的下载源，默认为None。

    返回:
        ImageModelFamilyV1: 匹配到的模型规格对象。

    功能:
        1. 首先检查用户自定义的图像模型。
        2. 根据指定的下载源和模型名称在不同的模型库中查找。
        3. 如果找不到指定的模型，抛出ValueError异常。

    异常:
        ValueError: 当找不到指定名称的模型时抛出。
    """
    from ..utils import download_from_modelscope
    from . import BUILTIN_IMAGE_MODELS, MODELSCOPE_IMAGE_MODELS
    from .custom import get_user_defined_images

    # 检查用户自定义模型
    for model_spec in get_user_defined_images():
        if model_spec.model_name == model_name:
            return model_spec

    # 根据下载源和模型名称匹配模型
    if download_hub == "modelscope" and model_name in MODELSCOPE_IMAGE_MODELS:
        logger.debug(f"Image model {model_name} found in ModelScope.")
        return MODELSCOPE_IMAGE_MODELS[model_name]
    elif download_hub == "huggingface" and model_name in BUILTIN_IMAGE_MODELS:
        logger.debug(f"Image model {model_name} found in Huggingface.")
        return BUILTIN_IMAGE_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_IMAGE_MODELS:
        logger.debug(f"Image model {model_name} found in ModelScope.")
        return MODELSCOPE_IMAGE_MODELS[model_name]
    elif model_name in BUILTIN_IMAGE_MODELS:
        logger.debug(f"Image model {model_name} found in Huggingface.")
        return BUILTIN_IMAGE_MODELS[model_name]
    else:
        raise ValueError(
            f"Image model {model_name} not found, available"
            f"model list: {BUILTIN_IMAGE_MODELS.keys()}"
        )


def cache(model_spec: ImageModelFamilyV1):
    """
    缓存指定的图像模型。

    参数:
        model_spec (ImageModelFamilyV1): 要缓存的模型规格。

    返回:
        返回缓存操作的结果。

    功能:
        调用通用的缓存函数来缓存图像模型。
    """
    from ..utils import cache

    return cache(model_spec, ImageModelDescription)


def get_cache_dir(model_spec: ImageModelFamilyV1):
    """
    获取模型的缓存目录。

    参数:
        model_spec (ImageModelFamilyV1): 模型规格对象。

    返回:
        str: 模型缓存的完整路径。

    功能:
        根据全局缓存目录和模型名称生成模型的缓存路径。
    """
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def get_cache_status(
    model_spec: ImageModelFamilyV1,
) -> bool:
    """
    检查模型的缓存状态。

    参数:
        model_spec (ImageModelFamilyV1): 模型规格对象。

    返回:
        bool: 如果模型已缓存且有效，返回True；否则返回False。

    功能:
        1. 获取模型的缓存目录和元数据文件路径。
        2. 检查模型是否在内置模型和ModelScope模型中都存在。
        3. 验证模型的修订版本是否有效。
    """
    cache_dir = get_cache_dir(model_spec)
    meta_path = os.path.join(cache_dir, "__valid_download")

    model_name = model_spec.model_name
    if model_name in BUILTIN_IMAGE_MODELS and model_name in MODELSCOPE_IMAGE_MODELS:
        hf_spec = BUILTIN_IMAGE_MODELS[model_name]
        ms_spec = MODELSCOPE_IMAGE_MODELS[model_name]

        return any(
            [
                valid_model_revision(meta_path, hf_spec.model_revision),
                valid_model_revision(meta_path, ms_spec.model_revision),
            ]
        )
    else:  # 通常用于单元测试
        return valid_model_revision(meta_path, model_spec.model_revision)


def create_image_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    peft_model_config: Optional[PeftModelConfig] = None,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[DiffusionModel, ImageModelDescription]:
    """
    创建图像模型实例。

    参数:
        subpool_addr (str): 子池地址。
        devices (List[str]): 可用设备列表。
        model_uid (str): 模型的唯一标识符。
        model_name (str): 模型名称。
        peft_model_config (Optional[PeftModelConfig]): PEFT模型配置，默认为None。
        download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 
            下载源，默认为None。
        model_path (Optional[str]): 模型路径，默认为None。
        **kwargs: 其他关键字参数。

    返回:
        Tuple[DiffusionModel, ImageModelDescription]: 
            包含创建的DiffusionModel实例和ImageModelDescription的元组。

    功能:
        1. 匹配指定的扩散模型。
        2. 处理ControlNet相关的配置。
        3. 缓存模型（如果需要）。
        4. 设置LoRA（如果提供了PEFT配置）。
        5. 创建DiffusionModel实例和ImageModelDescription。
        6. 返回创建的模型实例和描述。

    异常:
        ValueError: 当ControlNet配置无效时抛出。
    """
    model_spec = match_diffusion(model_name, download_hub)
    controlnet = kwargs.get("controlnet")
    
    # 处理ControlNet
    if controlnet is not None:
        if isinstance(controlnet, str):
            controlnet = [controlnet]
        elif not isinstance(controlnet, collections.abc.Sequence):
            raise ValueError("controlnet should be a str or a list of str.")
        elif set(controlnet) != len(controlnet):
            raise ValueError("controlnet should be a list of unique str.")
        elif not model_spec.controlnet:
            raise ValueError(f"Model {model_name} has empty controlnet list.")

        controlnet_model_paths = []
        assert model_spec.controlnet is not None
        for name in controlnet:
            for cn_model_spec in model_spec.controlnet:
                if cn_model_spec.model_name == name:
                    if not model_path:
                        model_path = cache(cn_model_spec)
                    controlnet_model_paths.append(model_path)
                    break
            else:
                raise ValueError(
                    f"controlnet `{name}` is not supported for model `{model_name}`."
                )
        if len(controlnet_model_paths) == 1:
            kwargs["controlnet"] = controlnet_model_paths[0]
        else:
            kwargs["controlnet"] = controlnet_model_paths

    # 缓存模型（如果需要）
    if not model_path:
        model_path = cache(model_spec)

    # 设置LoRA（如果提供）
    if peft_model_config is not None:
        lora_model = peft_model_config.peft_model
        lora_load_kwargs = peft_model_config.image_lora_load_kwargs
        lora_fuse_kwargs = peft_model_config.image_lora_fuse_kwargs
    else:
        lora_model = None
        lora_load_kwargs = None
        lora_fuse_kwargs = None

    # 创建DiffusionModel实例
    model = DiffusionModel(
        model_uid,
        model_path,
        lora_model_paths=lora_model,
        lora_load_kwargs=lora_load_kwargs,
        lora_fuse_kwargs=lora_fuse_kwargs,
        abilities=model_spec.model_ability,
        **kwargs,
    )

    # 创建ImageModelDescription
    model_description = ImageModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )

    return model, model_description
