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
# 导入所需的模块和类
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from ...constants import XINFERENCE_CACHE_DIR
from ..core import CacheableModelSpec, ModelDescription
from ..utils import valid_model_revision
from .chattts import ChatTTSModel
from .cosyvoice import CosyVoiceModel
from .fish_speech import FishSpeechModel
from .funasr import FunASRModel
from .whisper import WhisperModel

# 定义常量
MAX_ATTEMPTS = 3

# 设置日志记录器
logger = logging.getLogger(__name__)

# 用于检查模型是否已缓存的字典
# 在注册所有内置模型时初始化
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
AUDIO_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)


def get_audio_model_descriptions():
    """
    获取音频模型描述的深拷贝。

    返回:
        Dict[str, List[Dict]]: 音频模型描述的深拷贝。

    说明:
    - 导入copy模块。
    - 返回AUDIO_MODEL_DESCRIPTIONS的深拷贝，避免直接修改原始数据。
    """
    import copy

    return copy.deepcopy(AUDIO_MODEL_DESCRIPTIONS)


class AudioModelFamilyV1(CacheableModelSpec):
    """
    音频模型家族V1规格类，继承自CacheableModelSpec。

    属性:
    - model_family (str): 模型家族名称。
    - model_name (str): 模型名称。
    - model_id (str): 模型ID。
    - model_revision (str): 模型版本。
    - multilingual (bool): 是否支持多语言。
    - ability (str): 模型能力描述。
    - default_model_config (Optional[Dict[str, Any]]): 默认模型配置。
    - default_transcription_config (Optional[Dict[str, Any]]): 默认转录配置。
    """
    model_family: str
    model_name: str
    model_id: str
    model_revision: str
    multilingual: bool
    ability: str
    default_model_config: Optional[Dict[str, Any]]
    default_transcription_config: Optional[Dict[str, Any]]


class AudioModelDescription(ModelDescription):
    """
    音频模型描述类，继承自ModelDescription。

    方法:
    - __init__: 初始化方法。
    - to_dict: 将模型描述转换为字典格式。
    - to_version_info: 获取模型版本信息。
    """

    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: AudioModelFamilyV1,
        model_path: Optional[str] = None,
    ):
        """
        初始化AudioModelDescription实例。

        参数:
        - address (Optional[str]): 模型地址。
        - devices (Optional[List[str]]): 设备列表。
        - model_spec (AudioModelFamilyV1): 模型规格。
        - model_path (Optional[str]): 模型路径，默认为None。

        说明:
        - 调用父类的__init__方法。
        - 保存model_spec作为实例属性。
        """
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        """
        将模型描述转换为字典格式。

        返回:
        Dict: 包含模型信息的字典。

        说明:
        - 返回一个包含模型类型、地址、加速器、模型名称、模型家族和模型版本的字典。
        """
        return {
            "model_type": "audio",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "model_family": self._model_spec.model_family,
            "model_revision": self._model_spec.model_revision,
        }

    def to_version_info(self):
        """
        获取模型版本信息。

        返回:
        Dict: 包含模型版本、文件位置和缓存状态的字典。

        说明:
        - 导入get_model_version函数。
        - 根据model_path是否为None，确定模型的缓存状态和文件位置。
        - 返回包含模型版本、文件位置和缓存状态的字典。
        """
        from .utils import get_model_version

        if self._model_path is None:
            is_cached = get_cache_status(self._model_spec)
            file_location = get_cache_dir(self._model_spec)
        else:
            is_cached = True
            file_location = self._model_path

        return {
            "model_version": get_model_version(self._model_spec),
            "model_file_location": file_location,
            "cache_status": is_cached,
        }


def generate_audio_description(
    image_model: AudioModelFamilyV1,
) -> Dict[str, List[Dict]]:
    """
    生成音频模型描述。

    参数:
    - image_model (AudioModelFamilyV1): 音频模型规格。

    返回:
    Dict[str, List[Dict]]: 包含模型名称和版本信息的字典。

    说明:
    - 创建一个defaultdict来存储结果。
    - 使用模型名称作为键，将版本信息添加到对应的列表中。
    - 返回生成的描述字典。
    """
    res = defaultdict(list)
    res[image_model.model_name].append(
        AudioModelDescription(None, None, image_model).to_version_info()
    )
    return res


def match_audio(
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
) -> AudioModelFamilyV1:
    """
    匹配音频模型。

    参数:
    - model_name (str): 模型名称。
    - download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 下载中心，默认为None。

    返回:
    AudioModelFamilyV1: 匹配的音频模型规格。

    说明:
    - 导入必要的模块和函数。
    - 首先检查用户定义的音频模型。
    - 根据download_hub和模型名称在不同的模型库中查找匹配的模型。
    - 如果找到匹配的模型，返回相应的模型规格。
    - 如果未找到匹配的模型，抛出ValueError异常。
    """
    from ..utils import download_from_modelscope
    from . import BUILTIN_AUDIO_MODELS, MODELSCOPE_AUDIO_MODELS
    from .custom import get_user_defined_audios

    for model_spec in get_user_defined_audios():
        if model_spec.model_name == model_name:
            return model_spec

    if download_hub == "huggingface" and model_name in BUILTIN_AUDIO_MODELS:
        logger.debug(f"Audio model {model_name} found in huggingface.")
        return BUILTIN_AUDIO_MODELS[model_name]
    elif download_hub == "modelscope" and model_name in MODELSCOPE_AUDIO_MODELS:
        logger.debug(f"Audio model {model_name} found in ModelScope.")
        return MODELSCOPE_AUDIO_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_AUDIO_MODELS:
        logger.debug(f"Audio model {model_name} found in ModelScope.")
        return MODELSCOPE_AUDIO_MODELS[model_name]
    elif model_name in BUILTIN_AUDIO_MODELS:
        logger.debug(f"Audio model {model_name} found in huggingface.")
        return BUILTIN_AUDIO_MODELS[model_name]
    else:
        raise ValueError(
            f"Audio model {model_name} not found, available"
            f"model list: {BUILTIN_AUDIO_MODELS.keys()}"
        )


def cache(model_spec: AudioModelFamilyV1):
    """
    缓存音频模型。

    参数:
    - model_spec (AudioModelFamilyV1): 音频模型规格。

    返回:
    与utils.cache函数的返回值相同。

    说明:
    - 从..utils导入cache函数。
    - 调用cache函数，传入model_spec和AudioModelDescription。
    """
    from ..utils import cache

    return cache(model_spec, AudioModelDescription)


def get_cache_dir(model_spec: AudioModelFamilyV1):
    """
    获取模型缓存目录。

    参数:
    - model_spec (AudioModelFamilyV1): 音频模型规格。

    返回:
    str: 模型缓存目录的绝对路径。

    说明:
    - 使用os.path.realpath获取真实路径。
    - 将XINFERENCE_CACHE_DIR和模型名称拼接成完整的缓存路径。
    """
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def get_cache_status(
    model_spec: AudioModelFamilyV1,
) -> bool:
    """
    获取模型缓存状态。

    参数:
    - model_spec (AudioModelFamilyV1): 音频模型规格。

    返回:
    bool: 模型是否已缓存。

    说明:
    - 获取缓存目录。
    - 构造元数据文件路径。
    - 调用valid_model_revision函数检查模型版本是否有效。
    """
    cache_dir = get_cache_dir(model_spec)
    meta_path = os.path.join(cache_dir, "__valid_download")
    return valid_model_revision(meta_path, model_spec.model_revision)


def create_audio_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[
    Union[WhisperModel, FunASRModel, ChatTTSModel, CosyVoiceModel, FishSpeechModel],
    AudioModelDescription,
]:
    """
    创建音频模型实例。

    参数:
    - subpool_addr (str): 子池地址。
    - devices (List[str]): 设备列表。
    - model_uid (str): 模型唯一标识符。
    - model_name (str): 模型名称。
    - download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 下载中心，默认为None。
    - model_path (Optional[str]): 模型路径，默认为None。
    - **kwargs: 其他关键字参数。

    返回:
    Tuple[Union[WhisperModel, FunASRModel, ChatTTSModel, CosyVoiceModel, FishSpeechModel], AudioModelDescription]:
    包含模型实例和模型描述的元组。

    说明:
    - 匹配音频模型。
    - 如果model_path为None，则缓存模型。
    - 根据模型家族创建相应的模型实例。
    - 创建AudioModelDescription实例。
    - 返回模型实例和模型描述的元组。
    """
    model_spec = match_audio(model_name, download_hub)
    if model_path is None:
        model_path = cache(model_spec)
    model: Union[
        WhisperModel, FunASRModel, ChatTTSModel, CosyVoiceModel, FishSpeechModel
    ]
    if model_spec.model_family == "whisper":
        model = WhisperModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "funasr":
        model = FunASRModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "ChatTTS":
        model = ChatTTSModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "CosyVoice":
        model = CosyVoiceModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "FishAudio":
        model = FishSpeechModel(model_uid, model_path, model_spec, **kwargs)
    else:
        raise Exception(f"Unsupported audio model family: {model_spec.model_family}")
    model_description = AudioModelDescription(
        subpool_addr, devices, model_spec, model_path
    )
    return model, model_description
