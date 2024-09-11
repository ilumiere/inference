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
from threading import Lock
from typing import Any, List, Optional

from ..._compat import (
    ROOT_KEY,
    ErrorWrapper,
    Protocol,
    StrBytes,
    ValidationError,
    load_str_bytes,
)
from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from .core import AudioModelFamilyV1

# 设置日志记录器
logger = logging.getLogger(__name__)

# 创建一个全局锁，用于线程安全操作
UD_AUDIO_LOCK = Lock()

# CustomAudioModelFamilyV1 类
# 这个类继承自 AudioModelFamilyV1，用于定义自定义音频模型的规格
class CustomAudioModelFamilyV1(AudioModelFamilyV1):
    model_id: Optional[str]  # 模型ID，可选
    model_revision: Optional[str]  # 模型版本，可选
    model_uri: Optional[str]  # 模型URI，可选

    @classmethod
    def parse_raw(
        cls: Any,
        b: StrBytes,
        *,
        content_type: Optional[str] = None,
        encoding: str = "utf8",
        proto: Protocol = None,
        allow_pickle: bool = False,
    ) -> AudioModelFamilyV1:
        """
        See source code of BaseModel.parse_raw
        解析原始数据并创建 AudioModelFamilyV1 实例
        
        参数:
        - b: 要解析的原始字符串或字节
        - content_type: 内容类型
        - encoding: 编码方式
        - proto: 协议
        - allow_pickle: 是否允许pickle

        返回:
        - AudioModelFamilyV1 实例
        """
        # 尝试加载和解析原始数据
        try:
            obj = load_str_bytes(
                b,
                proto=proto,
                content_type=content_type,
                encoding=encoding,
                allow_pickle=allow_pickle,
                json_loads=cls.__config__.json_loads,
            )
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            # 如果解析失败，抛出验证错误
            raise ValidationError([ErrorWrapper(e, loc=ROOT_KEY)], cls)

        # 解析对象并创建 AudioModelFamilyV1 实例
        audio_spec: AudioModelFamilyV1 = cls.parse_obj(obj)

        # 检查 model_family 是否已指定
        if audio_spec.model_family is None:
            raise ValueError(
                f"You must specify `model_family` when registering custom Audio models."
            )
        assert isinstance(audio_spec.model_family, str)
        return audio_spec

# 存储用户定义的音频模型列表
UD_AUDIOS: List[CustomAudioModelFamilyV1] = []

def get_user_defined_audios() -> List[CustomAudioModelFamilyV1]:
    """
    获取用户定义的音频模型列表的副本
    
    返回:
    - 用户定义的音频模型列表的副本
    """
    with UD_AUDIO_LOCK:
        return UD_AUDIOS.copy()

def register_audio(model_spec: CustomAudioModelFamilyV1, persist: bool):
    """
    注册自定义音频模型
    
    参数:
    - model_spec: 自定义音频模型规格
    - persist: 是否持久化存储模型信息
    """
    from ...constants import XINFERENCE_MODEL_DIR
    from ..utils import is_valid_model_name, is_valid_model_uri
    from . import BUILTIN_AUDIO_MODELS, MODELSCOPE_AUDIO_MODELS

    # 验证模型名称
    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"Invalid model name {model_spec.model_name}.")

    # 验证模型 URI（如果存在）
    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    with UD_AUDIO_LOCK:
        # 检查模型名称是否与现有模型冲突
        for model_name in (
            list(BUILTIN_AUDIO_MODELS.keys())
            + list(MODELSCOPE_AUDIO_MODELS.keys())
            + [spec.model_name for spec in UD_AUDIOS]
        ):
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {model_spec.model_name}"
                )

        # 将新模型添加到用户定义的音频模型列表中
        UD_AUDIOS.append(model_spec)

    # 如果需要持久化存储，将模型信息保存到文件
    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "audio", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())

def unregister_audio(model_name: str, raise_error: bool = True):
    """
    注销自定义音频模型
    
    参数:
    - model_name: 要注销的模型名称
    - raise_error: 如果模型不存在，是否抛出错误
    """
    with UD_AUDIO_LOCK:
        # 查找要注销的模型
        model_spec = None
        for i, f in enumerate(UD_AUDIOS):
            if f.model_name == model_name:
                model_spec = f
                break
        
        if model_spec:
            # 从列表中移除模型
            UD_AUDIOS.remove(model_spec)

            # 删除持久化存储的模型文件
            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "audio", f"{model_spec.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            # 删除模型缓存
            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"Remove the cache of user-defined model {model_spec.model_name}. "
                    f"Cache directory: {cache_dir}"
                )
                if os.path.isdir(cache_dir):
                    os.rmdir(cache_dir)
                else:
                    logger.warning(
                        f"Cache directory is not a soft link, please remove it manually."
                    )
        else:
            # 如果模型不存在，根据 raise_error 参数决定是否抛出错误
            if raise_error:
                raise ValueError(f"Model {model_name} not found")
            else:
                logger.warning(f"Custom audio model {model_name} not found")
