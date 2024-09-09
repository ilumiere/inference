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

from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Tuple, Union

from .._compat import BaseModel
from ..types import PeftModelConfig


class ModelDescription(ABC):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_path: Optional[str] = None,
    ):
        # 初始化模型描述
        self.address = address  # 模型地址
        self.devices = devices  # 设备列表
        self._model_path = model_path  # 模型路径

    def to_dict(self):
        """
        返回一个字典来描述模型的一些信息。
        :return: 包含模型信息的字典
        """
        raise NotImplementedError

    @abstractmethod
    def to_version_info(self):
        """
        返回一个字典来描述模型实例的版本信息
        """


def create_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_type: str,
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
    peft_model_config: Optional[PeftModelConfig] = None,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[Any, ModelDescription]:
    # 导入各种模型实例创建函数
    from .audio.core import create_audio_model_instance
    from .embedding.core import create_embedding_model_instance
    from .flexible.core import create_flexible_model_instance
    from .image.core import create_image_model_instance
    from .llm.core import create_llm_model_instance
    from .rerank.core import create_rerank_model_instance
    from .video.core import create_video_model_instance

    # 根据模型类型创建相应的模型实例
    if model_type == "LLM":
        # 创建大语言模型实例
        return create_llm_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            model_engine,
            model_format,
            model_size_in_billions,
            quantization,
            peft_model_config,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "embedding":
        # 创建嵌入模型实例
        # 嵌入模型不接受trust_remote_code参数
        kwargs.pop("trust_remote_code", None)
        return create_embedding_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "image":
        # 创建图像模型实例
        kwargs.pop("trust_remote_code", None)
        return create_image_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            peft_model_config,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "rerank":
        # 创建重排序模型实例
        kwargs.pop("trust_remote_code", None)
        return create_rerank_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "audio":
        # 创建音频模型实例
        kwargs.pop("trust_remote_code", None)
        return create_audio_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "video":
        # 创建视频模型实例
        kwargs.pop("trust_remote_code", None)
        return create_video_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "flexible":
        # 创建灵活模型实例
        kwargs.pop("trust_remote_code", None)
        return create_flexible_model_instance(
            subpool_addr, devices, model_uid, model_name, model_path, **kwargs
        )
    else:
        # 如果提供了不支持的模型类型，则抛出异常
        raise ValueError(f"不支持的模型类型: {model_type}.")


class CacheableModelSpec(BaseModel):
    model_name: str  # 模型名称
    model_id: str  # 模型ID
    model_revision: Optional[str]  # 模型版本
    model_hub: str = "huggingface"  # 模型中心，默认为huggingface
