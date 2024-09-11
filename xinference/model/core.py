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
    """
    在这段代码中，ABC 是 Python 标准库 abc 模块中的 ABC 类（Abstract Base Class，抽象基类）的缩写。ModelDescription 类继承自 ABC，这表明 ModelDescription 是一个抽象基类。让我解释一下为什么要这样做：
    抽象基类的作用：
    定义接口：ABC 用于定义一个接口或者骨架，子类必须实现这个接口。
    强制实现：它可以强制子类实现某些方法。
    不能实例化：抽象基类本身不能被实例化。
    2. 为什么 ModelDescription 继承 ABC：
    定义通用接口：ModelDescription 定义了所有模型描述类应该具有的通用方法。
    强制实现某些方法：通过使用 @abstractmethod 装饰器，它要求所有子类必须实现 to_version_info 方法。
    提供部分实现：它提供了 __init__ 和 to_dict 方法的基本实现或框架。
    3. 好处：
    代码规范：确保所有模型描述类都遵循相同的接口。
    错误预防：如果子类忘记实现必要的方法，在实例化时会引发错误。
    清晰的设计：明确表示这个类是一个抽象概念，需要被具体的子类实现。
    示例：
    Ask
    总之，继承 ABC 使 ModelDescription 成为一个抽象基类，这有助于定义一个清晰的接口，确保所有派生的模型描述类都遵循相同的结构，同时提供了一定的灵活性来适应不同类型的模型。
    
    class ConcreteModelDescription(ModelDescription):
       def to_version_info(self):
           # 具体实现
           return {"version": "1.0", "other_info": "..."}
       
       def to_dict(self):
           # 覆盖基类的实现
           return {"address": self.address, "devices": self.devices}
    Args:
        ABC (_type_): _description_
    """
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
