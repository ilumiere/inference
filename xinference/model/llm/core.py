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

import abc
import inspect
import logging
import os
import platform
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from ...core.utils import parse_replica_model_uid
from ...types import PeftModelConfig
from ..core import ModelDescription

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV1, LLMSpecV1

logger = logging.getLogger(__name__)


LLM_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)


def get_llm_model_descriptions():
    import copy

    return copy.deepcopy(LLM_MODEL_DESCRIPTIONS)


class LLM(abc.ABC):
    """
    LLM（大型语言模型）的抽象基类。
    定义了LLM的基本结构和接口。
    """

    def __init__(
        self,
        replica_model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        *args,
        **kwargs,
    ):
        """
        初始化LLM实例。

        :param replica_model_uid: 复制模型的唯一标识符
        :param model_family: 模型家族
        :param model_spec: 模型规格
        :param quantization: 量化方法
        :param model_path: 模型路径
        :param args: 额外的位置参数
        :param kwargs: 额外的关键字参数
        """
        self.model_uid, self.replica, self.rep_id = parse_replica_model_uid(
            replica_model_uid
        )
        self.model_family = model_family
        self.model_spec = model_spec
        self.quantization = quantization
        self.model_path = model_path
        if args:
            raise ValueError(f"Unrecognized positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    @staticmethod
    def _is_darwin_and_apple_silicon():
        """
        检查当前系统是否为Darwin（macOS）且使用Apple Silicon处理器。

        :return: 如果是Darwin系统且使用Apple Silicon处理器，则返回True；否则返回False
        """
        return platform.system() == "Darwin" and platform.processor() == "arm"

    @staticmethod
    def _is_linux():
        """
        检查当前系统是否为Linux。

        :return: 如果是Linux系统，则返回True；否则返回False
        """
        return platform.system() == "Linux"

    @staticmethod
    @lru_cache
    def _has_cuda_device():
        """
        Use pynvml to impl this interface.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

        device_count = 0
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
        except:
            pass
        finally:
            try:
                nvmlShutdown()
            except:
                pass

        return device_count > 0

    @staticmethod
    @lru_cache
    def _get_cuda_count():
        """
        获取可用的CUDA设备数量。

        :return: 可用的CUDA设备数量
        """
        # 从utils模块导入cuda_count函数
        from ...utils import cuda_count

        # 先获取自定义的CUDA_VISIBLE_DEVICES的CUDA设备数量
        # 获取环境变量CUDA_VISIBLE_DEVICES的值
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        
        # 如果CUDA_VISIBLE_DEVICES未设置，获取设置的全部CUDA设备数量
        if cuda_visible_devices is None:
            # 直接返回cuda_count()的结果
            return cuda_count()

        # 如果CUDA_VISIBLE_DEVICES设置为"-1"，表示禁用所有CUDA设备
        if cuda_visible_devices == "-1":
            return 0
        else:
            # 否则，返回CUDA_VISIBLE_DEVICES中指定的设备数量
            # 通过分割字符串并计算长度来获得设备数量
            return len(cuda_visible_devices.split(","))

    @abstractmethod
    def load(self):
        """
        加载模型的抽象方法。
        子类必须实现此方法以加载特定的模型。
        """
        raise NotImplementedError

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        """
        检查给定的模型家族、规格和量化方法是否匹配当前LLM类。
        子类必须实现此方法以定义匹配条件。

        :param llm_family: 模型家族
        :param llm_spec: 模型规格
        :param quantization: 量化方法
        :return: 如果匹配，则返回True；否则返回False
        """
        raise NotImplementedError


class LLMDescription(ModelDescription):
    """
    LLMDescription类，用于描述大语言模型（LLM）的相关信息。
    继承自ModelDescription类。
    """

    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        llm_family: "LLMFamilyV1",
        llm_spec: "LLMSpecV1",
        quantization: Optional[str],
        model_path: Optional[str] = None,
    ):
        """
        初始化LLMDescription实例。

        :param address: 模型地址
        :param devices: 设备列表
        :param llm_family: LLM家族
        :param llm_spec: LLM规格
        :param quantization: 量化方法
        :param model_path: 模型路径（可选）
        """
        super().__init__(address, devices, model_path=model_path)
        self._llm_family = llm_family
        self._llm_spec = llm_spec
        self._quantization = quantization

    def to_dict(self):
        """
        将LLMDescription实例转换为字典形式。

        :return: 包含LLM描述信息的字典
        """
        return {
            "model_type": "LLM",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._llm_family.model_name,
            "model_lang": self._llm_family.model_lang,
            "model_ability": self._llm_family.model_ability,
            "model_description": self._llm_family.model_description,
            "model_format": self._llm_spec.model_format,
            "model_size_in_billions": self._llm_spec.model_size_in_billions,
            "model_family": self._llm_family.model_family
            or self._llm_family.model_name,
            "quantization": self._quantization,
            "model_hub": self._llm_spec.model_hub,
            "revision": self._llm_spec.model_revision,
            "context_length": self._llm_family.context_length,
        }

    def to_version_info(self):
        """
        生成模型版本信息。

        :return: 包含模型版本、文件位置、缓存状态等信息的字典
        """
        from .utils import get_file_location, get_model_version

        model_file_location, cache_status = get_file_location(
            self._llm_family, self._llm_spec, self._quantization
        )
        print(f"model_file_location: {model_file_location}")
        print(f"cache_status: {cache_status}")

        return {
            "model_version": get_model_version(
                self._llm_family, self._llm_spec, self._quantization
            ),
            "model_file_location": model_file_location,
            "cache_status": cache_status,
            "quantization": self._quantization,
            "model_format": self._llm_spec.model_format,
            "model_size_in_billions": self._llm_spec.model_size_in_billions,
        }


def generate_llm_description(llm_family: "LLMFamilyV1") -> Dict[str, List[Dict]]:
    """
    生成给定LLM家族的描述信息。

    :param llm_family: LLM家族对象
    :return: 包含LLM描述信息的字典，键为模型名称，值为版本信息列表
    """
    res = defaultdict(list)
    for spec in llm_family.model_specs:
        for q in spec.quantizations:
            res[llm_family.model_name].append(
                LLMDescription(None, None, llm_family, spec, q).to_version_info()
            )
    return res


def create_llm_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
    peft_model_config: Optional[PeftModelConfig] = None,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[LLM, LLMDescription]:
    """
    创建LLM模型实例。

    :param subpool_addr: 子池地址
    :param devices: 设备列表
    :param model_uid: 模型唯一标识符
    :param model_name: 模型名称
    :param model_engine: 模型引擎
    :param model_format: 模型格式（可选）
    :param model_size_in_billions: 模型大小（以十亿参数计）（可选）
    :param quantization: 量化方法（可选）
    :param peft_model_config: PEFT模型配置（可选）
    :param download_hub: 下载中心（可选）
    :param model_path: 模型路径（可选）
    :param kwargs: 其他关键字参数
    :return: LLM模型实例和LLM描述对象的元组
    """
    from .llm_family import cache, check_engine_by_spec_parameters, match_llm

    if model_engine is None:
        raise ValueError("model_engine is required for LLM model")
    match_result = match_llm(
        model_name, model_format, model_size_in_billions, quantization, download_hub
    )

    if not match_result:
        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format},"
            f" size: {model_size_in_billions}, quantization: {quantization}"
        )
    llm_family, llm_spec, quantization = match_result
    assert quantization is not None

    llm_cls = check_engine_by_spec_parameters(
        model_engine,
        llm_family.model_name,
        llm_spec.model_format,
        llm_spec.model_size_in_billions,
        quantization,
    )
    logger.debug(f"Launching {model_uid} with {llm_cls.__name__}")

    if not model_path:
        model_path = cache(llm_family, llm_spec, quantization)

    peft_model = peft_model_config.peft_model if peft_model_config else None
    if peft_model is not None:
        if "peft_model" in inspect.signature(llm_cls.__init__).parameters:
            model = llm_cls(
                model_uid,
                llm_family,
                llm_spec,
                quantization,
                model_path,
                kwargs,
                peft_model,
            )
        else:
            logger.warning(
                f"Model not supported with lora, name: {model_name}, format: {model_format}, engine: {model_engine}. "
                f"Load this without lora."
            )
            model = llm_cls(
                model_uid, llm_family, llm_spec, quantization, model_path, kwargs
            )
    else:
        model = llm_cls(
            model_uid, llm_family, llm_spec, quantization, model_path, kwargs
        )
    return model, LLMDescription(
        subpool_addr, devices, llm_family, llm_spec, quantization
    )
