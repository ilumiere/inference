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

import logging
import tempfile
from typing import TYPE_CHECKING, List, Optional

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class FunASRModel:
    """
    FunASRModel 类用于封装 FunASR 模型的加载、初始化和使用。
    该类提供了语音识别（转录）的功能。

    属性:
        _model_uid (str): 模型的唯一标识符
        _model_path (str): 模型文件的路径
        _model_spec (AudioModelFamilyV1): 模型规格
        _device (Optional[str]): 运行模型的设备，如 'cpu' 或 'cuda'
        _model: 加载的 FunASR 模型实例
        _kwargs (dict): 其他可选参数
    """

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV1",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化 FunASRModel 实例。

        参数:
            model_uid (str): 模型的唯一标识符
            model_path (str): 模型文件的路径
            model_spec (AudioModelFamilyV1): 模型规格
            device (Optional[str]): 运行模型的设备，如 'cpu' 或 'cuda'
            **kwargs: 其他可选参数
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    def load(self):
        """
        加载 FunASR 模型。

        该方法尝试导入 FunASR 库，设置适当的设备，并使用指定的参数加载模型。

        异常:
            ImportError: 如果无法导入 FunASR 库
            ValueError: 如果指定的设备不可用
        """
        try:
            from funasr import AutoModel
        except ImportError:
            error_message = "Failed to import module 'funasr'"
            installation_guide = [
                "Please make sure 'funasr' is installed. ",
                "You can install it by `pip install funasr`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # 如果没有指定设备，获取可用设备
        if self._device is None:
            self._device = get_available_device()
        else:
            # 检查指定的设备是否可用
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        # 准备模型配置参数
        kwargs = self._model_spec.default_model_config.copy()
        kwargs.update(self._kwargs)
        logger.debug("Loading FunASR model with kwargs: %s", kwargs)
        
        # 加载 FunASR 模型
        self._model = AutoModel(model=self._model_path, device=self._device, **kwargs)

    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        执行语音转录。

        参数:
            audio (bytes): 要转录的音频数据
            language (Optional[str]): 音频语言，默认为 None（自动检测）
            prompt (Optional[str]): 提示文本，目前被忽略
            response_format (str): 响应格式，默认为 "json"
            temperature (float): 采样温度，目前不支持
            timestamp_granularities (Optional[List[str]]): 时间戳粒度，目前不支持
            **kwargs: 其他可选参数

        返回:
            dict: 包含转录文本的字典

        异常:
            RuntimeError: 如果使用了不支持的参数
            ValueError: 如果指定了不支持的响应格式
        """
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        # 检查不支持的参数
        if temperature != 0:
            raise RuntimeError("`temperature`is not supported for FunASR")
        if timestamp_granularities is not None:
            raise RuntimeError("`timestamp_granularities`is not supported for FunASR")
        if prompt is not None:
            logger.warning(
                "Prompt for funasr transcriptions will be ignored: %s", prompt
            )

        # 设置语言，如果未指定则自动检测
        language = "auto" if language is None else language

        # 使用临时文件处理音频数据
        with tempfile.NamedTemporaryFile(buffering=0) as f:
            f.write(audio)

            # 准备转录配置
            kw = self._model_spec.default_transcription_config.copy()  # type: ignore
            kw.update(kwargs)
            logger.debug("Calling FunASR model with kwargs: %s", kw)
            
            # 执行转录
            result = self._model.generate(  # type: ignore
                input=f.name, cache={}, language=language, **kw
            )
            # 后处理转录结果
            text = rich_transcription_postprocess(result[0]["text"])

            # 根据指定格式返回结果
            if response_format == "json":
                return {"text": text}
            else:
                raise ValueError(f"Unsupported response format: {response_format}")

    def translations(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        """
        尝试执行语音翻译。

        该方法目前不支持，会抛出异常。

        异常:
            RuntimeError: FunASR 不支持翻译 API
        """
        raise RuntimeError("FunASR does not support translations API")
