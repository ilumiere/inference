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
# 导入所需的模块
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

# 导入设备相关的工具函数
from ...device_utils import (
    get_available_device,
    get_device_preferred_dtype,
    is_device_available,
)

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

# 设置日志记录器
logger = logging.getLogger(__name__)

# WhisperModel 类：用于加载和使用 Whisper 模型进行语音识别和翻译
class WhisperModel:
    """
    WhisperModel 类用于封装 Whisper 模型的加载、初始化和使用。
    它提供了语音转录和翻译的功能。
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
        初始化 WhisperModel 实例。

        参数:
        - model_uid (str): 模型的唯一标识符
        - model_path (str): 模型文件的路径
        - model_spec (AudioModelFamilyV1): 模型规格
        - device (Optional[str]): 运行模型的设备，如 'cpu' 或 'cuda'
        - **kwargs: 其他可选参数
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    def load(self):
        """
        加载 Whisper 模型。

        此方法初始化模型、处理器和管道，为后续的语音识别任务做准备。
        """
        # 导入必要的 transformers 组件
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        # 如果未指定设备，获取可用设备
        if self._device is None:
            self._device = get_available_device()
        else:
            # 检查指定的设备是否可用
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        # 获取设备首选的数据类型
        torch_dtype = get_device_preferred_dtype(self._device)

        # 加载预训练的语音识别模型
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        # 将模型移动到指定设备
        model.to(self._device)

        # 加载模型处理器
        processor = AutoProcessor.from_pretrained(self._model_path)

        # 创建语音识别管道
        self._model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=self._device,
        )

    def _call_model(
        self,
        audio: bytes,
        generate_kwargs: Dict,
        response_format: str,
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        """
        调用加载的模型进行语音识别。

        参数:
        - audio (bytes): 输入的音频数据
        - generate_kwargs (Dict): 生成配置参数
        - response_format (str): 响应格式，可以是 'json' 或 'verbose_json'
        - temperature (float): 生成的温度参数，控制随机性
        - timestamp_granularities (Optional[List[str]]): 时间戳粒度设置

        返回:
        Dict: 包含识别结果的字典
        """
        # 如果温度不为 0，更新生成参数
        if temperature != 0:
            generate_kwargs.update({"temperature": temperature, "do_sample": True})

        # 根据不同的响应格式处理结果
        if response_format == "json":
            logger.debug("Call whisper model with generate_kwargs: %s", generate_kwargs)
            assert callable(self._model)
            result = self._model(audio, generate_kwargs=generate_kwargs)
            return {"text": result["text"]}
        elif response_format == "verbose_json":
            # 处理时间戳设置
            return_timestamps: Union[bool, str] = False
            if not timestamp_granularities:
                return_timestamps = True
            elif timestamp_granularities == ["segment"]:
                return_timestamps = True
            elif timestamp_granularities == ["word"]:
                return_timestamps = "word"
            else:
                raise Exception(
                    f"Unsupported timestamp_granularities: {timestamp_granularities}"
                )
            
            # 调用模型进行识别
            assert callable(self._model)
            results = self._model(
                audio,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps,
            )

            language = generate_kwargs.get("language", "english")

            # 处理段级时间戳
            if return_timestamps is True:
                segments: List[dict] = []

                def _get_chunk_segment_json(idx, text, start, end):
                    find_start = 0
                    if segments:
                        find_start = segments[-1]["seek"] + len(segments[-1]["text"])
                    return {
                        "id": idx,
                        "seek": results["text"].find(text, find_start),
                        "start": start,
                        "end": end,
                        "text": text,
                        "tokens": [],
                        "temperature": temperature,
                        # 这些值无法提供，设为默认值
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }

                for idx, c in enumerate(results.get("chunks", [])):
                    text = c["text"]
                    start, end = c["timestamp"]
                    segments.append(_get_chunk_segment_json(idx, text, start, end))

                return {
                    "task": "transcribe",
                    "language": language,
                    "duration": segments[-1]["end"] if segments else 0,
                    "text": results["text"],
                    "segments": segments,
                }
            else:
                # 处理词级时间戳
                assert return_timestamps == "word"

                words = []
                for idx, c in enumerate(results.get("chunks", [])):
                    text = c["text"]
                    start, end = c["timestamp"]
                    words.append({"word": text, "start": start, "end": end})

                return {
                    "task": "transcribe",
                    "language": language,
                    "duration": words[-1]["end"] if words else 0,
                    "text": results["text"],
                    "words": words,
                }
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        """
        执行语音转录任务。

        参数:
        - audio (bytes): 输入的音频数据
        - language (Optional[str]): 音频语言
        - prompt (Optional[str]): 提示文本（目前被忽略）
        - response_format (str): 响应格式
        - temperature (float): 生成温度
        - timestamp_granularities (Optional[List[str]]): 时间戳粒度

        返回:
        Dict: 包含转录结果的字典
        """
        if prompt is not None:
            logger.warning(
                "Prompt for whisper transcriptions will be ignored: %s", prompt
            )
        return self._call_model(
            audio=audio,
            generate_kwargs=(
                {"language": language, "task": "transcribe"}
                if language is not None
                else {"task": "transcribe"}
            ),
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )

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
        执行语音翻译任务。

        参数:
        - audio (bytes): 输入的音频数据
        - language (Optional[str]): 音频语言
        - prompt (Optional[str]): 提示文本（目前被忽略）
        - response_format (str): 响应格式
        - temperature (float): 生成温度
        - timestamp_granularities (Optional[List[str]]): 时间戳粒度

        返回:
        Dict: 包含翻译结果的字典

        异常:
        RuntimeError: 如果模型不支持多语言翻译
        """
        if not self._model_spec.multilingual:
            raise RuntimeError(
                f"Model {self._model_spec.model_name} is not suitable for translations."
            )
        if prompt is not None:
            logger.warning(
                "Prompt for whisper transcriptions will be ignored: %s", prompt
            )
        return self._call_model(
            audio=audio,
            generate_kwargs=(
                {"language": language, "task": "translate"}
                if language is not None
                else {"task": "translate"}
            ),
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )
