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
import io
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from ..utils import set_all_random_seed

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)

# CosyVoiceModel 类
# 该类用于加载和使用 CosyVoice 模型进行文本到语音的转换
# CosyVoice 是一个支持多种推理模式的语音合成模型
class CosyVoiceModel:
    def __init__(
        self,
        model_uid: str,  # 模型的唯一标识符
        model_path: str,  # 模型文件的路径
        model_spec: "AudioModelFamilyV1",  # 模型规格
        device: Optional[str] = None,  # 运行模型的设备（如 CPU 或 GPU）
        **kwargs,  # 其他可能的参数
    ):
        # 初始化模型属性
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None  # 实际的 CosyVoice 模型实例，初始为 None
        self._kwargs = kwargs

    # 加载 CosyVoice 模型
    def load(self):
        import os
        import sys

        # The yaml config loaded from model has hard-coded the import paths. please refer to: load_hyperpyyaml
        # 将 thirdparty 目录添加到系统路径，以便导入 CosyVoice
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../thirdparty"))

        from cosyvoice.cli.cosyvoice import CosyVoice

        # 初始化 CosyVoice 模型
        self._model = CosyVoice(self._model_path)

    # 生成语音的主要方法
    def speech(
        self,
        input: str,  # 要转换为语音的输入文本
        voice: str,  # 指定的声音（说话人）
        response_format: str = "mp3",  # 输出音频格式，默认为 mp3
        speed: float = 1.0,  # 语音速度，默认为 1.0
        stream: bool = False,  # 是否使用流式输出，默认为 False
        **kwargs,  # 其他可能的参数
    ):
        # CosyVoice 不支持流式输出
        if stream:
            raise Exception("CosyVoiceModel does not support stream.")

        import torchaudio
        from cosyvoice.utils.file_utils import load_wav

        # 从 kwargs 中提取特定参数
        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_text: Optional[str] = kwargs.pop("prompt_text", None)
        instruct_text: Optional[str] = kwargs.pop("instruct_text", None)
        seed: Optional[int] = kwargs.pop("seed", 0)

        # 根据模型类型进行不同的参数验证
        if "SFT" in self._model_spec.model_name:
            # SFT 模型不支持 prompt_speech, prompt_text 和 instruct_text
            assert prompt_speech is None, "CosyVoice SFT model does not support prompt_speech"
            assert prompt_text is None, "CosyVoice SFT model does not support prompt_text"
            assert instruct_text is None, "CosyVoice SFT model does not support instruct_text"
        elif "Instruct" in self._model_spec.model_name:
            # Instruct 模型不支持 prompt_speech 和 prompt_text
            assert prompt_speech is None, "CosyVoice Instruct model does not support prompt_speech"
            assert prompt_text is None, "CosyVoice Instruct model does not support prompt_text"
        else:
            # inference_zero_shot
            # inference_cross_lingual
            # 其他模型（zero-shot 和 cross-lingual）需要 prompt_speech，不支持 instruct_text
            assert prompt_speech is not None, "CosyVoice model expect a prompt_speech"
            assert instruct_text is None, "CosyVoice model does not support instruct_text"

        # 确保模型已加载
        assert self._model is not None

        # 设置随机种子以确保结果可复现
        set_all_random_seed(seed)

        # 根据不同的参数组合选择不同的推理模式
        if prompt_speech:
            assert not voice, "voice can't be set with prompt speech."
            with io.BytesIO(prompt_speech) as prompt_speech_io:
                prompt_speech_16k = load_wav(prompt_speech_io, 16000)
                if prompt_text:
                    logger.info("CosyVoice inference_zero_shot")
                    output = self._model.inference_zero_shot(
                        input, prompt_text, prompt_speech_16k
                    )
                else:
                    logger.info("CosyVoice inference_cross_lingual")
                    output = self._model.inference_cross_lingual(
                        input, prompt_speech_16k
                    )
        else:
            # 获取可用的说话人列表
            available_speakers = self._model.list_avaliable_spks()
            if not voice:
                voice = available_speakers[0]  # 如果未指定 voice，使用第一个可用的说话人
            else:
                # 验证指定的 voice 是否有效
                assert voice in available_speakers, f"Invalid voice {voice}, CosyVoice available speakers: {available_speakers}"
            
            if instruct_text:
                logger.info("CosyVoice inference_instruct")
                output = self._model.inference_instruct(
                    input, voice, instruct_text=instruct_text
                )
            else:
                logger.info("CosyVoice inference_sft")
                output = self._model.inference_sft(input, voice)

        # 将生成的音频保存为指定格式
        with BytesIO() as out:
            torchaudio.save(out, output["tts_speech"], 22050, format=response_format)
            return out.getvalue()  # 返回生成的音频数据
