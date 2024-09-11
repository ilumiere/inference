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
import gc
import logging
import os.path
import queue
import sys
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)

# 生成WAV文件头部的函数
def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    """
    生成WAV文件的头部信息。

    参数:
    sample_rate (int): 采样率，默认为44100Hz
    bit_depth (int): 位深度，默认为16位
    channels (int): 声道数，默认为1（单声道）

    返回:
    bytes: WAV文件头部的字节数据
    """
    import wave

    buffer = BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes

# FishSpeechModel类：用于加载和使用FishSpeech模型进行文本到语音转换
class FishSpeechModel:
    """
    FishSpeechModel类封装了FishSpeech模型的加载、初始化和使用。
    它提供了文本到语音转换的功能。
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
        初始化FishSpeechModel实例。

        参数:
        model_uid (str): 模型的唯一标识符
        model_path (str): 模型文件的路径
        model_spec (AudioModelFamilyV1): 模型规格
        device (Optional[str]): 运行模型的设备，如'cpu'或'cuda'
        **kwargs: 其他可选参数
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._llama_queue = None
        self._model = None
        self._kwargs = kwargs

    def load(self):
        """
        加载FishSpeech模型。

        此方法执行以下步骤：
        1. 设置FishSpeech库的路径
        2. 导入必要的模块
        3. 确定并设置运行设备
        4. 加载Llama模型
        5. 加载VQ-GAN模型
        """
        # 将FishSpeech库的路径添加到系统路径
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../thirdparty/fish_speech")
        )

        from tools.llama.generate import launch_thread_safe_queue
        from tools.vqgan.inference import load_model as load_decoder_model

        # 确定运行设备
        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        # 加载Llama模型
        logger.info("Loading Llama model...")
        self._llama_queue = launch_thread_safe_queue(
            checkpoint_path=self._model_path,
            device=self._device,
            precision=torch.bfloat16,
            compile=False,
        )
        logger.info("Llama model loaded, loading VQ-GAN model...")

        # 加载VQ-GAN模型
        checkpoint_path = os.path.join(
            self._model_path,
            "firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
        )
        self._model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=checkpoint_path,
            device=self._device,
        )

    @torch.inference_mode()
    def _inference(
        self,
        text,
        enable_reference_audio,
        reference_audio,
        reference_text,
        max_new_tokens,
        chunk_length,
        top_p,
        repetition_penalty,
        temperature,
        streaming=False,
    ):
        """
        执行文本到语音的推理过程。

        参数:
        text (str): 要转换为语音的输入文本
        enable_reference_audio (bool): 是否启用参考音频
        reference_audio: 参考音频数据
        reference_text (str): 参考文本
        max_new_tokens (int): 生成的最大新token数
        chunk_length (int): 每个音频块的长度
        top_p (float): 用于nucleus采样的概率阈值
        repetition_penalty (float): 重复惩罚系数
        temperature (float): 采样温度
        streaming (bool): 是否启用流式输出

        生成器返回:
        tuple: 包含音频数据、采样率和音频的元组
        """
        from fish_speech.utils import autocast_exclude_mps
        from tools.api import decode_vq_tokens, encode_reference
        from tools.llama.generate import (
            GenerateRequest,
            GenerateResponse,
            WrappedGenerateResponse,
        )

        # Parse reference audio aka prompt
        # 解析参考音频（如果启用）
        prompt_tokens = encode_reference(
            decoder_model=self._model,
            reference_audio=reference_audio,
            enable_reference_audio=enable_reference_audio,
        )

        # LLAMA Inference
        request = dict(
            device=self._model.device,
            max_new_tokens=max_new_tokens,
            text=text,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,
            iterative_prompt=chunk_length > 0,
            chunk_length=chunk_length,
            max_length=2048,
            prompt_tokens=prompt_tokens if enable_reference_audio else None,
            prompt_text=reference_text if enable_reference_audio else None,
        )

        # 创建响应队列并发送生成请求
        response_queue = queue.Queue()
        self._llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        # 如果是流式输出，先yield WAV头部
        if streaming:
            yield wav_chunk_header(), None, None

        segments = []

        # 处理生成的响应
        while True:
            result: WrappedGenerateResponse = response_queue.get()
            if result.status == "error":
                raise Exception(str(result.response))

            result: GenerateResponse = result.response
            if result.action == "next":
                break

            # 解码VQ tokens生成音频
            with autocast_exclude_mps(
                device_type=self._model.device.type, dtype=torch.bfloat16
            ):
                fake_audios = decode_vq_tokens(
                    decoder_model=self._model,
                    codes=result.codes,
                )

            fake_audios = fake_audios.float().cpu().numpy()
            segments.append(fake_audios)

            # 如果是流式输出，yield每个音频段
            if streaming:
                yield (fake_audios * 32768).astype(np.int16).tobytes(), None, None

        if len(segments) == 0:
            raise Exception("No audio generated, please check the input text.")

        # 合并所有音频段
        # No matter streaming or not, we need to return the final audio
        audio = np.concatenate(segments, axis=0)
        yield None, (self._model.spec_transform.sample_rate, audio), None

        # 清理GPU内存（如果可用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        """
        将文本转换为语音。

        参数:
        input (str): 要转换为语音的输入文本
        voice (str): 语音类型（目前不支持）
        response_format (str): 输出音频格式，默认为"mp3"
        speed (float): 语音速度（目前不支持）
        stream (bool): 是否使用流式输出（目前不支持）
        **kwargs: 其他可选参数

        返回:
        bytes: 生成的音频数据
        """
        logger.warning("Fish speech does not support setting voice: %s.", voice)
        if speed != 1.0:
            logger.warning("Fish speech does not support setting speed: %s.", speed)
        if stream is True:
            logger.warning("stream mode is not implemented.")
        import torchaudio

        # 执行推理
        result = list(
            self._inference(
                text=input,
                enable_reference_audio=False,
                reference_audio=None,
                reference_text="",
                max_new_tokens=0,
                chunk_length=100,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
            )
        )
        sample_rate, audio = result[0][1]
        audio = np.array([audio])

        # 保存生成的音频
        with BytesIO() as out:
            torchaudio.save(
                out, torch.from_numpy(audio), sample_rate, format=response_format
            )
            return out.getvalue()
