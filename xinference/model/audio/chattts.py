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

# ChatTTSModel 类：用于处理文本到语音转换的模型
# 该类封装了 ChatTTS 模型的加载、初始化和推理功能

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from ..utils import set_all_random_seed

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class ChatTTSModel:
    """
    ChatTTSModel 类：用于加载和使用 ChatTTS 模型进行文本到语音转换
    
    属性:
    - _model_uid: 模型的唯一标识符
    - _model_path: 模型文件的路径
    - _model_spec: 模型规格
    - _device: 运行模型的设备（如 CPU 或 GPU）
    - _model: 加载的 ChatTTS 模型实例
    - _kwargs: 其他关键字参数
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
        初始化 ChatTTSModel 实例
        
        参数:
        - model_uid: 模型的唯一标识符
        - model_path: 模型文件的路径
        - model_spec: 模型规格
        - device: 运行模型的设备（可选）
        - **kwargs: 其他关键字参数
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    def load(self):
        """
        加载 ChatTTS 模型
        
        该方法导入必要的库，设置 PyTorch 配置，并加载 ChatTTS 模型
        """
        import ChatTTS
        import torch

        # 设置 PyTorch 动态优化器的缓存大小和错误抑制
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
        
        # 设置浮点数矩阵乘法精度
        torch.set_float32_matmul_precision("high")
        
        # 初始化 ChatTTS 模型
        self._model = ChatTTS.Chat()
        
        # 从指定路径加载模型，并编译
        self._model.load(source="custom", custom_path=self._model_path, compile=True)

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
    ):
        """
        将输入文本转换为语音
        
        参数:
        - input: 要转换为语音的输入文本
        - voice: 语音特征（可以是编码后的语音嵌入或用于生成随机说话人的种子）
        - response_format: 输出音频格式，默认为 mp3
        - speed: 语音速度，默认为 1.0
        - stream: 是否以流式方式生成音频，默认为 False
        
        返回:
        - 如果 stream 为 False，返回生成的音频数据
        - 如果 stream 为 True，返回一个生成器，用于流式输出音频数据
        """
        import ChatTTS
        import numpy as np
        import torch
        import torchaudio
        import xxhash

        rnd_spk_emb = None

        # 尝试解码和加载提供的语音嵌入
        if len(voice) > 400:
            try:
                assert self._model is not None
                b = base64.b64decode(voice)
                bio = BytesIO(b)
                tensor = torch.load(bio, map_location="cpu")
                rnd_spk_emb = self._model._encode_spk_emb(tensor)
                logger.info("Speech by input speaker")
            except Exception as e:
                logger.info("Fallback to random speaker due to %s", e)

        # 如果没有提供有效的语音嵌入，生成随机说话人
        if rnd_spk_emb is None:
            seed = xxhash.xxh32_intdigest(voice)

            # 设置随机种子以确保结果可重现
            set_all_random_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            assert self._model is not None
            rnd_spk_emb = self._model.sample_random_speaker()
            logger.info("Speech by voice %s", voice)

        # 设置推理参数
        default = 5
        infer_speed = int(default * speed)
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt=f"[speed_{infer_speed}]", spk_emb=rnd_spk_emb
        )

        assert self._model is not None
        if stream:
            # 流式生成音频
            iter = self._model.infer(
                [input], params_infer_code=params_infer_code, stream=True
            )

            def _generator():
                with BytesIO() as out:
                    writer = torchaudio.io.StreamWriter(out, format=response_format)
                    writer.add_audio_stream(sample_rate=24000, num_channels=1)
                    i = 0
                    last_pos = 0
                    with writer.open():
                        for it in iter:
                            for itt in it:
                                for chunk in itt:
                                    chunk = np.array([chunk]).transpose()
                                    writer.write_audio_chunk(i, torch.from_numpy(chunk))
                                    new_last_pos = out.tell()
                                    if new_last_pos != last_pos:
                                        out.seek(last_pos)
                                        encoded_bytes = out.read()
                                        yield encoded_bytes
                                        last_pos = new_last_pos

            return _generator()
        else:
            # 非流式生成音频
            wavs = self._model.infer([input], params_infer_code=params_infer_code)

            # 保存生成的音频
            with BytesIO() as out:
                torchaudio.save(
                    out, torch.from_numpy(wavs[0]), 24000, format=response_format
                )
                return out.getvalue()
