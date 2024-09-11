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

import base64
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, List, Union

import numpy as np
import PIL.Image

from ...constants import XINFERENCE_VIDEO_DIR
from ...device_utils import gpu_count, move_model_to_available_device
from ...types import Video, VideoList

if TYPE_CHECKING:
    from .core import VideoModelFamilyV1


logger = logging.getLogger(__name__)

def export_to_video_imageio(
    video_frames: Union[List[np.ndarray], List["PIL.Image.Image"]],
    output_video_path: str,
    fps: int = 8,
) -> str:
    """
    将视频帧导出为视频文件，使用imageio库以避免"绿屏"问题（例如CogVideoX）。

    此函数接受一系列视频帧，并将它们合成为一个视频文件。它使用imageio库来处理视频编码，
    这有助于避免某些视频生成模型（如CogVideoX）可能出现的"绿屏"问题。

    参数:
    video_frames (Union[List[np.ndarray], List["PIL.Image.Image"]]): 
        包含视频帧的列表。每一帧可以是numpy数组或PIL图像对象。
    output_video_path (str): 
        输出视频文件的保存路径，包括文件名和扩展名。
    fps (int, 可选): 
        视频的帧率，即每秒显示的帧数。默认值为8。

    返回:
    str: 输出视频文件的完整路径。

    函数流程:
    1. 导入imageio库，用于视频编码和写入。
    2. 检查输入帧的类型，如果是PIL图像，将其转换为numpy数组。
    3. 使用imageio创建视频写入器，设置输出路径和帧率。
    4. 遍历所有帧，将每一帧添加到视频中。
    5. 写入器会自动关闭，完成视频的创建。
    6. 返回输出视频文件的路径。

    注意:
    - 此函数依赖于imageio库，确保在使用前已安装该库。
    - 输入帧应该具有一致的尺寸和格式。
    - 输出视频的质量和大小可能受到fps参数的影响。
    """
    import imageio  # 导入imageio库，用于视频处理

    # 如果输入帧是PIL图像，将其转换为numpy数组
    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # 使用imageio创建视频写入器，设置输出路径和帧率
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        # 遍历所有帧，将每一帧添加到视频中
        for frame in video_frames:
            writer.append_data(frame)

    # 返回输出视频文件的路径
    return output_video_path


class DiffUsersVideoModel:
    """
    DiffUsersVideoModel类用于加载和使用视频生成模型

    属性:
    _model_uid: 模型的唯一标识符
    _model_path: 模型文件的路径
    _model_spec: 模型规格对象
    _model: 加载的模型实例
    _kwargs: 额外的模型参数

    方法:
    load(): 加载模型
    text_to_video(): 根据文本提示生成视频
    """

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "VideoModelFamilyV1",
        **kwargs,
    ):
        """
        初始化DiffUsersVideoModel实例

        参数:
        model_uid: 模型的唯一标识符
        model_path: 模型文件的路径
        model_spec: 模型规格对象
        **kwargs: 额外的模型参数
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._model = None
        self._kwargs = kwargs

    @property
    def model_spec(self):
        """返回模型规格对象"""
        return self._model_spec
    def load(self):
        """
        加载视频生成模型

        此方法负责加载和配置视频生成模型。它处理模型的初始化、调度器的设置、以及各种优化技术的应用。

        函数流程:
        1. 导入必要的库
        2. 准备模型配置参数
        3. 根据模型家族加载相应的模型
        4. 配置模型的调度器
        5. 根据需要进行模型优化（如图形编译、CPU卸载等）
        6. 启用注意力切片以优化内存使用

        参数:
        该方法不接受任何参数，但会使用类的属性来配置模型。

        返回值:
        该方法没有明确的返回值，但会设置类的 _model 属性。

        详细说明:
        """
        # 导入PyTorch库，用于深度学习操作
        import torch

        # 复制默认模型配置，并用用户提供的参数更新它
        kwargs = self._model_spec.default_model_config.copy()
        kwargs.update(self._kwargs)

        # 从配置中提取调度器类名，如果存在的话
        scheduler_cls_name = kwargs.pop("scheduler", None)

        # 处理torch_dtype参数，将字符串转换为实际的PyTorch数据类型
        torch_dtype = kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        logger.debug("Loading video model with kwargs: %s", kwargs)

        # 根据模型家族加载相应的模型
        if self._model_spec.model_family == "CogVideoX":
            # 导入必要的diffusers库
            import diffusers
            from diffusers import CogVideoXPipeline

            # 使用预训练的模型初始化pipeline
            pipeline = self._model = CogVideoXPipeline.from_pretrained(
                self._model_path, **kwargs
            )
        else:
            # 如果模型家族不支持，抛出异常
            raise Exception(
                f"Unsupported model family: {self._model_spec.model_family}"
            )

        # 如果指定了调度器，则配置它
        if scheduler_cls_name:
            logger.debug("Using scheduler: %s", scheduler_cls_name)
            pipeline.scheduler = getattr(diffusers, scheduler_cls_name).from_config(
                pipeline.scheduler.config, timestep_spacing="trailing"
            )

        # 如果启用了图形编译，对transformer进行编译优化
        if kwargs.get("compile_graph", False):
            pipeline.transformer = torch.compile(
                pipeline.transformer, mode="max-autotune", fullgraph=True
            )

        # 如果启用了CPU卸载，进行相关设置
        if kwargs.get("cpu_offload", False):
            logger.debug("CPU offloading model")
            pipeline.enable_model_cpu_offload()
            if kwargs.get("sequential_cpu_offload", True):
                pipeline.enable_sequential_cpu_offload()
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        # 如果没有指定设备映射，则将模型移动到可用设备
        elif not kwargs.get("device_map"):
            logger.debug("Loading model to available device")
            if gpu_count() > 1:
                kwargs["device_map"] = "balanced"
            else:
                pipeline = move_model_to_available_device(self._model)

        # 启用注意力切片以优化内存使用，特别适用于内存小于64GB的计算机
        pipeline.enable_attention_slicing()

    def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        num_inference_steps: int = 50,
        response_format: str = "b64_json",
        **kwargs,
    ) -> VideoList:
        """
        根据文本提示生成视频。

        此方法使用预训练的视频生成模型，将文本描述转换为视频序列。它支持生成多个视频，
        并可以自定义推理步骤数和其他生成参数。生成的视频可以以URL或base64编码的形式返回。

        参数:
        prompt (str): 用于生成视频的文本描述。
        n (int): 要生成的视频数量，默认为1。
        num_inference_steps (int): 模型推理的步骤数，默认为50。步骤数越多，生成质量可能越高，但耗时也越长。
        response_format (str): 返回视频的格式，可以是'url'或'b64_json'，默认为'b64_json'。
        **kwargs: 其他可选的生成参数，将与默认配置合并。

        返回:
        VideoList: 包含生成视频信息的对象。每个视频可以是URL链接或base64编码的字符串。

        函数流程:
        1. 导入必要的模块和函数。
        2. 准备生成参数，合并默认配置和用户提供的参数。
        3. 调用模型生成视频帧。
        4. 清理内存和GPU缓存。
        5. 将生成的视频帧导出为MP4文件。
        6. 根据指定的响应格式，返回视频URL或base64编码。
        """
        import gc

        # cv2 bug will cause the video cannot be normally displayed
        # thus we use the imageio one
        # from diffusers.utils import export_to_video
        from ...device_utils import empty_cache

        # 确保模型已加载
        assert self._model is not None
        assert callable(self._model)

        # 准备生成参数
        generate_kwargs = self._model_spec.default_generate_config.copy()
        generate_kwargs.update(kwargs)
        generate_kwargs["num_videos_per_prompt"] = n
        logger.debug(
            "diffusers text_to_video args: %s",
            generate_kwargs,
        )

        # 调用模型生成视频帧
        output = self._model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            **generate_kwargs,
        )

        # 清理内存和GPU缓存
        gc.collect()
        empty_cache()

        # 创建保存视频的目录
        os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)
        urls = []

        # 将生成的帧导出为视频文件
        for f in output.frames:
            path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".mp4")
            p = export_to_video_imageio(f, path, fps=8)
            urls.append(p)

        # 根据响应格式返回结果
        if response_format == "url":
            return VideoList(
                created=int(time.time()),
                data=[Video(url=url, b64_json=None) for url in urls],
            )
        elif response_format == "b64_json":
            def _gen_base64_video(_video_url):
                try:
                    with open(_video_url, "rb") as f:
                        return base64.b64encode(f.read()).decode()
                finally:
                    os.remove(_video_url)

            # 使用线程池并行处理视频编码
            with ThreadPoolExecutor() as executor:
                results = list(map(partial(executor.submit, _gen_base64_video), urls))  # type: ignore
                video_list = [Video(url=None, b64_json=s.result()) for s in results]
            return VideoList(created=int(time.time()), data=video_list)
        else:
            raise ValueError(f"Unsupported response format: {response_format}")
