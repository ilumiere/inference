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
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Dict, List, Optional, Union

import PIL.Image
from PIL import ImageOps

from ....constants import XINFERENCE_IMAGE_DIR
from ....device_utils import move_model_to_available_device
from ....types import Image, ImageList, LoRA

logger = logging.getLogger(__name__)


class DiffusionModel:
    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        lora_model: Optional[List[LoRA]] = None,
        lora_load_kwargs: Optional[Dict] = None,
        lora_fuse_kwargs: Optional[Dict] = None,
        abilities: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        初始化DiffusionModel实例。

        参数:
            model_uid (str): 模型的唯一标识符
            model_path (Optional[str]): 模型文件的路径
            device (Optional[str]): 模型运行的设备
            lora_model (Optional[List[LoRA]]): LoRA模型列表
            lora_load_kwargs (Optional[Dict]): LoRA加载参数
            lora_fuse_kwargs (Optional[Dict]): LoRA融合参数
            abilities (Optional[List[str]]): 模型支持的能力列表
            **kwargs: 其他模型初始化参数
            
            
            
            DiffusionModel 类：用于加载和管理稳定扩散模型，支持文本到图像、图像到图像和图像修复等功能。

            这个类封装了稳定扩散模型的加载、配置和推理过程，支持多种模型能力和LoRA（Low-Rank Adaptation）微调。

            属性:
                _model_uid (str): 模型的唯一标识符
                _model_path (Optional[str]): 模型文件的路径
                _device (Optional[str]): 模型运行的设备（如 'cpu' 或 'cuda'）
                _model: 主要的模型实例，用于文本到图像生成
                _i2i_model: 图像到图像模型实例
                _inpainting_model: 图像修复模型实例
                _lora_model (Optional[List[LoRA]]): LoRA模型列表
                _lora_load_kwargs (Dict): LoRA加载的参数
                _lora_fuse_kwargs (Dict): LoRA融合的参数
                _abilities (List[str]): 模型支持的能力列表
                _kwargs (Dict): 其他模型初始化参数
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        # when a model has text2image ability,
        # it will be loaded as AutoPipelineForText2Image
        # for image2image and inpainting,
        # we convert to the corresponding model
        self._model = None
        self._i2i_model = None  # image to image model
        self._inpainting_model = None  # inpainting model
        self._lora_model = lora_model
        self._lora_load_kwargs = lora_load_kwargs or {}
        self._lora_fuse_kwargs = lora_fuse_kwargs or {}
        self._abilities = abilities or []
        self._kwargs = kwargs

    def _apply_lora(self):
        """
        应用LoRA（Low-Rank Adaptation）到模型。

        这个方法加载并融合LoRA权重到主模型中，以实现模型的微调。
        """
        if self._lora_model is not None:
            logger.info(
                f"Loading the LoRA with load kwargs: {self._lora_load_kwargs}, fuse kwargs: {self._lora_fuse_kwargs}."
            )
            assert self._model is not None
            for lora_model in self._lora_model:
                # 加载每个LoRA模型
                self._model.load_lora_weights(
                    lora_model.local_path, **self._lora_load_kwargs
                )
            # 融合LoRA权重到主模型
            self._model.fuse_lora(**self._lora_fuse_kwargs)
            logger.info(f"Successfully loaded the LoRA for model {self._model_uid}.")

    def load(self):
        """
        加载稳定扩散模型。

        这个方法根据指定的能力加载适当的模型类型，并配置模型参数。
        它还处理ControlNet、量化和设备分配等高级功能。
        """
        import torch

        # 根据模型能力选择适当的管道类型
        if "text2image" in self._abilities or "image2image" in self._abilities:
            from diffusers import AutoPipelineForText2Image as AutoPipelineModel
        elif "inpainting" in self._abilities:
            from diffusers import AutoPipelineForInpainting as AutoPipelineModel
        else:
            raise ValueError(f"Unknown ability: {self._abilities}")

        # 处理ControlNet（如果指定）
        controlnet = self._kwargs.get("controlnet")
        if controlnet is not None:
            from diffusers import ControlNetModel
            logger.debug("Loading controlnet %s", controlnet)
            self._kwargs["controlnet"] = ControlNetModel.from_pretrained(controlnet)

        # 设置torch数据类型
        torch_dtype = self._kwargs.get("torch_dtype")
        if sys.platform != "darwin" and torch_dtype is None:
            # The following params crashes on Mac M2
            self._kwargs["torch_dtype"] = torch.float16
            self._kwargs["variant"] = "fp16"
            self._kwargs["use_safetensors"] = True
        if isinstance(torch_dtype, str):
            self._kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        # 处理文本编码器的量化
        quantize_text_encoder = self._kwargs.pop("quantize_text_encoder", None)
        if quantize_text_encoder:
            try:
                from transformers import BitsAndBytesConfig, T5EncoderModel
            except ImportError:
                error_message = "Failed to import module 'transformers'"
                installation_guide = [
                    "Please make sure 'transformers' is installed. ",
                    "You can install it by `pip install transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                error_message = "Failed to import module 'bitsandbytes'"
                installation_guide = [
                    "Please make sure 'bitsandbytes' is installed. ",
                    "You can install it by `pip install bitsandbytes`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            for text_encoder_name in quantize_text_encoder.split(","):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                quantization_kwargs = {}
                if torch_dtype:
                    quantization_kwargs["torch_dtype"] = torch_dtype
                # 加载并量化文本编码器
                text_encoder = T5EncoderModel.from_pretrained(
                    self._model_path,
                    subfolder=text_encoder_name,
                    quantization_config=quantization_config,
                    **quantization_kwargs,
                )
                self._kwargs[text_encoder_name] = text_encoder
                self._kwargs["device_map"] = "balanced"

        # 加载主模型
        logger.debug("Loading model %s", AutoPipelineModel)
        self._model = AutoPipelineModel.from_pretrained(
            self._model_path,
            **self._kwargs,
        )

        # 处理CPU卸载或设备分配
        if self._kwargs.get("cpu_offload", False):
            logger.debug("CPU offloading model")
            self._model.enable_model_cpu_offload()
        elif not self._kwargs.get("device_map"):
            logger.debug("Loading model to available device")
            self._model = move_model_to_available_device(self._model)
        # Recommended if your computer has < 64 GB of RAM
        # 启用注意力切片以节省内存
        self._model.enable_attention_slicing()

        # 应用LoRA（如果有）
        self._apply_lora()

    def _call_model(
        self,
        response_format: str,
        model=None,
        **kwargs,
    ):
        """
        调用模型并处理输出结果。

        参数:
        response_format (str): 输出格式，可以是 'url' 或 'b64_json'
        model: 要使用的模型，默认为 self._model
        **kwargs: 传递给模型的其他参数

        返回:
        ImageList: 包含生成图像信息的对象

        功能:
        1. 调用指定的模型生成图像
        2. 根据指定的格式处理生成的图像
        3. 清理内存缓存
        """
        import gc
        from ....device_utils import empty_cache

        # 记录调用参数
        logger.debug("stable diffusion args: %s", kwargs)
        
        # 确定使用的模型
        model = model if model is not None else self._model
        assert callable(model)
        
        # 调用模型生成图像
        images = model(**kwargs).images

        # 清理内存缓存
        gc.collect()
        empty_cache()

        # 根据指定格式处理图像
        if response_format == "url":
            # 将图像保存为文件并返回 URL
            os.makedirs(XINFERENCE_IMAGE_DIR, exist_ok=True)
            image_list = []
            with ThreadPoolExecutor() as executor:
                for img in images:
                    path = os.path.join(XINFERENCE_IMAGE_DIR, uuid.uuid4().hex + ".jpg")
                    image_list.append(Image(url=path, b64_json=None))
                    executor.submit(img.save, path, "jpeg")
            return ImageList(created=int(time.time()), data=image_list)
        elif response_format == "b64_json":
            # 将图像转换为 base64 编码
            def _gen_base64_image(_img):
                buffered = BytesIO()
                _img.save(buffered, format="jpeg")
                return base64.b64encode(buffered.getvalue()).decode()

            with ThreadPoolExecutor() as executor:
                results = list(map(partial(executor.submit, _gen_base64_image), images))  # type: ignore
                image_list = [Image(url=None, b64_json=s.result()) for s in results]
            return ImageList(created=int(time.time()), data=image_list)
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

    @classmethod
    def _filter_kwargs(cls, kwargs: dict):
        """
        过滤模型参数，移除空值。

        参数:
        kwargs (dict): 包含模型参数的字典

        功能:
        移除 'negative_prompt' 和 'num_inference_steps' 中的空值
        """
        for arg in ["negative_prompt", "num_inference_steps"]:
            if not kwargs.get(arg):
                kwargs.pop(arg, None)

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        """
        References:
        https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl
        文本到图像生成方法。

        参数:
        prompt (str): 用于生成图像的文本描述
        n (int): 生成图像的数量，默认为 1
        size (str): 图像尺寸，格式为 "宽度*高度"，默认为 "1024*1024"
        response_format (str): 返回格式，默认为 "url"
        **kwargs: 其他传递给模型的参数

        返回:
        ImageList: 包含生成图像信息的对象

        功能:
        1. 解析图像尺寸
        2. 过滤参数
        3. 调用模型生成图像
        """
        width, height = map(int, re.split(r"[^\d]+", size))
        self._filter_kwargs(kwargs)
        return self._call_model(
            prompt=prompt,
            height=height,
            width=width,
            num_images_per_prompt=n,
            response_format=response_format,
            **kwargs,
        )

    @staticmethod
    def pad_to_multiple(image, multiple=8):
        """
        将图像填充到指定倍数的尺寸。

        参数:
        image: 输入图像
        multiple (int): 填充的倍数，默认为 8

        返回:
        PIL.Image: 填充后的图像

        功能:
        确保图像的宽度和高度都是指定倍数的整数倍
        """
        x, y = image.size
        padding_x = (multiple - x % multiple) % multiple
        padding_y = (multiple - y % multiple) % multiple
        padding = (0, 0, padding_x, padding_y)
        return ImageOps.expand(image, padding)

    def image_to_image(
        self,
        image: PIL.Image,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ):
        """
        图像到图像转换方法。

        参数:
        image (PIL.Image): 输入图像
        prompt (Optional[Union[str, List[str]]]): 正面提示词
        negative_prompt (Optional[Union[str, List[str]]]): 负面提示词
        n (int): 生成图像的数量，默认为 1
        size (Optional[str]): 输出图像尺寸，格式为 "宽度*高度"
        response_format (str): 返回格式，默认为 "url"
        **kwargs: 其他传递给模型的参数

        返回:
        ImageList: 包含生成图像信息的对象

        功能:
        1. 选择合适的模型（controlnet 或 image2image）
        2. 处理图像填充（如果需要）
        3. 设置图像尺寸
        4. 调用模型生成图像
        """
        if "controlnet" in self._kwargs:
            model = self._model
        else:
            if "image2image" not in self._abilities:
                raise RuntimeError(f"{self._model_uid} does not support image2image")
            if self._i2i_model is not None:
                model = self._i2i_model
            else:
                from diffusers import AutoPipelineForImage2Image
                self._i2i_model = model = AutoPipelineForImage2Image.from_pipe(
                    self._model
                )

        if padding_image_to_multiple := kwargs.pop("padding_image_to_multiple", None):
            # Model like SD3 image to image requires image's height and width is times of 16
            # padding the image if specified
            # 对于需要图像尺寸为 16 的倍数的模型（如 SD3），进行填充
            image = self.pad_to_multiple(image, multiple=int(padding_image_to_multiple))

        if size:
            width, height = map(int, re.split(r"[^\d]+", size))
            if padding_image_to_multiple:
                width, height = image.size
            kwargs["width"] = width
            kwargs["height"] = height

        self._filter_kwargs(kwargs)
        return self._call_model(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=n,
            response_format=response_format,
            model=model,
            **kwargs,
        )

    def inpainting(
        self,
        image: PIL.Image,
        mask_image: PIL.Image,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        """
        执行图像修复（inpainting）操作。

        此方法用于在给定的图像上进行修复，根据提供的掩码和提示生成新的图像内容。

        参数:
        - image (PIL.Image): 需要修复的原始图像。
        - mask_image (PIL.Image): 指定需要修复区域的掩码图像。
        - prompt (Optional[Union[str, List[str]]]): 用于指导图像生成的文本提示。
        - negative_prompt (Optional[Union[str, List[str]]]): 用于避免生成特定内容的负面文本提示。
        - n (int): 要生成的图像数量，默认为1。
        - size (str): 输出图像的尺寸，格式为"宽度*高度"，默认为"1024*1024"。
        - response_format (str): 返回结果的格式，默认为"url"。
        - **kwargs: 其他可能传递给模型的参数。

        返回:
        - 调用self._call_model方法的结果，通常是包含生成图像信息的对象。

        异常:
        - RuntimeError: 如果当前模型不支持inpainting功能。

        功能流程:
        1. 检查模型是否支持inpainting。
        2. 根据模型能力选择或创建适当的inpainting模型。
        3. 处理图像尺寸，包括可能的填充操作。
        4. 调用模型执行inpainting操作。
        """
        # 检查模型是否支持inpainting功能
        if "inpainting" not in self._abilities:
            raise RuntimeError(f"{self._model_uid} does not support inpainting")

        # 根据模型能力选择或创建inpainting模型
        if (
            "text2image" in self._abilities or "image2image" in self._abilities
        ) and self._model is not None:
            from diffusers import AutoPipelineForInpainting

            if self._inpainting_model is not None:
                model = self._inpainting_model
            else:
                # 如果inpainting模型不存在，则从现有模型创建
                model = self._inpainting_model = AutoPipelineForInpainting.from_pipe(
                    self._model
                )
        else:
            # 如果模型不支持text2image或image2image，直接使用现有模型
            model = self._model

        # 解析size参数，获取宽度和高度
        width, height = map(int, re.split(r"[^\d]+", size))

        # 处理图像填充（如果需要）
        if padding_image_to_multiple := kwargs.pop("padding_image_to_multiple", None):
            # Model like SD3 inpainting requires image's height and width is times of 16
            # padding the image if specified
            # 某些模型（如SD3）要求图像尺寸为16的倍数
            image = self.pad_to_multiple(image, multiple=int(padding_image_to_multiple))
            mask_image = self.pad_to_multiple(
                mask_image, multiple=int(padding_image_to_multiple)
            )
            # 更新实际图像尺寸
            width, height = image.size

        # 调用模型执行inpainting操作
        return self._call_model(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=n,
            response_format=response_format,
            model=model,
            **kwargs,
        )
