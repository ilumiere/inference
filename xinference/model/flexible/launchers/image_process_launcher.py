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
from io import BytesIO

import PIL.Image
import PIL.ImageOps

from ....types import Image
from ..core import FlexibleModel, FlexibleModelSpec

class ImageRemoveBackgroundModel(FlexibleModel):
    def infer(self, **kwargs):
        # 获取是否需要反转图像的参数，默认为False
        invert = kwargs.get("invert", False)
        # 获取base64编码的图像数据
        b64_image: str = kwargs.get("image")  # type: ignore
        # 获取是否只返回蒙版的参数，默认为True
        only_mask = kwargs.pop("only_mask", True)
        # 获取输出图像格式，默认为PNG
        image_format = kwargs.pop("image_format", "PNG")
        # 如果没有提供图像数据，抛出异常
        if not b64_image:
            raise ValueError("No image found to remove background")
        # 解码base64图像数据
        image = base64.b64decode(b64_image)

        try:
            # 尝试导入rembg模块
            from rembg import remove
        except ImportError:
            # 如果导入失败，准备错误信息和安装指南
            error_message = "Failed to import module 'rembg'"
            installation_guide = [
                "Please make sure 'rembg' is installed. ",
                "You can install it by visiting the installation section of the git repo:\n",
                "https://github.com/danielgatis/rembg?tab=readme-ov-file#installation",
            ]
            # 抛出ImportError异常，包含错误信息和安装指南
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # 打开图像
        im = PIL.Image.open(BytesIO(image))
        # 使用rembg移除背景
        om = remove(im, only_mask=only_mask, **kwargs)
        # 如果需要反转图像，则进行反转
        if invert:
            om = PIL.ImageOps.invert(om)

        # 创建一个字节流缓冲区
        buffered = BytesIO()
        # 将处理后的图像保存到缓冲区
        om.save(buffered, format=image_format)
        # 将图像数据转换为base64编码的字符串
        img_str = base64.b64encode(buffered.getvalue()).decode()
        # 返回Image对象，包含base64编码的图像数据
        return Image(url=None, b64_json=img_str)


def launcher(model_uid: str, model_spec: FlexibleModelSpec, **kwargs) -> FlexibleModel:
    # 从kwargs中获取任务类型
    task = kwargs.get("task")
    # 从kwargs中获取设备信息
    device = kwargs.get("device")

    # 如果任务是移除背景
    if task == "remove_background":
        # 返回一个ImageRemoveBackgroundModel实例
        return ImageRemoveBackgroundModel(
            model_uid=model_uid,  # 设置模型唯一标识符
            model_path=model_spec.model_uri,  # type: ignore  # 设置模型路径
            device=device,  # 设置设备信息
            config=kwargs,  # 设置配置信息
        )
    else:
        # 如果任务类型未知，抛出ValueError异常
        raise ValueError(f"未知的图像处理任务: {task}")
