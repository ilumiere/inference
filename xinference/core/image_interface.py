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
import io
import logging
import os
from typing import Dict, List, Optional, Union

import gradio as gr
import PIL.Image
from gradio import Markdown

from ..client.restful.restful_client import RESTfulImageModelHandle

logger = logging.getLogger(__name__)


class ImageInterface:
    """
    图像生成界面类，用于构建和管理图像生成模型的用户界面。

    该类提供了一个通用的接口，用于创建基于Stable Diffusion模型的图像生成界面。
    它支持文本到图像和图像到图像的转换功能。

    属性:
        endpoint (str): API端点URL
        model_uid (str): 模型的唯一标识符
        model_family (str): 模型所属的系列（如'stable_diffusion'）
        model_name (str): 模型名称
        model_id (str): 模型ID
        model_revision (str): 模型版本
        model_ability (List[str]): 模型支持的能力列表
        controlnet (Union[None, List[Dict[str, Union[str, None]]]]): ControlNet配置
        access_token (Optional[str]): 用于API认证的访问令牌
    """

    def __init__(
        self,
        endpoint: str,
        model_uid: str,
        model_family: str,
        model_name: str,
        model_id: str,
        model_revision: str,
        model_ability: List[str],
        controlnet: Union[None, List[Dict[str, Union[str, None]]]],
        access_token: Optional[str],
    ):
        """
        初始化ImageInterface实例。

        参数:
            endpoint (str): API端点URL
            model_uid (str): 模型的唯一标识符
            model_family (str): 模型所属的系列
            model_name (str): 模型名称
            model_id (str): 模型ID
            model_revision (str): 模型版本
            model_ability (List[str]): 模型支持的能力列表
            controlnet (Union[None, List[Dict[str, Union[str, None]]]]): ControlNet配置
            access_token (Optional[str]): 用于API认证的访问令牌
        """
        self.endpoint = endpoint
        self.model_uid = model_uid
        self.model_family = model_family
        self.model_name = model_name
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_ability = model_ability
        self.controlnet = controlnet
        self.access_token = (
            access_token.replace("Bearer ", "") if access_token is not None else None
        )

    def build(self) -> gr.Blocks:
        """
        构建主界面。

        返回:
            gr.Blocks: 配置好的Gradio界面对象

        该方法执行以下步骤:
        1. 确保模型属于'stable_diffusion'系列
        2. 构建主界面
        3. 配置界面队列
        4. 手动触发启动事件
        5. 设置网页图标
        """
        assert "stable_diffusion" in self.model_family

        interface = self.build_main_interface()
        interface.queue()
        # Gradio initiates the queue during a startup event, but since the app has already been
        # started, that event will not run, so manually invoke the startup events.
        # See: https://github.com/gradio-app/gradio/issues/5228
        interface.startup_events()
        favicon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.pardir,
            "web",
            "ui",
            "public",
            "favicon.svg",
        )
        interface.favicon_path = favicon_path
        return interface

    def text2image_interface(self) -> "gr.Blocks":
        """
        构建文本到图像转换的界面。

        返回:
            gr.Blocks: 文本到图像转换的Gradio界面对象

        该方法定义了一个内部函数text_generate_image，用于处理文本到图像的转换请求。
        然后，它创建一个Gradio界面，包含输入字段（如提示词、图像数量、尺寸等）和输出图库。
        """

        def text_generate_image(
            prompt: str,
            n: int,
            size_width: int,
            size_height: int,
            num_inference_steps: int,
            negative_prompt: Optional[str] = None,
        ) -> PIL.Image.Image:
            """
            根据文本提示生成图像。

            参数:
                prompt (str): 用于生成图像的文本提示
                n (int): 要生成的图像数量
                size_width (int): 生成图像的宽度
                size_height (int): 生成图像的高度
                num_inference_steps (int): 推理步骤数
                negative_prompt (Optional[str]): 负面提示词，用于指定不希望出现在图像中的元素

            返回:
                List[PIL.Image.Image]: 生成的图像列表

            该函数执行以下步骤:
            1. 创建RESTful客户端并获取模型
            2. 设置图像生成参数
            3. 调用模型的text_to_image方法生成图像
            4. 将返回的base64编码图像数据转换为PIL.Image对象
            5. 返回生成的图像列表
            """
            from ..client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulImageModelHandle)

            size = f"{int(size_width)}*{int(size_height)}"
            num_inference_steps = (
                None if num_inference_steps == -1 else num_inference_steps  # type: ignore
            )

            response = model.text_to_image(
                prompt=prompt,
                n=n,
                size=size,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                response_format="b64_json",
            )

            images = []
            for image_dict in response["data"]:
                assert image_dict["b64_json"] is not None
                image_data = base64.b64decode(image_dict["b64_json"])
                image = PIL.Image.open(io.BytesIO(image_data))
                images.append(image)

            return images

        with gr.Blocks() as text2image_vl_interface:
            # 创建文本到图像界面的Gradio组件
            # 包括提示词输入、负面提示词输入、生成按钮、图像参数设置和输出图库
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=10):
                        prompt = gr.Textbox(
                            label="Prompt",
                            show_label=True,
                            placeholder="Enter prompt here...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            show_label=True,
                            placeholder="Enter negative prompt here...",
                        )
                    with gr.Column(scale=1):
                        generate_button = gr.Button("Generate")

                with gr.Row():
                    n = gr.Number(label="Number of Images", value=1)
                    size_width = gr.Number(label="Width", value=1024)
                    size_height = gr.Number(label="Height", value=1024)
                    num_inference_steps = gr.Number(
                        label="Inference Step Number", value=-1
                    )

                with gr.Column():
                    image_output = gr.Gallery()

            generate_button.click(
                text_generate_image,
                inputs=[
                    prompt,
                    n,
                    size_width,
                    size_height,
                    num_inference_steps,
                    negative_prompt,
                ],
                outputs=image_output,
            )

        return text2image_vl_interface

    def image2image_interface(self) -> "gr.Blocks":
        """
        构建图像到图像转换的界面。

        返回:
            gr.Blocks: 图像到图像转换的Gradio界面对象

        该方法定义了一个内部函数image_generate_image，用于处理图像到图像的转换请求。
        然后，它创建一个Gradio界面，包含输入字段（如提示词、图像上传、参数设置等）和输出图库。
        """

        def image_generate_image(
            prompt: str,
            negative_prompt: str,
            image: PIL.Image.Image,
            n: int,
            size_width: int,
            size_height: int,
            num_inference_steps: int,
            padding_image_to_multiple: int,
        ) -> PIL.Image.Image:
            """
            根据输入图像和文本提示生成新图像。

            参数:
                prompt (str): 用于生成图像的文本提示
                negative_prompt (str): 负面提示词
                image (PIL.Image.Image): 输入的源图像
                n (int): 要生成的图像数量
                size_width (int): 生成图像的宽度
                size_height (int): 生成图像的高度
                num_inference_steps (int): 推理步骤数
                padding_image_to_multiple (int): 图像填充到的倍数

            返回:
                List[PIL.Image.Image]: 生成的图像列表

            该函数执行以下步骤:
            1. 创建RESTful客户端并获取模型
            2. 设置图像生成参数
            3. 将输入图像转换为字节流
            4. 调用模型的image_to_image方法生成新图像
            5. 将返回的base64编码图像数据转换为PIL.Image对象
            6. 返回生成的图像列表
            """
            from ..client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulImageModelHandle)

            if size_width > 0 and size_height > 0:
                size = f"{int(size_width)}*{int(size_height)}"
            else:
                size = None
            num_inference_steps = (
                None if num_inference_steps == -1 else num_inference_steps  # type: ignore
            )
            padding_image_to_multiple = None if padding_image_to_multiple == -1 else padding_image_to_multiple  # type: ignore

            bio = io.BytesIO()
            image.save(bio, format="png")

            response = model.image_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                image=bio.getvalue(),
                size=size,
                response_format="b64_json",
                num_inference_steps=num_inference_steps,
                padding_image_to_multiple=padding_image_to_multiple,
            )

            images = []
            for image_dict in response["data"]:
                assert image_dict["b64_json"] is not None
                image_data = base64.b64decode(image_dict["b64_json"])
                image = PIL.Image.open(io.BytesIO(image_data))
                images.append(image)

            return images

        with gr.Blocks() as image2image_inteface:
            # 创建图像到图像界面的Gradio组件
            # 包括提示词输入、负面提示词输入、图像上传、参数设置和输出图库
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=10):
                        prompt = gr.Textbox(
                            label="Prompt",
                            show_label=True,
                            placeholder="Enter prompt here...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            show_label=True,
                            placeholder="Enter negative prompt here...",
                        )
                    with gr.Column(scale=1):
                        generate_button = gr.Button("Generate")

                with gr.Row():
                    n = gr.Number(label="Number of image", value=1)
                    size_width = gr.Number(label="Width", value=-1)
                    size_height = gr.Number(label="Height", value=-1)

                with gr.Row():
                    num_inference_steps = gr.Number(
                        label="Inference Step Number", value=-1
                    )
                    padding_image_to_multiple = gr.Number(
                        label="Padding image to multiple", value=-1
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_image = gr.Image(type="pil", label="Upload Image")
                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery()

            generate_button.click(
                image_generate_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    uploaded_image,
                    n,
                    size_width,
                    size_height,
                    num_inference_steps,
                    padding_image_to_multiple,
                ],
                outputs=output_gallery,
            )
        return image2image_inteface

    def build_main_interface(self) -> "gr.Blocks":
        """
        构建主界面，包括文本到图像和图像到图像的选项卡。

        返回:
            gr.Blocks: 主界面的Gradio对象

        该方法创建一个包含多个选项卡的主界面，每个选项卡对应一种图像生成功能。
        它还设置了界面的标题、CSS样式和其他元数据。
        """
        with gr.Blocks(
            title=f"🎨 Xinference Stable Diffusion: {self.model_name} 🎨",
            css="""
                    .center{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 0px;
                        color: #9ea4b0 !important;
                    }
                    """,
            analytics_enabled=False,
        ) as app:
            Markdown(
                f"""
                    <h1 class="center" style='text-align: center; margin-bottom: 1rem'>🎨 Xinference Stable Diffusion: {self.model_name} 🎨</h1>
                    """
            )
            Markdown(
                f"""
                    <div class="center">
                    Model ID: {self.model_uid}
                    </div>
                    """
            )
            if "text2image" in self.model_ability:
                with gr.Tab("Text to Image"):
                    self.text2image_interface()
            if "image2image" in self.model_ability:
                with gr.Tab("Image to Image"):
                    self.image2image_interface()

        return app
