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
from io import BytesIO
from typing import Generator, List, Optional

import gradio as gr
import PIL.Image
from gradio.components import Markdown, Textbox
from gradio.layouts import Accordion, Column, Row

from ..client.restful.restful_client import (
    RESTfulChatModelHandle,
    RESTfulGenerateModelHandle,
)
from ..types import ChatCompletionMessage

logger = logging.getLogger(__name__)


class GradioInterface:
    """
    GradioInterface 类用于构建基于 Gradio 的聊天界面。
    
    该类负责创建和配置不同类型的聊天界面，包括普通聊天、视觉语言聊天等。
    它根据模型的能力和特性来决定构建何种类型的界面。

    属性:
        endpoint (str): API 端点 URL
        model_uid (str): 模型的唯一标识符
        model_name (str): 模型名称
        model_size_in_billions (int): 模型参数量（以十亿为单位）
        model_type (str): 模型类型
        model_format (str): 模型格式
        quantization (str): 量化方法
        context_length (int): 上下文长度
        model_ability (List[str]): 模型能力列表
        model_description (str): 模型描述
        model_lang (List[str]): 模型支持的语言列表
        _access_token (Optional[str]): 访问令牌，用于 API 认证
    """

    def __init__(
        self,
        endpoint: str,
        model_uid: str,
        model_name: str,
        model_size_in_billions: int,
        model_type: str,
        model_format: str,
        quantization: str,
        context_length: int,
        model_ability: List[str],
        model_description: str,
        model_lang: List[str],
        access_token: Optional[str],
    ):
        """
        初始化 GradioInterface 实例。

        参数:
            endpoint (str): API 端点 URL
            model_uid (str): 模型的唯一标识符
            model_name (str): 模型名称
            model_size_in_billions (int): 模型参数量（以十亿为单位）
            model_type (str): 模型类型
            model_format (str): 模型格式
            quantization (str): 量化方法
            context_length (int): 上下文长度
            model_ability (List[str]): 模型能力列表
            model_description (str): 模型描述
            model_lang (List[str]): 模型支持的语言列表
            access_token (Optional[str]): 访问令牌，用于 API 认证
        """
        self.endpoint = endpoint
        self.model_uid = model_uid
        self.model_name = model_name
        self.model_size_in_billions = model_size_in_billions
        self.model_type = model_type
        self.model_format = model_format
        self.quantization = quantization
        self.context_length = context_length
        self.model_ability = model_ability
        self.model_description = model_description
        self.model_lang = model_lang
        self._access_token = (
            access_token.replace("Bearer ", "") if access_token is not None else None
        )

    def build(self) -> "gr.Blocks":
        """
        构建 Gradio 界面。

        根据模型能力选择合适的界面类型，并配置界面属性。

        返回:
            gr.Blocks: 配置好的 Gradio 界面对象
        """
        if "vision" in self.model_ability:
            interface = self.build_chat_vl_interface()
        elif "chat" in self.model_ability:
            interface = self.build_chat_interface()
        else:
            interface = self.build_generate_interface()

        interface.queue()
        # Gradio initiates the queue during a startup event, but since the app has already been
        # started, that event will not run, so manually invoke the startup events.
        # See: https://github.com/gradio-app/gradio/issues/5228
        interface.startup_events()
        # 设置网页图标
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

    def build_chat_interface(
        self,
    ) -> "gr.Blocks":
        """
        构建普通聊天界面。

        返回:
            gr.Blocks: 配置好的聊天界面对象
        """
        def flatten(matrix: List[List[str]]) -> List[str]:
            """
            将二维列表扁平化为一维列表。

            参数:
                matrix (List[List[str]]): 二维字符串列表

            返回:
                List[str]: 扁平化后的一维列表
            """
            flat_list = []
            for row in matrix:
                flat_list += row
            return flat_list

        def to_chat(lst: List[str]) -> List[ChatCompletionMessage]:
            """
            将字符串列表转换为聊天完成消息列表。

            参数:
                lst (List[str]): 字符串列表

            返回:
                List[ChatCompletionMessage]: 聊天完成消息列表
            """
            res = []
            for i in range(len(lst)):
                role = "assistant" if i % 2 == 1 else "user"
                res.append(ChatCompletionMessage(role=role, content=lst[i]))
            return res

        def generate_wrapper(
            message: str,
            history: List[List[str]],
            max_tokens: int,
            temperature: float,
            lora_name: str,
        ) -> Generator:
            """
            聊天生成函数的包装器。

            参数:
                message (str): 用户输入的消息
                history (List[List[str]]): 聊天历史
                max_tokens (int): 生成的最大标记数
                temperature (float): 生成的温度参数
                lora_name (str): LoRA 模型名称

            返回:
                Generator: 生成的响应内容生成器
            """
            from ..client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulChatModelHandle)

            response_content = ""
            for chunk in model.chat(
                prompt=message,
                chat_history=to_chat(flatten(history)),
                generate_config={
                    "max_tokens": int(max_tokens),
                    "temperature": temperature,
                    "stream": True,
                    "lora_name": lora_name,
                },
            ):
                assert isinstance(chunk, dict)
                delta = chunk["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                else:
                    response_content += delta["content"]
                    yield response_content

            yield response_content

        return gr.ChatInterface(
            fn=generate_wrapper,
            additional_inputs=[
                gr.Slider(
                    minimum=1,
                    maximum=self.context_length,
                    value=512,
                    step=1,
                    label="Max Tokens",
                ),
                gr.Slider(
                    minimum=0, maximum=2, value=1, step=0.01, label="Temperature"
                ),
                gr.Text(label="LoRA Name"),
            ],
            title=f"🚀 Xinference Chat Bot : {self.model_name} 🚀",
            css="""
            .center{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 0px;
                color: #9ea4b0 !important;
            }
            """,
            description=f"""
            <div class="center">
            Model ID: {self.model_uid}
            </div>
            <div class="center">
            Model Size: {self.model_size_in_billions} Billion Parameters
            </div>
            <div class="center">
            Model Format: {self.model_format}
            </div>
            <div class="center">
            Model Quantization: {self.quantization}
            </div>
            """,
            analytics_enabled=False,
        )

    def build_chat_vl_interface(
        self,
    ) -> "gr.Blocks":
        """
        构建视觉语言聊天界面。

        返回:
            gr.Blocks: 配置好的视觉语言聊天界面对象
        """
        def predict(history, bot, max_tokens, temperature, stream):
            """
            预测函数，用于生成聊天响应。

            参数:
                history (List): 聊天历史
                bot (List): 机器人响应列表
                max_tokens (int): 生成的最大标记数
                temperature (float): 生成的温度参数
                stream (bool): 是否使用流式输出

            返回:
                Generator: 生成的历史和机器人响应
            """
            from ..client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulChatModelHandle)

            prompt = history[-1]
            assert prompt["role"] == "user"
            prompt = prompt["content"]
            # 多模态聊天不支持流式输出
            if stream:
                response_content = ""
                for chunk in model.chat(
                    prompt=prompt,
                    chat_history=history[:-1],
                    generate_config={
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": stream,
                    },
                ):
                    assert isinstance(chunk, dict)
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue
                    else:
                        response_content += delta["content"]
                        bot[-1][1] = response_content
                        yield history, bot
                history.append(
                    {
                        "content": response_content,
                        "role": "assistant",
                    }
                )
                bot[-1][1] = response_content
                yield history, bot
            else:
                response = model.chat(
                    prompt=prompt,
                    chat_history=history[:-1],
                    generate_config={
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": stream,
                    },
                )
                history.append(response["choices"][0]["message"])
                bot[-1][1] = history[-1]["content"]
                yield history, bot

        def add_text(history, bot, text, image, video):
            """
            添加用户输入的文本、图片或视频到聊天历史记录中。

            此函数处理用户的输入，包括纯文本、图片和视频，并将其添加到聊天历史和机器人响应列表中。

            参数:
            history (List): 聊天历史记录列表
            bot (List): 机器人响应列表
            text (str): 用户输入的文本
            image (str): 用户上传的图片文件路径
            video (str): 用户上传的视频文件路径

            返回:
            Tuple[List, List, str, None, None]: 更新后的历史记录、机器人响应、清空的文本框、清空的图片和视频输入
            """
            logger.debug("Add text, text: %s, image: %s, video: %s", text, image, video)
            
            if image:
                # 处理图片输入
                buffered = BytesIO()
                with PIL.Image.open(image) as img:
                    # 调整图片大小
                    img.thumbnail((500, 500))
                    img.save(buffered, format="JPEG")
                # 将图片转换为base64编码
                img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                # 准备显示内容和消息
                display_content = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />\n{text}'
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64_str}"
                            },
                        },
                    ],
                }
            elif video:
                # 处理视频输入
                def video_to_base64(video_path):
                    """将视频文件转换为base64编码"""
                    with open(video_path, "rb") as video_file:
                        encoded_string = base64.b64encode(video_file.read()).decode(
                            "utf-8"
                        )
                    return encoded_string

                def generate_html_video(video_path):
                    """生成包含视频的HTML代码"""
                    base64_video = video_to_base64(video_path)
                    video_format = video_path.split(".")[-1]
                    html_code = f"""
                    <video controls>
                        <source src="data:video/{video_format};base64,{base64_video}" type="video/{video_format}">
                        Your browser does not support the video tag.
                    </video>
                    """
                    return html_code

                # 准备显示内容和消息
                display_content = f"{generate_html_video(video)}\n{text}"
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "video_url",
                            "video_url": {"url": video},
                        },
                    ],
                }
            else:
                # 处理纯文本输入
                display_content = text
                message = {"role": "user", "content": text}
            
            # 更新历史记录和机器人响应
            history = history + [message]
            bot = bot + [[display_content, None]]
            return history, bot, "", None, None

        def clear_history():
            """
            清空聊天历史记录。

            此函数用于重置聊天界面，清除所有历史记录和输入。

            返回:
            Tuple[List, None, str, None, None]: 空的历史记录、清空的机器人响应、空文本框、清空的图片和视频输入
            """
            logger.debug("Clear history.")
            return [], None, "", None, None

        def update_button(text):
            """
            更新发送按钮的状态。

            根据文本框是否有内容来启用或禁用发送按钮。

            参数:
            text (str): 文本框中的内容

            返回:
            gr.update: Gradio更新对象，用于更新按钮状态
            """
            return gr.update(interactive=bool(text))

        with gr.Blocks(
            title=f"🚀 Xinference Chat Bot : {self.model_name} 🚀",
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
        ) as chat_vl_interface:
            # 创建聊天界面的标题和模型信息
            Markdown(
                f"""
                <h1 style='text-align: center; margin-bottom: 1rem'>🚀 Xinference Chat Bot : {self.model_name} 🚀</h1>
                """
            )
            Markdown(
                f"""
                <div class="center">
                Model ID: {self.model_uid}
                </div>
                <div class="center">
                Model Size: {self.model_size_in_billions} Billion Parameters
                </div>
                <div class="center">
                Model Format: {self.model_format}
                </div>
                <div class="center">
                Model Quantization: {self.quantization}
                </div>
                """
            )

            # 初始化聊天状态
            state = gr.State([])
            with gr.Row():
                # 创建聊天机器人界面
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label=self.model_name, height=700, scale=7
                )
                with gr.Column(scale=3):
                    # 创建图片、视频和文本输入框
                    imagebox = gr.Image(type="filepath")
                    videobox = gr.Video()
                    textbox = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press ENTER",
                        container=False,
                    )
                    # 创建发送和清除按钮
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=False
                    )
                    clear_btn = gr.Button(value="Clear")

            # 创建额外输入选项（折叠面板）
            with gr.Accordion("Additional Inputs", open=False):
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=self.context_length,
                    value=512,
                    step=1,
                    label="Max Tokens",
                )
                temperature = gr.Slider(
                    minimum=0, maximum=2, value=1, step=0.01, label="Temperature"
                )
                stream = gr.Checkbox(label="Stream", value=False)

            # 设置文本框变化时更新按钮状态
            textbox.change(update_button, [textbox], [submit_btn], queue=False)

            # 设置文本框提交事件
            textbox.submit(
                add_text,
                [state, chatbot, textbox, imagebox, videobox],
                [state, chatbot, textbox, imagebox, videobox],
                queue=False,
            ).then(
                predict,
                [state, chatbot, max_tokens, temperature, stream],
                [state, chatbot],
            )

            # 设置发送按钮点击事件
            submit_btn.click(
                add_text,
                [state, chatbot, textbox, imagebox, videobox],
                [state, chatbot, textbox, imagebox, videobox],
                queue=False,
            ).then(
                predict,
                [state, chatbot, max_tokens, temperature, stream],
                [state, chatbot],
            )

            clear_btn.click(
                clear_history,
                None,
                [state, chatbot, textbox, imagebox, videobox],
                queue=False,
            )

        return chat_vl_interface

    def build_generate_interface(
        self,
    ):
        """
        构建生成式模型的交互界面。

        此方法创建一个用于文本生成的 Gradio 界面，包括文本输入、生成控制和历史记录管理。

        返回:
            gr.Blocks: 配置好的 Gradio 界面对象

        主要功能:
        1. 创建文本输入和输出区域
        2. 提供生成、撤销、重试和清除等操作按钮
        3. 允许用户调整生成参数（如最大标记数和温度）
        4. 管理生成历史记录
        """

        def undo(text, hist):
            """
            撤销上一次操作，恢复到前一个状态。

            参数:
                text (str): 当前文本框中的内容
                hist (list): 历史记录列表

            返回:
                dict: 包含更新后的文本框内容和历史记录的字典
            """
            if len(hist) == 0:
                return {
                    textbox: "",
                    history: [text],
                }
            if text == hist[-1]:
                hist = hist[:-1]

            return {
                textbox: hist[-1] if len(hist) > 0 else "",
                history: hist,
            }

        def clear(text, hist):
            """
            清除当前文本并更新历史记录。

            参数:
                text (str): 当前文本框中的内容
                hist (list): 历史记录列表

            返回:
                dict: 包含清空后的文本框内容和更新后的历史记录的字典
            """
            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)
            hist.append("")
            return {
                textbox: "",
                history: hist,
            }

        def complete(text, hist, max_tokens, temperature, lora_name) -> Generator:
            """
            生成文本并更新界面。

            参数:
                text (str): 当前文本框中的内容
                hist (list): 历史记录列表
                max_tokens (int): 生成的最大标记数
                temperature (float): 生成的温度参数
                lora_name (str): LoRA 模型名称

            返回:
                Generator: 生成的文本内容和更新后的历史记录
            """
            from ..client import RESTfulClient

            # 初始化客户端并获取模型
            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulGenerateModelHandle)

            # 更新历史记录
            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)

            response_content = text
            # 使用流式生成文本
            for chunk in model.generate(
                prompt=text,
                generate_config={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    "lora_name": lora_name,
                },
            ):
                assert isinstance(chunk, dict)
                choice = chunk["choices"][0]
                if "text" not in choice:
                    continue
                else:
                    response_content += choice["text"]
                    yield {
                        textbox: response_content,
                        history: hist,
                    }

            # 更新历史记录并返回最终结果
            hist.append(response_content)
            return {  # type: ignore
                textbox: response_content,
                history: hist,
            }

        def retry(text, hist, max_tokens, temperature, lora_name) -> Generator:
            """
            重新生成文本，使用历史记录中的前一个输入。

            参数:
                text (str): 当前文本框中的内容
                hist (list): 历史记录列表
                max_tokens (int): 生成的最大标记数
                temperature (float): 生成的温度参数
                lora_name (str): LoRA 模型名称

            返回:
                Generator: 重新生成的文本内容和更新后的历史记录
            """
            from ..client import RESTfulClient

            # 初始化客户端并获取模型
            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulGenerateModelHandle)

            # 更新历史记录并获取前一个输入
            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)
            text = hist[-2] if len(hist) > 1 else ""

            response_content = text
            # 使用流式生成文本
            for chunk in model.generate(
                prompt=text,
                generate_config={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    "lora_name": lora_name,
                },
            ):
                assert isinstance(chunk, dict)
                choice = chunk["choices"][0]
                if "text" not in choice:
                    continue
                else:
                    response_content += choice["text"]
                    yield {
                        textbox: response_content,
                        history: hist,
                    }

            # 更新历史记录并返回最终结果
            hist.append(response_content)
            return {  # type: ignore
                textbox: response_content,
                history: hist,
            }

        # 创建 Gradio 界面
        with gr.Blocks(
            title=f"🚀 Xinference Generate Bot : {self.model_name} 🚀",
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
        ) as generate_interface:
            # 初始化历史记录状态
            history = gr.State([])

            # 添加标题和模型信息
            Markdown(
                f"""
                <h1 style='text-align: center; margin-bottom: 1rem'>🚀 Xinference Generate Bot : {self.model_name} 🚀</h1>
                """
            )
            Markdown(
                f"""
                <div class="center">
                Model ID: {self.model_uid}
                </div>
                <div class="center">
                Model Size: {self.model_size_in_billions} Billion Parameters
                </div>
                <div class="center">
                Model Format: {self.model_format}
                </div>
                <div class="center">
                Model Quantization: {self.quantization}
                </div>
                """
            )

            # 创建主要界面元素
            with Column(variant="panel"):
                # 文本输入框
                textbox = Textbox(
                    container=False,
                    show_label=False,
                    label="Message",
                    placeholder="Type a message...",
                    lines=21,
                    max_lines=50,
                )

                # 操作按钮
                with Row():
                    btn_generate = gr.Button("Generate", variant="primary")
                with Row():
                    btn_undo = gr.Button("↩️  Undo")
                    btn_retry = gr.Button("🔄  Retry")
                    btn_clear = gr.Button("🗑️  Clear")

                # 附加输入选项（折叠面板）
                with Accordion("Additional Inputs", open=False):
                    length = gr.Slider(
                        minimum=1,
                        maximum=self.context_length,
                        value=1024,
                        step=1,
                        label="Max Tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0, maximum=2, value=1, step=0.01, label="Temperature"
                    )
                    lora_name = gr.Text(label="LoRA Name")

                # 设置按钮点击事件
                btn_generate.click(
                    fn=complete,
                    inputs=[textbox, history, length, temperature, lora_name],
                    outputs=[textbox, history],
                )

                btn_undo.click(
                    fn=undo,
                    inputs=[textbox, history],
                    outputs=[textbox, history],
                )

                btn_retry.click(
                    fn=retry,
                    inputs=[textbox, history, length, temperature, lora_name],
                    outputs=[textbox, history],
                )

                btn_clear.click(
                    fn=clear,
                    inputs=[textbox, history],
                    outputs=[textbox, history],
                )

        return generate_interface
