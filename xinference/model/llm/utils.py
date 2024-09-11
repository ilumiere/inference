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
import functools
import json
import logging
import os
import time
import uuid
from io import BytesIO
from typing import AsyncGenerator, Dict, Iterator, List, Optional, Tuple, cast

import requests
from PIL import Image

from ...types import (
    SPECIAL_TOOL_PROMPT,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)
from ..utils import ensure_cache_cleared
from .llm_family import (
    LlamaCppLLMSpecV1,
    LLMFamilyV1,
    LLMSpecV1,
    PromptStyleV1,
    _get_cache_dir,
    get_cache_status,
)

logger = logging.getLogger(__name__)


QWEN_TOOL_CALL_FAMILY = [
    "qwen-chat",
    "qwen1.5-chat",
    "qwen1.5-moe-chat",
    "qwen2-instruct",
    "qwen2-moe-instruct",
]

GLM4_TOOL_CALL_FAMILY = [
    "glm4-chat",
    "glm4-chat-1m",
]


class ChatModelMixin:
    @staticmethod
    def get_prompt(
        prompt: str,
        chat_history: List[ChatCompletionMessage],
        prompt_style: PromptStyleV1,
        tools: Optional[List[Dict]] = None,
    ):
        """
        根据不同模型的提示风格将聊天历史格式化为提示。

        此函数受FastChat启发，用于处理和格式化聊天历史，生成适合特定模型的提示。

        参数:
        prompt (str): 当前用户输入的提示。
        chat_history (List[ChatCompletionMessage]): 聊天历史记录列表。
        prompt_style (PromptStyleV1): 定义了特定模型的提示风格。
        tools (Optional[List[Dict]]): 可选的工具列表，用于某些模型的特殊处理。

        返回:
        str 或 Tuple[str, List]: 格式化后的提示字符串，或者对于某些特殊模型，返回提示字符串和图像列表的元组。

        函数流程:
        1. 确保prompt_style.roles不为None。
        2. 如果prompt不是特殊工具提示，将其添加到聊天历史中。
        3. 在聊天历史中添加一个空的助手消息。
        4. 定义get_role函数来获取角色名称。
        5. 根据不同的prompt_style.style_name处理聊天历史：
           - 每种风格都有特定的格式化逻辑
           - 处理系统提示、用户消息、助手消息等
           - 某些风格还包括工具调用的特殊处理
        6. 返回格式化后的提示字符串。

        注意:
        - 函数包含多个条件分支，每个分支对应不同的提示风格。
        - 某些风格（如INTERNVL）可能返回额外的图像信息。
        - 函数处理各种特殊情况，如工具调用、系统提示等。
        """
        # 确保prompt_style.roles不为None
        assert prompt_style.roles is not None
        # 如果prompt不是特殊工具提示，将其添加到聊天历史中
        if prompt != SPECIAL_TOOL_PROMPT:
            chat_history.append(
                ChatCompletionMessage(role=prompt_style.roles[0], content=prompt)
            )
        # 在聊天历史中添加一个空的助手消息
        chat_history.append(
            ChatCompletionMessage(role=prompt_style.roles[1], content="")
        )

        # 定义一个函数来获取角色名称
        def get_role(role_name: str):
            if role_name == "user":
                return prompt_style.roles[0]
            elif role_name == "assistant":
                return prompt_style.roles[1]
            else:
                return role_name

        # 根据不同的提示风格处理聊天历史
        if prompt_style.style_name == "ADD_COLON_SINGLE":
            ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + ": " + content + prompt_style.intra_message_sep
                else:
                    ret += role + ":"
            return ret
        elif prompt_style.style_name == "NO_COLON_TWO":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = prompt_style.system_prompt
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + content + seps[i % 2]
                else:
                    ret += role
            return ret
        elif prompt_style.style_name == "LLAMA2":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = ""
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    if i == 0:
                        ret += prompt_style.system_prompt + content
                    else:
                        ret += role + " " + content + seps[i % 2]
                else:
                    ret += role
            return ret
        elif prompt_style.style_name == "LLAMA3":
            ret = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"{prompt_style.intra_message_sep}{prompt_style.system_prompt}{prompt_style.inter_message_sep}"
            )
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += (
                        f"<|start_header_id|>{role}<|end_header_id|>"
                        f"{prompt_style.intra_message_sep}{content}{prompt_style.inter_message_sep}"
                    )
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>{prompt_style.intra_message_sep}"
            return ret
        elif prompt_style.style_name == "MIXTRAL_V01":
            ret = ""
            for i, message in enumerate(chat_history):
                content = message["content"]
                if i % 2 == 0:  # 用户消息
                    ret += f"<s> [INST] {content} [/INST]"
                else:  # 助手消息
                    ret += f"{content} </s>"
            return ret
        elif prompt_style.style_name == "CHATGLM3":
            # 如果有系统提示，将其添加到提示列表中
            prompts = (
                [f"<|system|>\n {prompt_style.system_prompt}"]
                if prompt_style.system_prompt
                else []
            )

            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    content = tool_calls[0]["function"]
                if content:
                    if role == "tool":
                        role = "observation"
                    prompts.append(f"<|{role}|>\n {content}")
                else:
                    prompts.append(f"<|{role}|>")
            return "\n".join(prompts)
        elif prompt_style.style_name == "XVERSE":
            ret = (
                f"<|system|> \n {prompt_style.system_prompt}"
                if prompt_style.system_prompt
                else ""
            )
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += f"<|{role}|> \n {content}"
                else:
                    ret += f"<|{role}|>"
            return ret
        elif prompt_style.style_name == "QWEN":
            # 如果提供了工具，生成工具描述和指令
            if tools:
                tool_desc = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

                react_instruction = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""
                tools_text = []
                tools_name_text = []
                for func_info in tools:
                    parameters = []
                    fp = func_info["function"].get("parameters", {})
                    if fp:
                        required_parameters = fp.get("required", [])
                        for name, p in fp["properties"].items():
                            param = dict({"name": name}, **p)
                            if name in required_parameters:
                                param["required"] = True
                            parameters.append(param)

                    name = func_info["function"]["name"]
                    desc = func_info["function"]["description"]
                    tool_string = tool_desc.format(
                        name_for_model=name,
                        name_for_human=name,
                        # Hint: You can add the following format requirements in description:
                        #   "Format the arguments as a JSON object."
                        #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                        description_for_model=desc,
                        parameters=json.dumps(parameters, ensure_ascii=False),
                    )
                    tools_text.append(tool_string)
                    tools_name_text.append(name)
                tools_text_string = "\n\n".join(tools_text)
                tools_name_text_string = ", ".join(tools_name_text)
                tool_system = react_instruction.format(
                    tools_text=tools_text_string,
                    tools_name_text=tools_name_text_string,
                )
            else:
                tool_system = ""

            # 生成提示
            ret = f"<|im_start|>system\n{prompt_style.system_prompt}<|im_end|>"
            for message in chat_history:
                role = get_role(message["role"])
                content = message.get("content")

                ret += prompt_style.intra_message_sep
                if tools:
                    if role == "user":
                        if tool_system:
                            content = tool_system + f"\n\nQuestion: {content}"
                            tool_system = ""
                        else:
                            content = f"Question: {content}"
                    elif role == "assistant":
                        tool_calls = message.get("tool_calls")
                        if tool_calls:
                            func_call = tool_calls[0]["function"]
                            f_name, f_args = (
                                func_call["name"],
                                func_call["arguments"],
                            )
                            content = f"Thought: I can use {f_name}.\nAction: {f_name}\nAction Input: {f_args}"
                        elif content:
                            content = f"Thought: I now know the final answer.\nFinal answer: {content}"
                    elif role == "tool":
                        role = "function"
                        content = f"Observation: {content}"
                    else:
                        raise Exception(f"Unsupported message role: {role}")
                if content:
                    content = content.lstrip("\n").rstrip()
                    ret += f"<|im_start|>{role}\n{content}<|im_end|>"
                else:
                    ret += f"<|im_start|>{role}\n"
            return ret
        elif prompt_style.style_name == "CHATML":
            ret = (
                ""
                if prompt_style.system_prompt == ""
                else prompt_style.system_prompt + prompt_style.intra_message_sep + "\n"
            )
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]

                if content:
                    ret += role + "\n" + content + prompt_style.intra_message_sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif prompt_style.style_name == "INTERNLM2":
            ret = (
                "<s>"
                if prompt_style.system_prompt == ""
                else "<s><|im_start|>system\n"
                + prompt_style.system_prompt
                + prompt_style.intra_message_sep
                + "\n"
            )
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]

                if content:
                    ret += role + "\n" + content + prompt_style.intra_message_sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif prompt_style.style_name == "ADD_COLON_SINGLE_COT":
            ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + ": " + content + prompt_style.intra_message_sep
                else:
                    ret += role + ": Let's think step by step."
            return ret
        elif prompt_style.style_name == "DEEPSEEK_CHAT":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = prompt_style.system_prompt
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + ": " + content + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif prompt_style.style_name == "DEEPSEEK_CODER":
            sep = prompt_style.inter_message_sep
            ret = prompt_style.system_prompt + sep
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + "\n" + content + sep
                else:
                    ret += role + "\n"
            return ret
        elif prompt_style.style_name == "GORILLA_OPENFUNCTIONS":
            if tools:
                gorilla_functions = []
                for tool in tools:
                    gorilla_functions.append(
                        {
                            "name": tool["function"]["name"],
                            "api_name": tool["function"]["name"],
                            "description": tool["function"]["description"],
                            "parameters": [
                                dict({"name": name}, **p)
                                for name, p in tool["function"]["parameters"][
                                    "properties"
                                ].items()
                            ],
                        }
                    )
                tools_string = json.dumps(gorilla_functions)
                return f"USER: <<question>> {prompt} <<function>> {tools_string}\nASSISTANT: "
            else:
                return f"USER: <<question>> {prompt}\nASSISTANT: "
        elif prompt_style.style_name == "orion":
            ret = "<s>"
            for i, message in enumerate(chat_history):
                content = message["content"]
                role = get_role(message["role"])
                if i % 2 == 0:  # Human
                    assert content is not None
                    ret += role + ": " + content + "\n\n"
                else:  # Assistant
                    if content:
                        ret += role + ": </s>" + content + "</s>"
                    else:
                        ret += role + ": </s>"
            return ret
        elif prompt_style.style_name == "gemma":
            ret = ""
            for message in chat_history:
                content = message["content"]
                role = get_role(message["role"])
                ret += "<start_of_turn>" + role + "\n"
                if content:
                    ret += content + "<end_of_turn>\n"
            return ret
        elif prompt_style.style_name == "CodeShell":
            ret = ""
            for message in chat_history:
                content = message["content"]
                role = get_role(message["role"])
                if content:
                    ret += f"{role}{content}|<end>|"
                else:
                    ret += f"{role}".rstrip()
            return ret
        elif prompt_style.style_name == "MINICPM-2B":
            ret = ""
            for message in chat_history:
                content = message["content"] or ""
                role = get_role(message["role"])
                if role == "user":
                    ret += "<用户>" + content.strip()
                else:
                    ret += "<AI>" + content.strip()
            return ret
        elif prompt_style.style_name == "PHI3":
            ret = f"<|system|>{prompt_style.intra_message_sep}{prompt_style.system_prompt}{prompt_style.inter_message_sep}"
            for message in chat_history:
                content = message["content"] or ""
                role = get_role(message["role"])
                if content:
                    ret += f"<|{role}|>{prompt_style.intra_message_sep}{content}{prompt_style.inter_message_sep}"
                else:
                    ret += f"<|{role}|>{prompt_style.intra_message_sep}"
            ret += "<|assistant|>\n"
            return ret
        elif prompt_style.style_name == "c4ai-command-r":
            ret = (
                f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
                f"{prompt_style.system_prompt}{prompt_style.inter_message_sep}"
            )
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += f"{role}{content}{prompt_style.inter_message_sep}"
                else:
                    ret += role
            return ret
        elif prompt_style.style_name == "mistral-nemo":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = "<s>"
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    if i == len(chat_history) - 2 and prompt_style.system_prompt:
                        ret += (
                            role
                            + " "
                            + prompt_style.system_prompt
                            + "\n\n"
                            + content
                            + seps[i % 2]
                        )
                    else:
                        ret += role + " " + content + seps[i % 2]
                else:
                    ret += role
            return ret
        elif prompt_style.style_name == "INTERNVL":
            ret = (
                "<s>"
                if prompt_style.system_prompt == ""
                else "<s><|im_start|>system\n"
                + prompt_style.system_prompt
                + prompt_style.intra_message_sep
                + "\n"
            )
            images = []  # type: ignore
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]
                if isinstance(content, str):
                    if content:
                        ret += (
                            role
                            + "\n"
                            + content
                            + prompt_style.intra_message_sep
                            + "\n"
                        )
                    else:
                        ret += role + "\n"
                elif isinstance(content, list):
                    text = ""
                    image_urls = []
                    for c in content:
                        c_type = c.get("type")
                        if c_type == "text":
                            text = c["text"]
                        elif c_type == "image_url":
                            image_urls.append(c["image_url"]["url"])
                    image_futures = []
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor() as executor:
                        for image_url in image_urls:
                            fut = executor.submit(_decode_image, image_url)
                            image_futures.append(fut)
                    images = [fut.result() for fut in image_futures]
                    if len(image_futures) == 0:
                        ret += (
                            role + "\n" + text + prompt_style.intra_message_sep + "\n"
                        )
                    else:
                        ret += (
                            role
                            + "\n"
                            + f"<image>\n{text}"
                            + prompt_style.intra_message_sep
                            + "\n"
                        )

            return (ret, images)
        else:
            raise ValueError(f"Invalid prompt style: {prompt_style.style_name}")

    @classmethod
    def _to_chat_completion_chunk(cls, chunk: CompletionChunk) -> ChatCompletionChunk:
        """
        将CompletionChunk转换为ChatCompletionChunk格式。

        此类方法用于处理和转换完成块（CompletionChunk）为聊天完成块（ChatCompletionChunk）格式。
        它处理两种情况：已经是ChatCompletionChunk格式的块和需要转换的CompletionChunk。

        参数:
        cls (type): 类本身，用于访问类方法。
        chunk (CompletionChunk): 需要转换的完成块。

        返回:
        ChatCompletionChunk: 转换后的聊天完成块。

        方法流程:
        1. 检查输入块是否已经是ChatCompletionChunk格式。
        2. 如果是，直接返回输入块。
        3. 如果不是，创建新的聊天完成块结构。
        4. 转换每个选择（choice）的内容，包括处理可能的工具调用。
        5. 返回转换后的ChatCompletionChunk。
        """
        choices = chunk.get("choices")
        # 检查是否已经是ChatCompletionChunk格式
        if (
            chunk.get("object") == "chat.completion.chunk"
            and choices
            and "delta" in choices[0]
        ):
            # 如果已经是ChatCompletionChunk，直接返回
            return cast(ChatCompletionChunk, chunk)
        
        # 创建新的聊天完成块结构
        chat_chunk = {
            "id": "chat" + chunk["id"],  # 添加"chat"前缀到ID
            "model": chunk["model"],     # 复制模型信息
            "created": chunk["created"], # 复制创建时间戳
            "object": "chat.completion.chunk",  # 设置对象类型
            "choices": [
                {
                    "index": i,  # 设置选择的索引
                    "delta": {
                        "content": choice.get("text"),  # 获取文本内容
                        # 如果存在工具调用，添加到delta中
                        **(
                            {"tool_calls": choice["tool_calls"]}
                            if "tool_calls" in choice
                            else {}
                        ),
                    },
                    "finish_reason": choice["finish_reason"],  # 设置完成原因
                }
                for i, choice in enumerate(chunk["choices"])
            ],
        }
        # 将结果转换为ChatCompletionChunk类型并返回
        return cast(ChatCompletionChunk, chat_chunk)

    @classmethod
    def _get_first_chat_completion_chunk(
        cls, chunk: CompletionChunk
    ) -> ChatCompletionChunk:
        """
        生成第一个聊天完成块。

        此类方法用于处理流式响应的第一个块，将其转换为聊天完成格式。

        参数:
        cls (type): 类本身，用于访问类方法。
        chunk (CompletionChunk): 输入的完成块，通常是流式响应的第一个块。

        返回:
        ChatCompletionChunk: 转换后的聊天完成块。

        方法流程:
        1. 创建基本的聊天完成块结构。
        2. 为每个选择创建初始delta信息。
        3. 将结果转换为ChatCompletionChunk类型并返回。
        """
        chat_chunk = {
            "id": "chat" + chunk["id"],  # 添加"chat"前缀到ID
            "model": chunk["model"],     # 复制模型信息
            "created": chunk["created"], # 复制创建时间戳
            "object": "chat.completion.chunk",  # 设置对象类型
            "choices": [
                {
                    "index": i,  # 设置选择的索引
                    "delta": {
                        "role": "assistant",  # 设置角色为助手
                        "content": "",        # 初始内容为空字符串
                    },
                    "finish_reason": None,  # 初始完成原因为None
                }
                for i, choice in enumerate(chunk["choices"])
            ],
        }
        return cast(ChatCompletionChunk, chat_chunk)  # 将结果转换为ChatCompletionChunk类型并返回

    @classmethod
    def _get_final_chat_completion_chunk(
        cls, chunk: CompletionChunk
    ) -> ChatCompletionChunk:
        """
        生成最终的聊天完成块。

        此类方法用于处理流式响应的最后一个块，将其转换为聊天完成格式。

        参数:
        cls (type): 类本身，用于访问类方法。
        chunk (CompletionChunk): 输入的完成块，通常是流式响应的最后一个块。

        返回:
        ChatCompletionChunk: 转换后的聊天完成块。

        方法流程:
        1. 创建基本的聊天完成块结构。
        2. 添加使用情况信息（如果有）。
        3. 将结果转换为ChatCompletionChunk类型并返回。
        """
        # 创建基本的聊天完成块结构
        chat_chunk = {
            "id": "chat" + chunk["id"],  # 添加"chat"前缀到ID
            "model": chunk["model"],     # 复制模型信息
            "created": chunk["created"], # 复制创建时间戳
            "object": "chat.completion.chunk",  # 设置对象类型
            "choices": [],  # 最终块通常没有选择，因此为空列表
        }
        
        # 获取使用情况信息（如果有）
        usage = chunk.get("usage")
        if usage is not None:
            chat_chunk["usage"] = usage  # 添加使用情况信息到聊天块
        
        # 将结果转换为ChatCompletionChunk类型并返回
        return cast(ChatCompletionChunk, chat_chunk)

    @classmethod
    @ensure_cache_cleared
    def _to_chat_completion_chunks(
        cls,
        chunks: Iterator[CompletionChunk],
    ) -> Iterator[ChatCompletionChunk]:
        """
        将CompletionChunk迭代器转换为ChatCompletionChunk迭代器。

        此类方法用于处理流式响应，将标准完成格式的chunks转换为聊天完成格式。
        它使用了@ensure_cache_cleared装饰器来确保在执行过程中清除缓存。

        参数:
        cls (type): 类本身，用于访问类方法。
        chunks (Iterator[CompletionChunk]): 输入的CompletionChunk迭代器。

        返回:
        Iterator[ChatCompletionChunk]: 转换后的ChatCompletionChunk迭代器。

        方法流程:
        1. 遍历输入的chunks迭代器。
        2. 对每个chunk进行处理和转换。
        3. 根据不同情况yield不同类型的ChatCompletionChunk。
        """
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 处理第一个chunk，生成并yield初始的聊天完成chunk
                yield cls._get_first_chat_completion_chunk(chunk)
            
            # 获取chunk中的choices
            choices = chunk.get("choices")
            if not choices:
                # 如果没有choices，说明是最后一个chunk，生成并yield最终的聊天完成chunk
                yield cls._get_final_chat_completion_chunk(chunk)
            else:
                # 否则，将当前chunk转换为聊天完成chunk并yield
                yield cls._to_chat_completion_chunk(chunk)

    @classmethod
    async def _async_to_chat_completion_chunks(
        cls,
        chunks: AsyncGenerator[CompletionChunk, None],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        异步方法，将CompletionChunk异步生成器转换为ChatCompletionChunk异步生成器。

        此方法用于处理流式响应，将标准完成格式的chunks转换为聊天完成格式。

        参数:
        cls (type): 类本身，用于访问类方法。
        chunks (AsyncGenerator[CompletionChunk, None]): 输入的CompletionChunk异步生成器。

        返回:
        AsyncGenerator[ChatCompletionChunk, None]: 转换后的ChatCompletionChunk异步生成器。

        方法流程:
        1. 初始化计数器i，用于标识第一个chunk。
        2. 异步迭代输入的chunks。
        3. 对每个chunk进行处理和转换。
        4. 根据不同情况yield不同类型的ChatCompletionChunk。
        """
        i = 0
        async for chunk in chunks:
            if i == 0:
                # 处理第一个chunk，生成初始的聊天完成chunk
                yield cls._get_first_chat_completion_chunk(chunk)
            
            # 获取chunk中的choices
            choices = chunk.get("choices")
            if not choices:
                # 如果没有choices，说明是最后一个chunk，生成最终的聊天完成chunk
                yield cls._get_final_chat_completion_chunk(chunk)
            else:
                # 否则，将当前chunk转换为聊天完成chunk
                yield cls._to_chat_completion_chunk(chunk)
            
            # 增加计数器
            i += 1

    @staticmethod
    @ensure_cache_cleared
    def _to_chat_completion(completion: Completion) -> ChatCompletion:
        """
        将Completion对象转换为ChatCompletion对象。

        此静态方法用于将标准的Completion响应格式转换为ChatCompletion格式，
        以适应聊天完成API的需求。

        参数:
        completion (Completion): 原始的完成响应对象。

        返回:
        ChatCompletion: 转换后的聊天完成响应对象。

        函数流程:
        1. 创建新的字典，设置基本属性（id, object, created, model）。
        2. 转换choices列表，将每个选择项适配为聊天格式。
        3. 复制usage信息。

        注意:
        - 使用@ensure_cache_cleared装饰器确保在执行此方法前清除缓存。
        - id字段前添加"chat"前缀，以区分普通完成和聊天完成。
        """
        return {
            # 为id添加"chat"前缀
            "id": "chat" + completion["id"],
            # 设置对象类型为聊天完成
            "object": "chat.completion",
            # 复制创建时间戳
            "created": completion["created"],
            # 复制模型信息
            "model": completion["model"],
            # 转换选择列表
            "choices": [
                {
                    "index": i,  # 保持原始索引
                    "message": {
                        "role": "assistant",  # 在聊天中，响应总是来自助手
                        "content": choice["text"],  # 将文本内容作为消息内容
                    },
                    "finish_reason": choice["finish_reason"],  # 复制完成原因
                }
                for i, choice in enumerate(completion["choices"])
            ],
            # 复制使用情况统计
            "usage": completion["usage"],
        }

    @staticmethod
    def _eval_gorilla_openfunctions_arguments(c, tools):
        """
        评估Gorilla OpenFunctions的参数。

        此静态方法用于解析和评估Gorilla OpenFunctions模型的输出，提取工具调用的相关信息。
        它尝试执行模型生成的代码，以获取工具调用的详细信息。

        参数:
        c (dict): 包含模型输出的字典，预期包含'choices'键。
        tools (list): 可用工具的列表，用于提取工具名称。

        返回:
        tuple: 包含以下三个元素：
            - a (Any): 通常为None，除非eval执行结果特别指定。
            - b (str or None): 被调用的工具名称，如果解析失败则为None。
            - c (dict or None): 工具调用的参数字典，如果解析失败则为None。

        工作流程:
        1. 从tools中提取所有工具的名称。
        2. 从模型输出中获取参数文本。
        3. 定义一个tool_call函数，用于模拟工具调用。
        4. 尝试执行（eval）模型生成的代码。
        5. 如果执行成功，返回解析后的工具调用信息。
        6. 如果执行失败，记录错误并返回原始参数文本。

        异常处理:
        - 捕获所有可能的异常，记录错误信息，并返回原始参数文本。

        注意:
        - 此方法使用eval执行模型生成的代码，可能存在安全风险，应在受控环境中使用。
        - 返回的a通常为None，除非eval执行的代码特别指定了返回值。
        """
        # 从tools中提取所有工具的名称
        tool_names = [tool["function"]["name"] for tool in tools]
        # 获取模型生成的参数文本
        arguments = c["choices"][0]["text"]

        # 定义tool_call函数，用于模拟工具调用
        def tool_call(n, **kwargs):
            return None, n, kwargs

        try:
            # 尝试执行模型生成的代码
            a, b, c = eval(
                arguments, {n: functools.partial(tool_call, n) for n in tool_names}
            )
            return a, b, c
        except Exception as e:
            # 如果执行失败，记录错误并返回原始参数文本
            logger.error("Eval tool calls completion failed: %s", e)
            return arguments, None, None

    @staticmethod
    def _eval_glm_chat_arguments(c, tools):
        """
        评估GLM聊天模型的工具调用参数。

        此静态方法用于解析GLM聊天模型的输出，提取工具调用的相关信息。
        它处理两种可能的输出格式：字符串或包含工具调用信息的字典。

        参数:
        c (list): 包含模型输出的列表。通常是一个单元素列表，其中包含字符串或字典。
        tools (list): 可用工具的列表。在当前实现中未直接使用。

        返回:
        tuple: 包含以下三个元素：
            - content (str or None): 如果输出是字符串，则返回该字符串；否则为None。
            - func_name (str or None): 如果输出是字典且包含'name'键，则返回函数名；否则为None。
            - func_args (dict or None): 如果输出是字典且包含'parameters'键，则返回参数字典；否则为None。

        异常处理:
        - 如果无法解析输出（例如，遇到KeyError），将记录错误并返回字符串化的输入作为内容。

        注意:
        - 此方法假设GLM模型的输出格式为列表的第一个元素，可能是字符串或字典。
        - 如果输出是字典，预期包含'name'和'parameters'键，分别对应函数名和参数。
        """
        try:
            # 检查输入的第一个元素是否为字符串
            if isinstance(c[0], str):
                # 如果是字符串，直接返回该字符串作为内容，函数名和参数均为None
                return c[0], None, None
            # 如果不是字符串，假定为字典，返回函数名和参数
            return None, c[0]["name"], c[0]["parameters"]
        except KeyError:
            # 如果解析失败（例如，字典中缺少预期的键），记录错误并返回字符串化的输入
            logger.error("无法解析GLM输出: %s", c)
            return str(c), None, None

    @staticmethod
    def _eval_qwen_chat_arguments(c, tools):
        """
        评估Qwen聊天模型的工具调用参数。

        此方法用于解析Qwen聊天模型的输出，提取工具调用的相关信息。它遵循Qwen模型的特定输出格式，
        包括Action、Action Input和Observation等关键词。

        参数:
        c (dict): 包含模型输出的字典，通常包含'choices'键。
        tools (list): 可用工具的列表，此参数在当前实现中未直接使用。

        返回:
        tuple: 包含以下三个元素：
            - content (str): 提取的内容或最终答案。
            - func_name (str or None): 被调用的函数名，如果没有则为None。
            - func_args (dict or None): 函数的参数字典，如果没有则为None。

        注意:
        - 此方法基于Qwen模型的特定输出格式进行解析，参考了Qwen官方仓库的实现。
        - 方法会处理可能的格式变化，如缺少Observation的情况。
        - 如果无法解析为工具调用，将返回文本内容作为答案。
        """
        # 获取模型输出的文本内容
        text = c["choices"][0]["text"]
        try:
            # Refer to:
            # https://github.com/QwenLM/Qwen/blob/main/examples/react_prompt.md
            # https://github.com/QwenLM/Qwen/blob/main/openai_api.py#L297
            func_name, func_args, content = "", "", ""
            
            # 查找关键词的位置
            i = text.rfind("\nAction:")
            j = text.rfind("\nAction Input:")
            k = text.rfind("\nObservation:")
            t = max(
                text.rfind("\nThought:", 0, i), text.rfind("Thought:", 0, i)
            )  # find the last thought just before Action, considering the Thought at the very beginning
            if 0 <= i < j:  # If the text has `Action` and `Action input`,
                if k < j:  # but does not contain `Observation`,
                    # then it is likely that `Observation` is omitted by the LLM,
                    # because the output text may have discarded the stop word.
                    text = text.rstrip() + "\nObservation:"  # Add it back.
                    k = text.rfind("\nObservation:")

            # 如果找到了完整的Action序列
            if 0 <= t < i < j < k:
                # 提取函数名、参数和内容
                func_name = text[i + len("\nAction:") : j].strip()
                func_args = text[j + len("\nAction Input:") : k].strip()
                content = text[
                    t + len("\nThought:") : i
                ].strip()  # len("\nThought:") and len("Thought:") both are OK since there is a space after :
            if func_name:
                return content, func_name, json.loads(func_args)
        except Exception as e:
            # 记录解析失败的错误
            logger.error("Eval tool calls completion failed: %s", e)

        # 如果无法解析为工具调用，提取文本内容作为答案
        t = max(text.rfind("\nThought:"), text.rfind("Thought:"))
        z = max(text.rfind("\nFinal Answer:"), text.rfind("Final Answer:"))
        if z >= 0:
            text = text[
                z + len("\nFinal Answer:") :
            ]  # len("\nFinal Answer::") and len("Final Answer::") both are OK since there is a space after :
        else:
            text = text[
                t + len("\nThought:") :
            ]  # There is only Thought: no Final Answer:
        return text, None, None

    @classmethod
    def _eval_tool_arguments(cls, model_family, c, tools):
        """
        评估工具参数并返回相应的内容、函数名和参数。

        此方法根据不同的模型家族来解析工具调用的结果，并返回标准化的输出。

        参数:
        cls (type): 类本身，用于访问类方法。
        model_family (object): 包含模型家族信息的对象。
        c (dict): 包含模型输出的字典。
        tools (list): 可用工具的列表。

        返回:
        tuple: 包含以下三个元素：
            - content (str): 工具调用的内容或结果。
            - func (str or None): 被调用的函数名，如果没有则为None。
            - args (dict or None): 函数的参数，如果没有则为None。

        异常:
        Exception: 如果模型不支持工具调用，则抛出异常。
        """
        # 获取模型家族名称，优先使用model_family属性，如果不存在则使用model_name
        family = model_family.model_family or model_family.model_name

        # 根据不同的模型家族选择相应的评估方法
        if family in ["gorilla-openfunctions-v1", "gorilla-openfunctions-v2"]:
            # 对于Gorilla OpenFunctions模型，使用专门的评估方法
            content, func, args = cls._eval_gorilla_openfunctions_arguments(c, tools)
        elif family in GLM4_TOOL_CALL_FAMILY:
            # 对于GLM4聊天模型，使用GLM聊天评估方法
            content, func, args = cls._eval_glm_chat_arguments(c, tools)
        elif family in QWEN_TOOL_CALL_FAMILY:
            # 对于Qwen聊天模型，使用Qwen聊天评估方法
            content, func, args = cls._eval_qwen_chat_arguments(c, tools)
        else:
            # 如果模型家族不在支持的列表中，抛出异常
            raise Exception(
                f"Model {model_family.model_name} is not support tool calls."
            )

        # 记录调试信息，包括工具调用的内容、函数名和参数
        logger.debug("Tool call content: %s, func: %s, args: %s", content, func, args)

        # 返回评估结果
        return content, func, args

    @classmethod
    def _tools_token_filter(cls, model_family):
        """
        Generates a filter function for Qwen series models to retain outputs after "\nFinal Answer:".

        Returns:
            A function that takes tokens (string output by the model so far) and delta (new tokens added) as input,
            returns the part after "\nFinal Answer:" if found, else returns delta.
        """
        family = model_family.model_family or model_family.model_name
        if family in QWEN_TOOL_CALL_FAMILY:
            # 封装函数以在每次调用后重置'found'
            found = False

            def process_tokens(tokens: str, delta: str):
                """
                处理输入的令牌，查找并返回"\nFinal Answer:"之后的内容。

                参数:
                tokens (str): 模型目前为止的输出字符串。
                delta (str): 新增的令牌。

                返回:
                str: 如果找到"\nFinal Answer:"，返回其后的内容；否则返回空字符串或delta。
                """
                nonlocal found
                # Once "Final Answer:" is found, future tokens are allowed.
                if found:
                    return delta
                # Check if the token ends with "\nFinal Answer:" and update `found`.
                final_answer_idx = tokens.lower().rfind("\nfinal answer:")
                if final_answer_idx != -1:
                    found = True
                    return tokens[final_answer_idx + len("\nfinal answer:") :]
                return ""

            return process_tokens
        else:
            # 对于非Qwen系列模型，直接返回所有新增的令牌
            return lambda tokens, delta: delta

    @classmethod
    def _tool_calls_completion_chunk(cls, model_family, model_uid, c, tools):
        """
        生成工具调用完成的分块响应。

        此方法用于处理工具调用的完成情况，并生成相应的分块响应格式。它处理工具调用的结果，
        并根据是否有工具被调用来构造不同的响应结构。

        参数:
        - cls: 类方法的类引用
        - model_family: 模型家族，用于评估工具参数
        - model_uid: 模型的唯一标识符
        - c: 包含完成信息的字典
        - tools: 可用的工具列表

        返回:
        - dict: 包含完成信息的字典，符合特定的分块响应格式

        执行步骤:
        1. 生成唯一标识符
        2. 评估工具参数
        3. 根据是否有工具被调用构造消息
        4. 尝试获取使用情况信息
        5. 构造并返回完整的分块响应字典
        """
        # 生成唯一标识符
        _id = str(uuid.uuid4())
        
        # 评估工具参数
        content, func, args = cls._eval_tool_arguments(model_family, c, tools)
        
        # 根据是否有工具被调用构造消息
        if func:
            # 如果有工具被调用，构造包含工具调用信息的消息
            d = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": f"call_{_id}",
                        "type": "function",
                        "function": {
                            "name": func,
                            "arguments": json.dumps(args),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            # 如果没有工具被调用，构造普通的助手消息
            d = {"role": "assistant", "content": content, "tool_calls": []}
            finish_reason = "stop"
        
        # 尝试获取使用情况信息
        try:
            usage = c.get("usage")
            assert "prompt_tokens" in usage
        except Exception:
            # 如果获取失败，使用默认值
            usage = {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            }
        
        # 构造并返回完整的分块响应字典
        return {
            "id": "chat" + f"cmpl-{_id}",
            "model": model_uid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "delta": d,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    @classmethod
    def _tool_calls_completion(cls, model_family, model_uid, c, tools):
        """
        生成工具调用完成的响应。

        此方法用于处理工具调用的完成情况，并生成相应的响应格式。它处理工具调用的结果，
        并根据是否有工具被调用来构造不同的响应结构。

        参数:
        - cls: 类方法的类引用
        - model_family: 模型家族，用于评估工具参数
        - model_uid: 模型的唯一标识符
        - c: 包含完成信息的字典
        - tools: 可用的工具列表

        返回:
        - dict: 包含完成信息的字典，符合特定的响应格式

        执行步骤:
        1. 生成唯一标识符
        2. 评估工具参数
        3. 根据是否有工具被调用构造消息
        4. 尝试获取使用情况信息
        5. 构造并返回完整的响应字典
        """
        # 生成唯一标识符
        _id = str(uuid.uuid4())
        
        # 评估工具参数
        content, func, args = cls._eval_tool_arguments(model_family, c, tools)
        
        # 根据是否有工具被调用构造消息
        if func:
            # 如果有工具被调用，构造包含工具调用信息的消息
            m = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": f"call_{_id}",
                        "type": "function",
                        "function": {
                            "name": func,
                            "arguments": json.dumps(args),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            # 如果没有工具被调用，构造普通的助手消息
            m = {"role": "assistant", "content": content, "tool_calls": []}
            finish_reason = "stop"
        
        # 尝试获取使用情况信息
        try:
            usage = c.get("usage")
            assert "prompt_tokens" in usage
        except Exception:
            # 如果获取失败，使用默认值
            usage = {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            }
        
        # 构造并返回完整的响应字典
        return {
            "id": "chat" + f"cmpl-{_id}",
            "model": model_uid,
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": m,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    @classmethod
    def get_full_prompt(cls, model_family, prompt, system_prompt, chat_history, tools):
        """
        生成完整的提示文本。

        此方法用于构建包含系统提示、聊天历史和用户提示的完整提示文本。它考虑了模型家族的提示风格，
        并可以包含可选的系统提示和工具信息。

        参数:
        - model_family: 模型家族对象，包含提示风格信息。
        - prompt: 用户的当前提示文本。
        - system_prompt: 可选的系统提示文本。
        - chat_history: 聊天历史记录列表，默认为空列表。
        - tools: 可选的工具信息。

        返回:
        - full_prompt: 构建的完整提示文本。

        执行步骤:
        1. 确保模型家族有定义的提示风格。
        2. 复制模型家族的提示风格以避免修改原始对象。
        3. 如果提供了系统提示，则更新提示风格中的系统提示。
        4. 确保聊天历史是一个列表（即使为空）。
        5. 调用cls.get_prompt方法生成完整的提示文本。
        """
        # 确保模型家族有定义的提示风格
        assert model_family.prompt_style is not None
        # 复制提示风格以避免修改原始对象
        prompt_style = model_family.prompt_style.copy()
        # 如果提供了系统提示，则更新提示风格
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        # 确保聊天历史是一个列表
        chat_history = chat_history or []
        # 生成完整的提示文本
        full_prompt = cls.get_prompt(prompt, chat_history, prompt_style, tools=tools)
        return full_prompt


def get_file_location(
    llm_family: LLMFamilyV1, spec: LLMSpecV1, quantization: str
) -> Tuple[str, bool]:
    # 获取缓存目录
    cache_dir = _get_cache_dir(
        llm_family, spec, quantization, create_if_not_exist=False
    )
    # 获取缓存状态
    cache_status = get_cache_status(llm_family, spec, quantization)
    if isinstance(cache_status, list):
        is_cached = None
        # 遍历量化列表，找到匹配的量化状态
        for q, cs in zip(spec.quantizations, cache_status):
            if q == quantization:
                is_cached = cs
                break
    else:
        is_cached = cache_status
    assert isinstance(is_cached, bool)

    # 根据模型格式返回不同的文件位置信息
    if spec.model_format in ["pytorch", "gptq", "awq", "fp8", "mlx"]:
        return cache_dir, is_cached
    elif spec.model_format in ["ggufv2"]:
        assert isinstance(spec, LlamaCppLLMSpecV1)
        # 构建模型文件名
        filename = spec.model_file_name_template.format(quantization=quantization)
        # 获取完整的模型路径
        model_path = os.path.join(cache_dir, filename)
        return model_path, is_cached
    else:
        # 不支持的模型格式抛出异常
        raise ValueError(f"不支持的模型格式 {spec.model_format}")


def get_model_version(
    llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
) -> str:
    # 生成并返回模型版本字符串
    # 格式: 模型名称--模型大小(单位:十亿参数)--模型格式--量化方式
    return f"{llm_family.model_name}--{llm_spec.model_size_in_billions}B--{llm_spec.model_format}--{quantization}"


def _decode_image(_url):
    # 解码图像URL或base64编码的图像数据
    if _url.startswith("data:"):
        logging.info("Parse url by base64 decoder.")
        # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
        # e.g. f"data:image/jpeg;base64,{base64_image}"
        _type, data = _url.split(";")
        _, ext = _type.split("/")
        # 移除base64前缀
        data = data[len("base64,") :]
        # 解码base64数据
        data = base64.b64decode(data.encode("utf-8"))
        # 打开并转换图像为RGB格式
        return Image.open(BytesIO(data)).convert("RGB")
    else:
        try:
            # 尝试从URL下载图像
            response = requests.get(_url)
        except requests.exceptions.MissingSchema:
            # 如果URL不合法，尝试直接打开本地文件
            return Image.open(_url).convert("RGB")
        else:
            # 从下载的内容中打开图像并转换为RGB格式
            return Image.open(BytesIO(response.content)).convert("RGB")
