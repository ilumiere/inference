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

import json
from typing import Any, Iterator


def streaming_response_iterator(
    response_lines: Iterator[bytes],
) -> Iterator[Any]:
    """
    创建一个迭代器来处理流式生成的响应。

    此函数用于处理模型生成器产生的流式响应，将其转换为与OpenAI API兼容的格式。
    它逐行解析响应，提取JSON数据，并生成相应的完成块（CompletionChunk）。

    注意：
    -----
    此方法旨在与OpenAI兼容。参考：
    https://github.com/openai/openai-python/blob/v0.28.1/openai/api_requestor.py#L99

    参数：
    -----
    response_lines: Iterator[bytes]
        模型生成器产生的字节流迭代器，每个元素代表一行响应数据。

    返回：
    -----
    Iterator[Any]
        返回一个迭代器，每个元素是解析后的完成块（CompletionChunk）。

    异常：
    -----
    Exception
        当解析的数据中包含错误信息时抛出。

    工作流程：
    1. 遍历输入的响应行
    2. 处理每一行数据：
       - 去除首尾空白字符
       - 检查是否以"data:"开头
       - 提取JSON字符串并解码
       - 处理特殊情况（如结束标记）
       - 解析JSON数据
       - 检查错误
       - 生成数据
    """

    for line in response_lines:
        # 去除行首尾的空白字符
        line = line.strip()
        
        # 检查行是否以"data:"开头
        if line.startswith(b"data:"):
            # 提取JSON字符串并去除首尾空白
            json_str = line[len(b"data:") :].strip()
                        
            # 检查是否为结束标记
            if json_str == b"[DONE]":
                continue  # 跳过结束标记
            
            # 解码JSON字符串并解析为Python对象
            data = json.loads(json_str.decode("utf-8"))
            
            # 检查是否存在错误
            error = data.get("error")
            if error is not None:
                raise Exception(str(error))  # 如果存在错误，抛出异常
            
            # 生成解析后的数据
            yield data
