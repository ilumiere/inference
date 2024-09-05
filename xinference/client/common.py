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
    创建一个迭代器来处理流式生成类型。

    注意
    ----------
    此方法是为了与openai兼容。请参考：
    https://github.com/openai/openai-python/blob/v0.28.1/openai/api_requestor.py#L99

    参数
    ----------
    response_lines: Iterator[bytes]
        由模型生成器生成的行。

    返回
    -------
    Iterator["CompletionChunk"]
        由模型生成的CompletionChunks的迭代器。

    """

    for line in response_lines:
        # 去除行两端的空白字符
        line = line.strip()
        # 检查行是否以'data:'开头
        if line.startswith(b"data:"):
            # 提取JSON字符串
            json_str = line[len(b"data:") :].strip()
            # 如果是结束标记，则跳过
            if json_str == b"[DONE]":
                continue
            # 解析JSON数据
            data = json.loads(json_str.decode("utf-8"))
            # 检查是否有错误
            error = data.get("error")
            if error is not None:
                raise Exception(str(error))
            # 生成数据
            yield data
