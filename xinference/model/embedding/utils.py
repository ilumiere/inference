# Copyright 2022-2024 XProbe Inc.
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
from .core import EmbeddingModelSpec

# 此模块提供了用于处理嵌入模型的实用函数

def get_model_version(embedding_model: EmbeddingModelSpec) -> str:
    """
    生成嵌入模型的版本字符串。

    此函数接受一个 EmbeddingModelSpec 对象作为输入，并返回一个唯一标识该模型的版本字符串。
    版本字符串由模型名称、最大令牌数和嵌入维度组成，以双破折号 "--" 分隔。

    参数:
    embedding_model (EmbeddingModelSpec): 包含嵌入模型规格的对象。

    返回:
    str: 由模型名称、最大令牌数和嵌入维度组成的版本字符串。

    示例:
    如果 embedding_model 的 model_name 为 "bert", max_tokens 为 512, dimensions 为 768,
    则返回的字符串将是 "bert--512--768"。

    注意:
    - 此函数假设 EmbeddingModelSpec 对象具有 model_name、max_tokens 和 dimensions 属性。
    - 返回的字符串可用于唯一标识和区分不同的嵌入模型配置。
    """
    # 使用 f-string 格式化版本字符串
    # 包含模型名称、最大令牌数和嵌入维度，以双破折号分隔
    return f"{embedding_model.model_name}--{embedding_model.max_tokens}--{embedding_model.dimensions}"
