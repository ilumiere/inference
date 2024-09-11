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
def _install():
    """
    安装模型相关的组件。

    这个函数是一个内部使用的安装函数，主要用于初始化和设置模型相关的组件。
    它的主要作用是调用大语言模型（LLM）模块中的安装函数。

    函数流程：
    1. 从当前包的 llm 子模块中导入 _install 函数，并将其重命名为 llm_install。
    2. 调用 llm_install 函数来完成大语言模型相关的安装和初始化。

    参数：
    该函数不接受任何参数。

    返回值：
    该函数没有明确的返回值。

    注意：
    - 这是一个内部函数，通常不应该在包外部直接调用。
    - 函数名前的下划线(_)表示这是一个内部使用的私有函数。
    - 这个函数可能在包的初始化过程中被自动调用，用于设置必要的环境和组件。
    """
    from .llm import _install as llm_install

    llm_install()
