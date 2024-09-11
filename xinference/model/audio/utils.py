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
# 导入 AudioModelFamilyV1 类，用于音频模型的规格定义
from .core import AudioModelFamilyV1

# 获取音频模型版本的函数
def get_model_version(audio_model: AudioModelFamilyV1) -> str:
    """
    获取给定音频模型的版本信息。

    此函数用于从 AudioModelFamilyV1 对象中提取模型版本。
    在当前实现中，函数直接返回模型名称作为版本信息。

    参数:
    audio_model (AudioModelFamilyV1): 包含音频模型信息的对象。

    返回:
    str: 表示模型版本的字符串，实际上是模型的名称。

    注意:
    - 这个函数假设模型名称可以作为版本标识符。
    - 如果需要更复杂的版本管理，可能需要修改此函数以返回更具体的版本信息。
    """
    # 直接返回模型名称作为版本信息
    return audio_model.model_name
