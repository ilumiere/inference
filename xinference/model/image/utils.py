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
from typing import Optional

from .core import ImageModelFamilyV1


def get_model_version(
    image_model: ImageModelFamilyV1, controlnet: Optional[ImageModelFamilyV1]
) -> str:
    """
    获取模型版本字符串。

    Args:
        image_model (ImageModelFamilyV1): 图像模型。
        controlnet (Optional[ImageModelFamilyV1]): 控制网络模型，可选。

    Returns:
        str: 模型版本字符串。
            如果没有controlnet，返回image_model的名称；
            如果有controlnet，返回image_model和controlnet名称的组合。
    """
    return (
        image_model.model_name
        if controlnet is None
        else f"{image_model.model_name}--{controlnet.model_name}"
    )
