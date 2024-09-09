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

# 导入必要的模块
import codecs
import json
import os

# 导入常量和核心功能
from ...constants import XINFERENCE_MODEL_DIR
from .core import (
    FLEXIBLE_MODEL_DESCRIPTIONS,
    FlexibleModel,
    FlexibleModelSpec,
    generate_flexible_model_description,
    get_flexible_model_descriptions,
    get_flexible_models,
    register_flexible_model,
    unregister_flexible_model,
)

# 设置灵活模型的目录路径
model_dir = os.path.join(XINFERENCE_MODEL_DIR, "flexible")
if os.path.isdir(model_dir):
    # 遍历目录中的所有文件
    for f in os.listdir(model_dir):
        # 使用 UTF-8 编码打开文件
        with codecs.open(os.path.join(model_dir, f), encoding="utf-8") as fd:
            # 解析文件内容为 FlexibleModelSpec 对象
            model_spec = FlexibleModelSpec.parse_obj(json.load(fd))
            # 注册灵活模型，但不持久化（因为已经存在文件）
            register_flexible_model(model_spec, persist=False)

# 注册模型描述
for model in get_flexible_models():
    # 为每个模型生成描述并更新到 FLEXIBLE_MODEL_DESCRIPTIONS
    FLEXIBLE_MODEL_DESCRIPTIONS.update(generate_flexible_model_description(model))
