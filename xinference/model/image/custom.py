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
# 导入所需的模块
import logging
import os
from threading import Lock
from typing import List, Optional

# 导入常量和基础模型类
from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from .core import ImageModelFamilyV1

# 设置日志记录器
logger = logging.getLogger(__name__)

# 创建用于同步的锁
UD_IMAGE_LOCK = Lock()


# 定义自定义图像模型家族类，继承自ImageModelFamilyV1
class CustomImageModelFamilyV1(ImageModelFamilyV1):
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]
    controlnet: Optional[List["CustomImageModelFamilyV1"]]


# 初始化用户定义的图像模型列表
UD_IMAGES: List[CustomImageModelFamilyV1] = []


# 获取用户定义的图像模型列表的函数
def get_user_defined_images() -> List[ImageModelFamilyV1]:
    with UD_IMAGE_LOCK:
        return UD_IMAGES.copy()  # 返回列表的副本以确保线程安全


def register_image(model_spec: CustomImageModelFamilyV1, persist: bool):
    # 导入必要的工具函数和模型列表
    from ..utils import is_valid_model_name, is_valid_model_uri
    from . import BUILTIN_IMAGE_MODELS, MODELSCOPE_IMAGE_MODELS

    # 验证模型名称是否有效
    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"无效的模型名称 {model_spec.model_name}。")

    # 验证模型URI是否有效（如果存在）
    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"无效的模型URI {model_uri}")

    # 使用锁确保线程安全
    with UD_IMAGE_LOCK:
        # 检查模型名称是否与现有模型冲突
        for model_name in (
            list(BUILTIN_IMAGE_MODELS.keys())
            + list(MODELSCOPE_IMAGE_MODELS.keys())
            + [spec.model_name for spec in UD_IMAGES]
        ):
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"模型名称与现有模型 {model_spec.model_name} 冲突"
                )
        # 将新模型规格添加到用户定义的图像模型列表中
        UD_IMAGES.append(model_spec)

    # 如果需要持久化，将模型规格保存到文件
    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "image", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, "w") as f:
            f.write(model_spec.json())


def unregister_image(model_name: str, raise_error: bool = True):
    # 使用锁确保线程安全
    with UD_IMAGE_LOCK:
        model_spec = None
        # 在用户定义的图像模型列表中查找指定名称的模型
        for i, f in enumerate(UD_IMAGES):
            if f.model_name == model_name:
                model_spec = f
                break
        if model_spec:
            # 从列表中移除找到的模型
            UD_IMAGES.remove(model_spec)

            # 构建模型持久化文件的路径
            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "image", f"{model_spec.model_id}.json"
            )

            # 如果持久化文件存在，则删除它
            if os.path.exists(persist_path):
                os.remove(persist_path)

            # 构建模型缓存目录的路径
            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                # 记录警告信息，表示正在删除用户定义模型的缓存
                logger.warning(
                    f"删除用户定义模型 {model_spec.model_name} 的缓存。"
                    f"缓存目录：{cache_dir}"
                )
                # 如果缓存目录是软链接，直接删除
                if os.path.islink(cache_dir):
                    os.remove(cache_dir)
                else:
                    # 如果不是软链接，提示用户手动删除
                    logger.warning(
                        f"缓存目录不是软链接，请手动删除。"
                    )
        else:
            # 如果未找到指定的模型
            if raise_error:
                # 如果设置了raise_error，则抛出异常
                raise ValueError(f"未找到模型 {model_name}。")
            else:
                # 否则只记录警告信息
                logger.warning(f"未找到自定义图像模型 {model_name}。")
