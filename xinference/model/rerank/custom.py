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
import logging
import os
from threading import Lock
from typing import List, Literal, Optional

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from .core import RerankModelSpec

# 设置日志记录器
logger = logging.getLogger(__name__)

# 创建一个线程锁，用于保护对自定义重排序模型的访问
UD_RERANK_LOCK = Lock()

# 定义自定义重排序模型规格类，继承自RerankModelSpec
class CustomRerankModelSpec(RerankModelSpec):
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]
    model_type: Literal["rerank"] = "rerank"  # 用于前端识别

# 存储自定义重排序模型的列表
UD_RERANKS: List[CustomRerankModelSpec] = []

# 获取用户定义的重排序模型列表
def get_user_defined_reranks() -> List[CustomRerankModelSpec]:
    with UD_RERANK_LOCK:
        return UD_RERANKS.copy()

# 注册新的重排序模型
def register_rerank(model_spec: CustomRerankModelSpec, persist: bool):
    from ...constants import XINFERENCE_MODEL_DIR
    from ..utils import is_valid_model_name, is_valid_model_uri
    from . import BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS

    # 验证模型名称
    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"无效的模型名称 {model_spec.model_name}。")

    # 验证模型URI（如果提供）
    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"无效的模型URI {model_uri}。")

    with UD_RERANK_LOCK:
        # 检查模型名称是否与现有模型冲突
        for model_name in (
            list(BUILTIN_RERANK_MODELS.keys())
            + list(MODELSCOPE_RERANK_MODELS.keys())
            + [spec.model_name for spec in UD_RERANKS]
        ):
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"模型名称与现有模型 {model_spec.model_name} 冲突"
                )

        # 添加新模型到列表
        UD_RERANKS.append(model_spec)

    # 如果需要持久化，将模型规格保存到文件
    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "rerank", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())

# 注销重排序模型
def unregister_rerank(model_name: str, raise_error: bool = True):
    with UD_RERANK_LOCK:
        model_spec = None
        # 查找要注销的模型
        for i, f in enumerate(UD_RERANKS):
            if f.model_name == model_name:
                model_spec = f
                break
        if model_spec:
            # 从列表中移除模型
            UD_RERANKS.remove(model_spec)

            # 删除持久化文件（如果存在）
            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "rerank", f"{model_spec.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            # 处理缓存目录
            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"正在移除用户定义模型 {model_spec.model_name} 的缓存。"
                    f"缓存目录：{cache_dir}"
                )
                if os.path.islink(cache_dir):
                    os.remove(cache_dir)
                else:
                    logger.warning(
                        f"缓存目录不是软链接，请手动删除。"
                    )
        else:
            if raise_error:
                raise ValueError(f"未找到模型 {model_name}")
            else:
                logger.warning(f"未找到自定义重排序模型 {model_name}")
