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
from typing import List, Optional

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from .core import EmbeddingModelSpec

# 设置日志记录器
logger = logging.getLogger(__name__)

# 用于保护用户定义的嵌入模型列表的线程锁
UD_EMBEDDING_LOCK = Lock()

# 自定义嵌入模型规格类，继承自EmbeddingModelSpec
class CustomEmbeddingModelSpec(EmbeddingModelSpec):
    model_id: Optional[str]  # 模型ID，可选
    model_revision: Optional[str]  # 模型版本，可选
    model_uri: Optional[str]  # 模型URI，可选

# 存储用户定义的嵌入模型列表
UD_EMBEDDINGS: List[CustomEmbeddingModelSpec] = []

# 获取用户定义的嵌入模型列表的函数
def get_user_defined_embeddings() -> List[EmbeddingModelSpec]:
    with UD_EMBEDDING_LOCK:
        # 这段代码中使用锁（UD_EMBEDDING_LOCK）是为了确保线程安全。让我解释一下为什么这里需要使用锁：
        # 并发访问保护：
        # 在多线程环境中，可能有多个线程同时尝试读取或修改 UD_EMBEDDINGS 列表。
        # 使用锁可以防止在一个线程读取列表时，另一个线程同时修改列表，从而避免数据不一致或竞态条件。
        # 一致性读取：
        # 锁确保在复制 UD_EMBEDDINGS 列表时，不会有其他线程同时修改它。
        # 这保证了返回的列表是某一时刻的完整快照，而不是部分更新的状态。
        return UD_EMBEDDINGS.copy()

# 注册新的嵌入模型
def register_embedding(model_spec: CustomEmbeddingModelSpec, persist: bool):
    from ...constants import XINFERENCE_MODEL_DIR
    from ..utils import is_valid_model_name, is_valid_model_uri
    from . import BUILTIN_EMBEDDING_MODELS, MODELSCOPE_EMBEDDING_MODELS

    # 验证模型名称
    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"Invalid model name {model_spec.model_name}.")

    # 验证模型URI（如果提供）
    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    with UD_EMBEDDING_LOCK:
        # 检查模型名称是否与现有模型冲突
        for model_name in (
            list(BUILTIN_EMBEDDING_MODELS.keys())
            + list(MODELSCOPE_EMBEDDING_MODELS.keys())
            + [spec.model_name for spec in UD_EMBEDDINGS]
        ):
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {model_spec.model_name}"
                )

        # 将新模型添加到用户定义的嵌入模型列表中
        UD_EMBEDDINGS.append(model_spec)

    # 如果需要持久化，将模型规格保存到文件
    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "embedding", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())

# 注销嵌入模型
def unregister_embedding(model_name: str, raise_error: bool = True):
    with UD_EMBEDDING_LOCK:
        model_spec = None
        # 查找要注销的模型
        for i, f in enumerate(UD_EMBEDDINGS):
            if f.model_name == model_name:
                model_spec = f
                break
        if model_spec:
            # 从列表中移除模型
            UD_EMBEDDINGS.remove(model_spec)

            # 删除持久化的模型文件
            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "embedding", f"{model_spec.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            # 删除模型缓存
            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"Remove the cache of user-defined model {model_spec.model_name}. "
                    f"Cache directory: {cache_dir}"
                )
                if os.path.islink(cache_dir):
                    os.remove(cache_dir)
                else:
                    logger.warning(
                        f"Cache directory is not a soft link, please remove it manually."
                    )
        else:
            # 如果未找到模型，根据raise_error参数决定是否抛出异常
            if raise_error:
                raise ValueError(f"Model {model_name} not found")
            else:
                logger.warning(f"Custom embedding model {model_name} not found")
