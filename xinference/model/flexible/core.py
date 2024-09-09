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

import json
import logging
import os
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Tuple

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from ..core import CacheableModelSpec, ModelDescription
from .utils import get_launcher

# 设置日志记录器
logger = logging.getLogger(__name__)

# 创建灵活模型的全局锁
FLEXIBLE_MODEL_LOCK = Lock()


class FlexibleModelSpec(CacheableModelSpec):
    model_id: Optional[str]  # type: ignore
    model_description: Optional[str]  # 模型描述
    model_uri: Optional[str]  # 模型URI
    launcher: str  # 启动器
    launcher_args: Optional[str]  # 启动器参数

    def parser_args(self):
        # 解析启动器参数
        return json.loads(self.launcher_args)


class FlexibleModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: FlexibleModelSpec,
        model_path: Optional[str] = None,
    ):
        # 初始化灵活模型描述
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        # 将模型描述转换为字典格式
        return {
            "model_type": "flexible",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "launcher": self._model_spec.launcher,
            "launcher_args": self._model_spec.launcher_args,
        }

    def get_model_version(self) -> str:
        # 获取模型版本
        return f"{self._model_spec.model_name}"

    def to_version_info(self):
        # 获取版本信息
        return {
            "model_version": self.get_model_version(),
            "cache_status": True,
            "model_file_location": self._model_spec.model_uri,
            "launcher": self._model_spec.launcher,
            "launcher_args": self._model_spec.launcher_args,
        }


def generate_flexible_model_description(
    model_spec: FlexibleModelSpec,
) -> Dict[str, List[Dict]]:
    # 生成灵活模型描述
    res = defaultdict(list)
    res[model_spec.model_name].append(
        FlexibleModelDescription(None, None, model_spec).to_version_info()
    )
    return res


# 存储灵活模型规格的列表
FLEXIBLE_MODELS: List[FlexibleModelSpec] = []
# 存储灵活模型描述的字典
FLEXIBLE_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)


def get_flexible_models():
    # 获取灵活模型列表的副本
    with FLEXIBLE_MODEL_LOCK:
        return FLEXIBLE_MODELS.copy()


def get_flexible_model_descriptions():
    # 获取灵活模型描述的深拷贝
    import copy

    return copy.deepcopy(FLEXIBLE_MODEL_DESCRIPTIONS)


def register_flexible_model(model_spec: FlexibleModelSpec, persist: bool):
    # 注册灵活模型
    from ..utils import is_valid_model_name, is_valid_model_uri

    # 验证模型名称
    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"无效的模型名称 {model_spec.model_name}。")

    # 验证模型URI
    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"无效的模型URI {model_uri}。")

    # 验证启动器参数
    if model_spec.launcher_args:
        try:
            model_spec.parser_args()
        except Exception:
            raise ValueError(f"无效的模型启动器参数 {model_spec.launcher_args}。")

    # 检查模型名称是否冲突
    with FLEXIBLE_MODEL_LOCK:
        for model_name in [spec.model_name for spec in FLEXIBLE_MODELS]:
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"模型名称与现有模型冲突 {model_spec.model_name}"
                )
        FLEXIBLE_MODELS.append(model_spec)

    # 如果需要持久化，将模型规格保存到文件
    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "flexible", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())


def unregister_flexible_model(model_name: str, raise_error: bool = True):
    # 使用锁确保线程安全
    with FLEXIBLE_MODEL_LOCK:
        model_spec = None
        # 遍历灵活模型列表，查找指定名称的模型
        for i, f in enumerate(FLEXIBLE_MODELS):
            if f.model_name == model_name:
                model_spec = f
                break
        
        if model_spec:
            # 从灵活模型列表中移除找到的模型
            FLEXIBLE_MODELS.remove(model_spec)

            # 构造模型持久化文件路径
            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "flexible", f"{model_spec.model_name}.json"
            )
            # 如果持久化文件存在，则删除它
            if os.path.exists(persist_path):
                os.remove(persist_path)

            # 构造模型缓存目录路径
            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                # 记录警告日志，提示正在删除用户定义模型的缓存
                logger.warning(
                    f"Remove the cache of user-defined model {model_spec.model_name}. "
                    f"Cache directory: {cache_dir}"
                )
                # 如果缓存目录是软链接，直接删除
                if os.path.islink(cache_dir):
                    os.remove(cache_dir)
                else:
                    # 如果不是软链接，提示用户手动删除
                    logger.warning(
                        f"Cache directory is not a soft link, please remove it manually."
                    )
        else:
            # 如果未找到指定名称的模型
            if raise_error:
                # 如果设置了raise_error，则抛出ValueError异常
                raise ValueError(f"Model {model_name} not found")
            else:
                # 否则只记录警告日志
                logger.warning(f"Model {model_name} not found")


class FlexibleModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        device: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        # 初始化灵活模型
        self._model_uid = model_uid  # 模型唯一标识符
        self._model_path = model_path  # 模型路径
        self._device = device  # 设备信息
        self._config = config  # 模型配置

    def load(self):
        """
        加载模型。
        """

    def infer(self, **kwargs):
        """
        调用模型进行推理。
        """
        raise NotImplementedError("推理方法尚未实现。")

    @property
    def model_uid(self):
        # 获取模型唯一标识符
        return self._model_uid

    @property
    def model_path(self):
        # 获取模型路径
        return self._model_path

    @property
    def device(self):
        # 获取设备信息
        return self._device

    @property
    def config(self):
        # 获取模型配置
        return self._config


def match_flexible_model(model_name):
    # 匹配灵活模型
    for model_spec in get_flexible_models():
        if model_name == model_spec.model_name:
            return model_spec  # 返回匹配的模型规格


def create_flexible_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[FlexibleModel, FlexibleModelDescription]:
    # 匹配灵活模型规格
    model_spec = match_flexible_model(model_name)
    
    # 如果未提供模型路径，则使用模型规格中的URI
    if not model_path:
        model_path = model_spec.model_uri
    
    # 获取启动器名称和解析参数
    launcher_name = model_spec.launcher
    launcher_args = model_spec.parser_args()
    
    # 更新kwargs，添加解析参数
    kwargs.update(launcher_args)

    # 使用启动器创建模型实例
    model = get_launcher(launcher_name)(
        model_uid=model_uid, model_spec=model_spec, **kwargs
    )

    # 创建模型描述
    model_description = FlexibleModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    
    # 返回模型实例和模型描述
    return model, model_description
