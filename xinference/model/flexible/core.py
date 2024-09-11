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
# FlexibleModelSpec 类定义了灵活模型的规格
class FlexibleModelSpec(CacheableModelSpec):
    model_id: Optional[str]  # type: ignore  # 模型ID，可选
    model_description: Optional[str]  # 模型描述，可选
    model_uri: Optional[str]  # 模型URI，可选
    launcher: str  # 启动器名称
    launcher_args: Optional[str]  # 启动器参数，可选

    def parser_args(self):
        """
        解析启动器参数
        
        返回:
            dict: 解析后的启动器参数字典
        """
        return json.loads(self.launcher_args)

# FlexibleModelDescription 类描述了灵活模型的详细信息
class FlexibleModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: FlexibleModelSpec,
        model_path: Optional[str] = None,
    ):
        """
        初始化灵活模型描述
        
        参数:
            address (Optional[str]): 模型地址
            devices (Optional[List[str]]): 可用设备列表
            model_spec (FlexibleModelSpec): 模型规格
            model_path (Optional[str]): 模型路径，默认为None
        """
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        """
        将模型描述转换为字典格式
        
        返回:
            dict: 包含模型描述信息的字典
        """
        return {
            "model_type": "flexible",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "launcher": self._model_spec.launcher,
            "launcher_args": self._model_spec.launcher_args,
        }

    def get_model_version(self) -> str:
        """
        获取模型版本
        
        返回:
            str: 模型版本字符串
        """
        return f"{self._model_spec.model_name}"

    def to_version_info(self):
        """
        获取版本信息
        
        返回:
            dict: 包含版本信息的字典
        """
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
    """
    生成灵活模型描述
    
    参数:
        model_spec (FlexibleModelSpec): 模型规格
    
    返回:
        Dict[str, List[Dict]]: 包含模型描述的字典
    """
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
    """
    获取灵活模型列表的副本
    
    返回:
        List[FlexibleModelSpec]: 灵活模型规格列表的副本
    """
    with FLEXIBLE_MODEL_LOCK:
        return FLEXIBLE_MODELS.copy()

def get_flexible_model_descriptions():
    """
    获取灵活模型描述的深拷贝
    
    返回:
        Dict[str, List[Dict]]: 灵活模型描述的深拷贝
    """
    import copy
    return copy.deepcopy(FLEXIBLE_MODEL_DESCRIPTIONS)
# 灵活模型注册和注销功能

def register_flexible_model(model_spec: FlexibleModelSpec, persist: bool):
    """
    注册一个新的灵活模型。

    此函数用于将新的灵活模型规格添加到系统中，并可选择将其持久化保存。

    参数:
        model_spec (FlexibleModelSpec): 要注册的模型规格。
        persist (bool): 是否将模型规格持久化保存到文件系统。

    异常:
        ValueError: 当模型名称无效、模型URI无效、启动器参数无效或模型名称与现有模型冲突时抛出。

    功能流程:
    1. 导入必要的验证函数。
    2. 验证模型名称的有效性。
    3. 如果提供了模型URI，验证其有效性。
    4. 验证启动器参数的有效性（如果提供）。
    5. 检查模型名称是否与现有模型冲突。
    6. 将新模型规格添加到全局模型列表中。
    7. 如果需要持久化，将模型规格保存到文件系统。
    """
    from ..utils import is_valid_model_name, is_valid_model_uri

    # 验证模型名称
    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"无效的模型名称 {model_spec.model_name}。")

    # 验证模型URI（如果提供）
    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"无效的模型URI {model_uri}。")

    # 验证启动器参数
    if model_spec.launcher_args:
        try:
            model_spec.parser_args()
        except Exception:
            raise ValueError(f"无效的模型启动器参数 {model_spec.launcher_args}。")

    # 检查模型名称是否与现有模型冲突
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
    """
    注销（移除）一个已注册的灵活模型。

    此函数用于从系统中移除指定名称的灵活模型，并清理相关的持久化文件和缓存。

    参数:
        model_name (str): 要注销的模型名称。
        raise_error (bool): 当模型未找到时是否抛出异常，默认为True。

    异常:
        ValueError: 当raise_error为True且指定的模型未找到时抛出。

    功能流程:
    1. 使用线程锁确保操作的线程安全。
    2. 在全局模型列表中查找指定名称的模型。
    3. 如果找到模型：
       a. 从全局列表中移除该模型。
       b. 删除模型的持久化文件（如果存在）。
       c. 尝试删除模型的缓存目录。
    4. 如果未找到模型：
       a. 根据raise_error参数决定是抛出异常还是记录警告日志。
    """
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
"""
FlexibleModel 类定义了一个灵活的模型接口，用于加载和推理各种类型的机器学习模型。
这个类提供了一个通用的结构，可以适应不同的模型实现和配置。

主要特点：
1. 支持模型的初始化、加载和推理。
2. 提供对模型基本属性的访问，如模型ID、路径、设备和配置。
3. 允许子类重写 load 和 infer 方法以实现特定模型的功能。
"""

class FlexibleModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        device: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        初始化 FlexibleModel 实例。

        参数:
        model_uid (str): 模型的唯一标识符，用于区分不同的模型实例。
        model_path (str): 模型文件或目录的路径，指定模型的存储位置。
        device (Optional[str]): 模型运行的设备，如 'cpu' 或 'cuda:0'。默认为 None。
        config (Optional[Dict]): 模型的配置参数字典。默认为 None。

        这个方法设置了模型的基本属性，为后续的加载和推理操作做准备。
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._config = config

    def load(self):
        """
        加载模型方法。

        这是一个占位方法，需要在子类中实现具体的模型加载逻辑。
        可能的实现包括从文件加载模型权重、初始化模型架构等。
        """
        pass

    def infer(self, **kwargs):
        """
        模型推理方法。

        参数:
        **kwargs: 可变关键字参数，用于传递推理所需的输入数据。

        这个方法在基类中抛出 NotImplementedError，
        要求子类必须实现自己的推理逻辑。

        抛出:
        NotImplementedError: 当这个方法在子类中没有被重写时抛出。
        """
        raise NotImplementedError("推理方法尚未实现。")

    @property
    def model_uid(self) -> str:
        """
        获取模型的唯一标识符。

        返回:
        str: 模型的唯一标识符。
        """
        return self._model_uid

    @property
    def model_path(self) -> str:
        """
        获取模型的文件路径。

        返回:
        str: 模型的文件或目录路径。
        """
        return self._model_path

    @property
    def device(self) -> Optional[str]:
        """
        获取模型运行的设备信息。

        返回:
        Optional[str]: 模型运行的设备，如果未指定则为 None。
        """
        return self._device

    @property
    def config(self) -> Optional[Dict]:
        """
        获取模型的配置信息。

        返回:
        Optional[Dict]: 模型的配置参数字典，如果未指定则为 None。
        """
        return self._config

# 这个模块包含两个函数，用于匹配和创建灵活模型实例

def match_flexible_model(model_name: str) -> Optional[FlexibleModelSpec]:
    """
    根据给定的模型名称匹配灵活模型规格。

    参数:
    model_name (str): 要匹配的模型名称。

    返回:
    Optional[FlexibleModelSpec]: 如果找到匹配的模型规格则返回，否则返回 None。

    描述:
    - 遍历所有可用的灵活模型规格。
    - 如果找到与给定名称匹配的模型规格，立即返回该规格。
    - 如果没有找到匹配的规格，函数将隐式返回 None。
    """
    for model_spec in get_flexible_models():
        if model_name == model_spec.model_name:
            return model_spec
    return None  # 显式返回 None，以提高代码清晰度


def create_flexible_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[FlexibleModel, FlexibleModelDescription]:
    """
    创建灵活模型实例及其描述。

    参数:
    subpool_addr (str): 子池地址。
    devices (List[str]): 可用设备列表。
    model_uid (str): 模型的唯一标识符。
    model_name (str): 模型名称。
    model_path (Optional[str]): 模型路径，如果为 None 则使用模型规格中的 URI。
    **kwargs: 额外的关键字参数。

    返回:
    Tuple[FlexibleModel, FlexibleModelDescription]: 包含创建的模型实例和模型描述的元组。

    描述:
    1. 匹配模型规格：调用 match_flexible_model 函数获取对应的模型规格。
    2. 确定模型路径：如果未提供 model_path，则使用模型规格中的 URI。
    3. 获取启动器信息：从模型规格中获取启动器名称和参数。
    4. 更新参数：将启动器参数添加到 kwargs 中。
    5. 创建模型实例：使用获取的启动器创建模型实例。
    6. 创建模型描述：使用提供的信息创建 FlexibleModelDescription 实例。
    7. 返回结果：将创建的模型实例和模型描述作为元组返回。
    """
    # 匹配灵活模型规格
    model_spec = match_flexible_model(model_name)
    
    # 如果未提供模型路径，则使用模型规格中的 URI
    if not model_path:
        model_path = model_spec.model_uri
    
    # 获取启动器名称和解析参数
    launcher_name = model_spec.launcher
    launcher_args = model_spec.parser_args()
    
    # 更新 kwargs，添加解析参数
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
