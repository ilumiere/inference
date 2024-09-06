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
from logging import getLogger
from typing import Any, Dict, List, Optional

import xoscar as xo

logger = getLogger(__name__)


class CacheTrackerActor(xo.Actor):
    """
    缓存跟踪系统，主要用于管理和跟踪模型的缓存状态。让我详细解释一下它的主要功能：
    1. 缓存状态管理：
    跟踪不同模型的缓存状态。
    记录模型的版本信息和文件位置。
    2. 模型版本记录：
    通过 record_model_version 方法记录模型的版本信息。
    处理模型版本信息的一致性，特别是在分布式环境中。
    3. 缓存状态更新：
    update_cache_status 方法用于更新特定模型的缓存状态。
    支持不同类型的模型（如图像模型）的缓存更新。
    4. 模型版本查询：
    提供方法来获取特定模型的版本信息和版本数量。
    5. 缓存模型列表：
    list_cached_models 方法用于列出所有缓存的模型。
    支持按模型名称和工作节点 IP 筛选。
    6. 模型删除管理：
    list_deletable_models 和 confirm_and_remove_model 方法用于管理可删除的模型。
    支持按版本和工作节点删除模型缓存。
    7. 分布式系统支持：
    设计考虑了分布式环境，支持多个工作节点。
    使用地址映射来跟踪不同节点上的模型文件位置。
    8. 错误处理和日志记录：
    包含错误处理逻辑和日志记录，以提高系统的可靠性和可调试性。
    
    

    Args:
        xo (_type_): _description_
        
        
    """
    def __init__(self):
        super().__init__()
        # 存储模型名称到版本信息的映射
        self._model_name_to_version_info: Dict[str, List[Dict]] = {}  # type: ignore

    @classmethod
    def uid(cls) -> str:
        return "cache_tracker"

    @staticmethod
    def _map_address_to_file_location(
        model_version: Dict[str, List[Dict]], address: str
    ):
        """
        将地址映射到文件位置。

        参数:
            model_version: Dict[str, List[Dict]] - 包含模型版本信息的字典
            address: str - 要映射的地址

        功能:
            遍历model_version字典,为每个模型版本的文件位置添加地址映射。
            如果模型已缓存,则将文件位置设置为{address: 原文件位置}的字典;
            如果未缓存,则设置为None。

        说明:
            这个方法用于在分布式环境中跟踪不同节点上模型文件的位置。
            
            输入:
            {
                "model_name_1": [
                    {
                        "model_file_location": "/path/to/model1_v1",
                        "cache_status": True,
                        # 其他版本信息...
                    },
                    {
                        "model_file_location": "/path/to/model1_v2",
                        "cache_status": False,
                        # 其他版本信息...
                    }
                ],
                "model_name_2": [
                    {
                        "model_file_location": "/path/to/model2_v1",
                        "cache_status": True,
                        # 其他版本信息...
                    }
                ]
            }
            
            
            输出:
            {
                "model_name_1": [
                    {
                        "model_file_location": {"192.168.1.100": "/path/to/model1_v1"},
                        "cache_status": True,
                        # 其他版本信息保持不变...
                    },
                    {
                        "model_file_location": None,
                        "cache_status": False,
                        # 其他版本信息保持不变...
                    }
                ],
                "model_name_2": [
                    {
                        "model_file_location": {"192.168.1.100": "/path/to/model2_v1"},
                        "cache_status": True,
                        # 其他版本信息保持不变...
                    }
                ]
            }  
        """
        # 遍历模型版本字典中的每个模型及其版本信息
        for model_name, model_versions in model_version.items():
            # 对每个版本的信息进行处理
            for info_dict in model_versions:
                # 更新模型文件位置信息
                # 如果模型已缓存，则创建一个新的字典，将地址映射到原始文件位置
                # 如果模型未缓存，则将文件位置设置为None
                info_dict["model_file_location"] = (
                    {address: info_dict["model_file_location"]}
                    if info_dict["cache_status"]
                    else None
                )

    @staticmethod
    def _update_file_location(data: Dict, origin_version_info: Dict):
        """
        更新模型文件位置信息

        参数:
            data: Dict - 新的文件位置数据
            origin_version_info: Dict - 原始版本信息字典

        说明:
            - 如果原始版本信息中的文件位置为None，直接用新数据替换
            - 如果原始版本信息中已有文件位置数据，则更新现有数据
        """
        # 如果原始版本信息中没有文件位置数据
        if origin_version_info["model_file_location"] is None:
            # 直接将新数据赋值给文件位置
            origin_version_info["model_file_location"] = data
        else:
            # 确保原始文件位置信息是字典类型
            assert isinstance(origin_version_info["model_file_location"], dict)
            # 更新原始文件位置信息，添加新的数据
            origin_version_info["model_file_location"].update(data)

    def record_model_version(self, version_info: Dict[str, List[Dict]], address: str):
        # 记录模型版本信息
        # 将地址映射到文件位置
        """
        记录新的模型版本信息
        更新现有模型的缓存状态和文件位置
        确保不同节点（supervisor 和 worker）之间的模型版本信息一致


        version_info = {
            "model_A": [
                {"model_version": "1.0", "cache_status": True, "model_file_location": "/path/to/A_v1"},
                {"model_version": "2.0", "cache_status": False, "model_file_location": None}
            ],
            "model_B": [
                {"model_version": "1.0", "cache_status": True, "model_file_location": "/path/to/B_v1"}
            ]
        }
        address = "192.168.1.100"
        
        Args:
            version_info (Dict[str, List[Dict]]): _description_
            address (str): _description_
        """
        self._map_address_to_file_location(version_info, address)
        
        # 遍历版本信息中的每个模型及其版本
        for model_name, model_versions in version_info.items():
            
            # 不存在,直接添加
            # 如果模型名称不在已记录的版本信息中，直接添加
            if model_name not in self._model_name_to_version_info:
                self._model_name_to_version_info[model_name] = model_versions
            
            # 其他: 更新版本信息
            else:
                # 确保supervisor和worker之间的模型版本信息一致
                assert len(model_versions) == len(
                    self._model_name_to_version_info[model_name]
                ), "Model version info inconsistency between supervisor and worker"
                
                # 遍历每个版本，更新缓存状态和文件位置
                # 实际效果：
                # 假设 model_versions 是 [v1, v2, v3]
                # self._model_name_to_version_info[model_name] 是 [o1, o2, o3]
                # 循环将依次处理 (v1, o1), (v2, o2), (v3, o3)
                for version, origin_version in zip(
                    model_versions, self._model_name_to_version_info[model_name]
                ):
                    # 如果版本已缓存且有文件位置信息
                    if (
                        version["cache_status"]
                        and version["model_file_location"] is not None
                    ):
                        # 更新原始版本的缓存状态
                        origin_version["cache_status"] = True
                        # 更新文件位置信息
                        self._update_file_location(
                            version["model_file_location"], origin_version
                        )

    def update_cache_status(
        self,
        address: str,
        model_name: str,
        model_version: Optional[str],
        model_path: str,
    ):
        """
        更新模型的缓存状态

        参数:
        address (str): 模型所在的地址
        model_name (str): 模型名称
        model_version (Optional[str]): 模型版本，对于图像模型可能为None
        model_path (str): 模型文件路径
        """
        # 检查模型名称是否在已记录的版本信息中
        if model_name not in self._model_name_to_version_info:
            logger.warning(f"目前没有记录 {model_name} 的版本信息。")
        else:
            # 遍历该模型的所有版本信息
            for version_info in self._model_name_to_version_info[model_name]:
                if model_version is None:  # 图像模型的情况
                    # 更新文件位置并设置缓存状态为True
                    self._update_file_location({address: model_path}, version_info)
                    version_info["cache_status"] = True
                else:
                    # 对于非图像模型，检查版本是否匹配
                    if version_info["model_version"] == model_version:
                        # 更新文件位置并设置缓存状态为True
                        self._update_file_location({address: model_path}, version_info)
                        version_info["cache_status"] = True

    def unregister_model_version(self, model_name: str):
        # 注销模型版本
        self._model_name_to_version_info.pop(model_name, None)

    def get_model_versions(self, model_name: str) -> List[Dict]:
        """
        获取指定模型名称的所有版本信息

        参数:
            model_name (str): 要查询的模型名称

        返回:
            List[Dict]: 包含模型所有版本信息的列表，每个版本信息是一个字典

        说明:
            - 如果模型名称不存在于记录中，会记录一个警告并返回空列表
            - 否则返回该模型名称对应的所有版本信息
        """
        # 检查模型名称是否在已记录的版本信息中
        if model_name not in self._model_name_to_version_info:
            # 如果不存在，记录警告信息
            logger.warning(f"未记录模型 {model_name} 的版本信息")
            # 返回空列表
            return []
        else:
            # 如果存在，返回该模型名称对应的所有版本信息
            return self._model_name_to_version_info[model_name]

    def get_model_version_count(self, model_name: str) -> int:
        # 获取模型版本数量
        return len(self.get_model_versions(model_name))

    def list_cached_models(
        self, worker_ip: str, model_name: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        """
        列出在某个worker ip下缓存的模型

        参数:
            worker_ip (str): 工作节点的IP地址
            model_name (Optional[str]): 可选的模型名称，用于筛选特定模型

        返回:
            List[Dict[Any, Any]]: 包含缓存模型信息的字典列表
        """
        cached_models = []
        # 遍历所有模型及其版本信息
        for name, versions in self._model_name_to_version_info.items():
            # 如果指定了模型名称，只返回该模型的缓存信息
            # 否则返回所有缓存的模型
            if model_name and model_name != name:
                continue
            # 遍历当前模型的所有版本信息
            for version_info in versions:
                # 获取缓存状态，默认为False
                cache_status = version_info.get("cache_status", False)
                # 如果模型已缓存
                if cache_status:
                    # 复制版本信息并添加模型名称
                    res = version_info.copy()
                    res["model_name"] = name
                    # 获取模型文件位置信息
                    paths = res.get("model_file_location", {})
                    # 如果指定的worker_ip存在于路径中
                    if worker_ip in paths.keys():
                        # 更新文件位置为指定worker的路径, 只修改worker, 
                        res["model_file_location"] = paths[worker_ip]
                        # 将结果添加到缓存模型列表
                        cached_models.append(res)
        # 返回缓存模型列表
        return cached_models
    def list_deletable_models(self, model_version: str, worker_ip: str) -> str:
        """
        列出某个workerip下 可删除的模型文件位置。

        参数:
            model_version (str): 要查找的模型版本
            worker_ip (str): 工作节点的IP地址

        返回:
            str: 可删除模型的文件位置，如果未找到则返回空字符串
        """
        # 初始化模型文件位置为空字符串
        model_file_location = ""
        # 遍历所有模型及其版本信息
        for model, model_versions in self._model_name_to_version_info.items():
            for version_info in model_versions:
                # 搜索指定的模型版本
                if model_version == version_info.get("model_version", None):
                    # 检查模型是否已缓存
                    if version_info.get("cache_status", False):
                        # 获取模型文件位置信息
                        paths = version_info.get("model_file_location", {})
                        # 只返回指定worker的设备路径
                        if worker_ip in paths.keys():
                            model_file_location = paths[worker_ip]
        # 返回找到的模型文件位置（如果未找到则为空字符串）
        return model_file_location
    
    def confirm_and_remove_model(self, model_version: str, worker_ip: str):
        """
        确认并移除指定版本和工作节点的模型。

        参数:
            model_version (str): 要移除的模型版本
            worker_ip (str): 工作节点的IP地址

        功能:
            1. 获取要移除的模型路径
            2. 在模型版本信息中查找并删除对应的路径
            3. 如果模型在该工作节点上不再有缓存，更新缓存状态
        """
        # 获取要移除的模型路径
        rm_path = self.list_deletable_models(model_version, worker_ip)
        
        # 遍历所有模型及其版本信息
        for model, model_versions in self._model_name_to_version_info.items():
            for version_info in model_versions:
                # 检查模型是否已缓存
                if version_info.get("cache_status", False):
                    paths = version_info.get("model_file_location", {})
                    # 检查是否为指定工作节点上的目标路径
                    if worker_ip in paths.keys() and rm_path == paths[worker_ip]:
                        # 删除该工作节点上的路径
                        del paths[worker_ip]
                        # 如果该模型在所有节点上都没有缓存了，将缓存状态设为False
                        if not paths:
                            version_info["cache_status"] = False
