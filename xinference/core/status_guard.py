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
from enum import Enum
from logging import getLogger
from typing import Dict, List, Optional

import xoscar as xo

from .._compat import BaseModel

logger = getLogger(__name__)



# 理和跟踪模型实例的状态。它的主要功能包括：
# 1. 状态定义：
# 定义了 LaunchStatus 枚举类，表示模型实例的不同状态（如创建中、更新中、终止中等）。
# 2. 实例信息管理：
# 定义了 InstanceInfo 类，用于存储模型实例的详细信息，如模型名称、版本、能力、副本数等。
# 3. 状态跟踪：
# 实现了 StatusGuardActor 类，用于管理和跟踪多个模型实例的状态。
# 4. 实例信息存储和更新：
# 提供了方法来设置、获取和更新实例信息。
# 5. 实例计数：
# 提供了获取特定模型实例数量的方法。
# 6. 状态过滤：
# 实现了过滤已终止实例的功能，确保只返回活跃的实例信息。
# 7. 查询功能：
# 支持根据模型名称或唯一标识符查询实例信息。
# 8. 并发安全：
# 作为一个 Actor，它可能设计用于在并发环境中安全地管理状态。
# status_guard" 这个名称可能是 "status guardian" 的缩写或变体，意为"状态守护者"或"状态监护者"。选择这个名称可能有以下几个原因

# Enum 的特性：
# 每个枚举成员都有两个主要属性：.name 和 .value。
# .name 返回枚举成员的名称（字符串）。
# .value 返回枚举成员的值。
# 例如，LaunchStatus.TERMINATED.name 会返回字符串 "TERMINATED"。
# 而 LaunchStatus.TERMINATED.value 会返回整数 4。
class LaunchStatus(Enum):
    CREATING = 1    # 模型实例正在创建中
    UPDATING = 2    # 模型实例正在更新中
    TERMINATING = 3 # 模型实例正在终止中
    TERMINATED = 4  # 模型实例已终止
    READY = 5       # 模型实例准备就绪，可以使用
    ERROR = 6       # 模型实例出现错误


class InstanceInfo(BaseModel):
    model_name: str             # 模型名称
    model_uid: str              # 模型唯一标识符
    model_version: Optional[str] # 模型版本（可选）
    model_ability: List[str]    # 模型能力列表
    replica: int                # 副本数量
    status: str                 # 实例状态
    instance_created_ts: int    # 实例创建时间戳

    def update(self, **kwargs):
        """
        更新实例信息
        
        参数:
            **kwargs: 需要更新的字段和对应的值
        
        说明:
            遍历传入的关键字参数，使用setattr更新对应的属性
        """
        for field, value in kwargs.items():
            setattr(self, field, value)


class StatusGuardActor(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        # 存储模型UID到实例信息的映射
        self._model_uid_to_info: Dict[str, InstanceInfo] = {}  # type: ignore

    @classmethod
    def uid(cls) -> str:
        # 返回Actor的唯一标识符
        return "status_guard"

    @staticmethod
    def _drop_terminated_info(instance_infos: List[InstanceInfo]) -> List[InstanceInfo]:
        # 从实例信息列表中移除已终止的实例
        return [
            info
            for info in instance_infos
            if info.status != LaunchStatus.TERMINATED.name
        ]

    def set_instance_info(self, model_uid: str, info: InstanceInfo):
        # 设置指定模型UID的实例信息
        self._model_uid_to_info[model_uid] = info

    def get_instance_info(
        self, model_name: Optional[str] = None, model_uid: Optional[str] = None
    ) -> List[InstanceInfo]:
        """
        获取实例信息。

        参数:
            model_name: 可选，模型名称
            model_uid: 可选，模型唯一标识符

        返回:
            List[InstanceInfo]: 包含实例信息的列表

        说明:
            - 如果提供了model_uid，则返回该特定模型的信息（如果存在）
            - 如果提供了model_name，则返回所有匹配该名称的模型信息
            - 如果两者都未提供，则返回所有模型的信息
            - 返回结果中不包含已终止的实例
        """
        if model_uid is not None:
            # 如果提供了model_uid，返回对应的实例信息（如果存在）
            return (
                self._drop_terminated_info([self._model_uid_to_info[model_uid]])
                if model_uid in self._model_uid_to_info
                else []
            )
        # 获取所有实例信息
        all_infos: List[InstanceInfo] = list(self._model_uid_to_info.values())
        # 如果提供了model_name，过滤出匹配的实例信息
        filtered_infos: List[InstanceInfo] = list(
            filter(lambda info: info.model_name == model_name, all_infos)
        )
        # 返回过滤后的实例信息，并移除已终止的实例
        return (
            self._drop_terminated_info(filtered_infos)
            if model_name is not None
            else self._drop_terminated_info(all_infos)
        )

    def get_instance_count(self, model_name: str) -> int:
        # 获取指定模型名称的实例数量
        return len(self.get_instance_info(model_name=model_name))

    def update_instance_info(self, model_uid: str, info: Dict):
        # 更新指定模型UID的实例信息
        self._model_uid_to_info[model_uid].update(**info)
