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

from collections import defaultdict, deque
from enum import Enum
from typing import Dict, List, TypedDict

import xoscar as xo

# 每个模型最大事件数量
MAX_EVENT_COUNT_PER_MODEL = 100


class EventType(Enum):
    # 事件类型枚举
    INFO = 1
    WARNING = 2
    ERROR = 3


class Event(TypedDict):
    # 事件数据结构
    event_type: EventType  # 事件类型
    event_ts: int  # 事件时间戳
    event_content: str  # 事件内容

# 继承自 xo.StatelessActor，表明这个类可能是为分布式系统设计的
# 帮助开发者和运维人员监控模型的运行状态
class EventCollectorActor(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        # 使用defaultdict存储每个模型的事件队列
        # `defaultdict` 是一个特殊的字典，当访问不存在的键时，它会自动创建一个默认值。
        # 这里的默认值是一个 `deque`，最大长度为 `MAX_EVENT_COUNT_PER_MODEL`。100
        # `lambda: deque(maxlen=MAX_EVENT_COUNT_PER_MODEL)` 是一个匿名函数，每次调用时创建一个新的 `deque`。
        self._model_uid_to_events: Dict[str, deque] = defaultdict(  # type: ignore
            lambda: deque(maxlen=MAX_EVENT_COUNT_PER_MODEL)
        )

    @classmethod
    def uid(cls) -> str:
        # 返回事件收集器的唯一标识符
        return "event_collector"

    def get_model_events(self, model_uid: str) -> List[Dict]:
        """
        获取模型事件

        参数:
            model_uid (str): 模型唯一标识符

        返回:
            List[Dict]: 模型事件列表
        """
        event_queue = self._model_uid_to_events.get(model_uid)
        if event_queue is None:
            return []
        else:
            # 将事件类型转换为字符串名称
            # dict(e, event_type=e["event_type"].name) 的作用：
            # 创建一个新的字典，包含 e 的所有键值对
            # 将 event_type 的值替换为枚举成员的名称（字符串）
            
            # 最终返回的格式：
            # [
            #     {
            #     "event_type": "INFO",  # 或 "WARNING", "ERROR"
            #     "event_ts": 1234567890,
            #     "event_content": "Some event description"
            #     },
            #  ]
            return [dict(e, event_type=e["event_type"].name) for e in iter(event_queue)]

    def report_event(self, model_uid: str, event: Event):
        """
        添加模型事件

        参数:
            model_uid (str): 模型唯一标识符
            event (Event): 事件对象
        """
        # 将事件添加到对应模型的事件队列中
        self._model_uid_to_events[model_uid].append(event)
