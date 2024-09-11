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

import asyncio
import functools
import logging
import uuid
from collections import deque
from enum import Enum
from typing import List, Optional, Set, Tuple

import xoscar as xo

logger = logging.getLogger(__name__)

XINFERENCE_STREAMING_DONE_FLAG = "<XINFERENCE_STREAMING_DONE>"
XINFERENCE_STREAMING_ERROR_FLAG = "<XINFERENCE_STREAMING_ERROR>"
XINFERENCE_STREAMING_ABORT_FLAG = "<XINFERENCE_STREAMING_ABORT>"
XINFERENCE_NON_STREAMING_ABORT_FLAG = "<XINFERENCE_NON_STREAMING_ABORT>"


class AbortRequestMessage(Enum):
    NOT_FOUND = 1
    DONE = 2
    NO_OP = 3


class InferenceRequest:
    """
    表示推理请求的类。
    
    这个类封装了一个推理请求的所有相关信息，包括输入、配置、状态和结果。
    """

    def __init__(self, prompt, future_or_queue, is_prefill, *args, **kwargs):
        """
        full prompt tokens
        初始化一个推理请求。

        参数:
        prompt: 原始提示文本
        future_or_queue: 用于获取返回结果的Future或Queue对象
        is_prefill: 是否为预填充阶段
        *args, **kwargs: 额外的参数和关键字参数
        """
        # 原始提示文本
        self._prompt = prompt
        # 包含聊天历史并应用聊天模板的完整提示
        self._full_prompt = None
        # 当前请求是否处于预填充阶段
        self._is_prefill = is_prefill
        # 完整提示的token
        self._prompt_tokens = None
        # 解码阶段生成的所有新token
        self._new_tokens = []
        # 解码阶段使用的KV缓存
        self._kv_cache = None
        # 从上游接口传递的参数
        self._inference_args = args
        # use passed kwargs from upstream interface, currently for getting raw generate config from upstream,
        # which is useful for some special models
        # 从上游接口传递的关键字参数，用于获取原始生成配置
        self._inference_kwargs = kwargs
        # should this request be stopped
        self._stopped = False
        # 完成原因。如果设置了此项，self._stopped为True
        self._finish_reason = None
        # 是否应该中止此请求
        # 注意：当此标志为True时，self._stopped也应为True
        self._aborted = False
        # sanitized generate config
        self._sanitized_generate_config = None
        # For calculate attention mask if needed
        # 结果的块ID。在流式模式下，所有块ID应该相同
        self._stream_chunk_id = str(uuid.uuid4())
        # For calculate attention mask if needed
        self.padding_len = 0
        # Use in stream mode
        self.last_output_length = 0
        # 推理结果，
        # 它是一个列表类型，因为当stream=True时，
        # self.completion包含一个解码轮次中的所有结果
        self.completion = []
        # 上游获取返回结果的方式，
        # 当stream=True时，它是一个asyncio.Queue，
        # 当stream=False时，它是一个asyncio future
        self.future_or_queue = future_or_queue
        # 当此请求出错时记录错误消息。
        # 设置此字段时必须将stopped设为True
        self.error_msg: Optional[str] = None
        # 为兼容性考虑。记录一些特殊情况下的额外参数
        self.extra_kwargs = {}

        # 检查从上游传递的参数的完整性
        self._check_args()

    def _check_args(self):
        """
        检查从上游传递的参数的完整性。
        """
        # 聊天模式
        if len(self._inference_args) == 3:
            # system prompt
            assert self._inference_args[0] is None or isinstance(
                self._inference_args[0], str
            )
            # chat history
            assert self._inference_args[1] is None or isinstance(
                self._inference_args[1], list
            )
            # generate config
            assert self._inference_args[2] is None or isinstance(
                self._inference_args[2], dict
            )
        else:  # 生成模式
            assert len(self._inference_args) == 1
            # 生成配置
            assert self._inference_args[0] is None or isinstance(
                self._inference_args[0], dict
            )

    @property
    def prompt(self):
        """获取原始提示文本。"""
        return self._prompt

    @property
    def system_prompt(self):
        """获取系统提示。"""
        return self._inference_args[0]

    @property
    def chat_history(self):
        """获取聊天历史。"""
        return self._inference_args[1]

    @property
    def full_prompt(self):
        """获取完整提示。"""
        return self._full_prompt

    @full_prompt.setter
    def full_prompt(self, value: str):
        """设置完整提示。"""
        self._full_prompt = value

    @property
    def is_prefill(self):
        """检查是否为预填充阶段。"""
        return self._is_prefill

    @is_prefill.setter
    def is_prefill(self, value: bool):
        """设置是否为预填充阶段。"""
        self._is_prefill = value

    @property
    def prompt_tokens(self):
        """获取提示的token。"""
        return self._prompt_tokens

    @prompt_tokens.setter
    def prompt_tokens(self, value: List[int]):
        """设置提示的token。"""
        self._prompt_tokens = value

    @property
    def kv_cache(self):
        """获取KV缓存。"""
        return self._kv_cache

    @kv_cache.setter
    def kv_cache(self, value):
        """设置KV缓存。"""
        self._kv_cache = value

    @property
    def new_tokens(self):
        """获取新生成的token。"""
        return self._new_tokens

    def append_new_token(self, token: int):
        """添加新生成的token。"""
        self._new_tokens.append(token)

    @property
    def generate_config(self):
        """获取生成配置。"""
        return (
            self._inference_args[2]
            if len(self._inference_args) == 3
            else self._inference_args[0]
        )

    @property
    def sanitized_generate_config(self):
        """获取经过清理的生成配置。"""
        return self._sanitized_generate_config

    @sanitized_generate_config.setter
    def sanitized_generate_config(self, value: dict):
        """设置经过清理的生成配置。"""
        self._sanitized_generate_config = value

    @property
    def inference_kwargs(self):
        """获取推理关键字参数。"""
        return self._inference_kwargs

    @property
    def stopped(self):
        """检查请求是否已停止。"""
        return self._stopped

    @stopped.setter
    def stopped(self, value: bool):
        """设置请求是否已停止。"""
        self._stopped = value

    @property
    def finish_reason(self):
        """获取完成原因。"""
        return self._finish_reason

    @finish_reason.setter
    def finish_reason(self, value: Optional[str]):
        """设置完成原因。"""
        self._finish_reason = value

    @property
    def chunk_id(self):
        """获取块ID。"""
        return self._stream_chunk_id

    @property
    def stream(self) -> bool:
        """检查是否为流式模式。"""
        return (
            False
            if self.generate_config is None
            else self.generate_config.get("stream", False)
        )

    @property
    def stream_interval(self) -> int:
        """获取流式间隔。"""
        return self.sanitized_generate_config.get("stream_interval", 2)

    @property
    def include_usage(self) -> bool:
        """检查是否包含使用情况。"""
        stream_options = self.sanitized_generate_config.get("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )
        return include_usage

    @property
    def aborted(self) -> bool:
        """检查请求是否已中止。"""
        return self._aborted

    @aborted.setter
    def aborted(self, value: bool):
        """设置请求是否已中止。"""
        self._aborted = value

    @property
    def request_id(self) -> Optional[str]:
        """获取请求ID。"""
        return (
            None
            if self.generate_config is None
            else self.generate_config.get("request_id", None)
        )
    @functools.lru_cache
    def get_generate_configs(
        self, eos_token_id: int, builtin_stop_token_ids: Optional[Tuple[int]] = None
    ):
        """
        获取并缓存文本生成的配置参数。

        此方法使用@functools.lru_cache装饰器来缓存结果，提高重复调用的效率。

        参数:
        eos_token_id (int): 表示序列结束的标记ID。
        builtin_stop_token_ids (Optional[Tuple[int]]): 内置的停止标记ID元组，默认为None。

        返回:
        Tuple: 包含以下生成配置参数的元组：
            - max_new_tokens (int): 生成的最大新标记数。
            - stream_interval (int): 流式生成的间隔。
            - include_usage (bool): 是否包含使用情况。
            - stop_str (Optional[str]): 停止生成的字符串。
            - stop_token_ids (Set[int]): 停止标记ID集合。
            - temperature (float): 生成的温度参数。
            - repetition_penalty (float): 重复惩罚参数。
            - top_p (float): Top-p采样参数。
            - top_k (int): Top-k采样参数。

        说明:
        该方法从self.sanitized_generate_config中提取各种生成参数，并进行必要的类型转换和默认值设置。
        它还合并了传入的eos_token_id和builtin_stop_token_ids到stop_token_ids集合中。
        """
        from ..types import max_tokens_field

        # 获取最大新标记数，默认值来自max_tokens_field
        max_new_tokens = int(
            self.sanitized_generate_config.get("max_tokens", max_tokens_field.default)
        )
        # 获取流式间隔，默认为2
        stream_interval = self.sanitized_generate_config.get("stream_interval", 2)
        # 获取是否包含使用情况
        include_usage = self.include_usage
        # 获取停止字符串，默认为None
        stop_str = self.sanitized_generate_config.get("stop", None)
        # 获取停止标记ID列表，如果不存在则使用空列表
        stop_token_ids = (
            self.sanitized_generate_config.get("stop_token_ids", None) or []
        )
        # 将停止标记ID转换为集合，并添加eos_token_id
        stop_token_ids = set(stop_token_ids)
        stop_token_ids.add(eos_token_id)
        # 如果提供了内置停止标记ID，则更新到集合中
        stop_token_ids.update(builtin_stop_token_ids or [])
        # 获取温度参数，默认为1.0
        temperature = float(self.sanitized_generate_config.get("temperature", 1.0))
        # 获取重复惩罚参数，默认为1.0
        repetition_penalty = float(
            self.sanitized_generate_config.get("repetition_penalty", 1.0)
        )
        # 获取top_p参数，默认为1.0
        top_p = float(self.sanitized_generate_config.get("top_p", 1.0))
        # 获取top_k参数，默认为-1（表示禁用）
        top_k = int(self.sanitized_generate_config.get("top_k", -1))  # -1表示禁用

        # 返回所有配置参数的元组
        return (
            max_new_tokens,
            stream_interval,
            include_usage,
            stop_str,
            stop_token_ids,
            temperature,
            repetition_penalty,
            top_p,
            top_k,
        )

def _get_valid_batch_kv_cache(data, skipped_indexes: Set[int]):
    """
    获取有效的批量KV缓存。

    参数:
    data: 原始KV缓存数据
    skipped_indexes: 需要跳过的索引集合

    返回:
    处理后的有效KV缓存

    此函数执行以下操作:
    1. 将传入的数据转换为DynamicCache对象
    2. 计算批量大小并确定有效的批次切片
    3. 对缓存中的每个层进行处理，移除跳过的索引对应的数据
    4. 将处理后的DynamicCache转换回原始格式并返回
    """
    from transformers.cache_utils import DynamicCache

    # 将传入的数据转换为DynamicCache对象
    cache = DynamicCache.from_legacy_cache(data)
    
    # 获取批量大小
    batch_size = cache.key_cache[0].shape[0]
    
    # 创建有效的批次切片列表，排除跳过的索引
    batch_slices = [num for num in range(batch_size) if num not in skipped_indexes]
    
    # 遍历缓存中的每一层
    for idx in range(len(cache)):
        # 更新key缓存，只保留有效的批次数据
        cache.key_cache[idx] = cache.key_cache[idx][batch_slices, ::]
        # 更新value缓存，只保留有效的批次数据
        cache.value_cache[idx] = cache.value_cache[idx][batch_slices, ::]
    
    # 将处理后的DynamicCache转换回原始格式并返回
    return cache.to_legacy_cache()


class SchedulerActor(xo.StatelessActor):
    """
    调度器Actor类，负责管理和执行推理请求。

    该类继承自xo.StatelessActor，用于处理模型推理的调度和执行。
    它维护了等待队列和运行队列，并管理请求的生命周期。
    """

    @classmethod
    def gen_uid(cls, model_uid: str, replica_id: str):
        """
        生成调度器Actor的唯一标识符。

        参数:
        model_uid (str): 模型的唯一标识符
        replica_id (str): 副本的标识符

        返回:
        str: 生成的唯一标识符
        """
        return f"{model_uid}-{replica_id}-scheduler-actor"

    def __init__(self):
        """
        初始化SchedulerActor实例。

        初始化各种队列、模型引用、请求映射和中止请求集合。
        """
        super().__init__()
        # 等待队列，用于存储待处理的推理请求
        self._waiting_queue: deque[InferenceRequest] = deque()  # type: ignore
        # 运行队列，用于存储正在处理的推理请求
        self._running_queue: deque[InferenceRequest] = deque()  # type: ignore
        # 模型实例，用于执行推理任务
        self._model = None
        # 请求ID到请求对象的映射字典
        self._id_to_req = {}
        # 存储需要中止的请求ID集合
        self._abort_req_ids: Set[str] = set()  # type: ignore
        # 隔离环境实例，用于运行推理任务
        self._isolation = None

    async def __post_create__(self):
        """
        Actor创建后的后处理方法。

        创建并启动隔离环境，并在新的事件循环中运行run方法。
        """
        # 从isolation模块导入Isolation类
        from ..isolation import Isolation

        # 创建一个新的Isolation实例
        # 使用新的事件循环，设置为线程模式和守护进程
        self._isolation = Isolation(
            asyncio.new_event_loop(), threaded=True, daemon=True
        )
        
        # 启动隔离环境
        self._isolation.start()
        
        # 在隔离环境的事件循环中异步运行self.run()方法
        # 使用run_coroutine_threadsafe确保在不同的线程中安全地运行协程
        asyncio.run_coroutine_threadsafe(self.run(), loop=self._isolation.loop)

    async def __pre_destroy__(self):
        """
        Actor销毁前的预处理方法。

        停止并删除隔离环境，处理可能出现的异常。
        """
        try:
            # 确保隔离环境存在
            assert self._isolation is not None
            # 停止隔离环境
            self._isolation.stop()
            # 删除隔离环境引用
            del self._isolation
        except Exception as e:
            # 如果销毁过程中出现异常，记录调试日志
            logger.debug(
                f"Destroy scheduler actor failed, address: {self.address}, error: {e}"
            )

    def set_model(self, model):
        """
        设置调度器使用的模型。

        参数:
        model: 要使用的模型实例
        """
        self._model = model

    def get_max_num_seqs(self):
        """
        获取模型支持的最大序列数。

        返回:
        int: 模型支持的最大序列数
        """
        # 确保模型已经被设置
        assert self._model is not None, "模型尚未设置，无法获取最大序列数"
        
        # 调用模型的方法获取最大序列数
        max_seqs = self._model.get_max_num_seqs()
        
        # 返回获取到的最大序列数
        return max_seqs
    def _check_request_aborted(self, req: InferenceRequest):
        """
        检查请求是否被中止。

        参数:
        req (InferenceRequest): 要检查的推理请求
        """
        # 检查请求是否有ID且该ID是否在需要中止的请求ID列表中
        if req.request_id and req.request_id in self._abort_req_ids:
            # 如果请求需要中止，设置请求的中止标志为True
            req.aborted = True
            # 同时设置请求的停止标志为True，表示请求已经结束
            req.stopped = True

    def _handle_request(self) -> Optional[List[InferenceRequest]]:
        """
        处理请求，从等待队列和运行队列中选择要执行的请求。

        返回:
        Optional[List[InferenceRequest]]: 要执行的请求列表，如果没有可执行的请求则返回None
        """
        # 如果模型未设置，直接返回None
        if self._model is None:
            return None
        
        # 获取模型支持的最大序列数
        max_num_seqs = self.get_max_num_seqs()
        
        # 使用先来先服务（FCFS）策略处理运行中的请求
        running_list: List[InferenceRequest] = []
        while len(self._running_queue) > 0:
            # 如果运行列表已达到最大序列数，停止添加
            if len(running_list) == max_num_seqs:
                break
            # 从运行队列中取出请求
            req = self._running_queue.popleft()
            # 检查请求是否被中止
            self._check_request_aborted(req)
            # 将请求添加到运行列表
            running_list.append(req)

        # 处理等待中的请求
        waiting_list: List[InferenceRequest] = []
        if len(running_list) < max_num_seqs:
            while len(self._waiting_queue) > 0:
                # 从等待队列中取出请求
                req = self._waiting_queue.popleft()
                # 检查请求是否被中止
                self._check_request_aborted(req)
                # 将请求添加到等待列表
                waiting_list.append(req)
                # 如果运行列表和等待列表的总数达到最大序列数，停止添加
                if len(running_list) + len(waiting_list) == max_num_seqs:
                    break
        # must waiting_list in front
        # 返回等待列表和运行列表的组合，确保等待列表在前
        return waiting_list + running_list

    @staticmethod
    def _empty_cache():
        """
        清空模型缓存。
        
        此方法用于释放GPU内存,通过清空模型的缓存来实现。
        在处理大量请求或切换不同模型时调用此方法可以有效管理内存使用。
        
        注意:
        - 此方法是静态方法,可以直接通过类调用,无需实例化对象。
        - 清空缓存可能会影响模型的推理速度,因为需要重新加载一些数据。
        - 在内存紧张的情况下,建议定期调用此方法。
        """
        from ..model.llm.transformers.utils import empty_cache

        empty_cache()

    async def step(self):
        """
        执行一步调度操作。

        此方法是调度器的核心，负责处理请求、执行批量推理，并处理完成的请求。
        它是一个异步方法，允许在处理大量请求时不阻塞其他操作。

        主要功能：
        1. 处理待处理的请求
        2. 执行批量推理
        3. 处理流式和非流式输出
        4. 管理已完成的请求
        5. 处理中止的请求
        6. 更新KV缓存
        7. 清理内存

        无参数
        无返回值
        """
        # 处理待处理的请求，获取可以进行推理的请求列表
        req_list = self._handle_request()
        if not req_list:
            # 如果没有待处理的请求，直接返回
            return
        
        # 使用模型对请求列表进行批量推理
        self._model.batch_inference(req_list)

        # 用于记录已停止的批次索引
        stopped_batch_indexes = set()

        # 遍历所有请求，处理推理结果
        for idx, r in enumerate(req_list):
            if r.stream:
                # 对于流式请求，逐个发送完成的部分
                for completion in r.completion:
                    await r.future_or_queue.put(completion)
                # 清空已发送的完成部分
                r.completion = []

            if not r.stopped:
                # 如果请求未停止，将其重新加入运行队列
                self._running_queue.append(r)
            else:
                # 处理已停止的请求
                if r.new_tokens:
                    # 记录已停止的批次索引，用于后续更新KV缓存
                    stopped_batch_indexes.add(idx)
                # 清除KV缓存以便垃圾回收
                r.kv_cache = None
                rid = r.request_id
                # 清理数据结构
                if rid is not None:
                    self._id_to_req.pop(rid, None)
                    self._abort_req_ids.discard(rid)

                if r.aborted:  
                    # 处理被中止的请求
                    if r.stream:
                        # 对于流式请求，发送中止标志
                        await r.future_or_queue.put(XINFERENCE_STREAMING_ABORT_FLAG)
                    else:
                        # 对于非流式请求，设置中止结果
                        r.future_or_queue.set_result(XINFERENCE_NON_STREAMING_ABORT_FLAG)
                else:
                    if r.error_msg is None:  
                        # 正常停止的情况
                        if not r.stream:
                            # 非流式请求，设置完成结果
                            r.future_or_queue.set_result(r.completion[0])
                        else:
                            # 流式请求，发送完成标志
                            await r.future_or_queue.put(XINFERENCE_STREAMING_DONE_FLAG)
                    # Abnormal stop, currently indicates that the parameter check does not pass,
                    # and does not participate in the inference
                    else:
                        # 异常停止，通常是参数检查未通过
                        if not r.stream:
                            # 非流式请求，抛出异常
                            r.future_or_queue.set_exception(ValueError(r.error_msg))
                        else:
                            # 流式请求，发送错误标志和消息
                            await r.future_or_queue.put(XINFERENCE_STREAMING_ERROR_FLAG + r.error_msg)

        # Some requests have been completed. Batch size needs to be reduced for kv cache.
        if stopped_batch_indexes and len(self._running_queue) > 0:
            kv_cache = self._running_queue[0].kv_cache
            # 获取更新后的KV缓存
            reduced_kv_cache = _get_valid_batch_kv_cache(kv_cache, stopped_batch_indexes)
            # 更新所有运行中请求的KV缓存
            for r in self._running_queue:
                r.kv_cache = reduced_kv_cache

        # 清空模型缓存，释放内存
        self._empty_cache()
    async def add_request(self, prompt: str, future_or_queue, *args, **kwargs):
        """
        添加新的推理请求到等待队列。

        此方法用于创建新的推理请求并将其添加到调度器的等待队列中。它还处理请求ID的唯一性检查。

        参数:
        prompt (str): 推理的提示文本，作为模型的输入。
        future_or_queue: 用于获取结果的Future或Queue对象，用于异步返回推理结果。
        *args, **kwargs: 额外的参数，用于传递给InferenceRequest构造函数。

        流程:
        1. 创建新的InferenceRequest对象。
        2. 检查请求ID的唯一性（如果存在）。
        3. 将请求添加到等待队列和ID映射字典中。

        异常:
        - 如果提供的请求ID已存在，则抛出KeyError。

        注意: 此方法是异步的，但主要操作是同步执行的。异步声明允许在更大的异步上下文中使用。
        """
        # 创建新的推理请求实例
        req = InferenceRequest(prompt, future_or_queue, True, *args, **kwargs)
        rid = req.request_id
        if rid is not None:
            # 如果请求ID已存在，抛出异常
            if rid in self._id_to_req:
                raise KeyError(f"Request id: {rid} has already existed!")
            # 将请求ID和请求对象添加到映射字典中
            self._id_to_req[rid] = req
        # 将请求添加到等待队列
        self._waiting_queue.append(req)

    async def abort_request(self, req_id: str) -> str:
        """
        中止一个已提交的请求。

        此方法用于尝试中止一个正在进行或等待中的推理请求。如果请求已完成或未找到，此方法不会执行任何操作。

        参数:
        req_id (str): 要中止的请求的唯一标识符。

        返回:
        str: 表示中止操作结果的消息。可能的返回值包括：
             - AbortRequestMessage.NOT_FOUND.name: 请求未找到
             - AbortRequestMessage.DONE.name: 请求已标记为中止

        流程:
        1. 检查请求ID是否存在于已知请求中。
        2. 如果不存在，记录日志并返回NOT_FOUND消息。
        3. 如果存在，将请求ID添加到待中止集合中，记录日志，并返回DONE消息。

        注意: 
        - 此方法不会立即停止正在进行的推理，而是标记请求为待中止状态。
        - 实际的中止操作将在调度器的下一个处理周期中执行。
        """
        if req_id not in self._id_to_req:
            # 如果请求ID不存在，记录日志并返回未找到的消息
            logger.info(f"请求ID: {req_id} 未找到。对xinference无操作。")
            return AbortRequestMessage.NOT_FOUND.name
        else:
            # 将请求ID添加到待中止集合中
            self._abort_req_ids.add(req_id)
            logger.info(f"Request id: {req_id} found to be aborted.")
            return AbortRequestMessage.DONE.name

    async def run(self):
        """
        运行调度器的主循环。

        此方法是调度器的核心，负责持续执行调度任务。它在一个无限循环中运行，
        定期调用step方法来处理待处理的请求和任务。

        主要功能:
        1. 持续执行调度任务
        2. 处理可能发生的异常
        3. 记录错误信息

        异常处理:
        - 捕获并记录所有可能发生的异常，确保调度器不会因单个错误而完全停止

        注意:
        - 此方法是异步的，使用了asyncio库来实现非阻塞操作
        - 循环中有一个短暂的休眠时间，以防止过度消耗CPU资源
        """
        try:
            while True:
                # 等待10毫秒，避免过度消耗CPU资源
                # 这个短暂的暂停允许其他异步任务有机会执行
                await asyncio.sleep(0.01)
                
                # 调用step方法执行实际的调度任务
                # step方法预期是异步的，负责处理单个调度周期的所有逻辑
                await self.step()
        except Exception as e:
            # 捕获并记录任何异常
            # 使用logger.exception可以同时记录异常信息和堆栈跟踪
            logger.exception(
                f"Scheduler actor uid: {self.uid}, address: {self.address} run with error: {e}"
            )
            # 注意：这里异常后循环会终止，可能需要考虑重启机制或更健壮的错误处理
