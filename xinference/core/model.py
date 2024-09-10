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

# 导入必要的模块
import asyncio  # 用于异步编程
import functools  # 提供高阶函数和操作可调用对象的工具
import inspect  # 提供获取对象信息的函数
import json  # 用于JSON数据的编码和解码
import os  # 提供与操作系统交互的功能
import time  # 提供各种时间相关函数
import types  # 定义了一些动态类型创建和名称绑定的工具
import weakref  # 提供弱引用对象

# 导入异步编程相关的类和函数
from asyncio.queues import Queue
from asyncio.tasks import wait_for
from concurrent.futures import Future as ConcurrentFuture

# 导入类型注解相关的模块和类型
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Union,
)

# 导入第三方库
import sse_starlette.sse  # 用于服务器发送事件（SSE）
import xoscar as xo  # XOscar库，可能用于分布式计算

# 导入项目内部常量
from ..constants import XINFERENCE_TRANSFORMERS_ENABLE_BATCHING

# 条件导入，仅在类型检查时使用
if TYPE_CHECKING:
    from .worker import WorkerActor
    from ..model.llm.core import LLM
    from ..model.core import ModelDescription
    import PIL

# 设置日志
import logging
logger = logging.getLogger(__name__)

# 导入项目内部工具函数
from ..device_utils import empty_cache
from .utils import json_dumps, log_async

# 尝试导入PyTorch的CUDA内存溢出错误，如果导入失败则定义一个自定义异常
try:
    from torch.cuda import OutOfMemoryError
except ImportError:
    class _OutOfMemoryError(Exception):
        pass
    OutOfMemoryError = _OutOfMemoryError

# 定义允许批处理的视觉模型列表
XINFERENCE_BATCHING_ALLOWED_VISION_MODELS = ["qwen-vl-chat", "cogvlm2", "glm-4v"]



def request_limit(fn):
    """
    用于ModelActor的装饰器。
    添加到ModelActor方法上，用于控制同时访问该方法的请求数量。
    每当一个新的请求开始处理时，如果没有超过限制，serve_count 就会增加 1。
    当请求处理完成时（无论成功与否），serve_count 会减少 1。
    """

    async def wrapped_func(self, *args, **kwargs):
        # 记录请求信息，包括方法名、当前服务请求数和请求限制
        logger.debug(
            f"请求 {fn.__name__}, 当前服务请求数: {self._serve_count}, 请求限制: {self._request_limits}, 模型: {self.model_uid()}"
        )
        
        # 如果设置了请求限制
        if self._request_limits is not None:
            # 检查是否超过请求限制
            if 1 + self._serve_count <= self._request_limits:
                self._serve_count += 1  # 增加当前服务请求数
            else:
                # 如果超过限制，抛出运行时错误
                raise RuntimeError(
                    f"模型达到速率限制。模型 {self.model_uid()} 的请求限制为 {self._request_limits}"
                )
        try:
            # 执行被装饰的方法
            ret = await fn(self, *args, **kwargs)
        finally:
            # 无论方法是否成功执行，都要减少服务请求数
            if self._request_limits is not None:
                self._serve_count -= 1
            # 记录请求完成后的信息
            logger.debug(
                f"请求 {fn.__name__} 完成后，当前服务请求数: {self._serve_count}, 模型: {self.model_uid()}"
            )
        return ret

    return wrapped_func

def oom_check(fn):
    """
    装饰器函数，用于检查内存溢出错误。
    如果发生OutOfMemoryError，记录异常并强制终止程序。

    :param fn: 被装饰的函数
    :return: 包装后的函数
    """
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        """
        同步函数的包装器
        """
        try:
            return fn(*args, **kwargs)
        except OutOfMemoryError:
            logger.exception("模型执行器内存不足。")
            os._exit(1)  # 强制终止程序

    @functools.wraps(fn)
    async def _async_wrapper(*args, **kwargs):
        """
        异步函数的包装器
        """
        try:
            return await fn(*args, **kwargs)
        except OutOfMemoryError:
            logger.exception("模型执行器内存不足。")
            os._exit(1)  # 强制终止程序

    # 确保被装饰的函数不是异步生成器或同步生成器
    assert not inspect.isasyncgen(fn), "不支持异步生成器"
    assert not inspect.isgenerator(fn), "不支持同步生成器"

    # 根据函数类型返回相应的包装器
    if asyncio.iscoroutinefunction(fn):
        return _async_wrapper
    else:
        return _wrapper


# ModelActor 类，这个类封装了与模型相关的各种操作。以下是该文件的主要功能：
# 1. 模型生命周期管理：
# 初始化模型（__init__方法）
# 加载模型（load方法）
# 销毁模型（__pre_destroy__方法）
# 模型操作：
# 文本生成（generate方法）
# 聊天（chat方法）
# 创建嵌入（create_embedding方法）
# 重新排序（rerank方法）
# 语音转文字（transcriptions方法）
# 翻译（translations方法）
# 文字转语音（speech方法）
# 文本生成图像（text_to_image方法）
# 文本生成视频（text_to_video方法）
# 性能监控和指标记录：
# 记录完成指标（_record_completion_metrics方法）
# 记录一般指标（record_metrics方法）
# 资源管理：
# 请求限制（使用@request_limit装饰器）
# 批处理支持（allow_batching方法）
# 错误处理：
# 内存溢出检查（oom_check装饰器）
# 异步操作支持：
# 大量使用async/await语法
# 使用asyncio进行异步操作
# 与其他组件的交互：
# 与工作节点的交互（_get_worker_ref方法）
# 与调度器的交互（在__post_create__方法中创建调度器）
# 日志记录：
# 使用@log_async装饰器记录异步操作的日志
class ModelActor(xo.StatelessActor):
    @classmethod
    def gen_uid(cls, model: "LLM"):
        # 生成模型的唯一标识符
        return f"{model.__class__}-model-actor"

    async def __pre_destroy__(self):
        # __pre_destroy__ 方法：
        # 这是一个特殊的方法，通常在 Actor 系统（如 xoscar）中使用。
        # 它在 Actor 被销毁之前自动调用，用于执行清理操作。
        # 这个方法允许 Actor 在被完全销毁之前释放资源、关闭连接等。
        # 在销毁模型之前执行的清理操作
        from ..model.embedding.core import EmbeddingModel
        from ..model.llm.sglang.core import SGLANGModel
        from ..model.llm.transformers.core import PytorchModel as LLMPytorchModel
        from ..model.llm.vllm.core import VLLMModel as LLMVLLMModel

        if self.allow_batching():
            try:
                # 销毁调度器
                assert self._scheduler_ref is not None
                await xo.destroy_actor(self._scheduler_ref)
                del self._scheduler_ref
            except Exception as e:
                logger.debug(
                    f"Destroy scheduler actor failed, address: {self.address}, error: {e}"
                )

        # 如果模型有stop方法，调用它
        if hasattr(self._model, "stop") and callable(self._model.stop):
            self._model.stop()

        # 对于特定类型的模型，执行额外的清理操作
        if (
            isinstance(self._model, (LLMPytorchModel, LLMVLLMModel, SGLANGModel))
            and self._model.model_spec.model_format == "pytorch"
        ) or isinstance(self._model, EmbeddingModel):
            try:
                import gc

                import torch  # noqa: F401
            except ImportError:
                error_message = "Failed to import module 'torch'"
                installation_guide = [
                    "Please make sure 'torch' is installed.\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            # 删除模型，执行垃圾回收和清空缓存
            # gc.collect() 强制执行Python的垃圾收集。
            # 在这里单独调用它是为了确保所有不再使用的对象（特别是大型的机器学习模型）被及时回收。
            del self._model
            # 大型对象：机器学习模型通常是内存密集型的，可能不会立即被Python的常规垃圾收集机制回收。
            # 内存释放：强制垃圾收集可以更快地释放内存，这在资源受限的环境中特别重要。
            # 确保清理：某些对象可能有复杂的引用关系，手动触发垃圾收集可以确保它们被正确处理。
            gc.collect()
            empty_cache()

    def __init__(
        self,
        worker_address: str,
        model: "LLM",
        model_description: Optional["ModelDescription"] = None,
        request_limits: Optional[int] = None,
    ):
        # 初始化ModelActor
        super().__init__()
        from ..model.llm.lmdeploy.core import LMDeployModel
        from ..model.llm.sglang.core import SGLANGModel
        from ..model.llm.transformers.core import PytorchModel
        from ..model.llm.vllm.core import VLLMModel

        # 设置基本属性
        self._worker_address = worker_address
        self._model = model
        self._model_description = (
            model_description.to_dict() if model_description else {}
        )
        self._request_limits = request_limits

        # 初始化生成器和锁
        self._generators: Dict[str, Union[Iterator, AsyncGenerator]] = {}
        self._current_generator = lambda: None
        self._lock = (
            None
            if isinstance(
                self._model, (PytorchModel, VLLMModel, SGLANGModel, LMDeployModel)
            )
            else asyncio.locks.Lock()
        )
        
        # 初始化其他属性
        self._worker_ref = None
        self._serve_count = 0
        self._metrics_labels = {
            "type": self._model_description.get("model_type", "unknown"),
            "model": self.model_uid(),
            "node": self._worker_address,
            "format": self._model_description.get("model_format", "unknown"),
            "quantization": self._model_description.get("quantization", "none"),
        }
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._scheduler_ref = None

    async def __post_create__(self):
        # 创建完成后的操作
        self._loop = asyncio.get_running_loop()

        if self.allow_batching():
            # 如果允许批处理，创建调度器
            from .scheduler import SchedulerActor

            self._scheduler_ref = await xo.create_actor(
                SchedulerActor,
                address=self.address,
                uid=SchedulerActor.gen_uid(self.model_uid(), self._model.rep_id),
            )

    async def _record_completion_metrics(
        self, duration, completion_tokens, prompt_tokens
    ):
        # 记录完成指标
        coros = []
        if completion_tokens > 0:
            coros.append(
                self.record_metrics(
                    "output_tokens_total_counter",
                    "add",
                    {
                        "labels": self._metrics_labels,
                        "value": completion_tokens,
                    },
                )
            )
        if prompt_tokens > 0:
            coros.append(
                self.record_metrics(
                    "input_tokens_total_counter",
                    "add",
                    {"labels": self._metrics_labels, "value": prompt_tokens},
                )
            )
        if completion_tokens > 0:
            generate_throughput = completion_tokens / duration
            coros.append(
                self.record_metrics(
                    "generate_throughput",
                    "set",
                    {
                        "labels": self._metrics_labels,
                        "value": generate_throughput,
                    },
                )
            )
        await asyncio.gather(*coros)

    async def _get_worker_ref(self) -> xo.ActorRefType["WorkerActor"]:
        # 获取工作节点的引用
        from .worker import WorkerActor

        if self._worker_ref is None:
            # 如果工作节点引用不存在，则创建一个新的引用
            self._worker_ref = await xo.actor_ref(
                address=self._worker_address,  # 使用预设的工作节点地址
                uid=WorkerActor.uid()  # 使用WorkerActor的唯一标识符
            )
        # 返回工作节点引用，可能是新创建的或已存在的
        return self._worker_ref

    def is_vllm_backend(self) -> bool:
        # 检查是否使用VLLM后端
        # 是VLLMModel模型实例就使用vllm后端
        from ..model.llm.vllm.core import VLLMModel

        return isinstance(self._model, VLLMModel)
    def allow_batching(self) -> bool:
        # 检查是否允许批处理
        from ..model.llm.transformers.core import PytorchModel

        # 获取模型能力列表, generate, chat, chat-vl, tools
        model_ability = self._model_description.get("model_ability", [])

        # 判断是否满足批处理条件：启用了Transformers批处理且模型是PytorchModel实例
        # 优化性能：
        # Transformer模型特别适合批处理，因为它们可以并行处理多个输入序列。批处理可以显著提高Transformer模型的计算效率。
        # 架构特性：
        # Transformer架构的自注意力机制天然支持并行处理多个序列。这使得Transformer模型在批处理时特别高效。
        # 内存利用：
        # 批处理可以更有效地利用GPU内存，特别是对于Transformer这样的大型模型。
        # 框架支持：
        # 许多深度学习框架（如PyTorch）对Transformer模型的批处理提供了优化支持。
        # 一致性：
        # Transformer模型通常有相似的输入和输出结构，使得批处理实现更加统一和简单。
        # 6. 特定优化：
        # 可能已经为Transformer模型实现了特定的批处理优化。
        # 限制范围：
        # 限制批处理只用于Transformer模型可以简化实现，减少潜在的错误。
        condition = XINFERENCE_TRANSFORMERS_ENABLE_BATCHING and isinstance(
            self._model, PytorchModel
        )
        
        # 如果满足批处理条件且模型具有视觉能力
        if condition and "vision" in model_ability:
            # 检查模型名称或模型家族是否在允许批处理的视觉模型列表中
            if (
                self._model.model_family.model_name
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
                or self._model.model_family.model_family
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
            ):
                return True
            else:
                # 如果不在允许列表中，记录警告日志并返回False
                logger.warning(
                    f"Currently for multimodal models, "
                    f"xinference only supports {', '.join(XINFERENCE_BATCHING_ALLOWED_VISION_MODELS)} for batching. "
                    f"Your model {self._model.model_family.model_name} with model family {self._model.model_family.model_family} is disqualified."
                )
                return False
        
        # 返回最终的批处理条件判断结果
        return condition

    async def load(self):
        # 加载模型
        self._model.load()
        if self.allow_batching():
            await self._scheduler_ref.set_model(self._model)
            logger.debug(
                f"Batching enabled for model: {self.model_uid()}, max_num_seqs: {self._model.get_max_num_seqs()}"
            )

    def model_uid(self):
        # 获取模型的唯一标识符
        return (
            self._model.model_uid
            if hasattr(self._model, "model_uid")
            else (
                self._model._model_uid
                if hasattr(self._model, "_model_uid")
                else None  # return None for UT
            )
        )

    def _to_generator(self, output_type: str, gen: types.GeneratorType):
        # 将生成器转换为指定输出类型的生成器
        start_time = time.time()  # 记录开始时间
        time_to_first_token = None  # 初始化首个token生成时间
        final_usage = None  # 初始化最终使用情况
        try:
            for v in gen:
                if time_to_first_token is None:
                    # 计算首个token生成时间（毫秒）
                    time_to_first_token = (time.time() - start_time) * 1000
                if output_type == "json":
                    # 如果输出类型是JSON，记录使用情况并转换为JSON格式
                    final_usage = v.get("usage", None)
                    v = dict(data=json.dumps(v, ensure_ascii=False))
                else:
                    # 如果不是JSON，确保输出类型为binary
                    assert (
                        output_type == "binary"
                    ), f"Unknown output type '{output_type}'"
                # 将数据转换为字节流并yield
                yield sse_starlette.sse.ensure_bytes(v, None)
        except OutOfMemoryError:
            # 捕获内存溢出错误，记录异常并强制退出程序
            logger.exception(
                "Model actor is out of memory, model id: %s", self.model_uid()
            )
            os._exit(1)
        finally:
            # 记录指标
            if self._loop is not None and time_to_first_token is not None:
                # 异步记录首个token生成时间
                coro = self.record_metrics(
                    "time_to_first_token",
                    "set",
                    {"labels": self._metrics_labels, "value": time_to_first_token},
                )
                asyncio.run_coroutine_threadsafe(coro, loop=self._loop)
            if self._loop is not None and final_usage is not None:
                # 异步记录完成指标（总时间、完成的token数和提示token数）
                coro = self._record_completion_metrics(
                    time.time() - start_time,
                    completion_tokens=final_usage["completion_tokens"],
                    prompt_tokens=final_usage["prompt_tokens"],
                )
                asyncio.run_coroutine_threadsafe(coro, loop=self._loop)

    async def _to_async_gen(self, output_type: str, gen: types.AsyncGeneratorType):
        # 将异步生成器转换为指定输出类型的异步生成器
        start_time = time.time()  # 记录开始时间
        time_to_first_token = None  # 初始化首个token生成时间
        final_usage = None  # 初始化最终使用情况
        try:
            async for v in gen:
                if time_to_first_token is None:
                    # 计算首个token生成时间（毫秒）
                    time_to_first_token = (time.time() - start_time) * 1000
                final_usage = v.get("usage", None)  # 更新最终使用情况
                if output_type == "json":
                    # 如果输出类型是JSON，将数据转换为JSON格式
                    v = await asyncio.to_thread(json.dumps, v, ensure_ascii=False)
                    v = dict(data=v)  # noqa: F821
                else:
                    # 如果不是JSON，确保输出类型为binary
                    assert (
                        output_type == "binary"
                    ), f"Unknown output type '{output_type}'"
                # 将数据转换为字节流并yield
                yield await asyncio.to_thread(sse_starlette.sse.ensure_bytes, v, None)
        except OutOfMemoryError:
            # 捕获内存溢出错误，记录异常并强制退出程序
            logger.exception(
                "Model actor is out of memory, model id: %s", self.model_uid()
            )
            os._exit(1)
        finally:
            # 记录指标
            coros = []
            if time_to_first_token is not None:
                # 添加记录首个token生成时间的协程
                coros.append(
                    self.record_metrics(
                        "time_to_first_token",
                        "set",
                        {"labels": self._metrics_labels, "value": time_to_first_token},
                    )
                )
            if final_usage is not None:
                # 添加记录完成指标的协程（总时间、完成的token数和提示token数）
                coros.append(
                    self._record_completion_metrics(
                        time.time() - start_time,
                        completion_tokens=final_usage["completion_tokens"],
                        prompt_tokens=final_usage["prompt_tokens"],
                    )
                )
            # 并发执行所有记录指标的协程
            await asyncio.gather(*coros)

    async def _call_wrapper_json(self, fn: Callable, *args, **kwargs):
        # JSON输出的调用包装器
        return await self._call_wrapper("json", fn, *args, **kwargs)

    async def _call_wrapper_binary(self, fn: Callable, *args, **kwargs):
        # 二进制输出的调用包装器
        return await self._call_wrapper("binary", fn, *args, **kwargs)

    @oom_check
    async def _call_wrapper(self, output_type: str, fn: Callable, *args, **kwargs):
        """
        个 _call_wrapper 函数是一个通用的调用包装器，它的主要作用是统一处理不同类型的函数调用和输出格式。将这个功能抽出为单独的函数有几个重要原因：
        统一处理异步和同步函数：
        函数可以处理协程函数（async）和普通函数。
        对于普通函数，它使用 asyncio.to_thread 来避免阻塞事件循环。
        并发控制：
        通过可选的锁机制（self._lock）来控制并发访问。
        这对于一些不支持并行操作的模型（如 llama-cpp-python）很重要。
        输出格式统一：
        根据指定的 output_type（"json" 或 "binary"）处理返回值。
        将生成器和异步生成器转换为适当的输出格式。
        错误处理：
        使用 @oom_check 装饰器来处理内存溢出错误。
        检查并发生成的限制。
        代码复用：
        这个包装器可以被多个方法使用，减少重复代码。
        灵活性：
        允许在一个地方统一管理所有调用的行为，便于未来的扩展和修改。
        7. 抽象化：
        将调用的复杂性隐藏在这个包装器中，使调用者不需要关心具体的实现细节。
        性能优化：
        通过在线程中执行同步函数，避免阻塞主事件循环。
        9. 状态管理：
        管理 self._current_generator，这对于跟踪当前活动的生成器很重要。
        10. 类型转换：
        根据需要将返回值转换为 JSON 或保持为二进制格式。
        示例使用：
        async def some_method(self, *args, **kwargs):
            return await self._call_wrapper("json", self._model.generate, *args, **kwargs)


        在这个例子中，_call_wrapper 处理了所有的复杂性，
        包括异步调用、锁管理、输出格式化等，使得调用代码保持简洁。
        总之，_call_wrapper 的存在使得代码更加模块化、易于维护，并提供了一个统一的接口来处理各种不同的调用场景。这种设计模式在处理复杂的异步操作和多样的函数调用时特别有用。
        Args:
            output_type (str): _description_
            fn (Callable): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        # 通用调用包装器，用于处理不同类型的函数调用和输出格式
        if self._lock is None:
            # 如果没有锁，直接执行函数
            if inspect.iscoroutinefunction(fn):
                # 如果是协程函数，使用await直接调用
                ret = await fn(*args, **kwargs)
            else:
                # 如果是普通函数，在线程中执行以避免阻塞
                ret = await asyncio.to_thread(fn, *args, **kwargs)
        else:
            # 如果有锁，在锁的上下文中执行函数
            async with self._lock:
                if inspect.iscoroutinefunction(fn):
                    ret = await fn(*args, **kwargs)
                else:
                    ret = await asyncio.to_thread(fn, *args, **kwargs)

        # 检查是否存在锁和当前生成器，如果都存在则不支持并行生成
        if self._lock is not None and self._current_generator():
            raise Exception("Parallel generation is not supported by llama-cpp-python.")

        # 处理不同类型的返回值
        if inspect.isgenerator(ret):
            # 如果返回值是生成器，转换为适当的输出类型
            gen = self._to_generator(output_type, ret)
            self._current_generator = weakref.ref(gen)
            return gen
        if inspect.isasyncgen(ret):
            # 如果返回值是异步生成器，转换为适当的输出类型
            gen = self._to_async_gen(output_type, ret)
            self._current_generator = weakref.ref(gen)
            return gen
        if output_type == "json":
            # 如果输出类型是JSON，将返回值转换为JSON字符串
            # 这个函数用于在单独的线程中执行可能阻塞的 I/O 操作，以避免阻塞事件循环。
            return await asyncio.to_thread(json_dumps, ret)
        else:
            # 如果输出类型是二进制，直接返回
            assert output_type == "binary", f"Unknown output type '{output_type}'"
            return ret
    @log_async(logger=logger)  # 异步日志装饰器
    @request_limit  # 请求限制装饰器
    @xo.generator  # 生成器装饰器
    async def generate(self, prompt: str, *args, **kwargs):
        # 生成文本的方法
        if self.allow_batching():
            # 如果允许批处理，使用批处理请求处理
            return await self.handle_batching_request(
                prompt, "generate", *args, **kwargs
            )
        else:
            # 如果不允许批处理，使用普通的生成方法
            kwargs.pop("raw_params", None)  # 移除原始参数，如果存在的话
            if hasattr(self._model, "generate"):
                # 如果模型有 generate 方法，使用同步生成
                return await self._call_wrapper_json(
                    self._model.generate, prompt, *args, **kwargs
                )
            if hasattr(self._model, "async_generate"):
                # 如果模型有 async_generate 方法，使用异步生成
                return await self._call_wrapper_json(
                    self._model.async_generate, prompt, *args, **kwargs
                )
            # 如果模型既没有 generate 也没有 async_generate 方法，抛出异常
            raise AttributeError(f"模型 {self._model.model_spec} 不支持生成操作。")

    @staticmethod
    async def _queue_consumer(
        queue: Queue, timeout: Optional[float] = None
    ) -> AsyncIterator[Any]:
        # 队列消费者方法，用于异步处理队列中的消息
        from .scheduler import (
            XINFERENCE_STREAMING_ABORT_FLAG,
            XINFERENCE_STREAMING_DONE_FLAG,
            XINFERENCE_STREAMING_ERROR_FLAG,
        )

        while True:
            # TODO: 需要实现超时设置
            # 从队列中异步获取消息，可能会有超时
            res = await wait_for(queue.get(), timeout)
            
            # 检查是否收到结束标志
            if res == XINFERENCE_STREAMING_DONE_FLAG:
                break  # 如果是结束标志，退出循环
            elif res == XINFERENCE_STREAMING_ABORT_FLAG:
                # 如果收到中止标志，抛出运行时错误
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            elif isinstance(res, str) and res.startswith(
                XINFERENCE_STREAMING_ERROR_FLAG
            ):
                # 如果收到错误标志，抛出运行时错误，包含错误信息
                raise RuntimeError(res[len(XINFERENCE_STREAMING_ERROR_FLAG) :])
            else:
                # 如果是正常消息，则yield出去
                yield res

    @staticmethod
    def _get_stream_from_args(ability: str, *args) -> bool:
        # 从参数中获取流式处理标志
        if ability == "chat":
            # 对于聊天能力，流标志在第三个参数中
            assert args[2] is None or isinstance(args[2], dict)
            # 如果第三个参数为None，返回False；否则从字典中获取'stream'的值，默认为False
            return False if args[2] is None else args[2].get("stream", False)
        else:
            # 对于其他能力，流标志在第一个参数中
            assert args[0] is None or isinstance(args[0], dict)
            # 如果第一个参数为None，返回False；否则从字典中获取'stream'的值，默认为False
            return False if args[0] is None else args[0].get("stream", False)

    async def handle_batching_request(self, prompt: str, ability: str, *args, **kwargs):
        # 处理批处理请求的方法
        # 从参数中获取是否为流式请求
        stream = self._get_stream_from_args(ability, *args)
        # 确保调度器引用存在
        assert self._scheduler_ref is not None
        if stream:
            # 处理流式请求
            # 再次确认调度器引用存在（可能是冗余检查）
            assert self._scheduler_ref is not None
            # 创建一个队列用于异步通信
            queue: Queue[Any] = Queue()
            # 创建一个队列消费者
            ret = self._queue_consumer(queue)
            # 向调度器添加请求
            await self._scheduler_ref.add_request(prompt, queue, *args, **kwargs)
            # 将返回结果转换为异步生成器
            gen = self._to_async_gen("json", ret)
            # 保存当前生成器的弱引用
            self._current_generator = weakref.ref(gen)
            return gen
        else:
            # 处理非流式请求
            # 导入非流式中止标志
            from .scheduler import XINFERENCE_NON_STREAMING_ABORT_FLAG

            # 确保事件循环存在
            assert self._loop is not None
            # 创建一个并发Future对象
            future = ConcurrentFuture()
            # 向调度器添加请求            await self._scheduler_ref.add_request(prompt, future, *args, **kwargs)

            # 将并发Future包装为asyncio Future
            fut = asyncio.wrap_future(future, loop=self._loop)
            # 等待结果
            result = await fut
            # 检查是否收到中止标志
            if result == XINFERENCE_NON_STREAMING_ABORT_FLAG:
                # 如果是中止标志，抛出运行时错误
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            # 将结果转换为JSON字符串并返回
            return await asyncio.to_thread(json_dumps, result)

    @log_async(logger=logger)  # 异步日志装饰器
    @request_limit  # 请求限制装饰器
    @xo.generator  # xo生成器装饰器
    async def chat(self, prompt: str, *args, **kwargs):
        # 聊天功能实现
        start_time = time.time()  # 记录开始时间
        response = None
        try:
            if self.allow_batching():
                # 如果允许批处理，调用批处理请求处理方法
                return await self.handle_batching_request(
                    prompt, "chat", *args, **kwargs
                )
            else:
                # 不允许批处理的情况
                kwargs.pop("raw_params", None)  # 移除原始参数
                if hasattr(self._model, "chat"):
                    # 如果模型有chat方法，调用它
                    response = await self._call_wrapper_json(
                        self._model.chat, prompt, *args, **kwargs
                    )
                    return response
                if hasattr(self._model, "async_chat"):
                    # 如果模型有async_chat方法，调用它
                    response = await self._call_wrapper_json(
                        self._model.async_chat, prompt, *args, **kwargs
                    )
                    return response
                # 如果模型既没有chat也没有async_chat方法，抛出异常
                raise AttributeError(f"Model {self._model.model_spec} is not for chat.")
        finally:
            # 处理非流式结果
            record = None
            if isinstance(response, Generator) or isinstance(response, AsyncGenerator):
                # 如果响应是生成器或异步生成器
                record = response
            elif isinstance(response, bytes):
                # 如果响应是字节串，解析为JSON
                record = json.loads(response)
            if record and isinstance(record, dict):
                # 如果记录是字典类型
                usage = record["usage"]
                # 某些后端可能没有有效的使用情况，我们跳过它们
                completion_tokens = usage["completion_tokens"]
                prompt_tokens = usage["prompt_tokens"]
                # 记录完成指标
                await self._record_completion_metrics(
                    time.time() - start_time,
                    completion_tokens,
                    prompt_tokens,
                )

    async def abort_request(self, request_id: str) -> str:
        # 中止请求的方法
        from .scheduler import AbortRequestMessage

        if self.allow_batching():
            # 如果允许批处理
            if self._scheduler_ref is None:
                # 如果调度器引用为空，返回未找到
                return AbortRequestMessage.NOT_FOUND.name
            # 调用调度器的中止请求方法
            return await self._scheduler_ref.abort_request(request_id)
        # 如果不允许批处理，返回无操作
        return AbortRequestMessage.NO_OP.name

    @log_async(logger=logger)  # 异步日志装饰器，用于记录函数调用
    @request_limit  # 请求限制装饰器，可能用于控制并发请求数量
    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        """
        创建文本嵌入的异步方法
        
        参数:
        input: 输入文本，可以是单个字符串或字符串列表
        *args: 可变位置参数
        **kwargs: 可变关键字参数
        
        返回:
        嵌入向量
        
        异常:
        AttributeError: 如果模型不支持创建嵌入
        """
        # 检查模型是否具有create_embedding方法
        if hasattr(self._model, "create_embedding"):
            # 调用模型的create_embedding方法并返回结果
            return await self._call_wrapper_json(
                self._model.create_embedding, input, *args, **kwargs
            )

        # 如果模型不支持创建嵌入，抛出AttributeError异常
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating embedding."
        )

    @log_async(logger=logger)
    @request_limit
    async def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        *args,
        **kwargs,
    ):
        # 重新排序
        if hasattr(self._model, "rerank"):
            return await self._call_wrapper_json(
                self._model.rerank,
                documents,
                query,
                top_n,
                max_chunks_per_doc,
                return_documents,
                return_len,
                *args,
                **kwargs,
            )
        raise AttributeError(f"Model {self._model.model_spec} is not for reranking.")

    @log_async(logger=logger, args_formatter=lambda _, kwargs: kwargs.pop("audio"))
    @request_limit
    async def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        # 转录
        if hasattr(self._model, "transcriptions"):
            return await self._call_wrapper_json(
                self._model.transcriptions,
                audio,
                language,
                prompt,
                response_format,
                temperature,
                timestamp_granularities,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating transcriptions."
        )

    @log_async(logger=logger, args_formatter=lambda _, kwargs: kwargs.pop("audio"))
    @request_limit
    async def translations(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        # 翻译
        if hasattr(self._model, "translations"):
            return await self._call_wrapper_json(
                self._model.translations,
                audio,
                language,
                prompt,
                response_format,
                temperature,
                timestamp_granularities,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating translations."
        )

    @log_async(
        logger=logger,
        args_formatter=lambda _, kwargs: kwargs.pop("prompt_speech", None),
    )
    @request_limit
    @xo.generator
    async def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        # 语音生成
        if hasattr(self._model, "speech"):
            return await self._call_wrapper_binary(
                self._model.speech,
                input,
                voice,
                response_format,
                speed,
                stream,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating speech."
        )

    @log_async(logger=logger)
    @request_limit
    async def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        *args,
        **kwargs,
    ):
        # 文本到图像生成
        if hasattr(self._model, "text_to_image"):
            return await self._call_wrapper_json(
                self._model.text_to_image,
                prompt,
                n,
                size,
                response_format,
                *args,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating image."
        )

    async def image_to_image(
        self,
        image: "PIL.Image",
        prompt: str,
        negative_prompt: str,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        *args,
        **kwargs,
    ):
        # 图像到图像生成
        if hasattr(self._model, "image_to_image"):
            return await self._call_wrapper_json(
                self._model.image_to_image,
                image,
                prompt,
                negative_prompt,
                n,
                size,
                response_format,
                *args,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating image."
        )

    async def inpainting(
        self,
        image: "PIL.Image",
        mask_image: "PIL.Image",
        prompt: str,
        negative_prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        *args,
        **kwargs,
    ):
        # 图像修复
        if hasattr(self._model, "inpainting"):
            return await self._call_wrapper_json(
                self._model.inpainting,
                image,
                mask_image,
                prompt,
                negative_prompt,
                n,
                size,
                response_format,
                *args,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating image."
        )

    @log_async(logger=logger)
    @request_limit
    async def infer(
        self,
        **kwargs,
    ):
        # 灵活推理
        if hasattr(self._model, "infer"):
            return await self._call_wrapper_json(
                self._model.infer,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for flexible infer."
        )

    @log_async(logger=logger)
    @request_limit
    async def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        *args,
        **kwargs,
    ):
        # 文本到视频生成
        if hasattr(self._model, "text_to_video"):
            return await self._call_wrapper_json(
                self._model.text_to_video,
                prompt,
                n,
                *args,
                **kwargs,
            )
        raise AttributeError(
            f"Model {self._model.model_spec} is not for creating video."
        )

    async def record_metrics(self, name, op, kwargs):
        """
        异步记录指标的方法

        参数:
        name: str - 指标名称
        op: str - 操作类型
        kwargs: dict - 额外的关键字参数

        该方法用于异步记录模型的各种指标。
        """
        # 获取工作者引用
        worker_ref = await self._get_worker_ref()
        # 调用工作者的记录指标方法
        await worker_ref.record_metrics(name, op, kwargs)
