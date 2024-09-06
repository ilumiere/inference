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

import asyncio
import functools
import inspect
import json
import os
import time
import types
import weakref
from asyncio.queues import Queue
from asyncio.tasks import wait_for
from concurrent.futures import Future as ConcurrentFuture
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

import sse_starlette.sse
import xoscar as xo

from ..constants import XINFERENCE_TRANSFORMERS_ENABLE_BATCHING

if TYPE_CHECKING:
    from .worker import WorkerActor
    from ..model.llm.core import LLM
    from ..model.core import ModelDescription
    import PIL

import logging

logger = logging.getLogger(__name__)

from ..device_utils import empty_cache
from .utils import json_dumps, log_async

try:
    from torch.cuda import OutOfMemoryError
except ImportError:

    class _OutOfMemoryError(Exception):
        pass

    OutOfMemoryError = _OutOfMemoryError


XINFERENCE_BATCHING_ALLOWED_VISION_MODELS = ["qwen-vl-chat", "cogvlm2", "glm-4v"]




def request_limit(fn):
    """
    Used by ModelActor.
    As a decorator, added to a ModelActor method to control
    how many requests are accessing that method at the same time.
    """

    async def wrapped_func(self, *args, **kwargs):
        logger.debug(
            f"Request {fn.__name__}, current serve request count: {self._serve_count}, request limit: {self._request_limits} for the model {self.model_uid()}"
        )
        if self._request_limits is not None:
            if 1 + self._serve_count <= self._request_limits:
                self._serve_count += 1
            else:
                raise RuntimeError(
                    f"Rate limit reached for the model. Request limit {self._request_limits} for the model: {self.model_uid()}"
                )
        try:
            ret = await fn(self, *args, **kwargs)
        finally:
            if self._request_limits is not None:
                self._serve_count -= 1
            logger.debug(
                f"After request {fn.__name__}, current serve request count: {self._serve_count} for the model {self.model_uid()}"
            )
        return ret

    return wrapped_func


def oom_check(fn):
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OutOfMemoryError:
            logger.exception("Model actor is out of memory.")
            os._exit(1)

    @functools.wraps(fn)
    async def _async_wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except OutOfMemoryError:
            logger.exception("Model actor is out of memory.")
            os._exit(1)

    assert not inspect.isasyncgen(fn)
    assert not inspect.isgenerator(fn)

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
            del self._model
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
            self._worker_ref = await xo.actor_ref(
                address=self._worker_address, uid=WorkerActor.uid()
            )
        return self._worker_ref

    def is_vllm_backend(self) -> bool:
        # 检查是否使用VLLM后端
        from ..model.llm.vllm.core import VLLMModel

        return isinstance(self._model, VLLMModel)

    def allow_batching(self) -> bool:
        # 检查是否允许批处理
        from ..model.llm.transformers.core import PytorchModel

        model_ability = self._model_description.get("model_ability", [])

        condition = XINFERENCE_TRANSFORMERS_ENABLE_BATCHING and isinstance(
            self._model, PytorchModel
        )
        if condition and "vision" in model_ability:
            if (
                self._model.model_family.model_name
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
                or self._model.model_family.model_family
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
            ):
                return True
            else:
                logger.warning(
                    f"Currently for multimodal models, "
                    f"xinference only supports {', '.join(XINFERENCE_BATCHING_ALLOWED_VISION_MODELS)} for batching. "
                    f"Your model {self._model.model_family.model_name} with model family {self._model.model_family.model_family} is disqualified."
                )
                return False
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
        start_time = time.time()
        time_to_first_token = None
        final_usage = None
        try:
            for v in gen:
                if time_to_first_token is None:
                    time_to_first_token = (time.time() - start_time) * 1000
                if output_type == "json":
                    final_usage = v.get("usage", None)
                    v = dict(data=json.dumps(v, ensure_ascii=False))
                else:
                    assert (
                        output_type == "binary"
                    ), f"Unknown output type '{output_type}'"
                yield sse_starlette.sse.ensure_bytes(v, None)
        except OutOfMemoryError:
            logger.exception(
                "Model actor is out of memory, model id: %s", self.model_uid()
            )
            os._exit(1)
        finally:
            # 记录指标
            if self._loop is not None and time_to_first_token is not None:
                coro = self.record_metrics(
                    "time_to_first_token",
                    "set",
                    {"labels": self._metrics_labels, "value": time_to_first_token},
                )
                asyncio.run_coroutine_threadsafe(coro, loop=self._loop)
            if self._loop is not None and final_usage is not None:
                coro = self._record_completion_metrics(
                    time.time() - start_time,
                    completion_tokens=final_usage["completion_tokens"],
                    prompt_tokens=final_usage["prompt_tokens"],
                )
                asyncio.run_coroutine_threadsafe(coro, loop=self._loop)

    async def _to_async_gen(self, output_type: str, gen: types.AsyncGeneratorType):
        # 将异步生成器转换为指定输出类型的异步生成器
        start_time = time.time()
        time_to_first_token = None
        final_usage = None
        try:
            async for v in gen:
                if time_to_first_token is None:
                    time_to_first_token = (time.time() - start_time) * 1000
                final_usage = v.get("usage", None)
                if output_type == "json":
                    v = await asyncio.to_thread(json.dumps, v, ensure_ascii=False)
                    v = dict(data=v)  # noqa: F821
                else:
                    assert (
                        output_type == "binary"
                    ), f"Unknown output type '{output_type}'"
                yield await asyncio.to_thread(sse_starlette.sse.ensure_bytes, v, None)
        except OutOfMemoryError:
            logger.exception(
                "Model actor is out of memory, model id: %s", self.model_uid()
            )
            os._exit(1)
        finally:
            # 记录指标
            coros = []
            if time_to_first_token is not None:
                coros.append(
                    self.record_metrics(
                        "time_to_first_token",
                        "set",
                        {"labels": self._metrics_labels, "value": time_to_first_token},
                    )
                )
            if final_usage is not None:
                coros.append(
                    self._record_completion_metrics(
                        time.time() - start_time,
                        completion_tokens=final_usage["completion_tokens"],
                        prompt_tokens=final_usage["prompt_tokens"],
                    )
                )
            await asyncio.gather(*coros)

    async def _call_wrapper_json(self, fn: Callable, *args, **kwargs):
        # JSON输出的调用包装器
        return await self._call_wrapper("json", fn, *args, **kwargs)

    async def _call_wrapper_binary(self, fn: Callable, *args, **kwargs):
        # 二进制输出的调用包装器
        return await self._call_wrapper("binary", fn, *args, **kwargs)

    @oom_check
    async def _call_wrapper(self, output_type: str, fn: Callable, *args, **kwargs):
        # 通用调用包装器
        if self._lock is None:
            if inspect.iscoroutinefunction(fn):
                ret = await fn(*args, **kwargs)
            else:
                ret = await asyncio.to_thread(fn, *args, **kwargs)
        else:
            async with self._lock:
                if inspect.iscoroutinefunction(fn):
                    ret = await fn(*args, **kwargs)
                else:
                    ret = await asyncio.to_thread(fn, *args, **kwargs)

        if self._lock is not None and self._current_generator():
            raise Exception("Parallel generation is not supported by llama-cpp-python.")

        if inspect.isgenerator(ret):
            gen = self._to_generator(output_type, ret)
            self._current_generator = weakref.ref(gen)
            return gen
        if inspect.isasyncgen(ret):
            gen = self._to_async_gen(output_type, ret)
            self._current_generator = weakref.ref(gen)
            return gen
        if output_type == "json":
            return await asyncio.to_thread(json_dumps, ret)
        else:
            assert output_type == "binary", f"Unknown output type '{output_type}'"
            return ret

    @log_async(logger=logger)
    @request_limit
    @xo.generator
    async def generate(self, prompt: str, *args, **kwargs):
        # 生成文本
        if self.allow_batching():
            return await self.handle_batching_request(
                prompt, "generate", *args, **kwargs
            )
        else:
            kwargs.pop("raw_params", None)
            if hasattr(self._model, "generate"):
                return await self._call_wrapper_json(
                    self._model.generate, prompt, *args, **kwargs
                )
            if hasattr(self._model, "async_generate"):
                return await self._call_wrapper_json(
                    self._model.async_generate, prompt, *args, **kwargs
                )
            raise AttributeError(f"Model {self._model.model_spec} is not for generate.")

    @staticmethod
    async def _queue_consumer(
        queue: Queue, timeout: Optional[float] = None
    ) -> AsyncIterator[Any]:
        # 队列消费者
        from .scheduler import (
            XINFERENCE_STREAMING_ABORT_FLAG,
            XINFERENCE_STREAMING_DONE_FLAG,
            XINFERENCE_STREAMING_ERROR_FLAG,
        )

        while True:
            # TODO: timeout setting
            res = await wait_for(queue.get(), timeout)
            if res == XINFERENCE_STREAMING_DONE_FLAG:
                break
            elif res == XINFERENCE_STREAMING_ABORT_FLAG:
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            elif isinstance(res, str) and res.startswith(
                XINFERENCE_STREAMING_ERROR_FLAG
            ):
                raise RuntimeError(res[len(XINFERENCE_STREAMING_ERROR_FLAG) :])
            else:
                yield res

    @staticmethod
    def _get_stream_from_args(ability: str, *args) -> bool:
        # 从参数中获取流式处理标志
        if ability == "chat":
            assert args[2] is None or isinstance(args[2], dict)
            return False if args[2] is None else args[2].get("stream", False)
        else:
            assert args[0] is None or isinstance(args[0], dict)
            return False if args[0] is None else args[0].get("stream", False)

    async def handle_batching_request(self, prompt: str, ability: str, *args, **kwargs):
        # 处理批处理请求
        stream = self._get_stream_from_args(ability, *args)
        assert self._scheduler_ref is not None
        if stream:
            # 处理流式请求
            assert self._scheduler_ref is not None
            queue: Queue[Any] = Queue()
            ret = self._queue_consumer(queue)
            await self._scheduler_ref.add_request(prompt, queue, *args, **kwargs)
            gen = self._to_async_gen("json", ret)
            self._current_generator = weakref.ref(gen)
            return gen
        else:
            # 处理非流式请求
            from .scheduler import XINFERENCE_NON_STREAMING_ABORT_FLAG

            assert self._loop is not None
            future = ConcurrentFuture()
            await self._scheduler_ref.add_request(prompt, future, *args, **kwargs)
            fut = asyncio.wrap_future(future, loop=self._loop)
            result = await fut
            if result == XINFERENCE_NON_STREAMING_ABORT_FLAG:
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            return await asyncio.to_thread(json_dumps, result)

    @log_async(logger=logger)
    @request_limit
    @xo.generator
    async def chat(self, prompt: str, *args, **kwargs):
        # 聊天功能
        start_time = time.time()
        response = None
        try:
            if self.allow_batching():
                return await self.handle_batching_request(
                    prompt, "chat", *args, **kwargs
                )
            else:
                kwargs.pop("raw_params", None)
                if hasattr(self._model, "chat"):
                    response = await self._call_wrapper_json(
                        self._model.chat, prompt, *args, **kwargs
                    )
                    return response
                if hasattr(self._model, "async_chat"):
                    response = await self._call_wrapper_json(
                        self._model.async_chat, prompt, *args, **kwargs
                    )
                    return response
                raise AttributeError(f"Model {self._model.model_spec} is not for chat.")
        finally:
            # For the non stream result.
            record = None
            if isinstance(response, Generator) or isinstance(response, AsyncGenerator):
                record = response
            elif isinstance(response, bytes):
                record = json.loads(response)
            if record and isinstance(record, dict):
                usage = record["usage"]
                # 某些后端可能没有有效的使用情况，我们跳过它们
                completion_tokens = usage["completion_tokens"]
                prompt_tokens = usage["prompt_tokens"]
                await self._record_completion_metrics(
                    time.time() - start_time,
                    completion_tokens,
                    prompt_tokens,
                )

    async def abort_request(self, request_id: str) -> str:
        # 中止请求
        from .scheduler import AbortRequestMessage

        if self.allow_batching():
            if self._scheduler_ref is None:
                return AbortRequestMessage.NOT_FOUND.name
            return await self._scheduler_ref.abort_request(request_id)
        return AbortRequestMessage.NO_OP.name

    @log_async(logger=logger)
    @request_limit
    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        # 创建嵌入
        if hasattr(self._model, "create_embedding"):
            return await self._call_wrapper_json(
                self._model.create_embedding, input, *args, **kwargs
            )

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
        # 记录指标
        worker_ref = await self._get_worker_ref()
        await worker_ref.record_metrics(name, op, kwargs)
