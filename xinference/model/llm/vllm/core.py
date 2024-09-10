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
import asyncio
import json
import logging
import multiprocessing
import os
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Iterable,
    List,
    Optional,
    TypedDict,
    Union,
)

# 导入自定义类型
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    LoRA,
    ToolCallFunction,
    ToolCalls,
)
# 导入LLM相关类和工具
from .. import LLM, LLMFamilyV1, LLMSpecV1
from ..llm_family import CustomLLMFamilyV1
from ..utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin

# 设置日志记录器
logger = logging.getLogger(__name__)

# 类型检查时导入RequestOutput
if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

# VLLM模型配置类
class VLLMModelConfig(TypedDict, total=False):
    tokenizer_mode: Optional[str]
    trust_remote_code: bool
    tensor_parallel_size: int
    block_size: int
    swap_space: int  # GiB
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    quantization: Optional[str]
    max_model_len: Optional[int]

# VLLM生成配置类
class VLLMGenerateConfig(TypedDict, total=False):
    lora_name: Optional[str]
    n: int
    best_of: Optional[int]
    presence_penalty: float
    frequency_penalty: float
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop_token_ids: Optional[List[int]]
    stop: Optional[Union[str, List[str]]]
    stream: bool  # 非采样参数，不应传递给引擎
    stream_options: Optional[Union[dict, None]]

# 尝试导入vllm模块，并设置安装标志
try:
    import vllm  # noqa: F401
    VLLM_INSTALLED = True
except ImportError:
    VLLM_INSTALLED = False

# VLLM支持的视觉模型列表
VLLM_SUPPORTED_VISION_MODEL_LIST: List[str] = [
    "internvl2",
]

# VLLM支持的基础模型列表
VLLM_SUPPORTED_MODELS = [
    "llama-2",
    "llama-3",
    "mistral-v0.1",
    "codestral-v0.1",
    "Yi",
    "Yi-1.5",
    "code-llama",
    "code-llama-python",
    "deepseek",
    "deepseek-coder",
]

# VLLM支持的聊天模型列表
VLLM_SUPPORTED_CHAT_MODELS = [
    "llama-2-chat",
    "llama-3-instruct",
    "baichuan-2-chat",
    "internlm2-chat",
    "internlm2.5-chat",
    "internlm2.5-chat-1m",
    "qwen-chat",
    "Yi-chat",
    "Yi-1.5-chat",
    "Yi-1.5-chat-16k",
    "code-llama-instruct",
    "mistral-instruct-v0.1",
    "mistral-instruct-v0.2",
    "mistral-instruct-v0.3",
    "mixtral-instruct-v0.1",
    "mixtral-8x22B-instruct-v0.1",
    "chatglm3",
    "chatglm3-32k",
    "chatglm3-128k",
    "glm4-chat",
    "glm4-chat-1m",
    "codegeex4",
    "deepseek-chat",
    "deepseek-coder-instruct",
]

# 根据VLLM版本添加支持的模型
if VLLM_INSTALLED and vllm.__version__ >= "0.3.0":
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-chat")
    VLLM_SUPPORTED_MODELS.append("codeqwen1.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("codeqwen1.5-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-instruct")

if VLLM_INSTALLED and vllm.__version__ >= "0.3.2":
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-it")

if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat-rag")

if VLLM_INSTALLED and vllm.__version__ >= "0.4.0":
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-moe-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-moe-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("c4ai-command-r-v01")

if VLLM_INSTALLED and vllm.__version__ >= "0.5.3":
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-2-it")
    VLLM_SUPPORTED_CHAT_MODELS.append("mistral-nemo-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("mistral-large-instruct")

if VLLM_INSTALLED and vllm.__version__ > "0.5.3":
    VLLM_SUPPORTED_MODELS.append("llama-3.1")
    VLLM_SUPPORTED_CHAT_MODELS.append("llama-3.1-instruct")


class VLLMModel(LLM):
    """
    VLLMModel类，用于处理VLLM模型的加载、配置和生成。
    继承自LLM基类。
    """

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[VLLMModelConfig],
        peft_model: Optional[List[LoRA]] = None,
    ):
        """
        初始化VLLMModel实例。

        :param model_uid: 模型的唯一标识符
        :param model_family: 模型家族
        :param model_spec: 模型规格
        :param quantization: 量化方法
        :param model_path: 模型路径
        :param model_config: VLLM模型配置
        :param peft_model: PEFT模型列表
        """
        try:
            # 尝试导入LoRARequest类
            from vllm.lora.request import LoRARequest
        except ImportError:
            # 如果导入失败，准备错误信息和安装指南
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            # 抛出ImportError异常，包含错误信息和安装指南
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        
        # 调用父类的初始化方法
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        
        # 初始化模型配置
        self._model_config = model_config
        # 初始化引擎为None
        self._engine = None
        # 存储PEFT模型
        # PEFT（Parameter-Efficient Fine-Tuning）：
        # 这是一种在大型预训练模型上进行微调的技术，它只更新模型的一小部分参数，而不是全部参数。
        # 这种方法可以显著减少计算资源的需求，同时保持良好的性能。
        self.lora_modules = peft_model
        # 初始化LoRA请求列表
        self.lora_requests: List[LoRARequest] = []

    def load(self):
        """
        加载VLLM模型。
        """
        try:
            import vllm
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.lora.request import LoRARequest
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        if vllm.__version__ >= "0.3.1":
            # from vllm v0.3.1, it uses cupy as NCCL backend
            # in which cupy will fork a process
            # only for xoscar >= 0.3.0, new process is allowed in subpool
            # besides, xinference set start method as forkserver for unix
            # we need to set it to fork to make cupy NCCL work
            multiprocessing.set_start_method("fork", force=True)
        
        # 清理并补充模型配置
        self._model_config = self._sanitize_model_config(self._model_config)

        if self.lora_modules is None:
            self.lora_requests = []
        else:
            # 创建LoRA请求列表
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.lora_name,
                    lora_int_id=i,
                    lora_local_path=lora.local_path,
                )
                for i, lora in enumerate(self.lora_modules, start=1)
            ]

        # 检查是否启用LoRA
        enable_lora = len(self.lora_requests) > 0
        # 设置LoRA的最大数量
        max_loras = len(self.lora_requests)

        # 记录加载模型时的配置信息
        logger.info(
            f"Loading {self.model_uid} with following model config: {self._model_config}"
            f"Enable lora: {enable_lora}. Lora count: {max_loras}."
        )

        # 创建AsyncEngineArgs实例
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            enable_lora=enable_lora,
            max_loras=max_loras,
            **self._model_config,
        )
        # 从引擎参数创建AsyncLLMEngine实例
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 创建健康检查任务
        self._check_health_task = None
        if hasattr(self._engine, "check_health"):
            # vLLM introduced `check_health` since v0.4.1
            self._check_health_task = asyncio.create_task(self._check_healthy())

    def stop(self):
        """
        停止VLLM引擎。
        """
        # though the vLLM engine will shutdown when deleted,
        # but some issue e.g. GH#1682 reported
        # when deleting, the engine exists still
        logger.info("Stopping vLLM engine")
        if self._check_health_task:
            self._check_health_task.cancel()
        if model_executor := getattr(self._engine.engine, "model_executor", None):
            model_executor.shutdown()
        self._engine = None

    async def _check_healthy(self, interval: int = 30):
        """
        定期检查VLLM引擎的健康状态。

        :param interval: 检查间隔时间（秒）
        """
        from vllm.engine.async_llm_engine import AsyncEngineDeadError

        logger.debug("Begin to check health of vLLM")

        while self._engine is not None:
            try:
                await self._engine.check_health()
            except (AsyncEngineDeadError, RuntimeError):
                logger.info("Detecting vLLM is not health, prepare to quit the process")
                try:
                    self.stop()
                except:
                    # ignore error when stop
                    pass
                # Just kill the process and let xinference auto-recover the model
                os._exit(1)
            else:
                await asyncio.sleep(interval)
    def _sanitize_model_config(
        self, model_config: Optional[VLLMModelConfig]
    ) -> VLLMModelConfig:
        """
        清理并补充模型配置。

        :param model_config: 原始模型配置
        :return: 清理后的模型配置
        """
        # 如果没有提供模型配置，创建一个新的VLLMModelConfig实例
        if model_config is None:
            model_config = VLLMModelConfig()

        # 获取可用的CUDA设备数量
        cuda_count = self._get_cuda_count()

        # 设置默认的模型配置参数
        model_config.setdefault("tokenizer_mode", "auto")  # 设置分词器模式为自动
        model_config.setdefault("trust_remote_code", True)  # 信任远程代码
        model_config.setdefault("tensor_parallel_size", cuda_count)  # 设置张量并行大小为CUDA设备数量
        model_config.setdefault("block_size", 16)  # 设置块大小
        model_config.setdefault("swap_space", 4)  # 设置交换空间大小
        model_config.setdefault("gpu_memory_utilization", 0.90)  # 设置GPU内存利用率
        model_config.setdefault("max_num_seqs", 256)  # 设置最大序列数
        model_config.setdefault("quantization", None)  # 设置量化方法为None
        model_config.setdefault("max_model_len", 4096)  # 设置最大模型长度

        return model_config  # 返回清理后的模型配置

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[Dict] = None,
    ) -> VLLMGenerateConfig:
        """
        清理并补充生成配置。

        :param generate_config: 原始生成配置

        
        安全性：防止使用未定义或不安全的配置值。
        默认值：为未指定的参数提供合理的默认值。
        标准化：确保所有后续处理都使用统一格式的配置。
        使用示例：
        original_config = {
            "temperature": 0.7,
            "max_tokens": 100
        }

        sanitized_config = VLLMModel._sanitize_generate_config(original_config)

        # sanitized_config 现在包含所有必要的参数，
        # 包括原始配置中的值和其他参数的默认值

        总之，_sanitize_generate_config 方法是一个重要的预处理步骤
        它确保了生成过程使用的配置是完整、一致和安全的。
        这种方法有助于提高代码的健壮性和可靠性，特别是在处理用户输入或外部配置
        
        :return: 清理后的生成配置
        """
        if not generate_config:
            generate_config = {}

        sanitized = VLLMGenerateConfig()
        sanitized.setdefault("lora_name", generate_config.get("lora_name", None))
        sanitized.setdefault("n", generate_config.get("n", 1))
        sanitized.setdefault("best_of", generate_config.get("best_of", None))
        sanitized.setdefault(
            "presence_penalty", generate_config.get("presence_penalty", 0.0)
        )
        sanitized.setdefault(
            "frequency_penalty", generate_config.get("frequency_penalty", 0.0)
        )
        sanitized.setdefault("temperature", generate_config.get("temperature", 1.0))
        sanitized.setdefault("top_p", generate_config.get("top_p", 1.0))
        sanitized.setdefault("top_k", generate_config.get("top_k", -1))
        sanitized.setdefault("max_tokens", generate_config.get("max_tokens", 1024))
        sanitized.setdefault("stop", generate_config.get("stop", None))
        sanitized.setdefault(
            "stop_token_ids", generate_config.get("stop_token_ids", None)
        )
        sanitized.setdefault("stream", generate_config.get("stream", False))
        sanitized.setdefault(
            "stream_options", generate_config.get("stream_options", None)
        )

        return sanitized

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        """
        判断给定的模型家族、规格和量化方法是否匹配VLLM模型。

        :param llm_family: 模型家族
        :param llm_spec: 模型规格
        :param quantization: 量化方法
        :return: 是否匹配
        """
        if not cls._has_cuda_device():
            return False
        if not cls._is_linux():
            return False
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8"]:
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False
            else:
                if "4" not in quantization:
                    return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in VLLM_SUPPORTED_MODELS:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_MODELS:
                return False
        if "generate" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    @staticmethod
    def _convert_request_output_to_completion_chunk(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> CompletionChunk:
        """
        将请求输出转换为完成块。

        :param request_id: 请求ID
        :param model: 模型名称
        :param request_output: 请求输出
        :return: 完成块
        """
        choices: List[CompletionChoice] = []
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=None,  # TODO: support logprobs.
                    finish_reason=output.finish_reason,
                )
            )
        return CompletionChunk(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
        )

    @staticmethod
    def _convert_request_output_to_completion(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> Completion:
        """
        将请求输出转换为完成对象。

        :param request_id: 请求ID
        :param model: 模型名称
        :param request_output: 请求输出
        :return: 完成对象
        """
        choices = []
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=None,  # TODO: support logprobs.
                    finish_reason=output.finish_reason,
                )
            )

        prompt_tokens = len(request_output.prompt_token_ids)
        completion_tokens = sum(
            len(output.token_ids) for output in request_output.outputs
        )
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return Completion(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )

    async def async_generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        generate_config: Optional[Dict] = None,
        tools: object = False,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        """
        异步生成文本。

        :param prompt: 提示文本或字典
        :param generate_config: 生成配置
        :param tools: 是否使用工具
        :return: 完成对象或异步生成器
        """
        try:
            from vllm.sampling_params import SamplingParams
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        sanitized_generate_config = self._sanitize_generate_config(generate_config)
        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        lora_model = sanitized_generate_config.pop("lora_name")

        lora_request = None
        if lora_model is not None:
            for lora in self.lora_requests:
                if lora_model == lora.lora_name:
                    lora_request = lora
                    break

        stream = sanitized_generate_config.pop("stream")
        stream_options = sanitized_generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )
        sampling_params = SamplingParams(**sanitized_generate_config)
        request_id = str(uuid.uuid1())

        assert self._engine is not None
        results_generator = self._engine.generate(
            prompt, sampling_params, request_id, lora_request=lora_request
        )

        async def stream_results() -> AsyncGenerator[CompletionChunk, None]:
            previous_texts = [""] * sanitized_generate_config["n"]
            tools_token_filter = ChatModelMixin._tools_token_filter(self.model_family)
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            async for _request_output in results_generator:
                chunk = self._convert_request_output_to_completion_chunk(
                    request_id=request_id,
                    model=self.model_uid,
                    request_output=_request_output,
                )

                for i, choice in enumerate(chunk["choices"]):
                    delta = choice["text"][len(previous_texts[i]) :]
                    previous_texts[i] = choice["text"]
                    choice["text"] = delta

                if tools:
                    # only handle the first choice
                    choice = chunk["choices"][0]
                    if choice["finish_reason"] is not None:
                        # use previous text for evaluation temporarily
                        choice_delta = choice["text"]
                        choice["text"] = previous_texts[0]
                        _content, func, args = ChatModelMixin._eval_tool_arguments(
                            self.model_family, chunk, tools
                        )
                        choice["text"] = tools_token_filter(
                            tokens=previous_texts[0], delta=choice_delta
                        )
                        if func is not None:
                            choice["text"] = None
                            choice["finish_reason"] = "tool_calls"
                            choice["tool_calls"] = [
                                ToolCalls(
                                    id=str(uuid.uuid4()),
                                    type="function",
                                    function=ToolCallFunction(
                                        name=func,
                                        arguments=json.dumps(args, ensure_ascii=False),
                                    ),
                                )
                            ]
                    else:
                        # use a filter function to skip Qwen's react thought process
                        choice["text"] = tools_token_filter(
                            tokens=previous_texts[0], delta=choice["text"]
                        )
                        if not choice["text"]:
                            continue
                prompt_tokens = len(_request_output.prompt_token_ids)
                completion_tokens = sum(
                    len(output.token_ids) for output in _request_output.outputs
                )
                total_tokens = prompt_tokens + completion_tokens
                chunk["usage"] = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                yield chunk
            if include_usage:
                chunk = CompletionChunk(
                    id=request_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=self.model_uid,
                    choices=[],
                )
                chunk["usage"] = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                yield chunk

        if stream:
            return stream_results()
        else:
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            assert final_output is not None
            return self._convert_request_output_to_completion(
                request_id, model=self.model_uid, request_output=final_output
            )


class VLLMChatModel(VLLMModel, ChatModelMixin):
    """
    VLLMChatModel类，用于处理VLLM聊天模型。
    继承自VLLMModel和ChatModelMixin。
    """

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        """
        判断给定的模型家族、规格和量化方法是否匹配VLLMChatModel。

        :param llm_family: LLM模型家族
        :param llm_spec: LLM模型规格
        :param quantization: 量化方法
        :return: 如果匹配返回True，否则返回False
        """
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8"]:
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False
            else:
                if "4" not in quantization:
                    return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in VLLM_SUPPORTED_CHAT_MODELS:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_CHAT_MODELS:
                return False
        if "chat" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        """
        清理并准备聊天配置。

        :param generate_config: 生成配置字典
        :return: 处理后的生成配置字典
        """
        if not generate_config:
            generate_config = {}
        if self.model_family.prompt_style:
            if (
                not generate_config.get("stop")
            ) and self.model_family.prompt_style.stop:
                generate_config["stop"] = self.model_family.prompt_style.stop.copy()
            if self.model_family.prompt_style.stop_token_ids:
                generate_config.setdefault(
                    "stop_token_ids",
                    self.model_family.prompt_style.stop_token_ids.copy(),
                )
        return generate_config

    async def async_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        异步执行聊天生成。

        :param prompt: 用户输入的提示
        :param system_prompt: 系统提示
        :param chat_history: 聊天历史记录
        :param generate_config: 生成配置
        :return: 聊天完成结果或异步生成器
        """
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        tools = generate_config.pop("tools", []) if generate_config else None
        full_prompt = self.get_prompt(prompt, chat_history, prompt_style, tools=tools)

        generate_config = self._sanitize_chat_config(generate_config)
        # TODO(codingl2k1): qwen hacky to set stop for function call.
        model_family = self.model_family.model_family or self.model_family.model_name
        if tools and model_family in QWEN_TOOL_CALL_FAMILY:
            stop = generate_config.get("stop")
            if isinstance(stop, str):
                generate_config["stop"] = [stop, "Observation:"]
            elif isinstance(stop, Iterable):
                assert not isinstance(stop, str)
                generate_config["stop"] = list(stop) + ["Observation:"]
            else:
                generate_config["stop"] = "Observation:"

        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(full_prompt, generate_config, tools)
            assert isinstance(agen, AsyncGenerator)
            return self._async_to_chat_completion_chunks(agen)
        else:
            c = await self.async_generate(full_prompt, generate_config)
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._tool_calls_completion(
                    self.model_family, self.model_uid, c, tools
                )
            return self._to_chat_completion(c)


class VLLMVisionModel(VLLMModel, ChatModelMixin):
    """
    VLLMVisionModel类，用于处理支持视觉功能的VLLM模型。
    继承自VLLMModel和ChatModelMixin。
    """

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        """
        判断给定的模型家族、规格和量化方法是否匹配VLLMVisionModel。

        :param llm_family: LLM模型家族
        :param llm_spec: LLM模型规格
        :param quantization: 量化方法
        :return: 如果匹配返回True，否则返回False
        """
        if llm_spec.model_format != "pytorch":
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in VLLM_SUPPORTED_VISION_MODEL_LIST:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_VISION_MODEL_LIST:
                return False
        if "vision" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        """
        清理并准备聊天配置。

        :param generate_config: 生成配置字典
        :return: 处理后的生成配置字典
        """
        if not generate_config:
            generate_config = {}
        if self.model_family.prompt_style:
            if self.model_family.prompt_style.stop_token_ids:
                generate_config.setdefault(
                    "stop_token_ids",
                    self.model_family.prompt_style.stop_token_ids.copy(),
                )
        return generate_config

    async def async_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        异步执行聊天生成。

        :param prompt: 用户输入的提示
        :param system_prompt: 系统提示（可选）
        :param chat_history: 聊天历史记录（可选）
        :param generate_config: 生成配置（可选）
        :return: 聊天完成结果或异步生成器
        """
        # 仅支持单张图片，等待vllm支持多图片输入
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        chat_history = chat_history or []
        prompt, images = self.get_prompt(prompt, chat_history, prompt_style)

        if len(images) == 0:
            inputs = {
                "prompt": prompt,
            }
        else:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images[-1]},  # type: ignore
            }
        generate_config = self._sanitize_chat_config(generate_config)

        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(inputs, generate_config)
            assert isinstance(agen, AsyncGenerator)
            return self._async_to_chat_completion_chunks(agen)
        else:
            c = await self.async_generate(inputs, generate_config)
            assert not isinstance(c, AsyncGenerator)
            return self._to_chat_completion(c)
