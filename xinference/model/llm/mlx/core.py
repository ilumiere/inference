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
import platform
import sys
import time
import uuid
from typing import Dict, Iterable, Iterator, List, Optional, TypedDict, Union

from ....fields import max_tokens_field
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    LoRA,
)
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)


class MLXModelConfig(TypedDict, total=False):
    revision: Optional[str]  # 模型的修订版本
    max_gpu_memory: str  # 最大GPU内存使用量
    trust_remote_code: bool  # 是否信任远程代码


class MLXGenerateConfig(TypedDict, total=False):
    max_tokens: int  # 生成的最大token数
    temperature: float  # 生成的温度参数
    repetition_penalty: Optional[float]  # 重复惩罚系数
    repetition_context_size: Optional[float]  # 重复惩罚的上下文大小
    top_p: float  # 用于nucleus采样的概率阈值
    logit_bias: Optional[Dict[int, float]]  # token的logit偏置
    stop: Optional[Union[str, List[str]]]  # 停止生成的字符串或字符串列表
    stop_token_ids: Optional[Union[int, List[int]]]  # 停止生成的token ID或ID列表
    stream: bool  # 是否使用流式输出
    stream_options: Optional[Union[dict, None]]  # 流式输出的选项


class MLXModel(LLM):
    """
    MLXModel类，用于处理MLX模型的加载、配置和生成。
    继承自LLM基类。
    """

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[MLXModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        """
        初始化MLXModel实例。

        :param model_uid: 模型的唯一标识符
        :param model_family: 模型家族
        :param model_spec: 模型规格
        :param quantization: 量化方法
        :param model_path: 模型路径
        :param model_config: MLX模型配置
        :param peft_model: PEFT模型列表
        """
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._use_fast_tokenizer = True
        self._model_config: MLXModelConfig = self._sanitize_model_config(model_config)
        if peft_model is not None:
            raise ValueError("MLX engine has not supported lora yet")

    def _sanitize_model_config(
        self, model_config: Optional[MLXModelConfig]
    ) -> MLXModelConfig:
        """
        清理并补充模型配置。

        :param model_config: MLX模型配置
        :return: 清理后的模型配置
        """
        if model_config is None:
            model_config = MLXModelConfig()
        model_config.setdefault("revision", self.model_spec.model_revision)
        model_config.setdefault("trust_remote_code", True)
        return model_config

    def _sanitize_generate_config(
        self,
        generate_config: Optional[MLXGenerateConfig],
    ) -> MLXGenerateConfig:
        """
        清理并补充生成配置。

        :param generate_config: 生成配置
        :return: 清理后的生成配置
        """
        if generate_config is None:
            generate_config = MLXGenerateConfig()

        generate_config.setdefault("max_tokens", max_tokens_field.default)
        # default config is adapted from
        # https://github.com/ml-explore/mlx-examples/blob/f212b770d8b5143e23102eda20400ae43340f844/llms/mlx_lm/utils.py#L129
        generate_config.setdefault("temperature", 0.0)
        generate_config.setdefault("repetition_penalty", None)
        generate_config.setdefault("repetition_context_size", 20)
        generate_config.setdefault("top_p", 1.0)
        generate_config.setdefault("logit_bias", None)
        return generate_config

    def _load_model(self, **kwargs):
        """
        加载MLX模型。

        :param kwargs: 额外的加载参数
        :return: 加载的模型和分词器
        """
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError:
            error_message = "Failed to import module 'mlx_lm'"
            installation_guide = [
                "Please make sure 'mlx_lm' is installed. ",
                "You can install it by `pip install mlx_lm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer_config = dict(
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        logger.debug(
            "loading model with tokenizer config: %s, model config: %s",
            tokenizer_config,
            self._model_config,
        )

        cache_limit_gb = kwargs.get("cache_limit_gb", None)
        if cache_limit_gb:
            logger.debug(f"Setting cache limit to {cache_limit_gb} GB")
            mx.metal.set_cache_limit(cache_limit_gb * 1024 * 1024 * 1024)

        return load(
            self.model_path,
            tokenizer_config=tokenizer_config,
            model_config=self._model_config,
        )

    def load(self):
        """
        加载MLX模型和分词器。
        """
        kwargs = {}
        kwargs["revision"] = self._model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._model_config.get("trust_remote_code")
        kwargs["cache_limit_gb"] = self._model_config.pop("cache_limit_gb", None)

        self._model, self._tokenizer = self._load_model(**kwargs)

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        """
        判断给定的模型家族、规格和量化方法是否匹配MLX模型。

        :param llm_family: 模型家族
        :param llm_spec: 模型规格
        :param quantization: 量化方法
        :return: 是否匹配
        """
        if llm_spec.model_format not in ["mlx"]:
            return False
        if sys.platform != "darwin" or platform.processor() != "arm":
            # only work for Mac M chips
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def _generate_stream(self, prompt: str, kwargs: MLXGenerateConfig):
        """
        生成流式输出。

        :param prompt: 输入提示
        :param kwargs: 生成配置
        :yield: 生成的完成块和使用情况
        """
        import mlx.core as mx
        from mlx_lm.utils import generate_step

        model = self._model
        model_uid = self.model_uid
        tokenizer = self._tokenizer
        max_tokens = kwargs["max_tokens"]
        chunk_id = str(uuid.uuid4())
        stop_token_ids = kwargs.get("stop_token_ids", [])
        stream = kwargs.get("stream", False)
        stream_options = kwargs.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        prompt_tokens = mx.array(tokenizer.encode(prompt))
        input_echo_len = len(prompt_tokens)

        i = 0
        start = time.time()
        output = ""
        for (token, _), i in zip(
            generate_step(
                prompt_tokens,
                model,
                temp=kwargs["temperature"],
                repetition_penalty=kwargs["repetition_penalty"],
                repetition_context_size=kwargs["repetition_context_size"],
                top_p=kwargs["top_p"],
                logit_bias=kwargs["logit_bias"],
            ),
            range(max_tokens),
        ):
            if token == tokenizer.eos_token_id or token in stop_token_ids:  # type: ignore
                break

            # Yield the last segment if streaming
            out = tokenizer.decode(
                token,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            if stream:
                # this special character is mainly for qwen
                out = out.strip("�")
                output = out
            else:
                output += out

            completion_choice = CompletionChoice(
                text=output, index=0, logprobs=None, finish_reason=None
            )
            completion_chunk = CompletionChunk(
                id=chunk_id,
                object="text_completion",
                created=int(time.time()),
                model=model_uid,
                choices=[completion_choice],
            )
            completion_usage = CompletionUsage(
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            )

            yield completion_chunk, completion_usage

        logger.info(
            f"Average generation speed: {i / (time.time() - start):.2f} tokens/s."
        )

        if i == max_tokens - 1:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        if stream:
            completion_choice = CompletionChoice(
                text="", index=0, logprobs=None, finish_reason=finish_reason
            )
        else:
            completion_choice = CompletionChoice(
                text=output, index=0, logprobs=None, finish_reason=finish_reason
            )

        completion_chunk = CompletionChunk(
            id=chunk_id,
            object="text_completion",
            created=int(time.time()),
            model=model_uid,
            choices=[completion_choice],
        )
        completion_usage = CompletionUsage(
            prompt_tokens=input_echo_len,
            completion_tokens=i,
            total_tokens=(input_echo_len + i),
        )

        yield completion_chunk, completion_usage

        if include_usage:
            completion_chunk = CompletionChunk(
                id=chunk_id,
                object="text_completion",
                created=int(time.time()),
                model=model_uid,
                choices=[],
            )
            completion_usage = CompletionUsage(
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            )
            yield completion_chunk, completion_usage

    def generate(
        self, prompt: str, generate_config: Optional[MLXGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """
        生成文本完成。

        :param prompt: 输入提示
        :param generate_config: 生成配置
        :return: 完成结果或完成块迭代器
        """
        def generator_wrapper(
            prompt: str, generate_config: MLXGenerateConfig
        ) -> Iterator[CompletionChunk]:
            for completion_chunk, completion_usage in self._generate_stream(
                prompt,
                generate_config,
            ):
                completion_chunk["usage"] = completion_usage
                yield completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)
        if not stream:
            for completion_chunk, completion_usage in self._generate_stream(
                prompt,
                generate_config,
            ):
                pass
            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
            return completion
        else:
            return generator_wrapper(prompt, generate_config)


class MLXChatModel(MLXModel, ChatModelMixin):
    """
    MLXChatModel类，用于处理MLX聊天模型。
    继承自MLXModel和ChatModelMixin。
    """

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[MLXModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        """
        初始化MLXChatModel实例。

        :param model_uid: 模型的唯一标识符
        :param model_family: 模型家族
        :param model_spec: 模型规格
        :param quantization: 量化方法
        :param model_path: 模型路径
        :param model_config: MLX模型配置
        :param peft_model: PEFT模型列表
        """
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            model_config,
            peft_model,
        )

    def _sanitize_generate_config(
        self,
        generate_config: Optional[MLXGenerateConfig],
    ) -> MLXGenerateConfig:
        """
        清理并补充生成配置。

        :param generate_config: 生成配置
        :return: 清理后的生成配置
        """
        generate_config = super()._sanitize_generate_config(generate_config)
        if (
            (not generate_config.get("stop"))
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop
        ):
            generate_config["stop"] = self.model_family.prompt_style.stop.copy()
        if (
            generate_config.get("stop_token_ids", None) is None
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop_token_ids
        ):
            generate_config[
                "stop_token_ids"
            ] = self.model_family.prompt_style.stop_token_ids.copy()

        return generate_config

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        """
        判断给定的模型家族、规格和量化方法是否匹配MLX聊天模型。

        :param llm_family: 模型家族
        :param llm_spec: 模型规格
        :param quantization: 量化方法
        :return: 是否匹配
        """
        if llm_spec.model_format not in ["mlx"]:
            return False
        if sys.platform != "darwin" or platform.processor() != "arm":
            # only work for Mac M chips
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[MLXGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        执行聊天生成。

        :param prompt: 用户输入的提示
        :param system_prompt: 系统提示
        :param chat_history: 聊天历史
        :param generate_config: 生成配置
        :return: 聊天完成或聊天完成块迭代器
        """
        tools = generate_config.pop("tools", []) if generate_config else None  # type: ignore
        full_prompt = self.get_full_prompt(
            self.model_family, prompt, system_prompt, chat_history, tools
        )

        generate_config = self._sanitize_generate_config(generate_config)
        # TODO(codingl2k1): qwen hacky to set stop for function call.
        model_family = self.model_family.model_family or self.model_family.model_name
        if tools and model_family in ["qwen-chat", "qwen1.5-chat"]:
            stop = generate_config.get("stop")
            if isinstance(stop, str):
                generate_config["stop"] = [stop, "Observation:"]
            elif isinstance(stop, Iterable):
                assert not isinstance(stop, str)
                generate_config["stop"] = list(stop) + ["Observation:"]
            else:
                generate_config["stop"] = "Observation:"

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._tool_calls_completion(
                    self.model_family, self.model_uid, c, tools
                )
            return self._to_chat_completion(c)
