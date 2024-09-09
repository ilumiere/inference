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
import time
import uuid
from typing import AsyncGenerator, Dict, Iterator, List, Optional, TypedDict, Union

import torch

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionUsage,
    LoRA,
)
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)

# noqa 的含义：
# "noqa" 是 "no quality assurance" 的缩写。
# 它告诉代码检查工具忽略这一行的特定警告。
# F401 的含义：
# 这是 Flake8（一个流行的 Python 代码检查工具）中的一个特定错误代码。
# F401 表示 "module imported but unused"（模块被导入但未使用）。
# 为什么要使用 # noqa: F401：
# 在这个例子中，lmdeploy 被导入但没有直接使用。
# 通常，这会触发一个未使用导入的警告。
# 但在这里，导入是为了检查模块是否可用，而不是为了直接使用它。
try:
    import lmdeploy  # noqa: F401

    LMDEPLOY_INSTALLED = True
except ImportError:
    LMDEPLOY_INSTALLED = False

LMDEPLOY_SUPPORTED_CHAT_MODELS = ["internvl2"]
LMDEPLOY_MODEL_CHAT_TEMPLATE_NAME = {
    "internvl2": "internvl-internlm2",
}


class LMDeployModelConfig(TypedDict, total=False):
    model_format: Optional[str]  # 模型格式
    tp: Optional[int]  # 张量并行度
    session_len: Optional[int]  # 会话长度
    max_batch_size: Optional[int]  # 最大批处理大小
    cache_max_entry_count: Optional[float]  # 缓存最大条目数
    cache_block_seq_len: Optional[int]  # 缓存块序列长度
    enable_prefix_caching: Optional[bool]  # 是否启用前缀缓存
    quant_policy: Optional[int]  # 量化策略
    rope_scaling_factor: Optional[float]  # RoPE缩放因子
    use_logn_attn: Optional[bool]  # 是否使用对数注意力
    download_dir: Optional[str]  # 下载目录
    revision: Optional[str]  # 模型版本
    max_prefill_token_num: Optional[int]  # 最大预填充token数
    num_tokens_per_iter: Optional[int]  # 每次迭代的token数
    max_prefill_iters: Optional[int]  # 最大预填充迭代次数


class LMDeployGenerateConfig(TypedDict, total=False):
    n: Optional[int]  # 生成的序列数量
    max_new_tokens: Optional[int]  # 最大新生成的token数
    top_p: Optional[float]  # 累积概率阈值
    top_k: Optional[int]  # 保留的最高概率token数
    temperature: Optional[float]  # 采样温度
    repetition_penalty: Optional[float]  # 重复惩罚系数
    ignore_eos: Optional[bool]  # 是否忽略结束符
    random_seed: Optional[int]  # 随机种子
    stop_words: Optional[List[str]]  # 停止词列表
    bad_words: Optional[List[str]]  # 禁用词列表
    min_new_tokens: Optional[int]  # 最小新生成的token数
    skip_special_tokens: Optional[bool]  # 是否跳过特殊token
    logprobs: Optional[int]  # 返回的对数概率数量


class LMDeployModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[LMDeployModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        # 调用父类的初始化方法
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        # 清理并设置模型配置
        self._model_config: LMDeployModelConfig = self._sanitize_model_config(
            model_config
        )
        # 目前不支持LoRA，如果提供了peft_model，则抛出异常
        if peft_model is not None:
            raise ValueError("LMDEPLOY engine has not supported lora yet.")

    def _sanitize_model_config(
        self, model_config: Optional[LMDeployModelConfig]
    ) -> LMDeployModelConfig:
        # 如果没有提供模型配置，则创建一个空的配置
        if model_config is None:
            model_config = LMDeployModelConfig()
        # 设置默认的会话长度
        model_config.setdefault("session_len", 8192)
        # 如果模型格式是AWQ，则设置模型格式为AWQ
        if self.model_spec.model_format == "awq":
            model_config.setdefault("model_format", "awq")
        return model_config

    def load(self):
        # 尝试导入lmdeploy模块
        try:
            import lmdeploy  # noqa: F401, F811
        except ImportError:
            # 如果导入失败，则提供安装指南
            error_message = "Failed to import module 'lmdeploy'"
            installation_guide = [
                "Please make sure 'lmdeploy' is installed. ",
                "You can install it by `pip install lmdeploy`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        # 目前不支持生成功能，抛出异常
        raise ValueError("LMDEPLOY engine has not supported generate yet.")

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        # 目前总是返回False，表示不匹配任何模型
        return False

    def generate(
        self,
        prompt: str,
        generate_config: Optional[Dict] = None,
    ) -> Union[Completion, Iterator[ChatCompletionChunk]]:
        # 生成功能尚未实现，抛出NotImplementedError异常
        raise NotImplementedError("LMDeploy generate ablility does not support now.")


class LMDeployChatModel(LMDeployModel, ChatModelMixin):
    def load(self):
        # 尝试导入必要的lmdeploy模块
        try:
            from lmdeploy import (
                ChatTemplateConfig,
                TurbomindEngineConfig,
                VisionConfig,
                pipeline,
            )
        except ImportError:
            # 如果导入失败，提供安装指南
            error_message = "Failed to import module 'lmdeploy'"
            installation_guide = [
                "Please make sure 'lmdeploy' is installed. ",
                "You can install it by `pip install lmdeploy`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # 确定正确的聊天模板名称
        chat_temp_name = ""
        family = self.model_family.model_family or self.model_family.model_name
        for key in LMDEPLOY_MODEL_CHAT_TEMPLATE_NAME.keys():
            if family in key:
                chat_temp_name = LMDEPLOY_MODEL_CHAT_TEMPLATE_NAME[key]
                break
        if chat_temp_name == "":
            raise ValueError(f"Can not find correct chat template.")

        # 配置聊天模板
        chat_template_config = ChatTemplateConfig(chat_temp_name)
        chat_template_config.meta_instruction = (
            self.model_family.prompt_style.system_prompt
        )
        
        # 检查是否有多个CUDA设备，如果有则设置张量并行
        count = torch.cuda.device_count()
        if count > 1:
            self._model_config.setdefault("tp", torch.cuda.device_count())

        # 创建模型管道
        self._model = pipeline(
            self.model_path,
            chat_template_config=chat_template_config,
            backend_config=TurbomindEngineConfig(**self._model_config),
            vision_config=VisionConfig(thread_safe=True),
        )

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        # 检查模型是否匹配LMDeploy支持的条件
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            # 目前AWQ只支持4位权重量化
            if "4" not in quantization:
                return False
        if llm_family.model_name not in LMDEPLOY_SUPPORTED_CHAT_MODELS:
            return False
        return LMDEPLOY_INSTALLED

    async def async_chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        # 异步聊天方法，支持流式和非流式输出
        stream = (
            generate_config.get("stream", False)
            if isinstance(generate_config, dict)
            else False
        )
        stream_options = (
            generate_config.get("stream_options", None)
            if isinstance(generate_config, dict)
            else False
        )
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        chat_history = chat_history or []

        if stream:
            chunk = self._chat_stream(prompt, chat_history, include_usage)
            return self._async_to_chat_completion_chunks(chunk)
        else:
            chunk = await self._chat(prompt, chat_history)
            return self._to_chat_completion(chunk)

    async def _chat_stream(self, prompt, chat_history, include_usage):
        # 流式聊天方法的实现
        from lmdeploy.messages import Response

        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        completion_id = str(uuid.uuid1())
        async for output in self._generate(
            prompt,
            chat_history,
            session_id=-1,
            stream_response=True,
        ):
            new_text = output.text if isinstance(output, Response) else output.response

            completion_choice = ChatCompletionChunkChoice(
                text=new_text,
                index=0,
                logprobs=None,
                finish_reason=output.finish_reason,
            )
            chunk = ChatCompletionChunk(
                id=completion_id,
                object="chat.completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[completion_choice],
            )
            prompt_tokens = output.input_token_len
            completion_tokens = output.generate_token_len
            total_tokens = prompt_tokens + completion_tokens
            completion_usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            chunk["usage"] = completion_usage
            print(chunk)
            yield chunk
        if include_usage:
            chunk = ChatCompletionChunk(
                id=completion_id,
                object="chat.completion",
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

    async def _chat(self, prompt, chat_history):
        # 非流式聊天方法的实现
        from lmdeploy.messages import Response

        response, finish_reason = "", ""
        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        async for output in self._generate(
            prompt,
            chat_history,
            session_id=-1,
            stream_response=False,
        ):
            response += output.text if isinstance(output, Response) else output.response
            prompt_tokens = output.input_token_len
            completion_tokens = output.generate_token_len
            total_tokens = output.input_token_len + output.generate_token_len
            finish_reason = output.finish_reason

        chunk = ChatCompletion(
            id=str(uuid.uuid1()),
            object="chat.completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[
                CompletionChoice(
                    index=0, text=response, finish_reason=finish_reason, logprobs=None
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )
        return chunk

    # 从lmdeploy复制
    # 参考: lmdeploy.serve.async_engine.py
    async def _generate(
        self,
        prompt,
        chat_history,
        session_id: int,
        generate_config: Optional[Dict] = None,
        tools: Optional[List[object]] = None,
        stream_response: bool = True,
        sequence_start: bool = True,
        sequence_end: bool = True,  # 默认不使用交互模式
        step: int = 0,
        do_preprocess: bool = False,
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        # 生成方法的核心实现
        import random

        from lmdeploy.messages import EngineGenerationConfig, GenerationConfig
        from lmdeploy.serve.async_engine import GenOut
        from lmdeploy.tokenizer import DetokenizeState

        session_id = -1

        # 初始化或更新会话状态
        if str(session_id) not in self._model.id2step:
            self._model.id2step[str(session_id)] = 0
        if generate_config is None:
            generate_config = GenerationConfig()
        if type(generate_config) is GenerationConfig:
            generate_config = EngineGenerationConfig.From(
                generate_config, self._model.tokenizer
            )
        if generate_config.stop_words is None:  # type: ignore
            generate_config.stop_words = self._model.stop_words  # type: ignore
        if generate_config.random_seed is None and sequence_start:  # type: ignore
            generate_config.random_seed = random.getrandbits(64)  # type: ignore
        if generate_config.n > 1:  # type: ignore
            logger.warning(
                f"n({generate_config.n}) > 1 hasn't been supported yet. "  # type: ignore
                f"Fallback to 1"
            )
            generate_config.n = 1  # type: ignore

        # 准备输入
        prompt_input = await self._get_prompt_input(prompt, chat_history)
        prompt = prompt_input["prompt"]
        input_ids = prompt_input["input_ids"]
        finish_reason = None
        logger.info(
            f"prompt={prompt!r}, "
            f"gen_config={generate_config}, "
            f"prompt_token_id={input_ids}, "
            f"adapter_name={adapter_name}."
        )
        logger.info(
            f"session_id={session_id}, "  # type: ignore
            f"history_tokens={self._model.id2step[str(session_id)]}, "
            f"input_tokens={len(input_ids)}, "
            f"max_new_tokens={generate_config.max_new_tokens}, "
            f"seq_start={sequence_start}, seq_end={sequence_end}, "
            f"step={step}, prep={do_preprocess}"
        )

        # 调整最大新token数
        if generate_config.max_new_tokens is None:  # type: ignore
            # 对于交互式端点，将尝试最大可能的token数
            generate_config.max_new_tokens = max(  # type: ignore
                128,
                self._model.session_len
                - self._model.id2step[str(session_id)]
                - len(input_ids),
            )
        elif (
            self._model.id2step[str(session_id)]
            + len(input_ids)
            + generate_config.max_new_tokens  # type: ignore
            > self._model.session_len
        ):
            generate_config.max_new_tokens = max(  # type: ignore
                self._model.session_len
                - self._model.id2step[str(session_id)]
                - len(input_ids),
                128,
            )
            logger.error(f"Truncate max_new_tokens to {generate_config.max_new_tokens}")  # type: ignore

        # 检查是否超出token限制
        if (
            self._model.id2step[str(session_id)]
            + len(input_ids)
            + generate_config.max_new_tokens  # type: ignore
            > self._model.session_len
        ):
            logger.error(f"run out of tokens. session_id={session_id}.")
            yield GenOut(
                "", self._model.id2step[str(session_id)], len(input_ids), 0, "length"
            )
            if sequence_end is True and sequence_start is False:
                await self._model.end_session(session_id)
        else:
            # 开始生成过程
            generator = await self._model.get_generator(False, session_id)
            async with self._model.safe_run(session_id):
                state = DetokenizeState(len(input_ids))
                start_ids_offset = state.ids_offset
                response = ""
                async for outputs in generator.async_stream_infer(
                    session_id=session_id,
                    **prompt_input,
                    gen_config=generate_config,
                    adapter_name=adapter_name,
                    stream_output=stream_response,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=self._model.id2step[str(session_id)],
                ):
                    # 解码结果
                    res, tokens = (
                        input_ids + outputs.token_ids,
                        outputs.num_token,
                    )  # noqa
                    if len(res) <= state.ids_offset:
                        continue

                    ids_offset = state.ids_offset
                    response, state = self._model.tokenizer.detokenize_incrementally(
                        res,
                        state,
                        skip_special_tokens=generate_config.skip_special_tokens,  # type: ignore
                    )

                    res = res[ids_offset:]
                    logprobs = None
                    if outputs.logprobs:
                        log_offset = ids_offset - start_ids_offset
                        logprobs = outputs.logprobs[log_offset:]

                    # response, history token len,
                    # input token len, gen token len
                    yield GenOut(
                        response,
                        self._model.id2step[str(session_id)],
                        len(input_ids),
                        tokens,
                        finish_reason,
                        res,
                        logprobs,
                    )

                # 确定结束原因
                finish_reason = (
                    "length" if tokens >= generate_config.max_new_tokens else "stop"  # type: ignore
                )
                # utf-8 char at the end means it's a potential unfinished
                # byte sequence
                if not response.endswith("�"):
                    response = ""  # avaid returning the last response twice
                yield GenOut(
                    response,
                    self._model.id2step[str(session_id)],
                    len(input_ids),
                    tokens,
                    finish_reason,
                )
                # 更新步骤
                self._model.id2step[str(session_id)] += len(input_ids) + tokens
                if sequence_end:
                    self._model.id2step[str(session_id)] = 0
                # 手动结束PyTorch会话
                # TODO 修改PyTorch或TurboMind API
                if self._model.backend == "pytorch" and sequence_end:
                    await self._model.end_session(session_id)

    # 从lmdeploy复制
    # 参考: lmdeploy.serve.vl_async_engine.py
    async def _get_prompt_input(
        self,
        prompt: Union[str, List[Dict]],
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        sequence_start: bool = True,
        tools: Optional[List[object]] = None,
        **kwargs,
    ):
        """获取input_ids、embeddings和offsets。"""
        IMAGE_TOKEN = "<IMAGE_TOKEN>"
        IMAGE_DUMMY_TOKEN_INDEX = 0
        import numpy as np

        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        chat_history = chat_history or []

        # 准备提示
        decorated, _ = self.get_prompt(prompt, chat_history, prompt_style)  # type: ignore
        chat_history.append(ChatCompletionMessage(role="user", content=prompt))  # type: ignore
        prompt = chat_history  # type: ignore

        decorated = decorated.replace("<image>", "<img><IMAGE_TOKEN></img>")

        segs = decorated.split(IMAGE_TOKEN)

        results = {}
        input_ids = []  # type: ignore
        if len(segs) > 1:
            images = await self._model.vl_prompt_template.async_collect_pil_images(
                prompt
            )

            features = await self._model.vl_encoder.async_infer(images)

            from lmdeploy.vl.templates import MiniCPMVTempateWrapper

            if isinstance(self._model.vl_prompt_template, MiniCPMVTempateWrapper):
                (
                    decorated,
                    features,
                ) = self._model.vl_prompt_template.update_image_token(  # noqa: E501
                    decorated, features
                )
                segs = decorated.split(IMAGE_TOKEN)

            features = [x.cpu().numpy() for x in features]
            input_ids = []
            begins = []
            ends = []
            if len(segs) != len(features) + 1:
                logger.error(
                    f"the number of {IMAGE_TOKEN} is not equal "
                    f"to input images, {len(segs) - 1} vs {len(features)}"
                )
                features = features[: len(segs) - 1]
            for i, seg in enumerate(segs):
                if i > 0 and i <= len(features):
                    image_dim = features[i - 1].shape[0]
                    begins.append(len(input_ids))
                    ends.append(begins[-1] + image_dim)
                    input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
                seg_ids = self._model.tokenizer.encode(
                    seg, add_bos=((i == 0) and sequence_start)
                )
                input_ids.extend(seg_ids)
            ranges = np.stack([begins, ends], axis=1).tolist()
            results["input_embeddings"] = features
            results["input_embedding_ranges"] = ranges
        else:
            input_ids = self._model.tokenizer.encode(decorated, add_bos=sequence_start)

        results["input_ids"] = input_ids
        results["prompt"] = decorated

        return results
