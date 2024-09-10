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

import json
import logging
import os
from functools import lru_cache
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import torch

from ....core.scheduler import InferenceRequest
from ....device_utils import (
    get_device_preferred_dtype,
    gpu_count,
    is_hf_accelerate_supported,
)
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CreateCompletionTorch,
    LoRA,
    PytorchGenerateConfig,
    PytorchModelConfig,
)
from ...utils import select_device
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin
from .utils import get_context_length, get_max_src_len, pad_prefill_tokens

logger = logging.getLogger(__name__)

NON_DEFAULT_MODEL_LIST: List[str] = [
    "chatglm3",
    "chatglm3-32k",
    "chatglm3-128k",
    "glm4-chat",
    "glm4-chat-1m",
    "llama-2",
    "llama-2-chat",
    "internlm2-chat",
    "internlm2.5-chat",
    "qwen-vl-chat",
    "OmniLMM",
    "yi-vl-chat",
    "deepseek-vl-chat",
    "internvl-chat",
    "internvl2",
    "cogvlm2",
    "cogvlm2-video-llama3-chat",
    "MiniCPM-Llama3-V-2_5",
    "MiniCPM-V-2.6",
    "glm-4v",
]


class PytorchModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        # 调用父类的初始化方法
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        # 设置是否使用快速分词器
        self._use_fast_tokenizer = True
        # 对模型配置进行清理和验证
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            pytorch_model_config
        )
        # 存储PEFT模型
        self._peft_model = peft_model

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        # 如果没有提供配置，创建一个新的配置对象
        if pytorch_model_config is None:
            pytorch_model_config = PytorchModelConfig()
        # 设置默认值
        pytorch_model_config.setdefault("revision", self.model_spec.model_revision)
        pytorch_model_config.setdefault("gptq_ckpt", None)
        pytorch_model_config.setdefault("gptq_wbits", 16)
        pytorch_model_config.setdefault("gptq_groupsize", -1)
        pytorch_model_config.setdefault("gptq_act_order", False)
        pytorch_model_config.setdefault("device", "auto")
        pytorch_model_config.setdefault("trust_remote_code", True)
        pytorch_model_config.setdefault("max_num_seqs", 16)
        pytorch_model_config.setdefault("enable_tensorizer", False)
        return pytorch_model_config

    def _sanitize_generate_config(
        self,
        generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        # 如果没有提供生成配置，创建一个新的配置对象
        # 默认值初始化：
        # CreateCompletionTorch() 创建了一个包含所有默认值的实例。
        # .dict() 将这个实例转换为字典，包含所有字段及其默认值。
        # 配置转换：
        # PytorchGenerateConfig(**...) 使用这个字典创建一个新的 PytorchGenerateConfig 对象。
        # ** 操作符将字典展开为关键字参数。
        # 3. 类型一致性：
        # 确保 generate_config 是 PytorchGenerateConfig 类型，这可能是模型期望的配置类型。
        # 灵活性：
        # 允许 CreateCompletionTorch 和 PytorchGenerateConfig 独立演化，只要它们的字段名称保持兼容。
        if generate_config is None:
            generate_config = PytorchGenerateConfig(**CreateCompletionTorch().dict())
        else:
            # Validate generate_config and fill default values to the generate config.
            generate_config = PytorchGenerateConfig(
                **CreateCompletionTorch(**generate_config).dict()
            )
        generate_config["model"] = self.model_uid
        return generate_config

    def _check_tensorizer_integrity(self):
        # 检查是否启用了tensorizer
        if not self._pytorch_model_config.get("enable_tensorizer"):
            return False

        from .tensorizer_utils import check_tensorizer_integrity

        # 检查tensorizer文件的完整性
        integrity = check_tensorizer_integrity(
            self.model_path,
            [component[0] for component in self._get_components()],
        )
        logger.info(f"Tensorizer files integrity: {integrity} {self.model_uid}")
        return integrity

    def _load_tensorizer(self, **kwargs):
        # 检查是否启用了tensorizer
        enable_tensorizer = self._pytorch_model_config.get("enable_tensorizer", None)
        if enable_tensorizer:
            from .tensorizer_utils import load_from_tensorizer

            # 获取组件元数据
            component_metadata = [
                (name, type, kwargs)
                for name, _, type, kwargs in self._get_components(**kwargs)
            ]
            # 从tensorizer加载模型和分词器
            model, tokenizer = load_from_tensorizer(
                self.model_path, component_metadata, self._get_model_class(), **kwargs
            )
            return model, tokenizer

    def _save_tensorizer(self, **kwargs):
        # 检查是否启用了tensorizer
        enable_tensorizer = self._pytorch_model_config.get("enable_tensorizer", None)
        if enable_tensorizer:
            from .tensorizer_utils import save_to_tensorizer

            # 获取组件
            components = [(name, obj) for name, obj, _, _ in self._get_components()]
            # 将模型保存为tensorizer格式
            save_to_tensorizer(self.model_path, self._model, components, **kwargs)

    def _get_model_class(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def _get_components(self, **kwargs):
        # 导入AutoTokenizer类
        """
        kwargs = {
         "trust_remote_code": False,
         "revision": "main",
         "code_revision": "v1.0"
        }

        result = self._get_components(**kwargs)
        
        解释：
        返回值是一个列表，包含一个元组。
        元组的四个元素分别是：
        组件名称 ("tokenizer")
        当前对象的tokenizer实例（如果存在）
        使用的Tokenizer类 (AutoTokenizer)
        配置字典
        配置字典包含：

        Returns:
            _type_: _description_
        """
        from transformers import AutoTokenizer

        # 返回分词器组件的配置
        return [
            (
                "tokenizer",  # 组件名称
                getattr(self, "_tokenizer", None),  # 获取当前对象的_tokenizer属性，如果不存在则返回None
                AutoTokenizer,  # 使用的分词器类
                {
                    # 分词器的配置参数
                    "use_fast": self._use_fast_tokenizer,  # 是否使用快速分词器
                    "trust_remote_code": kwargs.get("trust_remote_code", True),  # 是否信任远程代码，默认为True
                    "revision": kwargs.get("revision"),  # 模型的版本或分支
                    "code_revision": kwargs.get("code_revision", None),  # 代码的版本或分支，默认为None
                },
            )
        ]

    def _load_model(self, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )

        return model, tokenizer

    def _apply_lora(self):
        # 如果有PEFT模型，应用LoRA
        if self._peft_model is not None:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    f"Failed to import 'PeftModel' from 'peft'. Please make sure 'peft' is installed.\n\n"
                )

            for i, peft_model in enumerate(self._peft_model):
                if i == 0:
                    # 加载第一个PEFT模型
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        peft_model.local_path,
                        adapter_name=peft_model.lora_name,
                    )
                else:
                    # 加载额外的PEFT模型
                    self._model.load_adapter(
                        peft_model.local_path, adapter_name=peft_model.lora_name
                    )
                logger.info(
                    f"PEFT adaptor '{peft_model.lora_name}' successfully loaded for model '{self.model_uid}'."
                )

    def load(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
            )
        from .compression import load_compress_model

        quantization = self.quantization
        num_gpus = gpu_count()
        device = self._pytorch_model_config.get("device", "auto")
        self._pytorch_model_config["device"] = select_device(device)
        self._device = self._pytorch_model_config["device"]

        kwargs = {}

        # 获取设备首选的数据类型
        dtype = get_device_preferred_dtype(self._device)

        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        else:
            raise ValueError(f"Device {self._device} is not supported in temporary")

        kwargs["revision"] = self._pytorch_model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._pytorch_model_config.get(
            "trust_remote_code"
        )
        model_format = self.model_spec.model_format

        is_device_map_auto = False

        # This is required for Intel GPU to actually work with accelerate device_map until
        # https://github.com/intel/intel-extension-for-pytorch/issues/522
        # is resolved
        max_memory_env = os.getenv("ACCELERATE_MAX_MEMORY", None)

        if max_memory_env is not None:
            max_memory_raw = json.loads(max_memory_env)
            max_memory = {
                int(k) if k.isdigit() else k: max_memory_raw[k] for k in max_memory_raw
            }
            kwargs["max_memory"] = max_memory

        # 处理量化
        if quantization != "none" and model_format == "pytorch":
            if self._device == "cuda" and self._is_linux():
                kwargs["device_map"] = "auto"
                is_device_map_auto = True
                if quantization == "4-bit":
                    kwargs["load_in_4bit"] = True
                    kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    kwargs["bnb_4bit_use_double_quant"] = True
                    kwargs["llm_int8_skip_modules"] = [
                        "lm_head",
                        "encoder",
                        "EncDecAttention",
                    ]
                elif quantization == "8-bit":
                    kwargs["load_in_8bit"] = True
                else:
                    raise ValueError(
                        f"Quantization {quantization} is not supported in temporary"
                    )
            else:
                if num_gpus != 1 and self._device == "cuda":
                    raise ValueError(f"Quantization is not supported for multi-gpu")
                elif quantization != "8-bit":
                    raise ValueError(
                        f"Only 8-bit quantization is supported if it is not linux system or cuda device"
                    )
                else:
                    # 加载压缩模型
                    (
                        self._model,
                        self._tokenizer,
                    ) = load_compress_model(
                        model_path=self.model_path,
                        device=self._device,
                        torch_dtype=kwargs["torch_dtype"],
                        use_fast=self._use_fast_tokenizer,
                        revision=kwargs["revision"],
                    )
                    logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")
                    return

        if num_gpus > 0 and is_hf_accelerate_supported(self._device):
            kwargs.update({"device_map": "auto"})
            is_device_map_auto = True

        # 检查tensorizer完整性并加载模型
        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer(**kwargs)
        else:
            self._model, self._tokenizer = self._load_model(**kwargs)

        # 应用LoRA
        self._apply_lora()

        if not is_device_map_auto:
            self._model.to(self._device)

        # 保存为tensorizer格式
        self._save_tensorizer(**kwargs)

        logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        # 检查模型是否匹配当前类
        if llm_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        model_family = llm_family.model_family or llm_family.model_name
        if model_family in NON_DEFAULT_MODEL_LIST:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        from .utils import generate_stream

        def generator_wrapper(
            prompt: str, generate_config: PytorchGenerateConfig
        ) -> Iterator[CompletionChunk]:
            for completion_chunk, completion_usage in generate_stream(
                self.model_uid,
                self._model,
                self._tokenizer,
                prompt,
                self._device,
                generate_config,
            ):
                completion_chunk["usage"] = completion_usage
                yield completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        # 清理和验证生成配置
        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        # 处理LoRA模型
        lora_model = generate_config.pop("lora_name")

        if lora_model is not None and self._peft_model is not None:
            for lora in self._peft_model:
                if lora_model == lora.lora_name:
                    self._model.set_adapter(lora_model)
                    logger.info(f"Set lora model to {lora_model}")
                    break
            else:
                self._model.disable_adapter()
                logger.info(f"No lora model {lora_model} found, skip setting")

        # 生成文本
        stream = generate_config.get("stream", False)
        if not stream:
            for completion_chunk, completion_usage in generate_stream(
                self.model_uid,
                self._model,
                self._tokenizer,
                prompt,
                self._device,
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

    def build_prefill_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        构建预填充阶段的注意力掩码。
        在左侧填充 `0`。
        注意参数 `seq_length` 来自 `input_ids`。
        """
        data = []
        for r in reqs:
            real_len = seq_length - r.padding_len
            x = torch.cat(
                [
                    torch.full((r.padding_len,), 0, dtype=torch.long),
                    torch.ones((real_len,), dtype=torch.long),
                ]
            )
            data.append(x)
            r.extra_kwargs["attention_mask_seq_len"] = real_len
        return torch.stack(data).to(self._device)

    def build_decode_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        为解码阶段构建注意力掩码。
        注意参数 `seq_length` 来自合并的 kv_cache。
        因此我们需要再次在左侧填充 `0`。
        """
        data = []
        for r in reqs:
            # 增加注意力掩码序列长度
            r.extra_kwargs["attention_mask_seq_len"] += 1
            attention_mask_seq_len = r.extra_kwargs["attention_mask_seq_len"]
            # 计算需要填充的长度
            pad_len = seq_length - attention_mask_seq_len
            # 构建注意力掩码
            x = torch.cat(
                [
                    torch.full((pad_len,), 0, dtype=torch.long),  # 左侧填充0
                    torch.ones((attention_mask_seq_len,), dtype=torch.long),  # 右侧填充1
                ]
            )
            data.append(x)
        # 将所有注意力掩码堆叠并移动到指定设备
        return torch.stack(data).to(self._device)

    def build_prefill_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build position ids for prefill phase.
        Padding `0` on the left.
        Note that the parameter `seq_length` is from `input_ids`.
        Record the `max_position_id` on request for the decode phase.
        """
        res = []
        for r in reqs:
            real_seq_len = seq_length - r.padding_len
            res.append(
                torch.cat(
                    [
                        torch.full((r.padding_len,), 0, dtype=torch.long),  # 左侧填充0
                        torch.arange(0, real_seq_len, dtype=torch.long),  # 生成位置ID序列
                    ]
                )
            )
            # 记录最大位置ID
            r.extra_kwargs["max_position_id"] = real_seq_len - 1
        return torch.stack(res).to(self._device)

    def build_decode_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build position ids for decode phase.
        For most models, just let the `max_position_id` in previous step += 1 and use the latest `max_position_id`
        """
        data = []
        for r in reqs:
            # 增加最大位置ID
            r.extra_kwargs["max_position_id"] += 1
            data.append([r.extra_kwargs["max_position_id"]])
        position_ids = torch.as_tensor(data, dtype=torch.long, device=self._device)
        return position_ids

    def build_prefill_token_type_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        为预填充阶段构建token_type_ids。
        对于大多数模型，这不是必需的。
        """
        return None

    def build_decode_token_type_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        为解码阶段构建token_type_ids。
        对于大多数模型，这不是必需的。
        """
        return None

    def build_prefill_inputs(self, prompts: List, req_list: List[InferenceRequest]):
        """
        获取推理的输入。不同模型可能有自己的实现。
        """
        assert isinstance(prompts[0], str)
        # 对提示进行分词
        inputs = self._tokenizer(prompts, padding=False).input_ids
        context_len = self.get_context_len()
        # 对输入进行填充并转移到指定设备
        input_ids = torch.as_tensor(
            pad_prefill_tokens(inputs, context_len, req_list), device=self._device
        )
        return input_ids

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        """
        获取预填充阶段的所有输入参数。不同模型可能有自己的实现。
        """
        input_ids = self.build_prefill_inputs(prompts, req_list)
        res = {"input_ids": input_ids}
        batch_size, seq_len = input_ids.shape
        # 构建注意力掩码
        attention_mask = self.build_prefill_attention_mask(
            batch_size, seq_len, req_list
        )
        if attention_mask is not None:
            res["attention_mask"] = attention_mask
        # 构建位置ID
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            res["position_ids"] = position_ids
        # 构建token类型ID
        token_type_ids = self.build_prefill_token_type_ids(
            batch_size, seq_len, req_list
        )
        if token_type_ids is not None:
            res["token_type_ids"] = token_type_ids
        return res

    def build_decode_kwargs(
        self,
        prompts: List,
        req_list: List[InferenceRequest],
        batch_size: int,
        seq_len: int,
    ):
        """
        获取解码阶段的所有输入参数。不同模型可能有自己的实现。
        """
        res = {"input_ids": torch.as_tensor(prompts, device=self._device)}
        # 构建注意力掩码
        attention_mask = self.build_decode_attention_mask(batch_size, seq_len, req_list)
        if attention_mask is not None:
            res["attention_mask"] = attention_mask
        # 构建位置ID
        position_ids = self.build_decode_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            res["position_ids"] = position_ids
        # 构建token类型ID
        token_type_ids = self.build_decode_token_type_ids(batch_size, seq_len, req_list)
        if token_type_ids is not None:
            res["token_type_ids"] = token_type_ids
        return res

    @staticmethod
    def get_batch_size_and_seq_len_indexes_from_kv() -> Tuple[int, int]:
        """
        从huggingface transformers文档中，`pask_key_values` 的形状为
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`。
        然而，对于某些模型，形状可能会改变。
        """
        return 0, 2

    def get_dtype(self):
        raise NotImplementedError("未实现。")

    @lru_cache
    def get_context_len(self):
        return get_context_length(self._model.config)

    def get_max_num_seqs(self) -> int:
        return self._pytorch_model_config.get("max_num_seqs")  # type: ignore

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        return self._sanitize_generate_config(req.generate_config)

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        # 检查一些参数
        for r in req_list:
            try:
                if r.sanitized_generate_config is None:
                    r.sanitized_generate_config = self.prepare_sanitize_generate_config(
                        r
                    )
                if r.is_prefill:
                    # 检查一些生成参数
                    max_src_len = get_max_src_len(self.get_context_len(), r)  # type: ignore
                    if max_src_len < 0:
                        r.stopped = True
                        r.error_msg = "最大token数超过模型的最大长度"
                        continue
                    if r.stream_interval <= 0:
                        r.stopped = True
                        r.error_msg = "`stream_interval` 必须大于0"
                        continue
                    stop_str = r.sanitized_generate_config.get("stop", None)
                    if stop_str and (
                        not (
                            isinstance(stop_str, str) or isinstance(stop_str, Iterable)
                        )
                    ):
                        r.stopped = True
                        r.error_msg = "无效的 `stop` 字段类型"
                        continue
            # 在这里捕获异常。如果不捕获异常，请求将会挂起。
            except Exception as e:
                logger.exception(f"准备推理时出错：{e}")
                r.stopped = True
                r.error_msg = str(e)

    def get_builtin_stop_token_ids(self) -> Tuple:
        return (
            tuple(self.model_family.prompt_style.stop_token_ids)
            if self.model_family.prompt_style
            and self.model_family.prompt_style.stop_token_ids
            else tuple()
        )

    def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
        for req in req_list:
            if req.error_msg is None:
                # 非流式情况下不需要处理
                if req.stream:
                    results = []
                    for i, c in enumerate(req.completion):
                        if c == "<bos_stream>":
                            chunk = req.completion[i + 1]
                            results.append(
                                CompletionChunk(
                                    id=chunk["id"],
                                    object=chunk["object"],
                                    created=chunk["created"],
                                    model=chunk["model"],
                                    choices=[
                                        CompletionChoice(
                                            text="",
                                            index=0,
                                            logprobs=None,
                                            finish_reason=None,
                                        )
                                    ],
                                )
                            )
                            continue
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(c)

                    if req.stopped and req.include_usage:
                        results.append(req.completion[-1])
                    req.completion = results

    def batch_inference(self, req_list: List[InferenceRequest]):
        from .utils import batch_inference_one_step

        self.prepare_batch_inference(req_list)
        batch_inference_one_step(
            self, req_list, self.model_uid, self._model, self._tokenizer
        )
        self.handle_batch_inference_results(req_list)


class PytorchChatModel(PytorchModel, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            pytorch_model_config,
            peft_model,
        )

    def _sanitize_generate_config(
        self,
        generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
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
        if llm_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        model_family = llm_family.model_family or llm_family.model_name
        if model_family in NON_DEFAULT_MODEL_LIST:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        tools = generate_config.pop("tools", []) if generate_config else None
        full_prompt = self._get_full_prompt(prompt, system_prompt, chat_history, tools)

        generate_config = self._sanitize_generate_config(generate_config)
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

    def load(self):
        super().load()

    def _get_full_prompt(self, prompt, system_prompt, chat_history, tools):
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        full_prompt = ChatModelMixin.get_prompt(
            prompt, chat_history, prompt_style, tools=tools
        )
        return full_prompt

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        super().prepare_batch_inference(req_list)
        for r in req_list:
            try:
                if not r.stopped and r.is_prefill:
                    r.full_prompt = self._get_full_prompt(
                        r.prompt, r.system_prompt, r.chat_history, None
                    )
            except Exception as e:
                logger.exception(f"prepare inference error with {e}")
                r.stopped = True
                r.error_msg = str(e)

    def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
        for req in req_list:
            if req.error_msg is None and req.completion:
                if req.stream:
                    results = []
                    for i, c in enumerate(req.completion):
                        if c == "<bos_stream>":
                            results.append(
                                self._get_first_chat_completion_chunk(
                                    req.completion[i + 1]
                                )
                            )
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(self._to_chat_completion_chunk(c))

                    if req.stopped and req.include_usage:
                        results.append(
                            self._get_final_chat_completion_chunk(req.completion[-1])
                        )
                    req.completion = results
                else:
                    req.completion[0] = self._to_chat_completion(req.completion[0])
