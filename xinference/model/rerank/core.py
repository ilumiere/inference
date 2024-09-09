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

import gc
import logging
import os
import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from ...constants import XINFERENCE_CACHE_DIR
from ...device_utils import empty_cache
from ...types import Document, DocumentObj, Rerank, RerankTokens
from ..core import CacheableModelSpec, ModelDescription
from ..utils import is_model_cached

# 设置日志记录器
logger = logging.getLogger(__name__)

# 用于检查模型是否已缓存
# 在注册所有内置模型时初始化
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
RERANK_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
# 从环境变量获取清空缓存的计数，默认为10
RERANK_EMPTY_CACHE_COUNT = int(os.getenv("XINFERENCE_RERANK_EMPTY_CACHE_COUNT", "10"))
assert RERANK_EMPTY_CACHE_COUNT > 0


def get_rerank_model_descriptions():
    """
    获取重排序模型描述的深拷贝。

    :return: 重排序模型描述的深拷贝
    """
    import copy

    return copy.deepcopy(RERANK_MODEL_DESCRIPTIONS)


class RerankModelSpec(CacheableModelSpec):
    """
    重排序模型规格类，定义了模型的基本属性。
    """
    model_name: str
    language: List[str]
    type: Optional[str] = "unknown"
    model_id: str
    model_revision: Optional[str]
    model_hub: str = "huggingface"


class RerankModelDescription(ModelDescription):
    """
    重排序模型描述类，包含了模型的详细信息。
    """
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: RerankModelSpec,
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        """
        将模型描述转换为字典格式。

        :return: 包含模型信息的字典
        """
        return {
            "model_type": "rerank",
            "address": self.address,
            "accelerators": self.devices,
            "type": self._model_spec.type,
            "model_name": self._model_spec.model_name,
            "language": self._model_spec.language,
            "model_revision": self._model_spec.model_revision,
        }

    def to_version_info(self):
        """
        获取模型的版本信息。

        :return: 包含模型版本信息的字典
        """
        from .utils import get_model_version

        if self._model_path is None:
            is_cached = get_cache_status(self._model_spec)
            file_location = get_cache_dir(self._model_spec)
        else:
            is_cached = True
            file_location = self._model_path

        return {
            "model_version": get_model_version(self._model_spec),
            "model_file_location": file_location,
            "cache_status": is_cached,
            "language": self._model_spec.language,
        }


def generate_rerank_description(model_spec: RerankModelSpec) -> Dict[str, List[Dict]]:
    """
    生成重排序模型的描述。

    :param model_spec: 重排序模型规格
    :return: 包含模型描述的字典
    """
    res = defaultdict(list)
    res[model_spec.model_name].append(
        RerankModelDescription(None, None, model_spec).to_version_info()
    )
    return res


class RerankModel:
    """
    重排序模型类，用于加载和执行重排序任务。
    """

    def __init__(
        self,
        model_spec: RerankModelSpec,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        model_config: Optional[Dict] = None,
    ):
        """
        初始化重排序模型。

        :param model_spec: 模型规格
        :param model_uid: 模型唯一标识符
        :param model_path: 模型路径（可选）
        :param device: 设备（可选）
        :param use_fp16: 是否使用FP16精度（默认为False）
        :param model_config: 模型配置（可选）
        """
        self._model_spec = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model_config = model_config or dict()
        self._use_fp16 = use_fp16
        self._model = None
        self._counter = 0
        if model_spec.type == "unknown":
            model_spec.type = self._auto_detect_type(model_path)

    @staticmethod
    def _get_tokenizer(model_path):
        """
        获取模型的分词器。

        :param model_path: 模型路径
        :return: 分词器对象
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer

    @staticmethod
    def _auto_detect_type(model_path):
        """This method may not be stable due to the fact that the tokenizer name may be changed.
        Therefore, we only use this method for unknown model types."""

        type_mapper = {
            "LlamaTokenizerFast": "LLM-based layerwise",
            "GemmaTokenizerFast": "LLM-based",
            "XLMRobertaTokenizerFast": "normal",
        }

        tokenizer = RerankModel._get_tokenizer(model_path)
        rerank_type = type_mapper.get(type(tokenizer).__name__)
        if rerank_type is None:
            logger.warning(
                f"无法根据分词器 {tokenizer} 确定重排序类型，默认使用 normal 类型。"
            )
            return "normal"
        return rerank_type

    def load(self):
        if self._model_spec.type == "normal":
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError:
                error_message = "无法导入模块 'sentence-transformers'"
                installation_guide = [
                    "请确保已安装 'sentence-transformers'。 ",
                    "您可以通过 `pip install sentence-transformers` 安装它\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = CrossEncoder(
                self._model_path,
                device=self._device,
                trust_remote_code=True,
                **self._model_config,
            )
            if self._use_fp16:
                self._model.model.half()
        else:
            try:
                if self._model_spec.type == "LLM-based":
                    from FlagEmbedding import FlagLLMReranker as FlagReranker
                elif self._model_spec.type == "LLM-based layerwise":
                    from FlagEmbedding import LayerWiseFlagLLMReranker as FlagReranker
                else:
                    raise RuntimeError(
                        f"不支持的重排序模型类型: {self._model_spec.type}"
                    )
            except ImportError:
                error_message = "无法导入模块 'FlagEmbedding'"
                installation_guide = [
                    "请确保已安装 'FlagEmbedding'。 ",
                    "您可以通过 `pip install FlagEmbedding` 安装它\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ) -> Rerank:
        """
        执行重排序操作。

        :param documents: 待重排序的文档列表
        :param query: 查询字符串
        :param top_n: 返回的顶部结果数量（可选）
        :param max_chunks_per_doc: 每个文档的最大块数（可选，暂不支持）
        :param return_documents: 是否返回文档内容
        :param return_len: 是否返回标记长度
        :param kwargs: 额外的关键字参数
        :return: 重排序结果
        """
        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            logger.debug("清空重排序缓存。")
            gc.collect()
            empty_cache()
        assert self._model is not None
        if kwargs:
            raise ValueError("rerank 不支持额外参数。")
        if max_chunks_per_doc is not None:
            raise ValueError("rerank 不支持 `max_chunks_per_doc` 参数。")
        sentence_combinations = [[query, doc] for doc in documents]
        if self._model_spec.type == "normal":
            similarity_scores = self._model.predict(
                sentence_combinations, convert_to_numpy=False, convert_to_tensor=True
            ).cpu()
            if similarity_scores.dtype == torch.bfloat16:
                similarity_scores = similarity_scores.float()
        else:
            # 相关问题: https://github.com/xorbitsai/inference/issues/1775
            similarity_scores = self._model.compute_score(sentence_combinations)
            if not isinstance(similarity_scores, Sequence):
                similarity_scores = [similarity_scores]

        sim_scores_argsort = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            sim_scores_argsort = sim_scores_argsort[:top_n]
        if return_documents:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=Document(text=documents[arg]),
                )
                for arg in sim_scores_argsort
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=None,
                )
                for arg in sim_scores_argsort
            ]
        if return_len:
            tokenizer = self._get_tokenizer(self._model_path)
            input_len = sum([len(tokenizer.tokenize(t)) for t in documents])

            # Rerank Model output is just score or documents
            # while return_documents = True
            output_len = input_len

        # api_version, billed_units, warnings
        # 用于 Cohere API 兼容性，设置为 None
        metadata = {
            "api_version": None,
            "billed_units": None,
            "tokens": (
                RerankTokens(input_tokens=input_len, output_tokens=output_len)
                if return_len
                else None
            ),
            "warnings": None,
        }

        return Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)


def get_cache_dir(model_spec: RerankModelSpec):
    """
    获取模型缓存目录路径。

    :param model_spec: 重排序模型规格
    :return: 缓存目录的实际路径
    """
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def get_cache_status(
    model_spec: RerankModelSpec,
) -> bool:
    """
    检查模型是否已缓存。

    :param model_spec: 重排序模型规格
    :return: 如果模型已缓存则返回True，否则返回False
    """
    return is_model_cached(model_spec, MODEL_NAME_TO_REVISION)


def cache(model_spec: RerankModelSpec):
    """
    缓存重排序模型。

    :param model_spec: 重排序模型规格
    :return: 缓存操作的结果
    """
    from ..utils import cache

    return cache(model_spec, RerankModelDescription)


def create_rerank_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[RerankModel, RerankModelDescription]:
    """
    创建重排序模型实例。

    :param subpool_addr: 子池地址
    :param devices: 设备列表
    :param model_uid: 模型唯一标识符
    :param model_name: 模型名称
    :param download_hub: 下载中心（可选）
    :param model_path: 模型路径（可选）
    :param kwargs: 其他关键字参数
    :return: 重排序模型实例和模型描述的元组
    """
    from ..utils import download_from_modelscope
    from . import BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS
    from .custom import get_user_defined_reranks

    # 查找用户定义的模型规格
    model_spec = None
    for ud_spec in get_user_defined_reranks():
        if ud_spec.model_name == model_name:
            model_spec = ud_spec
            break

    # 如果未找到用户定义的模型，则在内置模型中查找
    if model_spec is None:
        if download_hub == "huggingface" and model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        elif download_hub == "modelscope" and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif download_from_modelscope() and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        else:
            raise ValueError(
                f"Rerank model {model_name} not found, available"
                f"Huggingface: {BUILTIN_RERANK_MODELS.keys()}"
                f"ModelScope: {MODELSCOPE_RERANK_MODELS.keys()}"
            )
    
    # 如果未提供模型路径，则缓存模型
    if not model_path:
        model_path = cache(model_spec)
    
    # 获取是否使用FP16的设置
    use_fp16 = kwargs.pop("use_fp16", False)
    
    # 创建重排序模型实例
    model = RerankModel(
        model_spec, model_uid, model_path, use_fp16=use_fp16, model_config=kwargs
    )
    
    # 创建模型描述实例
    model_description = RerankModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    
    return model, model_description
