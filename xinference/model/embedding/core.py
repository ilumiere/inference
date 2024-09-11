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
import gc
import logging
import os
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union, no_type_check

import numpy as np

# 导入自定义模块
from ...device_utils import empty_cache
from ...types import Embedding, EmbeddingData, EmbeddingUsage
from ..core import CacheableModelSpec, ModelDescription
from ..utils import get_cache_dir, is_model_cached

# 设置日志记录器
logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
EMBEDDING_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)

# 从环境变量获取嵌入模型清空缓存的计数，默认为10
EMBEDDING_EMPTY_CACHE_COUNT = int(
    os.getenv("XINFERENCE_EMBEDDING_EMPTY_CACHE_COUNT", "10")
)
assert EMBEDDING_EMPTY_CACHE_COUNT > 0

# 获取嵌入模型描述的函数
def get_embedding_model_descriptions():
    import copy
    return copy.deepcopy(EMBEDDING_MODEL_DESCRIPTIONS)

# 嵌入模型规格类
class EmbeddingModelSpec(CacheableModelSpec):
    model_name: str
    dimensions: int
    max_tokens: int
    language: List[str]
    model_id: str
    model_revision: Optional[str]
    model_hub: str = "huggingface"

# 嵌入模型描述类
class EmbeddingModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: EmbeddingModelSpec,
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    # 将模型描述转换为字典
    def to_dict(self):
        return {
            "model_type": "embedding",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "dimensions": self._model_spec.dimensions,
            "max_tokens": self._model_spec.max_tokens,
            "language": self._model_spec.language,
            "model_revision": self._model_spec.model_revision,
        }

    # 获取模型版本信息
    def to_version_info(self):
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
            "dimensions": self._model_spec.dimensions,
            "max_tokens": self._model_spec.max_tokens,
        }


def generate_embedding_description(
    model_spec: EmbeddingModelSpec,
) -> Dict[str, List[Dict]]:
    """
    生成嵌入模型的描述信息。

    参数:
    model_spec (EmbeddingModelSpec): 嵌入模型的规格。

    返回:
    Dict[str, List[Dict]]: 包含模型名称和版本信息的字典。
    """
    # 创建一个默认字典，用于存储结果
    res = defaultdict(list)
    
    # 使用模型名称作为键，将版本信息添加到对应的列表中
    res[model_spec.model_name].append(
        EmbeddingModelDescription(None, None, model_spec).to_version_info()
    )
    
    return res


def cache(model_spec: EmbeddingModelSpec):
    """
    缓存嵌入模型。

    参数:
    model_spec (EmbeddingModelSpec): 嵌入模型的规格。

    返回:
    返回缓存操作的结果。

    说明:
    这个函数使用从 ..utils 导入的 cache 函数来缓存指定的嵌入模型。
    它将 model_spec 和 EmbeddingModelDescription 作为参数传递给 cache 函数。
    """
    from ..utils import cache

    return cache(model_spec, EmbeddingModelDescription)


def get_cache_status(
    model_spec: EmbeddingModelSpec,
) -> bool:
    """
    获取嵌入模型的缓存状态。

    参数:
    model_spec (EmbeddingModelSpec): 嵌入模型的规格。

    返回:
    bool: 如果模型已缓存则返回True，否则返回False。

    说明:
    此函数使用is_model_cached函数检查指定的嵌入模型是否已被缓存。
    它使用model_spec和全局变量MODEL_NAME_TO_REVISION作为参数。
    """
    return is_model_cached(model_spec, MODEL_NAME_TO_REVISION)


class EmbeddingModel:
    """
    嵌入模型类，用于加载和使用嵌入模型。
    """

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: EmbeddingModelSpec,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化嵌入模型。

        Args:
            model_uid (str): 模型的唯一标识符。
            model_path (str): 模型文件的路径。
            model_spec (EmbeddingModelSpec): 模型规格。
            device (Optional[str], optional): 运行模型的设备。默认为None。
            **kwargs: 其他关键字参数。
        """
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._counter = 0
        self._model_spec = model_spec
        self._kwargs = kwargs

    def load(self):
        """
        加载嵌入模型。
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            error_message = "Failed to import module 'SentenceTransformer'"
            installation_guide = [
                "Please make sure 'sentence-transformers' is installed. ",
                "You can install it by `pip install sentence-transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        class XSentenceTransformer(SentenceTransformer):
            def to(self, *args, **kwargs):
                pass

        from ..utils import patch_trust_remote_code

        patch_trust_remote_code()
        if (
            "gte" in self._model_spec.model_name.lower()
            and "qwen2" in self._model_spec.model_name.lower()
        ):
            import torch

            torch_dtype_str = self._kwargs.get("torch_dtype")
            if torch_dtype_str is not None:
                try:
                    torch_dtype = getattr(torch, torch_dtype_str)
                    if torch_dtype not in [
                        torch.float16,
                        torch.float32,
                        torch.bfloat16,
                    ]:
                        logger.warning(
                            f"Load embedding model with unsupported torch dtype :  {torch_dtype_str}. Using default torch dtype: fp32."
                        )
                        torch_dtype = torch.float32
                except AttributeError:
                    logger.warning(
                        f"Load embedding model with  unknown torch dtype '{torch_dtype_str}'. Using default torch dtype: fp32."
                    )
                    torch_dtype = torch.float32
            else:
                torch_dtype = "auto"
            self._model = XSentenceTransformer(
                self._model_path,
                device=self._device,
                model_kwargs={"device_map": "auto", "torch_dtype": torch_dtype},
            )
        else:
            self._model = SentenceTransformer(self._model_path, device=self._device)

    def create_embedding(self, sentences: Union[str, List[str]], **kwargs):
        """
        为给定的句子创建嵌入。

        Args:
            sentences (Union[str, List[str]]): 要创建嵌入的句子或句子列表。
            **kwargs: 其他关键字参数。

        Returns:
            Embedding: 包含嵌入数据和使用信息的对象。
        """
        self._counter += 1
        if self._counter % EMBEDDING_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty embedding cache.")
            gc.collect()
            empty_cache()
        from sentence_transformers import SentenceTransformer

        kwargs.setdefault("normalize_embeddings", True)

        # 从sentence-transformers复制并修改以返回token数量
        @no_type_check
        def encode(
            model: SentenceTransformer,
            sentences: Union[str, List[str]],
            prompt_name: Optional[str] = None,
            prompt: Optional[str] = None,
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: str = "sentence_embedding",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
        ):
            """
            计算句子嵌入

            参数:
            :param model: SentenceTransformer模型
            :param sentences: 要嵌入的句子
            :param prompt_name: 提示名称
            :param prompt: 提示文本
            :param batch_size: 用于计算的批次大小
            :param show_progress_bar: 编码句子时是否输出进度条
            :param output_value: 默认为"sentence_embedding"以获取句子嵌入。可以设置为"token_embeddings"以获取词片段token嵌入。设置为None以获取所有输出值
            :param convert_to_numpy: 如果为True，输出为numpy向量列表。否则为PyTorch张量列表
            :param convert_to_tensor: 如果为True，返回一个大的张量。覆盖convert_to_numpy的设置
            :param device: 用于计算的torch.device
            :param normalize_embeddings: 如果设置为True，返回的向量将具有长度1。在这种情况下，可以使用更快的点积(util.dot_score)而不是余弦相似度

            返回:
               默认返回张量列表。如果convert_to_tensor为True，返回堆叠的张量。如果convert_to_numpy为True，返回numpy矩阵。
            """
            import torch
            from sentence_transformers.util import batch_to_device
            from tqdm.autonotebook import trange

            # 设置模型为评估模式
            model.eval()
            
            # 根据日志级别决定是否显示进度条
            if show_progress_bar is None:
                show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO
                    or logger.getEffectiveLevel() == logging.DEBUG
                )

            # 如果需要转换为张量，则不转换为numpy
            if convert_to_tensor:
                convert_to_numpy = False

            # 如果输出值不是句子嵌入，则不进行转换
            if output_value != "sentence_embedding":
                convert_to_tensor = False
                convert_to_numpy = False

            # 处理单个句子输入的情况
            input_was_string = False
            if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
            ):  # 将单个句子转换为长度为1的列表
                sentences = [sentences]
                input_was_string = True

            # 处理提示（prompt）
            if prompt is None:
                if prompt_name is not None:
                    try:
                        prompt = model.prompts[prompt_name]
                    except KeyError:
                        raise ValueError(
                            f"提示名称'{prompt_name}'在配置的提示字典中未找到，可用的键为{list(model.prompts.keys())!r}。"
                        )
                elif model.default_prompt_name is not None:
                    prompt = model.prompts.get(model.default_prompt_name, None)
            else:
                if prompt_name is not None:
                    logger.warning(
                        "编码时使用'prompt'、'prompt_name'或两者都不使用，但不能同时使用两者。"
                        "忽略'prompt_name'，使用'prompt'。"
                    )

            # 处理额外特征
            extra_features = {}
            if prompt is not None:
                sentences = [prompt + sentence for sentence in sentences]

                # 一些模型（如INSTRUCTOR、GRIT）需要在池化之前移除提示
                # 跟踪提示长度允许我们在池化期间移除提示
                tokenized_prompt = model.tokenize([prompt])
                if "input_ids" in tokenized_prompt:
                    extra_features["prompt_length"] = (
                        tokenized_prompt["input_ids"].shape[-1] - 1
                    )

            # 设置设备
            if device is None:
                device = model._target_device

            # 对于特定模型，将模型移动到指定设备
            if (
                "gte" in self._model_spec.model_name.lower()
                and "qwen2" in self._model_spec.model_name.lower()
            ):
                model.to(device)

            all_embeddings = []
            all_token_nums = 0
            # 按句子长度排序以提高效率
            length_sorted_idx = np.argsort(
                [-model._text_length(sen) for sen in sentences]
            )
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            # 批处理编码
            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[
                    start_index : start_index + batch_size
                ]
                features = model.tokenize(sentences_batch)
                features = batch_to_device(features, device)
                features.update(extra_features)
                all_token_nums += sum([len(f) for f in features])

                with torch.no_grad():
                    out_features = model.forward(features)

                    # 处理不同的输出值类型
                    if output_value == "token_embeddings":
                        embeddings = []
                        for token_emb, attention in zip(
                            out_features[output_value], out_features["attention_mask"]
                        ):
                            last_mask_id = len(attention) - 1
                            while (
                                last_mask_id > 0 and attention[last_mask_id].item() == 0
                            ):
                                last_mask_id -= 1

                            embeddings.append(token_emb[0 : last_mask_id + 1])
                    elif output_value is None:  # 返回所有输出
                        embeddings = []
                        for sent_idx in range(len(out_features["sentence_embedding"])):
                            row = {
                                name: out_features[name][sent_idx]
                                for name in out_features
                            }
                            embeddings.append(row)
                    else:  # 句子嵌入
                        embeddings = out_features[output_value]
                        embeddings = embeddings.detach()
                        if normalize_embeddings:
                            embeddings = torch.nn.functional.normalize(
                                embeddings, p=2, dim=1
                            )

                        # 修复#522和#487以避免大数据集在GPU上的内存溢出问题
                        if convert_to_numpy:
                            embeddings = embeddings.cpu()

                    all_embeddings.extend(embeddings)

            # 恢复原始顺序
            all_embeddings = [
                all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
            ]

            # 根据需要转换输出格式
            if convert_to_tensor:
                if len(all_embeddings):
                    all_embeddings = torch.stack(all_embeddings)
                else:
                    all_embeddings = torch.Tensor()
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

            # 如果输入是单个字符串，返回单个嵌入
            if input_was_string:
                all_embeddings = all_embeddings[0]

            return all_embeddings, all_token_nums

        # 根据模型类型选择不同的编码方式
        if (
            "gte" in self._model_spec.model_name.lower()
            and "qwen2" in self._model_spec.model_name.lower()
        ):
            all_embeddings, all_token_nums = encode(
                self._model,
                sentences,
                prompt_name="query",
                convert_to_numpy=False,
                **kwargs,
            )
        else:
            all_embeddings, all_token_nums = encode(
                self._model,
                sentences,
                convert_to_numpy=False,
                **kwargs,
            )
        
        # 处理单个句子输入的情况
        if isinstance(sentences, str):
            all_embeddings = [all_embeddings]
        
        # 构建嵌入数据列表
        embedding_list = []
        for index, data in enumerate(all_embeddings):
            embedding_list.append(
                EmbeddingData(index=index, object="embedding", embedding=data.tolist())
            )
        
        # 创建使用情况对象
        usage = EmbeddingUsage(
            prompt_tokens=all_token_nums, total_tokens=all_token_nums
        )
        
        # 返回最终的嵌入对象
        return Embedding(
            object="list",
            model=self._model_uid,
            data=embedding_list,
            usage=usage,
        )

def match_embedding(
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
) -> EmbeddingModelSpec:
    """
    匹配给定的嵌入模型名称，并返回相应的EmbeddingModelSpec。

    参数:
    model_name (str): 要匹配的嵌入模型名称。
    download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 指定下载模型的平台，默认为None。

    返回:
    EmbeddingModelSpec: 匹配到的嵌入模型规格。

    异常:
    ValueError: 如果未找到匹配的嵌入模型。
    """
    from ..utils import download_from_modelscope
    from . import BUILTIN_EMBEDDING_MODELS, MODELSCOPE_EMBEDDING_MODELS
    from .custom import get_user_defined_embeddings

    # 首先检查是否为用户自定义的嵌入模型
    for model_spec in get_user_defined_embeddings():
        if model_name == model_spec.model_name:
            return model_spec

    # 根据指定的下载平台和模型名称进行匹配
    if download_hub == "modelscope" and model_name in MODELSCOPE_EMBEDDING_MODELS:
        logger.debug(f"在ModelScope中找到嵌入模型 {model_name}。")
        return MODELSCOPE_EMBEDDING_MODELS[model_name]
    elif download_hub == "huggingface" and model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"在Huggingface中找到嵌入模型 {model_name}。")
        return BUILTIN_EMBEDDING_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_EMBEDDING_MODELS:
        logger.debug(f"在ModelScope中找到嵌入模型 {model_name}。")
        return MODELSCOPE_EMBEDDING_MODELS[model_name]
    elif model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"在Huggingface中找到嵌入模型 {model_name}。")
        return BUILTIN_EMBEDDING_MODELS[model_name]
    else:
        # 如果未找到匹配的模型，抛出ValueError异常
        raise ValueError(
            f"未找到嵌入模型 {model_name}，可用的模型有："
            f"Huggingface: {BUILTIN_EMBEDDING_MODELS.keys()}"
            f"ModelScope: {MODELSCOPE_EMBEDDING_MODELS.keys()}"
        )


def create_embedding_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[EmbeddingModel, EmbeddingModelDescription]:
    """
    创建嵌入模型实例。

    参数:
    subpool_addr (str): 子池地址。
    devices (List[str]): 设备列表。
    model_uid (str): 模型唯一标识符。
    model_name (str): 模型名称。
    download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 下载中心，默认为None。
    model_path (Optional[str]): 模型路径，默认为None。
    **kwargs: 其他关键字参数。

    返回:
    Tuple[EmbeddingModel, EmbeddingModelDescription]: 返回嵌入模型实例和模型描述的元组。
    """
    # 匹配嵌入模型规格
    model_spec = match_embedding(model_name, download_hub)
    
    # 如果未提供模型路径，则缓存模型
    if model_path is None:
        model_path = cache(model_spec)

    # 创建嵌入模型实例
    model = EmbeddingModel(model_uid, model_path, model_spec, **kwargs)
    
    # 创建嵌入模型描述
    model_description = EmbeddingModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    
    return model, model_description
