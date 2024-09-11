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
import os
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from typing_extensions import Annotated, Literal

from ..._compat import (
    ROOT_KEY,
    BaseModel,
    ErrorWrapper,
    Field,
    Protocol,
    StrBytes,
    ValidationError,
    load_str_bytes,
    validator,
)
from ...constants import (
    XINFERENCE_CACHE_DIR,
    XINFERENCE_CSG_ENDPOINT,
    XINFERENCE_ENV_CSG_TOKEN,
    XINFERENCE_MODEL_DIR,
)
from ..utils import (
    IS_NEW_HUGGINGFACE_HUB,
    create_symlink,
    download_from_csghub,
    download_from_modelscope,
    is_valid_model_uri,
    parse_uri,
    retry_download,
    symlink_local_file,
    valid_model_revision,
)
from . import LLM

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 设置默认上下文长度
DEFAULT_CONTEXT_LENGTH = 2048
# 初始化内置LLM提示风格字典
BUILTIN_LLM_PROMPT_STYLE: Dict[str, "PromptStyleV1"] = {}
# 初始化内置LLM模型聊天家族集合
BUILTIN_LLM_MODEL_CHAT_FAMILIES: Set[str] = set()
# 初始化内置LLM模型生成家族集合
BUILTIN_LLM_MODEL_GENERATE_FAMILIES: Set[str] = set()
# 初始化内置LLM模型工具调用家族集合
BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES: Set[str] = set()


class LlamaCppLLMSpecV1(BaseModel):
    """LlamaCpp LLM规格V1类"""
    model_format: Literal["ggufv2"]
    # 必须按顺序：先`str`，然后`int`
    model_size_in_billions: Union[str, int]
    quantizations: List[str]
    model_id: Optional[str]
    model_file_name_template: str
    model_file_name_split_template: Optional[str]
    quantization_parts: Optional[Dict[str, List[str]]]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]

    @validator("model_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        """验证模型大小，处理带下划线的字符串"""
        if isinstance(v, str):
            if "_" in v:  # 例如，"1_8"直接返回"1_8"，否则int("1_8")会返回18
                return v
            else:
                return int(v)
        return v


class PytorchLLMSpecV1(BaseModel):
    """Pytorch LLM规格V1类"""
    model_format: Literal["pytorch", "gptq", "awq", "fp8"]
    # 必须按顺序：先`str`，然后`int`
    model_size_in_billions: Union[str, int]
    quantizations: List[str]
    model_id: Optional[str]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]

    @validator("model_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        """验证模型大小，处理带下划线的字符串"""
        if isinstance(v, str):
            if "_" in v:  # 例如，"1_8"直接返回"1_8"，否则int("1_8")会返回18
                return v
            else:
                return int(v)
        return v


class MLXLLMSpecV1(BaseModel):
    """MLX LLM规格V1类"""
    model_format: Literal["mlx"]
    # 必须按顺序：先`str`，然后`int`
    model_size_in_billions: Union[str, int]
    quantizations: List[str]
    model_id: Optional[str]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]

    @validator("model_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        """验证模型大小，处理带下划线的字符串"""
        if isinstance(v, str):
            if "_" in v:  # 例如，"1_8"直接返回"1_8"，否则int("1_8")会返回18
                return v
            else:
                return int(v)
        return v


class PromptStyleV1(BaseModel):
    """提示风格V1类"""
    style_name: str
    system_prompt: str = ""
    roles: List[str]
    intra_message_sep: str = ""
    inter_message_sep: str = ""
    stop: Optional[List[str]]
    stop_token_ids: Optional[List[int]]


class LLMFamilyV1(BaseModel):
    """LLM家族V1类"""
    version: Literal[1]
    context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH
    model_name: str
    model_lang: List[str]
    model_ability: List[Literal["embed", "generate", "chat", "tools", "vision"]]
    model_description: Optional[str]
    # reason for not required str here: legacy registration
    model_family: Optional[str]
    model_specs: List["LLMSpecV1"]
    prompt_style: Optional["PromptStyleV1"]


class CustomLLMFamilyV1(LLMFamilyV1):
    # 允许prompt_style为PromptStyleV1对象或字符串
    prompt_style: Optional[Union["PromptStyleV1", str]]  # type: ignore

    @classmethod
    def parse_raw(
        cls: Any,
        b: StrBytes,
        *,
        content_type: Optional[str] = None,
        encoding: str = "utf8",
        proto: Protocol = None,
        allow_pickle: bool = False,
    ) -> LLMFamilyV1:
        # See source code of BaseModel.parse_raw
        try:
            obj = load_str_bytes(
                b,
                proto=proto,
                content_type=content_type,
                encoding=encoding,
                allow_pickle=allow_pickle,
                json_loads=cls.__config__.json_loads,
            )
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            raise ValidationError([ErrorWrapper(e, loc=ROOT_KEY)], cls)
        llm_spec: CustomLLMFamilyV1 = cls.parse_obj(obj)

        # 检查model_family是否已指定
        if llm_spec.model_family is None:
            raise ValueError(
                f"You must specify `model_family` when registering custom LLM models."
            )
        assert isinstance(llm_spec.model_family, str)
        
        # 检查聊天模型的model_family是否有效
        if (
            llm_spec.model_family != "other"
            and "chat" in llm_spec.model_ability
            and llm_spec.model_family not in BUILTIN_LLM_MODEL_CHAT_FAMILIES
        ):
            raise ValueError(
                f"`model_family` for chat model must be `other` or one of the following values: \n"
                f"{', '.join(list(BUILTIN_LLM_MODEL_CHAT_FAMILIES))}"
            )
        
        # 检查工具调用模型的model_family是否有效
        if (
            llm_spec.model_family != "other"
            and "tools" in llm_spec.model_ability
            and llm_spec.model_family not in BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES
        ):
            raise ValueError(
                f"`model_family` for tool call model must be `other` or one of the following values: \n"
                f"{', '.join(list(BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES))}"
            )
        
        # 检查生成模型的model_family是否有效
        if (
            llm_spec.model_family != "other"
            and "chat" not in llm_spec.model_ability
            and llm_spec.model_family not in BUILTIN_LLM_MODEL_GENERATE_FAMILIES
        ):
            raise ValueError(
                f"`model_family` for generate model must be `other` or one of the following values: \n"
                f"{', '.join(list(BUILTIN_LLM_MODEL_GENERATE_FAMILIES))}"
            )
        # set prompt style when it is the builtin model family
        if (
            llm_spec.prompt_style is None
            and llm_spec.model_family != "other"
            and "chat" in llm_spec.model_ability
        ):
            llm_spec.prompt_style = llm_spec.model_family

        # handle prompt style when user choose existing style
        if llm_spec.prompt_style is not None and isinstance(llm_spec.prompt_style, str):
            prompt_style_name = llm_spec.prompt_style
            if prompt_style_name not in BUILTIN_LLM_PROMPT_STYLE:
                raise ValueError(
                    f"Xinference does not support the prompt style name: {prompt_style_name}"
                )
            llm_spec.prompt_style = BUILTIN_LLM_PROMPT_STYLE[prompt_style_name]

        # check model ability, registering LLM only provides generate and chat
        # but for vision models, we add back the abilities so that
        # gradio chat interface can be generated properly
        if (
            llm_spec.model_family != "other"
            and llm_spec.model_family
            in {
                family.model_name
                for family in BUILTIN_LLM_FAMILIES
                if "vision" in family.model_ability
            }
            and "vision" not in llm_spec.model_ability
        ):
            llm_spec.model_ability.append("vision")

        return llm_spec

# 定义LLMSpecV1类型，包含LlamaCppLLMSpecV1、PytorchLLMSpecV1和MLXLLMSpecV1
LLMSpecV1 = Annotated[
    Union[LlamaCppLLMSpecV1, PytorchLLMSpecV1, MLXLLMSpecV1],
    Field(discriminator="model_format"),
]

# 更新LLMFamilyV1和CustomLLMFamilyV1的前向引用
LLMFamilyV1.update_forward_refs()
CustomLLMFamilyV1.update_forward_refs()

# 定义各种LLM相关的类和家族列表
LLAMA_CLASSES: List[Type[LLM]] = []  # LLAMA类列表
BUILTIN_LLM_FAMILIES: List["LLMFamilyV1"] = []  # 内置LLM家族列表
BUILTIN_MODELSCOPE_LLM_FAMILIES: List["LLMFamilyV1"] = []  # 内置ModelScope LLM家族列表
BUILTIN_CSGHUB_LLM_FAMILIES: List["LLMFamilyV1"] = []  # 内置CSGHub LLM家族列表
SGLANG_CLASSES: List[Type[LLM]] = []  # SGLang类列表
TRANSFORMERS_CLASSES: List[Type[LLM]] = []  # Transformers类列表
UD_LLM_FAMILIES: List["LLMFamilyV1"] = []  # 用户定义的LLM家族列表
UD_LLM_FAMILIES_LOCK = Lock()  # 用户定义LLM家族的锁
VLLM_CLASSES: List[Type[LLM]] = []  # VLLM类列表
MLX_CLASSES: List[Type[LLM]] = []  # MLX类列表
LMDEPLOY_CLASSES: List[Type[LLM]] = []  # LMDeploy类列表

# 定义LLM引擎和支持的引擎字典
LLM_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[LLM]]] = {}

# 定义LLM启动版本字典
LLM_LAUNCH_VERSIONS: Dict[str, List[str]] = {}

def download_from_self_hosted_storage() -> bool:
    """
    检查是否从自托管存储下载模型
    
    返回:
    bool: 如果XINFERENCE_ENV_MODEL_SRC环境变量设置为"xorbits"，则返回True，否则返回False
    """
    from ...constants import XINFERENCE_ENV_MODEL_SRC
    return os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "xorbits"


def get_legacy_cache_path(
    model_name: str,
    model_format: str,
    model_size_in_billions: Optional[Union[str, int]] = None,
    quantization: Optional[str] = None,
) -> str:
    """
    获取旧版缓存路径。

    参数:
    model_name (str): 模型名称
    model_format (str): 模型格式
    model_size_in_billions (Optional[Union[str, int]]): 模型大小（以十亿参数计）
    quantization (Optional[str]): 量化方法

    返回:
    str: 旧版缓存路径
    """
    # 构建完整的模型名称，包括模型名、格式、大小和量化方法
    full_name = f"{model_name}-{model_format}-{model_size_in_billions}b-{quantization}"
    
    # 使用os.path.join将XINFERENCE_CACHE_DIR、完整模型名和"model.bin"组合成完整的路径
    return os.path.join(XINFERENCE_CACHE_DIR, full_name, "model.bin")


def cache(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    缓存LLM模型并返回缓存路径。

    参数:
    llm_family: LLM家族信息
    llm_spec: LLM规格信息
    quantization: 量化方法（可选）

    返回:
    str: 缓存路径
    """
    # 获取旧版缓存路径
    legacy_cache_path = get_legacy_cache_path(
        llm_family.model_name,
        llm_spec.model_format,
        llm_spec.model_size_in_billions,
        quantization,
    )
    
    # 检查旧版缓存是否存在
    if os.path.exists(legacy_cache_path):
        logger.info("旧版缓存路径存在: %s", legacy_cache_path)
        return os.path.dirname(legacy_cache_path)
    else:
        # 如果指定了模型URI，从URI缓存
        if llm_spec.model_uri is not None:
            logger.info(f"从URI缓存: {llm_spec.model_uri}")
            return cache_from_uri(llm_family, llm_spec)
        else:
            # 根据不同的模型中心选择缓存方法
            if llm_spec.model_hub == "huggingface":
                logger.info(f"从Hugging Face缓存: {llm_spec.model_id}")
                return cache_from_huggingface(llm_family, llm_spec, quantization)
            elif llm_spec.model_hub == "modelscope":
                logger.info(f"从Modelscope缓存: {llm_spec.model_id}")
                return cache_from_modelscope(llm_family, llm_spec, quantization)
            elif llm_spec.model_hub == "csghub":
                logger.info(f"从CSGHub缓存: {llm_spec.model_id}")
                return cache_from_csghub(llm_family, llm_spec, quantization)
            else:
                # 如果是未知的模型中心，抛出异常
                raise ValueError(f"未知的模型中心: {llm_spec.model_hub}")

def cache_from_uri(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
) -> str:
    # 构建缓存目录名称
    cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b"
    )
    # 获取完整的缓存目录路径
    cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name))

    # 确保model_uri不为空
    assert llm_spec.model_uri is not None
    # 解析URI的scheme和root
    src_scheme, src_root = parse_uri(llm_spec.model_uri)
    if src_root.endswith("/"):
        # 移除尾部的路径分隔符
        src_root = src_root[:-1]

    if src_scheme == "file":
        # 如果是文件URI
        if not os.path.isabs(src_root):
            # 如果不是绝对路径，抛出异常
            raise ValueError(
                f"Model URI cannot be a relative path: {llm_spec.model_uri}"
            )
        # 创建缓存目录
        os.makedirs(XINFERENCE_CACHE_DIR, exist_ok=True)
        if os.path.exists(cache_dir):
            # 如果缓存目录已存在，直接返回
            logger.info(f"Cache {cache_dir} exists")
            return cache_dir
        else:
            # 创建符号链接
            os.symlink(src_root, cache_dir, target_is_directory=True)
        return cache_dir
    else:
        # 如果是不支持的URI scheme，抛出异常
        raise ValueError(f"Unsupported URL scheme: {src_scheme}")


def cache_model_config(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
):
    """
    下载模型的config.json文件到缓存目录，并返回本地文件路径
    
    参数:
    llm_family: LLM家族对象
    llm_spec: LLM规格对象
    
    返回:
    str: config.json文件的本地路径
    """
    # 获取缓存目录
    cache_dir = _get_cache_dir_for_model_mem(llm_family, llm_spec)
    # 构建config.json文件的完整路径
    config_file = os.path.join(cache_dir, "config.json")
    
    # 如果config.json文件不是符号链接且不存在，则下载
    if not os.path.islink(config_file) and not os.path.exists(config_file):
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 根据模型中心选择不同的下载方式
        if llm_spec.model_hub == "huggingface":
            # 从Hugging Face下载
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=llm_spec.model_id, filename="config.json", local_dir=cache_dir
            )
        else:
            # 从ModelScope下载
            from modelscope.hub.file_download import model_file_download

            download_path = model_file_download(
                model_id=llm_spec.model_id, file_path="config.json"
            )
            # 创建符号链接
            os.symlink(download_path, config_file)
    
    # 返回config.json文件的本地路径
    return config_file


def _get_cache_dir_for_model_mem(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    create_if_not_exist=True,
):
    """
    仅用于计算模型内存。（可能由supervisor / cli调用）
    由于符号链接样式不同的问题，暂时使用与worker的cache_dir分开的目录。

    参数:
    llm_family: LLM家族对象
    llm_spec: LLM规格对象
    create_if_not_exist: 如果目录不存在是否创建，默认为True

    返回:
    str: 缓存目录的路径
    """
    # 初始化量化后缀
    quant_suffix = ""
    # 遍历所有量化选项
    for q in llm_spec.quantizations:
        # 如果模型ID存在且包含当前量化选项
        if llm_spec.model_id and q in llm_spec.model_id:
            quant_suffix = q
            break
    
    # 构建缓存目录名
    cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b"
    )
    # 如果有量化后缀，添加到目录名中
    if quant_suffix:
        cache_dir_name += f"-{quant_suffix}"
    
    # 获取完整的缓存目录路径
    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, "model_mem", cache_dir_name)
    )
    
    # 如果需要创建目录且目录不存在，则创建目录
    if create_if_not_exist and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    return cache_dir



def _get_cache_dir(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
    create_if_not_exist=True,
):
    """
    llm_family = LLMFamilyV1(model_name="qwen1.5-chat")
    llm_spec = LLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=7,
        quantizations=["4bit", "8bit"],
        model_id="Qwen/Qwen1.5-7B-Chat"
    )
    quantization = "4bit"
    XINFERENCE_CACHE_DIR = "/home/user/.xinference/cache"

    调用函数
    cache_dir = _get_cache_dir(llm_family, llm_spec, quantization)
    可能的输出：
    1. 如果旧目录存在：
    2. 如果旧目录不存在，创建新目录：
    注意在第二种情况下，模型名称中的"."被替换为""。
    这个函数的主要用途是确保每个模型版本都有一个唯一的缓存目录
    ，同时处理可能导致导入问题的特殊字符。它在模型下载、加载和管理过程中起着重要作用，
    确保不同版本和量化的模型能够正确地被存储和访问。


    Args:
        llm_family (LLMFamilyV1): _description_
        llm_spec (LLMSpecV1): _description_
        quantization (Optional[str], optional): _description_. Defaults to None.
        create_if_not_exist (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # 如果模型ID包含量化信息，我们应该为每种量化方法提供专用的缓存目录
    quant_suffix = ""
    # model_id = "Qwen/Qwen1.5-7B-Chat-{quantization}"
    # 于4位量化版本，最终的model_id可能是：
    # "Qwen/Qwen1.5-7B-Chat-4bit"
    # 对于8位量化版本，最终的model_id可能是：
    # "Qwen/Qwen1.5-7B-Chat-8bit"
    # 因此，我们需要检查model_id是否包含量化信息，并根据需要设置quant_suffix
    if llm_spec.model_id and "{" in llm_spec.model_id and quantization is not None:
        quant_suffix = quantization
    else:
        # 遍历所有量化选项，查找匹配的量化后缀
        # 例如：
        # llm_spec.quantizations = ["4bit", "8bit"]
        # llm_spec.model_id = "Qwen/Qwen1.5-7B-Chat-4bit"
        # 在这种情况下，quant_suffix将被设置为"4bit"

        
        # quant_suffix = ""
        # if llm_spec.model_id:
        #     for q in llm_spec.quantizations:
        #         if q in llm_spec.model_id:
        #     quant_suffix = q
        #     break
        for q in llm_spec.quantizations:
            if llm_spec.model_id and q in llm_spec.model_id:
                quant_suffix = q
                break

    # 一些模型名称包含"."，例如qwen1.5-chat
    # 如果模型不需要trust_remote_code，这没问题
    # 因为不需要从路径导入modeling_xxx.py
    # 但当模型需要trust_remote_code时，
    # 例如internlm2.5-chat，导入将失败，
    # 但在此之前模型可能已经被下载，
    # 因此我们首先检查它，如果存在，则返回它，
    # 否则，我们将模型名称中的"."替换为"_"
    old_cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b"
    )
    if quant_suffix:
        old_cache_dir_name += f"-{quant_suffix}"
    # 获取旧版缓存目录的完整路径
    old_cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, old_cache_dir_name)
    )
    if os.path.exists(old_cache_dir):
        return old_cache_dir
    else:
        # 创建新的缓存目录名，将"."替换为"_"
        # 字符串字面量的隐式连接：
        # 在Python中，当两个或多个字符串字面量（包括原始字符串）相邻时，
        # 它们会自动连接成一个字符串。这种连接发生在词法分析阶段，不需要任何额外的语法。
        #    cache_dir_name = f"{llm_family.model_name.replace('.', '_')}-{llm_spec.model_format}" + f"-{llm_spec.model_size_in_billions}b"
        cache_dir_name = (
            f"{llm_family.model_name.replace('.', '_')}-{llm_spec.model_format}"
            f"-{llm_spec.model_size_in_billions}b"
        )
        if quant_suffix:
            cache_dir_name += f"-{quant_suffix}"
        cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name))
        # 如果需要创建目录且目录不存在，则创建目录
        if create_if_not_exist and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir


def _get_meta_path(
    cache_dir: str,
    model_format: str,
    model_hub: str,
    quantization: Optional[str] = None,
):
    """
    获取元数据文件的路径。

    参数:
    cache_dir (str): 缓存目录
    model_format (str): 模型格式
    model_hub (str): 模型仓库
    quantization (Optional[str]): 量化方法（可选）

    返回:
    str: 元数据文件的路径
    """
    if model_format == "pytorch":
        # 对于PyTorch格式的模型
        if model_hub == "huggingface":
            # Hugging Face仓库使用默认的验证文件名
            return os.path.join(cache_dir, "__valid_download")
        else:
            # 其他仓库在验证文件名中包含仓库名
            return os.path.join(cache_dir, f"__valid_download_{model_hub}")
    elif model_format in ["ggufv2", "gptq", "awq", "fp8", "mlx"]:
        assert quantization is not None
        if model_hub == "huggingface":
            # Hugging Face仓库的验证文件名包含量化方法
            return os.path.join(cache_dir, f"__valid_download_{quantization}")
        else:
            # 其他仓库的验证文件名包含仓库名和量化方法
            return os.path.join(
                cache_dir, f"__valid_download_{model_hub}_{quantization}"
            )
    else:
        # 不支持的模型格式
        raise ValueError(f"不支持的模型格式: {model_format}")


def _skip_download(
    cache_dir: str,
    model_format: str,
    model_hub: str,
    model_revision: Optional[str],
    quantization: Optional[str] = None,
) -> bool:
    """
    判断是否跳过下载模型。

    参数:
    cache_dir: 缓存目录
    model_format: 模型格式
    model_hub: 模型仓库
    model_revision: 模型版本
    quantization: 量化方法（可选）

    返回:
    bool: 是否跳过下载
    """
    if model_format == "pytorch":
        # 为不同的模型仓库创建元数据路径字典
        model_hub_to_meta_path = {
            "huggingface": _get_meta_path(
                cache_dir, model_format, "huggingface", quantization
            ),
            "modelscope": _get_meta_path(
                cache_dir, model_format, "modelscope", quantization
            ),
            "csghub": _get_meta_path(cache_dir, model_format, "csghub", quantization),
        }
        # 检查指定仓库的模型版本是否有效
        if valid_model_revision(model_hub_to_meta_path[model_hub], model_revision):
            logger.info(f"缓存 {cache_dir} 存在")
            return True
        else:
            # 检查其他仓库的缓存是否存在
            for hub, meta_path in model_hub_to_meta_path.items():
                if hub != model_hub and os.path.exists(meta_path):
                    # PyTorch模型也可以从modelscope加载
                    logger.warning(f"缓存 {cache_dir} 存在，但来自 {hub}")
                    return True
            return False
    elif model_format in ["ggufv2", "gptq", "awq", "fp8", "mlx"]:
        # 对于特定格式，检查量化方法是否存在
        assert quantization is not None
        return os.path.exists(
            _get_meta_path(cache_dir, model_format, model_hub, quantization)
        )
    else:
        # 不支持的格式抛出异常
        raise ValueError(f"不支持的格式: {model_format}")

def _generate_meta_file(
    meta_path: str,
    llm_family: "LLMFamilyV1",
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
):
    # 确保元数据文件不是有效的模型版本
    assert not valid_model_revision(
        meta_path, llm_spec.model_revision
    ), f"meta file {meta_path} should not be valid"
    
    # 打开元数据文件进行写入
    with open(meta_path, "w") as f:
        import json
        
        from .core import LLMDescription

        # 创建LLM描述对象
        desc = LLMDescription(None, None, llm_family, llm_spec, quantization)
        
        # 将LLM描述对象转换为字典并写入JSON文件
        json.dump(desc.to_dict(), f)


def _generate_model_file_names(
    llm_spec: "LLMSpecV1", quantization: Optional[str] = None
) -> Tuple[List[str], str, bool]:
    """
    生成模型文件名列表、最终文件名和是否需要合并的标志。

    参数:
    llm_spec: LLM规格对象
    quantization: 量化方法（可选）

    返回:
    Tuple[List[str], str, bool]: 文件名列表、最终文件名和是否需要合并的标志
    """
    file_names = []
    # 使用模板生成最终文件名
    final_file_name = llm_spec.model_file_name_template.format(
        quantization=quantization
    )
    need_merge = False

    if llm_spec.quantization_parts is None:
        # 如果没有量化部分，直接使用最终文件名
        file_names.append(final_file_name)
    elif quantization is not None and quantization in llm_spec.quantization_parts:
        # 如果有量化部分且指定的量化方法存在
        parts = llm_spec.quantization_parts[quantization]
        need_merge = True

        logger.info(
            f"模型 {llm_spec.model_id} {llm_spec.model_format} {quantization} 有 {len(parts)} 个部分。"
        )

        if llm_spec.model_file_name_split_template is None:
            raise ValueError(
                f"No model_file_name_split_template for model spec {llm_spec.model_id}"
            )

        # 为每个部分生成文件名
        for part in parts:
            file_name = llm_spec.model_file_name_split_template.format(
                quantization=quantization, part=part
            )
            file_names.append(file_name)

    return file_names, final_file_name, need_merge


def _merge_cached_files(
    cache_dir: str, input_file_names: List[str], output_file_name: str
):
    """
    合并缓存的文件。
    
    由于llama.cpp现在可以自动找到gguf的各个部分，
    我们只需要提供第一个部分即可。
    因此，我们创建一个指向第一个部分的符号链接。

    参数:
    cache_dir (str): 缓存目录的路径
    input_file_names (List[str]): 输入文件名列表
    output_file_name (str): 输出文件名
    """
    # 创建一个符号链接，将输出文件名链接到第一个输入文件
    symlink_local_file(
        os.path.join(cache_dir, input_file_names[0]), cache_dir, output_file_name
    )

    # 记录合并完成的日志
    logger.info("合并完成。")


def cache_from_csghub(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    从CSGHub缓存模型。返回缓存目录。

    参数:
    llm_family: LLM家族对象
    llm_spec: LLM规格对象
    quantization: 量化方法（可选）

    返回:
    str: 缓存目录路径
    """
    from pycsghub.file_download import file_download
    from pycsghub.snapshot_download import snapshot_download

    # 获取缓存目录
    cache_dir = _get_cache_dir(llm_family, llm_spec)

    # 检查是否需要跳过下载
    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    # 处理不同的模型格式
    if llm_spec.model_format in ["pytorch", "gptq", "awq", "fp8", "mlx"]:
        # 下载模型快照
        download_dir = retry_download(
            snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            endpoint=XINFERENCE_CSG_ENDPOINT,
            token=os.environ.get(XINFERENCE_ENV_CSG_TOKEN),
        )
        # 创建符号链接
        create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggufv2"]:
        # 生成模型文件名
        file_names, final_file_name, need_merge = _generate_model_file_names(
            llm_spec, quantization
        )

        # 下载每个文件
        for filename in file_names:
            download_path = retry_download(
                file_download,
                llm_family.model_name,
                {
                    "model_size": llm_spec.model_size_in_billions,
                    "model_format": llm_spec.model_format,
                },
                llm_spec.model_id,
                file_name=filename,
                endpoint=XINFERENCE_CSG_ENDPOINT,
                token=os.environ.get(XINFERENCE_ENV_CSG_TOKEN),
            )
            # 创建符号链接
            symlink_local_file(download_path, cache_dir, filename)

        # 如果需要合并文件
        if need_merge:
            _merge_cached_files(cache_dir, file_names, final_file_name)
    else:
        # 如果是不支持的格式，抛出异常
        raise ValueError(f"不支持的格式: {llm_spec.model_format}")

    # 生成元数据文件
    meta_path = _get_meta_path(
        cache_dir, llm_spec.model_format, llm_spec.model_hub, quantization
    )
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir


def cache_from_modelscope(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    从ModelScope缓存模型。返回缓存目录。

    参数:
    llm_family: LLM家族对象
    llm_spec: LLM规格对象
    quantization: 量化方法（可选）

    返回:
    str: 缓存目录路径
    """
    from modelscope.hub.file_download import model_file_download
    from modelscope.hub.snapshot_download import snapshot_download

    # 获取缓存目录
    cache_dir = _get_cache_dir(llm_family, llm_spec)
    # 检查是否需要跳过下载
    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    # 处理不同的模型格式
    if llm_spec.model_format in ["pytorch", "gptq", "awq", "fp8", "mlx"]:
        # 下载模型快照
        download_dir = retry_download(
            snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            revision=llm_spec.model_revision,
        )
        # 创建符号链接
        create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggufv2"]:
        # 生成模型文件名
        file_names, final_file_name, need_merge = _generate_model_file_names(
            llm_spec, quantization
        )

        # 下载每个文件
        for filename in file_names:
            download_path = retry_download(
                model_file_download,
                llm_family.model_name,
                {
                    "model_size": llm_spec.model_size_in_billions,
                    "model_format": llm_spec.model_format,
                },
                llm_spec.model_id,
                filename,
                revision=llm_spec.model_revision,
            )
            # 创建本地文件的符号链接
            symlink_local_file(download_path, cache_dir, filename)

        # 如果需要合并文件
        if need_merge:
            _merge_cached_files(cache_dir, file_names, final_file_name)
    else:
        raise ValueError(f"不支持的格式: {llm_spec.model_format}")

    # 生成元数据文件路径
    meta_path = _get_meta_path(
        cache_dir, llm_spec.model_format, llm_spec.model_hub, quantization
    )
    # 生成元数据文件
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir


def cache_from_huggingface(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    从Hugging Face缓存模型。返回缓存目录。

    参数:
    llm_family: LLM家族对象
    llm_spec: LLM规格对象
    quantization: 量化方法（可选）

    返回:
    str: 缓存目录路径
    """
    import huggingface_hub

    # 获取缓存目录
    cache_dir = _get_cache_dir(llm_family, llm_spec)
    # 检查是否需要跳过下载
    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    # 设置符号链接选项
    use_symlinks = {}
    if not IS_NEW_HUGGINGFACE_HUB:
        use_symlinks = {"local_dir_use_symlinks": True, "local_dir": cache_dir}

    # 处理不同的模型格式
    if llm_spec.model_format in ["pytorch", "gptq", "awq", "fp8", "mlx"]:
        assert isinstance(llm_spec, (PytorchLLMSpecV1, MLXLLMSpecV1))
        # 下载模型快照
        download_dir = retry_download(
            huggingface_hub.snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            revision=llm_spec.model_revision,
            **use_symlinks,
        )
        # 对于新版本的huggingface_hub，创建符号链接
        if IS_NEW_HUGGINGFACE_HUB:
            create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggufv2"]:
        assert isinstance(llm_spec, LlamaCppLLMSpecV1)
        # 生成模型文件名
        file_names, final_file_name, need_merge = _generate_model_file_names(
            llm_spec, quantization
        )

        # 下载每个文件
        for file_name in file_names:
            download_file_path = retry_download(
                huggingface_hub.hf_hub_download,
                llm_family.model_name,
                {
                    "model_size": llm_spec.model_size_in_billions,
                    "model_format": llm_spec.model_format,
                },
                llm_spec.model_id,
                revision=llm_spec.model_revision,
                filename=file_name,
                **use_symlinks,
            )
            # 对于新版本的huggingface_hub，创建符号链接
            if IS_NEW_HUGGINGFACE_HUB:
                symlink_local_file(download_file_path, cache_dir, file_name)

        # 如果需要，合并缓存的文件
        if need_merge:
            _merge_cached_files(cache_dir, file_names, final_file_name)
    else:
        raise ValueError(f"不支持的模型格式: {llm_spec.model_format}")

    # 生成元数据文件路径
    meta_path = _get_meta_path(
        cache_dir, llm_spec.model_format, llm_spec.model_hub, quantization
    )
    # 生成元数据文件
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir

def _check_revision(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    builtin: list,
    meta_path: str,
    quantization: Optional[str] = None,
) -> bool:
    """
    检查模型版本是否有效。
    验证模型版本：确保使用的模型版本是有效的和预期的。
    兼容性检查：确保所请求的模型配置（格式、大小、量化）与内置模型列表中的配置相匹配。
    安全性：防止使用未经验证或不兼容的模型版本。


    参数:
    llm_family: LLM家族对象
    llm_spec: LLM规格对象
    builtin: 内置模型列表
    meta_path: 元数据文件路径
    quantization: 量化方法（可选）
    
    
    llm_family = LLMFamilyV1(model_name="bert")
    llm_spec = LLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=0.3,
        quantizations=["8bit", "4bit"]
    )
    builtin = [
        LLMFamilyV1(
            model_name="bert",
            model_specs=[
                LLMSpecV1(
                    model_format="pytorch",
                    model_size_in_billions=0.3,
                    quantizations=["8bit", "4bit"],
                    model_revision="v2.1"
                )
            ]
        )
    ]
    meta_path = "/path/to/meta/file"
    quantization = None

    # 假设 valid_model_revision 函数返回 False
    result = _check_revision(llm_family, llm_spec, builtin, meta_path, quantization)
    # 输出: False

    返回:
    bool: 如果找到匹配的模型规格并且版本有效，则返回True；否则返回False
    """
    # 遍历内置模型家族
    for family in builtin:
        # 检查模型名称是否匹配
        if llm_family.model_name == family.model_name:
            specs = family.model_specs
            # 遍历模型规格
            for spec in specs:
                # 检查模型格式、大小和量化方法是否匹配
                if (
                    spec.model_format == "pytorch"
                    and spec.model_size_in_billions == llm_spec.model_size_in_billions
                    and (quantization is None or quantization in spec.quantizations)
                ):
                    # 如果找到匹配的规格，验证模型版本
                    return valid_model_revision(meta_path, spec.model_revision)
    # 如果没有找到匹配的规格，返回False
    return False


def get_cache_status(
    llm_family: LLMFamilyV1, llm_spec: "LLMSpecV1", quantization: Optional[str] = None
) -> Union[bool, List[bool]]:
    """
    根据模型格式和量化检查模型的缓存状态是否可用。
    支持不同的目录和模型格式。
    """

    def check_file_status(meta_path: str) -> bool:
        """检查元数据文件是否存在"""
        return os.path.exists(meta_path)

    def check_revision_status(
        meta_path: str, families: list, quantization: Optional[str] = None
    ) -> bool:
        """检查模型版本是否有效"""
        return _check_revision(llm_family, llm_spec, families, meta_path, quantization)

    def handle_quantization(q: Union[str, None]) -> bool:
        """处理特定量化的缓存状态"""
        # 获取特定缓存目录
        specific_cache_dir = _get_cache_dir(
            llm_family, llm_spec, q, create_if_not_exist=False
        )
        # 获取不同模型中心的元数据路径
        meta_paths = {
            "huggingface": _get_meta_path(
                specific_cache_dir, llm_spec.model_format, "huggingface", q
            ),
            "modelscope": _get_meta_path(
                specific_cache_dir, llm_spec.model_format, "modelscope", q
            ),
        }
        if llm_spec.model_format == "pytorch":
            # 对于PyTorch格式，检查版本状态
            return check_revision_status(
                meta_paths["huggingface"], BUILTIN_LLM_FAMILIES, q
            ) or check_revision_status(
                meta_paths["modelscope"], BUILTIN_MODELSCOPE_LLM_FAMILIES, q
            )
        else:
            # 对于其他格式，检查文件状态
            return check_file_status(meta_paths["huggingface"]) or check_file_status(
                meta_paths["modelscope"]
            )

    if llm_spec.model_id and "{" in llm_spec.model_id:
        # 如果模型ID包含占位符
        return (
            [handle_quantization(q) for q in llm_spec.quantizations]
            if quantization is None
            else handle_quantization(quantization)
        )
    else:
        # 如果模型ID不包含占位符
        return (
            [handle_quantization(q) for q in llm_spec.quantizations]
            if llm_spec.model_format != "pytorch"
            else handle_quantization(None)
        )


def get_user_defined_llm_families():
    """
    获取用户定义的LLM家族列表的副本。

    返回:
    List[LLMFamilyV1]: 用户定义的LLM家族列表的副本
    """
    with UD_LLM_FAMILIES_LOCK:
        return UD_LLM_FAMILIES.copy()


def match_model_size(
    model_size: Union[int, str], spec_model_size: Union[int, str]
) -> bool:
    """
    比较两个模型大小是否匹配。

    参数:
    model_size (Union[int, str]): 要比较的模型大小
    spec_model_size (Union[int, str]): 规格中的模型大小

    返回:
    bool: 如果模型大小匹配返回True，否则返回False
    """
    # 将字符串形式的模型大小中的下划线替换为点
    if isinstance(model_size, str):
        model_size = model_size.replace("_", ".")
    if isinstance(spec_model_size, str):
        spec_model_size = spec_model_size.replace("_", ".")

    # 如果字符串完全相同，直接返回True
    if model_size == spec_model_size:
        return True

    try:
        # 尝试将模型大小转换为整数进行比较
        ms = int(model_size)
        ss = int(spec_model_size)
        return ms == ss
    except ValueError:
        # 如果转换失败（可能是浮点数），返回False
        return False


def convert_model_size_to_float(
    model_size_in_billions: Union[float, int, str]
) -> float:
    """
    将模型大小转换为浮点数。

    参数:
    model_size_in_billions (Union[float, int, str]): 模型大小，可以是浮点数、整数或字符串。

    返回:
    float: 转换后的浮点数形式的模型大小。
    """
    if isinstance(model_size_in_billions, str):
        if "_" in model_size_in_billions:
            # 将形如 "3_5" 的字符串转换为 "3.5"，然后转为浮点数
            ms = model_size_in_billions.replace("_", ".")
            return float(ms)
        elif "." in model_size_in_billions:
            # 如果字符串中已经包含小数点，直接转换为浮点数
            return float(model_size_in_billions)
        else:
            # 如果是整数形式的字符串，转换为整数
            return int(model_size_in_billions)
    # 如果输入已经是浮点数或整数，直接返回
    return model_size_in_billions


def match_llm(
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
) -> Optional[Tuple[LLMFamilyV1, LLMSpecV1, str]]:
    """
    根据给定的条件查找匹配的LLM家族、规格和量化方法。

    参数:
    model_name (str): 模型名称
    model_format (Optional[str]): 模型格式
    model_size_in_billions (Optional[Union[int, str]]): 模型大小（以十亿参数计）
    quantization (Optional[str]): 量化方法
    download_hub (Optional[Literal["huggingface", "modelscope", "csghub"]]): 下载中心

    返回:
    Optional[Tuple[LLMFamilyV1, LLMSpecV1, str]]: 匹配的LLM家族、规格和量化方法，如果没有匹配项则返回None
    """
    # 获取用户定义的LLM家族
    user_defined_llm_families = get_user_defined_llm_families()

    def _match_quantization(q: Union[str, None], quantizations: List[str]):
        # Currently, the quantization name could include both uppercase and lowercase letters,
        # so it is necessary to ensure that the case sensitivity does not
        # affect the matching results.
        if q is None:
            return q
        for quant in quantizations:
            if q.lower() == quant.lower():
                return quant

    def _apply_format_to_model_id(spec: LLMSpecV1, q: str) -> LLMSpecV1:
        # Different quantized versions of some models use different model ids,
        # Here we check the `{}` in the model id to format the id.
        # 为某些模型的不同量化版本应用不同的模型ID
        if spec.model_id and "{" in spec.model_id:
            spec.model_id = spec.model_id.format(quantization=q)
        return spec

    # priority: download_hub > download_from_modelscope() and download_from_csghub()
    if download_hub == "modelscope":
        all_families = (
            BUILTIN_MODELSCOPE_LLM_FAMILIES
            + BUILTIN_LLM_FAMILIES
            + user_defined_llm_families
        )
    elif download_hub == "csghub":
        all_families = (
            BUILTIN_CSGHUB_LLM_FAMILIES
            + BUILTIN_LLM_FAMILIES
            + user_defined_llm_families
        )
    elif download_hub == "huggingface":
        all_families = BUILTIN_LLM_FAMILIES + user_defined_llm_families
    elif download_from_modelscope():
        all_families = (
            BUILTIN_MODELSCOPE_LLM_FAMILIES
            + BUILTIN_LLM_FAMILIES
            + user_defined_llm_families
        )
    elif download_from_csghub():
        all_families = (
            BUILTIN_CSGHUB_LLM_FAMILIES
            + BUILTIN_LLM_FAMILIES
            + user_defined_llm_families
        )
    else:
        all_families = BUILTIN_LLM_FAMILIES + user_defined_llm_families

    # 遍历所有LLM家族，查找匹配项
    for family in all_families:
        if model_name != family.model_name:
            continue
        for spec in family.model_specs:
            # 匹配量化方法
            matched_quantization = _match_quantization(quantization, spec.quantizations)
            # 检查是否满足所有匹配条件
            if (
                model_format
                and model_format != spec.model_format
                or model_size_in_billions
                and not match_model_size(
                    model_size_in_billions, spec.model_size_in_billions
                )
                or quantization
                and matched_quantization is None
            ):
                continue
            # Copy spec to avoid _apply_format_to_model_id modify the original spec.
            spec = spec.copy()
            if quantization:
                # 如果指定了量化方法，返回匹配的结果
                return (
                    family,
                    _apply_format_to_model_id(spec, matched_quantization),
                    matched_quantization,
                )
            else:
                # TODO: If user does not specify quantization, just use the first one
                _q = "none" if spec.model_format == "pytorch" else spec.quantizations[0]
                return family, _apply_format_to_model_id(spec, _q), _q
    # 如果没有找到匹配项，返回None
    return None


def register_llm(llm_family: LLMFamilyV1, persist: bool):
    from ..utils import is_valid_model_name
    from . import generate_engine_config_by_model_family

    # 验证模型名称是否有效
    if not is_valid_model_name(llm_family.model_name):
        raise ValueError(f"无效的模型名称 {llm_family.model_name}。")

    # 验证模型规格中的URI是否有效
    for spec in llm_family.model_specs:
        model_uri = spec.model_uri
        if model_uri and not is_valid_model_uri(model_uri):
            raise ValueError(f"无效的模型URI {model_uri}。")

    with UD_LLM_FAMILIES_LOCK:
        # 检查是否与现有模型名称冲突
        for family in BUILTIN_LLM_FAMILIES + UD_LLM_FAMILIES:
            if llm_family.model_name == family.model_name:
                raise ValueError(
                    f"模型名称与现有模型 {family.model_name} 冲突"
                )

        # 将新的LLM家族添加到用户定义列表中
        UD_LLM_FAMILIES.append(llm_family)
        # 为新添加的模型家族生成引擎配置
        generate_engine_config_by_model_family(llm_family)

    # 如果需要持久化，将模型信息保存到文件
    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "llm", f"{llm_family.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(llm_family.json())


def unregister_llm(model_name: str, raise_error: bool = True):
    """
    注销指定的LLM模型。

    参数:
    model_name (str): 要注销的模型名称
    raise_error (bool): 如果模型未找到，是否抛出异常，默认为True

    """
    with UD_LLM_FAMILIES_LOCK:
        # 查找要注销的LLM家族
        llm_family = None
        for i, f in enumerate(UD_LLM_FAMILIES):
            if f.model_name == model_name:
                llm_family = f
                break
        
        if llm_family:
            # 从用户定义的LLM家族列表中移除
            UD_LLM_FAMILIES.remove(llm_family)
            # 从LLM引擎字典中删除
            del LLM_ENGINES[model_name]

            # 删除持久化文件
            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "llm", f"{llm_family.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            # 处理缓存目录
            llm_spec = llm_family.model_specs[0]
            cache_dir_name = (
                f"{llm_family.model_name}-{llm_spec.model_format}"
                f"-{llm_spec.model_size_in_billions}b"
            )
            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"移除用户定义模型 {llm_family.model_name} 的缓存。"
                    f"缓存目录: {cache_dir}"
                )
                if os.path.islink(cache_dir):
                    # 如果是软链接，直接删除
                    os.remove(cache_dir)
                else:
                    # 如果不是软链接，提示用户手动删除
                    logger.warning(
                        f"缓存目录不是软链接，请手动删除。"
                    )
        else:
            # 模型未找到的处理
            if raise_error:
                raise ValueError(f"未找到模型 {model_name}")
            else:
                logger.warning(f"未找到自定义模型 {model_name}")


def check_engine_by_spec_parameters(
    model_engine: str,
    model_name: str,
    model_format: str,
    model_size_in_billions: Union[str, int],
    quantization: str,
) -> Type[LLM]:
    # 定义一个内部函数，用于获取正确的模型引擎名称
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in LLM_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    # 检查模型名称是否存在
    if model_name not in LLM_ENGINES:
        raise ValueError(f"模型 {model_name} 未找到。")
    
    # 获取正确的模型引擎名称
    model_engine = get_model_engine_from_spell(model_engine)
    
    # 检查模型是否可以在指定引擎上运行
    if model_engine not in LLM_ENGINES[model_name]:
        raise ValueError(f"模型 {model_name} 无法在引擎 {model_engine} 上运行。")
    
    # 获取匹配参数
    match_params = LLM_ENGINES[model_name][model_engine]
    
    # 遍历匹配参数，查找符合条件的LLM类
    for param in match_params:
        if (
            model_name == param["model_name"]
            and model_format == param["model_format"]
            and model_size_in_billions == param["model_size_in_billions"]
            and quantization in param["quantizations"]
        ):
            return param["llm_class"]
    
    # 如果没有找到匹配的LLM类，抛出异常
    raise ValueError(
        f"模型 {model_name} 无法在引擎 {model_engine} 上运行，格式为 {model_format}，大小为 {model_size_in_billions}，量化方法为 {quantization}。"
    )
