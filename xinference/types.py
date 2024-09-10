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

from typing import Any, Callable, Dict, ForwardRef, Iterable, List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict

from ._compat import (
    BaseModel,
    create_model,
    create_model_from_typeddict,
    validate_arguments,
)
from .fields import (
    echo_field,
    frequency_penalty_field,
    logprobs_field,
    max_tokens_field,
    none_field,
    presence_penalty_field,
    repeat_penalty_field,
    stop_field,
    stream_field,
    stream_interval_field,
    stream_option_field,
    temperature_field,
    top_k_field,
    top_p_field,
)

# 特殊工具提示符
SPECIAL_TOOL_PROMPT = "<TOOL>"


class Image(TypedDict):
    url: Optional[str]  # 图像的URL
    b64_json: Optional[str]  # Base64编码的图像数据


class ImageList(TypedDict):
    created: int  # 创建时间戳
    data: List[Image]  # 图像列表


class Video(TypedDict):
    url: Optional[str]  # 视频的URL
    b64_json: Optional[str]  # Base64编码的视频数据


class VideoList(TypedDict):
    created: int  # 创建时间戳
    data: List[Video]  # 视频列表


class EmbeddingUsage(TypedDict):
    prompt_tokens: int  # 提示词的token数量
    total_tokens: int  # 总token数量


class EmbeddingData(TypedDict):
    index: int  # 嵌入向量的索引
    object: str  # 对象类型
    embedding: List[float]  # 嵌入向量


class Embedding(TypedDict):
    object: Literal["list"]  # 对象类型，固定为"list"
    model: str  # 使用的模型名称
    data: List[EmbeddingData]  # 嵌入数据列表
    usage: EmbeddingUsage  # 使用情况


class Document(TypedDict):
    text: str  # 文档文本内容


class DocumentObj(TypedDict):
    index: int  # 文档索引
    relevance_score: float  # 相关性得分
    document: Optional[Document]  # 文档对象


# Cohere API 兼容性
class ApiVersion(TypedDict):
    version: str  # API版本号
    is_deprecated: bool  # 是否已废弃
    is_experimental: bool  # 是否为实验性版本


# Cohere API 兼容性
class BilledUnit(TypedDict):
    input_tokens: int  # 输入token数量
    output_tokens: int  # 输出token数量
    search_units: int  # 搜索单元数量
    classifications: int  # 分类数量


class RerankTokens(TypedDict):
    input_tokens: int  # 输入token数量
    output_tokens: int  # 输出token数量


class Meta(TypedDict):
    api_version: Optional[ApiVersion]  # API版本信息
    billed_units: Optional[BilledUnit]  # 计费单位
    tokens: RerankTokens  # 重排序token信息
    warnings: Optional[List[str]]  # 警告信息列表


class Rerank(TypedDict):
    id: str  # 重排序ID
    results: List[DocumentObj]  # 重排序结果列表
    meta: Meta  # 元数据信息


class CompletionLogprobs(TypedDict):
    text_offset: List[int]  # 文本偏移量列表
    token_logprobs: List[Optional[float]]  # token对数概率列表
    tokens: List[str]  # token列表
    top_logprobs: List[Optional[Dict[str, float]]]  # 顶部对数概率列表


class ToolCallFunction(TypedDict):
    name: str  # 函数名称
    arguments: str  # 函数参数


class ToolCalls(TypedDict):
    id: str  # 工具调用ID
    type: Literal["function"]  # 工具类型，固定为"function"
    function: ToolCallFunction  # 函数调用信息


class CompletionChoice(TypedDict):
    text: str  # 生成的文本
    index: int  # 选择索引
    logprobs: Optional[CompletionLogprobs]  # 对数概率信息
    finish_reason: Optional[str]  # 完成原因
    tool_calls: NotRequired[List[ToolCalls]]  # 工具调用列表（可选）


class CompletionUsage(TypedDict):
    prompt_tokens: int  # 提示词token数量
    completion_tokens: int  # 完成token数量
    total_tokens: int  # 总token数量


class CompletionChunk(TypedDict):
    id: str  # 完成块ID
    object: Literal["text_completion"]  # 对象类型，固定为"text_completion"
    created: int  # 创建时间戳
    model: str  # 使用的模型名称
    choices: List[CompletionChoice]  # 完成选择列表
    usage: NotRequired[CompletionUsage]  # 使用情况（可选）


class Completion(TypedDict):
    id: str  # 完成ID
    object: Literal["text_completion"]  # 对象类型，固定为"text_completion"
    created: int  # 创建时间戳
    model: str  # 使用的模型名称
    choices: List[CompletionChoice]  # 完成选择列表
    usage: CompletionUsage  # 使用情况


class ChatCompletionMessage(TypedDict):
    role: str  # 消息角色
    content: Optional[str]  # 消息内容
    user: NotRequired[str]  # 用户标识（可选）
    tool_calls: NotRequired[List]  # 工具调用列表（可选）


class ChatCompletionChoice(TypedDict):
    index: int  # 选择索引
    message: ChatCompletionMessage  # 聊天完成消息
    finish_reason: Optional[str]  # 完成原因


class ChatCompletion(TypedDict):
    id: str  # 聊天完成ID
    object: Literal["chat.completion"]  # 对象类型，固定为"chat.completion"
    created: int  # 创建时间戳
    model: str  # 使用的模型名称
    choices: List[ChatCompletionChoice]  # 聊天完成选择列表
    usage: CompletionUsage  # 使用情况


class ChatCompletionChunkDelta(TypedDict):
    role: NotRequired[str]  # 角色（可选）
    content: NotRequired[str]  # 内容（可选）
    tool_calls: NotRequired[List[ToolCalls]]  # 工具调用列表（可选）


class ChatCompletionChunkChoice(TypedDict):
    index: int  # 选择索引
    delta: ChatCompletionChunkDelta  # 增量信息
    finish_reason: Optional[str]  # 完成原因


class ChatCompletionChunk(TypedDict):
    id: str  # 聊天完成块ID
    model: str  # 使用的模型名称
    object: Literal["chat.completion.chunk"]  # 对象类型，固定为"chat.completion.chunk"
    created: int  # 创建时间戳
    choices: List[ChatCompletionChunkChoice]  # 聊天完成块选择列表
    usage: NotRequired[CompletionUsage]  # 使用情况（可选）

    
    
# # 定义一个停止条件
# def length_limit(input_ids, logits):
#     return len(input_ids) >= 100  # 当生成的文本长度达到100时停止

# # 定义一个对数概率处理器
# def temperature_scaling(input_ids, scores, temperature=0.7):
#     return [score / temperature for score in scores]

# # 创建停止条件列表
# stop_criteria = StoppingCriteriaList([length_limit])

# # 创建对数概率处理器列表
# logits_processors = LogitsProcessorList([
#     lambda ids, scores: temperature_scaling(ids, scores, 0.7)
# ])

# # 在文本生成循环中使用
# while not stop_criteria(current_ids, current_logits):
#     # 处理对数概率
#     processed_logits = logits_processors(current_ids, current_logits)
#     # 使用处理后的对数概率生成下一个词
#     # ...

# 定义停止标准的类型，接受输入ID列表和对数概率列表，返回布尔值
# Callable:
# Callable 是 Python 的类型提示（type hint）之一，来自 typing 模块。
# 它用于表示可调用对象，如函数、方法或实现了 __call__ 方法的类。
# 在这里，Callable[[参数类型], 返回类型] 表示一个接受特定类型参数并返回特定类型结果的可调用对象
StoppingCriteria = Callable[[List[int], List[float]], bool]


# 这是一个自定义类，继承自 List[StoppingCriteria]，即停止条件函数的列表。
# 它重写了 __call__ 方法，使得这个类的实例可以像函数一样被调用。
# 当被调用时，它会遍历所有的停止条件，如果任何一个条件满足（返回 True），整个列表就返回 True
class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(self, input_ids: List[int], logits: List[float]) -> bool:
        # 如果任何一个停止标准满足，则返回True
        return any([stopping_criteria(input_ids, logits) for stopping_criteria in self])


# 定义对数概率处理器的类型，接受输入ID列表和分数列表，返回处理后的分数列表
LogitsProcessor = Callable[[List[int], List[float]], List[float]]

# 这是一个自定义类，继承自 List[LogitsProcessor]，即对数概率处理器的列表。
# 它重写了 __call__ 方法，使得这个类的实例可以像函数一样被调用。
# 当被调用时，它会遍历所有的对数概率处理器，依次应用每个处理器，并返回处理后的分数列表。
class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        # 依次应用每个处理器
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class LlamaCppGenerateConfig(TypedDict, total=False):
    suffix: Optional[str]  # 可选的后缀
    max_tokens: int  # 生成的最大token数
    temperature: float  # 采样温度
    top_p: float  # 累积概率阈值
    logprobs: Optional[int]  # 返回的对数概率数量
    echo: bool  # 是否回显输入
    stop: Optional[Union[str, List[str]]]  # 停止生成的标记
    frequency_penalty: float  # 频率惩罚
    presence_penalty: float  # 存在惩罚
    repetition_penalty: float  # 重复惩罚
    top_k: int  # 考虑的最高概率词数
    stream: bool  # 是否流式输出
    stream_options: Optional[Union[dict, None]]  # 流式输出选项
    tfs_z: float  # Tail-free sampling参数
    mirostat_mode: int  # Mirostat采样模式
    mirostat_tau: float  # Mirostat目标熵
    mirostat_eta: float  # Mirostat学习率
    model: Optional[str]  # 模型名称
    grammar: Optional[Any]  # 语法约束
    stopping_criteria: Optional["StoppingCriteriaList"]  # 停止标准列表
    logits_processor: Optional["LogitsProcessorList"]  # logits处理器列表
    tools: Optional[List[Dict]]  # 可用工具列表


class LlamaCppModelConfig(TypedDict, total=False):
    n_ctx: int  # 上下文窗口大小
    n_parts: int  # 模型分割数
    n_gpu_layers: int  # GPU层数
    split_mode: int  # 分割模式
    main_gpu: int  # 主GPU索引
    seed: int  # 随机种子
    f16_kv: bool  # 是否使用float16 key/value缓存
    logits_all: bool  # 是否计算所有logits
    vocab_only: bool  # 是否只加载词汇表
    use_mmap: bool  # 是否使用内存映射
    use_mlock: bool  # 是否锁定内存
    n_threads: Optional[int]  # 线程数
    n_batch: int  # 批处理大小
    last_n_tokens_size: int  # 最后N个token的大小
    lora_base: Optional[str]  # LoRA基础模型路径
    lora_path: Optional[str]  # LoRA模型路径
    low_vram: bool  # 是否使用低显存模式
    n_gqa: Optional[int]  # GQA头数（临时，必须为llama2 70b设置为8）
    rms_norm_eps: Optional[float]  # RMS归一化epsilon（临时）
    verbose: bool  # 是否输出详细日志


class PytorchGenerateConfig(TypedDict, total=False):
    temperature: float  # 采样温度
    repetition_penalty: float  # 重复惩罚
    top_p: float  # 累积概率阈值
    top_k: int  # 考虑的最高概率词数
    stream: bool  # 是否流式输出
    max_tokens: int  # 生成的最大token数
    echo: bool  # 是否回显输入
    stop: Optional[Union[str, List[str]]]  # 停止生成的标记
    stop_token_ids: Optional[Union[int, List[int]]]  # 停止生成的token ID
    stream_interval: int  # 流式输出间隔
    model: Optional[str]  # 模型名称
    tools: Optional[List[Dict]]  # 可用工具列表
    lora_name: Optional[str]  # LoRA模型名称
    stream_options: Optional[Union[dict, None]]  # 流式输出选项
    request_id: Optional[str]  # 请求ID


class PytorchModelConfig(TypedDict, total=False):
    revision: Optional[str]  # 模型版本
    device: str  # 设备类型（如'cuda'或'cpu'）
    gpus: Optional[str]  # GPU设备列表
    num_gpus: int  # 使用的GPU数量
    max_gpu_memory: str  # 最大GPU内存使用量
    gptq_ckpt: Optional[str]  # GPTQ检查点路径
    gptq_wbits: int  # GPTQ量化位数
    gptq_groupsize: int  # GPTQ分组大小
    gptq_act_order: bool  # 是否使用GPTQ激活顺序
    trust_remote_code: bool  # 是否信任远程代码
    max_num_seqs: int  # 最大序列数
    enable_tensorizer: Optional[bool]  # 是否启用tensorizer

def get_pydantic_model_from_method(
    meth,
    exclude_fields: Optional[Iterable[str]] = None,
    include_fields: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """

    从给定的方法创建一个Pydantic模型。
    这个函数`get_pydantic_model_from_method`的主要目的是从一个方法创建一个Pydantic模型，
    同时允许自定义包含或排除某些字段。

    从现有方法创建API模型：
    当你有一个现有的方法，想要为其创建一个对应的API请求或响应模型时。
    b. 自动生成文档：
    可以用于自动生成API文档，因为Pydantic模型可以很容易地转换为OpenAPI规范。
    c. 参数验证：
    创建的模型可以用于验证传入的参数。
    d. 模型定制：
    允许你基于现有方法创建模型，同时灵活地添加或删除字段。

    from pydantic import BaseModel
    from typing import Optional

    def example_method(param1: int, param2: str, optional_param: Optional[float] = None):
        pass

    # 创建基本模型
    basic_model = get_pydantic_model_from_method(example_method)
    print(basic_model.__fields__.keys())
    # 输出可能是: dict_keys(['param1', 'param2', 'optional_param'])

    # 创建排除某些字段的模型
    excluded_model = get_pydantic_model_from_method(example_method, exclude_fields=['optional_param'])
    print(excluded_model.__fields__.keys())
    # 输出可能是: dict_keys(['param1', 'param2'])

    # 创建包含额外字段的模型
    included_model = get_pydantic_model_from_method(
        example_method, 
        include_fields={'extra_field': (str, ...)}
    )
    print(included_model.__fields__.keys())
    # 输出可能是: dict_keys(['param1', 'param2', 'optional_param', 'extra_field'])

    # 使用生成的模型
    model_instance = basic_model(param1=1, param2="test")
    print(model_instance)

    参数:
    meth: 方法对象
    exclude_fields: 要排除的字段列表
    include_fields: 要包含的字段字典

    返回:
    一个Pydantic模型

    """
    # 使用validate_arguments装饰器验证方法参数
    f = validate_arguments(meth, config={"arbitrary_types_allowed": True})
    model = f.model
    
    # 移除一些不需要的字段
    model.__fields__.pop("self", None)
    model.__fields__.pop("args", None)
    model.__fields__.pop("kwargs", None)
    
    # 移除以"v__"开头的Pydantic私有字段
    pydantic_private_keys = [
        key for key in model.__fields__.keys() if key.startswith("v__")
    ]
    for key in pydantic_private_keys:
        model.__fields__.pop(key)
    
    # 如果提供了exclude_fields，从模型中移除这些字段
    if exclude_fields is not None:
        for key in exclude_fields:
            model.__fields__.pop(key, None)
    
    # 如果提供了include_fields，将这些字段添加到模型中
    if include_fields is not None:
        dummy_model = create_model("DummyModel", **include_fields)
        model.__fields__.update(dummy_model.__fields__)
    
    return model


def fix_forward_ref(model):
    """
    修复 pydantic 在 Python 3.8 中生成的 ForwardRef 字段问题。
    我们将这些字段替换为 Optional[Any] 类型。

    ForwardRef（前向引用）：
    ForwardRef是Python类型提示系统中的一个概念，用于处理循环引用或者在定义时还不存在的类型。
    在Pydantic中，当你在一个类中引用这个类本身或者还未定义的类时，就会使用ForwardRef。
    ForwardRef在某些Python版本（特别是3.8）中可能会导致问题，尤其是在处理复杂的嵌套模型时。
    将ForwardRef替换为Optional[Any]可以避免这些问题，因为Any类型基本上告诉Python"这可以是任何类型"。
    Optional表示这个字段可以是None或者任何其他类型。

    class Node(BaseModel):
         value: int
         next: Optional['Node'] = None  # 这里的 'Node' 就是一个 ForwardRef


    修复后的模型
    class Node(BaseModel):
         value: int
         next: Optional[Any] = None

    参数:
    model: pydantic 模型

    返回:
    修复后的 pydantic 模型
    """
    exclude_fields = []  # 存储需要排除的字段
    include_fields = {}  # 存储需要包含的字段

    # 遍历模型的所有字段
    for key, field in model.__fields__.items():
        if isinstance(field.annotation, ForwardRef):
            exclude_fields.append(key)
            include_fields[key] = (Optional[Any], None)

    # 从模型中移除需要排除的字段
    if exclude_fields:
        for key in exclude_fields:
            model.__fields__.pop(key, None)

    # 添加需要包含的字段
    if include_fields:
        # 创建一个临时的模型，用于包含需要包含的字段。DummyModel是临时模型的名称。
        dummy_model = create_model("DummyModel", **include_fields)

        # 将`dummy_model`的字段添加到原始模型中。
        # `update`方法将`dummy_model`的字段合并到原始模型的字段中。
        model.__fields__.update(dummy_model.__fields__)

    return model  # 返回修复后的模型


class ModelAndPrompt(BaseModel):
    model: str  # 模型名称
    prompt: str  # 提示文本


# 当 echo 设置为 True 时，模型的输出会包含原始的输入文本，然后再跟上生成的新文本。
# 当 echo 设置为 False 时，输出中只包含新生成的文本，不包括原始输入。
class CreateCompletionTorch(BaseModel):
    echo: bool = echo_field  # 是否回显输入
    max_tokens: Optional[int] = max_tokens_field  # 生成的最大token数
    repetition_penalty: float = repeat_penalty_field  # 重复惩罚系数
    stop: Optional[Union[str, List[str]]] = stop_field  # 停止生成的标记
    stop_token_ids: Optional[Union[int, List[int]]] = none_field  # 停止生成的token ID
    stream: bool = stream_field  # 是否使用流式输出
    stream_options: Optional[Union[dict, None]] = stream_option_field  # 流式输出选项
    stream_interval: int = stream_interval_field  # 流式输出间隔
    temperature: float = temperature_field  # 采样温度
    top_p: float = top_p_field  # 累积概率阈值
    top_k: int = top_k_field  # 考虑的最高概率词数
    lora_name: Optional[str]  # LoRA模型名称
    request_id: Optional[str]  # 请求ID


CreateCompletionLlamaCpp: BaseModel
try:
    from llama_cpp import Llama

    CreateCompletionLlamaCpp = get_pydantic_model_from_method(
        Llama.create_completion,
        exclude_fields=["model", "prompt", "grammar", "max_tokens"],
        include_fields={
            "grammar": (Optional[Any], None),  # 语法约束
            "max_tokens": (Optional[int], max_tokens_field),  # 生成的最大token数
            "lora_name": (Optional[str], None),  # LoRA模型名称
            "stream_options": (Optional[Union[dict, None]], None),  # 流式输出选项
        },
    )
except ImportError:
    CreateCompletionLlamaCpp = create_model("CreateCompletionLlamaCpp")


# 这个类型用于OpenAI API兼容性
CreateCompletionOpenAI: BaseModel


class _CreateCompletionOpenAIFallback(BaseModel):
    # OpenAI's create completion request body, we define it by pydantic
    # model to verify the input params.
    # https://platform.openai.com/docs/api-reference/completions/object
    model: str  # 模型名称
    prompt: str  # 提示文本
    best_of: Optional[int] = 1  # 生成多个完成并返回最佳结果
    echo: bool = echo_field  # 是否回显输入
    frequency_penalty: Optional[float] = frequency_penalty_field  # 频率惩罚
    logit_bias: Optional[Dict[str, float]] = none_field  # 对特定token的偏好
    logprobs: Optional[int] = logprobs_field  # 返回的对数概率数量
    max_tokens: int = max_tokens_field  # 生成的最大token数
    n: Optional[int] = 1  # 生成的完成数量
    presence_penalty: Optional[float] = presence_penalty_field  # 存在惩罚
    seed: Optional[int] = none_field  # 随机种子
    stop: Optional[Union[str, List[str]]] = stop_field  # 停止生成的标记
    stream: bool = stream_field  # 是否使用流式输出
    stream_options: Optional[Union[dict, None]] = stream_option_field  # 流式输出选项
    suffix: Optional[str] = none_field  # 生成文本的后缀
    temperature: float = temperature_field  # 采样温度
    top_p: float = top_p_field  # 累积概率阈值
    user: Optional[str] = none_field  # 用户标识符


try:
    # 适用于 openai > 1 版本
    from openai.types.completion_create_params import CompletionCreateParamsNonStreaming

    CreateCompletionOpenAI = create_model_from_typeddict(
        CompletionCreateParamsNonStreaming,
    )
    CreateCompletionOpenAI = fix_forward_ref(CreateCompletionOpenAI)
except ImportError:
    # TODO(codingl2k1): 如果不再支持 openai < 1 版本，则移除此部分
    CreateCompletionOpenAI = _CreateCompletionOpenAIFallback


class CreateCompletion(
    ModelAndPrompt,
    CreateCompletionTorch,
    CreateCompletionLlamaCpp,
    CreateCompletionOpenAI,
):
    pass  # 创建完成类，继承多个基类以支持不同后端


class CreateChatModel(BaseModel):
    model: str  # 聊天模型名称


# 目前，聊天调用生成，所以参数共享相同的类
CreateChatCompletionTorch = CreateCompletionTorch
CreateChatCompletionLlamaCpp: BaseModel = CreateCompletionLlamaCpp

# 这个类型用于OpenAI API兼容性
CreateChatCompletionOpenAI: BaseModel


# 仅支持 openai > 1 版本
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)

CreateChatCompletionOpenAI = create_model_from_typeddict(
    CompletionCreateParamsNonStreaming,
)
CreateChatCompletionOpenAI = fix_forward_ref(CreateChatCompletionOpenAI)

class CreateChatCompletion(
    CreateChatModel,
    CreateChatCompletionTorch,
    CreateChatCompletionLlamaCpp,
    CreateChatCompletionOpenAI,
):
    """
    多重继承：
    CreateCompletion类继承自四个不同的类：
    ModelAndPrompt
    CreateCompletionTorch
    CreateCompletionLlamaCpp
    CreateCompletionOpenAI
    目的：

    这个类的主要目的是创建一个统一的接口，用于处理不同后端（PyTorch、LlamaCpp、OpenAI）的文本生成请求。

    功能：

    它结合了不同后端的配置选项和参数。
    允许使用一个统一的类来处理多种不同的文本生成模型和API。
    灵活性：
    通过继承多个类，CreateCompletion可以支持各种不同的文本生成场景和配置选

    Args:
        CreateChatModel (_type_): _description_
        CreateChatCompletionTorch (_type_): _description_
        CreateChatCompletionLlamaCpp (_type_): _description_
        CreateChatCompletionOpenAI (_type_): _description_
    """
    pass


class LoRA:
    def __init__(self, lora_name: str, local_path: str):
        """
        初始化LoRA对象。
        # 直接创建LoRA对象
        lora = LoRA(lora_name="my_lora", local_path="/path/to/lora/model")

        # 打印LoRA信息
        print(f"LoRA Name: {lora.lora_name}")
        print(f"Local Path: {lora.local_path}")
        :param lora_name: LoRA的名称
        :param local_path: LoRA的本地路径
        """
        self.lora_name = lora_name
        self.local_path = local_path

    def to_dict(self):
        """
        将LoRA对象转换为字典。
        lora_dict = lora.to_dict()
        print(lora_dict)
        输出: {'lora_name': 'my_lora', 'local_path': '/path/to/lora/model'}

        :return: 包含LoRA信息的字典

        """
        return {
            "lora_name": self.lora_name,
            "local_path": self.local_path,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """
        从字典创建LoRA对象。

        lora_data = {
            "lora_name": "another_lora",
            "local_path": "/path/to/another/lora/model"
        }
        new_lora = LoRA.from_dict(lora_data)

        print(f"New LoRA Name: {new_lora.lora_name}")
        print(f"New Local Path: {new_lora.local_path}")

        :param data: 包含LoRA信息的字典
        :return: 新创建的LoRA对象
        """
        return cls(
            lora_name=data["lora_name"],
            local_path=data["local_path"],
        )


class PeftModelConfig:
    def __init__(
        self,
        peft_model: Optional[List[LoRA]] = None,
        image_lora_load_kwargs: Optional[Dict] = None,
        image_lora_fuse_kwargs: Optional[Dict] = None,
    ):
        """
        初始化PeftModelConfig对象。

        :param peft_model: LoRA模型列表
        :param image_lora_load_kwargs: 加载LoRA模型时的配置参数
        :param image_lora_fuse_kwargs: 融合LoRA模型时的配置参数
        """
        self.peft_model = peft_model
        self.image_lora_load_kwargs = image_lora_load_kwargs
        self.image_lora_fuse_kwargs = image_lora_fuse_kwargs

    def to_dict(self):
        """
        将PeftModelConfig对象转换为字典。

        :return: 包含PeftModelConfig信息的字典
        """
        return {
            "lora_list": [lora.to_dict() for lora in self.peft_model]
            if self.peft_model
            else None,
            "image_lora_load_kwargs": self.image_lora_load_kwargs,
            "image_lora_fuse_kwargs": self.image_lora_fuse_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """
        从字典创建PeftModelConfig对象。

        :param data: 包含PeftModelConfig信息的字典
        :return: 新创建的PeftModelConfig对象
        """
        peft_model_list = data.get("lora_list", None)
        peft_model = (
            [LoRA.from_dict(lora_dict) for lora_dict in peft_model_list]
            if peft_model_list is not None
            else None
        )

        return cls(
            peft_model=peft_model,
            image_lora_load_kwargs=data.get("image_lora_load_kwargs"),
            image_lora_fuse_kwargs=data.get("image_lora_fuse_kwargs"),
        )
