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
import copy
import logging
import os
import random
import string
from typing import Dict, Generator, List, Tuple, Union

import orjson
from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

from .._compat import BaseModel

logger = logging.getLogger(__name__)


def log_async(logger, args_formatter=None):
    """
    创建一个异步日志装饰器。

    参数:
    logger: 用于记录日志的logger对象
    args_formatter: 可选的参数格式化函数

    返回:
    function: 装饰器函数
    """
    import time
    from functools import wraps

    def decorator(func):
        """
        实际的装饰器函数。

        参数:
        func: 被装饰的异步函数

        返回:
        function: 包装后的异步函数
        """
        @wraps(func)
        async def wrapped(*args, **kwargs):
            """
            
            @wraps(func) 的作用：
            保留被装饰函数的元数据（如函数名、文档字符串等）
            确保被装饰函数的 __name__、__doc__ 等属性不会被改变
            这对于调试和文档生成很重要
            
            
            包装函数，添加异步日志记录功能。

            参数:
            *args: 原函数的位置参数
            **kwargs: 原函数的所有额外的关键字参数

            返回:
            任意类型: 原异步函数的返回值
            """
            # 如果提供了参数格式化函数，则使用它
            if args_formatter is not None:
                formatted_args, formatted_kwargs = copy.copy(args), copy.copy(kwargs)
                args_formatter(formatted_args, formatted_kwargs)
            else:
                formatted_args, formatted_kwargs = args, kwargs
            
            # 记录函数进入的日志
            logger.debug(
                f"进入 {func.__name__}, 参数: {formatted_args}, 关键字参数: {formatted_kwargs}"
            )
            
            # 记录开始时间
            start = time.time()
            
            # 执行原异步函数
            ret = await func(*args, **kwargs)
            
            # 记录函数退出的日志，包括执行时间
            logger.debug(
                f"离开 {func.__name__}, 执行时间: {int(time.time() - start)} 秒"
            )
            return ret

        return wrapped

    return decorator


def log_sync(logger):
    """
    创建一个同步日志装饰器。

    参数:
    logger: 用于记录日志的logger对象

    返回:
    function: 装饰器函数
    """
    import time
    from functools import wraps

    def decorator(func):
        """
        实际的装饰器函数。

        参数:
        func: 被装饰的函数

        返回:
        function: 包装后的函数
        """
        @wraps(func)
        def wrapped(*args, **kwargs):
            """
            包装函数，添加日志记录功能。

            参数:
            *args: 原函数的位置参数
            **kwargs: 原函数的关键字参数

            返回:
            任意类型: 原函数的返回值
            """
            # 记录函数进入的日志
            logger.debug(f"进入 {func.__name__}, 参数: {args}, 关键字参数: {kwargs}")
            
            # 记录开始时间
            start = time.time()
            
            # 执行原函数
            ret = func(*args, **kwargs)
            
            # 记录函数退出的日志，包括执行时间
            logger.debug(
                f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} s"
            )
            
            return ret

        return wrapped

    return decorator


def iter_replica_model_uid(model_uid: str, replica: int) -> Generator[str, None, None]:
    """
    生成所有副本模型的唯一标识符。

    参数:
    model_uid (str): 原始模型的唯一标识符
    replica (int): 副本数量

    返回:
    Generator[str, None, None]: 生成器，用于迭代所有副本模型的唯一标识符

    说明:
    - 将replica转换为整数，确保输入的副本数是有效的整数
    - 使用range(replica)生成从0到replica-1的序列，作为副本ID
    - 对每个副本ID，生成格式为"{model_uid}-{replica}-{rep_id}"的唯一标识符
    
    
    使用yield所以只有需要的时候才会生成:
    for uid in iter_replica_model_uid(model_uid, 10000):
    # 这里的 uid 是在每次循环时才生成的
    process_model(uid)
    
    如果您只需要前100个副本，您可以只迭代100次，而不是生成全部10,000个标识符：

    for i, uid in enumerate(iter_replica_model_uid(model_uid, 10000)):
    if i >= 100:
        break
    process_model(uid)
    
    
    
    在分布式系统或大规模部署中，通常需要同一模型的多个副本来处理高并发请求。
    每个副本需要一个唯一的标识符，以便系统可以区分和管理它们。
    
    标识符生成：
    函数生成的标识符格式为 "{model_uid}-{replica}-{rep_id}"。
    这种格式包含了原始模型ID、总副本数和当前副本的编号。
    
    一致性：
    确保所有副本的标识符遵循相同的命名规则，便于系统统一管理。
    灵活性：
    使用生成器（Generator）允许按需生成标识符，而不是一次性生成所有标识符。
    """
    replica = int(replica)  # 确保replica是整数
    for rep_id in range(replica):
        yield f"{model_uid}-{replica}-{rep_id}"


def build_replica_model_uid(model_uid: str, replica: int, rep_id: int) -> str:
    """
    Build a replica model uid.
    """
    return f"{model_uid}-{replica}-{rep_id}"


def parse_replica_model_uid(replica_model_uid: str) -> Tuple[str, int, int]:
    """
    解析复制模型的唯一标识符，返回模型UID、副本数和副本ID。

    参数:
    replica_model_uid (str): 复制模型的唯一标识符

    返回:
    Tuple[str, int, int]: 包含模型UID、副本数和副本ID的元组

    说明:
    - 如果输入的标识符不包含副本信息，则返回原始标识符和默认值(-1, -1)
    - 否则，解析出模型UID、副本数和副本ID
    
    
    根据代码的实现，replica_model_uid 的格式应该是这样的：
    model_uid-replica-rep_id

    qwen2-instruct-2-1
    表示qwen2-instruct-2-1 的第2个副本，总共有2个副本

    qwen2-instruct-3-0
    表示qwen2-instruct-2-1 的第1个副本，总共有3个副本
    
    """
    # 将输入的标识符按"-"分割
    parts = replica_model_uid.split("-")
    
    # 如果分割后只有一个部分，说明没有副本信息
    if len(parts) == 1:
        return replica_model_uid, -1, -1
    
    # 从后往前解析副本ID和副本数
    rep_id = int(parts.pop())
    replica = int(parts.pop())
    
    # 剩余部分重新组合为模型UID
    model_uid = "-".join(parts)
    
    return model_uid, replica, rep_id


def is_valid_model_uid(model_uid: str) -> bool:
    """
    检查模型UID是否有效。

    参数:
    model_uid (str): 要检查的模型UID

    返回:
    bool: 如果模型UID有效则返回True，否则返回False

    说明:
    - 移除模型UID两端的空白字符
    - 检查模型UID是否为空或长度超过100个字符
    - 空UID或长度超过100的UID被认为是无效的
    """
    # 移除两端空白字符
    model_uid = model_uid.strip()
    # 检查是否为空或长度超过100
    if not model_uid or len(model_uid) > 100:
        return False
    return True


def gen_random_string(length: int) -> str:
    """
    生成指定长度的随机字符串。

    参数:
    length (int): 要生成的随机字符串的长度

    返回:
    str: 生成的随机字符串

    说明:
    - 使用 string.ascii_letters 和 string.digits 作为字符池
    - 从字符池中随机采样指定数量的字符
    - 将采样的字符连接成一个字符串并返回
    string.ascii_letters 和 string.digits 都是字符串,所以可以直接相加。
    备注:
    1. string.ascii_letters 是一个包含所有 ASCII 字母(大小写)的字符串:
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    string.digits 是一个包含所有数字字符的字符串:
    '0123456789'
    
    在 Python 中,字符串可以直接用 + 运算符连接。
    
    所以 string.ascii_letters + string.digits 会生成一个新的字符串,包含所有字母和数字:
    
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    这种方式提供了一个简便的方法来创建包含所有字母和数字的字符池,用于随机字符串生成。它比手动输入所有字符更方便,也不容易出错。
    
    """
    return "".join(random.sample(string.ascii_letters + string.digits, length))


def json_dumps(o):
    """
    将对象转换为JSON字符串。

    参数:
    o: 要序列化的对象

    返回:
    str: 序列化后的JSON字符串

    说明:
    - 使用orjson库进行JSON序列化
    - 对于BaseModel类型的对象，会调用其dict()方法进行序列化
    - 对于其他类型的对象，如果无法序列化则抛出TypeError异常
    
    
    关于 _default 函数：
    1. 作用：这是一个自定义的序列化处理函数，用于处理 orjson 无法直接序列化的对象。
    2. 实现：
    如果对象是 BaseModel 的实例（可能是 Pydantic 模型），它会调用 dict() 方法将其转换为字典。
    对于其他类型的对象，它会抛出 TypeError，表示无法序列化。
    3. 使用：通过 default=_default 参数传递给 orjson.dumps，当遇到无法序列化的对象时，orjson 会调用这个函数。
    """
    def _default(obj):
        # 如果对象是BaseModel的实例，返回其字典表示
        if isinstance(obj, BaseModel):
            return obj.dict()
        # 对于其他类型的对象，抛出TypeError异常
        # 对于无法序列化的对象，会明确抛出异常
        raise TypeError

    # 使用orjson.dumps进行序列化，并使用自定义的_default函数处理特殊对象
    return orjson.dumps(o, default=_default)


def purge_dir(d):
    """
    清理指定目录，删除空子目录或无效的符号链接。

    参数:
    d (str): 要清理的目录路径

    说明:
    - 如果目录不存在或不是一个目录，则直接返回
    - 遍历目录中的所有项目
    - 删除符合以下条件的子目录:
      1. 是一个无效的符号链接（指向不存在的位置）
      2. 是一个空目录
    - 删除操作可能会失败，但不会中断整个过程
    """
    # 检查目录是否存在且是一个目录
    if not os.path.exists(d) or not os.path.isdir(d):
        return
    
    # 遍历目录中的所有项目
    for name in os.listdir(d):
        subdir = os.path.join(d, name)
        try:
            # 检查是否是无效的符号链接或空目录
            if (os.path.islink(subdir) and not os.path.exists(subdir)) or (
                len(os.listdir(subdir)) == 0
            ):
                # 记录删除操作
                logger.info("Remove empty directory: %s", subdir)
                # 删除目录
                os.rmdir(subdir)
        except Exception:
            # 忽略删除过程中的任何错误
            pass

def parse_model_version(model_version: str, model_type: str) -> Tuple:
    """
    解析模型版本字符串，根据不同的模型类型返回相应的信息。

    参数:
    model_version (str): 模型版本字符串
    model_type (str): 模型类型

    返回:
    Tuple: 包含解析后的模型信息的元组
    返回元组而不是单个字符串，保持了返回值类型的一致性。
    可扩展性：
    如果将来需要为嵌入模型添加更多信息，可以轻松地在元组中添加新的元素。

    异常:
    ValueError: 当解析失败或不支持的模型类型时抛出
    
    
    
    所有类型的 model_version 都使用双破折号 -- 分隔不同部分
    
    model_type 可以是以下几种：
    - "LLM"（大型语言模型）
    - "embedding"（嵌入模型）
    - "rerank"（重排序模型）
    - "image"（图像模型）
    
    a. 对于 "LLM" 类型：
    格式：model_name--size--model_format--quantization
    例如：chatglm--6B--ggmlv3--q4_0
    
    - model_name: 模型名称
    - size: 模型大小，以 "B" 结尾（表示十亿参数）
    - model_format: 模型格式
    - quantization: 量化信息
    
    b. 对于 "embedding" 和 "rerank" 类型：
    格式：model_name
    例如：bge-large-en-v1.5
    
    c. 对于 "image" 类型：
    格式：model_name 或 model_name--additional_info
    例如：stable-diffusion-v1.5 或 stable-diffusion-v1.5--fp16
    """
    # 使用双破折号分割模型版本字符串
    results: List[str] = model_version.split("--")

    if model_type == "LLM":
        # LLM模型需要4个部分的信息
        if len(results) != 4:
            raise ValueError(
                f"LLM model_version parses failed! model_version: {model_version}"
            )
        model_name = results[0]  # 模型名称
        size = results[1]  # 模型大小
        # 确保大小以"B"结尾（表示十亿）
        if not size.endswith("B"):
            raise ValueError(f"Cannot parse model_size_in_billions: {size}")
        size = size.rstrip("B")  # 移除"B"后缀
        # 如果包含下划线，保持为字符串；否则转换为整数
        size_in_billions: Union[int, str] = size if "_" in size else int(size)
        model_format = results[2]  # 模型格式
        quantization = results[3]  # 量化信息
        return model_name, size_in_billions, model_format, quantization
    elif model_type == "embedding":
        # 嵌入模型至少需要一个部分
        assert len(results) > 0, "Embedding model_version parses failed!"
        return (results[0],)  # 返回嵌入模型名称
    elif model_type == "rerank":
        # 重排序模型至少需要一个部分
        assert len(results) > 0, "Rerank model_version parses failed!"
        return (results[0],)  # 返回重排序模型名称
    elif model_type == "image":
        # 图像模型需要1或2个部分
        assert 2 >= len(results) >= 1, "Image model_version parses failed!"
        return tuple(results)  # 返回图像模型信息
    else:
        # 不支持的模型类型
        raise ValueError(f"Not supported model_type: {model_type}")


def _get_nvidia_gpu_mem_info(gpu_id: int) -> Dict[str, float]:
    """
    获取指定NVIDIA GPU的内存信息。

    参数:
    gpu_id (int): GPU的索引ID

    返回:
    Dict[str, float]: 包含GPU内存总量、已使用量和空闲量的字典

    说明:
    - 使用NVIDIA管理库(NVML)获取GPU内存信息
    - 返回的内存数值单位为字节
    """
    from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

    # 获取指定GPU的设备句柄
    handler = nvmlDeviceGetHandleByIndex(gpu_id)
    # 获取GPU的内存信息
    mem_info = nvmlDeviceGetMemoryInfo(handler)
    # 返回包含总内存、已用内存和空闲内存的字典
    return {"total": mem_info.total, "used": mem_info.used, "free": mem_info.free}


def get_nvidia_gpu_info() -> Dict:
    """
    获取所有NVIDIA GPU的内存信息。

    返回:
    Dict: 包含所有GPU内存信息的字典，键为'gpu-{i}'，值为对应GPU的内存信息

    说明:
    - 使用NVIDIA管理库(NVML)获取所有GPU的内存信息
    - 如果无法初始化NVML（可能是由于缺少NVIDIA GPU或CUDA安装不正确），则返回空字典
    """
    try:
        # 初始化NVML
        nvmlInit()
        # 获取GPU数量
        device_count = nvmlDeviceGetCount()
        res = {}
        # 遍历每个GPU，获取其内存信息
        for i in range(device_count):
            res[f"gpu-{i}"] = _get_nvidia_gpu_mem_info(i)
        return res
    except:
        # TODO: 在此处添加日志
        # logger.debug(f"无法初始化NVML。可能是由于缺少NVIDIA GPU或CUDA安装不正确。")
        return {}
    finally:
        # 虽然 Python 的垃圾回收机制很强大，但对于管理外部资源
        # 特别是像 NVML 这样的系统级资源，显式的资源管理通常是更安全和可靠的方法。
        # 这种做法确保了资源的及时和可预测的释放，无论程序如何结束
        # 确保在结束时关闭NVML，即使发生异常
        try:
            nvmlShutdown()
        except:
            pass


def assign_replica_gpu(
    _replica_model_uid: str, gpu_idx: Union[int, List[int]]
) -> List[int]:
    """
    为复制模型分配GPU。

    参数:
    _replica_model_uid (str): 复制模型的唯一标识符
    gpu_idx (Union[int, List[int]]): 可用的GPU索引，可以是单个整数或整数列表

    返回:
    List[int]: 分配给该复制模型的GPU索引列表

    说明:
    - 解析复制模型的UID以获取模型信息
    - 根据复制ID和总复制数分配GPU
    - 如果gpu_idx是单个整数，将其转换为列表
    - 如果gpu_idx是非空列表，则按复制ID和总复制数进行切片
    - 如果gpu_idx为空或无效，则返回原始gpu_idx
    
    
    
    
     gpu_idx[rep_id::replica] 的含义：
    从索引 rep_id 开始，每隔 replica 个元素取一个。
    例如，如果 gpu_idx = [0, 1, 2, 3, 4, 5]，replica = 3，rep_id = 1：
    对于第二个副本（rep_id = 1），结果将是 [1, 4]
    对于第一个副本（rep_id = 0），结果将是 [0, 3]
    对于第三个副本（rep_id = 2），结果将是 [2, 5]
    
    
    这种分配方式确保了：
    每个副本都分配到至少一个 GPU。
    GPU 资源在副本之间均匀分布。
    如果 GPU 数量多于副本数，每个副本可能获得多个 GPU。

    """
    # 在分布式系统中，"replica" 通常指的是同一模型的多个副本。
    # 解析复制模型的UID
    model_uid, replica, rep_id = parse_replica_model_uid(_replica_model_uid)
    rep_id, replica = int(rep_id), int(replica)
    
    # 如果gpu_idx是单个整数，将其转换为列表
    if isinstance(gpu_idx, int):
        gpu_idx = [gpu_idx]
    
    # 如果gpu_idx是非空列表，则按复制ID和总复制数进行切片
    if isinstance(gpu_idx, list) and gpu_idx:
        return gpu_idx[rep_id::replica]
    
    # 如果gpu_idx为空或无效，则返回原始gpu_idx
    return gpu_idx
