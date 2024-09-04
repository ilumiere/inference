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
import time
import typing
from typing import TYPE_CHECKING, Any, Optional

import xoscar as xo

from ..constants import XINFERENCE_DEFAULT_LOG_FILE_NAME, XINFERENCE_LOG_DIR

if TYPE_CHECKING:
    from xoscar.backends.pool import MainActorPoolType

logger = logging.getLogger(__name__)

# mainly for k8s
XINFERENCE_POD_NAME_ENV_KEY = "XINFERENCE_POD_NAME"

class LoggerNameFilter(logging.Filter):
    # 定义一个日志过滤器类，继承自logging.Filter
    def filter(self, record):
        # 定义过滤方法，接受一个日志记录作为参数
        return record.name.startswith("xinference") or (
            # 返回True如果日志记录的名称以"xinference"开头
            record.name.startswith("uvicorn.error")
            # 或者日志记录的名称以"uvicorn.error"开头
            and record.getMessage().startswith("Uvicorn running on")
            # 并且日志消息以"Uvicorn running on"开头
        )
        # 如果以上条件满足，则保留该日志记录；否则过滤掉

def get_log_file(sub_dir: str):
    """
    获取日志文件路径。
    
    参数:
    sub_dir: str - 应包含时间戳的子目录名。
    
    返回:
    str - 完整的日志文件路径。
    """
    # 从环境变量中获取 pod 名称，如果不存在则为 None
    pod_name = os.environ.get(XINFERENCE_POD_NAME_ENV_KEY, None)
    
    # 如果 pod 名称存在，将其添加到子目录名中
    if pod_name is not None:
        sub_dir = sub_dir + "_" + pod_name
    
    # 构建完整的日志目录路径
    log_dir = os.path.join(XINFERENCE_LOG_DIR, sub_dir)
    
    # 创建新的日志目录，如果目录已存在则抛出异常
    # 这里应该每次都创建一个新目录，所以 `exist_ok=False`
    os.makedirs(log_dir, exist_ok=False)
    
    # 返回完整的日志文件路径
    return os.path.join(log_dir, XINFERENCE_DEFAULT_LOG_FILE_NAME)


def get_config_dict(
    log_level: str, log_file_path: str, log_backup_count: int, log_max_bytes: int
) -> dict:
    # 定义一个函数，用于生成日志配置字典
    
    # 对于Windows系统，需要对路径进行特殊处理
    # 对于windows而言os.name == 'nt', nt代表New Technology 是windows NT及后续版本的操作系统的内核名称
    log_file_path = (
        log_file_path.encode("unicode-escape").decode()
        if os.name == "nt"
        else log_file_path
    )
    
    # 将日志级别转换为大写
    log_level = log_level.upper()
    
    # 创建配置字典
    config_dict = {
        "version": 1,  # 配置版本
        "disable_existing_loggers": False,  # 不禁用现有的日志记录器
        "formatters": {
            "formatter": {
                "format": (
                    "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
                )
            },
        },  # 定义日志格式
        "filters": {
            "logger_name_filter": {
                "()": __name__ + ".LoggerNameFilter",
            },
        },  # 定义日志过滤器
        "handlers": {
            "stream_handler": {
                "class": "logging.StreamHandler",
                "formatter": "formatter",
                "level": log_level,
                "stream": "ext://sys.stderr",
                "filters": ["logger_name_filter"],
            },  # 定义流处理器，输出到标准错误
            "console_handler": {
                "class": "logging.StreamHandler",
                "formatter": "formatter",
                "level": log_level,
                "stream": "ext://sys.stderr",
            },  # 定义控制台处理器，输出到标准错误
            "file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "formatter",
                "level": log_level,
                "filename": log_file_path,
                "mode": "a",
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf8",
            },  # 定义文件处理器，支持日志轮转
        },
        "loggers": {
            "xinference": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },  # 配置xinference日志记录器
            "uvicorn": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },  # 配置uvicorn日志记录器
            "uvicorn.error": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },  # 配置uvicorn.error日志记录器
            "uvicorn.access": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },  # 配置uvicorn.access日志记录器
            "transformers": {
                "handlers": ["console_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },  # 配置transformers日志记录器
            "vllm": {
                "handlers": ["console_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            },  # 配置vllm日志记录器
        },
        "root": {
            "level": "WARN",
            "handlers": ["stream_handler", "file_handler"],
        },  # 配置根日志记录器
    }
    return config_dict  # 返回配置字典

async def create_worker_actor_pool(
    address: str, logging_conf: Optional[dict] = None
) -> "MainActorPoolType":
    # 定义一个异步函数，用于创建工作者actor池
    # 参数：address - 地址字符串，logging_conf - 可选的日志配置字典
    # 返回类型：MainActorPoolType

    # 根据操作系统类型选择子进程启动方法
    subprocess_start_method = "forkserver" if os.name != "nt" else "spawn"
    # 如果不是Windows系统(nt)，使用"forkserver"，否则使用"spawn"

    # 创建并返回actor池
    return await xo.create_actor_pool(
        address=address,  # 设置地址
        n_process=0,  # 进程数设为0，可能表示使用默认值或动态分配
        auto_recover="process",  # 设置自动恢复策略为"process"
        subprocess_start_method=subprocess_start_method,  # 设置子进程启动方法
        logging_conf={"dict": logging_conf},  # 设置日志配置
    )
    # 等待xo.create_actor_pool异步函数完成并返回结果

def health_check(address: str, max_attempts: int, sleep_interval: int = 3) -> bool:
    # 定义健康检查函数，接受地址、最大尝试次数和睡眠间隔作为参数
    async def health_check_internal():
        # 定义内部异步健康检查函数
        import time
        
        attempts = 0
        while attempts < max_attempts:
            # 循环尝试，直到达到最大尝试次数
            time.sleep(sleep_interval)
            # 在每次尝试之间休眠指定的时间
            try:
                from ..core.supervisor import SupervisorActor
                # 导入SupervisorActor
                
                supervisor_ref: xo.ActorRefType[SupervisorActor] = await xo.actor_ref(  # type: ignore
                    address=address, uid=SupervisorActor.uid()
                )
                # 获取SupervisorActor的引用
                # 通过调用 supervisor_ref.get_status()，可以获取系统的当前状态。这对于健康检查非常重要
                await supervisor_ref.get_status()
                # 尝试获取supervisor的状态
                return True
                # 如果成功获取状态，返回True表示健康
            except Exception as e:
                logger.debug(f"Error while checking cluster: {e}")
                # 如果出现异常，记录错误日志
            
            attempts += 1
            # 增加尝试次数
            if attempts < max_attempts:
                logger.debug(
                    f"Cluster not available, will try {max_attempts - attempts} more times"
                )
                # 如果还有剩余尝试次数，记录日志
        
        return False
        # 如果所有尝试都失败，返回False表示不健康

    import asyncio
    from ..isolation import Isolation
    # 导入所需的模块

    isolation = Isolation(asyncio.new_event_loop(), threaded=True)
    # 创建一个新的事件循环，并使用它初始化Isolation对象
    isolation.start()
    # 启动隔离环境
    available = isolation.call(health_check_internal())
    # 在隔离环境中调用health_check_internal函数
    isolation.stop()
    # 停止隔离环境
    return available
    # 返回健康检查结果


def get_timestamp_ms():
    # 获取当前时间戳（以毫秒为单位）的函数
    t = time.time()  # 获取当前时间（以秒为单位）
    return int(round(t * 1000))  # 将秒转换为毫秒，四舍五入后转为整数


# 当函数返回类型不固定，取决于输入参数时
# 替代方案：
# 使用 Union 类型：def func(arg: str) -> Union[int, float, bool, str, None]
# 使用 Any 类型：def func(arg: str) -> Any
# 这些方法可能更精确，但可能不如 @no_type_check 方便。
# 由于函数可能返回多种类型（int, float, bool, str, None），使用 @no_type_check 是合理的
@typing.no_type_check  # 装饰器，用于禁用类型检查
def handle_click_args_type(arg: str) -> Any:
    # 处理命令行参数类型的函数
    # 函数的主要作用是处理命令行参数的类型转换。
    # 它的设计目的是将字符串形式的命令行参数转换为适当的 Python 数据类型。
    if arg == "None":
        return None  # 如果参数是字符串"None"，返回Python的None
    if arg in ("True", "true"):
        return True  # 如果参数是"True"或"true"，返回布尔值True
    if arg in ("False", "false"):
        return False  # 如果参数是"False"或"false"，返回布尔值False
    try:
        result = int(arg)
        return result  # 尝试将参数转换为整数，如果成功则返回
    except:
        pass  # 如果转换失败，继续下一步

    try:
        result = float(arg)
        return result  # 尝试将参数转换为浮点数，如果成功则返回
    except:
        pass  # 如果转换失败，继续下一步

    return arg  # 如果以上所有转换都失败，则返回原始字符串
