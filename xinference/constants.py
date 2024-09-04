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

import os  # 导入操作系统相关的功能模块
from pathlib import Path  # 导入处理文件路径的模块

# 定义环境变量名称常量
XINFERENCE_ENV_ENDPOINT = "XINFERENCE_ENDPOINT"  # Xinference 端点环境变量
XINFERENCE_ENV_MODEL_SRC = "XINFERENCE_MODEL_SRC"  # Xinference 模型源环境变量

# CSG 在这个上下文中可能指的是 "Cloud Service Gateway" 或类似的概念。但是没有足够的上下文信息来确定它的具体含义。

XINFERENCE_ENV_CSG_TOKEN = "XINFERENCE_CSG_TOKEN"  # Xinference CSG 令牌环境变量
XINFERENCE_ENV_CSG_ENDPOINT = "XINFERENCE_CSG_ENDPOINT"  # Xinference CSG 端点环境变量
XINFERENCE_ENV_HOME_PATH = "XINFERENCE_HOME"  # Xinference 主目录路径环境变量
XINFERENCE_ENV_HEALTH_CHECK_FAILURE_THRESHOLD = (  # Xinference 健康检查失败阈值环境变量
    "XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD"
)
XINFERENCE_ENV_HEALTH_CHECK_INTERVAL = "XINFERENCE_HEALTH_CHECK_INTERVAL"  # Xinference 健康检查间隔环境变量
XINFERENCE_ENV_HEALTH_CHECK_TIMEOUT = "XINFERENCE_HEALTH_CHECK_TIMEOUT"  # Xinference 健康检查超时环境变量
XINFERENCE_ENV_DISABLE_HEALTH_CHECK = "XINFERENCE_DISABLE_HEALTH_CHECK"  # Xinference 禁用健康检查环境变量
XINFERENCE_ENV_DISABLE_METRICS = "XINFERENCE_DISABLE_METRICS"  # Xinference 禁用指标环境变量
XINFERENCE_ENV_TRANSFORMERS_ENABLE_BATCHING = "XINFERENCE_TRANSFORMERS_ENABLE_BATCHING"  # Xinference Transformers 启用批处理环境变量


def get_xinference_home() -> str:  # 定义获取 Xinference 主目录路径的函数
    """
    获取 Xinference 的主目录路径
    如果未设置环境变量，则使用默认路径
    如果已设置环境变量，则同时更新 Hugging Face 和 ModelScope 的下载路径
    """
    home_path = os.environ.get(XINFERENCE_ENV_HOME_PATH)  # 尝试从环境变量获取主目录路径
    if home_path is None:  # 如果环境变量未设置
        home_path = str(Path.home() / ".xinference")  # 使用默认路径
    else:  # 如果环境变量已设置
        # 如果用户已设置 `XINFERENCE_HOME` 环境变量，更改 huggingface 和 modelscope 的默认下载路径
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(home_path, "huggingface")  # 设置 Hugging Face 缓存路径
        os.environ["MODELSCOPE_CACHE"] = os.path.join(home_path, "modelscope")  # 设置 ModelScope 缓存路径
    return home_path  # 返回主目录路径


# 定义 Xinference 相关的目录路径
XINFERENCE_HOME = get_xinference_home()  # 获取 Xinference 主目录路径
XINFERENCE_CACHE_DIR = os.path.join(XINFERENCE_HOME, "cache")  # Xinference 缓存目录
XINFERENCE_TENSORIZER_DIR = os.path.join(XINFERENCE_HOME, "tensorizer")  # Xinference tensorizer 目录
XINFERENCE_MODEL_DIR = os.path.join(XINFERENCE_HOME, "model")  # Xinference 模型目录
XINFERENCE_LOG_DIR = os.path.join(XINFERENCE_HOME, "logs")  # Xinference 日志目录
XINFERENCE_IMAGE_DIR = os.path.join(XINFERENCE_HOME, "image")  # Xinference 图像目录
XINFERENCE_VIDEO_DIR = os.path.join(XINFERENCE_HOME, "video")  # Xinference 视频目录
XINFERENCE_AUTH_DIR = os.path.join(XINFERENCE_HOME, "auth")  # Xinference 认证目录
XINFERENCE_CSG_ENDPOINT = str(  # Xinference CSG 端点
    os.environ.get(XINFERENCE_ENV_CSG_ENDPOINT, "https://hub-stg.opencsg.com/")  # 从环境变量获取或使用默认值
)

# 定义 Xinference 的默认配置
XINFERENCE_DEFAULT_LOCAL_HOST = "127.0.0.1"  # Xinference 默认本地主机地址
XINFERENCE_DEFAULT_DISTRIBUTED_HOST = "0.0.0.0"  # Xinference 默认分布式主机地址
XINFERENCE_DEFAULT_ENDPOINT_PORT = 9997  # Xinference 默认端点端口
XINFERENCE_DEFAULT_LOG_FILE_NAME = "xinference.log"  # Xinference 默认日志文件名
XINFERENCE_LOG_MAX_BYTES = 100 * 1024 * 1024  # 日志文件最大大小：100MB
XINFERENCE_LOG_BACKUP_COUNT = 30  # 日志文件备份数量

# 从环境变量获取健康检查相关配置，如果未设置则使用默认值
XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD = int(  # Xinference 健康检查失败阈值
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_FAILURE_THRESHOLD, 5)  # 从环境变量获取或使用默认值 5
)
XINFERENCE_HEALTH_CHECK_INTERVAL = int(  # Xinference 健康检查间隔
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_INTERVAL, 5)  # 从环境变量获取或使用默认值 5
)
XINFERENCE_HEALTH_CHECK_TIMEOUT = int(  # Xinference 健康检查超时
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_TIMEOUT, 10)  # 从环境变量获取或使用默认值 10
)
XINFERENCE_DISABLE_HEALTH_CHECK = bool(  # Xinference 是否禁用健康检查
    int(os.environ.get(XINFERENCE_ENV_DISABLE_HEALTH_CHECK, 0))  # 从环境变量获取或使用默认值 False
)
XINFERENCE_DISABLE_METRICS = bool(  # Xinference 是否禁用指标
    int(os.environ.get(XINFERENCE_ENV_DISABLE_METRICS, 0))  # 从环境变量获取或使用默认值 False
)
XINFERENCE_TRANSFORMERS_ENABLE_BATCHING = bool(  # Xinference Transformers 是否启用批处理
    int(os.environ.get(XINFERENCE_ENV_TRANSFORMERS_ENABLE_BATCHING, 0))  # 从环境变量获取或使用默认值 False
)

print(f"XINFERENCE_HOME: {XINFERENCE_HOME}")