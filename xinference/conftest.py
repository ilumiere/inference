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
import asyncio  # 用于异步编程
import json  # 用于JSON数据处理
import logging  # 用于日志记录
import multiprocessing  # 用于多进程处理
import os  # 用于操作系统相关功能
import signal  # 用于处理信号
import sys  # 用于系统相关功能
import tempfile  # 用于创建临时文件
from typing import Dict, Optional  # 用于类型注解

import pytest  # 用于测试框架
import xoscar as xo  # 导入xoscar库

# 如果在 GitHub Actions 环境中运行，跳过健康检查
if os.environ.get("GITHUB_ACTIONS"):
    os.environ["XINFERENCE_DISABLE_HEALTH_CHECK"] = "1"

# 导入自定义模块和类
from .api.oauth2.types import AuthConfig, AuthStartupConfig, User  # 导入认证相关类型
from .constants import XINFERENCE_LOG_BACKUP_COUNT, XINFERENCE_LOG_MAX_BYTES  # 导入日志相关常量
from .core.supervisor import SupervisorActor  # 导入监督者Actor
from .deploy.utils import create_worker_actor_pool, get_log_file, get_timestamp_ms  # 导入工具函数
from .deploy.worker import start_worker_components  # 导入工作组件启动函数

# 定义测试日志配置
TEST_LOGGING_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "formatter": {
            "format": "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s",
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "formatter",
            "level": "DEBUG",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "xinference": {
            "handlers": ["stream_handler"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
    "root": {
        "level": "WARN",
        "handlers": ["stream_handler"],
    },
}

# 获取测试日志文件路径
TEST_LOG_FILE_PATH = get_log_file(f"test_{get_timestamp_ms()}")
if os.name == "nt":
    TEST_LOG_FILE_PATH = TEST_LOG_FILE_PATH.encode("unicode-escape").decode()

# 定义测试文件日志配置
TEST_FILE_LOGGING_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "formatter": {
            "format": "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "formatter",
            "level": "DEBUG",
            "stream": "ext://sys.stderr",
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "formatter",
            "level": "DEBUG",
            "filename": TEST_LOG_FILE_PATH,
            "mode": "a",
            "maxBytes": XINFERENCE_LOG_MAX_BYTES,
            "backupCount": XINFERENCE_LOG_BACKUP_COUNT,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "xinference": {
            "handlers": ["stream_handler", "file_handler"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
    "root": {
        "level": "WARN",
        "handlers": ["stream_handler", "file_handler"],
    },
}

# API 健康检查函数
def api_health_check(endpoint: str, max_attempts: int, sleep_interval: int = 3):
    import time
    import requests

    attempts = 0
    while attempts < max_attempts:
        time.sleep(sleep_interval)
        try:
            response = requests.get(f"{endpoint}/status")
            if response.status_code == 200:
                return True
        except requests.RequestException as e:
            print(f"Error while checking endpoint: {e}")

        attempts += 1
        if attempts < max_attempts:
            print(
                f"Endpoint not available, will try {max_attempts - attempts} more times"
            )

    return False

# 启动测试集群的异步函数
async def _start_test_cluster(
    address: str,
    logging_conf: Optional[Dict] = None,
):
    """
    启动测试集群的异步函数。

    参数:
    address (str): 集群地址
    logging_conf (Optional[Dict]): 日志配置字典

    功能:
    1. 配置日志
    2. 创建工作者actor池
    3. 创建监督者actor
    4. 启动工作者组件
    5. 等待池完成
    6. 处理取消异常
    """
    # 配置日志
    logging.config.dictConfig(logging_conf)  # type: ignore
    pool = None
    try:
        # 创建工作者actor池
        pool = await create_worker_actor_pool(
            address=f"test://{address}", logging_conf=logging_conf
        )
        # 创建监督者actor
        await xo.create_actor(
            SupervisorActor, address=address, uid=SupervisorActor.uid()
        )
        # 启动工作者组件
        await start_worker_components(
            address=address,
            supervisor_address=address,
            main_pool=pool,
            metrics_exporter_host=None,
            metrics_exporter_port=None,
        )
        # 等待池完成
        await pool.join()
    except asyncio.CancelledError:
        # 处理取消异常,停止池
        if pool is not None:
            await pool.stop()

# 运行测试集群的函数
def run_test_cluster(address: str, logging_conf: Optional[Dict] = None):
    def sigterm_handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        _start_test_cluster(address=address, logging_conf=logging_conf)
    )
    loop.run_until_complete(task)

# 在子进程中运行测试集群的函数
def run_test_cluster_in_subprocess(
    address: str, logging_conf: Optional[Dict] = None
) -> multiprocessing.Process:
    # 防止重新初始化 CUDA 错误
    multiprocessing.set_start_method(method="spawn", force=True)

    p = multiprocessing.Process(target=run_test_cluster, args=(address, logging_conf))
    p.start()
    return p

# 设置测试环境的 pytest fixture
@pytest.fixture
def setup():
    from .api.restful_api import run_in_subprocess as run_restful_api
    from .deploy.utils import health_check as cluster_health_check

    logging.config.dictConfig(TEST_LOGGING_CONF)  # type: ignore

    supervisor_addr = f"localhost:{xo.utils.get_next_port()}"
    local_cluster_proc = run_test_cluster_in_subprocess(
        supervisor_addr, TEST_LOGGING_CONF
    )
    if not cluster_health_check(supervisor_addr, max_attempts=10, sleep_interval=5):
        raise RuntimeError("Cluster is not available after multiple attempts")

    port = xo.utils.get_next_port()
    restful_api_proc = run_restful_api(
        supervisor_addr,
        host="localhost",
        port=port,
        logging_conf=TEST_LOGGING_CONF,
    )
    endpoint = f"http://localhost:{port}"
    if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
        raise RuntimeError("Endpoint is not available after multiple attempts")

    try:
        yield f"http://localhost:{port}", supervisor_addr
    finally:
        local_cluster_proc.kill()
        restful_api_proc.kill()

# 设置带文件日志的测试环境的 pytest fixture
@pytest.fixture
def setup_with_file_logging():
    from .api.restful_api import run_in_subprocess as run_restful_api
    from .deploy.utils import health_check as cluster_health_check

    logging.config.dictConfig(TEST_FILE_LOGGING_CONF)  # type: ignore

    supervisor_addr = f"localhost:{xo.utils.get_next_port()}"
    local_cluster_proc = run_test_cluster_in_subprocess(
        supervisor_addr, TEST_FILE_LOGGING_CONF
    )
    if not cluster_health_check(supervisor_addr, max_attempts=10, sleep_interval=5):
        raise RuntimeError("Cluster is not available after multiple attempts")

    port = xo.utils.get_next_port()
    restful_api_proc = run_restful_api(
        supervisor_addr,
        host="localhost",
        port=port,
        logging_conf=TEST_FILE_LOGGING_CONF,
    )
    endpoint = f"http://localhost:{port}"
    if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
        raise RuntimeError("Endpoint is not available after multiple attempts")

    try:
        yield f"http://localhost:{port}", supervisor_addr, TEST_LOG_FILE_PATH
    finally:
        local_cluster_proc.kill()
        restful_api_proc.kill()

# 设置带认证的测试环境的 pytest fixture
@pytest.fixture
def setup_with_auth():
    from .api.restful_api import run_in_subprocess as run_restful_api
    from .deploy.utils import health_check as cluster_health_check

    logging.config.dictConfig(TEST_LOGGING_CONF)  # type: ignore

    supervisor_addr = f"localhost:{xo.utils.get_next_port()}"
    local_cluster_proc = run_test_cluster_in_subprocess(
        supervisor_addr, TEST_LOGGING_CONF
    )
    if not cluster_health_check(supervisor_addr, max_attempts=10, sleep_interval=5):
        raise RuntimeError("Cluster is not available after multiple attempts")

    # 创建测试用户
    user1 = User(
        username="user1",
        password="pass1",
        permissions=["admin"],
        api_keys=["sk-3sjLbdwqAhhAF", "sk-0HCRO1rauFQDL"],
    )
    user2 = User(
        username="user2",
        password="pass2",
        permissions=["models:list"],
        api_keys=["sk-72tkvudyGLPMi"],
    )
    user3 = User(
        username="user3",
        password="pass3",
        permissions=["models:list", "models:read", "models:start"],
        api_keys=["sk-m6jEzEwmCc4iQ", "sk-ZOTLIY4gt9w11"],
    )
    
    # 创建认证配置
    auth_config = AuthConfig(
        algorithm="HS256",
        secret_key="09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",
        token_expire_in_minutes=30,
    )
    startup_config = AuthStartupConfig(
        auth_config=auth_config, user_config=[user1, user2, user3]
    )
    
    # 创建临时认证配置文件
    _, auth_file = tempfile.mkstemp()
    with open(auth_file, "w") as fd:
        fd.write(json.dumps(startup_config.dict()))

    port = xo.utils.get_next_port()
    restful_api_proc = run_restful_api(
        supervisor_addr,
        host="localhost",
        port=port,
        logging_conf=TEST_LOGGING_CONF,
        auth_config_file=auth_file,
    )
    endpoint = f"http://localhost:{port}"
    if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
        raise RuntimeError("Endpoint is not available after multiple attempts")

    try:
        yield f"http://localhost:{port}", supervisor_addr
    finally:
        local_cluster_proc.kill()
        restful_api_proc.kill()
        try:
            os.remove(auth_file)
        except:
            pass
