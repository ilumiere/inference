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

import asyncio
import logging
import multiprocessing
import signal
import sys
from typing import Dict, Optional

import xoscar as xo
from xoscar.utils import get_next_port

from ..constants import (
    XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
)
from ..core.supervisor import SupervisorActor
from .utils import health_check

logger = logging.getLogger(__name__)


async def _start_supervisor(address: str, logging_conf: Optional[Dict] = None):
    # 配置日志
    logging.config.dictConfig(logging_conf)  # type: ignore

    pool = None
    try:
        # 创建actor池
        pool = await xo.create_actor_pool(
            address=address, n_process=0, logging_conf={"dict": logging_conf}
        )
        # 创建SupervisorActor
        await xo.create_actor(
            SupervisorActor, address=address, uid=SupervisorActor.uid()
        )
        # 等待池中的所有任务完成
        await pool.join()
    except asyncio.exceptions.CancelledError:
        # 如果收到取消信号
        if pool is not None:
            # 停止actor池
            await pool.stop()


def run(address: str, logging_conf: Optional[Dict] = None):
    # 定义SIGTERM信号处理函数
    def sigterm_handler(signum, frame):
        print("Received SIGTERM signal. Exiting...")
        sys.exit(0)  # 收到SIGTERM信号时退出程序

    # 注册SIGTERM信号处理函数
    signal.signal(signal.SIGTERM, sigterm_handler)

    # 获取事件循环
    loop = asyncio.get_event_loop()
    # 创建启动监督器的异步任务
    task = loop.create_task(
        _start_supervisor(address=address, logging_conf=logging_conf)
    )
    # 运行任务直到完成
    loop.run_until_complete(task)


def run_in_subprocess(
    address: str, logging_conf: Optional[Dict] = None
) -> multiprocessing.Process:
    # 创建一个新的进程来运行监督器
    # 参数:
    #   address: 监督器的地址
    #   logging_conf: 可选的日志配置字典
    # 返回:
    #   multiprocessing.Process 对象

    # 创建一个新的进程，目标函数为run，参数为address和logging_conf
    p = multiprocessing.Process(target=run, args=(address, logging_conf))
    
    # 启动新创建的进程
    p.start()
    
    # 返回进程对象
    return p


def main(
    host: str,
    port: int,
    supervisor_port: Optional[int],
    logging_conf: Optional[Dict] = None,
    auth_config_file: Optional[str] = None,
):
    # 定义主函数,接收主机、端口、监督器端口、日志配置和认证配置文件作为参数
    
    supervisor_address = f"{host}:{supervisor_port or get_next_port()}"
    # 构造监督器地址,如果未指定端口则获取下一个可用端口
    
    local_cluster = run_in_subprocess(supervisor_address, logging_conf)
    # 在子进程中运行本地集群,传入监督器地址和日志配置

    if not health_check(
        address=supervisor_address,
        max_attempts=XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
        sleep_interval=XINFERENCE_HEALTH_CHECK_INTERVAL,
    ):
        # 执行健康检查,检查监督器是否可用
        raise RuntimeError("Supervisor is not available after multiple attempts")
        # 如果多次尝试后监督器仍不可用,则抛出运行时错误

    try:
        from ..api import restful_api
        # 再supervisor中启动restapi服务
        # 导入restful_api模块
        # 这里会阻塞，这里会阻塞住，只有当restful_api.run 返回之后才能往下继续执行
        # 才会把集群停掉
        # 可以考虑优化如下: 
        # def signal_handler(signum, frame):
        #     # 触发 restful_api 的关闭逻辑
        #     # 这需要 restful_api 提供一个关闭方法
        #     restful_api.shutdown()
            

        # signal.signal(signal.SIGINT, signal_handler)
        # signal.signal(signal.SIGTERM, signal_handler)


        restful_api.run(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            logging_conf=logging_conf,
            auth_config_file=auth_config_file,
        )
        # 运行RESTful API,传入监督器地址、主机、端口、日志配置和认证配置文件
    finally:
        local_cluster.kill()
        # 无论try块是否成功执行,最后都会终止本地集群进程
