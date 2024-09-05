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
from .worker import start_worker_components

# 创建日志记录器
print(f"local.py 34: {__name__}")
logger = logging.getLogger(__name__)


async def _start_local_cluster(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
):
    # 导入创建工作者actor池的函数
    from .utils import create_worker_actor_pool

    # 配置日志
    print(f"local: logging_conf: {logging_conf}")
    logging.config.dictConfig(logging_conf)  # type: ignore

    pool = None
    try:
        # 创建工作者actor池
        pool = await create_worker_actor_pool(
            address=address, logging_conf=logging_conf
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
            metrics_exporter_host=metrics_exporter_host,
            metrics_exporter_port=metrics_exporter_port,
        )
        # 等待池完成
        await pool.join()
    except asyncio.CancelledError:
        # 如果发生取消错误，停止池
        if pool is not None:
            await pool.stop()


def run(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
):
    # 定义SIGTERM信号处理函数
    def sigterm_handler(signum, frame):
        sys.exit(0)

    # 注册SIGTERM信号处理函数
    signal.signal(signal.SIGTERM, sigterm_handler)

    # 获取事件循环
    loop = asyncio.get_event_loop()
    # 创建启动本地集群的任务
    task = loop.create_task(
        _start_local_cluster(
            address=address,
            metrics_exporter_host=metrics_exporter_host,
            metrics_exporter_port=metrics_exporter_port,
            logging_conf=logging_conf,
        )
    )
    # 运行任务直到完成
    loop.run_until_complete(task)


def run_in_subprocess(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
) -> multiprocessing.Process:
    # 创建子进程运行集群
    p = multiprocessing.Process(
        target=run,
        args=(address, metrics_exporter_host, metrics_exporter_port, logging_conf),
    )
    # 启动子进程
    p.start()
    return p


def main(
    host: str,
    port: int,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
    auth_config_file: Optional[str] = None,
):

    print(f"local: start main {host}, {port}, {metrics_exporter_host}, {metrics_exporter_port}, {logging_conf}, {auth_config_file}")
    # 生成监督者地址
    supervisor_address = f"{host}:{get_next_port()}"
    # 在子进程中运行本地集群
    local_cluster = run_in_subprocess(
        supervisor_address, metrics_exporter_host, metrics_exporter_port, logging_conf
    )

    # 检查集群健康状态
    if not health_check(
        address=supervisor_address,
        max_attempts=XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
        sleep_interval=XINFERENCE_HEALTH_CHECK_INTERVAL,
    ):
        raise RuntimeError("Cluster is not available after multiple attempts")

    try:
        # 导入并运行RESTful API
        from ..api import restful_api

        restful_api.run(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            logging_conf=logging_conf,
            auth_config_file=auth_config_file,
        )
    finally:
        # 终止本地集群进程
        local_cluster.kill()
