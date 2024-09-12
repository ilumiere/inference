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
import os
from typing import Any, Optional

import xoscar as xo
from xoscar import MainActorPoolType

from ..core.worker import WorkerActor
from ..device_utils import gpu_count

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)



async def start_worker_components(
    address: str,
    supervisor_address: str,
    main_pool: MainActorPoolType,
    metrics_exporter_host: Optional[str],
    metrics_exporter_port: Optional[int],
):
    """
    异步启动工作者组件的函数。

    这个函数的主要用途是初始化GPU设备索引列表，并根据CUDA_VISIBLE_DEVICES环境变量或所有可用的GPU设备创建WorkerActor。

    :param address: 工作者的地址字符串。
    :param supervisor_address: 监督器的地址字符串。
    :param main_pool: 主Actor池类型，用于管理工作者Actor。
    :param metrics_exporter_host: 指标导出器的主机地址，非必填。
    :param metrics_exporter_port: 指标导出器的端口号，非必填。
    """
    # 初始化GPU设备索引列表
    gpu_device_indices = []
    
    # 获取CUDA_VISIBLE_DEVICES环境变量
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    if cuda_visible_devices is not None and cuda_visible_devices != "-1":
        # 如果CUDA_VISIBLE_DEVICES存在且不为-1，解析并添加到GPU设备索引列表
        gpu_device_indices.extend([int(i) for i in cuda_visible_devices.split(",")])
    else:
        # 否则，使用所有可用的GPU设备
        gpu_device_indices = list(range(gpu_count()))

    # 创建WorkerActor
    await xo.create_actor(
        WorkerActor,
        address=address,
        uid=WorkerActor.uid(),
        supervisor_address=supervisor_address,
        main_pool=main_pool,
        gpu_devices=gpu_device_indices,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
    )

async def _start_worker(
    address: str,
    supervisor_address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Any = None,
):
    """
    异步启动工作者组件的函数。

    这个函数的主要用途是启动工作者组件，包括创建工作者actor池、启动工作者组件和等待池完成。它接受工作者地址、监督器地址、指标导出器主机、指标导出器端口和日志配置作为参数。

    :param address: 工作者的地址字符串。
    :param supervisor_address: 监督器的地址字符串。
    :param metrics_exporter_host: 指标导出器的主机地址，非必填。
    :param metrics_exporter_port: 指标导出器的端口号，非必填。
    :param logging_conf: 日志配置，非必填。
    """
    # 导入创建工作者actor池的函数
    from .utils import create_worker_actor_pool

    # 创建工作者actor池
    pool = await create_worker_actor_pool(address=address, logging_conf=logging_conf)
    # 启动工作者组件
    await start_worker_components(
        address=address,
        supervisor_address=supervisor_address,
        main_pool=pool,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
    )
    # 等待池完成
    await pool.join()


def main(
    address: str,
    supervisor_address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[dict] = None,
):
    # 获取事件循环
    loop = asyncio.get_event_loop()
    # 创建启动工作者的任务
    task = loop.create_task(
        _start_worker(
            address,
            supervisor_address,
            metrics_exporter_host,
            metrics_exporter_port,
            logging_conf,
        )
    )

    try:
        # 运行任务直到完成
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        # 如果接收到键盘中断，取消任务
        task.cancel()
        # 优雅关闭：当任务被取消后，loop.run_until_complete(task) 允许任务有机会执行清理操作
        # 确保取消完成：task.cancel() 只是发送取消信号，但不会立即停止任务。
        # loop.run_until_complete(task) 会等待任务真正结束
        # 处理异常：如果任务在取消过程中抛出异常（除了 CancelledError），这个调用可以捕获并处理这些异常
        # 状态同步：确保事件循环和任务状态保持一致，避免潜在的竞态条件。
        loop.run_until_complete(task)
        # 获取任务异常以避免显示未处理的异常警告
        task.exception()
