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
import os
import platform
import queue
import shutil
import signal
import threading
import time
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import xoscar as xo
from async_timeout import timeout
from xoscar import MainActorPoolType

from ..constants import (
    XINFERENCE_CACHE_DIR,
    XINFERENCE_DISABLE_HEALTH_CHECK,
    XINFERENCE_DISABLE_METRICS,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
)
from ..core.model import ModelActor
from ..core.status_guard import LaunchStatus
from ..device_utils import get_available_device_env_name, gpu_count
from ..model.core import ModelDescription, create_model_instance
from ..types import PeftModelConfig
from .cache_tracker import CacheTrackerActor
from .event import Event, EventCollectorActor, EventType
from .metrics import launch_metrics_export_server, record_metrics
from .resource import gather_node_info
from .status_guard import StatusGuardActor
from .utils import log_async, log_sync, parse_replica_model_uid, purge_dir

logger = getLogger(__name__)


# 定义模型 Actor 自动恢复的限制次数
MODEL_ACTOR_AUTO_RECOVER_LIMIT: Optional[int]

# 从环境变量获取自动恢复限制次数
_MODEL_ACTOR_AUTO_RECOVER_LIMIT = os.getenv("XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT")

# 如果环境变量存在，将其转换为整数
if _MODEL_ACTOR_AUTO_RECOVER_LIMIT is not None:
    MODEL_ACTOR_AUTO_RECOVER_LIMIT = int(_MODEL_ACTOR_AUTO_RECOVER_LIMIT)
else:
    # 如果环境变量不存在，设置为 None（表示无限制）
    MODEL_ACTOR_AUTO_RECOVER_LIMIT = None


class WorkerActor(xo.StatelessActor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        gpu_devices: List[int],
        metrics_exporter_host: Optional[str] = None,
        metrics_exporter_port: Optional[int] = None,
    ):
        super().__init__()
        # 静态属性
        self._total_gpu_devices = gpu_devices  # 可用的GPU设备列表
        self._supervisor_address = supervisor_address  # 监督者地址
        self._supervisor_ref: Optional[xo.ActorRefType] = None  # 监督者引用
        self._main_pool = main_pool  # 主Actor池
        self._main_pool.recover_sub_pool = self.recover_sub_pool  # 设置子池恢复方法
        self._status_guard_ref: xo.ActorRefType["StatusGuardActor"] = (  # type: ignore
            None
        )  # 状态守卫引用
        self._event_collector_ref: xo.ActorRefType[  # type: ignore
            EventCollectorActor
        ] = None  # 事件收集器引用
        self._cache_tracker_ref: xo.ActorRefType[CacheTrackerActor] = (  # type: ignore
            None
        )  # 缓存追踪器引用

        # 内部状态
        # 模型启动过程中的临时占位符
        self._model_uid_launching_guard: Dict[str, bool] = {}
        # 模型启动后维护的属性
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}  # 模型UID到模型Actor的映射
        self._model_uid_to_model_spec: Dict[str, ModelDescription] = {}  # 模型UID到模型描述的映射
        self._gpu_to_model_uid: Dict[int, str] = {}  # GPU索引到模型UID的映射
        self._gpu_to_embedding_model_uids: Dict[int, Set[str]] = defaultdict(set)  # GPU索引到嵌入模型UID集合的映射
        # 字典结构: gpu_index: {(replica_model_uid, model_type)}
        self._user_specified_gpu_to_model_uids: Dict[
            int, Set[Tuple[str, str]]
        ] = defaultdict(set)  # 用户指定的GPU到模型UID和类型的映射
        self._model_uid_to_addr: Dict[str, str] = {}  # 模型UID到地址的映射
        self._model_uid_to_recover_count: Dict[str, Optional[int]] = {}  # 模型UID到恢复次数的映射
        self._model_uid_to_launch_args: Dict[str, Dict] = {}  # 模型UID到启动参数的映射

        # 检查是否禁用指标
        if XINFERENCE_DISABLE_METRICS:
            logger.info(
                "Worker metrics is disabled due to the environment XINFERENCE_DISABLE_METRICS=1"
            )
        elif metrics_exporter_host is not None or metrics_exporter_port is not None:
            # 启动指标导出服务器
            logger.info(
                f"Starting metrics export server at {metrics_exporter_host}:{metrics_exporter_port}"
            )
            q: queue.Queue = queue.Queue()
            self._metrics_thread = threading.Thread(
                name="Metrics Export Server",
                target=launch_metrics_export_server,
                args=(q, metrics_exporter_host, metrics_exporter_port),
                daemon=True,
            )
            self._metrics_thread.start()
            logger.info("Checking metrics export server...")
            while self._metrics_thread.is_alive():
                try:
                    host, port = q.get(block=False)[:2]
                    logger.info(f"Metrics server is started at: http://{host}:{port}")
                    break
                except queue.Empty:
                    pass
            else:
                raise Exception("Metrics server thread exit.")

        self._lock = asyncio.Lock()  # 异步锁，用于并发控制

    async def recover_sub_pool(self, address):
        # 记录进程崩溃的警告日志
        logger.warning("Process %s is down.", address)
        # Xoscar 不会从 sub_processes 中移除地址，所以我们需要手动移除
        try:
            await self._main_pool.remove_sub_pool(address)
        except Exception:
            pass
        # 遍历所有模型，检查是否有需要恢复的模型
        for model_uid, addr in self._model_uid_to_addr.items():
            if addr == address:
                # 获取模型的启动参数
                launch_args = self._model_uid_to_launch_args.get(model_uid)
                if launch_args is None:
                    # 如果没有启动参数，说明模型在启动过程中崩溃，不进行恢复
                    logger.warning(
                        "Not recreate model because the it is down during launch."
                    )
                else:
                    # 获取模型的恢复次数
                    recover_count = self._model_uid_to_recover_count.get(model_uid)
                    try:
                        # 尝试终止已崩溃的模型
                        await self.terminate_model(model_uid, is_model_die=True)
                    except Exception:
                        pass
                    if recover_count is not None:
                        if recover_count > 0:
                            # 如果还有恢复次数，进行模型重建
                            logger.warning(
                                "Recreating model actor %s, remain %s times ...",
                                model_uid,
                                recover_count - 1,
                            )
                            # 解析模型UID
                            event_model_uid, _, __ = parse_replica_model_uid(model_uid)
                            try:
                                # 如果存在事件收集器，报告模型重建事件
                                if self._event_collector_ref is not None:
                                    await self._event_collector_ref.report_event(
                                        event_model_uid,
                                        Event(
                                            event_type=EventType.WARNING,
                                            event_ts=int(time.time()),
                                            event_content="Recreate model",
                                        ),
                                    )
                            except Exception as e:
                                # 报告回调错误可以记录并忽略，不应中断进程
                                logger.error("report_event error: %s" % (e))
                            finally:
                                del event_model_uid

                            # 更新剩余恢复次数
                            self._model_uid_to_recover_count[model_uid] = (
                                recover_count - 1
                            )
                            # 重新启动内置模型
                            await self.launch_builtin_model(**launch_args)
                        else:
                            # 如果恢复次数用完，停止重建
                            logger.warning("Stop recreating model actor.")
                    else:
                        # 如果恢复次数为None，无限次重建模型
                        logger.warning("Recreating model actor %s ...", model_uid)
                        await self.launch_builtin_model(**launch_args)
                break

    @classmethod
    def uid(cls) -> str:
        return "worker"

    async def __post_create__(self):
        # 导入各种模型相关的模块和函数
        from ..model.audio import (
            CustomAudioModelFamilyV1,
            generate_audio_description,
            register_audio,
            unregister_audio,
        )
        from ..model.embedding import (
            CustomEmbeddingModelSpec,
            generate_embedding_description,
            register_embedding,
            unregister_embedding,
        )
        from ..model.flexible import (
            FlexibleModelSpec,
            generate_flexible_model_description,
            register_flexible_model,
            unregister_flexible_model,
        )
        from ..model.image import (
            CustomImageModelFamilyV1,
            generate_image_description,
            register_image,
            unregister_image,
        )
        from ..model.llm import (
            CustomLLMFamilyV1,
            generate_llm_description,
            register_llm,
            unregister_llm,
        )
        from ..model.rerank import (
            CustomRerankModelSpec,
            generate_rerank_description,
            register_rerank,
            unregister_rerank,
        )

        # 定义自定义注册类型到相应类和函数的映射
        self._custom_register_type_to_cls: Dict[str, Tuple] = {  # type: ignore
            "LLM": (
                CustomLLMFamilyV1,
                register_llm,
                unregister_llm,
                generate_llm_description,
            ),
            "embedding": (
                CustomEmbeddingModelSpec,
                register_embedding,
                unregister_embedding,
                generate_embedding_description,
            ),
            "rerank": (
                CustomRerankModelSpec,
                register_rerank,
                unregister_rerank,
                generate_rerank_description,
            ),
            "image": (
                CustomImageModelFamilyV1,
                register_image,
                unregister_image,
                generate_image_description,
            ),
            "audio": (
                CustomAudioModelFamilyV1,
                register_audio,
                unregister_audio,
                generate_audio_description,
            ),
            "flexible": (
                FlexibleModelSpec,
                register_flexible_model,
                unregister_flexible_model,
                generate_flexible_model_description,
            ),
        }

        # 清理缓存目录
        logger.info("Purge cache directory: %s", XINFERENCE_CACHE_DIR)
        purge_dir(XINFERENCE_CACHE_DIR)

        # 尝试连接到supervisor
        try:
            await self.get_supervisor_ref(add_worker=True)
        except Exception as e:
            # 如果supervisor宕机，不要使worker崩溃，稍后自动重连
            logger.error(f"cannot connect to supervisor {e}")

        # 如果未禁用健康检查
        if not XINFERENCE_DISABLE_HEALTH_CHECK:
            from ..isolation import Isolation

            # 在专用线程中运行_periodical_report_status()
            self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            self._isolation.start()
            asyncio.run_coroutine_threadsafe(
                self._periodical_report_status(), loop=self._isolation.loop
            )
        logger.info(f"Xinference worker {self.address} started")

        # Windows does not have signal handler
        if os.name != "nt":

            # 定义信号处理函数
            async def signal_handler():
                try:
                    supervisor_ref = await self.get_supervisor_ref(add_worker=False)
                    await supervisor_ref.remove_worker(self.address)
                except Exception as e:
                    # 忽略RPC错误，因为我们正在退出
                    logger.exception("remove worker rpc error: %s", e)
                os._exit(0)

            # 添加SIGINT信号处理器
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGINT, lambda: asyncio.create_task(signal_handler())
            )

    async def __pre_destroy__(self):
        self._isolation.stop()

    async def trigger_exit(self) -> bool:
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception as e:
            logger.info(f"trigger exit error: {e}")
            return False
        return True

    async def get_supervisor_ref(self, add_worker: bool = True) -> xo.ActorRefType:
        """
        Try connect to supervisor and return ActorRef. Raise exception on error
        Params:
            add_worker: By default will call supervisor.add_worker after first connect
        """
        from .supervisor import SupervisorActor

        # 如果已经有supervisor引用,直接返回
        if self._supervisor_ref is not None:
            return self._supervisor_ref
        
        # 获取supervisor的ActorRef
        supervisor_ref = await xo.actor_ref(  # type: ignore
            address=self._supervisor_address, uid=SupervisorActor.uid()
        )
        # Prevent concurrent operations leads to double initialization, check again.
        if self._supervisor_ref is not None:
            return self._supervisor_ref
        
        self._supervisor_ref = supervisor_ref
        
        # 如果是新启动(或重启)且没有模型,通知supervisor
        if add_worker and len(self._model_uid_to_model) == 0:
            # Newly started (or restarted), has no model, notify supervisor
            await self._supervisor_ref.add_worker(self.address)
            logger.info("Connected to supervisor as a fresh worker")

        # 获取其他必要的ActorRef
        self._status_guard_ref = await xo.actor_ref(
            address=self._supervisor_address, uid=StatusGuardActor.uid()
        )
        self._event_collector_ref = await xo.actor_ref(
            address=self._supervisor_address, uid=EventCollectorActor.uid()
        )
        self._cache_tracker_ref = await xo.actor_ref(
            address=self._supervisor_address, uid=CacheTrackerActor.uid()
        )
        # cache_tracker is on supervisor
        from ..model.audio import get_audio_model_descriptions
        from ..model.embedding import get_embedding_model_descriptions
        from ..model.flexible import get_flexible_model_descriptions
        from ..model.image import get_image_model_descriptions
        from ..model.llm import get_llm_model_descriptions
        from ..model.rerank import get_rerank_model_descriptions

        # 记录模型版本信息
        model_version_infos: Dict[str, List[Dict]] = {}  # type: ignore
        model_version_infos.update(get_llm_model_descriptions())
        model_version_infos.update(get_embedding_model_descriptions())
        model_version_infos.update(get_rerank_model_descriptions())
        model_version_infos.update(get_image_model_descriptions())
        model_version_infos.update(get_audio_model_descriptions())
        model_version_infos.update(get_flexible_model_descriptions())
        
        # 向缓存追踪器记录模型版本信息
        await self._cache_tracker_ref.record_model_version(
            model_version_infos, self.address
        )
        
        return self._supervisor_ref

    @staticmethod
    def get_devices_count():
        from ..device_utils import gpu_count

        return gpu_count()

    @log_sync(logger=logger)
    def get_model_count(self) -> int:
        return len(self._model_uid_to_model)

    async def is_model_vllm_backend(self, model_uid: str) -> bool:
        _model_uid, _, _ = parse_replica_model_uid(model_uid)
        supervisor_ref = await self.get_supervisor_ref()
        model_ref = await supervisor_ref.get_model(_model_uid)
        return await model_ref.is_vllm_backend()

    async def allocate_devices_for_embedding(self, model_uid: str) -> int:
        """
        为嵌入模型分配设备。我们假设嵌入模型只占用1个GPU槽位。
        """
        candidates = []
        for _dev in self._total_gpu_devices:
            if (
                _dev not in self._gpu_to_model_uid
                and _dev not in self._user_specified_gpu_to_model_uids
            ):  # 该设备上没有可能的vllm模型，将其添加到候选列表
                candidates.append(_dev)
            else:  # 需要判断该设备上是否有vllm模型
                has_vllm_model = False
                
                # 检查设备是否已分配给模型
                if _dev in self._gpu_to_model_uid:
                    # 获取分配给该设备的模型UID
                    existing_model_uid = self._gpu_to_model_uid[_dev]
                    # 检查该模型是否使用vllm后端
                    has_vllm_model = await self.is_model_vllm_backend(
                        existing_model_uid
                    )
                
                # 如果设备上没有vllm模型，且设备在用户指定的GPU-模型映射中
                if (
                    not has_vllm_model
                    and _dev in self._user_specified_gpu_to_model_uids
                ):
                    # 遍历该设备上用户指定的所有模型
                    for rep_uid, _ in self._user_specified_gpu_to_model_uids[_dev]:
                        # 检查每个模型是否使用vllm后端
                        has_vllm_model = await self.is_model_vllm_backend(rep_uid)
                        if has_vllm_model:
                            break  # 如果找到vllm模型，立即退出循环
                
                # 如果设备上没有vllm模型，将其添加到候选列表
                if not has_vllm_model:
                    candidates.append(_dev)

        if len(candidates) == 0:
            raise RuntimeError(
                "没有找到可用的槽位来加载嵌入模型。"
                "我们建议先启动嵌入模型，然后再启动LLM模型。"
            )

        device, min_cnt = -1, -1
        # 在所有候选设备中选择已有模型最少的设备
        for _dev in candidates:
            existing_cnt = 0
            # 检查设备是否已分配给嵌入模型
            if _dev in self._gpu_to_embedding_model_uids:
                existing_cnt += len(self._gpu_to_embedding_model_uids[_dev])
            # 检查设备是否已分配给非嵌入模型
            if _dev in self._gpu_to_model_uid:
                existing_cnt += 1
            # 检查设备是否有用户指定的模型
            if _dev in self._user_specified_gpu_to_model_uids:
                existing_cnt += len(self._user_specified_gpu_to_model_uids[_dev])
            
            # 如果当前设备上已有的模型数量最少，则选择该设备
            if min_cnt == -1 or existing_cnt < min_cnt:
                device, min_cnt = _dev, existing_cnt

        # 将选定的设备分配给嵌入模型
        self._gpu_to_embedding_model_uids[device].add(model_uid)
        return device
    def allocate_devices(self, model_uid: str, n_gpu: int) -> List[int]:
        """
        为非嵌入模型分配设备。
        """
        # 初始化用户指定的已分配设备集合
        user_specified_allocated_devices: Set[int] = set()
        
        # 遍历用户指定的GPU-模型映射
        for dev, model_infos in self._user_specified_gpu_to_model_uids.items():
            allocated_non_embedding_rerank_models = False
            # 检查每个设备上的模型类型
            for _, model_type in model_infos:
                # 判断是否为非嵌入和非重排序模型
                allocated_non_embedding_rerank_models = model_type not in [
                    "embedding",
                    "rerank",
                ]
                if allocated_non_embedding_rerank_models:
                    break
            # 如果设备上有非嵌入和非重排序模型，将其添加到已分配设备集合
            if allocated_non_embedding_rerank_models:
                user_specified_allocated_devices.add(dev)
        
        # 合并已分配给模型的设备和用户指定的已分配设备
        allocated_devices = set(self._gpu_to_model_uid.keys()).union(
            user_specified_allocated_devices
        )
        
        # 检查是否有足够的可用设备
        if n_gpu > len(self._total_gpu_devices) - len(allocated_devices):
            raise RuntimeError("No available slot found for the model")

        # 从总设备中选择可用的设备
        devices: List[int] = [
            dev
            for dev in self._total_gpu_devices
            if dev not in self._gpu_to_model_uid
            and dev not in user_specified_allocated_devices
        ][:n_gpu]
        
        # 将选定的设备分配给模型
        for dev in devices:
            self._gpu_to_model_uid[int(dev)] = model_uid

        # 返回排序后的设备列表
        return sorted(devices)

    async def allocate_devices_with_gpu_idx(
        self, model_uid: str, model_type: str, gpu_idx: List[int]
    ) -> List[int]:
        """
        When user specifies the gpu_idx, allocate models on user-specified GPUs whenever possible
        """
        # must be subset of total devices visible to this worker
        if not set(gpu_idx) <= set(self._total_gpu_devices):
            raise ValueError(
                f"Worker {self.address} cannot use the GPUs with these indexes: {gpu_idx}. "
                f"Worker {self.address} can only see these GPUs: {self._total_gpu_devices}."
            )
        # currently just report a warning log when there are already models on these GPUs
        for idx in gpu_idx:
            existing_model_uids = []
            if idx in self._gpu_to_model_uid:
                rep_uid = self._gpu_to_model_uid[idx]
                is_vllm_model = await self.is_model_vllm_backend(rep_uid)
                if is_vllm_model:
                    raise RuntimeError(
                        f"GPU index {idx} has been occupied with a vLLM model: {rep_uid}, "
                        f"therefore cannot allocate GPU memory for a new model."
                    )
                existing_model_uids.append(rep_uid)
            if idx in self._gpu_to_embedding_model_uids:
                existing_model_uids.extend(self._gpu_to_embedding_model_uids[idx])
            # If user has run the vLLM model on the GPU that was forced to be specified,
            # it is not possible to force this GPU to be allocated again
            if idx in self._user_specified_gpu_to_model_uids:
                for rep_uid, _ in self._user_specified_gpu_to_model_uids[idx]:
                    is_vllm_model = await self.is_model_vllm_backend(rep_uid)
                    if is_vllm_model:
                        raise RuntimeError(
                            f"User specified GPU index {idx} has been occupied with a vLLM model: {rep_uid}, "
                            f"therefore cannot allocate GPU memory for a new model."
                        )

            if existing_model_uids:
                logger.warning(
                    f"WARNING!!! GPU index {idx} has been occupied "
                    f"with these models on it: {existing_model_uids}"
                )

        for idx in gpu_idx:
            self._user_specified_gpu_to_model_uids[idx].add((model_uid, model_type))
        return sorted(gpu_idx)

    def release_devices(self, model_uid: str):
        # 释放被指定模型占用的GPU设备
        
        # 释放普通模型占用的GPU设备
        devices = [
            dev
            for dev in self._gpu_to_model_uid
            if self._gpu_to_model_uid[dev] == model_uid
        ]
        for dev in devices:
            del self._gpu_to_model_uid[dev]

        # 释放嵌入模型占用的GPU设备
        for dev in self._gpu_to_embedding_model_uids:
            if model_uid in self._gpu_to_embedding_model_uids[dev]:
                self._gpu_to_embedding_model_uids[dev].remove(model_uid)

        # check user-specified slots
        for dev in self._user_specified_gpu_to_model_uids:
            # 找出与当前模型UID匹配的所有模型信息
            model_infos = list(
                filter(
                    lambda x: x[0] == model_uid,
                    self._user_specified_gpu_to_model_uids[dev],
                )
            )
            # 从用户指定的GPU-模型映射中移除这些模型信息
            for model_info in model_infos:
                self._user_specified_gpu_to_model_uids[dev].remove(model_info)

    async def _create_subpool(
        self,
        model_uid: str,
        model_type: Optional[str] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
        gpu_idx: Optional[List[int]] = None,
    ) -> Tuple[str, List[str]]:
        # 初始化环境变量和设备列表
        env = {}
        devices = []
        # 获取可用的设备环境变量名称
        env_name = get_available_device_env_name() or "CUDA_VISIBLE_DEVICES"
        
        if gpu_idx is None:
            # 如果没有指定GPU索引
            if isinstance(n_gpu, int) or (n_gpu == "auto" and gpu_count() > 0):
                # Currently, n_gpu=auto means using 1 GPU
                gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
                # 根据模型类型分配设备
                devices = (
                    [await self.allocate_devices_for_embedding(model_uid)]
                    if model_type in ["embedding", "rerank"]
                    else self.allocate_devices(model_uid=model_uid, n_gpu=gpu_cnt)
                )
                # 设置环境变量
                env[env_name] = ",".join([str(dev) for dev in devices])
                logger.debug(f"GPU selected: {devices} for model {model_uid}")
            if n_gpu is None:
                # 如果不使用GPU，设置环境变量为-1
                env[env_name] = "-1"
                logger.debug(f"GPU disabled for model {model_uid}")
        else:
            # 如果指定了GPU索引
            assert isinstance(gpu_idx, list)
            # 使用指定的GPU索引分配设备
            devices = await self.allocate_devices_with_gpu_idx(
                model_uid, model_type, gpu_idx  # type: ignore
            )
            env[env_name] = ",".join([str(dev) for dev in devices])

        # 根据操作系统选择子进程启动方法
        if os.name != "nt" and platform.system() != "Darwin":
            # Linux系统使用forkserver
            start_method = "forkserver"
        else:
            # Windows和macOS系统使用spawn
            start_method = "spawn"
        
        # 创建子进程池并获取地址
        subpool_address = await self._main_pool.append_sub_pool(
            env=env, start_method=start_method
        )
        # 返回子进程池地址和分配的设备列表
        return subpool_address, [str(dev) for dev in devices]

    def _check_model_is_valid(self, model_name: str, model_format: Optional[str]):
        # baichuan-base and baichuan-chat depend on `cpm_kernels` module,
        # but `cpm_kernels` cannot run on Darwin system.
        if platform.system() == "Darwin" and model_format == "pytorch":
            if "baichuan" in model_name:
                raise ValueError(f"{model_name} model can't run on Darwin system.")

    @log_sync(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        # TODO: 集中管理模型注册
        if model_type in self._custom_register_type_to_cls:
            # 从自定义注册类型中获取相关函数和类
            (
                model_spec_cls,  # 模型规格类
                register_fn,     # 注册函数
                unregister_fn,   # 注销函数
                generate_fn,     # 生成函数
            ) = self._custom_register_type_to_cls[model_type]
            
            # 解析模型规格
            model_spec = model_spec_cls.parse_raw(model)
            
            try:
                # 注册模型
                register_fn(model_spec, persist)
                
                # 记录模型版本信息
                await self._cache_tracker_ref.record_model_version(
                    generate_fn(model_spec), self.address
                )
            except ValueError as e:
                # 如果发生ValueError，直接抛出
                raise e
            except Exception as e:
                # 如果发生其他异常，尝试注销模型并重新抛出异常
                unregister_fn(model_spec.model_name, raise_error=False)
                raise e
        else:
            # 如果模型类型不支持，抛出ValueError
            raise ValueError(f"不支持的模型类型: {model_type}")

    @log_sync(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        # TODO: centralized model registrations
        if model_type in self._custom_register_type_to_cls:
            _, _, unregister_fn, _ = self._custom_register_type_to_cls[model_type]
            unregister_fn(model_name, False)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def list_model_registrations(
        self, model_type: str, detailed: bool = False
    ) -> List[Dict[str, Any]]:
        def sort_helper(item):
            assert isinstance(item["model_name"], str)
            return item.get("model_name").lower()

        if model_type == "LLM":
            from ..model.llm import get_user_defined_llm_families

            ret = []

            for family in get_user_defined_llm_families():
                ret.append({"model_name": family.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "embedding":
            from ..model.embedding.custom import get_user_defined_embeddings

            ret = []

            for model_spec in get_user_defined_embeddings():
                ret.append({"model_name": model_spec.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "image":
            from ..model.image.custom import get_user_defined_images

            ret = []

            for model_spec in get_user_defined_images():
                ret.append({"model_name": model_spec.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "audio":
            from ..model.audio.custom import get_user_defined_audios

            ret = []

            for model_spec in get_user_defined_audios():
                ret.append({"model_name": model_spec.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "video":
            return []
        elif model_type == "rerank":
            from ..model.rerank.custom import get_user_defined_reranks

            ret = []

            for model_spec in get_user_defined_reranks():
                ret.append({"model_name": model_spec.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def get_model_registration(self, model_type: str, model_name: str) -> Any:
        if model_type == "LLM":
            from ..model.llm import get_user_defined_llm_families

            for f in get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f
        elif model_type == "embedding":
            from ..model.embedding.custom import get_user_defined_embeddings

            for f in get_user_defined_embeddings():
                if f.model_name == model_name:
                    return f
        elif model_type == "image":
            from ..model.image.custom import get_user_defined_images

            for f in get_user_defined_images():
                if f.model_name == model_name:
                    return f
        elif model_type == "audio":
            from ..model.audio.custom import get_user_defined_audios

            for f in get_user_defined_audios():
                if f.model_name == model_name:
                    return f
        elif model_type == "video":
            return None
        elif model_type == "rerank":
            from ..model.rerank.custom import get_user_defined_reranks

            for f in get_user_defined_reranks():
                if f.model_name == model_name:
                    return f
        return None

    @log_async(logger=logger)
    async def query_engines_by_model_name(self, model_name: str):
        from copy import deepcopy

        from ..model.llm.llm_family import LLM_ENGINES

        if model_name not in LLM_ENGINES:
            return None

        # filter llm_class
        engine_params = deepcopy(LLM_ENGINES[model_name])
        for engine in engine_params:
            params = engine_params[engine]
            for param in params:
                del param["llm_class"]

        return engine_params

    async def _get_model_ability(self, model: Any, model_type: str) -> List[str]:
        from ..model.llm.core import LLM

        if model_type == "embedding":
            return ["embed"]
        elif model_type == "rerank":
            return ["rerank"]
        elif model_type == "image":
            return ["text_to_image"]
        elif model_type == "audio":
            return [model._model_spec.ability]
        elif model_type == "video":
            return ["text_to_video"]
        elif model_type == "flexible":
            return ["flexible"]
        else:
            assert model_type == "LLM"
            assert isinstance(model, LLM)
            return model.model_family.model_ability  # type: ignore

    async def update_cache_status(
        self, model_name: str, model_description: ModelDescription
    ):
        version_info = model_description.to_version_info()
        if isinstance(version_info, list):  # image model
            model_path = version_info[0]["model_file_location"]
            await self._cache_tracker_ref.update_cache_status(
                self.address, model_name, None, model_path
            )
        else:
            await self._cache_tracker_ref.update_cache_status(
                self.address,
                model_name,
                version_info["model_version"],
                version_info["model_file_location"],
            )

    @log_async(logger=logger)
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[Union[int, str]],
        model_format: Optional[str],
        quantization: Optional[str],
        model_engine: Optional[str],
        model_type: str = "LLM",
        n_gpu: Optional[Union[int, str]] = "auto",
        peft_model_config: Optional[PeftModelConfig] = None,
        request_limits: Optional[int] = None,
        gpu_idx: Optional[Union[int, List[int]]] = None,
        download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        # !!! Note that The following code must be placed at the very beginning of this function,
        # or there will be problems with auto-recovery.
        # Because `locals()` will collect all the local parameters of this function and pass to this function again.
        launch_args = locals()
        launch_args.pop("self")
        launch_args.pop("kwargs")
        launch_args.update(kwargs)

        try:
            origin_uid, _, _ = parse_replica_model_uid(model_uid)
        except Exception as e:
            logger.exception(e)
            raise
        try:
            _ = await self.get_supervisor_ref()
            if self._event_collector_ref is not None:
                await self._event_collector_ref.report_event(
                    origin_uid,
                    Event(
                        event_type=EventType.INFO,
                        event_ts=int(time.time()),
                        event_content="Launch model",
                    ),
                )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e))

        if gpu_idx is not None:
            logger.info(
                f"You specify to launch the model: {model_name} on GPU index: {gpu_idx} "
                f"of the worker: {self.address}, "
                f"xinference will automatically ignore the `n_gpu` option."
            )
            if isinstance(gpu_idx, int):
                gpu_idx = [gpu_idx]
            assert isinstance(gpu_idx, list)

        if n_gpu is not None:
            if isinstance(n_gpu, int) and (n_gpu <= 0 or n_gpu > gpu_count()):
                raise ValueError(
                    f"The parameter `n_gpu` must be greater than 0 and "
                    f"not greater than the number of GPUs: {gpu_count()} on the machine."
                )
            if isinstance(n_gpu, str) and n_gpu != "auto":
                raise ValueError("Currently `n_gpu` only supports `auto`.")

        if peft_model_config is not None:
            if model_type in ("embedding", "rerank"):
                raise ValueError(
                    f"PEFT adaptors cannot be applied to embedding or rerank models."
                )
            if model_type == "LLM" and model_format in ("ggufv2",):
                raise ValueError(
                    f"PEFT adaptors can only be applied to pytorch-like models"
                )
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    f"Invalid input. `model_path`: {model_path} File or directory does not exist."
                )

        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name, model_format)

        if self.get_model_launch_status(model_uid) is not None:
            raise ValueError(f"{model_uid} is running")

        try:
            self._model_uid_launching_guard[model_uid] = True
            subpool_address, devices = await self._create_subpool(
                model_uid, model_type, n_gpu=n_gpu, gpu_idx=gpu_idx
            )

            try:
                model, model_description = await asyncio.to_thread(
                    create_model_instance,
                    subpool_address,
                    devices,
                    model_uid,
                    model_type,
                    model_name,
                    model_engine,
                    model_format,
                    model_size_in_billions,
                    quantization,
                    peft_model_config,
                    download_hub,
                    model_path,
                    **kwargs,
                )
                await self.update_cache_status(model_name, model_description)
                model_ref = await xo.create_actor(
                    ModelActor,
                    address=subpool_address,
                    uid=model_uid,
                    worker_address=self.address,
                    model=model,
                    model_description=model_description,
                    request_limits=request_limits,
                )
                await model_ref.load()
            except:
                logger.error(f"Failed to load model {model_uid}", exc_info=True)
                self.release_devices(model_uid=model_uid)
                await self._main_pool.remove_sub_pool(subpool_address)
                raise
            self._model_uid_to_model[model_uid] = model_ref
            self._model_uid_to_model_spec[model_uid] = model_description
            self._model_uid_to_addr[model_uid] = subpool_address
            self._model_uid_to_recover_count.setdefault(
                model_uid, MODEL_ACTOR_AUTO_RECOVER_LIMIT
            )
            self._model_uid_to_launch_args[model_uid] = launch_args
        finally:
            del self._model_uid_launching_guard[model_uid]

        # update status to READY
        abilities = await self._get_model_ability(model, model_type)
        _ = await self.get_supervisor_ref(add_worker=False)

        if self._status_guard_ref is None:
            _ = await self.get_supervisor_ref()
        assert self._status_guard_ref is not None
        await self._status_guard_ref.update_instance_info(
            origin_uid,
            {"model_ability": abilities, "status": LaunchStatus.READY.name},
        )

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str, is_model_die=False):
        # Terminate model while its launching is not allow
        if model_uid in self._model_uid_launching_guard:
            raise ValueError(f"{model_uid} is launching")
        origin_uid, _, __ = parse_replica_model_uid(model_uid)
        try:
            _ = await self.get_supervisor_ref()
            if self._event_collector_ref is not None:
                await self._event_collector_ref.report_event(
                    origin_uid,
                    Event(
                        event_type=EventType.INFO,
                        event_ts=int(time.time()),
                        event_content="Terminate model",
                    ),
                )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e))

        if self._status_guard_ref is not None:
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": LaunchStatus.TERMINATING.name}
            )
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            logger.debug("Model not found, uid: %s", model_uid)

        try:
            await xo.destroy_actor(model_ref)
        except Exception as e:
            logger.debug(
                "Destroy model actor failed, model uid: %s, error: %s", model_uid, e
            )
        try:
            subpool_address = self._model_uid_to_addr[model_uid]
            await self._main_pool.remove_sub_pool(subpool_address)
        except Exception as e:
            logger.debug(
                "Remove sub pool failed, model uid: %s, error: %s", model_uid, e
            )
        finally:
            self._model_uid_to_model.pop(model_uid, None)
            self._model_uid_to_model_spec.pop(model_uid, None)
            self.release_devices(model_uid)
            self._model_uid_to_addr.pop(model_uid, None)
            self._model_uid_to_recover_count.pop(model_uid, None)
            self._model_uid_to_launch_args.pop(model_uid, None)

            if is_model_die:
                status = LaunchStatus.ERROR.name
            else:
                status = LaunchStatus.TERMINATED.name

            if self._status_guard_ref is None:
                _ = await self.get_supervisor_ref()
            assert self._status_guard_ref is not None
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": status}
            )

    # Provide an interface for future version of supervisor to call
    def get_model_launch_status(self, model_uid: str) -> Optional[str]:
        """
        returns:
            CREATING: model is launching
            RREADY: model is running
            None: model is not running (launch error might have happened)
        """

        if model_uid in self._model_uid_launching_guard:
            return LaunchStatus.CREATING.name
        if model_uid in self._model_uid_to_model:
            return LaunchStatus.READY.name
        return None

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}

        items = list(self._model_uid_to_model_spec.items())
        for k, v in items:
            ret[k] = v.to_dict()
        return ret

    @log_sync(logger=logger)
    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            raise ValueError(f"Model not found, uid: {model_uid}")
        return model_ref

    @log_sync(logger=logger)
    def describe_model(self, model_uid: str) -> Dict[str, Any]:
        model_desc = self._model_uid_to_model_spec.get(model_uid, None)
        if model_desc is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        return model_desc.to_dict()

    async def report_status(self):
        status = dict()
        try:
            # asyncio.timeout is only available in Python >= 3.11
            async with timeout(2):
                status = await asyncio.to_thread(gather_node_info)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Report status got error.")
        supervisor_ref = await self.get_supervisor_ref()
        await supervisor_ref.report_worker_status(self.address, status)

    async def _periodical_report_status(self):
        while True:
            try:
                await self.report_status()
            except asyncio.CancelledError:  # pragma: no cover
                break
            except RuntimeError as ex:  # pragma: no cover
                if "cannot schedule new futures" not in str(ex):
                    # when atexit is triggered, the default pool might be shutdown
                    # and to_thread will fail
                    break
            except (
                Exception
            ) as ex:  # pragma: no cover  # noqa: E722  # nosec  # pylint: disable=bare-except
                logger.error(f"Failed to upload node info: {ex}")
            try:
                await asyncio.sleep(XINFERENCE_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break

    async def list_cached_models(
        self, model_name: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        # 从缓存跟踪器获取缓存的模型列表
        lists = await self._cache_tracker_ref.list_cached_models(
            self.address, model_name
        )
        cached_models = []
        for list in lists:
            # 创建缓存模型的基本信息字典
            cached_model = {
                "model_name": list.get("model_name"),
                "model_size_in_billions": list.get("model_size_in_billions"),
                "model_format": list.get("model_format"),
                "quantization": list.get("quantization"),
                "model_version": list.get("model_version"),
            }
            path = list.get("model_file_location")
            cached_model["path"] = path
            # 解析软链接
            if os.path.isdir(path):
                files = os.listdir(path)
                # 如果目录有文件
                if files:
                    # 获取第一个文件的真实路径
                    resolved_file = os.path.realpath(os.path.join(path, files[0]))
                    if resolved_file:
                        # 设置真实路径为解析后文件的目录
                        cached_model["real_path"] = os.path.dirname(resolved_file)
            else:
                # 如果不是目录，直接获取真实路径
                cached_model["real_path"] = os.path.realpath(path)
            # 添加当前worker的地址
            cached_model["actor_ip_address"] = self.address
            cached_models.append(cached_model)
        return cached_models

    async def list_deletable_models(self, model_version: str) -> List[str]:
        # 初始化一个集合来存储可删除的路径
        paths = set()
        
        # 从缓存跟踪器获取可删除模型的路径
        path = await self._cache_tracker_ref.list_deletable_models(
            model_version, self.address
        )
        
        # 如果路径是文件，获取其所在目录
        if os.path.isfile(path):
            path = os.path.dirname(path)

        # 如果路径是目录
        if os.path.isdir(path):
            # 获取目录中的所有文件
            files = os.listdir(path)
            # 将所有文件的完整路径添加到集合中
            paths.update([os.path.join(path, file) for file in files])
            
            # 如果paths不为空，添加所有文件的真实路径（解析软链接）
            if paths:
                paths.update([os.path.realpath(path) for path in paths])

            # 获取tensorizer路径
            from ..model.llm.transformers.tensorizer_utils import get_tensorizer_dir

            tensorizer_path = get_tensorizer_dir(path)
            # 如果tensorizer路径存在且是目录
            if os.path.isdir(tensorizer_path):
                # 获取tensorizer目录中的所有文件
                files = os.listdir(tensorizer_path)
                # 将所有tensorizer文件的完整路径添加到集合中
                paths.update([os.path.join(tensorizer_path, file) for file in files])

        # 返回可删除路径的列表
        return list(paths)

    async def confirm_and_remove_model(self, model_version: str) -> bool:
        """
        确认并删除指定版本的模型。

        参数:
        model_version (str): 要删除的模型版本

        返回:
        bool: 如果删除成功返回True，否则返回False
        """
        # 获取可删除的模型路径列表
        paths = await self.list_deletable_models(model_version)
        
        # 遍历每个路径并尝试删除
        for path in paths:
            try:
                if os.path.islink(path):
                    # 如果是符号链接，解除链接
                    os.unlink(path)
                elif os.path.isfile(path):
                    # 如果是文件，直接删除
                    os.remove(path)
                elif os.path.isdir(path):
                    # 如果是目录，递归删除整个目录
                    shutil.rmtree(path)
                else:
                    # 如果不是有效路径，记录调试信息
                    logger.debug(f"{path} is not a valid path.")
            except Exception as e:
                # 如果删除过程中出现错误，记录错误信息并返回False
                logger.error(f"Fail to delete {path} with error:{e}.")
                return False
        
        # 通知缓存追踪器确认并移除模型
        await self._cache_tracker_ref.confirm_and_remove_model(
            model_version, self.address
        )
        
        # 所有操作成功完成，返回True
        return True
    async def get_workers_info(self) -> Dict[str, Any]:
        """
        获取当前工作节点的信息。

        返回:
        Dict[str, Any]: 包含工作节点IP地址和已加载模型列表的字典
        """
        ret = {
            "work-ip": self.address,  # 当前工作节点的IP地址
            "models": await self.list_models(),  # 异步获取当前节点上已加载的模型列表
        }
        return ret  # 返回包含工作节点信息的字典

    @staticmethod
    def record_metrics(name, op, kwargs):
        record_metrics(name, op, kwargs)
