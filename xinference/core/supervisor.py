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
import itertools
import os
import signal
import time
import typing
from dataclasses import dataclass
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import xoscar as xo

from ..constants import (
    XINFERENCE_DISABLE_HEALTH_CHECK,
    XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
    XINFERENCE_HEALTH_CHECK_TIMEOUT,
)
from ..core.model import ModelActor
from ..core.status_guard import InstanceInfo, LaunchStatus
from ..types import PeftModelConfig
from .metrics import record_metrics
from .resource import GPUStatus, ResourceStatus
from .utils import (
    assign_replica_gpu,
    build_replica_model_uid,
    gen_random_string,
    is_valid_model_uid,
    iter_replica_model_uid,
    log_async,
    log_sync,
    parse_model_version,
    parse_replica_model_uid,
)

if TYPE_CHECKING:
    from ..model.audio import AudioModelFamilyV1
    from ..model.embedding import EmbeddingModelSpec
    from ..model.flexible import FlexibleModelSpec
    from ..model.image import ImageModelFamilyV1
    from ..model.llm import LLMFamilyV1
    from ..model.rerank import RerankModelSpec
    from ..model.video import VideoModelFamilyV1
    from .worker import WorkerActor


logger = getLogger(__name__)


ASYNC_LAUNCH_TASKS = {}  # type: ignore


def callback_for_async_launch(model_uid: str):
    ASYNC_LAUNCH_TASKS.pop(model_uid, None)
    logger.debug(f"Model uid: {model_uid} async launch completes.")


@dataclass
class WorkerStatus:
    update_time: float
    failure_remaining_count: int
    status: Dict[str, Union[ResourceStatus, GPUStatus]]


@dataclass
class ReplicaInfo:
    replica: int
    scheduler: Iterator


class SupervisorActor(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        self._worker_address_to_worker: Dict[str, xo.ActorRefType["WorkerActor"]] = {}  # type: ignore
        self._worker_status: Dict[str, WorkerStatus] = {}  # type: ignore
        self._replica_model_uid_to_worker: Dict[  # type: ignore
            str, xo.ActorRefType["WorkerActor"]
        ] = {}
        self._model_uid_to_replica_info: Dict[str, ReplicaInfo] = {}  # type: ignore
        self._uptime = None
        self._lock = asyncio.Lock()

    @classmethod
    def uid(cls) -> str:
        return "supervisor"

    def _get_worker_ref_by_ip(
        self, ip: str
    ) -> Optional[xo.ActorRefType["WorkerActor"]]:
        for addr, ref in self._worker_address_to_worker.items():
            existing_ip = addr.split(":")[0]
            if existing_ip == ip:
                return ref
        return None

    async def __post_create__(self):
        self._uptime = time.time()
        if not XINFERENCE_DISABLE_HEALTH_CHECK:
            # Run _check_dead_nodes() in a dedicated thread.
            from ..isolation import Isolation

            self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            self._isolation.start()
            asyncio.run_coroutine_threadsafe(
                self._check_dead_nodes(), loop=self._isolation.loop
            )
        logger.info(f"Xinference supervisor {self.address} started")
        from .cache_tracker import CacheTrackerActor
        from .status_guard import StatusGuardActor

        self._status_guard_ref: xo.ActorRefType[  # type: ignore
            "StatusGuardActor"
        ] = await xo.create_actor(
            StatusGuardActor, address=self.address, uid=StatusGuardActor.uid()
        )
        self._cache_tracker_ref: xo.ActorRefType[  # type: ignore
            "CacheTrackerActor"
        ] = await xo.create_actor(
            CacheTrackerActor, address=self.address, uid=CacheTrackerActor.uid()
        )

        from .event import EventCollectorActor

        self._event_collector_ref: xo.ActorRefType[  # type: ignore
            EventCollectorActor
        ] = await xo.create_actor(
            EventCollectorActor, address=self.address, uid=EventCollectorActor.uid()
        )

        from ..model.audio import (
            CustomAudioModelFamilyV1,
            generate_audio_description,
            get_audio_model_descriptions,
            register_audio,
            unregister_audio,
        )
        from ..model.embedding import (
            CustomEmbeddingModelSpec,
            generate_embedding_description,
            get_embedding_model_descriptions,
            register_embedding,
            unregister_embedding,
        )
        from ..model.flexible import (
            FlexibleModelSpec,
            generate_flexible_model_description,
            get_flexible_model_descriptions,
            register_flexible_model,
            unregister_flexible_model,
        )
        from ..model.image import (
            CustomImageModelFamilyV1,
            generate_image_description,
            get_image_model_descriptions,
            register_image,
            unregister_image,
        )
        from ..model.llm import (
            CustomLLMFamilyV1,
            generate_llm_description,
            get_llm_model_descriptions,
            register_llm,
            unregister_llm,
        )
        from ..model.rerank import (
            CustomRerankModelSpec,
            generate_rerank_description,
            get_rerank_model_descriptions,
            register_rerank,
            unregister_rerank,
        )

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

        # record model version
        model_version_infos: Dict[str, List[Dict]] = {}  # type: ignore
        model_version_infos.update(get_llm_model_descriptions())
        model_version_infos.update(get_embedding_model_descriptions())
        model_version_infos.update(get_rerank_model_descriptions())
        model_version_infos.update(get_image_model_descriptions())
        model_version_infos.update(get_audio_model_descriptions())
        model_version_infos.update(get_flexible_model_descriptions())
        await self._cache_tracker_ref.record_model_version(
            model_version_infos, self.address
        )

        # Windows does not have signal handler
        if os.name != "nt":

            async def signal_handler():
                os._exit(0)

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGTERM, lambda: asyncio.create_task(signal_handler())
            )

    @typing.no_type_check
    async def get_cluster_device_info(self, detailed: bool = False) -> List:
        """
        获取集群设备信息的异步方法。

        参数:
            detailed (bool): 是否返回详细信息，默认为False。

        返回:
            List: 包含集群中所有节点（Supervisor和Worker）设备信息的列表。

        说明:
        1. 首先导入psutil库，用于获取系统信息。
        2. 创建supervisor_device_info字典，包含基本信息如IP地址、GPU数量和总VRAM。
        3. 如果detailed为True，添加更多详细信息，包括CPU和内存使用情况。
        4. 将supervisor信息添加到结果列表res中。
        5. 遍历所有worker节点，收集它们的设备信息：
           - 基本信息包括节点类型、IP地址、GPU数量和总VRAM。
           - 如果detailed为True，还包括CPU和内存的详细使用情况。
        6. 将所有worker信息添加到res列表中。
        7. 返回包含所有节点信息的列表。
        """
        import psutil

        supervisor_device_info = {
            "ip_address": self.address.split(":")[0],
            "gpu_count": 0,
            "gpu_vram_total": 0,
        }
        if detailed:
            supervisor_device_info["gpu_vram_total"] = 0
            supervisor_device_info["gpu_vram_available"] = 0
            supervisor_device_info["cpu_available"] = psutil.cpu_count() * (
                1 - psutil.cpu_percent() / 100.0
            )
            supervisor_device_info["cpu_count"] = psutil.cpu_count()
            mem_info = psutil.virtual_memory()
            supervisor_device_info["mem_used"] = mem_info.used
            supervisor_device_info["mem_available"] = mem_info.available
            supervisor_device_info["mem_total"] = mem_info.total
        res = [{"node_type": "Supervisor", **supervisor_device_info}]
        for worker_addr, worker_status in self._worker_status.items():
            vram_total: float = sum(
                [v.mem_total for k, v in worker_status.status.items() if k != "cpu"]  # type: ignore
            )
            total = (
                vram_total if vram_total == 0 else f"{int(vram_total / 1024 / 1024)}MiB"
            )
            info = {
                "node_type": "Worker",
                "ip_address": worker_addr.split(":")[0],
                "gpu_count": len(worker_status.status) - 1,
                "gpu_vram_total": total,
            }
            if detailed:
                cpu_info = worker_status.status["cpu"]
                info["cpu_available"] = cpu_info.total * (1 - cpu_info.usage)
                info["cpu_count"] = cpu_info.total
                info["mem_used"] = cpu_info.memory_used
                info["mem_available"] = cpu_info.memory_available
                info["mem_total"] = cpu_info.memory_total
                info["gpu_vram_total"] = vram_total
                info["gpu_vram_available"] = sum(
                    [v.mem_free for k, v in worker_status.status.items() if k != "cpu"]
                )
            res.append(info)
        return res
    @staticmethod
    async def get_builtin_prompts() -> Dict[str, Any]:
        """
        获取内置提示的异步静态方法。

        返回:
            Dict[str, Any]: 包含内置提示样式的字典。

        说明:
        - 从 BUILTIN_LLM_PROMPT_STYLE 导入预定义的提示样式。
        - 创建一个空字典 data 来存储结果。
        - 遍历 BUILTIN_LLM_PROMPT_STYLE 中的每个键值对：
          - 键 (k) 作为新字典的键。
          - 值 (v) 通过调用 dict() 方法转换为字典，作为新字典的值。
        - 返回包含所有转换后提示样式的字典。
        """
        from ..model.llm.llm_family import BUILTIN_LLM_PROMPT_STYLE

        data = {}
        for k, v in BUILTIN_LLM_PROMPT_STYLE.items():
            data[k] = v.dict()
        return data

    @staticmethod
    async def get_builtin_families() -> Dict[str, List[str]]:
        from ..model.llm.llm_family import (
            BUILTIN_LLM_MODEL_CHAT_FAMILIES,
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES,
        )

        return {
            "chat": list(BUILTIN_LLM_MODEL_CHAT_FAMILIES),
            "generate": list(BUILTIN_LLM_MODEL_GENERATE_FAMILIES),
            "tools": list(BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES),
        }

    async def get_devices_count(self) -> int:
        from ..device_utils import gpu_count

        if self.is_local_deployment():
            print(f"local: deployment:===> !!!")
            return gpu_count()
        # distributed deployment, choose a worker and return its device_count.
        # Assume that each worker has the same count of cards.
        worker_ref = await self._choose_worker()
        return await worker_ref.get_devices_count()

    async def _choose_worker(self) -> xo.ActorRefType["WorkerActor"]:
        """
        选择一个工作节点的异步方法。

        返回:
            xo.ActorRefType["WorkerActor"]: 选中的工作节点的引用。

        异常:
            RuntimeError: 当没有可用的工作节点时抛出。

        说明:
        - 该方法使用简单的负载均衡策略选择一个工作节点。
        - 遍历所有工作节点，选择运行模型数量最少的节点。
        - 如果找到合适的工作节点，返回该节点的引用。
        - 如果没有可用的工作节点，抛出RuntimeError异常。

        注意:
        - TODO: 未来可以实现更好的分配策略。
        """
        # TODO: better allocation strategy.
        min_running_model_count = None
        target_worker = None

        workers = list(self._worker_address_to_worker.values())
        for worker in workers:
            running_model_count = await worker.get_model_count()
            if (
                min_running_model_count is None
                or running_model_count < min_running_model_count
            ):
                min_running_model_count = running_model_count
                target_worker = worker

        if target_worker:
            return target_worker

        raise RuntimeError("No available worker found")

    @log_sync(logger=logger)
    def get_status(self) -> Dict:
        return {
            "uptime": int(time.time() - self._uptime),
            "workers": self._worker_status,
        }

    async def _to_llm_reg(
        self, llm_family: "LLMFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.llm import get_cache_status

        instance_cnt = await self.get_instance_count(llm_family.model_name)
        version_cnt = await self.get_model_version_count(llm_family.model_name)

        if self.is_local_deployment():
            specs = []
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            for spec in llm_family.model_specs:
                cache_status = get_cache_status(llm_family, spec)
                specs.append({**spec.dict(), "cache_status": cache_status})
            res = {**llm_family.dict(), "is_builtin": is_builtin, "model_specs": specs}
        else:
            res = {**llm_family.dict(), "is_builtin": is_builtin}
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_embedding_model_reg(
        self, model_spec: "EmbeddingModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.embedding import get_cache_status

        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_spec)
            res = {
                **model_spec.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_rerank_model_reg(
        self, model_spec: "RerankModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.rerank import get_cache_status

        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_spec)
            res = {
                **model_spec.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_image_model_reg(
        self, model_family: "ImageModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.image import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_audio_model_reg(
        self, model_family: "AudioModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.audio import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_video_model_reg(
        self, model_family: "VideoModelFamilyV1", is_builtin: bool
    ) -> Dict[str, Any]:
        from ..model.video import get_cache_status

        instance_cnt = await self.get_instance_count(model_family.model_name)
        version_cnt = await self.get_model_version_count(model_family.model_name)

        if self.is_local_deployment():
            # TODO: does not work when the supervisor and worker are running on separate nodes.
            cache_status = get_cache_status(model_family)
            res = {
                **model_family.dict(),
                "cache_status": cache_status,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_family.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    async def _to_flexible_model_reg(
        self, model_spec: "FlexibleModelSpec", is_builtin: bool
    ) -> Dict[str, Any]:
        instance_cnt = await self.get_instance_count(model_spec.model_name)
        version_cnt = await self.get_model_version_count(model_spec.model_name)

        if self.is_local_deployment():
            res = {
                **model_spec.dict(),
                "cache_status": True,
                "is_builtin": is_builtin,
            }
        else:
            res = {
                **model_spec.dict(),
                "is_builtin": is_builtin,
            }
        res["model_version_count"] = version_cnt
        res["model_instance_count"] = instance_cnt
        return res

    @log_async(logger=logger)
    async def list_model_registrations(
        self, model_type: str, detailed: bool = False
    ) -> List[Dict[str, Any]]:
        def sort_helper(item):
            assert isinstance(item["model_name"], str)
            return item.get("model_name").lower()

        ret = []
        if not self.is_local_deployment():
            workers = list(self._worker_address_to_worker.values())
            for worker in workers:
                ret.extend(await worker.list_model_registrations(model_type, detailed))

        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            for family in BUILTIN_LLM_FAMILIES:
                if detailed:
                    ret.append(await self._to_llm_reg(family, True))
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": True})

            for family in get_user_defined_llm_families():
                if detailed:
                    ret.append(await self._to_llm_reg(family, False))
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            for model_name, family in BUILTIN_EMBEDDING_MODELS.items():
                if detailed:
                    ret.append(
                        await self._to_embedding_model_reg(family, is_builtin=True)
                    )
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_embeddings():
                if detailed:
                    ret.append(
                        await self._to_embedding_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.custom import get_user_defined_images

            for model_name, family in BUILTIN_IMAGE_MODELS.items():
                if detailed:
                    ret.append(await self._to_image_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_images():
                if detailed:
                    ret.append(
                        await self._to_image_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios

            for model_name, family in BUILTIN_AUDIO_MODELS.items():
                if detailed:
                    ret.append(await self._to_audio_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_audios():
                if detailed:
                    ret.append(
                        await self._to_audio_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "video":
            from ..model.video import BUILTIN_VIDEO_MODELS

            for model_name, family in BUILTIN_VIDEO_MODELS.items():
                if detailed:
                    ret.append(await self._to_video_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            for model_name, family in BUILTIN_RERANK_MODELS.items():
                if detailed:
                    ret.append(await self._to_rerank_model_reg(family, is_builtin=True))
                else:
                    ret.append({"model_name": model_name, "is_builtin": True})

            for model_spec in get_user_defined_reranks():
                if detailed:
                    ret.append(
                        await self._to_rerank_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "flexible":
            from ..model.flexible import get_flexible_models

            ret = []

            for model_spec in get_flexible_models():
                if detailed:
                    ret.append(
                        await self._to_flexible_model_reg(model_spec, is_builtin=False)
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def get_model_registration(self, model_type: str, model_name: str) -> Any:
        # search in worker first
        if not self.is_local_deployment():
            workers = list(self._worker_address_to_worker.values())
            for worker in workers:
                f = await worker.get_model_registration(model_type, model_name)
                if f is not None:
                    return f

        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            for f in BUILTIN_LLM_FAMILIES + get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f

            raise ValueError(f"Model {model_name} not found")
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            for f in (
                list(BUILTIN_EMBEDDING_MODELS.values()) + get_user_defined_embeddings()
            ):
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.custom import get_user_defined_images

            for f in list(BUILTIN_IMAGE_MODELS.values()) + get_user_defined_images():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios

            for f in list(BUILTIN_AUDIO_MODELS.values()) + get_user_defined_audios():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            for f in list(BUILTIN_RERANK_MODELS.values()) + get_user_defined_reranks():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        elif model_type == "flexible":
            from ..model.flexible import get_flexible_models

            for f in get_flexible_models():
                if f.model_name == model_name:
                    return f
            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def query_engines_by_model_name(self, model_name: str):
        # 这个方法用于根据模型名称查询引擎参数
        from copy import deepcopy

        from ..model.llm.llm_family import LLM_ENGINES

        # 首先在工作节点中搜索
        workers = list(self._worker_address_to_worker.values())
        for worker in workers:
            # 对每个工作节点异步调用query_engines_by_model_name方法
            res = await worker.query_engines_by_model_name(model_name)
            if res is not None:
                # 如果找到结果，直接返回
                return res

        # 如果在工作节点中没有找到，检查是否在LLM_ENGINES中
        if model_name not in LLM_ENGINES:
            # 如果模型名称不在LLM_ENGINES中，抛出ValueError异常
            raise ValueError(f"Model {model_name} not found")

        # filter llm_class
        engine_params = deepcopy(LLM_ENGINES[model_name])
        # 遍历所有引擎参数，移除"llm_class"键
        for engine in engine_params:
            params = engine_params[engine]
            for param in params:
                del param["llm_class"]

        # 返回处理后的引擎参数
        return engine_params

    @log_async(logger=logger)
    async def register_model(
        self,
        model_type: str,
        model: str,
        persist: bool,
        worker_ip: Optional[str] = None,
    ):
        if model_type in self._custom_register_type_to_cls:
            (
                model_spec_cls,
                register_fn,
                unregister_fn,
                generate_fn,
            ) = self._custom_register_type_to_cls[model_type]

            target_ip_worker_ref = (
                self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
            )
            if (
                worker_ip is not None
                and not self.is_local_deployment()
                and target_ip_worker_ref is None
            ):
                raise ValueError(
                    f"Worker ip address {worker_ip} is not in the cluster."
                )

            if target_ip_worker_ref:
                await target_ip_worker_ref.register_model(model_type, model, persist)
                return

            model_spec = model_spec_cls.parse_raw(model)
            try:
                register_fn(model_spec, persist)
                await self._cache_tracker_ref.record_model_version(
                    generate_fn(model_spec), self.address
                )
            except ValueError as e:
                raise e
            except Exception as e:
                unregister_fn(model_spec.model_name, raise_error=False)
                raise e
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        if model_type in self._custom_register_type_to_cls:
            _, _, unregister_fn, _ = self._custom_register_type_to_cls[model_type]
            unregister_fn(model_name, False)

            if not self.is_local_deployment():
                workers = list(self._worker_address_to_worker.values())
                for worker in workers:
                    await worker.unregister_model(model_type, model_name)

            await self._cache_tracker_ref.unregister_model_version(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _gen_model_uid(self, model_name: str) -> str:
        if model_name not in self._model_uid_to_replica_info:
            return model_name
        logger.debug(
            f"{model_name} exists in xinference. Generate suffix to {model_name} for model_uid."
        )
        return f"{model_name}-{gen_random_string(8)}"

    # TODO 获取指定模型的所有版本信息 这里的mode_type没有使用到
    async def get_model_versions(self, model_type: str, model_name: str) -> List[Dict]:
        """
        获取指定模型的所有版本信息。

        参数:
            model_type: str - 模型类型
            model_name: str - 模型名称

        返回:
            List[Dict] - 包含模型所有版本信息的字典列表

        说明:
            - 该方法通过缓存追踪器获取指定模型名称的所有版本信息
            - model_type 参数目前未使用，保留以便未来扩展
        """
        return await self._cache_tracker_ref.get_model_versions(model_name)

    async def get_model_version_count(self, model_name: str) -> int:
        return await self._cache_tracker_ref.get_model_version_count(model_name)

    @log_async(logger=logger)
    async def launch_model_by_version(
        self,
        model_uid: Optional[str],
        model_type: str,
        model_engine: Optional[str],
        model_version: str,
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        wait_ready: bool = True,
    ):
        # 解析模型版本信息
        parse_results = parse_model_version(model_version, model_type)

        # 如果是图像模型且解析结果有两个元素，设置controlnet参数
        if model_type == "image" and len(parse_results) == 2:
            kwargs = {"controlnet": parse_results[1]}
        else:
            kwargs = {}

        # 调用launch_builtin_model方法启动模型
        return await self.launch_builtin_model(
            model_uid=model_uid,
            model_name=parse_results[0],
            model_engine=model_engine,
            # 根据模型类型设置不同的参数
            model_size_in_billions=parse_results[1] if model_type == "LLM" else None,
            model_format=parse_results[2] if model_type == "LLM" else None,
            quantization=parse_results[3] if model_type == "LLM" else None,
            model_type=model_type,
            replica=replica,
            n_gpu=n_gpu,
            wait_ready=wait_ready,
            model_version=model_version,
            **kwargs,
        )

    async def launch_builtin_model(
        self,
        model_uid: Optional[str],
        model_name: str,
        model_size_in_billions: Optional[Union[int, str]],
        model_format: Optional[str],
        quantization: Optional[str],
        model_engine: Optional[str],
        model_type: Optional[str],
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        request_limits: Optional[int] = None,
        wait_ready: bool = True,
        model_version: Optional[str] = None,
        peft_model_config: Optional[PeftModelConfig] = None,
        worker_ip: Optional[str] = None,
        gpu_idx: Optional[Union[int, List[int]]] = None,
        download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        # search in worker first
        if not self.is_local_deployment():
            workers = list(self._worker_address_to_worker.values())
            for worker in workers:
                res = await worker.get_model_registration(model_type, model_name)
                if res is not None:
                    worker_ip = worker.address.split(":")[0]

        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")
        if worker_ip is not None and self.is_local_deployment():
            logger.warning(
                f"You specified the worker ip: {worker_ip} in local mode, "
                f"xinference will ignore this option."
            )

        if kwargs.get("enable_tensorizer", None) and (
            (
                model_engine is None
                or model_engine.lower() != "transformers"
                or model_format != "pytorch"
                or quantization != "none"
                or model_type != "LLM"
            )
        ):
            raise ValueError(
                "Tensorizer can only be enabled for LLM models with Transformers engine, PyTorch format, and none quantization."
            )

        if kwargs.get("enable_tensorizer", None) and model_name in [
            "OmniLMM",
            "yi-vl-chat",
            "deepseek-vl-chat",
        ]:
            raise ValueError("Tensorizer is not supported for %s." % model_name)

        if model_uid is None:
            model_uid = self._gen_model_uid(model_name)

        model_size = str(model_size_in_billions) if model_size_in_billions else ""
        logger.debug(
            f"Enter launch_builtin_model, model_uid: {model_uid}, model_name: {model_name}, model_size: {model_size}, "
            f"model_format: {model_format}, quantization: {quantization}, replica: {replica}, "
            f"kwargs: {kwargs}"
        )

        async def _launch_one_model(_replica_model_uid):
            if _replica_model_uid in self._replica_model_uid_to_worker:
                raise ValueError(
                    f"Model is already in the model list, uid: {_replica_model_uid}"
                )
            replica_gpu_idx = assign_replica_gpu(_replica_model_uid, gpu_idx)
            nonlocal model_type

            worker_ref = (
                target_ip_worker_ref
                if target_ip_worker_ref is not None
                else await self._choose_worker()
            )
            # LLM as default for compatibility
            model_type = model_type or "LLM"
            await worker_ref.launch_builtin_model(
                model_uid=_replica_model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                model_engine=model_engine,
                model_type=model_type,
                n_gpu=n_gpu,
                request_limits=request_limits,
                peft_model_config=peft_model_config,
                gpu_idx=replica_gpu_idx,
                download_hub=download_hub,
                model_path=model_path,
                **kwargs,
            )
            self._replica_model_uid_to_worker[_replica_model_uid] = worker_ref

        async def _launch_model():
            try:
                for rep_model_uid in iter_replica_model_uid(model_uid, replica):
                    await _launch_one_model(rep_model_uid)
            except Exception:
                # terminate_model will remove the replica info.
                await self.terminate_model(model_uid, suppress_exception=True)
                await self._status_guard_ref.update_instance_info(
                    model_uid, {"status": LaunchStatus.ERROR.name}
                )
                raise

        if not is_valid_model_uid(model_uid):
            raise ValueError(
                "The model UID is invalid. Please specify the model UID by 0 < length <= 100."
            )

        if request_limits is not None and request_limits < 0:
            raise ValueError(
                "The `request_limits` parameter must be greater or equal than 0."
            )

        if model_uid in self._model_uid_to_replica_info:
            raise ValueError(f"Model is already in the model list, uid: {model_uid}")
        # Set replica info first for exception handler to terminate model.
        self._model_uid_to_replica_info[model_uid] = ReplicaInfo(
            replica=replica, scheduler=itertools.cycle(range(replica))
        )
        instance_info = InstanceInfo(
            model_name=model_name,
            model_uid=model_uid,
            model_version=model_version,
            model_ability=[],
            replica=replica,
            status=LaunchStatus.CREATING.name,
            instance_created_ts=int(time.time()),
        )
        await self._status_guard_ref.set_instance_info(model_uid, instance_info)
        if wait_ready:
            await _launch_model()
        else:
            task = asyncio.create_task(_launch_model())
            ASYNC_LAUNCH_TASKS[model_uid] = task
            task.add_done_callback(lambda _: callback_for_async_launch(model_uid))
        return model_uid

    async def get_instance_info(
        self, model_name: Optional[str], model_uid: Optional[str]
    ) -> List[Dict]:
        """
        获取模型实例信息。

        参数:
            model_name: 可选，模型名称
            model_uid: 可选，模型唯一标识符

        返回:
            包含模型实例信息的字典列表，按模型UID排序
        """
        # 从状态守卫获取实例信息
        infos = await self._status_guard_ref.get_instance_info(
            model_name=model_name, model_uid=model_uid
        )
        # 将实例信息转换为字典并按模型UID排序
        return [info.dict() for info in sorted(infos, key=lambda info: info.model_uid)]

    async def get_instance_count(self, model_name: str) -> int:
        return await self._status_guard_ref.get_instance_count(model_name)

    async def _check_dead_nodes(self):
        while True:
            try:
                dead_nodes = []
                for address, status in self._worker_status.items():
                    if (
                        time.time() - status.update_time
                        > XINFERENCE_HEALTH_CHECK_TIMEOUT
                    ):
                        status.failure_remaining_count -= 1
                    else:
                        status.failure_remaining_count = (
                            XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
                        )

                    if status.failure_remaining_count <= 0:
                        dead_models = []
                        for model_uid in self._replica_model_uid_to_worker:
                            if (
                                self._replica_model_uid_to_worker[model_uid].address
                                == address
                            ):
                                dead_models.append(model_uid)
                        logger.error(
                            "Worker dead. address: %s, influenced models: %s",
                            address,
                            dead_models,
                        )
                        for replica_model_uid in dead_models:
                            model_uid, _, _ = parse_replica_model_uid(replica_model_uid)
                            self._model_uid_to_replica_info.pop(model_uid, None)
                            self._replica_model_uid_to_worker.pop(
                                replica_model_uid, None
                            )
                        dead_nodes.append(address)
                    elif (
                        status.failure_remaining_count
                        != XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
                    ):
                        logger.error(
                            "Worker timeout. address: %s, check count remaining %s...",
                            address,
                            status.failure_remaining_count,
                        )

                for address in dead_nodes:
                    self._worker_status.pop(address, None)
                    self._worker_address_to_worker.pop(address, None)
            finally:
                await asyncio.sleep(XINFERENCE_HEALTH_CHECK_INTERVAL)

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str, suppress_exception=False):
        async def _terminate_one_model(_replica_model_uid):
            worker_ref = self._replica_model_uid_to_worker.get(_replica_model_uid, None)

            if worker_ref is None:
                raise ValueError(
                    f"Model not found in the model list, uid: {_replica_model_uid}"
                )
            await worker_ref.terminate_model(model_uid=_replica_model_uid)
            del self._replica_model_uid_to_worker[_replica_model_uid]

        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        for rep_model_uid in iter_replica_model_uid(model_uid, replica_info.replica):
            try:
                await _terminate_one_model(rep_model_uid)
            except Exception:
                if not suppress_exception:
                    raise
        self._model_uid_to_replica_info.pop(model_uid, None)

    @log_async(logger=logger)
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        replica_model_uid = build_replica_model_uid(
            model_uid, replica_info.replica, next(replica_info.scheduler)
        )

        worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
        if worker_ref is None:
            raise ValueError(
                f"Model not found in the model list, uid: {replica_model_uid}"
            )
        return await worker_ref.get_model(model_uid=replica_model_uid)

    @log_async(logger=logger)
    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        # 从模型UID到副本信息的映射中获取副本信息
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if replica_info is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        # Use rep id 0 to instead of next(replica_info.scheduler) to avoid
        # consuming the generator.
        replica_model_uid = build_replica_model_uid(model_uid, replica_info.replica, 0)
        
        # 从副本模型UID到工作节点引用的映射中获取工作节点引用
        worker_ref = self._replica_model_uid_to_worker.get(replica_model_uid, None)
        if worker_ref is None:
            # 如果找不到工作节点引用，抛出ValueError异常
            raise ValueError(
                f"Model not found in the model list, uid: {replica_model_uid}"
            )
        
        # 调用工作节点的describe_model方法获取模型信息
        info = await worker_ref.describe_model(model_uid=replica_model_uid)
        
        # 在返回的信息中添加副本数量
        info["replica"] = replica_info.replica
        
        # 返回模型描述信息
        return info

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        # 初始化返回结果字典
        ret = {}

        # 获取所有工作节点的引用列表
        workers = list(self._worker_address_to_worker.values())
        # 遍历每个工作节点，获取其模型列表并更新结果字典
        for worker in workers:
            ret.update(await worker.list_models())
        
        # 解析复制模型UID，创建运行中模型信息字典
        running_model_info = {parse_replica_model_uid(k)[0]: v for k, v in ret.items()}
        
        # 为每个运行中的模型添加副本数量信息
        for k, v in running_model_info.items():
            v["replica"] = self._model_uid_to_replica_info[k].replica
        
        # 返回包含所有运行中模型信息的字典
        return running_model_info

    def is_local_deployment(self) -> bool:
        # TODO: temporary.
        return (
            len(self._worker_address_to_worker) == 1
            and list(self._worker_address_to_worker)[0] == self.address
        )

    @log_async(logger=logger)
    async def list_cached_models(
        self, model_name: Optional[str] = None, worker_ip: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")

        # search assigned worker and return
        if target_ip_worker_ref:
            cached_models = await target_ip_worker_ref.list_cached_models(model_name)
            cached_models = sorted(cached_models, key=lambda x: x["model_name"])
            return cached_models

        # search all worker
        cached_models = []
        for worker in self._worker_address_to_worker.values():
            res = await worker.list_cached_models(model_name)
            cached_models.extend(res)
        cached_models = sorted(cached_models, key=lambda x: x["model_name"])
        return cached_models

    @log_async(logger=logger)
    async def abort_request(self, model_uid: str, request_id: str) -> Dict:
        # 导入中止请求消息枚举
        from .scheduler import AbortRequestMessage

        # 初始化结果字典，默认为无操作
        res = {"msg": AbortRequestMessage.NO_OP.name}
        
        # 获取模型的副本信息
        replica_info = self._model_uid_to_replica_info.get(model_uid, None)
        if not replica_info:
            return res
        replica_cnt = replica_info.replica

        # 遍历所有副本
        for rep_mid in iter_replica_model_uid(model_uid, replica_cnt):
            # 获取工作节点引用
            worker_ref = self._replica_model_uid_to_worker.get(rep_mid, None)
            if worker_ref is None:
                continue
            
            # 获取模型引用
            model_ref = await worker_ref.get_model(model_uid=rep_mid)
            
            # 尝试中止请求
            result_info = await model_ref.abort_request(request_id)
            res["msg"] = result_info
            
            # 根据结果进行相应处理
            if result_info == AbortRequestMessage.DONE.name:
                break  # 成功中止，退出循环
            elif result_info == AbortRequestMessage.NOT_FOUND.name:
                logger.debug(f"Request id: {request_id} not found for model {rep_mid}")
            else:
                logger.debug(f"No-op for model {rep_mid}")
        
        return res

    @log_async(logger=logger)
    async def add_worker(self, worker_address: str):
        # 从worker模块导入WorkerActor类
        from .worker import WorkerActor

        # 断言检查：确保worker_address不在已存在的worker列表中
        assert (
            worker_address not in self._worker_address_to_worker
        ), f"Worker {worker_address} exists"

        # 创建一个指向新worker的actor引用
        worker_ref = await xo.actor_ref(address=worker_address, uid=WorkerActor.uid())
        
        # 将新worker添加到worker字典中
        self._worker_address_to_worker[worker_address] = worker_ref
        
        # 记录成功添加worker的日志
        logger.debug("Worker %s has been added successfully", worker_address)

    @log_async(logger=logger)
    async def remove_worker(self, worker_address: str):
        # 初始化一个列表，用于存储需要移除的模型 UID
        uids_to_remove = []
        
        # 遍历所有模型 UID，找出与要移除的 worker 相关的模型
        for model_uid in self._replica_model_uid_to_worker:
            if self._replica_model_uid_to_worker[model_uid].address == worker_address:
                uids_to_remove.append(model_uid)

        # 移除与该 worker 相关的所有模型信息
        for replica_model_uid in uids_to_remove:
            model_uid, _, _ = parse_replica_model_uid(replica_model_uid)
            # 从副本信息字典中移除模型信息
            self._model_uid_to_replica_info.pop(model_uid, None)
            # 从副本模型到 worker 的映射中移除信息
            self._replica_model_uid_to_worker.pop(replica_model_uid, None)

        # 如果 worker 地址存在于 worker 字典中，则移除它
        if worker_address in self._worker_address_to_worker:
            del self._worker_address_to_worker[worker_address]
            logger.debug("Worker %s has been removed successfully", worker_address)
        else:
            # 如果 worker 不在字典中，记录警告日志
            logger.warning(
                f"Worker {worker_address} cannot be removed since it is not registered to supervisor."
            )

    async def report_worker_status(
        self, worker_address: str, status: Dict[str, Union[ResourceStatus, GPUStatus]]
    ):
        # 检查worker_address是否已存在于_worker_status字典中
        if worker_address not in self._worker_status:
            # 如果不存在，记录日志并创建新的WorkerStatus对象
            logger.debug("Worker %s resources: %s", worker_address, status)
            self._worker_status[worker_address] = WorkerStatus(
                update_time=time.time(),  # 设置更新时间为当前时间
                failure_remaining_count=XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,  # 设置剩余失败次数
                status=status,  # 设置worker状态
            )
        else:
            # 如果worker_address已存在，更新其状态
            worker_status = self._worker_status[worker_address]
            worker_status.update_time = time.time()  # 更新最后更新时间
            worker_status.status = status  # 更新worker状态

    async def list_deletable_models(
        self, model_version: str, worker_ip: Optional[str] = None
    ) -> List[str]:
        # 获取指定IP的worker引用，如果未指定IP则为None
        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        
        # 检查指定的worker IP是否有效
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")

        ret = []
        # 如果指定了特定的worker
        if target_ip_worker_ref:
            # 从指定的worker获取可删除的模型列表
            ret = await target_ip_worker_ref.list_deletable_models(
                model_version=model_version,
            )
            return ret

        # 如果未指定特定worker，则遍历所有worker
        for worker in self._worker_address_to_worker.values():
            # 从每个worker获取可删除的模型路径
            path = await worker.list_deletable_models(model_version=model_version)
            # 将获取的路径添加到结果列表中
            ret.extend(path)
        return ret

    async def confirm_and_remove_model(
        self, model_version: str, worker_ip: Optional[str] = None
    ) -> bool:
        target_ip_worker_ref = (
            self._get_worker_ref_by_ip(worker_ip) if worker_ip is not None else None
        )
        if (
            worker_ip is not None
            and not self.is_local_deployment()
            and target_ip_worker_ref is None
        ):
            raise ValueError(f"Worker ip address {worker_ip} is not in the cluster.")

        if target_ip_worker_ref:
            ret = await target_ip_worker_ref.confirm_and_remove_model(
                model_version=model_version,
            )
            return ret
        ret = True
        for worker in self._worker_address_to_worker.values():
            ret = ret and await worker.confirm_and_remove_model(
                model_version=model_version,
            )
        return ret

    async def get_workers_info(self) -> List[Dict[str, Any]]:
        ret = []
        for worker in self._worker_address_to_worker.values():
            ret.append(await worker.get_workers_info())
        return ret

    async def get_supervisor_info(self) -> Dict[str, Any]:
        ret = {
            "supervisor_ip": self.address,
        }
        return ret

    async def trigger_exit(self) -> bool:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            logger.info(f"trigger exit error: {e}")
            return False
        return True

    async def abort_cluster(self) -> bool:
        ret = True
        for worker in self._worker_address_to_worker.values():
            ret = ret and await worker.trigger_exit()

        ret = ret and await self.trigger_exit()
        return ret

    @staticmethod
    def record_metrics(name, op, kwargs):
        record_metrics(name, op, kwargs)
