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

# 导入所需的模块和库
import asyncio
import inspect
import json
import logging
import multiprocessing
import os
import pprint
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import gradio as gr
import xoscar as xo
from aioprometheus import REGISTRY, MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse as StarletteJSONResponse
from starlette.responses import PlainTextResponse, RedirectResponse
from uvicorn import Config, Server
from xoscar.utils import get_next_port

from .._compat import BaseModel, Field
from .._version import get_versions
from ..constants import XINFERENCE_DEFAULT_ENDPOINT_PORT, XINFERENCE_DISABLE_METRICS
from ..core.event import Event, EventCollectorActor, EventType
from ..core.supervisor import SupervisorActor
from ..core.utils import json_dumps
from ..types import (
    SPECIAL_TOOL_PROMPT,
    ChatCompletion,
    ChatCompletionMessage,
    Completion,
    CreateChatCompletion,
    CreateCompletion,
    ImageList,
    PeftModelConfig,
    VideoList,
    max_tokens_field,
)
from .oauth2.auth_service import AuthService
from .oauth2.types import LoginUserForm

# 配置日志记录器
logger = logging.getLogger(__name__)


# 自定义 JSONResponse 类，用于处理 JSON 响应
class JSONResponse(StarletteJSONResponse):
    """
    自定义 JSONResponse 类，继承自 StarletteJSONResponse。
    用于处理 JSON 响应，使用自定义的 json_dumps 函数进行 JSON 序列化。
    """
    def render(self, content: Any) -> bytes:
        """
        将内容渲染为 JSON 格式的字节串。
        
        参数:
            content (Any): 要序列化的内容。
        
        返回:
            bytes: JSON 格式的字节串。
        """
        return json_dumps(content)


# 创建完成请求的模型类
class CreateCompletionRequest(CreateCompletion):
    """
    用于创建完成请求的模型类，继承自 CreateCompletion。
    定义了请求的结构和示例。
    """
    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


# 创建嵌入请求的模型类
class CreateEmbeddingRequest(BaseModel):
    """
    用于创建嵌入请求的模型类。
    定义了请求的结构，包括模型、输入和用户信息。
    """
    model: str
    input: Union[str, List[str], List[int], List[List[int]]] = Field(
        description="The input to embed."
    )
    user: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "input": "The food was delicious and the waiter...",
            }
        }


# 重新排序请求的模型类
class RerankRequest(BaseModel):
    """
    用于重新排序请求的模型类。
    定义了请求的结构，包括模型、查询、文档列表和其他可选参数。
    """
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False
    return_len: Optional[bool] = False
    max_chunks_per_doc: Optional[int] = None


# 文本到图像请求的模型类
class TextToImageRequest(BaseModel):
    """
    用于文本到图像请求的模型类。
    定义了请求的结构，包括模型、提示、生成数量、响应格式等参数。
    """
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024*1024"
    kwargs: Optional[str] = None
    user: Optional[str] = None


# 文本到视频请求的模型类
class TextToVideoRequest(BaseModel):
    """
    用于文本到视频请求的模型类。
    定义了请求的结构，包括模型、提示、生成数量等参数。
    """
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    kwargs: Optional[str] = None
    user: Optional[str] = None


# 语音请求的模型类
class SpeechRequest(BaseModel):
    """
    用于语音请求的模型类。
    定义了请求的结构，包括模型、输入文本、语音、响应格式等参数。
    """
    model: str
    input: str
    voice: Optional[str]
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False
    kwargs: Optional[str] = None


# 注册模型请求的模型类
class RegisterModelRequest(BaseModel):
    """
    用于注册模型请求的模型类。
    定义了请求的结构，包括模型、工作节点 IP 和持久化选项。
    """
    model: str
    worker_ip: Optional[str]
    persist: bool


# 构建 Gradio 接口请求的模型类（用于文本模型）
class BuildGradioInterfaceRequest(BaseModel):
    """
    用于构建 Gradio 接口请求的模型类（针对文本模型）。
    定义了请求的结构，包括模型类型、名称、大小、格式等多个属性。
    """
    model_type: str
    model_name: str
    model_size_in_billions: int
    model_format: str
    quantization: str
    context_length: int
    model_ability: List[str]
    model_description: str
    model_lang: List[str]


# 构建 Gradio 图像接口请求的模型类
class BuildGradioImageInterfaceRequest(BaseModel):
    """
    用于构建 Gradio 图像接口请求的模型类。
    定义了请求的结构，包括模型类型、名称、家族、ID 等多个属性。
    """
    model_type: str
    model_name: str
    model_family: str
    model_id: str
    controlnet: Union[None, List[Dict[str, Union[str, None]]]]
    model_revision: str
    model_ability: List[str]

class RESTfulAPI:
    """
    RESTfulAPI 类用于处理 Xinference 的 RESTful API 请求。
    该类负责初始化 API 路由、处理认证、管理模型实例等核心功能。
    """

    def __init__(
        self,
        supervisor_address: str,
        host: str,
        port: int,
        auth_config_file: Optional[str] = None,
    ):
        """
        初始化 RESTfulAPI 实例。

        参数:
        - supervisor_address: str, 监督者地址
        - host: str, API 服务器主机地址
        - port: int, API 服务器端口
        - auth_config_file: Optional[str], 认证配置文件路径，默认为 None

        初始化过程:
        1. 调用父类初始化方法
        2. 设置基本属性（地址、端口等）
        3. 初始化监督者和事件收集器引用为 None
        4. 创建认证服务实例
        5. 初始化 API 路由器和 FastAPI 应用实例
        """
        super().__init__()
        self._supervisor_address = supervisor_address
        self._host = host
        self._port = port
        self._supervisor_ref = None
        self._event_collector_ref = None
        self._auth_service = AuthService(auth_config_file)
        self._router = APIRouter()
        self._app = FastAPI()

    def is_authenticated(self):
        """
        检查是否启用了认证。

        返回:
        - bool: 如果认证配置存在则返回 True，否则返回 False
        """
        return False if self._auth_service.config is None else True

    @staticmethod
    def handle_request_limit_error(e: Exception):
        """
        处理请求限制错误。

        参数:
        - e: Exception, 捕获的异常

        行为:
        如果异常消息中包含 "Rate limit reached"，则抛出 429 状态码的 HTTPException
        """
        if "Rate limit reached" in str(e):
            raise HTTPException(status_code=429, detail=str(e))

    async def _get_supervisor_ref(self) -> xo.ActorRefType[SupervisorActor]:
        """
        获取或创建监督者 Actor 的引用。

        返回:
        - xo.ActorRefType[SupervisorActor]: 监督者 Actor 的引用

        行为:
        如果引用不存在，则创建新的引用并存储
        """
        if self._supervisor_ref is None:
            # 如果监督者引用不存在，创建一个新的引用
            # 通过supervisor的地址和uid获取引用
            self._supervisor_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=SupervisorActor.uid()
            )
        return self._supervisor_ref

    async def _get_event_collector_ref(self) -> xo.ActorRefType[EventCollectorActor]:
        """
        获取或创建事件收集器 Actor 的引用。

        返回:
        - xo.ActorRefType[EventCollectorActor]: 事件收集器 Actor 的引用

        行为:
        如果引用不存在，则创建新的引用并存储
        """
        if self._event_collector_ref is None:
            # 如果事件收集器引用不存在，创建一个新的引用
            # 通过supervisor的地址和EventCollectorActor的uid获取引用
            self._event_collector_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=EventCollectorActor.uid()
            )
        return self._event_collector_ref

    async def _report_error_event(self, model_uid: str, content: str):
        """
        报告错误事件。

        参数:
        - model_uid: str, 模型的唯一标识符
        - content: str, 错误内容

        行为:
        1. 尝试获取事件收集器引用
        2. 向事件收集器报告错误事件
        3. 如果报告失败，记录异常信息
        """
        try:
            event_collector_ref = await self._get_event_collector_ref()
            await event_collector_ref.report_event(
                model_uid,
                Event(
                    event_type=EventType.ERROR,
                    event_ts=int(time.time()),
                    event_content=content,
                ),
            )
        except Exception:
            logger.exception(
                "Report error event failed, model: %s, content: %s", model_uid, content
            )

    async def login_for_access_token(self, request: Request) -> JSONResponse:
        """
        处理用户登录并生成访问令牌。

        参数:
        - request: Request, FastAPI 请求对象

        返回:
        - JSONResponse: 包含访问令牌的 JSON 响应

        行为:
        1. 解析请求中的登录表单数据
        2. 调用认证服务生成用户令牌
        3. 返回包含令牌的 JSON 响应
        """
        form_data = LoginUserForm.parse_obj(await request.json())
        result = self._auth_service.generate_token_for_user(
            form_data.username, form_data.password
        )
        return JSONResponse(content=result)

    async def is_cluster_authenticated(self) -> JSONResponse:
        """
        检查集群是否启用了认证。

        返回:
        - JSONResponse: 包含认证状态的 JSON 响应

        行为:
        返回一个 JSON 响应，指示集群是否启用了认证
        """
        return JSONResponse(content={"auth": self.is_authenticated()})

    def serve(self, logging_conf: Optional[dict] = None):
        """
        启动 RESTful API 服务。

        参数:
        - logging_conf: Optional[dict], 日志配置，默认为 None

        行为:
        1. 添加 CORS 中间件
        2. 设置内部异常处理器
        3. 添加各种 API 路由，包括内部接口和用户接口
        4. 根据环境变量决定是否启用指标中间件
        5. 检查所有路由的返回值类型
        6. 设置静态文件服务
        """
        # 添加 CORS 中间件
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 设置内部异常处理器
        @self._app.exception_handler(500)
        async def internal_exception_handler(request: Request, exc: Exception):
            logger.exception("Handling request %s failed: %s", request.url, exc)
            return PlainTextResponse(
                status_code=500, content=f"Internal Server Error: {exc}"
            )

        # 内部接口路由配置
        
        # 获取服务状态
        self._router.add_api_route("/status", self.get_status, methods=["GET"])
        
        # 获取内置提示词列表
        # 注意:此路由需要先于 "/v1/models/{model_uid}" 注册,以避免冲突
        self._router.add_api_route(
            "/v1/models/prompts", self._get_builtin_prompts, methods=["GET"]
        )
        
        # 获取内置模型系列列表
        self._router.add_api_route(
            "/v1/models/families", self._get_builtin_families, methods=["GET"]
        )
        
        # 列出 vLLM 支持的模型系列
        self._router.add_api_route(
            "/v1/models/vllm-supported",
            self.list_vllm_supported_model_families,
            methods=["GET"],
        )
        
        # 获取集群设备信息
        # 如果启用了认证,则需要管理员权限
        self._router.add_api_route(
            "/v1/cluster/info",
            self.get_cluster_device_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 获取集群版本信息
        # 如果启用了认证,则需要管理员权限
        self._router.add_api_route(
            "/v1/cluster/version",
            self.get_cluster_version,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 获取设备数量
        # 如果启用了认证,则需要模型列表查看权限
        self._router.add_api_route(
            "/v1/cluster/devices",
            self._get_devices_count,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 获取服务地址
        # 获取supervisor的地址
        self._router.add_api_route("/v1/address", self.get_address, methods=["GET"])

        # 用户接口路由配置
        
        # 构建 Gradio 界面
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/ui/{model_uid}",
            self.build_gradio_interface,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 构建 Gradio 图像界面
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/ui/images/{model_uid}",
            self.build_gradio_images_interface,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 用户登录获取访问令牌
        self._router.add_api_route(
            "/token", self.login_for_access_token, methods=["POST"]
        )
        
        # 检查集群是否启用认证
        self._router.add_api_route(
            "/v1/cluster/auth", self.is_cluster_authenticated, methods=["GET"]
        )
        
        # 根据模型名称查询引擎
        # 如果启用了认证,则需要模型列表查看权限
        self._router.add_api_route(
            "/v1/engines/{model_name}",
            self.query_engines_by_model_name,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 获取运行实例信息
        # 如果启用了认证,则需要模型列表查看权限
        self._router.add_api_route(
            "/v1/models/instances",
            self.get_instance_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 获取模型版本列表
        # 如果启用了认证,则需要模型列表查看权限
        # http://127.0.0.1:9997/v1/models/vllm/qwen2-instruct/versions
        self._router.add_api_route(
            "/v1/models/{model_type}/{model_name}/versions",
            self.get_model_versions,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 列出所有模型
        # 如果启用了认证,则需要模型列表查看权限
        self._router.add_api_route(
            "/v1/models",
            self.list_models,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 描述特定模型
        # 如果启用了认证,则需要模型列表查看权限
        self._router.add_api_route(
            "/v1/models/{model_uid}",
            self.describe_model,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 获取模型事件
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/models/{model_uid}/events",
            self.get_model_events,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 中止特定请求
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/models/{model_uid}/requests/{request_id}/abort",
            self.abort_request,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 根据版本启动模型实例
        # 如果启用了认证,则需要模型启动权限
        self._router.add_api_route(
            "/v1/models/instance",
            self.launch_model_by_version,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:start"])]
                if self.is_authenticated()
                else None
            ),
        )
        # 启动模型实例
        # 如果启用了认证,则需要模型启动权限
        self._router.add_api_route(
            "/v1/models",
            self.launch_model,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:start"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 终止模型实例
        # 如果启用了认证,则需要模型停止权限
        self._router.add_api_route(
            "/v1/models/{model_uid}",
            self.terminate_model,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:stop"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 创建文本补全
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/completions",
            self.create_completion,
            methods=["POST"],
            response_model=Completion,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 创建文本嵌入
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/embeddings",
            self.create_embedding,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 重新排序
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/rerank",
            self.rerank,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 创建音频转录
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/audio/transcriptions",
            self.create_transcriptions,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 创建音频翻译
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/audio/translations",
            self.create_translations,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 创建语音
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/audio/speech",
            self.create_speech,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 生成图像
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/images/generations",
            self.create_images,
            methods=["POST"],
            response_model=ImageList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 创建图像变体
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/images/variations",
            self.create_variations,
            methods=["POST"],
            response_model=ImageList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        # 创建图像修复
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/images/inpainting",
            self.create_inpainting,
            methods=["POST"],
            response_model=ImageList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        
        
        # 生成视频
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/video/generations",
            self.create_videos,
            methods=["POST"],
            response_model=VideoList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        # 创建聊天补全
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/chat/completions",
            self.create_chat_completion,
            methods=["POST"],
            response_model=ChatCompletion,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 创建灵活推理
        # 如果启用了认证,则需要模型读取权限
        self._router.add_api_route(
            "/v1/flexible/infers",
            self.create_flexible_infer,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 自定义模型相关路由

        # 注册模型
        # 如果启用了认证,则需要模型注册权限
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}",
            self.register_model,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:register"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 注销模型
        # 如果启用了认证,则需要模型注销权限
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}/{model_name}",
            self.unregister_model,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:unregister"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 列出模型注册信息
        # 如果启用了认证,则需要模型列表查看权限
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}",
            self.list_model_registrations,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 获取特定模型的注册信息
        # 如果启用了认证,则需要模型列表查看权限
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}/{model_name}",
            self.get_model_registrations,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 列出缓存的模型
        # 如果启用了认证,则需要缓存列表查看权限
        self._router.add_api_route(
            "/v1/cache/models",
            self.list_cached_models,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["cache:list"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 列出模型文件
        # 如果启用了认证,则需要缓存列表查看权限
        self._router.add_api_route(
            "/v1/cache/models/files",
            self.list_model_files,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["cache:list"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 确认并移除模型
        # 如果启用了认证,则需要缓存删除权限
        self._router.add_api_route(
            "/v1/cache/models",
            self.confirm_and_remove_model,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["cache:delete"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 获取工作节点信息
        # 如果启用了认证,则需要管理员权限
        self._router.add_api_route(
            "/v1/workers",
            self.get_workers_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 获取监督节点信息
        # 如果启用了认证,则需要管理员权限
        self._router.add_api_route(
            "/v1/supervisor",
            self.get_supervisor_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 中止集群
        # 如果启用了认证,则需要管理员权限
        self._router.add_api_route(
            "/v1/clusters",
            self.abort_cluster,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )

        # 根据环境变量决定是否启用指标中间件
        if XINFERENCE_DISABLE_METRICS:
            # 如果禁用指标,则记录日志并直接包含路由
            logger.info(
                "Supervisor metrics is disabled due to the environment XINFERENCE_DISABLE_METRICS=1"
            )
            self._app.include_router(self._router)
        else:
            # 如果启用指标,则清除全局注册表,添加指标中间件和路由
            # 清除全局注册表是为了避免端口冲突时重复注册指标
            REGISTRY.clear()
            self._app.add_middleware(MetricsMiddleware)
            self._app.include_router(self._router)
            self._app.add_route("/metrics", metrics)

        # Check all the routes returns Response.
        # This is to avoid `jsonable_encoder` performance issue:
        # https://github.com/xorbitsai/inference/issues/647
        invalid_routes = []
        try:
            for router in self._router.routes:
                return_annotation = router.endpoint.__annotations__.get("return")
                if not inspect.isclass(return_annotation) or not issubclass(
                    return_annotation, Response
                ):
                    invalid_routes.append(
                        (router.path, router.endpoint, return_annotation)
                    )
        except Exception:
            pass  # 某些 Python 版本可能没有 __annotations__
        if invalid_routes:
            raise Exception(
                f"以下路由的返回值类型不是 Response:\n"
                f"{pprint.pformat(invalid_routes)}"
            )

        # 定义用于处理单页应用的静态文件类
        class SPAStaticFiles(StaticFiles):
            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                if response.status_code == 404:
                    response = await super().get_response(".", scope)
                return response

        # 尝试定位 UI 文件
        try:
            package_file_path = __import__("xinference").__file__
            assert package_file_path is not None
            lib_location = os.path.abspath(os.path.dirname(package_file_path))
            ui_location = os.path.join(lib_location, "web/ui/build/")
        except ImportError as e:
            raise ImportError(f"Xinference 导入不正确: {e}")

        # 如果 UI 文件存在,设置相关路由
        if os.path.exists(ui_location):
            @self._app.get("/")
            def read_main():
                response = RedirectResponse(url="/ui/")
                return response

            self._app.mount(
                "/ui/",
                SPAStaticFiles(directory=ui_location, html=True),
            )
        else:
            # 如果 UI 文件不存在,发出警告
            warnings.warn(
                f"""
            Xinference ui is not built at expected directory: {ui_location}
            To resolve this warning, navigate to {os.path.join(lib_location, "web/ui/")}
            And build the Xinference ui by running "npm run build"
            """
            )

        # 配置并运行服务器
        config = Config(
            app=self._app, host=self._host, port=self._port, log_config=logging_conf
        )
        server = Server(config)
        server.run()

    async def _get_builtin_prompts(self) -> JSONResponse:
        """
        获取内置提示词列表
        
        返回:
            JSONResponse: 包含内置提示词数据的 JSON 响应
        
        异常:
            HTTPException: 如果获取过程中发生错误,则抛出 500 状态码的异常
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_prompts()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_builtin_families(self) -> JSONResponse:
        """
        获取内置模型系列列表
        
        返回:
            JSONResponse: 包含内置模型系列数据的 JSON 响应
        
        异常:
            HTTPException: 如果获取过程中发生错误,则抛出 500 状态码的异常
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_families()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_devices_count(self) -> JSONResponse:
        """
        获取设备数量
        
        返回:
            JSONResponse: 包含设备数量数据的 JSON 响应
        
        异常:
            HTTPException: 如果获取过程中发生错误,则抛出 500 状态码的异常
        """
        try:
            # 从监督者获取设备数量数据
            data = await (await self._get_supervisor_ref()).get_devices_count()
            return JSONResponse(content=data)
        except Exception as e:
            # 记录错误日志
            logger.error(e, exc_info=True)
            # 抛出HTTP 500错误
            raise HTTPException(status_code=500, detail=str(e))
    async def get_status(self) -> JSONResponse:
        """
        获取服务状态
        
        返回:
            JSONResponse: 包含服务状态数据的 JSON 响应
        
        异常:
            HTTPException: 如果获取过程中发生错误,则抛出 500 状态码的异常
        """
        try:
            # 从监督者获取状态数据
            data = await (await self._get_supervisor_ref()).get_status()
            # 将数据包装成JSON响应返回
            return JSONResponse(content=data)
        except Exception as e:
            # 记录错误日志
            logger.error(e, exc_info=True)
            # 抛出HTTP 500错误
            raise HTTPException(status_code=500, detail=str(e))

    async def list_models(self) -> JSONResponse:
        """
        列出所有可用模型
        
        返回:
            JSONResponse: 包含模型列表的 JSON 响应,每个模型包含 id、object、created、owned_by 等信息
        
        异常:
            HTTPException: 如果获取过程中发生错误,则抛出 500 状态码的异常
        """
        try:
            models = await (await self._get_supervisor_ref()).list_models()

            model_list = []
            for model_id, model_info in models.items():
                model_list.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "xinference",
                        **model_info,
                    }
                )
            response = {"object": "list", "data": model_list}

            return JSONResponse(content=response)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def describe_model(self, model_uid: str) -> JSONResponse:
        """
        描述指定模型的详细信息。

        参数:
            model_uid (str): 模型的唯一标识符。

        返回:
            JSONResponse: 包含模型详细信息的 JSON 响应。

        异常:
            HTTPException: 
                - 400 状态码: 如果提供的 model_uid 无效。
                - 500 状态码: 如果在获取模型信息过程中发生其他错误。
        """
        try:
            # 通过 supervisor 获取模型的详细信息
            data = await (await self._get_supervisor_ref()).describe_model(model_uid)
            # 返回模型描述的JSON响应
            return JSONResponse(content=data)
        except ValueError as ve:
            # 捕获并处理无效的 model_uid 错误
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 捕获并处理其他类型的错误
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def launch_model(
        self, request: Request, wait_ready: bool = Query(True)
    ) -> JSONResponse:
        """
        启动一个新的模型实例。

        参数:
            request (Request): FastAPI 请求对象，包含模型启动所需的参数。
            wait_ready (bool): 是否等待模型准备就绪，默认为 True。

        返回:
            JSONResponse: 包含启动的模型 UID 的 JSON 响应。

        异常:
            HTTPException: 
                - 400 状态码: 如果请求参数无效。
                - 503 状态码: 如果模型启动过程中发生运行时错误。
                - 500 状态码: 如果发生其他类型的错误。
        """
        # 从请求中提取模型参数
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_name = payload.get("model_name")
        model_engine = payload.get("model_engine")
        model_size_in_billions = payload.get("model_size_in_billions")
        model_format = payload.get("model_format")
        quantization = payload.get("quantization")
        model_type = payload.get("model_type", "LLM")
        replica = payload.get("replica", 1)
        n_gpu = payload.get("n_gpu", "auto")
        request_limits = payload.get("request_limits", None)
        peft_model_config = payload.get("peft_model_config", None)
        worker_ip = payload.get("worker_ip", None)
        gpu_idx = payload.get("gpu_idx", None)
        download_hub = payload.get("download_hub", None)
        model_path = payload.get("model_path", None)

        # 定义需要排除的键
        exclude_keys = {
            "model_uid",
            "model_name",
            "model_engine",
            "model_size_in_billions",
            "model_format",
            "quantization",
            "model_type",
            "replica",
            "n_gpu",
            "request_limits",
            "peft_model_config",
            "worker_ip",
            "gpu_idx",
            "download_hub",
            "model_path",
        }

        # 构建额外参数字典
        kwargs = {
            key: value for key, value in payload.items() if key not in exclude_keys
        }

        # 验证必要参数
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the `model_name` field.",
            )
        if not model_engine and model_type == "LLM":
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the `model_engine` field.",
            )

        # 处理 GPU 索引
        # 如果 len(gpu_idx) % replica 不等于 0，则抛出一个 HTTPException，状态码为 400，
        # 提示用户分配的 GPU 数量必须是副本数量的倍数。


        if isinstance(gpu_idx, int):
            gpu_idx = [gpu_idx]
        if gpu_idx:
            if len(gpu_idx) % replica:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid input. Allocated gpu must be a multiple of replica.",
                )

        # 处理 PEFT 模型配置
        if peft_model_config is not None:
            peft_model_config = PeftModelConfig.from_dict(peft_model_config)
        else:
            peft_model_config = None

        try:
            # 通过 supervisor 启动内置模型
            model_uid = await (await self._get_supervisor_ref()).launch_builtin_model(
                model_uid=model_uid,
                model_name=model_name,
                model_engine=model_engine,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                model_type=model_type,
                replica=replica,
                n_gpu=n_gpu,
                request_limits=request_limits,
                wait_ready=wait_ready,
                peft_model_config=peft_model_config,
                worker_ip=worker_ip,
                gpu_idx=gpu_idx,
                download_hub=download_hub,
                model_path=model_path,
                **kwargs,
            )
        except ValueError as ve:
            # 处理参数验证错误
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(str(re), exc_info=True)
            raise HTTPException(status_code=503, detail=str(re))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        # 返回启动的模型 UID
        return JSONResponse(content={"model_uid": model_uid})

    async def get_instance_info(
        self,
        model_name: Optional[str] = Query(None),
        model_uid: Optional[str] = Query(None),
    ) -> JSONResponse:
        """
        获取模型实例的信息。

        参数:
            model_name (Optional[str]): 模型名称，可选。
            model_uid (Optional[str]): 模型的唯一标识符，可选。

        返回:
            JSONResponse: 包含模型实例信息的 JSON 响应。

        异常:
            HTTPException: 
                - 500 状态码: 如果在获取实例信息过程中发生错误。
        """
        try:
            # 通过 supervisor 获取实例信息
            infos = await (await self._get_supervisor_ref()).get_instance_info(
                model_name, model_uid
            )
        except Exception as e:
            # 处理获取实例信息过程中的错误
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=infos)
    async def launch_model_by_version(
        self, request: Request, wait_ready: bool = Query(True)
    ) -> JSONResponse:
        """
        根据版本启动模型实例。

        参数:
            request (Request): FastAPI 请求对象，包含模型启动所需的参数。
            wait_ready (bool): 是否等待模型准备就绪，默认为 True。

        返回:
            JSONResponse: 包含启动的模型 UID 的 JSON 响应。

        异常:
            HTTPException: 
                - 500 状态码: 如果在启动模型过程中发生错误。
        """
        # 从请求中提取模型参数
        payload = await request.json()
        
        # 获取模型相关参数
        model_uid = payload.get("model_uid")
        model_engine = payload.get("model_engine")
        model_type = payload.get("model_type")
        model_version = payload.get("model_version")
        replica = payload.get("replica", 1)  # 默认值为1
        n_gpu = payload.get("n_gpu", "auto")  # 默认值为"auto"

        try:
            # 通过 supervisor 根据版本启动模型
            model_uid = await (
                await self._get_supervisor_ref()
            ).launch_model_by_version(
                model_uid=model_uid,
                model_engine=model_engine,
                model_type=model_type,
                model_version=model_version,
                replica=replica,
                n_gpu=n_gpu,
                wait_ready=wait_ready,
            )
        except Exception as e:
            # 处理启动过程中的错误
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        
        # 返回包含模型UID的JSON响应
        return JSONResponse(content={"model_uid": model_uid})

    async def get_model_versions(
        self, model_type: str, model_name: str
    ) -> JSONResponse:
        """
        获取指定模型类型和名称的所有版本信息。

        参数:
            model_type (str): 模型类型。
            model_name (str): 模型名称。

        返回:
            JSONResponse: 包含模型版本信息的 JSON 响应。

        异常:
            HTTPException: 
                - 500 状态码: 如果在获取模型版本信息过程中发生错误。
        """
        try:
            # 通过 supervisor 获取模型版本信息
            content = await (await self._get_supervisor_ref()).get_model_versions(
                model_type, model_name
            )
            return JSONResponse(content=content)
        except Exception as e:
            # 处理获取版本信息过程中的错误
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def build_gradio_interface(
        self, model_uid: str, request: Request
    ) -> JSONResponse:
        """
        Separate build_interface with launch_model
        build_interface requires RESTful Client for API calls
        but calling API in async function does not return
        
        为指定的模型构建 Gradio 接口。

        参数:
            model_uid (str): 模型的唯一标识符。
            request (Request): FastAPI 请求对象，包含构建接口所需的参数。

        返回:
            JSONResponse: 包含构建的模型 UID 的 JSON 响应。

        异常:
            HTTPException: 
                - 400 状态码: 如果请求参数无效。
                - 500 状态码: 如果在构建接口过程中发生其他错误。

        注意:
            此方法需要单独的 RESTful 客户端进行 API 调用，因为在异步函数中调用 API 不会返回结果。
        """
        # 解析请求体
        payload = await request.json()
        body = BuildGradioInterfaceRequest.parse_obj(payload)
        assert self._app is not None
        assert body.model_type == "LLM"

        # asyncio.Lock() behaves differently in 3.9 than 3.10+
        # A event loop is required in 3.9 but not 3.10+
        if sys.version_info < (3, 10):
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                warnings.warn(
                    "asyncio.Lock() requires an event loop in Python 3.9"
                    + "a placeholder event loop has been created"
                )
                asyncio.set_event_loop(asyncio.new_event_loop())

        from ..core.chat_interface import GradioInterface

        try:
            # 获取访问令牌并设置内部主机
            access_token = request.headers.get("Authorization")
            internal_host = "localhost" if self._host == "0.0.0.0" else self._host
            
            # 构建 Gradio 接口
            interface = GradioInterface(
                endpoint=f"http://{internal_host}:{self._port}",
                model_uid=model_uid,
                model_name=body.model_name,
                model_size_in_billions=body.model_size_in_billions,
                model_type=body.model_type,
                model_format=body.model_format,
                quantization=body.quantization,
                context_length=body.context_length,
                model_ability=body.model_ability,
                model_description=body.model_description,
                model_lang=body.model_lang,
                access_token=access_token,
            ).build()
            
            # 将 Gradio 应用挂载到 FastAPI 应用
            gr.mount_gradio_app(self._app, interface, f"/{model_uid}")
        except ValueError as ve:
            # 处理参数验证错误
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        # 返回包含模型UID的JSON响应
        return JSONResponse(content={"model_uid": model_uid})

    async def build_gradio_images_interface(
        self, model_uid: str, request: Request
    ) -> JSONResponse:
        """
        为图像处理模型构建 Gradio 接口。

        参数:
            model_uid (str): 模型的唯一标识符。
            request (Request): FastAPI 请求对象，包含构建接口所需的参数。

        返回:
            JSONResponse: 包含模型 UID 的 JSON 响应。

        异常:
            HTTPException: 
                - 400 状态码: 如果请求参数无效。
                - 500 状态码: 如果在构建接口过程中发生其他错误。
        """
        # 解析请求体并验证模型类型
        payload = await request.json()
        body = BuildGradioImageInterfaceRequest.parse_obj(payload)
        assert self._app is not None
        assert body.model_type == "image"

        # asyncio.Lock() behaves differently in 3.9 than 3.10+
        # A event loop is required in 3.9 but not 3.10+
        if sys.version_info < (3, 10):
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                warnings.warn(
                    "asyncio.Lock() requires an event loop in Python 3.9"
                    + "a placeholder event loop has been created"
                )
                asyncio.set_event_loop(asyncio.new_event_loop())

        from ..core.image_interface import ImageInterface

        try:
            # 获取访问令牌并设置内部主机
            access_token = request.headers.get("Authorization")
            internal_host = "localhost" if self._host == "0.0.0.0" else self._host
            
            # 构建图像接口
            interface = ImageInterface(
                endpoint=f"http://{internal_host}:{self._port}",
                model_uid=model_uid,
                model_family=body.model_family,
                model_name=body.model_name,
                model_id=body.model_id,
                model_revision=body.model_revision,
                controlnet=body.controlnet,
                access_token=access_token,
                model_ability=body.model_ability,
            ).build()

            # 将 Gradio 应用挂载到 FastAPI 应用
            gr.mount_gradio_app(self._app, interface, f"/{model_uid}")
        except ValueError as ve:
            # 处理参数验证错误
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={"model_uid": model_uid})

    async def terminate_model(self, model_uid: str) -> JSONResponse:
        """
        终止指定的模型实例。

        参数:
            model_uid (str): 要终止的模型的唯一标识符。

        返回:
            JSONResponse: 空内容的 JSON 响应，表示操作成功。

        异常:
            HTTPException: 
                - 400 状态码: 如果提供的 model_uid 无效。
                - 500 状态码: 如果在终止模型过程中发生其他错误。
        """
        try:
            assert self._app is not None
            # 通过 supervisor 终止模型
            await (await self._get_supervisor_ref()).terminate_model(model_uid)
            
            # 从 FastAPI 应用中移除相关的路由
            self._app.router.routes = [
                route
                for route in self._app.router.routes
                if not (
                    hasattr(route, "path")
                    and isinstance(route.path, str)
                    and route.path == "/" + model_uid
                )
            ]
        except ValueError as ve:
            # 处理无效的 model_uid 错误
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def get_address(self) -> JSONResponse:
        """
        获取 supervisor 的地址。

        返回:
            JSONResponse: 包含 supervisor 地址的 JSON 响应。
        """
        return JSONResponse(content=self._supervisor_address)

    async def create_completion(self, request: Request) -> Response:
        """
        创建文本补全。

        参数:
            request (Request): FastAPI 请求对象，包含补全所需的参数。

        返回:
            Response: 包含补全结果的响应。

        异常:
            HTTPException: 
                - 400 状态码: 如果请求参数无效或模型不存在。
                - 500 状态码: 如果在补全过程中发生其他错误。
                - 501 状态码: 如果请求了未实现的功能。
        """
        # 解析请求体
        raw_body = await request.json()
        body = CreateCompletionRequest.parse_obj(raw_body)
        
        # 提取需要的参数
        exclude = {
            "prompt",
            "model",
            "n",
            "best_of",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        raw_kwargs = {k: v for k, v in raw_body.items() if k not in exclude}
        kwargs = body.dict(exclude_unset=True, exclude=exclude)

        # TODO: Decide if this default value override is necessary #1061
        if body.max_tokens is None:
            kwargs["max_tokens"] = max_tokens_field.default

        # 检查是否使用了未实现的 logit_bias 功能
        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        model_uid = body.model

        try:
            # 获取模型实例
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        if body.stream:
            # 处理流式响应
            async def stream_results():
                iterator = None
                try:
                    try:
                        iterator = await model.generate(
                            body.prompt, kwargs, raw_params=raw_kwargs
                        )
                    except RuntimeError as re:
                        self.handle_request_limit_error(re)
                    async for item in iterator:
                        yield item
                except asyncio.CancelledError:
                    logger.info(
                        f"Disconnected from client (via refresh/close) {request.client} during generate."
                    )
                    return
                except Exception as ex:
                    logger.exception("Completion stream got an error: %s", ex)
                    await self._report_error_event(model_uid, str(ex))
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": str(ex)}))
                    return

            return EventSourceResponse(stream_results())
        else:
            # 处理非流式响应
            try:
                data = await model.generate(body.prompt, kwargs, raw_params=raw_kwargs)
                return Response(data, media_type="application/json")
            except Exception as e:
                logger.error(e, exc_info=True)
                await self._report_error_event(model_uid, str(e))
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    async def create_embedding(self, request: Request) -> Response:
        """
        创建文本嵌入。

        参数:
            request (Request): FastAPI 请求对象，包含创建嵌入所需的参数。

        返回:
            Response: 包含嵌入结果的响应。

        异常:
            HTTPException: 
                - 400 状态码: 如果请求参数无效或模型不存在。
                - 500 状态码: 如果在创建嵌入过程中发生其他错误。
        """
        # 解析请求体
        payload = await request.json()
        body = CreateEmbeddingRequest.parse_obj(payload)
        model_uid = body.model
        
        # 提取需要的参数,过滤不需要的参数
        # {
        #     "model": "model_uid_123",
        #     "input": "some_input",
        #     "user": "user_1",
        #     "encoding_format": "utf-8",
        #     "param1": "value1",
        #     "param2": "value2"
        # }
        
        # {
        #     "param1": "value1",
        #     "param2": "value2"
        # }
        exclude = {
            "model",
            "input",
            "user",
            "encoding_format",
        }
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        try:
            # 获取模型实例
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 创建嵌入
            embedding = await model.create_embedding(body.input, **kwargs)
            return Response(embedding, media_type="application/json")
        except RuntimeError as re:
            # 处理请求限制错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def rerank(self, request: Request) -> Response:
        """
        对文档进行重新排序。

        参数:
            request (Request): FastAPI 请求对象，包含重新排序所需的参数。

        返回:
            Response: 包含重新排序结果的响应。

        异常:
            HTTPException: 
                - 400 状态码: 如果请求参数无效或模型不存在。
                - 500 状态码: 如果在重新排序过程中发生其他错误。
        """
        # 解析请求体
        payload = await request.json()
        body = RerankRequest.parse_obj(payload)
        model_uid = body.model
        
        # 提取额外参数
        kwargs = {
            key: value
            for key, value in payload.items()
            if key not in RerankRequest.__annotations__.keys()
        }

        try:
            # 获取模型实例
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 执行重新排序
            scores = await model.rerank(
                body.documents,
                body.query,
                top_n=body.top_n,
                max_chunks_per_doc=body.max_chunks_per_doc,
                return_documents=body.return_documents,
                return_len=body.return_len,
                **kwargs,
            )
            return Response(scores, media_type="application/json")
        except RuntimeError as re:
            # 处理请求限制错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他类型的错误
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))
    # 创建语音转录接口
    async def create_transcriptions(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        """
        处理语音转录请求的异步方法。

        参数:
        - request: FastAPI的请求对象
        - model: 使用的模型标识符
        - file: 上传的音频文件
        - language: 音频语言（可选）
        - prompt: 转录提示（可选）
        - response_format: 响应格式，默认为json
        - temperature: 采样温度，默认为0
        - kwargs: 额外的关键字参数（JSON字符串）

        返回:
        - Response: 包含转录结果的响应对象
        """
        # 从请求表单中获取时间戳粒度参数
        form = await request.form()
        timestamp_granularities = form.get("timestamp_granularities[]")
        if timestamp_granularities:
            timestamp_granularities = [timestamp_granularities]
        
        model_uid = model
        try:
            # 获取模型引用
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 解析额外的关键字参数
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            
            # 执行转录
            transcription = await model_ref.transcriptions(
                audio=await file.read(),
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                **parsed_kwargs,
            )
            return Response(content=transcription, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 创建翻译接口
    async def create_translations(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        """
        处理语音翻译请求的异步方法。

        参数:
        - request: FastAPI的请求对象
        - model: 使用的模型标识符
        - file: 上传的音频文件
        - language: 目标语言（可选）
        - prompt: 翻译提示（可选）
        - response_format: 响应格式，默认为json
        - temperature: 采样温度，默认为0
        - kwargs: 额外的关键字参数（JSON字符串）

        返回:
        - Response: 包含翻译结果的响应对象
        """
        # 从请求表单中获取时间戳粒度参数
        form = await request.form()
        timestamp_granularities = form.get("timestamp_granularities[]")
        if timestamp_granularities:
            timestamp_granularities = [timestamp_granularities]
        
        model_uid = model
        try:
            # 获取模型引用
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 解析额外的关键字参数
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            
            # 执行翻译
            translation = await model_ref.translations(
                audio=await file.read(),
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                **parsed_kwargs,
            )
            return Response(content=translation, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 创建语音生成接口
    async def create_speech(
        self,
        request: Request,
        prompt_speech: Optional[UploadFile] = File(
            None, media_type="application/octet-stream"
        ),
    ) -> Response:
        """
        处理语音生成请求的异步方法。

        参数:
        - request: FastAPI的请求对象
        - prompt_speech: 可选的语音提示文件

        返回:
        - Response: 包含生成的语音数据的响应对象
        """
        # 根据是否有语音提示文件来决定如何解析请求
        if prompt_speech:
            f = await request.form()
        else:
            f = await request.json()
        
        # 解析请求体
        body = SpeechRequest.parse_obj(f)
        model_uid = body.model
        try:
            # 获取模型引用
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 解析额外的关键字参数
            if body.kwargs is not None:
                parsed_kwargs = json.loads(body.kwargs)
            else:
                parsed_kwargs = {}
            
            # 如果有语音提示文件，将其添加到参数中
            if prompt_speech is not None:
                parsed_kwargs["prompt_speech"] = await prompt_speech.read()
            
            # 执行语音生成
            out = await model.speech(
                input=body.input,
                voice=body.voice,
                response_format=body.response_format,
                speed=body.speed,
                stream=body.stream,
                **parsed_kwargs,
            )
            
            # 根据是否为流式响应返回不同类型的响应
            if body.stream:
                return EventSourceResponse(
                    media_type="application/octet-stream", content=out
                )
            else:
                return Response(media_type="application/octet-stream", content=out)
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 创建图像生成接口
    async def create_images(self, request: Request) -> Response:
        """
        处理图像生成请求的异步方法。

        参数:
        - request: FastAPI的请求对象

        返回:
        - Response: 包含生成的图像数据的响应对象
        """
        # 解析请求体
        body = TextToImageRequest.parse_obj(await request.json())
        model_uid = body.model
        try:
            # 获取模型引用
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 解析额外的关键字参数
            kwargs = json.loads(body.kwargs) if body.kwargs else {}
            
            # 执行文本到图像的转换
            image_list = await model.text_to_image(
                prompt=body.prompt,
                n=body.n,
                size=body.size,
                response_format=body.response_format,
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Stable Diffusion API选项接口
    async def sdapi_options(self, request: Request) -> Response:
        """
        处理Stable Diffusion API选项请求的异步方法。

        参数:
        - request: FastAPI的请求对象

        返回:
        - Response: 空响应对象，表示操作成功
        """
        # 解析请求体
        body = SDAPIOptionsRequest.parse_obj(await request.json())
        model_uid = body.sd_model_checkpoint

        try:
            # 检查模型是否存在
            if not model_uid:
                raise ValueError("Unknown model")
            await (await self._get_supervisor_ref()).get_model(model_uid)
            return Response()
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Stable Diffusion API文本到图像接口
    async def sdapi_txt2img(self, request: Request) -> Response:
        """
        处理Stable Diffusion API文本到图像请求的异步方法。

        参数:
        - request: FastAPI的请求对象

        返回:
        - Response: 包含生成的图像数据的响应对象
        """
        # 解析请求体
        body = SDAPITxt2imgRequst.parse_obj(await request.json())
        model_uid = body.model or body.override_settings.get("sd_model_checkpoint")

        try:
            # 检查模型是否存在并获取模型引用
            if not model_uid:
                raise ValueError("Unknown model")
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 准备参数
            kwargs = dict(body)
            kwargs.update(json.loads(body.kwargs) if body.kwargs else {})
            
            # 执行文本到图像的转换
            image_list = await model.txt2img(
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))
    # 图像变体生成接口
    async def create_variations(
        self,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        response_format: Optional[str] = Form("url"),
        size: Optional[str] = Form(None),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        """
        处理图像变体生成请求的异步方法。

        参数:
        - model: 使用的模型标识符
        - image: 上传的原始图像文件
        - prompt: 正向提示词，用于引导图像生成
        - negative_prompt: 负向提示词，用于避免特定元素出现
        - n: 生成的图像变体数量
        - response_format: 响应格式，默认为"url"
        - size: 生成图像的尺寸
        - kwargs: 额外的关键字参数（JSON字符串）

        返回:
        - Response: 包含生成的图像变体数据的响应对象
        """
        model_uid = model
        try:
            # 获取模型引用
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 解析额外的关键字参数
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            
            # 执行图像到图像的转换
            image_list = await model_ref.image_to_image(
                image=Image.open(image.file),
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                response_format=response_format,
                **parsed_kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 图像修复接口
    async def create_inpainting(
        self,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        mask_image: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        response_format: Optional[str] = Form("url"),
        size: Optional[str] = Form(None),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        """
        处理图像修复请求的异步方法。

        参数:
        - model: 使用的模型标识符
        - image: 上传的原始图像文件
        - mask_image: 上传的遮罩图像文件
        - prompt: 正向提示词，用于引导图像修复
        - negative_prompt: 负向提示词，用于避免特定元素出现
        - n: 生成的修复图像数量
        - response_format: 响应格式，默认为"url"
        - size: 生成图像的尺寸
        - kwargs: 额外的关键字参数（JSON字符串）

        返回:
        - Response: 包含修复后的图像数据的响应对象
        """
        model_uid = model
        try:
            # 获取模型引用
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 解析额外的关键字参数
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            
            # 打开原始图像和遮罩图像
            im = Image.open(image.file)
            mask_im = Image.open(mask_image.file)
            
            # 如果未指定尺寸，使用原始图像的尺寸
            if not size:
                w, h = im.size
                size = f"{w}*{h}"
            
            # 执行图像修复
            image_list = await model_ref.inpainting(
                image=im,
                mask_image=mask_im,
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                response_format=response_format,
                **parsed_kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 灵活推理接口
    async def create_flexible_infer(self, request: Request) -> Response:
        """
        处理灵活推理请求的异步方法。

        参数:
        - request: FastAPI的请求对象

        返回:
        - Response: 包含推理结果的响应对象
        """
        # 解析请求体
        payload = await request.json()

        # 获取模型标识符
        model_uid = payload.get("model")

        # 排除特定键，构建kwargs字典
        exclude = {
            "model",
        }
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        try:
            # 获取模型引用
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 执行推理
            result = await model.infer(**kwargs)
            return Response(result, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 视频生成接口
    async def create_videos(self, request: Request) -> Response:
        """
        处理文本到视频生成请求的异步方法。

        参数:
        - request: FastAPI的请求对象

        返回:
        - Response: 包含生成的视频数据的响应对象
        """
        # 解析请求体
        body = TextToVideoRequest.parse_obj(await request.json())
        model_uid = body.model
        try:
            # 获取模型引用
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 解析额外的关键字参数
            kwargs = json.loads(body.kwargs) if body.kwargs else {}
            
            # 执行文本到视频的转换
            video_list = await model.text_to_video(
                prompt=body.prompt,
                n=body.n,
                **kwargs,
            )
            return Response(content=video_list, media_type="application/json")
        except RuntimeError as re:
            # 处理运行时错误
            logger.error(re, exc_info=True)
            await self._report_error_event(model_uid, str(re))
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def create_chat_completion(self, request: Request) -> Response:
        """
        创建聊天完成的异步方法。

        参数:
        - request: FastAPI的请求对象，包含聊天完成所需的参数

        返回:
        - Response: 包含聊天完成结果的响应对象

        异常:
        - HTTPException: 当请求参数无效或处理过程中出现错误时抛出
        """
        # 解析请求体
        raw_body = await request.json()
        body = CreateChatCompletion.parse_obj(raw_body)
        
        # 提取不需要传递给模型的参数
        exclude = {
            "prompt",
            "model",
            "n",
            "messages",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        # 构建传递给模型的参数字典
        raw_kwargs = {k: v for k, v in raw_body.items() if k not in exclude}
        kwargs = body.dict(exclude_unset=True, exclude=exclude)

        # TODO: Decide if this default value override is necessary #1061
        if body.max_tokens is None:
            kwargs["max_tokens"] = max_tokens_field.default

        # 检查是否使用了未实现的logit_bias功能
        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        # 提取消息列表
        messages = body.messages and list(body.messages) or None

        # 验证消息列表的有效性
        if not messages or messages[-1].get("role") not in ["user", "system", "tool"]:
            raise HTTPException(
                status_code=400, detail="Invalid input. Please specify the prompt."
            )

        system_messages: List["ChatCompletionMessage"] = []
        system_messages_contents = []
        non_system_messages = []
        for msg in messages:
            assert (
                msg.get("content") != SPECIAL_TOOL_PROMPT
            ), f"Invalid message content {SPECIAL_TOOL_PROMPT}"
            if msg["role"] == "system":
                system_messages_contents.append(msg["content"])
            else:
                non_system_messages.append(msg)
        system_messages.append(
            {"role": "system", "content": ". ".join(system_messages_contents)}
        )

        has_tool_message = messages[-1].get("role") == "tool"
        if has_tool_message:
            prompt = SPECIAL_TOOL_PROMPT
            system_prompt = system_messages[0]["content"] if system_messages else None
            chat_history = non_system_messages  # exclude the prompt
        else:
            prompt = None
            if non_system_messages:
                prompt = non_system_messages[-1]["content"]
            system_prompt = system_messages[0]["content"] if system_messages else None
            chat_history = non_system_messages[:-1]  # exclude the prompt

        model_uid = body.model

        try:
            # 获取模型实例
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            # 处理模型不存在的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            # 获取模型描述
            desc = await (await self._get_supervisor_ref()).describe_model(model_uid)
        except ValueError as ve:
            # 处理模型描述获取失败的错误
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        # 导入工具调用相关的模型家族
        from ..model.llm.utils import GLM4_TOOL_CALL_FAMILY, QWEN_TOOL_CALL_FAMILY

        model_family = desc.get("model_family", "")
        function_call_models = (
            ["gorilla-openfunctions-v1"] + QWEN_TOOL_CALL_FAMILY + GLM4_TOOL_CALL_FAMILY
        )

        # 检查模型是否支持工具调用
        if model_family not in function_call_models:
            if body.tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only {function_call_models} support tool calls",
                )
            if has_tool_message:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only {function_call_models} support tool messages",
                )
        
        # 检查是否支持流式工具调用
        if body.tools and body.stream:
            is_vllm = await model.is_vllm_backend()

            if not (
                (is_vllm and model_family in QWEN_TOOL_CALL_FAMILY)
                or (not is_vllm and model_family in GLM4_TOOL_CALL_FAMILY)
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Streaming support for tool calls is available only when using "
                    "Qwen models with vLLM backend or GLM4-chat models without vLLM backend.",
                )

        if body.stream:
            # 处理流式响应
            async def stream_results():
                iterator = None
                try:
                    try:
                        # 获取聊天结果迭代器
                        iterator = await model.chat(
                            prompt,
                            system_prompt,
                            chat_history,
                            kwargs,
                            raw_params=raw_kwargs,
                        )
                    except RuntimeError as re:
                        # 处理运行时错误
                        await self._report_error_event(model_uid, str(re))
                        self.handle_request_limit_error(re)
                    # 逐个yield结果
                    async for item in iterator:
                        yield item
                    yield "[DONE]"
                # Note that asyncio.CancelledError does not inherit from Exception.
                # When the user uses ctrl+c to cancel the streaming chat, asyncio.CancelledError would be triggered.
                # See https://github.com/sysid/sse-starlette/blob/main/examples/example.py#L48
                except asyncio.CancelledError:
                    # 处理用户取消请求
                    logger.info(
                        f"Disconnected from client (via refresh/close) {request.client} during chat."
                    )
                    # See https://github.com/sysid/sse-starlette/blob/main/examples/error_handling.py#L13
                    # Use return to stop the generator from continuing.
                    # TODO: Cannot yield here. Yield here would leads to error for the next streaming request.
                    return
                except Exception as ex:
                    # 处理其他异常
                    logger.exception("Chat completion stream got an error: %s", ex)
                    await self._report_error_event(model_uid, str(ex))
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": str(ex)}))
                    return

            return EventSourceResponse(stream_results())
        else:
            # 处理非流式响应
            try:
                # 获取聊天结果
                data = await model.chat(
                    prompt,
                    system_prompt,
                    chat_history,
                    kwargs,
                    raw_params=raw_kwargs,
                )
                return Response(content=data, media_type="application/json")
            except Exception as e:
                # 处理异常
                logger.error(e, exc_info=True)
                await self._report_error_event(model_uid, str(e))
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    # TODO quwery_engines_by_model_name
    async def query_engines_by_model_name(self, model_name: str) -> JSONResponse:
        """
        根据模型名称查询引擎。

        参数:
        - model_name: 要查询的模型名称

        返回:
        - JSONResponse: 包含查询结果的JSON响应

        异常:
        - HTTPException: 当查询过程中出现错误时抛出
        """
        try:
            # 通过supervisor查询引擎
            content = await (
                await self._get_supervisor_ref()
            ).query_engines_by_model_name(model_name)
            return JSONResponse(content=content)
        except ValueError as re:
            # 处理值错误
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def register_model(self, model_type: str, request: Request) -> JSONResponse:
        """
        注册模型的异步方法。

        参数:
        - model_type: 模型类型
        - request: FastAPI的请求对象，包含注册所需的参数

        返回:
        - JSONResponse: 表示注册成功的空JSON响应

        异常:
        - HTTPException: 当注册过程中出现错误时抛出
        """
        # 解析请求体
        body = RegisterModelRequest.parse_obj(await request.json())
        model = body.model
        worker_ip = body.worker_ip
        persist = body.persist

        try:
            # 通过supervisor注册模型
            await (await self._get_supervisor_ref()).register_model(
                model_type, model, persist, worker_ip
            )
        except ValueError as re:
            # 处理值错误
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def unregister_model(self, model_type: str, model_name: str) -> JSONResponse:
        """
        注销模型的异步方法。

        参数:
        - model_type: 模型类型
        - model_name: 要注销的模型名称

        返回:
        - JSONResponse: 表示注销成功的空JSON响应

        异常:
        - HTTPException: 当注销过程中出现错误时抛出
        """
        try:
            # 通过supervisor注销模型
            await (await self._get_supervisor_ref()).unregister_model(
                model_type, model_name
            )
        except ValueError as re:
            # 处理值错误
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def list_model_registrations(
        self, model_type: str, detailed: bool = Query(False)
    ) -> JSONResponse:
        """
        列出模型注册信息的异步方法。

        参数:
        - model_type: 模型类型
        - detailed: 是否返回详细信息，默认为False

        返回:
        - JSONResponse: 包含模型注册列表的JSON响应

        异常:
        - HTTPException: 当列出过程中出现错误时抛出
        """
        try:
            # 通过supervisor获取模型注册列表
            data = await (await self._get_supervisor_ref()).list_model_registrations(
                model_type, detailed=detailed
            )
            return JSONResponse(content=data)
        except ValueError as re:
            # 处理值错误
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_registrations(
        self, model_type: str, model_name: str
    ) -> JSONResponse:
        """
        获取特定模型注册信息的异步方法。

        参数:
        - model_type: 模型类型
        - model_name: 模型名称

        返回:
        - JSONResponse: 包含特定模型注册信息的JSON响应

        异常:
        - HTTPException: 当获取过程中出现错误时抛出
        """
        try:
            # 通过supervisor获取特定模型的注册信息
            data = await (await self._get_supervisor_ref()).get_model_registration(
                model_type, model_name
            )
            return JSONResponse(content=data)
        except ValueError as re:
            # 处理值错误
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            # 处理其他异常
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    # RESTfulAPI 类的方法

    async def list_cached_models(
        self, model_name: str = Query(None), worker_ip: str = Query(None)
    ) -> JSONResponse:
        """
        列出缓存的模型。

        参数:
            model_name (str, 可选): 模型名称，用于过滤结果。
            worker_ip (str, 可选): 工作节点IP，用于过滤结果。

        返回:
            JSONResponse: 包含缓存模型列表的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出，包括400（客户端错误）和500（服务器错误）。
        """
        try:
            data = await (await self._get_supervisor_ref()).list_cached_models(
                model_name, worker_ip
            )
            resp = {
                "list": data,
            }
            return JSONResponse(content=resp)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_events(self, model_uid: str) -> JSONResponse:
        """
        获取指定模型的事件。

        参数:
            model_uid (str): 模型的唯一标识符。

        返回:
            JSONResponse: 包含模型事件的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出，包括400（客户端错误）和500（服务器错误）。
        """
        try:
            event_collector_ref = await self._get_event_collector_ref()
            events = await event_collector_ref.get_model_events(model_uid)
            return JSONResponse(content=events)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def abort_request(self, model_uid: str, request_id: str) -> JSONResponse:
        """
        中止指定模型的特定请求。

        参数:
            model_uid (str): 模型的唯一标识符。
            request_id (str): 请求的唯一标识符。

        返回:
            JSONResponse: 包含中止操作结果的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出500（服务器错误）。
        """
        try:
            supervisor_ref = await self._get_supervisor_ref()
            res = await supervisor_ref.abort_request(model_uid, request_id)
            return JSONResponse(content=res)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    async def list_vllm_supported_model_families(self) -> JSONResponse:
        """
        列出VLLM支持的模型系列。

        返回:
            JSONResponse: 包含支持的聊天和生成模型系列的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出500（服务器错误）。
        """
        try:
            # 从vllm核心模块导入支持的模型列表
            from ..model.llm.vllm.core import (
                VLLM_SUPPORTED_CHAT_MODELS,
                VLLM_SUPPORTED_MODELS,
            )

            # 构建包含聊天和生成模型的数据字典
            data = {
                "chat": VLLM_SUPPORTED_CHAT_MODELS,
                "generate": VLLM_SUPPORTED_MODELS,
            }
            # 返回JSON响应
            return JSONResponse(content=data)
        except Exception as e:
            # 记录错误日志
            logger.error(e, exc_info=True)
            # 抛出HTTP 500错误
            raise HTTPException(status_code=500, detail=str(e))

    async def get_cluster_device_info(
        self, detailed: bool = Query(False)
    ) -> JSONResponse:
        """
        获取集群设备信息。

        参数:
            detailed (bool, 可选): 是否返回详细信息，默认为False。

        返回:
            JSONResponse: 包含集群设备信息的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出500（服务器错误）。
        """
        try:
            data = await (await self._get_supervisor_ref()).get_cluster_device_info(
                detailed=detailed
            )
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_cluster_version(self) -> JSONResponse:
        try:
            # 获取集群版本实际上是通过version获取的realease版本的信息
            # {
            #     "version": "0.14.4+6.g66612b9.dirty",
            #     "full-revisionid": "66612b9c9abd8598999490b1a2d34cfd3084d9ed",
            #     "dirty": true,
            #     "error": null,
            #     "date": "2024-09-04T17:45:32+0800"
            # }
            data = get_versions()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_model_files(
        self, model_version: str = Query(None), worker_ip: str = Query(None)
    ) -> JSONResponse:
        """
        列出可删除的模型文件。

        参数:
            model_version (str, 可选): 模型版本，用于过滤结果。
            worker_ip (str, 可选): 工作节点IP，用于过滤结果。

        返回:
            JSONResponse: 包含可删除模型文件列表的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出，包括400（客户端错误）和500（服务器错误）。
        """
        try:
            data = await (await self._get_supervisor_ref()).list_deletable_models(
                model_version, worker_ip
            )
            response = {
                "model_version": model_version,
                "worker_ip": worker_ip,
                "paths": data,
            }
            return JSONResponse(content=response)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def confirm_and_remove_model(
        self, model_version: str = Query(None), worker_ip: str = Query(None)
    ) -> JSONResponse:
        """
        确认并删除指定的模型。

        参数:
            model_version (str, 可选): 要删除的模型版本。
            worker_ip (str, 可选): 执行删除操作的工作节点IP。

        返回:
            JSONResponse: 包含删除操作结果的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出，包括400（客户端错误）和500（服务器错误）。
        """
        try:
            res = await (await self._get_supervisor_ref()).confirm_and_remove_model(
                model_version=model_version, worker_ip=worker_ip
            )
            return JSONResponse(content={"result": res})
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_workers_info(self) -> JSONResponse:
        """
        获取所有工作节点的信息。

        返回:
            JSONResponse: 包含工作节点信息的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出，包括400（客户端错误）和500（服务器错误）。
        """
        try:
            res = await (await self._get_supervisor_ref()).get_workers_info()
            return JSONResponse(content=res)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_supervisor_info(self) -> JSONResponse:
        """
        获取监督者（supervisor）的信息。

        返回:
            JSONResponse: 包含监督者信息的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出，包括400（客户端错误）和500（服务器错误）。
        """
        try:
            res = await (await self._get_supervisor_ref()).get_supervisor_info()
            return res
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def abort_cluster(self) -> JSONResponse:
        """
        中止整个集群。

        返回:
            JSONResponse: 包含中止操作结果的JSON响应。

        异常:
            HTTPException: 当发生错误时抛出，包括400（客户端错误）和500（服务器错误）。
        """
        import os
        import signal

        try:
            res = await (await self._get_supervisor_ref()).abort_cluster()
            os.kill(os.getpid(), signal.SIGINT)
            return JSONResponse(content={"result": res})
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


def run(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict] = None,
    auth_config_file: Optional[str] = None,
):
    """
    运行RESTful API服务。

    参数:
        supervisor_address (str): 监督者地址。
        host (str): 主机地址。
        port (int): 端口号。
        logging_conf (Optional[dict]): 日志配置，默认为None。
        auth_config_file (Optional[str]): 认证配置文件路径，默认为None。

    注意:
        如果指定的端口不可用，且为默认端口，将尝试使用下一个可用端口。
    """
    logger.info(f"Starting Xinference at endpoint: http://{host}:{port}")
    try:
        api = RESTfulAPI(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            auth_config_file=auth_config_file,
        )
        api.serve(logging_conf=logging_conf)
    except SystemExit:
        logger.warning("Failed to create socket with port %d", port)
        # compare the reference to differentiate between the cases where the user specify the
        # default port and the user does not specify the port.
        if port is XINFERENCE_DEFAULT_ENDPOINT_PORT:
            port = get_next_port()
            logger.info(f"Found available port: {port}")
            logger.info(f"Starting Xinference at endpoint: http://{host}:{port}")
            api = RESTfulAPI(
                supervisor_address=supervisor_address,
                host=host,
                port=port,
                auth_config_file=auth_config_file,
            )
            api.serve(logging_conf=logging_conf)
        else:
            # 果不是默认端口，说明用户明确指定了想要使用的端口。在这种情况下，
            # 如果指定的端口不可用，代码选择抛出异常而不是自动选择其他端口。
            raise


def run_in_subprocess(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict] = None,
    auth_config_file: Optional[str] = None,
) -> multiprocessing.Process:
    """
    在子进程中运行RESTful API服务。

    参数:
        supervisor_address (str): 监督者地址。
        host (str): 主机地址。
        port (int): 端口号。
        logging_conf (Optional[dict]): 日志配置，默认为None。
        auth_config_file (Optional[str]): 认证配置文件路径，默认为None。

    返回:
        multiprocessing.Process: 运行API服务的子进程。

    注意:
        创建的子进程是守护进程，会随主进程的结束而结束。
    """
    p = multiprocessing.Process(
        target=run,
        args=(supervisor_address, host, port, logging_conf, auth_config_file),
    )
    p.daemon = True
    p.start()
    return p
