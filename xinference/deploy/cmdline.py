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
import logging  # 用于日志记录
import os  # 用于操作系统相关功能
import sys  # 用于系统相关功能
import warnings  # 用于发出警告
from typing import List, Optional, Sequence, Tuple, Union  # 用于类型注解

import click  # 用于创建命令行界面
from xoscar.utils import get_next_port  # 用于获取下一个可用端口

from .. import __version__  # 导入版本信息
from ..client import RESTfulClient  # 导入RESTful客户端
from ..client.restful.restful_client import (  # 导入RESTful客户端相关类
    RESTfulChatModelHandle,
    RESTfulGenerateModelHandle,
)
from ..constants import (  # 导入常量
    XINFERENCE_AUTH_DIR,
    XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    XINFERENCE_DEFAULT_ENDPOINT_PORT,
    XINFERENCE_DEFAULT_LOCAL_HOST,
    XINFERENCE_ENV_ENDPOINT,
    XINFERENCE_LOG_BACKUP_COUNT,
    XINFERENCE_LOG_MAX_BYTES,
)
from ..isolation import Isolation  # 导入隔离相关功能
from ..types import ChatCompletionMessage  # 导入聊天完成消息类型
from .utils import (  # 导入工具函数
    get_config_dict,
    get_log_file,
    get_timestamp_ms,
    handle_click_args_type,
)

try:
    # 尝试导入readline模块，提供更好的行编辑和历史功能
    import readline  # noqa: F401
except ImportError:
    pass  # 如果导入失败，则忽略

# 定义获取端点的函数
def get_endpoint(endpoint: Optional[str]) -> str:
    # 如果未指定端点
    if endpoint is None:
        # 检查环境变量中是否有端点设置
        if XINFERENCE_ENV_ENDPOINT in os.environ:
            return os.environ[XINFERENCE_ENV_ENDPOINT]
        else:
            # 使用默认端点
            default_endpoint = f"http://{XINFERENCE_DEFAULT_LOCAL_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}"
            return default_endpoint
    else:
        return endpoint

# 定义获取端点哈希值的函数
def get_hash_endpoint(endpoint: str) -> str:
    import hashlib

    m = hashlib.sha256()
    m.update(bytes(endpoint, "utf-8"))
    return m.hexdigest()

# 定义获取存储令牌的函数
def get_stored_token(
    endpoint: str, client: Optional[RESTfulClient] = None
) -> Optional[str]:
    rest_client = RESTfulClient(endpoint) if client is None else client
    authed = rest_client._cluster_authed
    if not authed:
        return None

    token_path = os.path.join(XINFERENCE_AUTH_DIR, get_hash_endpoint(endpoint))
    if not os.path.exists(token_path):
        raise RuntimeError("Cannot find access token, please login first!")
    with open(token_path, "r") as f:
        access_token = str(f.read())
    return access_token

# 定义启动本地集群的函数
def start_local_cluster(
    log_level: str,
    host: str,
    port: int,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    auth_config_file: Optional[str] = None,
):
    from .local import main

    dict_config = get_config_dict(
        log_level,
        get_log_file(f"local_{get_timestamp_ms()}"),
        XINFERENCE_LOG_BACKUP_COUNT,
        XINFERENCE_LOG_MAX_BYTES,
    )
    logging.config.dictConfig(dict_config)  # type: ignore

    main(
        host=host,
        port=port,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
        logging_conf=dict_config,
        auth_config_file=auth_config_file,
    )

# 使用click创建命令行界面
@click.group(
    invoke_without_command=True,
    name="xinference",
    help="Xinference command-line interface for serving and deploying models.",
)
@click.pass_context
@click.version_option(
    __version__,
    "--version",
    "-v",
    help="Show the current version of the Xinference tool.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_LOCAL_HOST,
    type=str,
    help="Specify the host address for the Xinference server.",
)
@click.option(
    "--port",
    "-p",
    default=XINFERENCE_DEFAULT_ENDPOINT_PORT,
    type=int,
    help="Specify the port number for the Xinference server.",
)
def cli(
    ctx,
    log_level: str,
    host: str,
    port: int,
):
    # 定义命令行界面的主函数
    # ctx: Click的上下文对象
    # log_level: 日志级别
    # host: 主机地址
    # port: 端口号
    
    if ctx.invoked_subcommand is None:
        # 如果没有调用子命令，执行以下代码
        
        # 使用warnings.catch_warnings()上下文管理器来控制警告的行为
        with warnings.catch_warnings():
            # 将所有DeprecationWarning设置为always显示
            warnings.simplefilter("always", DeprecationWarning)
            
            # 发出一个废弃警告
            warnings.warn(
                "Starting a local 'xinference' cluster via the 'xinference' command line is "
                "deprecated and will be removed in a future release. Please use the new "
                "'xinference-local' command.",
                category=DeprecationWarning,
            )

        # 调用start_local_cluster函数启动本地集群
        start_local_cluster(log_level=log_level, host=host, port=port)

# 定义启动本地集群的命令
@click.command(help="Starts an Xinference local cluster.")
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_LOCAL_HOST,
    type=str,
    help="Specify the host address for the Xinference server.",
)
@click.option(
    "--port",
    "-p",
    default=XINFERENCE_DEFAULT_ENDPOINT_PORT,
    type=int,
    help="Specify the port number for the Xinference server.",
)
@click.option(
    "--metrics-exporter-host",
    "-MH",
    default=None,
    type=str,
    help="Specify the host address for the Xinference metrics exporter server, default is the same as --host.",
)
@click.option(
    "--metrics-exporter-port",
    "-mp",
    type=int,
    help="Specify the port number for the Xinference metrics exporter server.",
)
@click.option(
    "--auth-config",
    type=str,
    help="Specify the auth config json file.",
)
def local(
    log_level: str,
    host: str,
    port: int,
    metrics_exporter_host: Optional[str],
    metrics_exporter_port: Optional[int],
    auth_config: Optional[str],
):
    if metrics_exporter_host is None:
        metrics_exporter_host = host
    start_local_cluster(
        log_level=log_level,
        host=host,
        port=port,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
        auth_config_file=auth_config,
    )

# 定义启动监督器的命令
@click.command(
    help="Starts an Xinference supervisor to control and monitor the worker actors."
)
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level for the supervisor. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    type=str,
    help="Specify the host address for the supervisor.",
)
@click.option(
    "--port",
    "-p",
    default=XINFERENCE_DEFAULT_ENDPOINT_PORT,
    type=int,
    help="Specify the port number for the Xinference web ui and service.",
)
@click.option(
    "--supervisor-port",
    type=int,
    help="Specify the port number for the Xinference supervisor.",
)
@click.option(
    "--auth-config",
    type=str,
    help="Specify the auth config json file.",
)
def supervisor(
    log_level: str,
    host: str,
    port: int,
    supervisor_port: Optional[int],
    auth_config: Optional[str],
):
    # 从 ..deploy.supervisor 模块导入 main 函数
    from ..deploy.supervisor import main

    # 获取配置字典，包括日志级别、日志文件路径、备份数量和最大字节数
    dict_config = get_config_dict(
        log_level,
        get_log_file(f"supervisor_{get_timestamp_ms()}"),
        XINFERENCE_LOG_BACKUP_COUNT,
        XINFERENCE_LOG_MAX_BYTES,
    )
    # 使用配置字典设置日志配置
    logging.config.dictConfig(dict_config)  # type: ignore

    # 调用 main 函数启动监督器，传入主机、端口、监督器端口、日志配置和认证配置文件
    main(
        host=host,
        port=port,
        supervisor_port=supervisor_port,
        logging_conf=dict_config,
        auth_config_file=auth_config,
    )

# 定义启动工作节点的命令
@click.command(
    help="Starts an Xinference worker to execute tasks assigned by the supervisor in a distributed setup."
)
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level for the worker. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    type=str,
    help="Specify the host address for the worker.",
)
@click.option(
    "--worker-port",
    type=int,
    help="Specify the port number for the Xinference worker.",
)
@click.option(
    "--metrics-exporter-host",
    "-MH",
    default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    type=str,
    help="Specify the host address for the metrics exporter server.",
)
@click.option(
    "--metrics-exporter-port",
    type=int,
    help="Specify the port number for the Xinference metrics exporter worker.",
)
def worker(
    log_level: str,
    endpoint: Optional[str],
    host: str,
    worker_port: Optional[int],
    metrics_exporter_host: Optional[str],
    metrics_exporter_port: Optional[int],
):
    from ..deploy.worker import main

    dict_config = get_config_dict(
        log_level,
        get_log_file(f"worker_{get_timestamp_ms()}"),
        XINFERENCE_LOG_BACKUP_COUNT,
        XINFERENCE_LOG_MAX_BYTES,
    )
    logging.config.dictConfig(dict_config)  # type: ignore

    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    supervisor_internal_addr = client._get_supervisor_internal_address()

    address = f"{host}:{worker_port or get_next_port()}"
    main(
        address=address,
        supervisor_address=supervisor_internal_addr,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
        logging_conf=dict_config,
    )

# 定义注册模型的命令
@cli.command("register", help="Register a new model with Xinference for deployment.")
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--model-type",
    "-t",
    default="LLM",
    type=str,
    help="Type of model to register (default is 'LLM').",
)
@click.option("--file", "-f", type=str, help="Path to the model configuration file.")
@click.option(
    "--worker-ip", "-w", type=str, help="Specify the ip address of the worker."
)
@click.option(
    "--persist",
    "-p",
    is_flag=True,
    help="Persist the model configuration to the filesystem, retains the model registration after server restarts.",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def register_model(
    endpoint: Optional[str],
    model_type: str,
    file: str,
    worker_ip: str,
    persist: bool,
    api_key: Optional[str],
):
    endpoint = get_endpoint(endpoint)
    with open(file) as fd:
        model = fd.read()

    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))
    client.register_model(
        model_type=model_type,
        model=model,
        worker_ip=worker_ip,
        persist=persist,
    )

# 定义注销模型的命令
@cli.command(
    "unregister",
    help="Unregister a model from Xinference, removing it from deployment.",
)
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--model-type",
    "-t",
    default="LLM",
    type=str,
    help="Type of model to unregister (default is 'LLM').",
)
@click.option("--model-name", "-n", type=str, help="Name of the model to unregister.")
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def unregister_model(
    endpoint: Optional[str],
    model_type: str,
    model_name: str,
    api_key: Optional[str],
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))
    client.unregister_model(
        model_type=model_type,
        model_name=model_name,
    )

# 定义列出已注册模型的命令
@cli.command("registrations", help="List all registered models in Xinference.")
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model-type",
    "-t",
    default="LLM",
    type=str,
    help="Filter by model type (default is 'LLM').",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def list_model_registrations(
    endpoint: Optional[str],
    model_type: str,
    api_key: Optional[str],
):
    from tabulate import tabulate

    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    registrations = client.list_model_registrations(model_type=model_type)

    table = []
    if model_type == "LLM":
        for registration in registrations:
            model_name = registration["model_name"]
            model_family = client.get_model_registration(model_type, model_name)
            table.append(
                [
                    model_type,
                    model_family["model_name"],
                    model_family["model_lang"],
                    model_family["model_ability"],
                    registration["is_builtin"],
                ]
            )
        print(
            tabulate(
                table, headers=["Type", "Name", "Language", "Ability", "Is-built-in"]
            ),
            file=sys.stderr,
        )
    elif model_type == "embedding":
        for registration in registrations:
            model_name = registration["model_name"]
            model_family = client.get_model_registration(model_type, model_name)
            table.append(
                [
                    model_type,
                    model_family["model_name"],
                    model_family["language"],
                    model_family["dimensions"],
                    registration["is_builtin"],
                ]
            )
        print(
            tabulate(
                table, headers=["Type", "Name", "Language", "Dimensions", "Is-built-in"]
            ),
            file=sys.stderr,
        )
    elif model_type == "rerank":
        for registration in registrations:
            model_name = registration["model_name"]
            model_family = client.get_model_registration(model_type, model_name)
            table.append(
                [
                    model_type,
                    model_family["model_name"],
                    model_family["language"],
                    registration["is_builtin"],
                ]
            )
        print(
            tabulate(table, headers=["Type", "Name", "Language", "Is-built-in"]),
            file=sys.stderr,
        )
    elif model_type == "image":
        for registration in registrations:
            model_name = registration["model_name"]
            model_family = client.get_model_registration(model_type, model_name)
            table.append(
                [
                    model_type,
                    model_family["model_name"],
                    model_family["model_family"],
                    registration["is_builtin"],
                ]
            )
        print(
            tabulate(table, headers=["Type", "Name", "Family", "Is-built-in"]),
            file=sys.stderr,
        )
    elif model_type == "audio":
        for registration in registrations:
            model_name = registration["model_name"]
            model_family = client.get_model_registration(model_type, model_name)
            table.append(
                [
                    model_type,
                    model_family["model_name"],
                    model_family["model_family"],
                    model_family["multilingual"],
                    registration["is_builtin"],
                ]
            )
        print(
            tabulate(
                table, headers=["Type", "Name", "Family", "Multilingual", "Is-built-in"]
            ),
            file=sys.stderr,
        )
    else:
        raise NotImplementedError(f"List {model_type} is not implemented.")


@cli.command("cached", help="List all cached models in Xinference.")
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model_name",
    "-n",
    type=str,
    help="Provide the name of the models to be removed.",
)
@click.option(
    "--worker-ip",
    default=None,
    type=str,
    help="Specify which worker this model runs on by ip, for distributed situation.",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def list_cached_models(
    endpoint: Optional[str],
    api_key: Optional[str],
    model_name: Optional[str],
    worker_ip: Optional[str],
):
    from tabulate import tabulate

    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    cached_models = client.list_cached_models(model_name, worker_ip)
    if not cached_models:
        print("There are no cache files.")
        return
    headers = list(cached_models[0].keys())

    print("cached_model: ")
    table_data = []
    for model in cached_models:
        row_data = [
            str(value) if value is not None else "-" for value in model.values()
        ]
        table_data.append(row_data)
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))


@cli.command("remove-cache", help="Remove selected cached models in Xinference.")
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model_version",
    "-n",
    type=str,
    help="Provide the version of the models to be removed.",
)
@click.option(
    "--worker-ip",
    default=None,
    type=str,
    help="Specify which worker this model runs on by ip, for distributed situation.",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
@click.option("--check", is_flag=True, help="Confirm the deletion of the cache.")
def remove_cache(
    endpoint: Optional[str],
    model_version: str,
    api_key: Optional[str],
    check: bool,
    worker_ip: Optional[str] = None,
):
    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    if not check:
        response = client.list_deletable_models(
            model_version=model_version, worker_ip=worker_ip
        )
        paths: List[str] = response.get("paths", [])
        if not paths:
            click.echo(f"There is no model version named {model_version}.")
            return
        click.echo(f"Model {model_version} cache directory to be deleted:")
        for path in response.get("paths", []):
            click.echo(f"{path}")

        if click.confirm("Do you want to proceed with the deletion?", abort=True):
            check = True
    try:
        result = client.confirm_and_remove_model(
            model_version=model_version, worker_ip=worker_ip
        )
        if result:
            click.echo(f"Cache directory {model_version} has been deleted.")
        else:
            click.echo(
                f"Cache directory {model_version} fail to be deleted. Please check the log."
            )
    except Exception as e:
        click.echo(f"An error occurred while deleting the cache: {e}")


@cli.command(
    "launch",
    help="Launch a model with the Xinference framework with the given parameters.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model-name",
    "-n",
    type=str,
    required=True,
    help="Provide the name of the model to be launched.",
)
@click.option(
    "--model-type",
    "-t",
    type=str,
    default="LLM",
    help="Specify type of model, LLM as default.",
)
@click.option(
    "--model-engine",
    "-en",
    type=str,
    default=None,
    help="Specify the inference engine of the model when launching LLM.",
)
@click.option(
    "--model-uid",
    "-u",
    type=str,
    default=None,
    help="Specify UID of model, default is None.",
)
@click.option(
    "--size-in-billions",
    "-s",
    default=None,
    type=str,
    help="Specify the model size in billions of parameters.",
)
@click.option(
    "--model-format",
    "-f",
    default=None,
    type=str,
    help="Specify the format of the model, e.g. pytorch, ggufv2, etc.",
)
@click.option(
    "--quantization",
    "-q",
    default=None,
    type=str,
    help="Define the quantization settings for the model.",
)
@click.option(
    "--replica",
    "-r",
    default=1,
    type=int,
    help="The replica count of the model, default is 1.",
)
@click.option(
    "--n-gpu",
    default="auto",
    type=str,
    help='The number of GPUs used by the model, default is "auto".',
)
@click.option(
    "--lora-modules",
    "-lm",
    multiple=True,
    type=(str, str),
    help="LoRA module configurations in the format name=path. Multiple modules can be specified.",
)
@click.option(
    "--image-lora-load-kwargs",
    "-ld",
    "image_lora_load_kwargs",
    type=(str, str),
    multiple=True,
)
@click.option(
    "--image-lora-fuse-kwargs",
    "-fd",
    "image_lora_fuse_kwargs",
    type=(str, str),
    multiple=True,
)
@click.option(
    "--worker-ip",
    default=None,
    type=str,
    help="Specify which worker this model runs on by ip, for distributed situation.",
)
@click.option(
    "--gpu-idx",
    default=None,
    type=str,
    help="Specify which GPUs of a worker this model can run on, separated with commas.",
)
@click.option(
    "--trust-remote-code",
    default=True,
    type=bool,
    help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
@click.pass_context
def model_launch(
    ctx,
    endpoint: Optional[str],
    model_name: str,
    model_type: str,
    model_engine: Optional[str],
    model_uid: str,
    size_in_billions: str,
    model_format: str,
    quantization: str,
    replica: int,
    n_gpu: str,
    lora_modules: Optional[Tuple],
    image_lora_load_kwargs: Optional[Tuple],
    image_lora_fuse_kwargs: Optional[Tuple],
    worker_ip: Optional[str],
    gpu_idx: Optional[str],
    trust_remote_code: bool,
    api_key: Optional[str],
):
    kwargs = {}
    for i in range(0, len(ctx.args), 2):
        if not ctx.args[i].startswith("--"):
            raise ValueError("You must specify extra kwargs with `--` prefix.")
        kwargs[ctx.args[i][2:]] = handle_click_args_type(ctx.args[i + 1])
    print(f"Launch model name: {model_name} with kwargs: {kwargs}", file=sys.stderr)

    if model_type == "LLM" and model_engine is None:
        raise ValueError("--model-engine is required for LLM models.")

    if n_gpu.lower() == "none":
        _n_gpu: Optional[Union[int, str]] = None
    elif n_gpu == "auto":
        _n_gpu = n_gpu
    else:
        _n_gpu = int(n_gpu)

    image_lora_load_params = (
        {k: handle_click_args_type(v) for k, v in dict(image_lora_load_kwargs).items()}
        if image_lora_load_kwargs
        else None
    )
    image_lora_fuse_params = (
        {k: handle_click_args_type(v) for k, v in dict(image_lora_fuse_kwargs).items()}
        if image_lora_fuse_kwargs
        else None
    )

    lora_list = (
        [{"lora_name": k, "local_path": v} for k, v in dict(lora_modules).items()]
        if lora_modules
        else []
    )

    peft_model_config = (
        {
            "image_lora_load_kwargs": image_lora_load_params,
            "image_lora_fuse_kwargs": image_lora_fuse_params,
            "lora_list": lora_list,
        }
        if lora_list or image_lora_load_params or image_lora_fuse_params
        else None
    )

    _gpu_idx: Optional[List[int]] = (
        None if gpu_idx is None else [int(idx) for idx in gpu_idx.split(",")]
    )

    endpoint = get_endpoint(endpoint)
    model_size: Optional[Union[str, int]] = (
        size_in_billions
        if size_in_billions is None
        or "_" in size_in_billions
        or "." in size_in_billions
        else int(size_in_billions)
    )
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    model_uid = client.launch_model(
        model_name=model_name,
        model_type=model_type,
        model_engine=model_engine,
        model_uid=model_uid,
        model_size_in_billions=model_size,
        model_format=model_format,
        quantization=quantization,
        replica=replica,
        n_gpu=_n_gpu,
        peft_model_config=peft_model_config,
        worker_ip=worker_ip,
        gpu_idx=_gpu_idx,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    print(f"Model uid: {model_uid}", file=sys.stderr)


@cli.command(
    "list",
    help="List all running models in Xinference.",
)
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def model_list(endpoint: Optional[str], api_key: Optional[str]):
    from tabulate import tabulate

    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    llm_table = []
    embedding_table = []
    rerank_table = []
    image_table = []
    audio_table = []
    models = client.list_models()
    for model_uid, model_spec in models.items():
        if model_spec["model_type"] == "LLM":
            llm_table.append(
                [
                    model_uid,
                    model_spec["model_type"],
                    model_spec["model_name"],
                    model_spec["model_format"],
                    model_spec["model_size_in_billions"],
                    model_spec["quantization"],
                ]
            )
        elif model_spec["model_type"] == "embedding":
            embedding_table.append(
                [
                    model_uid,
                    model_spec["model_type"],
                    model_spec["model_name"],
                    model_spec["dimensions"],
                ]
            )
        elif model_spec["model_type"] == "rerank":
            rerank_table.append(
                [model_uid, model_spec["model_type"], model_spec["model_name"]]
            )
        elif model_spec["model_type"] == "image":
            image_table.append(
                [
                    model_uid,
                    model_spec["model_type"],
                    model_spec["model_name"],
                    str(model_spec["controlnet"]),
                ]
            )
        elif model_spec["model_type"] == "audio":
            audio_table.append(
                [model_uid, model_spec["model_type"], model_spec["model_name"]]
            )
    if llm_table:
        print(
            tabulate(
                llm_table,
                headers=[
                    "UID",
                    "Type",
                    "Name",
                    "Format",
                    "Size (in billions)",
                    "Quantization",
                ],
            ),
            file=sys.stderr,
        )
        print()  # add a blank line for better visual experience
    if embedding_table:
        print(
            tabulate(
                embedding_table,
                headers=[
                    "UID",
                    "Type",
                    "Name",
                    "Dimensions",
                ],
            ),
            file=sys.stderr,
        )
        print()
    if rerank_table:
        print(
            tabulate(
                rerank_table,
                headers=["UID", "Type", "Name"],
            ),
            file=sys.stderr,
        )
        print()
    if image_table:
        print(
            tabulate(
                image_table,
                headers=["UID", "Type", "Name", "Controlnet"],
            ),
            file=sys.stderr,
        )
        print()
    if audio_table:
        print(
            tabulate(
                audio_table,
                headers=["UID", "Type", "Name"],
            ),
            file=sys.stderr,
        )
        print()


@cli.command(
    "terminate",
    help="Terminate a deployed model through unique identifier (UID) of the model.",
)
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model-uid",
    type=str,
    required=True,
    help="The unique identifier (UID) of the model.",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def model_terminate(
    endpoint: Optional[str],
    model_uid: str,
    api_key: Optional[str],
):
    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))
    client.terminate_model(model_uid=model_uid)


@cli.command("generate", help="Generate text using a running LLM.")
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--model-uid",
    type=str,
    help="The unique identifier (UID) of the model.",
)
@click.option(
    "--max_tokens",
    default=512,
    type=int,
    help="Maximum number of tokens in the generated text (default is 512).",
)
@click.option(
    "--stream",
    default=True,
    type=bool,
    help="Whether to stream the generated text. Use 'True' for streaming (default is True).",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def model_generate(
    endpoint: Optional[str],
    model_uid: str,
    max_tokens: int,
    stream: bool,
    api_key: Optional[str],
):
    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))
    if stream:
        # TODO: when stream=True, RestfulClient cannot generate words one by one.
        # So use Client in temporary. The implementation needs to be changed to
        # RestfulClient in the future.
        async def generate_internal():
            while True:
                # the prompt will be written to stdout.
                # https://docs.python.org/3.10/library/functions.html#input
                prompt = input("Prompt: ")
                if prompt == "":
                    break
                print(f"Completion: {prompt}", end="", file=sys.stdout)
                for chunk in model.generate(
                    prompt=prompt,
                    generate_config={"stream": stream, "max_tokens": max_tokens},
                ):
                    choice = chunk["choices"][0]
                    if "text" not in choice:
                        continue
                    else:
                        print(choice["text"], end="", flush=True, file=sys.stdout)
                print("", file=sys.stdout)

        model = client.get_model(model_uid=model_uid)

        loop = asyncio.get_event_loop()
        coro = generate_internal()

        if loop.is_running():
            isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            isolation.start()
            isolation.call(coro)
        else:
            task = loop.create_task(coro)
            try:
                loop.run_until_complete(task)
            except KeyboardInterrupt:
                task.cancel()
                loop.run_until_complete(task)
                # avoid displaying exception-unhandled warnings
                task.exception()
    else:
        restful_model = client.get_model(model_uid=model_uid)
        if not isinstance(
            restful_model, (RESTfulChatModelHandle, RESTfulGenerateModelHandle)
        ):
            raise ValueError(f"model {model_uid} has no generate method")

        while True:
            prompt = input("User: ")
            if prompt == "":
                break
            print(f"Assistant: {prompt}", end="", file=sys.stdout)
            response = restful_model.generate(
                prompt=prompt,
                generate_config={"stream": stream, "max_tokens": max_tokens},
            )
            if not isinstance(response, dict):
                raise ValueError("generate result is not valid")
            print(f"{response['choices'][0]['text']}\n", file=sys.stdout)


@cli.command("chat", help="与运行的LLM进行聊天。")
@click.option("--endpoint", "-e", type=str, help="Xinference的端点。")
@click.option("--model-uid", type=str, help="模型的唯一标识符（UID）。")
@click.option(
    "--max_tokens",
    default=512,
    type=int,
    help="每条消息的最大token数（默认是512）。",
)
@click.option(
    "--stream",
    default=True,
    type=bool,
    help="是否流式传输聊天消息。使用'True'进行流式传输（默认是True）。",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="用于访问xinference api的Api-Key，带有授权。",
)
def model_chat(
    endpoint: Optional[str],
    model_uid: str,
    max_tokens: int,
    stream: bool,
    api_key: Optional[str],
):
    """
    与运行的LLM进行聊天。

    参数:
    - endpoint: Xinference的端点。
    - model_uid: 模型的唯一标识符（UID）。
    - max_tokens: 每条消息的最大token数（默认是512）。
    - stream: 是否流式传输聊天消息。使用'True'进行流式传输（默认是True）。
    - api-key: 用于访问xinference api的Api-Key，带有授权。

    功能:
    1. 获取模型的端点和客户端。
    2. 如果api-key为None，则设置存储的token。
    3. 根据stream参数的不同，选择不同的聊天方式：
       - 如果stream为True，使用异步方式进行聊天，逐字生成回复。
       - 如果stream为False，使用同步方式进行聊天，一次性生成完整回复。
    4. 在聊天过程中，记录聊天历史，并根据用户的输入生成回复。
    """
    # TODO: 聊天模型的角色可能不是用户和助手。
    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    chat_history: "List[ChatCompletionMessage]" = []
    if stream:
        # TODO: 当stream=True时，RestfulClient无法逐字生成单词。
        # 因此暂时使用Client。未来的实现需要改为RestfulClient。
        async def chat_internal():
            while True:
                # 提示将写入stdout。
                # https://docs.python.org/3.10/library/functions.html#input
                prompt = input("User: ")
                if prompt == "":
                    break
                print("Assistant: ", end="", file=sys.stdout)
                response_content = ""
                for chunk in model.chat(
                    prompt=prompt,
                    chat_history=chat_history,
                    generate_config={"stream": stream, "max_tokens": max_tokens},
                ):
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue
                    else:
                        response_content += delta["content"]
                        print(delta["content"], end="", flush=True, file=sys.stdout)
                print("", file=sys.stdout)
                chat_history.append(ChatCompletionMessage(role="user", content=prompt))
                chat_history.append(
                    ChatCompletionMessage(role="assistant", content=response_content)
                )

        model = client.get_model(model_uid=model_uid)

        loop = asyncio.get_event_loop()
        coro = chat_internal()

        if loop.is_running():
            isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            isolation.start()
            isolation.call(coro)
        else:
            task = loop.create_task(coro)
            try:
                loop.run_until_complete(task)
            except KeyboardInterrupt:
                task.cancel()
                loop.run_until_complete(task)
                # 避免显示未处理的异常警告
                task.exception()
    else:
        restful_model = client.get_model(model_uid=model_uid)
        if not isinstance(restful_model, RESTfulChatModelHandle):
            raise ValueError(f"模型 {model_uid} 没有聊天方法")

        while True:
            prompt = input("User: ")
            if prompt == "":
                break
            chat_history.append(ChatCompletionMessage(role="user", content=prompt))
            print("Assistant: ", end="", file=sys.stdout)
            response = restful_model.chat(
                prompt=prompt,
                chat_history=chat_history,
                generate_config={"stream": stream, "max_tokens": max_tokens},
            )
            if not isinstance(response, dict):
                raise ValueError("聊天结果无效")
            response_content = response["choices"][0]["message"]["content"]
            print(f"{response_content}\n", file=sys.stdout)
            chat_history.append(
                ChatCompletionMessage(role="assistant", content=response_content)
            )


@cli.command("vllm-models", help="Query and display models compatible with vLLM.")
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def vllm_models(endpoint: Optional[str], api_key: Optional[str]):
    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))
    vllm_models_dict = client.vllm_models()
    print("VLLM supported model families:")
    chat_models = vllm_models_dict["chat"]
    supported_models = vllm_models_dict["generate"]

    print("VLLM supported chat model families:", chat_models)
    print("VLLM supported generate model families:", supported_models)


@cli.command("login", help="Login when the cluster is authenticated.")
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option("--username", type=str, required=True, help="Username.")
@click.option(
    "--password",
    type=str,
    required=True,
    help="Password.",
)
def cluster_login(
    endpoint: Optional[str],
    username: str,
    password: str,
):
    endpoint = get_endpoint(endpoint)
    restful_client = RESTfulClient(base_url=endpoint)
    if restful_client._cluster_authed:
        restful_client.login(username, password)
        access_token = restful_client._get_token()
        assert access_token is not None
        os.makedirs(XINFERENCE_AUTH_DIR, exist_ok=True)
        hashed_ep = get_hash_endpoint(endpoint)
        with open(os.path.join(XINFERENCE_AUTH_DIR, hashed_ep), "w") as f:
            f.write(access_token)


@cli.command(name="engine", help="Query the applicable inference engine by model name.")
@click.option(
    "--model-name",
    "-n",
    type=str,
    required=True,
    help="The model name you want to query.",
)
@click.option(
    "--model-engine",
    "-en",
    type=str,
    default=None,
    help="Specify the `model_engine` to query the corresponding combination of other parameters.",
)
@click.option(
    "--model-format",
    "-f",
    type=str,
    default=None,
    help="Specify the `model_format` to query the corresponding combination of other parameters.",
)
@click.option(
    "--model-size-in-billions",
    "-s",
    type=str,
    default=None,
    help="Specify the `model_size_in_billions` to query the corresponding combination of other parameters.",
)
@click.option(
    "--quantization",
    "-q",
    type=str,
    default=None,
    help="Specify the `quantization` to query the corresponding combination of other parameters.",
)
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="Api-Key for access xinference api with authorization.",
)
def query_engine_by_model_name(
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str],
    model_size_in_billions: Optional[Union[str, int]],
    quantization: Optional[str],
    endpoint: Optional[str],
    api_key: Optional[str],
):
    """
    查询指定模型名称的适用推理引擎。

    该函数的主要用途是根据用户提供的模型名称和其他可选参数（如模型引擎、模型格式、模型大小和量化设置），查询并显示支持的推理引擎及其相关参数。

    :param model_name: 要查询的模型名称。
    :param model_engine: 可选参数，指定要查询的模型引擎。
    :param model_format: 可选参数，指定要查询的模型格式。
    :param model_size_in_billions: 可选参数，指定要查询的模型大小（以十亿参数为单位）。
    :param quantization: 可选参数，指定要查询的量化设置。
    :param endpoint: 可选参数，指定Xinference的端点。
    :param api_key: 可选参数，指定用于访问Xinference API的API密钥。
    """
    from tabulate import tabulate

    def match_engine_from_spell(value: str, target: Sequence[str]) -> Tuple[bool, str]:
        """
        匹配用户输入的引擎名称与支持的引擎名称，忽略大小写。

        :param value: 用户输入的引擎名称。
        :param target: 支持的引擎名称列表。
        :return: 一个元组，包含匹配结果（布尔值）和匹配到的引擎名称。
        """
        for t in target:
            if value.lower() == t.lower():
                return True, t
        return False, value

    def handle_user_passed_parameters() -> List[str]:
        """
        处理用户传递的参数，生成用户指定的参数列表。

        :return: 用户指定的参数列表。
        """
        user_specified_parameters = []
        if model_engine is not None:
            user_specified_parameters.append(f"--model-engine {model_engine}")
        if model_format is not None:
            user_specified_parameters.append(f"--model-format {model_format}")
        if model_size_in_billions is not None:
            user_specified_parameters.append(
                f"--model-size-in-billions {model_size_in_billions}"
            )
        if quantization is not None:
            user_specified_parameters.append(f"--quantization {quantization}")
        return user_specified_parameters

    user_specified_params = handle_user_passed_parameters()

    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    llm_engines = client.query_engine_by_model_name(model_name)
    if model_engine is not None:
        is_matched, model_engine = match_engine_from_spell(
            model_engine, list(llm_engines.keys())
        )
        if not is_matched:
            print(
                f'Xinference does not support this inference engine "{model_engine}".',
                file=sys.stderr,
            )
            return

    table = []
    engines = [model_engine] if model_engine is not None else list(llm_engines.keys())
    for engine in engines:
        params = llm_engines[engine]
        for param in params:
            if (
                (model_format is None or model_format == param["model_format"])
                and (
                    model_size_in_billions is None
                    or model_size_in_billions == str(param["model_size_in_billions"])
                )
                and (quantization is None or quantization in param["quantizations"])
            ):
                if quantization is not None:
                    table.append(
                        [
                            model_name,
                            engine,
                            param["model_format"],
                            param["model_size_in_billions"],
                            quantization,
                        ]
                    )
                else:
                    for quant in param["quantizations"]:
                        table.append(
                            [
                                model_name,
                                engine,
                                param["model_format"],
                                param["model_size_in_billions"],
                                quant,
                            ]
                        )
    if len(table) == 0:
        print(
            f"Xinference does not support "
            f"your provided params: {', '.join(user_specified_params)} for the model {model_name}.",
            file=sys.stderr,
        )
    else:
        print(
            tabulate(
                table,
                headers=[
                    "Name",
                    "Engine",
                    "Format",
                    "Size (in billions)",
                    "Quantization",
                ],
            ),
            file=sys.stderr,
        )


@cli.command(
    "cal-model-mem",
    help="calculate gpu mem usage with specified model size and context_length",
)
@click.option(
    "--model-name",
    "-n",
    type=str,
    help="The model name is optional.\
    If provided, fetch model config from huggingface/modelscope;\
    If not specified, use default model layer to estimate.",
)
@click.option(
    "--size-in-billions",
    "-s",
    type=str,
    required=True,
    help="Specify the model size in billions of parameters. Format accept 1_8 and 1.8",
)
@click.option(
    "--model-format",
    "-f",
    type=str,
    required=True,
    help="Specify the format of the model, e.g. pytorch, ggufv2, etc.",
)
@click.option(
    "--quantization",
    "-q",
    type=str,
    default=None,
    help="Define the quantization settings for the model.",
)
@click.option(
    "--context-length",
    "-c",
    type=int,
    required=True,
    help="Specify the context length",
)
@click.option(
    "--kv-cache-dtype",
    type=int,
    default=16,
    help="Specified the kv_cache_dtype, one of: 8, 16, 32",
)
def cal_model_mem(
    model_name: Optional[str],
    size_in_billions: str,
    model_format: str,
    quantization: Optional[str],
    context_length: int,
    kv_cache_dtype: int,
):
    if kv_cache_dtype not in [8, 16, 32]:
        print("Invalid kv_cache_dtype:", kv_cache_dtype)
        os._exit(1)

    import math

    from ..model.llm.llm_family import convert_model_size_to_float
    from ..model.llm.memory import estimate_llm_gpu_memory

    mem_info = estimate_llm_gpu_memory(
        model_size_in_billions=size_in_billions,
        quantization=quantization,
        context_length=context_length,
        model_format=model_format,
        model_name=model_name,
        kv_cache_dtype=kv_cache_dtype,
    )
    if mem_info is None:
        print("The Specified model parameters is not match: `%s`" % model_name)
        os._exit(1)
    total_mem_g = math.ceil(mem_info.total / 1024.0)
    print("model_name:", model_name)
    print("kv_cache_dtype:", kv_cache_dtype)
    print("model size: %.1f B" % (convert_model_size_to_float(size_in_billions)))
    print("quant: %s" % (quantization))
    print("context: %d" % (context_length))
    print("gpu mem usage:")
    print("  model mem: %d MB" % (mem_info.model_mem))
    print("  kv_cache: %d MB" % (mem_info.kv_cache_mem))
    print("  overhead: %d MB" % (mem_info.overhead))
    print("  active: %d MB" % (mem_info.activation_mem))
    print("  total: %d MB (%d GB)" % (mem_info.total, total_mem_g))


@cli.command(
    "stop-cluster",
    help="Stop a cluster using the Xinference framework with the given parameters.",
)
@click.option(
    "--endpoint",
    "-e",
    type=str,
    required=True,
    help="Xinference endpoint.",
)
@click.option(
    "--api-key",
    "-ak",
    default=None,
    type=str,
    help="API key for accessing the Xinference API with authorization.",
)
@click.option("--check", is_flag=True, help="Confirm the deletion of the cache.")
def stop_cluster(endpoint: str, api_key: Optional[str], check: bool):
    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint, api_key=api_key)
    if api_key is None:
        client._set_token(get_stored_token(endpoint, client))

    if not check:
        click.echo(
            f"This command will stop Xinference cluster in {endpoint}.", err=True
        )
        supervisor_info = client.get_supervisor_info()
        click.echo("Supervisor information: ")
        click.echo(supervisor_info)

        workers_info = client.get_workers_info()
        click.echo("Workers information:")
        click.echo(workers_info)

        click.confirm("Continue?", abort=True)
    try:
        result = client.abort_cluster()
        result = result.get("result")
        click.echo(f"Cluster stopped: {result}")
    except Exception as e:
        click.echo(e)


if __name__ == "__main__":
    cli()
