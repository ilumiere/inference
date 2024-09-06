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

import uvicorn
from aioprometheus import Counter, Gauge
from aioprometheus.asgi.starlette import metrics
from fastapi import FastAPI
from fastapi.responses import RedirectResponse



# 实现性能指标的收集、记录和导出功能。它的主要作用包括：
# 1. 定义指标：
# 使用 Gauge 和 Counter 定义了几个关键的性能指标，如生成吞吐量、首个令牌的延迟、输入和输出令牌总数等。
# 2. 记录指标：
# 提供了 record_metrics 函数，用于动态更新各种指标的值。
# 3. 导出指标：
# 实现了 launch_metrics_export_server 函数，用于启动一个专门的服务器来导出这些指标。


# 设置默认的指标服务器日志级别
DEFAULT_METRICS_SERVER_LOG_LEVEL = "warning"

# Gauge 用于表示可以任意上下波动的单一数值。
# 它可以增加也可以减少。
# 在这里用于测量：
# a) 生成吞吐量（tokens/s）
# b) 首个令牌的延迟（ms）
# 这些值可能随时间变化，既可能增加也可能减少。
# 2. Counter（计数器）:
# Counter 用于表示单调递增的计数器。
# 它只能增加或被重置为零，不能减少。

# 定义生成吞吐量的指标
generate_throughput = Gauge(
    "xinference:generate_tokens_per_s", "Generate throughput in tokens/s."
)
# Latency
time_to_first_token = Gauge(
    "xinference:time_to_first_token_ms", "First token latency in ms."
)
# 定义输入令牌总数的计数器
input_tokens_total_counter = Counter(
    "xinference:input_tokens_total_counter", "Total number of input tokens."
)
# 定义输出令牌总数的计数器
output_tokens_total_counter = Counter(
    "xinference:output_tokens_total_counter", "Total number of output tokens."
)

# 记录指标的函数
def record_metrics(name, op, kwargs):
    """
    globals() 函数:
    globals() 是 Python 的内置函数，返回当前模块的全局变量字典。
    在这个上下文中，它用来访问在模块级别定义的变量。
    为什么能获得 collector:
    在代码的前面部分，定义了几个全局变量作为指标收集器：
        generate_throughput = Gauge(...)
        time_to_first_token = Gauge(...)
        input_tokens_total_counter = Counter(...)
        output_tokens_total_counter = Counter(...)
    - 这些变量存在于全局命名空间中，因此可以通过 globals() 访问。
    
    
    在这个函数中，getattr(collector, op) 用于动态获取 collector 对象的方法。
    例如，如果 op 是 "inc"，那么 getattr(collector, "inc") 会返回 collector 的 inc 方法。
    然后，通过传入的 kwargs 参数调用该方法。
    
    
    假设调用 record_metrics("generate_throughput", "set", {"value": 100})
    globals().get("generate_throughput") 返回之前定义的 Gauge 对象。
    getattr(collector, "set") 获取这个 Gauge 对象的 set 方法。
    最后调用 set(value=100) 来设置指标的值。
    
    
    record_metrics("input_tokens_total_counter", "inc", {"value": 50})
    这会将 input_tokens_total_counter 增加 50。
    
    Args:
        name (_type_): _description_
        op (_type_): _description_
        kwargs (_type_): _description_
    """
    # 从全局变量中获取指标收集器
    collector = globals().get(name)
    # 动态调用收集器的方法并传入参数
    getattr(collector, op)(**kwargs)

# 启动指标导出服务器的函数
def launch_metrics_export_server(q, host=None, port=None):
    # 创建FastAPI应用
    app = FastAPI()
    # 添加/metrics路由，用于暴露指标
    app.add_route("/metrics", metrics)

    # 根路由重定向到 /metrics
    @app.get("/")
    async def root():
        response = RedirectResponse(url="/metrics")
        return response

    # 主异步函数
    async def main():
        # 根据提供的主机和端口配置服务器
        if host is not None and port is not None:
            # 如果主机和端口都指定了
            config = uvicorn.Config(
                app, host=host, port=port, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL
            )
        elif host is not None:
            # 如果只指定了主机，端口使用0（系统自动分配）
            config = uvicorn.Config(
                app, host=host, port=0, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL
            )
        elif port is not None:
            # 如果只指定了端口
            config = uvicorn.Config(
                app, port=port, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL
            )
        else:
            # 如果既没有指定主机也没有指定端口
            config = uvicorn.Config(app, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL)

        # 创建uvicorn服务器实例
        server = uvicorn.Server(config)
        # 创建异步任务来运行服务器
        task = asyncio.create_task(server.serve())

        # 等待服务器启动
        # 这个循环会持续检查服务器是否已启动，每0.1秒检查一次。这确保了在继续执行后续代码之前，服务器已经完全启动。

        while not server.started and not task.done():
            await asyncio.sleep(0.1)

        # 获取服务器的套接字地址并放入队列
        # q参数是从外
        # 部传入launch_metrics_export_server函数的一个队列(queue)对象。这个队列的主要用途是:
        # 作为一个通信渠道,用于将启动的指标服务器的地址信息传递回调用者。
        # 具体来说,代码中这部分在使用队列:
        
        # 调用:

        # import queue

        # q = queue.Queue()
        # launch_metrics_export_server(q)
        
        # # 稍后获取地址信息
        # server_address = q.get()
        # 总之,q是一个用于进程间通信的队列对象,用来传递服务器地址信息给调用者
        
        for server in server.servers:
            for socket in server.sockets:
                q.put(socket.getsockname())
        # 等待服务器任务完成
        await task

    # 运行主异步函数
    asyncio.run(main())
