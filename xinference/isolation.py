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
import threading
from typing import Any, Coroutine


class Isolation:
    # TODO: 将隔离移至xoscar更好。
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        threaded: bool = True,
        daemon: bool = True,
    ):
        self._loop = loop  # 事件循环
        self._threaded = threaded  # 是否使用线程

        self._stopped = None  # 停止事件
        self._thread = None  # 线程对象
        self._thread_ident = None  # 线程标识符
        self._daemon = daemon  # 是否为守护线程

    def _run(self):
        # 设置事件循环并等待停止事件
        asyncio.set_event_loop(self._loop)
        self._stopped = asyncio.Event()
        # 运行事件循环直到某个特定的 Future 或 Task 完成
        self._loop.run_until_complete(self._stopped.wait())

    def start(self):
        # 如果使用线程，则启动线程
        if self._threaded:
            self._thread = thread = threading.Thread(target=self._run)
            if self._daemon:
                thread.daemon = True
            thread.start()
            # 存储线程的标识符
            self._thread_ident = thread.ident
            print(f"thread.ident: {self._thread_ident}")
    def call(self, coro: Coroutine) -> Any:
        # 在事件循环中运行协程并返回结果
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @property
    def thread_ident(self):
        # 获取线程标识符
        return self._thread_ident

    @property
    def loop(self):
        # 获取事件循环
        return self._loop

    async def _stop(self):
        # 设置停止事件
        self._stopped.set()

    def stop(self):
        # 如果使用线程，则停止线程
        if self._threaded:
            asyncio.run_coroutine_threadsafe(self._stop(), self._loop).result()
            self._thread.join()


# import asyncio
# import threading
# from typing import Any, Coroutine
# 测试用例
# 异步函数示例
# async def async_task():
#     print("异步任务开始")
#     await asyncio.sleep(1)  # 模拟异步操作
#     print("异步任务结束")
#     return "任务完成"

# # 主程序
# def main():
#     loop = asyncio.new_event_loop()  # 创建新的事件循环
#     isolation = Isolation(loop)      # 实例化Isolation类
#     isolation.start()                # 启动隔离的事件循环

#     try:
#         # 在隔离的事件循环中调用异步任务
#         result = isolation.call(async_task())
#         print("异步任务的结果：", result)
#     finally:
#         # 停止隔离的事件循环并等待线程完成
#         isolation.stop()

#     print("主线程继续执行其他任务")



# # 修改 async_task 以接收一个回调函数作为参数
# async def async_task_callback(callback):
#     print("1. 异步任务开始")
#     await asyncio.sleep(1)  # 模拟异步操作
#     print("1. 异步任务结束")
#     callback("1. 任务完成")  # 调用回调函数并传递结果

# def task_callback(result):
#     print("1. 异步任务的结果：", result)
#     # 处理异步任务结果
    

# def callback_main():
#     loop = asyncio.new_event_loop()
#     isolation = Isolation(loop)
#     isolation.start()

#     # 使用回调而不是直接等待结果
#     asyncio.run_coroutine_threadsafe(async_task_callback(task_callback), loop)

#     import time
#     for i in range(5):
#         print(f"1.主线程任务 {i + 1} 执行中...")
#         # threading.sleep(0.5)
#         time.sleep(0.5)

#     isolation.stop()
#     print("1.主线程任务完成")

# if __name__ == "__main__":
#     main()
    
#     callback_main()
