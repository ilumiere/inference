import asyncio
import threading

# 创建时间循环
# loop = asyncio.new_event_loop()


class Isolation:
    def __init__(self, loop) -> None:
        self._loop = loop
        self._thread = None

        
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self):
        if not self._thread:
            # 创建并启动线程
            self._thread = threading.Thread(target=self._run_loop)
            self._thread.start()
           
    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
    
    def run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
    
    
async def async_task():
    print("Async task running")
    await asyncio.sleep(1)
    print("Async task completed")
    return "Task Result"

def main():
    loop = asyncio.new_event_loop()
    isolation = Isolation(loop)
    isolation.start()

    result = isolation.run_async(async_task())
    print(f"Async task result: {result}")

    isolation.stop()

if __name__ == "__main__":
    main()
