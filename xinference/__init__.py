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

# 导入版本信息模块
from . import _version

# 获取并设置当前包的版本号
__version__ = _version.get_versions()["version"]


# 尝试导入 Intel 的 PyTorch 扩展
# 这可能用于优化特定硬件上的性能
try:
    import intel_extension_for_pytorch  # noqa: F401
except:
    pass  # 如果导入失败，静默忽略


def _install():
    # 从 xoscar 库导入 Router 类
    from xoscar.backends.router import Router

    # 从当前包的 model 模块导入 _install 函数，并重命名为 install_model
    from .model import _install as install_model

    # 获取默认的路由器实例，如果不存在则返回空实例
    default_router = Router.get_instance_or_empty()
    # 设置获取到的路由器实例为当前使用的实例
    Router.set_instance(default_router)

    # 调用 model 模块的安装函数
    install_model()


# 执行安装函数
_install()
# 删除 _install 函数，防止外部直接调用
del _install
