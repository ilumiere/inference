# Copyright 2022-2024 XProbe Inc.
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

import importlib


def get_launcher(launcher_name: str):
    """
    根据给定的启动器名称获取启动器函数。

    :param launcher_name: 启动器的名称，可以是完整的模块路径或单个函数名
    :return: 启动器函数
    :raises ValueError: 如果找不到指定的启动器
    :raises ImportError: 如果导入启动器模块失败
    """
    try:
        # 查找最后一个点的位置，用于分割模块名和函数名
        i = launcher_name.rfind(".")
        if i != -1:
            # 如果存在点，则分别导入模块和获取函数
            module = importlib.import_module(launcher_name[:i])
            fn = getattr(module, launcher_name[i + 1 :])
        else:
            # 如果不存在点，则直接导入整个模块
            importlib.import_module(launcher_name)
            fn = locals().get(launcher_name)

        # 检查是否成功获取到启动器函数
        if fn is None:
            raise ValueError(f"启动器 {launcher_name} 未找到。")

        return fn
    except ImportError as e:
        # 如果导入失败，抛出带有详细信息的ImportError
        raise ImportError(f"导入 {launcher_name} 失败: {e}")
