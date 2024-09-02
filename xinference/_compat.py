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
# 导入 Pydantic 的版本信息
from pydantic.version import VERSION as PYDANTIC_VERSION

# 判断是否为 Pydantic V2 版本
PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")


if PYDANTIC_V2:
    # 如果是 Pydantic V2 版本，从 pydantic.v1 导入兼容性模块
    from pydantic.v1 import (  # noqa: F401
        BaseModel,  # 基础模型类
        Field,  # 字段定义
        Protocol,  # 协议类
        ValidationError,  # 验证错误
        create_model,  # 创建模型函数
        create_model_from_namedtuple,  # 从命名元组创建模型
        create_model_from_typeddict,  # 从类型字典创建模型
        parse_file_as,  # 解析文件
        validate_arguments,  # 参数验证装饰器
        validator,  # 验证器装饰器
    )
    from pydantic.v1.error_wrappers import ErrorWrapper  # noqa: F401  # 错误包装器
    from pydantic.v1.parse import load_str_bytes  # noqa: F401  # 加载字符串或字节
    from pydantic.v1.types import StrBytes  # noqa: F401  # 字符串或字节类型
    from pydantic.v1.utils import ROOT_KEY  # noqa: F401  # 根键常量
else:
    # 如果不是 Pydantic V2 版本，直接从 pydantic 导入模块
    from pydantic import (  # noqa: F401
        BaseModel,  # 基础模型类
        Field,  # 字段定义
        Protocol,  # 协议类
        ValidationError,  # 验证错误
        create_model,  # 创建模型函数
        create_model_from_namedtuple,  # 从命名元组创建模型
        create_model_from_typeddict,  # 从类型字典创建模型
        parse_file_as,  # 解析文件
        validate_arguments,  # 参数验证装饰器
        validator,  # 验证器装饰器
    )
    from pydantic.error_wrappers import ErrorWrapper  # noqa: F401  # 错误包装器
    from pydantic.parse import load_str_bytes  # noqa: F401  # 加载字符串或字节
    from pydantic.types import StrBytes  # noqa: F401  # 字符串或字节类型
    from pydantic.utils import ROOT_KEY  # noqa: F401  # 根键常量
