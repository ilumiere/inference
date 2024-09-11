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
from typing import List

from ..._compat import BaseModel


class LoginUserForm(BaseModel):
    """
    用户登录表单模型。

    属性:
        username (str): 用户名。
        password (str): 用户密码。
    """

    username: str
    password: str


class User(LoginUserForm):
    """
    用户模型，继承自LoginUserForm。

    属性:
        username (str): 用户名，继承自LoginUserForm。
        password (str): 用户密码，继承自LoginUserForm。
        permissions (List[str]): 用户权限列表。
        api_keys (List[str]): 用户的API密钥列表。
    """

    permissions: List[str]
    api_keys: List[str]


class AuthConfig(BaseModel):
    """
    认证配置模型。

    属性:
        algorithm (str): 用于令牌加密的算法，默认为"HS256"。
        secret_key (str): 用于令牌加密和解密的密钥。
        token_expire_in_minutes (int): 令牌过期时间，以分钟为单位。
    """

    algorithm: str = "HS256"
    secret_key: str
    token_expire_in_minutes: int


class AuthStartupConfig(BaseModel):
    """
    认证启动配置模型。

    属性:
        auth_config (AuthConfig): 认证配置对象。
        user_config (List[User]): 用户配置列表。
    """

    auth_config: AuthConfig
    user_config: List[User]
