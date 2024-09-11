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
import re
from datetime import timedelta
from typing import List, Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from typing_extensions import Annotated

from ..._compat import BaseModel, ValidationError, parse_file_as
from .types import AuthStartupConfig, User
from .utils import create_access_token, get_password_hash, verify_password

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# TokenData 类用于存储令牌中的用户信息
class TokenData(BaseModel):
    username: str  # 用户名
    scopes: List[str] = []  # 用户权限范围列表，默认为空列表

class AuthService:
    """
    认证服务类，用于处理用户认证、API密钥验证和令牌生成等功能。
    
    该类提供了一系列方法来管理和验证用户身份，包括API密钥验证、用户认证、
    令牌生成等。它是整个认证系统的核心组件。

    属性:
        _auth_config_file (Optional[str]): 认证配置文件的路径。
        _config (AuthStartupConfig): 解析后的认证配置对象，包含所有认证相关的设置。
    """

    def __init__(self, auth_config_file: Optional[str]):
        """
        初始化AuthService实例。
        
        该方法设置认证配置文件路径并初始化认证配置。

        参数:
            auth_config_file (Optional[str]): 认证配置文件的路径。如果为None，则不使用配置文件。
        """
        self._auth_config_file = auth_config_file
        self._config = self.init_auth_config()

    @property
    def config(self):
        """
        获取认证配置。
        
        这是一个属性装饰器，允许直接访问认证配置。

        返回:
            AuthStartupConfig: 认证配置对象，包含所有认证相关的设置。
        """
        return self._config

    @staticmethod
    def is_legal_api_key(key: str) -> bool:
        """
        验证API密钥是否合法。
        
        使用正则表达式检查API密钥的格式是否符合要求。

        参数:
            key (str): 待验证的API密钥。

        返回:
            bool: 如果API密钥合法则返回True，否则返回False。
        """
        pattern = re.compile("^sk-[a-zA-Z0-9]{13}$")
        return re.match(pattern, key) is not None

    def init_auth_config(self):
        """
        初始化认证配置。
        
        该方法读取配置文件，解析用户配置，并进行必要的验证。

        返回:
            AuthStartupConfig: 解析后的认证配置对象。

        异常:
            ValueError: 当API密钥格式不正确或存在重复时抛出。
        """
        if self._auth_config_file:
            # 解析配置文件
            config: AuthStartupConfig = parse_file_as(  # type: ignore
                path=self._auth_config_file, type_=AuthStartupConfig
            )
            all_api_keys = set()
            for user in config.user_config:
                # 对每个用户的密码进行哈希处理
                user.password = get_password_hash(user.password)
                for api_key in user.api_keys:
                    # 验证API密钥的合法性
                    if not self.is_legal_api_key(api_key):
                        raise ValueError(
                            "Api-Key should be a string started with 'sk-' with a total length of 16"
                        )
                    # 检查API密钥是否重复
                    if api_key in all_api_keys:
                        raise ValueError(
                            "Duplicate api-keys exists, please check your configuration"
                        )
                    else:
                        all_api_keys.add(api_key)
            return config
        # 如果没有配置文件，返回None
        return None

    def __call__(
        self,
        security_scopes: SecurityScopes,
        token: Annotated[str, Depends(oauth2_scheme)],
    ):
        """
        Advanced dependencies. See: https://fastapi.tiangolo.com/advanced/advanced-dependencies/
        认证服务的调用方法，用于验证用户令牌和权限。

        此方法作为FastAPI的高级依赖项使用，负责处理用户认证和授权。
        它支持两种认证方式：API密钥和JWT令牌。

        参数:
            security_scopes (SecurityScopes): 包含当前请求所需的安全作用域。
            token (str): 用户提供的认证令牌，可能是API密钥或JWT令牌。

        返回:
            User: 认证成功的用户对象。

        异常:
            HTTPException: 
                - 当认证失败时，抛出401 Unauthorized异常。
                - 当权限不足时，抛出403 Forbidden异常。

        工作流程:
        1. 设置认证头
        2. 验证令牌类型（API密钥或JWT）
        3. 解析令牌并获取用户信息
        4. 验证用户权限
        5. 返回认证用户或抛出相应异常
        """
        # 根据安全作用域设置认证头
        if security_scopes.scopes:
            authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
        else:
            authenticate_value = "Bearer"
        
        # 定义认证失败时的异常
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )

        # 判断令牌类型并进行相应的认证
        if self.is_legal_api_key(token):
            # 如果是合法的API密钥，直接获取用户和权限
            user, token_scopes = self.get_user_and_scopes_with_api_key(token)
        else:
            # 如果不是API密钥，尝试解析为JWT令牌
            try:
                assert self._config is not None
                # 解码JWT令牌
                payload = jwt.decode(
                    token,
                    self._config.auth_config.secret_key,
                    algorithms=[self._config.auth_config.algorithm],
                    options={"verify_exp": False},  # TODO: 支持令牌过期
                )
                # 从payload中提取用户名和权限范围
                username: str = payload.get("sub")
                if username is None:
                    raise credentials_exception
                token_scopes = payload.get("scopes", [])
                # 获取用户对象
                user = self.get_user(username)
            except (JWTError, ValidationError):
                # JWT解码失败或验证错误时抛出异常
                raise credentials_exception
        
        # 验证用户是否存在
        if user is None:
            raise credentials_exception
        
        # 检查用户权限
        if "admin" in token_scopes:
            # 管理员拥有所有权限
            return user
        
        # 验证用户是否具有所需的所有权限
        for scope in security_scopes.scopes:
            if scope not in token_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions",
                    headers={"WWW-Authenticate": authenticate_value},
                )
        
        # 认证成功，返回用户对象
        return user

    def get_user(self, username: str) -> Optional[User]:
        """
        根据用户名获取用户对象。

        参数:
            username (str): 用户名。

        返回:
            Optional[User]: 如果找到用户则返回User对象，否则返回None。
        """
        for user in self._config.user_config:
            if user.username == username:
                return user
        return None

    def get_user_and_scopes_with_api_key(
        self, api_key: str
    ) -> Tuple[Optional[User], List]:
        """
        根据API密钥获取用户对象和权限范围。

        此方法用于验证API密钥并返回相应的用户信息和权限。它遍历所有用户配置，
        检查提供的API密钥是否匹配任何用户的API密钥。

        参数:
            api_key (str): 待验证的API密钥。

        返回:
            Tuple[Optional[User], List]: 
            - 如果找到匹配的API密钥，返回一个元组，包含User对象和该用户的权限列表。
            - 如果未找到匹配的API密钥，返回(None, [])。

        注意:
            - 此方法假设每个用户可能有多个API密钥。
            - 返回的权限列表直接来自用户配置，未经进一步处理。
        """
        # 遍历所有用户配置
        for user in self._config.user_config:
            # 检查当前用户的所有API密钥
            for key in user.api_keys:
                # 如果找到匹配的API密钥
                if api_key == key:
                    # 返回用户对象和该用户的权限列表
                    return user, user.permissions
        # 如果未找到匹配的API密钥，返回None和空列表
        return None, []

    def authenticate_user(self, username: str, password: str):
        """
        验证用户名和密码。

        此方法用于用户认证，检查提供的用户名和密码是否匹配。

        参数:
            username (str): 待验证的用户名。
            password (str): 待验证的密码。

        返回:
            Union[User, bool]: 
            - 如果认证成功，返回对应的User对象。
            - 如果认证失败，返回False。

        流程:
            1. 首先通过用户名获取用户对象。
            2. 如果用户不存在，直接返回False。
            3. 如果用户存在，验证密码。
            4. 如果密码正确，返回用户对象；否则返回False。
        """
        # 尝试获取用户对象
        user = self.get_user(username)
        # 如果用户不存在，返回False
        if not user:
            return False
        # 验证密码
        if not verify_password(password, user.password):
            return False
        # 认证成功，返回用户对象
        return user

    def generate_token_for_user(self, username: str, password: str):
        """
        为用户生成访问令牌。

        此方法首先验证用户凭据，然后为通过验证的用户生成一个JWT访问令牌。

        参数:
            username (str): 用户名。
            password (str): 密码。

        返回:
            dict: 包含访问令牌和令牌类型的字典。
                  格式: {"access_token": <token>, "token_type": "bearer"}

        异常:
            HTTPException: 
                - 当用户名或密码不正确时抛出，状态码401。

        流程:
            1. 调用authenticate_user方法验证用户凭据。
            2. 如果验证失败，抛出HTTPException。
            3. 设置访问令牌的过期时间。
            4. 创建包含用户信息和权限的访问令牌。
            5. 返回包含访问令牌和类型的字典。
        """
        # 验证用户凭据
        user = self.authenticate_user(username, password)
        if not user:
            # 验证失败，抛出异常
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # 确保user是User类型的实例
        assert user is not None and isinstance(user, User)
        
        # 设置访问令牌的过期时间
        access_token_expires = timedelta(
            minutes=self._config.auth_config.token_expire_in_minutes
        )
        
        # 创建访问令牌
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.permissions},
            secret_key=self._config.auth_config.secret_key,
            algorithm=self._config.auth_config.algorithm,
            expires_delta=access_token_expires,
        )
        
        # 返回包含访问令牌和类型的字典
        return {"access_token": access_token, "token_type": "bearer"}
