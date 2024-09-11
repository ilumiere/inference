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
from datetime import datetime, timedelta
from typing import Union

from jose import jwt
from passlib.context import CryptContext

# 创建密码上下文，使用bcrypt算法进行密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    data: dict,
    secret_key: str,
    algorithm: str,
    expires_delta: Union[timedelta, None] = None,
) -> str:
    """
    创建访问令牌。

    参数:
        data (dict): 要编码到令牌中的数据。
        secret_key (str): 用于签名令牌的密钥。
        algorithm (str): 用于签名令牌的算法。
        expires_delta (Union[timedelta, None]): 令牌的过期时间增量，默认为None。

    返回:
        str: 编码后的JWT令牌。

    逻辑:
        1. 复制输入数据。
        2. 设置过期时间（如果提供了expires_delta，则使用它；否则默认15分钟）。
        3. 将过期时间添加到待编码数据中。
        4. 使用提供的密钥和算法对数据进行编码。
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码是否与哈希密码匹配。

    参数:
        plain_password (str): 待验证的明文密码。
        hashed_password (str): 存储的哈希密码。

    返回:
        bool: 如果密码匹配返回True，否则返回False。
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    对给定的密码进行哈希处理。

    参数:
        password (str): 需要哈希的明文密码。

    返回:
        str: 哈希后的密码。
    """
    return pwd_context.hash(password)
