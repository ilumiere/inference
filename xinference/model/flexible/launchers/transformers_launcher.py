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

from transformers import pipeline

from ..core import FlexibleModel, FlexibleModelSpec


class MockModel(FlexibleModel):
    def infer(self, **kwargs):
        # 模拟推理，直接返回输入的参数
        return kwargs


class AutoModel(FlexibleModel):
    def load(self):
        # 获取配置，如果没有则使用空字典
        config = self.config or {}
        # 使用transformers的pipeline创建模型
        # 传入模型路径、设备信息和其他配置参数
        self._pipeline = pipeline(model=self.model_path, device=self.device, **config)

    def infer(self, **kwargs):
        # 使用创建的pipeline进行推理
        # 将所有传入的参数传递给pipeline
        return self._pipeline(**kwargs)


class TransformersTextClassificationModel(FlexibleModel):
    def load(self):
        # 获取配置，如果没有则使用空字典
        config = self.config or {}

        # 使用transformers的pipeline创建文本分类模型
        # 传入模型路径、设备信息和其他配置参数
        self._pipeline = pipeline(model=self._model_path, device=self._device, **config)

    def infer(self, **kwargs):
        # 使用创建的pipeline进行推理
        # 将所有传入的参数传递给pipeline
        return self._pipeline(**kwargs)


def launcher(model_uid: str, model_spec: FlexibleModelSpec, **kwargs) -> FlexibleModel:
    # 从kwargs中获取任务类型和设备信息
    task = kwargs.get("task")
    device = kwargs.get("device")

    # 获取模型路径
    model_path = model_spec.model_uri
    # 如果模型路径为空，抛出异常
    if model_path is None:
        raise ValueError("model_path required")

    # 根据任务类型返回相应的模型实例
    if task == "text-classification":
        # 返回文本分类模型实例
        return TransformersTextClassificationModel(
            model_uid=model_uid, model_path=model_path, device=device, config=kwargs
        )
    elif task == "mock":
        # 返回模拟模型实例
        return MockModel(
            model_uid=model_uid, model_path=model_path, device=device, config=kwargs
        )
    else:
        # 返回自动模型实例
        return AutoModel(
            model_uid=model_uid, model_path=model_path, device=device, config=kwargs
        )
