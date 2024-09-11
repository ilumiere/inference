# 通过使用 as，可以为导入的类提供一个更简洁或更通用的名称。
# 例如，RESTfulAudioModelHandle 被简化为 AudioModelHandle。
# 导入音频模型处理器
# 通过在 handlers.py 中进行这种导入和重命名,您可以在项目的其他部分使用更简洁和抽象的接口,
# 而无需关心具体的实现细节。这提高了代码的可维护性和灵活性

from .restful.restful_client import (  # noqa: F401
    RESTfulAudioModelHandle as AudioModelHandle,
)

# 导入并重命名RESTfulChatModelHandle为ChatModelHandle
# ChatModelHandle用于处理聊天相关的AI模型操作
from .restful.restful_client import (  # noqa: F401
    RESTfulChatModelHandle as ChatModelHandle,
)

# 导入并重命名RESTfulEmbeddingModelHandle为EmbeddingModelHandle
# EmbeddingModelHandle用于处理嵌入相关的AI模型操作
from .restful.restful_client import (  # noqa: F401
    RESTfulEmbeddingModelHandle as EmbeddingModelHandle,
)

# 导入并重命名RESTfulGenerateModelHandle为GenerateModelHandle
# GenerateModelHandle用于处理生成相关的AI模型操作
from .restful.restful_client import (  # noqa: F401
    RESTfulGenerateModelHandle as GenerateModelHandle,
)

# 导入并重命名RESTfulImageModelHandle为ImageModelHandle
# ImageModelHandle用于处理图像相关的AI模型操作
from .restful.restful_client import (  # noqa: F401
    RESTfulImageModelHandle as ImageModelHandle,
)

# 注意：每个导入语句后的 # noqa: F401 注释是为了抑制未使用导入的警告
# 这表明虽然这些导入可能在当前文件中未直接使用，但它们对于项目的其他部分是必要的
