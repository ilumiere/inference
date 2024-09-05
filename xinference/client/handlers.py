# 通过使用 as，可以为导入的类提供一个更简洁或更通用的名称。
# 例如，RESTfulAudioModelHandle 被简化为 AudioModelHandle。
# 导入音频模型处理器
# 通过在 handlers.py 中进行这种导入和重命名,您可以在项目的其他部分使用更简洁和抽象的接口,
# 而无需关心具体的实现细节。这提高了代码的可维护性和灵活性

from .restful.restful_client import (  # noqa: F401
    RESTfulAudioModelHandle as AudioModelHandle,
)

# 导入聊天模型处理器
from .restful.restful_client import (  # noqa: F401
    RESTfulChatModelHandle as ChatModelHandle,
)

# 导入嵌入模型处理器
from .restful.restful_client import (  # noqa: F401
    RESTfulEmbeddingModelHandle as EmbeddingModelHandle,
)

# 导入生成模型处理器
from .restful.restful_client import (  # noqa: F401
    RESTfulGenerateModelHandle as GenerateModelHandle,
)

# 导入图像模型处理器
from .restful.restful_client import (  # noqa: F401
    RESTfulImageModelHandle as ImageModelHandle,
)
