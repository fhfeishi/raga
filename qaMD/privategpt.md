# private_gpt 源码学习
## 2025-09-01-0000
private_gpt 的python sdk包源码的结构如下

```python
private_gpt/
    + components/
    + open_ai/
    + server/
    + settings/
    + ui/
    + utils/
    + __init__.py
    + __main__.py
    + constants.py
    + di.py
    + launcher.py
    + main.py
    + path.py
```

首先是工程root目录下的几个py文件

-  `__init__.py` 
   -  定义了logging，
   -  Disable gradio analytics， 
   -  adding tiktoken cache path within repo to be able to run in offline environment.

- `__main__.py`
  -  from private_gpt.main import app
  -  from private_gpt.settings.settings import settings
  -  fastapi  uvicorn启动，  加载了  settings , 导入了 app

- `constants.py` 定义了工程主目录 PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]

现在请你解析，工程主目录下的其余几个py文件：

```python
# di.py
from injector import Injector

from private_gpt.settings.settings import Settings, unsafe_typed_settings


def create_application_injector() -> Injector:
    _injector = Injector(auto_bind=True)
    _injector.binder.bind(Settings, to=unsafe_typed_settings)
    return _injector


"""
Global injector for the application.

Avoid using this reference, it will make your code harder to test.

Instead, use the `request.state.injector` reference, which is bound to every request
"""
global_injector: Injector = create_application_injector()

```



```python
# launcher.py
"""FastAPI app creation, logger configuration and main API routes."""

import logging

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.global_handlers import create_global_handler
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.server.recipes.summarize.summarize_router import summarize_router
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def create_app(root_injector: Injector) -> FastAPI:

    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app = FastAPI(dependencies=[Depends(bind_injector_to_request)])

    app.include_router(completions_router)
    app.include_router(chat_router)
    app.include_router(chunks_router)
    app.include_router(ingest_router)
    app.include_router(summarize_router)
    app.include_router(embeddings_router)
    app.include_router(health_router)

    # Add LlamaIndex simple observability
    global_handler = create_global_handler("simple")
    if global_handler:
        LlamaIndexSettings.callback_manager = CallbackManager([global_handler])

    settings = root_injector.get(Settings)
    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=settings.server.cors.allow_credentials,
            allow_origins=settings.server.cors.allow_origins,
            allow_origin_regex=settings.server.cors.allow_origin_regex,
            allow_methods=settings.server.cors.allow_methods,
            allow_headers=settings.server.cors.allow_headers,
        )

    if settings.ui.enabled:
        logger.debug("Importing the UI module")
        try:
            from private_gpt.ui.ui import PrivateGptUi
        except ImportError as e:
            raise ImportError(
                "UI dependencies not found, install with `poetry install --extras ui`"
            ) from e

        ui = root_injector.get(PrivateGptUi)
        ui.mount_in_app(app, settings.ui.path)

    return app


```



```python
# main.py
"""FastAPI app creation, logger configuration and main API routes."""

from private_gpt.di import global_injector
from private_gpt.launcher import create_app

app = create_app(global_injector)

```


```python
# paths.py
from pathlib import Path

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.settings import settings


def _absolute_or_from_project_root(path: str) -> Path:
    if path.startswith("/"):
        return Path(path)
    return PROJECT_ROOT_PATH / path


models_path: Path = PROJECT_ROOT_PATH / "models"
models_cache_path: Path = models_path / "cache"
docs_path: Path = PROJECT_ROOT_PATH / "docs"
local_data_path: Path = _absolute_or_from_project_root(
    settings().data.local_data_folder
)

```


## 2025-09-02-2229

今天我们来继续拆解private-gpt的工程源码，
目标是：拆解和重构private-gpt的数据处理部分（数据读取、数据分块、数据节点llama-index、检索等），然后尽可能的学习private-gpt的工程设计（代码设计）

首先是settings部分，我希望通过一个settings.py文件设置来代替，我们可以一步一步实现，你给出你的代码，然后告诉我你的理由

我的工程需求：
- 工程框架使用llama-index，考虑到其优质的RAG数据/信息处理接口
- chat_model embedding_model  reranker_model  等 均是本地
- 向量库：目前先使用chroma，之后可能会考虑qdrant、等
- 数据解析、切块、检索，我们来一步一步实现
- 由于前端接口我初步使用了react ts实现，现在先不考虑前端界面展示


private-gpt   工程结构，
    + components/  # 各组件
    + open_ai/     # openai适配
    + server/      # 汇总成服务
    + settings/    各组件的Settings，强制通过settings.yaml读取
    + ui/          # 内置的gradio ui
    + utils/       # 工具部分
    + __init__.py  # logging, gradio设置等
    + __main__.py  # 启动uvicorn
    + constants.py # 路径
    + di.py        # 依赖注入器
    + launcher.py  # def create_app
    + main.py      # creat_app
    + paths.py     # 路径

```python
# private_gpt/settings/settings_loader.py
import functools
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic.v1.utils import deep_update, unique_list

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.yaml import load_yaml_with_envvars

logger = logging.getLogger(__name__)

_settings_folder = os.environ.get("PGPT_SETTINGS_FOLDER", PROJECT_ROOT_PATH)

# if running in unittest, use the test profile
_test_profile = ["test"] if "tests.fixtures" in sys.modules else []

active_profiles: list[str] = unique_list(
    ["default"]
    + [
        item.strip()
        for item in os.environ.get("PGPT_PROFILES", "").split(",")
        if item.strip()
    ]
    + _test_profile
)


def merge_settings(settings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return functools.reduce(deep_update, settings, {})


def load_settings_from_profile(profile: str) -> dict[str, Any]:
    if profile == "default":
        profile_file_name = "settings.yaml"
    else:
        profile_file_name = f"settings-{profile}.yaml"

    path = Path(_settings_folder) / profile_file_name
    with Path(path).open("r") as f:
        config = load_yaml_with_envvars(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file has no top-level mapping: {path}")
    return config


def load_active_settings() -> dict[str, Any]:
    """Load active profiles and merge them."""
    logger.info("Starting application with profiles=%s", active_profiles)
    loaded_profiles = [
        load_settings_from_profile(profile) for profile in active_profiles
    ]
    merged: dict[str, Any] = merge_settings(loaded_profiles)
    return merged


# private_gpt/settings/yaml.py
from typing import Any, Literal

from pydantic import BaseModel, Field

from private_gpt.settings.settings_loader import load_active_settings


class CorsSettings(BaseModel):
    """CORS configuration.

    For more details on the CORS configuration, see:
    # * https://fastapi.tiangolo.com/tutorial/cors/
    # * https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
    """

    enabled: bool = Field(
        description="Flag indicating if CORS headers are set or not."
        "If set to True, the CORS headers will be set to allow all origins, methods and headers.",
        default=False,
    )
    allow_credentials: bool = Field(
        description="Indicate that cookies should be supported for cross-origin requests",
        default=False,
    )
    allow_origins: list[str] = Field(
        description="A list of origins that should be permitted to make cross-origin requests.",
        default=[],
    )
    allow_origin_regex: list[str] = Field(
        description="A regex string to match against origins that should be permitted to make cross-origin requests.",
        default=None,
    )
    allow_methods: list[str] = Field(
        description="A list of HTTP methods that should be allowed for cross-origin requests.",
        default=[
            "GET",
        ],
    )
    allow_headers: list[str] = Field(
        description="A list of HTTP request headers that should be supported for cross-origin requests.",
        default=[],
    )


class AuthSettings(BaseModel):
    """Authentication configuration.

    The implementation of the authentication strategy must
    """

    enabled: bool = Field(
        description="Flag indicating if authentication is enabled or not.",
        default=False,
    )
    secret: str = Field(
        description="The secret to be used for authentication. "
        "It can be any non-blank string. For HTTP basic authentication, "
        "this value should be the whole 'Authorization' header that is expected"
    )


class IngestionSettings(BaseModel):
    """Ingestion configuration.

    This configuration is used to control the ingestion of data into the system
    using non-server methods. This is useful for local development and testing;
    or to ingest in bulk from a folder.

    Please note that this configuration is not secure and should be used in
    a controlled environment only (setting right permissions, etc.).
    """

    enabled: bool = Field(
        description="Flag indicating if local ingestion is enabled or not.",
        default=False,
    )
    allow_ingest_from: list[str] = Field(
        description="A list of folders that should be permitted to make ingest requests.",
        default=[],
    )


class ServerSettings(BaseModel):
    env_name: str = Field(
        description="Name of the environment (prod, staging, local...)"
    )
    port: int = Field(description="Port of PrivateGPT FastAPI server, defaults to 8001")
    cors: CorsSettings = Field(
        description="CORS configuration", default=CorsSettings(enabled=False)
    )
    auth: AuthSettings = Field(
        description="Authentication configuration",
        default_factory=lambda: AuthSettings(enabled=False, secret="secret-key"),
    )


class DataSettings(BaseModel):
    local_ingestion: IngestionSettings = Field(
        description="Ingestion configuration",
        default_factory=lambda: IngestionSettings(allow_ingest_from=["*"]),
    )
    local_data_folder: str = Field(
        description="Path to local storage."
        "It will be treated as an absolute path if it starts with /"
    )


class LLMSettings(BaseModel):
    mode: Literal[
        "llamacpp",
        "openai",
        "openailike",
        "azopenai",
        "sagemaker",
        "mock",
        "ollama",
        "gemini",
    ]
    max_new_tokens: int = Field(
        256,
        description="The maximum number of token that the LLM is authorized to generate in one completion.",
    )
    context_window: int = Field(
        3900,
        description="The maximum number of context tokens for the model.",
    )
    tokenizer: str = Field(
        None,
        description="The model id of a predefined tokenizer hosted inside a model repo on "
        "huggingface.co. Valid model ids can be located at the root-level, like "
        "`bert-base-uncased`, or namespaced under a user or organization name, "
        "like `HuggingFaceH4/zephyr-7b-beta`. If not set, will load a tokenizer matching "
        "gpt-3.5-turbo LLM.",
    )
    temperature: float = Field(
        0.1,
        description="The temperature of the model. Increasing the temperature will make the model answer more creatively. A value of 0.1 would be more factual.",
    )
    prompt_style: Literal["default", "llama2", "llama3", "tag", "mistral", "chatml"] = (
        Field(
            "llama2",
            description=(
                "The prompt style to use for the chat engine. "
                "If `default` - use the default prompt style from the llama_index. It should look like `role: message`.\n"
                "If `llama2` - use the llama2 prompt style from the llama_index. Based on `<s>`, `[INST]` and `<<SYS>>`.\n"
                "If `llama3` - use the llama3 prompt style from the llama_index."
                "If `tag` - use the `tag` prompt style. It should look like `<|role|>: message`. \n"
                "If `mistral` - use the `mistral prompt style. It shoudl look like <s>[INST] {System Prompt} [/INST]</s>[INST] { UserInstructions } [/INST]"
                "`llama2` is the historic behaviour. `default` might work better with your custom models."
            ),
        )
    )


class VectorstoreSettings(BaseModel):
    database: Literal["chroma", "qdrant", "postgres", "clickhouse", "milvus"]


class NodeStoreSettings(BaseModel):
    database: Literal["simple", "postgres"]


class LlamaCPPSettings(BaseModel):
    llm_hf_repo_id: str
    llm_hf_model_file: str
    tfs_z: float = Field(
        1.0,
        description="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting.",
    )
    top_k: int = Field(
        40,
        description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
    )
    top_p: float = Field(
        0.9,
        description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
    )
    repeat_penalty: float = Field(
        1.1,
        description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
    )


class HuggingFaceSettings(BaseModel):
    embedding_hf_model_name: str = Field(
        description="Name of the HuggingFace model to use for embeddings"
    )
    access_token: str = Field(
        None,
        description="Huggingface access token, required to download some models",
    )
    trust_remote_code: bool = Field(
        False,
        description="If set to True, the code from the remote model will be trusted and executed.",
    )


class EmbeddingSettings(BaseModel):
    mode: Literal[
        "huggingface",
        "openai",
        "azopenai",
        "sagemaker",
        "ollama",
        "mock",
        "gemini",
        "mistralai",
    ]
    ingest_mode: Literal["simple", "batch", "parallel", "pipeline"] = Field(
        "simple",
        description=(
            "The ingest mode to use for the embedding engine:\n"
            "If `simple` - ingest files sequentially and one by one. It is the historic behaviour.\n"
            "If `batch` - if multiple files, parse all the files in parallel, "
            "and send them in batch to the embedding model.\n"
            "In `pipeline` - The Embedding engine is kept as busy as possible\n"
            "If `parallel` - parse the files in parallel using multiple cores, and embedd them in parallel.\n"
            "`parallel` is the fastest mode for local setup, as it parallelize IO RW in the index.\n"
            "For modes that leverage parallelization, you can specify the number of "
            "workers to use with `count_workers`.\n"
        ),
    )
    count_workers: int = Field(
        2,
        description=(
            "The number of workers to use for file ingestion.\n"
            "In `batch` mode, this is the number of workers used to parse the files.\n"
            "In `parallel` mode, this is the number of workers used to parse the files and embed them.\n"
            "In `pipeline` mode, this is the number of workers that can perform embeddings.\n"
            "This is only used if `ingest_mode` is not `simple`.\n"
            "Do not go too high with this number, as it might cause memory issues. (especially in `parallel` mode)\n"
            "Do not set it higher than your number of threads of your CPU."
        ),
    )
    embed_dim: int = Field(
        384,
        description="The dimension of the embeddings stored in the Postgres database",
    )


class SagemakerSettings(BaseModel):
    llm_endpoint_name: str
    embedding_endpoint_name: str


class OpenAISettings(BaseModel):
    api_base: str = Field(
        None,
        description="Base URL of OpenAI API. Example: 'https://api.openai.com/v1'.",
    )
    api_key: str
    model: str = Field(
        "gpt-3.5-turbo",
        description="OpenAI Model to use. Example: 'gpt-4'.",
    )
    request_timeout: float = Field(
        120.0,
        description="Time elapsed until openailike server times out the request. Default is 120s. Format is float. ",
    )
    embedding_api_base: str = Field(
        None,
        description="Base URL of OpenAI API. Example: 'https://api.openai.com/v1'.",
    )
    embedding_api_key: str
    embedding_model: str = Field(
        "text-embedding-ada-002",
        description="OpenAI embedding Model to use. Example: 'text-embedding-3-large'.",
    )


class GeminiSettings(BaseModel):
    api_key: str
    model: str = Field(
        "models/gemini-pro",
        description="Google Model to use. Example: 'models/gemini-pro'.",
    )
    embedding_model: str = Field(
        "models/embedding-001",
        description="Google Embedding Model to use. Example: 'models/embedding-001'.",
    )


class OllamaSettings(BaseModel):
    api_base: str = Field(
        "http://localhost:11434",
        description="Base URL of Ollama API. Example: 'https://localhost:11434'.",
    )
    embedding_api_base: str = Field(
        "http://localhost:11434",
        description="Base URL of Ollama embedding API. Example: 'https://localhost:11434'.",
    )
    llm_model: str = Field(
        None,
        description="Model to use. Example: 'llama2-uncensored'.",
    )
    embedding_model: str = Field(
        None,
        description="Model to use. Example: 'nomic-embed-text'.",
    )
    keep_alive: str = Field(
        "5m",
        description="Time the model will stay loaded in memory after a request. examples: 5m, 5h, '-1' ",
    )
    tfs_z: float = Field(
        1.0,
        description="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting.",
    )
    num_predict: int = Field(
        None,
        description="Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)",
    )
    top_k: int = Field(
        40,
        description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
    )
    top_p: float = Field(
        0.9,
        description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
    )
    repeat_last_n: int = Field(
        64,
        description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)",
    )
    repeat_penalty: float = Field(
        1.1,
        description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
    )
    request_timeout: float = Field(
        120.0,
        description="Time elapsed until ollama times out the request. Default is 120s. Format is float. ",
    )
    autopull_models: bool = Field(
        False,
        description="If set to True, the Ollama will automatically pull the models from the API base.",
    )


class AzureOpenAISettings(BaseModel):
    api_key: str
    azure_endpoint: str
    api_version: str = Field(
        "2023_05_15",
        description="The API version to use for this operation. This follows the YYYY-MM-DD format.",
    )
    embedding_deployment_name: str
    embedding_model: str = Field(
        "text-embedding-ada-002",
        description="OpenAI Model to use. Example: 'text-embedding-ada-002'.",
    )
    llm_deployment_name: str
    llm_model: str = Field(
        "gpt-35-turbo",
        description="OpenAI Model to use. Example: 'gpt-4'.",
    )


class UISettings(BaseModel):
    enabled: bool
    path: str
    default_mode: Literal["RAG", "Search", "Basic", "Summarize"] = Field(
        "RAG",
        description="The default mode.",
    )
    default_chat_system_prompt: str = Field(
        None,
        description="The default system prompt to use for the chat mode.",
    )
    default_query_system_prompt: str = Field(
        None, description="The default system prompt to use for the query mode."
    )
    default_summarization_system_prompt: str = Field(
        None,
        description="The default system prompt to use for the summarization mode.",
    )
    delete_file_button_enabled: bool = Field(
        True, description="If the button to delete a file is enabled or not."
    )
    delete_all_files_button_enabled: bool = Field(
        False, description="If the button to delete all files is enabled or not."
    )


class RerankSettings(BaseModel):
    enabled: bool = Field(
        False,
        description="This value controls whether a reranker should be included in the RAG pipeline.",
    )
    model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-2-v2",
        description="Rerank model to use. Limited to SentenceTransformer cross-encoder models.",
    )
    top_n: int = Field(
        2,
        description="This value controls the number of documents returned by the RAG pipeline.",
    )


class RagSettings(BaseModel):
    similarity_top_k: int = Field(
        2,
        description="This value controls the number of documents returned by the RAG pipeline or considered for reranking if enabled.",
    )
    similarity_value: float = Field(
        None,
        description="If set, any documents retrieved from the RAG must meet a certain match score. Acceptable values are between 0 and 1.",
    )
    rerank: RerankSettings


class SummarizeSettings(BaseModel):
    use_async: bool = Field(
        True,
        description="If set to True, the summarization will be done asynchronously.",
    )


class ClickHouseSettings(BaseModel):
    host: str = Field(
        "localhost",
        description="The server hosting the ClickHouse database",
    )
    port: int = Field(
        8443,
        description="The port on which the ClickHouse database is accessible",
    )
    username: str = Field(
        "default",
        description="The username to use to connect to the ClickHouse database",
    )
    password: str = Field(
        "",
        description="The password to use to connect to the ClickHouse database",
    )
    database: str = Field(
        "__default__",
        description="The default database to use for connections",
    )
    secure: bool | str = Field(
        False,
        description="Use https/TLS for secure connection to the server",
    )
    interface: str | None = Field(
        None,
        description="Must be either 'http' or 'https'. Determines the protocol to use for the connection",
    )
    settings: dict[str, Any] | None = Field(
        None,
        description="Specific ClickHouse server settings to be used with the session",
    )
    connect_timeout: int | None = Field(
        None,
        description="Timeout in seconds for establishing a connection",
    )
    send_receive_timeout: int | None = Field(
        None,
        description="Read timeout in seconds for http connection",
    )
    verify: bool | None = Field(
        None,
        description="Verify the server certificate in secure/https mode",
    )
    ca_cert: str | None = Field(
        None,
        description="Path to Certificate Authority root certificate (.pem format)",
    )
    client_cert: str | None = Field(
        None,
        description="Path to TLS Client certificate (.pem format)",
    )
    client_cert_key: str | None = Field(
        None,
        description="Path to the private key for the TLS Client certificate",
    )
    http_proxy: str | None = Field(
        None,
        description="HTTP proxy address",
    )
    https_proxy: str | None = Field(
        None,
        description="HTTPS proxy address",
    )
    server_host_name: str | None = Field(
        None,
        description="Server host name to be checked against the TLS certificate",
    )


class PostgresSettings(BaseModel):
    host: str = Field(
        "localhost",
        description="The server hosting the Postgres database",
    )
    port: int = Field(
        5432,
        description="The port on which the Postgres database is accessible",
    )
    user: str = Field(
        "postgres",
        description="The user to use to connect to the Postgres database",
    )
    password: str = Field(
        "postgres",
        description="The password to use to connect to the Postgres database",
    )
    database: str = Field(
        "postgres",
        description="The database to use to connect to the Postgres database",
    )
    schema_name: str = Field(
        "public",
        description="The name of the schema in the Postgres database to use",
    )


class QdrantSettings(BaseModel):
    location: str | None = Field(
        None,
        description=(
            "If `:memory:` - use in-memory Qdrant instance.\n"
            "If `str` - use it as a `url` parameter.\n"
        ),
    )
    url: str | None = Field(
        None,
        description=(
            "Either host or str of 'Optional[scheme], host, Optional[port], Optional[prefix]'."
        ),
    )
    port: int | None = Field(6333, description="Port of the REST API interface.")
    grpc_port: int | None = Field(6334, description="Port of the gRPC interface.")
    prefer_grpc: bool | None = Field(
        False,
        description="If `true` - use gRPC interface whenever possible in custom methods.",
    )
    https: bool | None = Field(
        None,
        description="If `true` - use HTTPS(SSL) protocol.",
    )
    api_key: str | None = Field(
        None,
        description="API key for authentication in Qdrant Cloud.",
    )
    prefix: str | None = Field(
        None,
        description=(
            "Prefix to add to the REST URL path."
            "Example: `service/v1` will result in "
            "'http://localhost:6333/service/v1/{qdrant-endpoint}' for REST API."
        ),
    )
    timeout: float | None = Field(
        None,
        description="Timeout for REST and gRPC API requests.",
    )
    host: str | None = Field(
        None,
        description="Host name of Qdrant service. If url and host are None, set to 'localhost'.",
    )
    path: str | None = Field(None, description="Persistence path for QdrantLocal.")
    force_disable_check_same_thread: bool | None = Field(
        True,
        description=(
            "For QdrantLocal, force disable check_same_thread. Default: `True`"
            "Only use this if you can guarantee that you can resolve the thread safety outside QdrantClient."
        ),
    )


class MilvusSettings(BaseModel):
    uri: str = Field(
        "local_data/private_gpt/milvus/milvus_local.db",
        description="The URI of the Milvus instance. For example: 'local_data/private_gpt/milvus/milvus_local.db' for Milvus Lite.",
    )
    token: str = Field(
        "",
        description=(
            "A valid access token to access the specified Milvus instance. "
            "This can be used as a recommended alternative to setting user and password separately. "
        ),
    )
    collection_name: str = Field(
        "make_this_parameterizable_per_api_call",
        description="The name of the collection in Milvus. Default is 'make_this_parameterizable_per_api_call'.",
    )
    overwrite: bool = Field(
        True, description="Overwrite the previous collection schema if it exists."
    )


class Settings(BaseModel):
    server: ServerSettings
    data: DataSettings
    ui: UISettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    llamacpp: LlamaCPPSettings
    huggingface: HuggingFaceSettings
    sagemaker: SagemakerSettings
    openai: OpenAISettings
    gemini: GeminiSettings
    ollama: OllamaSettings
    azopenai: AzureOpenAISettings
    vectorstore: VectorstoreSettings
    nodestore: NodeStoreSettings
    rag: RagSettings
    summarize: SummarizeSettings
    qdrant: QdrantSettings | None = None
    postgres: PostgresSettings | None = None
    clickhouse: ClickHouseSettings | None = None
    milvus: MilvusSettings | None = None


"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_settings = load_active_settings()

"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_typed_settings = Settings(**unsafe_settings)


def settings() -> Settings:
    """Get the current loaded settings from the DI container.

    This method exists to keep compatibility with the existing code,
    that require global access to the settings.

    For regular components use dependency injection instead.
    """
    from private_gpt.di import global_injector

    return global_injector.get(Settings)


# private_gpt/settings/settings.py
import os
import re
import typing
from typing import Any, TextIO

from yaml import SafeLoader

_env_replace_matcher = re.compile(r"\$\{(\w|_)+:?.*}")


@typing.no_type_check  # pyaml does not have good hints, everything is Any
def load_yaml_with_envvars(
    stream: TextIO, environ: dict[str, Any] = os.environ
) -> dict[str, Any]:
    """Load yaml file with environment variable expansion.

    The pattern ${VAR} or ${VAR:default} will be replaced with
    the value of the environment variable.
    """
    loader = SafeLoader(stream)

    def load_env_var(_, node) -> str:
        """Extract the matched value, expand env variable, and replace the match."""
        value = str(node.value).removeprefix("${").removesuffix("}")
        split = value.split(":", 1)
        env_var = split[0]
        value = environ.get(env_var)
        default = None if len(split) == 1 else split[1]
        if value is None and default is None:
            raise ValueError(
                f"Environment variable {env_var} is not set and not default was provided"
            )
        return value or default

    loader.add_implicit_resolver("env_var_replacer", _env_replace_matcher, None)
    loader.add_constructor("env_var_replacer", load_env_var)

    try:
        return loader.get_single_data()
    finally:
        loader.dispose()

```


## 2025-09-02-2352
现在我想要 根据settings.py（可能还需要进一步修改）继续重构其余的private-gpt的工程逻辑   ingest
我自己的拆解private-gpt的工程架构设计(初步设想、还有待完善或者就是有错的)如下：
```python
module_zzz
    + components/  # 部分组件的初始化与集成, 比如仿照private-gpt的 embedding, ingest,llm,node_store,vector_store 等等，
    + api/  # api接口的逻辑封装、 某某功能，比如仿照private-gpt的 chat, ingest、chunks 等等，
    + settings.py

``` 

private-gpt的部分源码
```python
private-gpt/components/ingest/ingest_components.py
import abc
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Any

from llama_index.core.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings
from private_gpt.utils.eta import eta

logger = logging.getLogger(__name__)


class BaseIngestComponent(abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Initializing base ingest component type=%s", type(self).__name__)
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.transformations = transformations

    @abc.abstractmethod
    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        pass

    @abc.abstractmethod
    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        pass

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None:
        pass


class BaseIngestComponentWithIndex(BaseIngestComponent, abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

        self.show_progress = True
        self._index_thread_lock = (
            threading.Lock()
        )  # Thread lock! Not Multiprocessing lock
        self._index = self._initialize_index()

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        """Initialize the index from the storage context."""
        try:
            # Load the index with store_nodes_override=True to be able to delete them
            index = load_index_from_storage(
                storage_context=self.storage_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
        except ValueError:
            # There are no index in the storage context, creating a new one
            logger.info("Creating a new vector store index")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            index.storage_context.persist(persist_dir=local_data_path)
        return index

    def _save_index(self) -> None:
        self._index.storage_context.persist(persist_dir=local_data_path)

    def delete(self, doc_id: str) -> None:
        with self._index_thread_lock:
            # Delete the document from the index
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)

            # Save the index
            self._save_index()


class SimpleIngestComponent(BaseIngestComponentWithIndex):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        saved_documents = []
        for file_name, file_data in files:
            documents = IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        with self._index_thread_lock:
            for document in documents:
                self._index.insert(document, show_progress=True)
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents


class BatchIngestComponent(BaseIngestComponentWithIndex):
    """Parallelize the file reading and parsing on multiple CPU core.

    This also makes the embeddings to be computed in batches (on GPU or CPU).
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        # Make an efficient use of the CPU and GPU, the embedding
        # must be in the transformations
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers

        self._file_to_documents_work_pool = multiprocessing.Pool(
            processes=self.count_workers
        )

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        documents = list(
            itertools.chain.from_iterable(
                self._file_to_documents_work_pool.starmap(
                    IngestionHelper.transform_file_into_documents, files
                )
            )
        )
        logger.info(
            "Transformed count=%s files into count=%s documents",
            len(files),
            len(documents),
        )
        return self._save_docs(documents)

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,  # type: ignore[arg-type]
            self.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.info("Inserting count=%s nodes in the index", len(nodes))
            self._index.insert_nodes(nodes, show_progress=True)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents


class ParallelizedIngestComponent(BaseIngestComponentWithIndex):
    """Parallelize the file ingestion (file reading, embeddings, and index insertion).

    This use the CPU and GPU in parallel (both running at the same time), and
    reduce the memory pressure by not loading all the files in memory at the same time.
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        # To make an efficient use of the CPU and GPU, the embeddings
        # must be in the transformations (to be computed in batches)
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers
        # We are doing our own multiprocessing
        # To do not collide with the multiprocessing of huggingface, we disable it
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self._ingest_work_pool = multiprocessing.pool.ThreadPool(
            processes=self.count_workers
        )

        self._file_to_documents_work_pool = multiprocessing.Pool(
            processes=self.count_workers
        )

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        # Running in a single (1) process to release the current
        # thread, and take a dedicated CPU core for computation
        documents = self._file_to_documents_work_pool.apply(
            IngestionHelper.transform_file_into_documents, (file_name, file_data)
        )
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        # Lightweight threads, used for parallelize the
        # underlying IO calls made in the ingestion

        documents = list(
            itertools.chain.from_iterable(
                self._ingest_work_pool.starmap(self.ingest, files)
            )
        )
        return documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,  # type: ignore[arg-type]
            self.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.info("Inserting count=%s nodes in the index", len(nodes))
            self._index.insert_nodes(nodes, show_progress=True)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents

    def __del__(self) -> None:
        # We need to do the appropriate cleanup of the multiprocessing pools
        # when the object is deleted. Using root logger to avoid
        # the logger to be deleted before the pool
        logging.debug("Closing the ingest work pool")
        self._ingest_work_pool.close()
        self._ingest_work_pool.join()
        self._ingest_work_pool.terminate()
        logging.debug("Closing the file to documents work pool")
        self._file_to_documents_work_pool.close()
        self._file_to_documents_work_pool.join()
        self._file_to_documents_work_pool.terminate()


class PipelineIngestComponent(BaseIngestComponentWithIndex):
    """Pipeline ingestion - keeping the embedding worker pool as busy as possible.

    This class implements a threaded ingestion pipeline, which comprises two threads
    and two queues. The primary thread is responsible for reading and parsing files
    into documents. These documents are then placed into a queue, which is
    distributed to a pool of worker processes for embedding computation. After
    embedding, the documents are transferred to another queue where they are
    accumulated until a threshold is reached. Upon reaching this threshold, the
    accumulated documents are flushed to the document store, index, and vector
    store.

    Exception handling ensures robustness against erroneous files. However, in the
    pipelined design, one error can lead to the discarding of multiple files. Any
    discarded files will be reported.
    """

    NODE_FLUSH_COUNT = 5000  # Save the index every # nodes.

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        self.count_workers = count_workers
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers
        # We are doing our own multiprocessing
        # To do not collide with the multiprocessing of huggingface, we disable it
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # doc_q stores parsed files as Document chunks.
        # Using a shallow queue causes the filesystem parser to block
        # when it reaches capacity. This ensures it doesn't outpace the
        # computationally intensive embeddings phase, avoiding unnecessary
        # memory consumption.  The semaphore is used to bound the async worker
        # embedding computations to cause the doc Q to fill and block.
        self.doc_semaphore = multiprocessing.Semaphore(
            self.count_workers
        )  # limit the doc queue to # items.
        self.doc_q: Queue[tuple[str, str | None, list[Document] | None]] = Queue(20)
        # node_q stores documents parsed into nodes (embeddings).
        # Larger queue size so we don't block the embedding workers during a slow
        # index update.
        self.node_q: Queue[
            tuple[str, str | None, list[Document] | None, list[BaseNode] | None]
        ] = Queue(40)
        threading.Thread(target=self._doc_to_node, daemon=True).start()
        threading.Thread(target=self._write_nodes, daemon=True).start()

    def _doc_to_node(self) -> None:
        # Parse documents into nodes
        with multiprocessing.pool.ThreadPool(processes=self.count_workers) as pool:
            while True:
                try:
                    cmd, file_name, documents = self.doc_q.get(
                        block=True
                    )  # Documents for a file
                    if cmd == "process":
                        # Push CPU/GPU embedding work to the worker pool
                        # Acquire semaphore to control access to worker pool
                        self.doc_semaphore.acquire()
                        pool.apply_async(
                            self._doc_to_node_worker, (file_name, documents)
                        )
                    elif cmd == "quit":
                        break
                finally:
                    if cmd != "process":
                        self.doc_q.task_done()  # unblock Q joins

    def _doc_to_node_worker(self, file_name: str, documents: list[Document]) -> None:
        # CPU/GPU intensive work in its own process
        try:
            nodes = run_transformations(
                documents,  # type: ignore[arg-type]
                self.transformations,
                show_progress=self.show_progress,
            )
            self.node_q.put(("process", file_name, documents, list(nodes)))
        finally:
            self.doc_semaphore.release()
            self.doc_q.task_done()  # unblock Q joins

    def _save_docs(
        self, files: list[str], documents: list[Document], nodes: list[BaseNode]
    ) -> None:
        try:
            logger.info(
                f"Saving {len(files)} files ({len(documents)} documents / {len(nodes)} nodes)"
            )
            self._index.insert_nodes(nodes)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            self._save_index()
        except Exception:
            # Tell the user so they can investigate these files
            logger.exception(f"Processing files {files}")
        finally:
            # Clearing work, even on exception, maintains a clean state.
            nodes.clear()
            documents.clear()
            files.clear()

    def _write_nodes(self) -> None:
        # Save nodes to index.  I/O intensive.
        node_stack: list[BaseNode] = []
        doc_stack: list[Document] = []
        file_stack: list[str] = []
        while True:
            try:
                cmd, file_name, documents, nodes = self.node_q.get(block=True)
                if cmd in ("flush", "quit"):
                    if file_stack:
                        self._save_docs(file_stack, doc_stack, node_stack)
                    if cmd == "quit":
                        break
                elif cmd == "process":
                    node_stack.extend(nodes)  # type: ignore[arg-type]
                    doc_stack.extend(documents)  # type: ignore[arg-type]
                    file_stack.append(file_name)  # type: ignore[arg-type]
                    # Constant saving is heavy on I/O - accumulate to a threshold
                    if len(node_stack) >= self.NODE_FLUSH_COUNT:
                        self._save_docs(file_stack, doc_stack, node_stack)
            finally:
                self.node_q.task_done()

    def _flush(self) -> None:
        self.doc_q.put(("flush", None, None))
        self.doc_q.join()
        self.node_q.put(("flush", None, None, None))
        self.node_q.join()

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        self.doc_q.put(("process", file_name, documents))
        self._flush()
        return documents

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        docs = []
        for file_name, file_data in eta(files):
            try:
                documents = IngestionHelper.transform_file_into_documents(
                    file_name, file_data
                )
                self.doc_q.put(("process", file_name, documents))
                docs.extend(documents)
            except Exception:
                logger.exception(f"Skipping {file_data.name}")
        self._flush()
        return docs


def get_ingestion_component(
    storage_context: StorageContext,
    embed_model: EmbedType,
    transformations: list[TransformComponent],
    settings: Settings,
) -> BaseIngestComponent:
    """Get the ingestion component for the given configuration."""
    ingest_mode = settings.embedding.ingest_mode
    if ingest_mode == "batch":
        return BatchIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    elif ingest_mode == "parallel":
        return ParallelizedIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    elif ingest_mode == "pipeline":
        return PipelineIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    else:
        return SimpleIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
        )




private-gpt/components/ingest/ingest_helper.py
import logging
from pathlib import Path

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


# Inspired by the `llama_index.core.readers.file.base` module
def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])

        logger.debug("Specific reader found for extension=%s", extension)
        documents = reader_cls().load_data(file_data)

        # Sanitize NUL bytes in text which can't be stored in Postgres
        for i in range(len(documents)):
            documents[i].text = documents[i].text.replace("\u0000", "")

        return documents

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]




private-gpt/server/ingest/ingest_router.py
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class IngestTextBody(BaseModel):
    file_name: str = Field(examples=["Avatar: The Last Airbender"])
    text: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of the four elements—water, "
            "earth, fire or air—through practices known as 'bending', inspired by "
            "Chinese martial arts."
        ]
    )


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]


@ingest_router.post("/ingest", tags=["Ingestion"], deprecated=True)
def ingest(request: Request, file: UploadFile) -> IngestResponse:
    """Ingests and processes a file.

    Deprecated. Use ingest/file instead.
    """
    return ingest_file(request, file)


@ingest_router.post("/ingest/file", tags=["Ingestion"])
def ingest_file(request: Request, file: UploadFile) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    Most common document
    formats are supported, but you may be prompted to install an extra dependency to
    manage a specific file type.

    A file can generate different Documents (for example a PDF generates one Document
    per page). All Documents IDs are returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). Those IDs
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_bin_data(file.filename, file.file)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.post("/ingest/text", tags=["Ingestion"])
def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    A Document will be generated with the given text. The Document
    ID is returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). That ID
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    if len(body.file_name) == 0:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_text(body.file_name, body.text)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(request: Request) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(request: Request, doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)




private-gpt/server/ingest/ingest_server.py
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr, BinaryIO

from injector import inject, singleton
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.storage import StorageContext

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

logger = logging.getLogger(__name__)


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        node_parser = SentenceWindowNodeParser.from_defaults()

        self.ingest_component = get_ingestion_component(
            self.storage_context,
            embed_model=embedding_component.embedding_model,
            transformations=[node_parser, embedding_component.embedding_model],
            settings=settings(),
        )

    def _ingest_data(self, file_name: str, file_data: AnyStr) -> list[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        # llama-index mainly supports reading from files, so
        # we have to create a tmp file to read for it to work
        # delete=False to avoid a Windows 11 permission error.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                return self.ingest_file(file_name, path_to_tmp)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    def ingest_file(self, file_name: str, file_data: Path) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = self.ingest_component.ingest(file_name, file_data)
        logger.info("Finished ingestion file_name=%s", file_name)
        return [IngestedDoc.from_document(document) for document in documents]

    def ingest_text(self, file_name: str, text: str) -> list[IngestedDoc]:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        return self._ingest_data(file_name, text)

    def ingest_bin_data(
        self, file_name: str, raw_file_data: BinaryIO
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        return self._ingest_data(file_name, file_data)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[IngestedDoc]:
        logger.info("Ingesting file_names=%s", [f[0] for f in files])
        documents = self.ingest_component.bulk_ingest(files)
        logger.info("Finished ingestion file_name=%s", [f[0] for f in files])
        return [IngestedDoc.from_document(document) for document in documents]

    def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs: list[IngestedDoc] = []
        try:
            docstore = self.storage_context.docstore
            ref_docs: dict[str, RefDocInfo] | None = docstore.get_all_ref_doc_info()

            if not ref_docs:
                return ingested_docs

            for doc_id, ref_doc_info in ref_docs.items():
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
            pass
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    def delete(self, doc_id: str) -> None:
        """Delete an ingested document.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s in the doc and index store", doc_id
        )
        self.ingest_component.delete(doc_id)




private-gpt/server/ingest/ingest_watcher.py
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileCreatedEvent,
    FileModifiedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer


class IngestWatcher:
    def __init__(
        self, watch_path: Path, on_file_changed: Callable[[Path], None]
    ) -> None:
        self.watch_path = watch_path
        self.on_file_changed = on_file_changed

        class Handler(FileSystemEventHandler):
            def on_modified(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileModifiedEvent):
                    on_file_changed(Path(event.src_path))

            def on_created(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileCreatedEvent):
                    on_file_changed(Path(event.src_path))

        event_handler = Handler()
        observer: Any = Observer()
        self._observer = observer
        self._observer.schedule(event_handler, str(watch_path), recursive=True)

    def start(self) -> None:
        self._observer.start()
        while self._observer.is_alive():
            try:
                self._observer.join(1)
            except KeyboardInterrupt:
                break

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()




private-gpt/server/ingest/model.py
from typing import Any, Literal

from llama_index.core.schema import Document
from pydantic import BaseModel, Field


class IngestedDoc(BaseModel):
    object: Literal["ingest.document"]
    doc_id: str = Field(examples=["c202d5e6-7b69-4869-81cc-dd574ee8ee11"])
    doc_metadata: dict[str, Any] | None = Field(
        examples=[
            {
                "page_label": "2",
                "file_name": "Sales Report Q3 2023.pdf",
            }
        ]
    )

    @staticmethod
    def curate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove unwanted metadata keys."""
        for key in ["doc_id", "window", "original_text"]:
            metadata.pop(key, None)
        return metadata

    @staticmethod
    def from_document(document: Document) -> "IngestedDoc":
        return IngestedDoc(
            object="ingest.document",
            doc_id=document.doc_id,
            doc_metadata=IngestedDoc.curate_metadata(document.metadata),
        )
```


## 2025-09-03-2315
我依旧不是很清楚private-gpt中文件处理的流程，
components模块中有：embedding ingest  llm node_store  vector_store
server模块中有：chat chunks completions embeddings  health ingest recopies utils

然后我又研究起private-gpt 的components模块中的ingest部分源码了，我认为这个代码处理的是加载文件的逻辑，是整个工程流程的开始

我发现了llama-index一个很不错的设计from llama_index.core.schema import Document这个Document类，继承自Node，也就是llama_index节点的设计，所以有了下面 private_gpt/components/ingest_helper.py ：


```python
import logging
from pathlib import Path

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


# Inspired by the `llama_index.core.readers.file.base` module
def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])

        logger.debug("Specific reader found for extension=%s", extension)
        documents = reader_cls().load_data(file_data)

        # Sanitize NUL bytes in text which can't be stored in Postgres
        for i in range(len(documents)):
            documents[i].text = documents[i].text.replace("\u0000", "")

        return documents

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
```
我希望你介绍一下Document  Node  BaseNode  

## 2025-09-03-2332
随后我们继续解读components模块的inget_component.py，这里我希望你先整理一下应用到的包、接口，先介绍一下他们，重点是llama-index相关的接口、包，
然后梳理一下代码 类 方法 做了什么事情，
```python
import abc
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Any

from llama_index.core.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings
from private_gpt.utils.eta import eta

logger = logging.getLogger(__name__)


class BaseIngestComponent(abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Initializing base ingest component type=%s", type(self).__name__)
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.transformations = transformations

    @abc.abstractmethod
    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        pass

    @abc.abstractmethod
    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        pass

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None:
        pass


class BaseIngestComponentWithIndex(BaseIngestComponent, abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

        self.show_progress = True
        self._index_thread_lock = (
            threading.Lock()
        )  # Thread lock! Not Multiprocessing lock
        self._index = self._initialize_index()

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        """Initialize the index from the storage context."""
        try:
            # Load the index with store_nodes_override=True to be able to delete them
            index = load_index_from_storage(
                storage_context=self.storage_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
        except ValueError:
            # There are no index in the storage context, creating a new one
            logger.info("Creating a new vector store index")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            index.storage_context.persist(persist_dir=local_data_path)
        return index

    def _save_index(self) -> None:
        self._index.storage_context.persist(persist_dir=local_data_path)

    def delete(self, doc_id: str) -> None:
        with self._index_thread_lock:
            # Delete the document from the index
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)

            # Save the index
            self._save_index()


class SimpleIngestComponent(BaseIngestComponentWithIndex):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        saved_documents = []
        for file_name, file_data in files:
            documents = IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        with self._index_thread_lock:
            for document in documents:
                self._index.insert(document, show_progress=True)
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents


class BatchIngestComponent(BaseIngestComponentWithIndex):
    """Parallelize the file reading and parsing on multiple CPU core.

    This also makes the embeddings to be computed in batches (on GPU or CPU).
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        # Make an efficient use of the CPU and GPU, the embedding
        # must be in the transformations
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers

        self._file_to_documents_work_pool = multiprocessing.Pool(
            processes=self.count_workers
        )

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        documents = list(
            itertools.chain.from_iterable(
                self._file_to_documents_work_pool.starmap(
                    IngestionHelper.transform_file_into_documents, files
                )
            )
        )
        logger.info(
            "Transformed count=%s files into count=%s documents",
            len(files),
            len(documents),
        )
        return self._save_docs(documents)

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,  # type: ignore[arg-type]
            self.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.info("Inserting count=%s nodes in the index", len(nodes))
            self._index.insert_nodes(nodes, show_progress=True)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents


class ParallelizedIngestComponent(BaseIngestComponentWithIndex):
    """Parallelize the file ingestion (file reading, embeddings, and index insertion).

    This use the CPU and GPU in parallel (both running at the same time), and
    reduce the memory pressure by not loading all the files in memory at the same time.
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        # To make an efficient use of the CPU and GPU, the embeddings
        # must be in the transformations (to be computed in batches)
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers
        # We are doing our own multiprocessing
        # To do not collide with the multiprocessing of huggingface, we disable it
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self._ingest_work_pool = multiprocessing.pool.ThreadPool(
            processes=self.count_workers
        )

        self._file_to_documents_work_pool = multiprocessing.Pool(
            processes=self.count_workers
        )

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        # Running in a single (1) process to release the current
        # thread, and take a dedicated CPU core for computation
        documents = self._file_to_documents_work_pool.apply(
            IngestionHelper.transform_file_into_documents, (file_name, file_data)
        )
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        # Lightweight threads, used for parallelize the
        # underlying IO calls made in the ingestion

        documents = list(
            itertools.chain.from_iterable(
                self._ingest_work_pool.starmap(self.ingest, files)
            )
        )
        return documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,  # type: ignore[arg-type]
            self.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.info("Inserting count=%s nodes in the index", len(nodes))
            self._index.insert_nodes(nodes, show_progress=True)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents

    def __del__(self) -> None:
        # We need to do the appropriate cleanup of the multiprocessing pools
        # when the object is deleted. Using root logger to avoid
        # the logger to be deleted before the pool
        logging.debug("Closing the ingest work pool")
        self._ingest_work_pool.close()
        self._ingest_work_pool.join()
        self._ingest_work_pool.terminate()
        logging.debug("Closing the file to documents work pool")
        self._file_to_documents_work_pool.close()
        self._file_to_documents_work_pool.join()
        self._file_to_documents_work_pool.terminate()


class PipelineIngestComponent(BaseIngestComponentWithIndex):
    """Pipeline ingestion - keeping the embedding worker pool as busy as possible.

    This class implements a threaded ingestion pipeline, which comprises two threads
    and two queues. The primary thread is responsible for reading and parsing files
    into documents. These documents are then placed into a queue, which is
    distributed to a pool of worker processes for embedding computation. After
    embedding, the documents are transferred to another queue where they are
    accumulated until a threshold is reached. Upon reaching this threshold, the
    accumulated documents are flushed to the document store, index, and vector
    store.

    Exception handling ensures robustness against erroneous files. However, in the
    pipelined design, one error can lead to the discarding of multiple files. Any
    discarded files will be reported.
    """

    NODE_FLUSH_COUNT = 5000  # Save the index every # nodes.

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        self.count_workers = count_workers
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers
        # We are doing our own multiprocessing
        # To do not collide with the multiprocessing of huggingface, we disable it
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # doc_q stores parsed files as Document chunks.
        # Using a shallow queue causes the filesystem parser to block
        # when it reaches capacity. This ensures it doesn't outpace the
        # computationally intensive embeddings phase, avoiding unnecessary
        # memory consumption.  The semaphore is used to bound the async worker
        # embedding computations to cause the doc Q to fill and block.
        self.doc_semaphore = multiprocessing.Semaphore(
            self.count_workers
        )  # limit the doc queue to # items.
        self.doc_q: Queue[tuple[str, str | None, list[Document] | None]] = Queue(20)
        # node_q stores documents parsed into nodes (embeddings).
        # Larger queue size so we don't block the embedding workers during a slow
        # index update.
        self.node_q: Queue[
            tuple[str, str | None, list[Document] | None, list[BaseNode] | None]
        ] = Queue(40)
        threading.Thread(target=self._doc_to_node, daemon=True).start()
        threading.Thread(target=self._write_nodes, daemon=True).start()

    def _doc_to_node(self) -> None:
        # Parse documents into nodes
        with multiprocessing.pool.ThreadPool(processes=self.count_workers) as pool:
            while True:
                try:
                    cmd, file_name, documents = self.doc_q.get(
                        block=True
                    )  # Documents for a file
                    if cmd == "process":
                        # Push CPU/GPU embedding work to the worker pool
                        # Acquire semaphore to control access to worker pool
                        self.doc_semaphore.acquire()
                        pool.apply_async(
                            self._doc_to_node_worker, (file_name, documents)
                        )
                    elif cmd == "quit":
                        break
                finally:
                    if cmd != "process":
                        self.doc_q.task_done()  # unblock Q joins

    def _doc_to_node_worker(self, file_name: str, documents: list[Document]) -> None:
        # CPU/GPU intensive work in its own process
        try:
            nodes = run_transformations(
                documents,  # type: ignore[arg-type]
                self.transformations,
                show_progress=self.show_progress,
            )
            self.node_q.put(("process", file_name, documents, list(nodes)))
        finally:
            self.doc_semaphore.release()
            self.doc_q.task_done()  # unblock Q joins

    def _save_docs(
        self, files: list[str], documents: list[Document], nodes: list[BaseNode]
    ) -> None:
        try:
            logger.info(
                f"Saving {len(files)} files ({len(documents)} documents / {len(nodes)} nodes)"
            )
            self._index.insert_nodes(nodes)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            self._save_index()
        except Exception:
            # Tell the user so they can investigate these files
            logger.exception(f"Processing files {files}")
        finally:
            # Clearing work, even on exception, maintains a clean state.
            nodes.clear()
            documents.clear()
            files.clear()

    def _write_nodes(self) -> None:
        # Save nodes to index.  I/O intensive.
        node_stack: list[BaseNode] = []
        doc_stack: list[Document] = []
        file_stack: list[str] = []
        while True:
            try:
                cmd, file_name, documents, nodes = self.node_q.get(block=True)
                if cmd in ("flush", "quit"):
                    if file_stack:
                        self._save_docs(file_stack, doc_stack, node_stack)
                    if cmd == "quit":
                        break
                elif cmd == "process":
                    node_stack.extend(nodes)  # type: ignore[arg-type]
                    doc_stack.extend(documents)  # type: ignore[arg-type]
                    file_stack.append(file_name)  # type: ignore[arg-type]
                    # Constant saving is heavy on I/O - accumulate to a threshold
                    if len(node_stack) >= self.NODE_FLUSH_COUNT:
                        self._save_docs(file_stack, doc_stack, node_stack)
            finally:
                self.node_q.task_done()

    def _flush(self) -> None:
        self.doc_q.put(("flush", None, None))
        self.doc_q.join()
        self.node_q.put(("flush", None, None, None))
        self.node_q.join()

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        self.doc_q.put(("process", file_name, documents))
        self._flush()
        return documents

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        docs = []
        for file_name, file_data in eta(files):
            try:
                documents = IngestionHelper.transform_file_into_documents(
                    file_name, file_data
                )
                self.doc_q.put(("process", file_name, documents))
                docs.extend(documents)
            except Exception:
                logger.exception(f"Skipping {file_data.name}")
        self._flush()
        return docs


def get_ingestion_component(
    storage_context: StorageContext,
    embed_model: EmbedType,
    transformations: list[TransformComponent],
    settings: Settings,
) -> BaseIngestComponent:
    """Get the ingestion component for the given configuration."""
    ingest_mode = settings.embedding.ingest_mode
    if ingest_mode == "batch":
        return BatchIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    elif ingest_mode == "parallel":
        return ParallelizedIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    elif ingest_mode == "pipeline":
        return PipelineIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    else:
        return SimpleIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
        )

```

## 2025-09-04-2114
今天我们继续拆解private_gpt的ingest部分，在这之前，
我想先请你介绍一下llama-index中reranker模型的地位，好像并没有很有用的样子，
我找到了官方的接口
from llama_index.core.postprocessor import SentenceTransformerRerank
定义SentenceTransformerRerank类的模块sbert_rerank.py
也请你介绍一下这个类，要怎么用呢，有什么样的限制，然后这个类适配Qwen3-Reranker-0.6B模型吗？
好像现在reranker的实现方式并不完全一样呢，请你分析分析
```python
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.utils import infer_torch_device

DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH = 512


class SentenceTransformerRerank(BaseNodePostprocessor):
    model: str = Field(description="Sentence transformer model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    device: str = Field(
        default="cpu",
        description="Device to use for sentence transformer.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code.",
    )
    _model: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "cross-encoder/stsb-distilroberta-base",
        device: Optional[str] = None,
        keep_retrieval_score: bool = False,
        trust_remote_code: bool = True,
    ):
        try:
            from sentence_transformers import CrossEncoder  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers or torch package,",
                "please `pip install torch sentence-transformers`",
            )
        device = infer_torch_device() if device is None else device
        super().__init__(
            top_n=top_n,
            model=model,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
        )
        self._model = CrossEncoder(
            model,
            max_length=DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = self._model.predict(query_and_nodes)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
```


## 2025-09-04


## 2025-09-04


## 2025-09-04


## 2025-09-04


## 2025-09-04


## 


## 


## 


## 







## 



## 


## 

## 



## 


## 


## 

# 专利解析工作流设计
## 2025-09-07-1018

用一个智能体工作流 结合 强大的LLM 来结构化提取 专利文件pdf（打印版、普通版）的信息

输入文件：  mineru解析之后的 专利pdf文件.md  
        - 专利pdf文件.md  存在于一个子文件夹中，附带的还有原始的PDF文件.pdf、images/ 等  

```text
文件系统：
~ patent_root/
    + sub_dir/
        + *.md          # 解析之后的markdown  （对于打印版pdf的解析不够好）
        + *_origin.pdf  # 原pdf 
        + images/       # 抽出来的配图 (不存在遗漏)

-->
文件戳： 词条解析、需要配置一个 英文-中文对照的prompt、 文件类型， --> 写入原 sub_dir  metadata.json
正文  ： 打印版的pdf可能几乎全部都是配图，几乎没有文字说明         --> 写入原 sub_dir  textdata.md
```



## 


## 



## 


## 

## 



## 


## 

## 



## 


## 


## 



## 


## 


## 



## 


## 

## 



## 


## 

## 

# 
## 


## 


## 



## 


## 


## 



## 


## 

## 



## 


## 

## 



## 


## 


## 

# 

## 


## 


## 



## 


## 

## 



## 


## 

## 



## 


## 


## 



## 


## 

# 
## 



## 


## 

## 



## 


## 

## 



## 


## 


## 



## 


## 


## 



## 


## 

## 



## 


## 

## 

# 

## 


## 


## 



## 


## 


## 



## 


## 