import logging
from collections.abc import Callable
from typing import Any

from injector import inject, singleton
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.settings import Settings as LlamaIndexSettings
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer  # type: ignore

from components.languagelm.prompt_helper import get_prompt_style

from settings import AppSettings

logger = logging.getLogger(__name__)




@singleton
class LLMComponent:
    llm: LLM

    @inject
    def __init__(self, settings: AppSettings) -> None:
        llm_mode = settings.languagelm.load_method
        assert llm_mode == "hf_cache", f"Only 'hf_cache' is supported for llm.load_method, got {llm_mode}"
        #### tokenizer
        if settings.languagelm.model_name:
            # Try to download the tokenizer. If it fails, the LLM will still work
            # using the default one, which is less accurate.
            try:
                set_global_tokenizer(
                    AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=settings.languagelm.model_name,
                        cache_dir=str(settings.hf_cache.hf_models_cache_root),
                        trust_remote_code=settings.languagelm.trust_remote_code,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to download tokenizer {settings.languagelm.model_name}: {e!s}"
                    f"Please follow the instructions in the documentation to download it if needed: "
                    f"https://docs.privategpt.dev/installation/getting-started/troubleshooting#tokenizer-setup."
                    f"Falling back to default tokenizer."
                )

        logger.info("Initializing the LLM in mode=%s", llm_mode)
        ##### language lm
        match settings.languagelm.load_method:

            case "hf_cache":
                LLMCls = settings._import([
                "llama_index.llms.huggingface.HuggingFaceLLM",
                ])
                self.llm = LLMCls(
                    model_name=settings.languagelm.model_name,
                    tokenizer_name=settings.languagelm.model_name,
                    context_window=settings.languagelm.context_window,
                    max_new_tokens=settings.languagelm.max_tokens,
                    system_prompt=settings.languagelm.system_prompt,
                    generate_kwargs={"temperature": settings.languagelm.temperature, 
                                    "top_k": settings.languagelm.top_k, 
                                    "top_p": settings.languagelm.top_p, 
                                    "do_sample": settings},
                    device_map=settings.languagelm.device,
                )
            case "llamacpp":
                try:
                    from llama_index.llms.llama_cpp import LlamaCPP  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras llms-llama-cpp`"
                    ) from e

                prompt_style = get_prompt_style(settings.llm.prompt_style)
                settings_kwargs = {
                    "tfs_z": settings.llamacpp.tfs_z,  # ollama and llama-cpp
                    "top_k": settings.llamacpp.top_k,  # ollama and llama-cpp
                    "top_p": settings.llamacpp.top_p,  # ollama and llama-cpp
                    "repeat_penalty": settings.llamacpp.repeat_penalty,  # ollama llama-cpp
                    "n_gpu_layers": -1,
                    "offload_kqv": True,
                }
                self.llm = LlamaCPP(
                    model_path=str(models_path / settings.llamacpp.llm_hf_model_file),
                    temperature=settings.llm.temperature,
                    max_new_tokens=settings.llm.max_new_tokens,
                    context_window=settings.llm.context_window,
                    generate_kwargs={},
                    callback_manager=LlamaIndexSettings.callback_manager,
                    # All to GPU
                    model_kwargs=settings_kwargs,
                    # transform inputs into Llama2 format
                    messages_to_prompt=prompt_style.messages_to_prompt,
                    completion_to_prompt=prompt_style.completion_to_prompt,
                    verbose=True,
                )

            case "sagemaker":
                try:
                    from components.languagelm.custom.sagemaker import SagemakerLLM
                except ImportError as e:
                    raise ImportError(
                        "Sagemaker dependencies not found, install with `poetry install --extras llms-sagemaker`"
                    ) from e

                self.llm = SagemakerLLM(
                    endpoint_name=settings.sagemaker.llm_endpoint_name,
                    max_new_tokens=settings.llm.max_new_tokens,
                    context_window=settings.llm.context_window,
                )
            